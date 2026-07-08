// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Graduate closed live nodes into the cold bucket-index (tiered V4 metadata).
//!
//! With append-only live nodes (the V4 "one-file commit" model), each commit
//! writes its new files into a fresh child manifest ("node") and keeps only the
//! node *reference* in the root. This action relocates the **closed** live nodes
//! — those whose newest event time has fallen behind the cutoff — out of the hot
//! root and into the cold bucket-index. The move is reference-only: a closed
//! node is immutable, so graduating it just moves its `ManifestFile` ref from
//! the root into the bucket-index (no data read, no re-write).
//!
//! For robustness it also handles any residual **inline** entries (the older
//! "live = inline" shape): closed inline files are materialized into new
//! partition-tight cold leaves.
//!
//! The close decision keys on **time** — the max value of `ts_field_id` across a
//! node's data files — so it is fully partition-spec-agnostic. The caller
//! computes `cutoff_micros = now − bucket_window`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use super::rebalance_root_manifest::write_entries_clustered;
use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    reconstruct_root, write_root_manifest, RootManifestEntry, RootManifestMetadata,
};
use crate::spec::{
    DataFile, FormatVersion, ManifestEntry, ManifestFile, ManifestStatus, Operation,
    PrimitiveLiteral, Snapshot, SnapshotReference, SnapshotRetention, Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Action that relocates closed live nodes (and any closed inline files) into
/// the cold bucket-index. Use via `Transaction::graduate_buckets(ts_field_id,
/// cutoff_micros)`.
pub struct GraduateBucketsAction {
    ts_field_id: i32,
    cutoff_micros: i64,
    commit_uuid: Uuid,
}

impl GraduateBucketsAction {
    /// Create with the event-time field id and the close cutoff (micros). A node
    /// (or inline file) graduates when its max value for `ts_field_id` is below
    /// `cutoff_micros`.
    pub fn new(ts_field_id: i32, cutoff_micros: i64) -> Self {
        Self {
            ts_field_id,
            cutoff_micros,
            commit_uuid: Uuid::now_v7(),
        }
    }
}

/// Max value of the `ts_field_id` upper-bound statistic across `files` (newest
/// event time). `None` if no file carries the stat. Pure.
fn max_ts_of(files: &[DataFile], ts_field_id: i32) -> Option<i64> {
    files
        .iter()
        .filter_map(
            |f| match f.upper_bounds().get(&ts_field_id).map(|d| d.literal()) {
                Some(PrimitiveLiteral::Long(v)) => Some(*v),
                _ => None,
            },
        )
        .max()
}

/// Max event time of a single data file for `ts_field_id`, if present. Pure.
fn file_max_ts(df: &DataFile, ts_field_id: i32) -> Option<i64> {
    match df.upper_bounds().get(&ts_field_id).map(|d| d.literal()) {
        Some(PrimitiveLiteral::Long(v)) => Some(*v),
        _ => None,
    }
}

/// Outcome of a graduation fold when something was actually moved to cold.
pub(crate) struct FoldOutcome {
    /// Path of the freshly-written bucket-index (existing cold leaves + newly
    /// closed nodes/inline). The caller sets this as the new root's
    /// `bucket_index_path`.
    pub bucket_index_path: String,
    pub nodes_moved: usize,
    pub inline_leaves: usize,
    pub cold_leaves_total: usize,
}

/// Fold the "closed" entries of a reconstructed live set (those whose max
/// `ts_field_id` event time is below `cutoff_micros`) into the cold bucket-index,
/// returning the entries that stay hot (`kept`) and — if anything was moved — a
/// [`FoldOutcome`] with the updated bucket-index path.
///
/// This is the durable core of graduation: it reads any existing bucket-index at
/// `carried_bucket_index_path`, relocates closed clean nodes by reference,
/// materializes closed inline files into partition-tight cold leaves, and writes
/// a new bucket-index — but it does NOT write a root. The caller folds this into
/// its own root commit (a collapse, or a standalone base), so graduation is
/// atomic with the root rewrite and cannot be orphaned by a concurrent append.
/// Partition-spec-agnostic (keys purely on the time cutoff).
///
/// Returns `(kept, None)` when nothing is closed (caller keeps its carried
/// pointer and its entries unchanged). `manifest_counter`/`commit_uuid` are
/// threaded from the caller so cold-leaf manifests share the caller's namespace.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn fold_closed_into_bucket_index(
    table: &Table,
    entries: Vec<RootManifestEntry>,
    carried_bucket_index_path: Option<&str>,
    cutoff_micros: i64,
    ts_field_id: i32,
    snapshot_id: i64,
    commit_uuid: Uuid,
    manifest_counter: &mut u64,
) -> Result<(Vec<RootManifestEntry>, Option<FoldOutcome>)> {
    // Existing cold leaves (graduated nodes get appended to these).
    let mut cold_leaves: Vec<ManifestFile> = match carried_bucket_index_path {
        Some(path) => {
            let b = table.file_io().new_input(path)?.read().await?;
            read_bucket_index(b)?.leaves().to_vec()
        }
        None => Vec::new(),
    };

    // Partition into: closed live nodes (→ cold by reference), closed inline
    // files (→ materialize as cold leaves), and kept (stay hot).
    let mut kept: Vec<RootManifestEntry> = Vec::new();
    let mut graduated_nodes: Vec<ManifestFile> = Vec::new();
    let mut closed_inline_files: Vec<DataFile> = Vec::new();

    for entry in entries {
        match entry {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                let manifest = manifest_file.load_manifest(table.file_io()).await?;
                let files: Vec<DataFile> = manifest
                    .entries()
                    .iter()
                    .filter(|e| e.is_alive())
                    .map(|e| e.data_file().clone())
                    .collect();
                let closed = max_ts_of(&files, ts_field_id)
                    .map(|mx| mx < cutoff_micros)
                    .unwrap_or(false);
                // Only graduate clean nodes; an MDV-carrying node has pending
                // deletes and is left for the rebalance/compaction path.
                if closed && mdv.is_none() {
                    graduated_nodes.push(manifest_file);
                } else {
                    kept.push(RootManifestEntry::ManifestRef { manifest_file, mdv });
                }
            }
            RootManifestEntry::Inline(me) => {
                let closed = file_max_ts(&me.data_file, ts_field_id)
                    .map(|mx| mx < cutoff_micros)
                    .unwrap_or(false);
                if closed {
                    closed_inline_files.push(me.data_file.clone());
                } else {
                    kept.push(RootManifestEntry::Inline(me));
                }
            }
        }
    }

    if graduated_nodes.is_empty() && closed_inline_files.is_empty() {
        return Ok((kept, None));
    }
    let nodes_moved = graduated_nodes.len();

    let schema = table.metadata().current_schema().clone();
    let format_version = table.metadata().format_version();
    let spec = table.metadata().default_partition_spec().clone();
    let partition_type = spec.partition_type(&schema)?;
    let next_seq_num = table.metadata().next_sequence_number();

    // Materialize any closed inline files into new partition-tight cold leaves.
    let mut inline_leaves = 0usize;
    if !closed_inline_files.is_empty() {
        let grad_entries: Vec<ManifestEntry> = closed_inline_files
            .into_iter()
            .map(|df| {
                ManifestEntry::builder()
                    .status(ManifestStatus::Existing)
                    .data_file(df)
                    .build()
            })
            .collect();
        let new_leaves = write_entries_clustered(
            table,
            &schema,
            spec.as_ref(),
            format_version,
            snapshot_id,
            commit_uuid,
            manifest_counter,
            false,
            grad_entries,
            true,
        )
        .await?;
        inline_leaves = new_leaves.len();
        cold_leaves.extend(new_leaves);
    }
    // Move graduated live nodes into cold by reference (immutable, no rewrite).
    cold_leaves.extend(graduated_nodes);

    let bucket_index_path = format!(
        "{}/{}/bucket-index-{}-{}.parquet",
        table.metadata().location(),
        META_ROOT_PATH,
        snapshot_id,
        commit_uuid,
    );
    let bi_metadata = RootManifestMetadata {
        schema: schema.clone(),
        schema_id: table.metadata().current_schema_id(),
        partition_spec: spec.clone(),
        format_version: FormatVersion::V4,
        snapshot_id,
        sequence_number: next_seq_num,
        parent_snapshot_id: table.metadata().current_snapshot_id(),
        bucket_index_path: None,
        prev_root_path: None,
        chain_depth: 0,
        node_level: 0,
        removed_paths: Vec::new(),
    };
    let bi_bytes = write_bucket_index(&cold_leaves, &bi_metadata, &partition_type)?;
    table
        .file_io()
        .new_output(&bucket_index_path)?
        .write(bi_bytes.into())
        .await?;

    let cold_leaves_total = cold_leaves.len();
    Ok((
        kept,
        Some(FoldOutcome {
            bucket_index_path,
            nodes_moved,
            inline_leaves,
            cold_leaves_total,
        }),
    ))
}

#[async_trait]
impl TransactionAction for GraduateBucketsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "graduate_buckets requires format version V4 (effective={:?})",
                    table.effective_format_version()
                ),
            ));
        }

        let current_snapshot = match table.metadata().current_snapshot() {
            Some(s) => s,
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        let root_path = current_snapshot.manifest_list();
        let (rm_metadata, entries) = reconstruct_root(table.file_io(), root_path).await?;

        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let commit_uuid = self.commit_uuid;
        let mut manifest_counter: u64 = 0;

        // Fold closed entries into the cold bucket-index (shared with the
        // commit_v4 collapse path). Returns the entries that stay hot plus the
        // updated bucket-index; None ⇒ nothing closed this pass.
        let (kept, fold) = fold_closed_into_bucket_index(
            table,
            entries,
            rm_metadata.bucket_index_path.as_deref(),
            self.cutoff_micros,
            self.ts_field_id,
            snapshot_id,
            commit_uuid,
            &mut manifest_counter,
        )
        .await?;
        let fold = match fold {
            Some(f) => f,
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        let schema = table.metadata().current_schema().clone();
        let next_seq_num = table.metadata().next_sequence_number();
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(&schema)?;

        // Standalone base root (this action resets the chain to a base). NOTE:
        // the durable path is commit_v4's collapse, which folds graduation
        // atomically with the chain rewrite; this standalone action is retained
        // for manual / maintenance-window use where no concurrent appends race it.
        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: Some(fold.bucket_index_path.clone()),
            prev_root_path: None,
            chain_depth: 0,
            node_level: 0,
            removed_paths: rm_metadata.removed_paths.clone(),
        };
        let new_root_path = format!(
            "{}/{}/root-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            snapshot_id,
            commit_uuid,
        );
        let root_bytes = write_root_manifest(&kept, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_path)?
            .write(root_bytes.into())
            .await?;

        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "graduate-nodes-moved".to_string(),
                    fold.nodes_moved.to_string(),
                ),
                (
                    "graduate-inline-leaves".to_string(),
                    fold.inline_leaves.to_string(),
                ),
                (
                    "graduate-cold-leaves-total".to_string(),
                    fold.cold_leaves_total.to_string(),
                ),
            ]),
        };

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let first_row_id = table.metadata().next_row_id();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(new_root_path.clone())
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(table.metadata().current_schema_id())
            .with_timestamp_ms(commit_ts)
            .with_row_range(first_row_id, 0)
            .build();

        let updates = vec![
            TableUpdate::AddSnapshot {
                snapshot: new_snapshot,
            },
            TableUpdate::SetSnapshotRef {
                ref_name: MAIN_BRANCH.to_string(),
                reference: SnapshotReference::new(
                    snapshot_id,
                    SnapshotRetention::branch(None, None, None),
                ),
            },
        ];
        let requirements = vec![
            TableRequirement::UuidMatch {
                uuid: table.metadata().uuid(),
            },
            TableRequirement::RefSnapshotIdMatch {
                r#ref: MAIN_BRANCH.to_string(),
                snapshot_id: table.metadata().current_snapshot_id(),
            },
        ];

        Ok(ActionCommit::new(updates, requirements)
            .with_manifest_paths(vec![new_root_path, fold.bucket_index_path]))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Datum, Struct};

    fn df(path: &str, ts_max: Option<(i32, i64)>) -> DataFile {
        let mut b = DataFileBuilder::default();
        b.content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty());
        if let Some((fid, micros)) = ts_max {
            b.upper_bounds(HashMap::from([(fid, Datum::timestamp_micros(micros))]));
        }
        b.build().unwrap()
    }

    #[test]
    fn max_ts_takes_newest() {
        let fid = 5;
        let files = vec![
            df("a", Some((fid, 100))),
            df("b", Some((fid, 900))),
            df("c", Some((fid, 500))),
        ];
        assert_eq!(max_ts_of(&files, fid), Some(900));
    }

    #[test]
    fn max_ts_none_without_stat() {
        assert_eq!(max_ts_of(&[df("a", None)], 5), None);
    }

    #[test]
    fn file_max_ts_reads_field() {
        assert_eq!(file_max_ts(&df("a", Some((5, 42))), 5), Some(42));
        assert_eq!(file_max_ts(&df("a", Some((5, 42))), 9), None);
    }
}
