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
    PartitionSpec, PrimitiveLiteral, Snapshot, SnapshotReference, SnapshotRetention, Summary,
    Transform, MAIN_BRANCH,
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

/// Push a dropped data file AND its per-file puffin sidecar into the tombstone
/// set. A data file `X.parquet` carries a stats sidecar at `X.parquet.stats`
/// (see merge_puffin); it orphans when the data file is dropped, so tombstone
/// both. A file with no sidecar → the extra path is an idempotent no-op delete.
fn push_data_file_and_sidecar(paths: &mut Vec<String>, data_file_path: &str) {
    paths.push(format!("{data_file_path}.stats"));
    paths.push(data_file_path.to_string());
}

fn le_i32(b: &[u8]) -> Option<i64> {
    (b.len() >= 4).then(|| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
}
fn le_i64(b: &[u8]) -> Option<i64> {
    (b.len() >= 8).then(|| i64::from_le_bytes(b[..8].try_into().unwrap()))
}

/// Exclusive upper bound (micros) of event time covered by a partition
/// upper-bound value (Iceberg little-endian single-value encoding), for the
/// contiguous, monotonic time transforms. `None` for transforms whose bucket
/// doesn't map to a simple time range (month/year variable width, bucket,
/// truncate, …) — the caller then falls back to loading the manifest. Pure.
fn transform_upper_micros(transform: &Transform, bytes: &[u8]) -> Option<i64> {
    const HOUR_US: i64 = 3_600 * 1_000_000;
    const DAY_US: i64 = 86_400 * 1_000_000;
    match transform {
        // hour/day: value is the bucket ordinal from the epoch; the newest
        // possible event is the END of that bucket (exclusive).
        Transform::Hour => Some((le_i32(bytes)? + 1) * HOUR_US),
        Transform::Day => Some((le_i32(bytes)? + 1) * DAY_US),
        // identity on a timestamp column: the bound IS the max event time.
        Transform::Identity => le_i64(bytes),
        _ => None,
    }
}

/// Newest event time (micros, exclusive upper bound) a manifest can hold,
/// derived from its PARTITION SUMMARY for the `ts_field_id`-derived partition
/// field — WITHOUT loading the manifest. `None` when the timestamp isn't
/// partitioned by a supported time transform (caller falls back to the manifest).
/// This is what keeps graduation's classification O(refs) cheap reads instead of
/// O(refs) full manifest loads. Pure.
fn ref_max_event_micros(mf: &ManifestFile, spec: &PartitionSpec, ts_field_id: i32) -> Option<i64> {
    let parts = mf.partitions.as_ref()?;
    for (idx, pf) in spec.fields().iter().enumerate() {
        if pf.source_id != ts_field_id {
            continue;
        }
        let ub = parts.get(idx)?.upper_bound.as_ref()?;
        if let Some(m) = transform_upper_micros(&pf.transform, ub) {
            return Some(m);
        }
    }
    None
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
    /// TTL-drop (Type 1 retention): S3 object paths that this fold removed from
    /// the tree because their newest `ts_field_id` value fell below the retention
    /// cutoff — the data files (and the manifest/leaf files that held them). The
    /// caller surfaces these to the sole tombstone writer (laminar) so they are
    /// reclaimed after grace. Empty unless a retention cutoff was supplied.
    pub ttl_dropped_paths: Vec<String>,
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
    max_graduate: Option<usize>,
    // TTL (Type 1 retention): entries/leaves whose newest `ts_field_id` value is
    // below this cutoff are DROPPED from the tree (not graduated, not kept) and
    // their object paths returned in `FoldOutcome::ttl_dropped_paths`. `None`
    // disables TTL (graduation-only fold). `max_ttl_drop_files` bounds how many
    // paths one fold may drop so the caller's snapshot summary stays small — the
    // rest drop on later collapses (eventually consistent).
    retention_cutoff_micros: Option<i64>,
    max_ttl_drop_files: usize,
    snapshot_id: i64,
    commit_uuid: Uuid,
    manifest_counter: &mut u64,
) -> Result<(Vec<RootManifestEntry>, Option<FoldOutcome>)> {
    let schema = table.metadata().current_schema().clone();
    let format_version = table.metadata().format_version();
    let spec = table.metadata().default_partition_spec().clone();
    let partition_type = spec.partition_type(&schema)?;
    let next_seq_num = table.metadata().next_sequence_number();

    // Accumulates TTL-dropped object paths (data files + the manifest/leaf files
    // that referenced them), bounded by `max_ttl_drop_files`. `ttl_budget_left`
    // returns whether another leaf/entry may still be dropped this fold.
    let mut ttl_dropped_paths: Vec<String> = Vec::new();
    let ttl_budget_left =
        |dropped: &Vec<String>| retention_cutoff_micros.is_some() && dropped.len() < max_ttl_drop_files;

    // Existing cold leaves (graduated nodes get appended to these).
    let mut cold_leaves: Vec<ManifestFile> = match carried_bucket_index_path {
        Some(path) => {
            let b = table.file_io().new_input(path)?.read().await?;
            read_bucket_index(b)?.leaves().to_vec()
        }
        None => Vec::new(),
    };

    // ── TTL prune of the cold tier (steady-state path). Data ages into cold via
    // graduation long before it hits retention (retention ≫ bucket-window), so
    // expired data almost always lives here. Drop each cold leaf whose newest
    // `ts_field_id` value is below the retention cutoff; tombstone its data files
    // and the leaf manifest itself. Bounded by the file budget. ──
    if let Some(retention_cutoff) = retention_cutoff_micros {
        let cold_total = cold_leaves.len();
        let mut with_ts_count = 0usize;
        let mut none_ts_count = 0usize;
        let mut newest_max_ts: Option<i64> = None;
        let mut oldest_max_ts: Option<i64> = None;
        let mut surviving: Vec<ManifestFile> = Vec::with_capacity(cold_leaves.len());
        // Coldest-first so the budget reclaims the oldest data before newer.
        let mut with_ts: Vec<(ManifestFile, Option<i64>)> = Vec::with_capacity(cold_leaves.len());
        for leaf in std::mem::take(&mut cold_leaves) {
            let mx = match ref_max_event_micros(&leaf, &spec, ts_field_id) {
                Some(m) => Some(m),
                None => {
                    let manifest = leaf.load_manifest(table.file_io()).await?;
                    let files: Vec<DataFile> = manifest
                        .entries()
                        .iter()
                        .filter(|e| e.is_alive())
                        .map(|e| e.data_file().clone())
                        .collect();
                    max_ts_of(&files, ts_field_id)
                }
            };
            match mx {
                Some(m) => {
                    with_ts_count += 1;
                    newest_max_ts = Some(newest_max_ts.map_or(m, |n| n.max(m)));
                    oldest_max_ts = Some(oldest_max_ts.map_or(m, |o| o.min(m)));
                }
                None => none_ts_count += 1,
            }
            with_ts.push((leaf, mx));
        }
        with_ts.sort_by_key(|(_, mx)| mx.unwrap_or(i64::MAX));
        let mut dropped_leaves = 0usize;
        for (leaf, mx) in with_ts {
            let expired = mx.map(|m| m < retention_cutoff).unwrap_or(false);
            if expired && ttl_budget_left(&ttl_dropped_paths) {
                let manifest = leaf.load_manifest(table.file_io()).await?;
                for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                    push_data_file_and_sidecar(&mut ttl_dropped_paths, e.data_file().file_path());
                }
                ttl_dropped_paths.push(leaf.manifest_path.clone());
                dropped_leaves += 1;
            } else {
                surviving.push(leaf);
            }
        }
        // Diagnostic: makes the TTL prune self-explaining. If dropped=0 despite
        // old data, this line disambiguates the cause: ts_field_id resolution
        // (none_ts high), poisoned bounds (oldest_max_ts recent/future), or an
        // already-pruned cold tier (cold_total low).
        log::info!(
            "ttl cold-leaf prune: ts_field_id={} cutoff_us={} cold_leaves={} with_ts={} none_ts={} oldest_max_ts={:?} newest_max_ts={:?} dropped_leaves={} dropped_paths={}",
            ts_field_id, retention_cutoff, cold_total, with_ts_count, none_ts_count,
            oldest_max_ts, newest_max_ts, dropped_leaves, ttl_dropped_paths.len()
        );
        cold_leaves = surviving;
    }

    // Partition into: closed live nodes (→ cold by reference), closed inline
    // files (→ materialize as cold leaves), and kept (stay hot). `closed_refs`
    // carries each node's newest event time so we can graduate the COLDEST
    // first when the per-collapse cap trims the batch.
    let mut kept: Vec<RootManifestEntry> = Vec::new();
    let mut closed_refs: Vec<(ManifestFile, i64)> = Vec::new();
    let mut closed_inline_files: Vec<DataFile> = Vec::new();

    for entry in entries {
        match entry {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                // Only graduate clean nodes; an MDV-carrying node has pending
                // deletes and is left for the rebalance/compaction path.
                if mdv.is_some() {
                    kept.push(RootManifestEntry::ManifestRef { manifest_file, mdv });
                    continue;
                }
                // Fast path: newest event time straight from the partition
                // summary (no manifest read). Fall back to loading the manifest
                // only when the ts field isn't partitioned by a time transform.
                let max_micros = match ref_max_event_micros(&manifest_file, &spec, ts_field_id) {
                    Some(m) => Some(m),
                    None => {
                        let manifest = manifest_file.load_manifest(table.file_io()).await?;
                        let files: Vec<DataFile> = manifest
                            .entries()
                            .iter()
                            .filter(|e| e.is_alive())
                            .map(|e| e.data_file().clone())
                            .collect();
                        max_ts_of(&files, ts_field_id)
                    }
                };
                // Expired past retention (edge case for hot data: normally it
                // graduates to cold long before this). Drop + tombstone rather
                // than graduate. `retention_cutoff < cutoff_micros`, so an expired
                // ref is always also "closed".
                let expired = matches!((retention_cutoff_micros, max_micros),
                    (Some(rc), Some(mx)) if mx < rc);
                if expired && ttl_budget_left(&ttl_dropped_paths) {
                    let manifest = manifest_file.load_manifest(table.file_io()).await?;
                    for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                        push_data_file_and_sidecar(&mut ttl_dropped_paths, e.data_file().file_path());
                    }
                    ttl_dropped_paths.push(manifest_file.manifest_path.clone());
                    continue;
                }
                match max_micros {
                    Some(mx) if mx < cutoff_micros => closed_refs.push((manifest_file, mx)),
                    _ => kept.push(RootManifestEntry::ManifestRef { manifest_file, mdv }),
                }
            }
            RootManifestEntry::Inline(me) => {
                let mx = file_max_ts(&me.data_file, ts_field_id);
                let expired = matches!((retention_cutoff_micros, mx),
                    (Some(rc), Some(m)) if m < rc);
                if expired && ttl_budget_left(&ttl_dropped_paths) {
                    push_data_file_and_sidecar(&mut ttl_dropped_paths, me.data_file.file_path());
                    continue;
                }
                let closed = mx.map(|m| m < cutoff_micros).unwrap_or(false);
                if closed {
                    closed_inline_files.push(me.data_file.clone());
                } else {
                    kept.push(RootManifestEntry::Inline(me));
                }
            }
        }
    }

    // Bound the graduation per collapse. On a long-history table the first
    // graduation can otherwise move most of the table in one commit; cap it and
    // graduate the COLDEST refs first — the rest stay hot and graduate on later
    // collapses (eventually consistent, bounded per-commit work).
    if let Some(cap) = max_graduate {
        if closed_refs.len() > cap {
            closed_refs.sort_by_key(|(_, mx)| *mx);
            for (mf, _) in closed_refs.split_off(cap) {
                kept.push(RootManifestEntry::ManifestRef {
                    manifest_file: mf,
                    mdv: None,
                });
            }
        }
    }
    let graduated_nodes: Vec<ManifestFile> = closed_refs.into_iter().map(|(mf, _)| mf).collect();

    // Nothing to do only if no graduation AND no TTL prune happened. A TTL-only
    // fold (cold leaves dropped, nothing graduated) still must persist the
    // pruned bucket-index and return the dropped paths.
    if graduated_nodes.is_empty()
        && closed_inline_files.is_empty()
        && ttl_dropped_paths.is_empty()
    {
        return Ok((kept, None));
    }
    let nodes_moved = graduated_nodes.len();

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
            ttl_dropped_paths,
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
            None, // manual/maintenance use: no per-collapse cap
            None, // TTL retention drop is driven only by the commit_v4 collapse
            0,
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

    #[test]
    fn transform_upper_micros_decodes_time_buckets() {
        const H: i64 = 3_600 * 1_000_000;
        const D: i64 = 86_400 * 1_000_000;
        // hour ordinal 10 → exclusive upper = END of hour 10 = hour 11 start.
        assert_eq!(
            transform_upper_micros(&Transform::Hour, &10i32.to_le_bytes()),
            Some(11 * H)
        );
        // day ordinal 3 → end of day 3 = day 4 start.
        assert_eq!(
            transform_upper_micros(&Transform::Day, &3i32.to_le_bytes()),
            Some(4 * D)
        );
        // identity on a timestamp column: bound IS the max event time (micros).
        assert_eq!(
            transform_upper_micros(&Transform::Identity, &1_700_000_000_000_000i64.to_le_bytes()),
            Some(1_700_000_000_000_000)
        );
        // variable-width / non-time transforms fall back (None → load manifest).
        assert_eq!(
            transform_upper_micros(&Transform::Month, &10i32.to_le_bytes()),
            None
        );
        // truncated/garbage bytes → None (safe fallback), never a bogus bound.
        assert_eq!(transform_upper_micros(&Transform::Hour, &[1u8, 2]), None);
        assert_eq!(transform_upper_micros(&Transform::Identity, &[1u8, 2, 3, 4]), None);
    }
}
