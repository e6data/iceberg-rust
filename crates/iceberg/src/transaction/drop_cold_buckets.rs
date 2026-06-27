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

//! Time-based retention for the cold tier of a tiered V4 table.
//!
//! Old data lives in immutable leaf manifests referenced by the bucket-index,
//! each bucket-index row carrying that leaf's partition/time summary. Enforcing
//! retention is therefore a cheap, metadata-only filter on the bucket-index —
//! it never touches the root's live entries or any live data file:
//!
//!   1. read the bucket-index,
//!   2. keep the leaves the predicate says to keep, drop the rest,
//!   3. write a trimmed bucket-index and repoint the root at it, one commit.
//!
//! The dropped leaf manifests (and the data files they reference) become
//! unreferenced and are reclaimed by a separate orphan-GC pass — keeping this
//! action O(bucket-index rows) with **no leaf loads and no data reads**.
//!
//! The predicate is supplied by the caller (`Fn(&ManifestFile) -> bool`, "keep
//! this leaf?"), which decodes the leaf's partition summary against a retention
//! cutoff using the table's partition spec. Keeping the time interpretation in
//! the caller makes this action partition-spec-agnostic — same shape whether
//! buckets are day-, hour-, or otherwise-partitioned.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    read_root_manifest, write_root_manifest, RootManifestMetadata,
};
use crate::spec::{
    FormatVersion, ManifestFile, Operation, Snapshot, SnapshotReference, SnapshotRetention,
    Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Predicate over a cold leaf: return `true` to KEEP it, `false` to drop it for
/// retention. The caller builds this from the leaf's partition/time summary and
/// the retention cutoff.
pub type KeepLeafPredicate = Arc<dyn Fn(&ManifestFile) -> bool + Send + Sync>;

/// Action that drops cold leaves failing the keep-predicate (retention), trims
/// the bucket-index, and repoints the root. Live tier untouched.
///
/// Use via `Transaction::drop_cold_buckets(keep)`.
pub struct DropColdBucketsAction {
    keep: KeepLeafPredicate,
    commit_uuid: Uuid,
}

impl DropColdBucketsAction {
    /// Create with the keep-predicate (`true` ⇒ retain the leaf).
    pub fn new(keep: KeepLeafPredicate) -> Self {
        Self {
            keep,
            commit_uuid: Uuid::now_v7(),
        }
    }
}

/// Split leaves into (kept, dropped) by the keep-predicate. Pure.
fn partition_leaves(
    leaves: Vec<ManifestFile>,
    keep: &(dyn Fn(&ManifestFile) -> bool + Send + Sync),
) -> (Vec<ManifestFile>, Vec<ManifestFile>) {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for leaf in leaves {
        if keep(&leaf) {
            kept.push(leaf);
        } else {
            dropped.push(leaf);
        }
    }
    (kept, dropped)
}

#[async_trait]
impl TransactionAction for DropColdBucketsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "drop_cold_buckets requires format version V4 (effective={:?})",
                    table.effective_format_version()
                ),
            ));
        }

        let current_snapshot = match table.metadata().current_snapshot() {
            Some(s) => s,
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        let root_path = current_snapshot.manifest_list();
        let bytes = table.file_io().new_input(root_path)?.read().await?;
        let (rm_metadata, root_entries) = read_root_manifest(bytes)?;

        // No cold tier => nothing to retain.
        let leaves: Vec<ManifestFile> = match &rm_metadata.bucket_index_path {
            Some(path) => {
                let b = table.file_io().new_input(path)?.read().await?;
                read_bucket_index(b)?.leaves().to_vec()
            }
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        let (kept, dropped) = partition_leaves(leaves, self.keep.as_ref());
        if dropped.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }
        let dropped_count = dropped.len();

        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(table.metadata().current_schema())?;
        let commit_uuid = self.commit_uuid;

        // Trimmed bucket-index (None if everything aged out).
        let new_bucket_index_path = if kept.is_empty() {
            None
        } else {
            let path = format!(
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
            };
            let bi_bytes = write_bucket_index(&kept, &bi_metadata, &partition_type)?;
            table
                .file_io()
                .new_output(&path)?
                .write(bi_bytes.into())
                .await?;
            Some(path)
        };

        // New root = live entries unchanged + trimmed bucket-index pointer.
        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: new_bucket_index_path.clone(),
        };
        let new_root_path = format!(
            "{}/{}/root-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            snapshot_id,
            commit_uuid,
        );
        let root_bytes = write_root_manifest(&root_entries, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_path)?
            .write(root_bytes.into())
            .await?;

        // Record dropped leaf paths in the summary so an orphan-GC pass can
        // reclaim their manifests + data files.
        let dropped_paths = dropped
            .iter()
            .map(|m| m.manifest_path.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                ("retention-leaves-dropped".to_string(), dropped_count.to_string()),
                ("retention-leaves-kept".to_string(), kept.len().to_string()),
                ("retention-dropped-leaf-paths".to_string(), dropped_paths),
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

        let mut manifest_paths = vec![new_root_path];
        if let Some(p) = new_bucket_index_path {
            manifest_paths.push(p);
        }
        Ok(ActionCommit::new(updates, requirements).with_manifest_paths(manifest_paths))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ManifestContentType, ManifestFile};

    fn leaf(path: &str, seq: i64) -> ManifestFile {
        ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: 1,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: seq,
            min_sequence_number: seq,
            added_snapshot_id: 1,
            added_files_count: Some(1),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        }
    }

    #[test]
    fn partition_leaves_keeps_and_drops() {
        let leaves = vec![leaf("old-1", 1), leaf("new", 9), leaf("old-2", 2)];
        // Keep leaves whose sequence_number >= 5 (stand-in for "newer than cutoff").
        let (kept, dropped) = partition_leaves(leaves, &|m: &ManifestFile| m.sequence_number >= 5);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].manifest_path, "new");
        let dpaths: Vec<&str> = dropped.iter().map(|m| m.manifest_path.as_str()).collect();
        assert_eq!(dpaths, vec!["old-1", "old-2"]);
    }

    #[test]
    fn partition_leaves_nothing_dropped() {
        let leaves = vec![leaf("a", 9), leaf("b", 9)];
        let (kept, dropped) = partition_leaves(leaves, &|_m: &ManifestFile| true);
        assert_eq!(kept.len(), 2);
        assert!(dropped.is_empty());
    }

    #[test]
    fn partition_leaves_all_dropped() {
        let leaves = vec![leaf("a", 1), leaf("b", 1)];
        let (kept, dropped) = partition_leaves(leaves, &|_m: &ManifestFile| false);
        assert!(kept.is_empty());
        assert_eq!(dropped.len(), 2);
    }
}
