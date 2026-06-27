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

//! Compact the **cold tier** of a tiered V4 table (tessellate's job).
//!
//! The hot path keeps the live bucket inline in the root; `graduate_buckets`
//! moves closed buckets into immutable leaf manifests under the bucket-index.
//! Over time a cold partition accumulates many small *data files* (one per live
//! rotation), which makes cold scans slow. The fix is data-file compaction —
//! but the cold files live under the bucket-index, not in the root, so the hot
//! `replace_data_files` (which only touches the root's entries) can't reach
//! them.
//!
//! This action is the cold-tier equivalent, run by the compaction service
//! (tessellate), **off the hot commit path**. The caller does the actual data
//! merge (read N small parquet → write 1 big parquet via streaming concat) and
//! passes the result here:
//!
//!   - `removed`: file paths of the small cold files that were merged away,
//!   - `added`:   the merged replacement `DataFile`s (already written to S3).
//!
//! The action rewrites only the **affected** leaf manifests (those containing a
//! removed file) — dropping the removed entries and re-clustering the survivors
//! plus the merged files per partition — carries unaffected leaves forward by
//! reference, writes a new bucket-index, and points the root at it. The live
//! tier (root inline + refs) is untouched.
//!
//! v1 loads every leaf manifest to find the affected ones. That is read-only
//! and runs at the compaction-service cadence (not per commit); a future
//! optimization can target leaves by partition or a file→leaf index.

use std::collections::{HashMap, HashSet};
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
    DataFile, FormatVersion, ManifestEntry, ManifestFile, ManifestStatus, Operation, Snapshot,
    SnapshotReference, SnapshotRetention, Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Action that compacts the cold tier: replaces `removed` data files (already
/// merged into `added` by the caller) within the bucket-index's leaf manifests.
///
/// Use via `Transaction::compact_cold_tier()`.
pub struct CompactColdTierAction {
    removed: HashSet<String>,
    added: Vec<DataFile>,
    commit_uuid: Uuid,
}

impl CompactColdTierAction {
    /// Create an empty cold-compaction action (no removals or additions yet).
    pub fn new() -> Self {
        Self {
            removed: HashSet::new(),
            added: Vec::new(),
            commit_uuid: Uuid::now_v7(),
        }
    }

    /// Mark cold data-file paths to remove (the small files merged away).
    pub fn remove_files(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.removed.extend(paths);
        self
    }

    /// Add the merged replacement data files (already written to storage).
    pub fn add_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added.extend(files);
        self
    }
}

impl Default for CompactColdTierAction {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter a leaf's data files against the removal set. Returns the survivors and
/// whether any file was removed (i.e. this leaf must be rewritten). Pure.
fn apply_removals(files: Vec<DataFile>, removed: &HashSet<String>) -> (Vec<DataFile>, bool) {
    let mut survivors = Vec::with_capacity(files.len());
    let mut affected = false;
    for f in files {
        if removed.contains(&f.file_path) {
            affected = true;
        } else {
            survivors.push(f);
        }
    }
    (survivors, affected)
}

fn existing_entry(df: DataFile) -> ManifestEntry {
    ManifestEntry::builder()
        .status(ManifestStatus::Existing)
        .data_file(df)
        .build()
}

#[async_trait]
impl TransactionAction for CompactColdTierAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "compact_cold_tier requires format version V4 (effective={:?})",
                    table.effective_format_version()
                ),
            ));
        }
        if self.removed.is_empty() && self.added.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        let current_snapshot = match table.metadata().current_snapshot() {
            Some(s) => s,
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        // Load the root and the cold bucket-index it points to.
        let root_path = current_snapshot.manifest_list();
        let (rm_metadata, root_entries) = reconstruct_root(table.file_io(), root_path).await?;

        let leaves: Vec<ManifestFile> = match &rm_metadata.bucket_index_path {
            Some(path) => {
                let b = table.file_io().new_input(path)?.read().await?;
                read_bucket_index(b)?.leaves().to_vec()
            }
            None => Vec::new(),
        };

        // Partition leaves into unaffected (carry forward) vs affected (rewrite),
        // collecting survivors from the affected ones.
        let mut kept_leaf_refs: Vec<RootManifestEntry> = Vec::new();
        let mut survivors: Vec<ManifestEntry> = Vec::new();
        let mut any_affected = false;

        for leaf in leaves {
            let manifest = leaf.load_manifest(table.file_io()).await?;
            let files: Vec<DataFile> = manifest
                .entries()
                .iter()
                .filter(|e| e.is_alive())
                .map(|e| e.data_file().clone())
                .collect();
            let (leaf_survivors, affected) = apply_removals(files, &self.removed);
            if affected {
                any_affected = true;
                survivors.extend(leaf_survivors.into_iter().map(existing_entry));
            } else {
                // Unaffected — keep the leaf as-is (no rewrite, no re-read cost).
                kept_leaf_refs.push(RootManifestEntry::ManifestRef {
                    manifest_file: leaf,
                    mdv: None,
                });
            }
        }

        // If nothing matched the removal set and there's nothing to add, no-op.
        if !any_affected && self.added.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // Re-cluster the affected survivors + the merged additions into new,
        // partition-tight leaf manifests.
        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let format_version = table.metadata().format_version();
        let commit_uuid = self.commit_uuid;
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(table.metadata().current_schema())?;
        let mut manifest_counter: u64 = 0;

        let mut rewrite_entries = survivors;
        rewrite_entries.extend(self.added.iter().cloned().map(existing_entry));

        let new_leaves = write_entries_clustered(
            table,
            &schema,
            spec.as_ref(),
            format_version,
            snapshot_id,
            commit_uuid,
            &mut manifest_counter,
            false,
            rewrite_entries,
            true, // partition-tight cold leaves
        )
        .await?;

        // New bucket-index = unaffected leaf refs + the freshly written leaves.
        let mut all_leaves: Vec<ManifestFile> = kept_leaf_refs
            .iter()
            .filter_map(|e| match e {
                RootManifestEntry::ManifestRef { manifest_file, .. } => Some(manifest_file.clone()),
                _ => None,
            })
            .collect();
        all_leaves.extend(new_leaves);

        let new_bucket_index_path = if all_leaves.is_empty() {
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
                prev_root_path: None,
                chain_depth: 0,
            };
            let bi_bytes = write_bucket_index(&all_leaves, &bi_metadata, &partition_type)?;
            table
                .file_io()
                .new_output(&path)?
                .write(bi_bytes.into())
                .await?;
            Some(path)
        };

        // New root = the live entries unchanged + the new bucket-index pointer.
        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: new_bucket_index_path.clone(),
            prev_root_path: None,
            chain_depth: 0,
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

        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                ("cold-compact-removed".to_string(), self.removed.len().to_string()),
                ("cold-compact-added".to_string(), self.added.len().to_string()),
                ("cold-compact-leaves-after".to_string(), all_leaves.len().to_string()),
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
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Struct};

    fn df(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .unwrap()
    }

    #[test]
    fn apply_removals_filters_and_flags() {
        let files = vec![df("keep-1"), df("merged-away"), df("keep-2")];
        let removed: HashSet<String> = ["merged-away".to_string()].into_iter().collect();
        let (survivors, affected) = apply_removals(files, &removed);
        assert!(affected);
        let paths: Vec<&str> = survivors.iter().map(|f| f.file_path.as_str()).collect();
        assert_eq!(paths, vec!["keep-1", "keep-2"]);
    }

    #[test]
    fn apply_removals_unaffected_leaf() {
        let files = vec![df("a"), df("b")];
        let removed: HashSet<String> = ["x".to_string()].into_iter().collect();
        let (survivors, affected) = apply_removals(files, &removed);
        assert!(!affected);
        assert_eq!(survivors.len(), 2);
    }

    #[test]
    fn builder_accumulates() {
        let a = CompactColdTierAction::new()
            .remove_files(["p1".to_string(), "p2".to_string()])
            .add_files([df("merged")]);
        assert_eq!(a.removed.len(), 2);
        assert_eq!(a.added.len(), 1);
    }
}
