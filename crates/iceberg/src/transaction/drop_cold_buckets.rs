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
//! Old data lives in immutable leaf manifests under the bucket-index. Retention
//! drops the leaves whose data is entirely older than a cutoff, trims the
//! bucket-index, and repoints the root — **never touching the root's live
//! entries or any live data file**. The dropped leaves + their data files become
//! unreferenced and are reclaimed by a separate orphan-GC pass.
//!
//! The cutoff is **time**, which is universal — it does NOT depend on how the
//! table is partitioned. A leaf's event-time is read from its data files'
//! statistics for a configured timestamp field (`ts_field_id`), so retention is
//! fully partition-spec-agnostic: a table partitioned by tenant, by region, by
//! day, or not at all all retain identically.
//!
//! A leaf is KEPT when its max event-time is at/after the cutoff (it still holds
//! data within the window); leaves whose max event-time has fallen behind the
//! cutoff are dropped. A leaf lacking the timestamp statistic is kept
//! (conservative — never drop data we can't time-bound).
//!
//! Cost: O(bucket-index leaves) manifest reads per pass (to read each leaf's
//! timestamp stat) — read-only and run at the retention cadence, never on the
//! hot path. (A future optimization caches each leaf's ts bound in the
//! bucket-index row, making this O(1).)

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    reconstruct_root, write_root_manifest, RootManifestMetadata,
};
use crate::spec::{
    DataFile, FormatVersion, ManifestFile, Operation, PrimitiveLiteral, Snapshot,
    SnapshotReference, SnapshotRetention, Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Action that drops cold leaves whose data is entirely older than the cutoff
/// (time-based retention), trims the bucket-index, and repoints the root.
///
/// Use via `Transaction::drop_cold_buckets(ts_field_id, cutoff_micros)`.
pub struct DropColdBucketsAction {
    ts_field_id: i32,
    cutoff_micros: i64,
    commit_uuid: Uuid,
}

impl DropColdBucketsAction {
    /// Create with the timestamp field id and the retention cutoff (micros).
    /// Leaves whose max value for `ts_field_id` is `< cutoff_micros` are dropped.
    pub fn new(ts_field_id: i32, cutoff_micros: i64) -> Self {
        Self {
            ts_field_id,
            cutoff_micros,
            commit_uuid: Uuid::now_v7(),
        }
    }
}

/// Max value of the `ts_field_id` upper-bound statistic across `files` (event
/// time of the newest row). `None` if no file carries the stat. Pure.
fn max_ts_of(files: &[DataFile], ts_field_id: i32) -> Option<i64> {
    files
        .iter()
        .filter_map(|f| match f.upper_bounds().get(&ts_field_id).map(|d| d.literal()) {
            Some(PrimitiveLiteral::Long(v)) => Some(*v),
            _ => None,
        })
        .max()
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
        let (rm_metadata, root_entries) = reconstruct_root(table.file_io(), root_path).await?;

        let leaves: Vec<ManifestFile> = match &rm_metadata.bucket_index_path {
            Some(path) => {
                let b = table.file_io().new_input(path)?.read().await?;
                read_bucket_index(b)?.leaves().to_vec()
            }
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        // Keep a leaf when its newest event-time is at/after the cutoff (or it
        // lacks the stat — conservative). Reading the leaf's data-file stats
        // makes the decision partition-agnostic.
        let mut kept: Vec<ManifestFile> = Vec::new();
        let mut dropped: Vec<ManifestFile> = Vec::new();
        for leaf in leaves {
            let manifest = leaf.load_manifest(table.file_io()).await?;
            let files: Vec<DataFile> = manifest
                .entries()
                .iter()
                .filter(|e| e.is_alive())
                .map(|e| e.data_file().clone())
                .collect();
            let keep = match max_ts_of(&files, self.ts_field_id) {
                Some(max_ts) => max_ts >= self.cutoff_micros,
                None => true,
            };
            if keep {
                kept.push(leaf);
            } else {
                dropped.push(leaf);
            }
        }

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
                prev_root_path: None,
                chain_depth: 0,
                node_level: 0,
            };
            let bi_bytes = write_bucket_index(&kept, &bi_metadata, &partition_type)?;
            table
                .file_io()
                .new_output(&path)?
                .write(bi_bytes.into())
                .await?;
            Some(path)
        };

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
            node_level: 0,
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
                ("retention-cutoff-micros".to_string(), self.cutoff_micros.to_string()),
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
        let files = vec![df("a", None), df("b", None)];
        assert_eq!(max_ts_of(&files, 5), None);
    }

    #[test]
    fn max_ts_ignores_other_fields() {
        // Only field 5 carries a stat; asking for field 9 → None.
        let files = vec![df("a", Some((5, 100)))];
        assert_eq!(max_ts_of(&files, 9), None);
    }
}
