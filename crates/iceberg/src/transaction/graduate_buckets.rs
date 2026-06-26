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

//! Graduate closed live blocks into the cold bucket-index (tiered V4 metadata).
//!
//! The hot commit path keeps the current bucket's data as inline entries in the
//! root manifest. When a bucket *closes* (a policy the caller owns — every N
//! hours, every M commits, …), this action moves the closed data out of the
//! hot root and into immutable, partition-tight **leaf manifests** referenced by
//! the cold **bucket-index**:
//!
//! ```text
//!   before:  root = [ inline(closed) … , inline(open) … ]   (+ maybe a bucket-index)
//!   after:   root = [ inline(open) … ]  ──bucket_index_ptr──► bucket-index
//!                                                               + new closed leaves
//! ```
//!
//! This is what keeps per-commit cost bounded by the *live* window: the bucket-
//! index (O(total closed leaves)) is rewritten only here, at bucket-close
//! cadence — never on the hot commit path.
//!
//! The action is cadence- and partition-agnostic: the caller supplies an
//! `is_closed(&Struct) -> bool` predicate over partition values.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use super::rebalance_root_manifest::write_entries_clustered;
use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    read_root_manifest, write_root_manifest, RootManifestEntry, RootManifestMetadata,
};
use crate::spec::{
    DataContentType, DataFile, FormatVersion, ManifestEntry, ManifestFile, ManifestStatus,
    Operation, PrimitiveLiteral, Snapshot, SnapshotReference, SnapshotRetention, Summary,
    MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Predicate deciding whether a data file belongs to a *closed* bucket — i.e.
/// it should graduate to cold storage. Evaluated per file so the close policy
/// can key on the file's own statistics (typically its timestamp max), which
/// makes it independent of the partition spec. See [`closed_before`].
pub type ClosedPredicate = Arc<dyn Fn(&DataFile) -> bool + Send + Sync>;

/// Build a time-cutoff close predicate: a file is closed when its **max value**
/// for `timestamp_field_id` is strictly below `cutoff_micros`. Files lacking
/// that statistic stay live (conservative — never graduate data we can't
/// time-bound). This is the partition-agnostic close policy the scheduler uses;
/// the caller computes `cutoff_micros = now − bucket_window` and resolves the
/// timestamp field id from the schema.
pub fn closed_before(timestamp_field_id: i32, cutoff_micros: i64) -> ClosedPredicate {
    Arc::new(move |df: &DataFile| {
        df.upper_bounds()
            .get(&timestamp_field_id)
            .map(|d| matches!(d.literal(), PrimitiveLiteral::Long(v) if *v < cutoff_micros))
            .unwrap_or(false)
    })
}

/// Action that graduates closed live blocks into the cold bucket-index.
///
/// Use via `Transaction::graduate_buckets(is_closed)`.
pub struct GraduateBucketsAction {
    is_closed: ClosedPredicate,
    commit_uuid: Uuid,
}

impl GraduateBucketsAction {
    /// Create with the close predicate. `is_closed(partition)` returns true for
    /// partition values whose bucket has closed and should move to cold storage.
    pub fn new(is_closed: ClosedPredicate) -> Self {
        Self {
            is_closed,
            commit_uuid: Uuid::now_v7(),
        }
    }
}

/// Split root entries into (data files to graduate to cold leaves, entries that
/// stay live in the root). Only **inline data** entries with a closed partition
/// graduate; manifest refs, delete entries, and open inline entries stay.
///
/// Pure — no I/O — so the close decision is unit testable.
fn split_for_graduation(
    entries: &[RootManifestEntry],
    is_closed: &(dyn Fn(&DataFile) -> bool + Send + Sync),
) -> (Vec<DataFile>, Vec<RootManifestEntry>) {
    let mut graduate = Vec::new();
    let mut stay = Vec::new();
    for entry in entries {
        match entry {
            RootManifestEntry::Inline(me)
                if me.data_file.content == DataContentType::Data
                    && is_closed(&me.data_file) =>
            {
                graduate.push(me.data_file.clone());
            }
            other => stay.push(other.clone()),
        }
    }
    (graduate, stay)
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

        // Load current root.
        let root_path = current_snapshot.manifest_list();
        let bytes = table.file_io().new_input(root_path)?.read().await?;
        let (rm_metadata, entries) = read_root_manifest(bytes)?;

        // Decide what graduates (pure).
        let (graduate_files, stay_entries) =
            split_for_graduation(&entries, self.is_closed.as_ref());
        if graduate_files.is_empty() {
            // Nothing closed this round.
            return Ok(ActionCommit::new(vec![], vec![]));
        }
        let graduated_file_count = graduate_files.len();

        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let format_version = table.metadata().format_version();
        let commit_uuid = self.commit_uuid;
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(table.metadata().current_schema())?;
        let mut manifest_counter: u64 = 0;

        // 1. Write graduating data files as cold, partition-tight leaf manifests.
        let grad_entries: Vec<ManifestEntry> = graduate_files
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
            &mut manifest_counter,
            false, // data, not deletes
            grad_entries,
            true, // partition_scoped: cold leaves are one-per-partition (tight summaries)
        )
        .await?;
        let new_leaf_count = new_leaves.len();

        // 2. Append the new leaves to the bucket-index (read existing if present).
        let mut leaves: Vec<ManifestFile> = match &rm_metadata.bucket_index_path {
            Some(path) => {
                let b = table.file_io().new_input(path)?.read().await?;
                read_bucket_index(b)?.leaves().to_vec()
            }
            None => Vec::new(),
        };
        leaves.extend(new_leaves);

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
            // The bucket-index is itself a single tier today (no nested pointer).
            bucket_index_path: None,
        };
        let bi_bytes = write_bucket_index(&leaves, &bi_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&bucket_index_path)?
            .write(bi_bytes.into())
            .await?;

        // 3. New root = the entries that stayed live + the new bucket-index pointer.
        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: Some(bucket_index_path.clone()),
        };
        let new_root_path = format!(
            "{}/{}/root-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            snapshot_id,
            commit_uuid,
        );
        let root_bytes = write_root_manifest(&stay_entries, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_path)?
            .write(root_bytes.into())
            .await?;

        // 4. Build snapshot + ActionCommit.
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "graduate-files".to_string(),
                    graduated_file_count.to_string(),
                ),
                ("graduate-leaves-added".to_string(), new_leaf_count.to_string()),
                (
                    "graduate-leaves-total".to_string(),
                    leaves.len().to_string(),
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
            .with_manifest_paths(vec![new_root_path, bucket_index_path]))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Datum, ManifestContentType,
        ManifestFile, Struct,
    };

    /// Build an inline data entry. `ts_max` optionally sets an upper-bound
    /// statistic (field_id, micros) so `closed_before` can be exercised.
    fn inline_with(path: &str, ts_max: Option<(i32, i64)>) -> RootManifestEntry {
        let mut builder = DataFileBuilder::default();
        builder
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty());
        if let Some((fid, micros)) = ts_max {
            builder.upper_bounds(HashMap::from([(fid, Datum::timestamp_micros(micros))]));
        }
        let df = builder.build().unwrap();
        RootManifestEntry::Inline(
            ManifestEntry::builder()
                .status(ManifestStatus::Added)
                .data_file(df)
                .build(),
        )
    }

    fn a_ref() -> RootManifestEntry {
        RootManifestEntry::ManifestRef {
            manifest_file: ManifestFile {
                manifest_path: "s3://b/leaf.parquet".to_string(),
                manifest_length: 1,
                partition_spec_id: 0,
                content: ManifestContentType::Data,
                sequence_number: 1,
                min_sequence_number: 1,
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
            },
            mdv: None,
        }
    }

    #[test]
    fn graduates_only_closed_inline_data() {
        let entries = vec![
            inline_with("s3://b/data/closed-a.parquet", None),
            inline_with("s3://b/data/open.parquet", None),
            inline_with("s3://b/data/closed-b.parquet", None),
            a_ref(), // refs always stay
        ];

        // Close decision can key on anything in the DataFile.
        let (grad, stay) =
            split_for_graduation(&entries, &|df: &DataFile| df.file_path.contains("closed"));

        assert_eq!(grad.len(), 2);
        let paths: Vec<&str> = grad.iter().map(|d| d.file_path.as_str()).collect();
        assert!(paths.contains(&"s3://b/data/closed-a.parquet"));
        assert!(paths.contains(&"s3://b/data/closed-b.parquet"));
        assert_eq!(stay.len(), 2);
    }

    #[test]
    fn closed_before_uses_timestamp_max() {
        let fid = 5;
        let entries = vec![
            inline_with("closed.parquet", Some((fid, 1_000))), // max ts < cutoff
            inline_with("open.parquet", Some((fid, 9_000))),   // max ts >= cutoff
            inline_with("nostat.parquet", None),               // no ts -> stays live
        ];

        let pred = closed_before(fid, 5_000);
        let (grad, stay) = split_for_graduation(&entries, pred.as_ref());

        assert_eq!(grad.len(), 1);
        assert_eq!(grad[0].file_path, "closed.parquet");
        assert_eq!(stay.len(), 2);
    }

    #[test]
    fn refs_never_graduate_even_if_predicate_true() {
        let (grad, stay) = split_for_graduation(&[a_ref()], &|_df: &DataFile| true);
        assert!(grad.is_empty());
        assert_eq!(stay.len(), 1);
    }
}
