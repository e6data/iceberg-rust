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

//! Background rebalancing for V4 root manifests.
//!
//! Over time, fast-append commits accumulate inline entries in the root manifest.
//! When the inline count exceeds a threshold, this action flushes them into child
//! manifest files grouped by partition spec, replacing the inline entries with
//! manifest references. It also compacts child manifests whose manifest delete
//! vector (MDV) deleted fraction exceeds a threshold, rewriting them without the
//! deleted entries.
//!
//! This is a metadata-only operation — no data files are read or written.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::root_manifest::{
    ManifestDeleteVector, RootManifest, RootManifestEntry, RootManifestMetadata,
    read_root_manifest, write_root_manifest,
};
use crate::spec::{
    DataContentType, FormatVersion, ManifestContentType, ManifestEntry, ManifestFile,
    ManifestStatus, ManifestWriterBuilder, Operation, Snapshot, SnapshotReference,
    SnapshotRetention, Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Default number of inline entries before triggering a flush to child manifests.
const DEFAULT_INLINE_THRESHOLD: usize = 1000;

/// Default fraction of MDV-deleted entries in a child manifest before rewriting it.
const DEFAULT_MDV_COMPACTION_THRESHOLD: f64 = 0.3;

const META_ROOT_PATH: &str = "metadata";

/// Action that rebalances a V4 root manifest by flushing accumulated inline
/// entries into child manifest files and compacting MDV-heavy manifest refs.
///
/// Use via `Transaction::rebalance_root_manifest()` or apply directly.
///
/// # When to run
///
/// Schedule periodically (e.g. every N commits or on a timer) when the table
/// uses V4 format and streaming ingest accumulates many inline entries.
///
/// # What it does
///
/// 1. Loads the current root manifest from the table's current snapshot.
/// 2. If `inline_count < inline_threshold` AND no MDV exceeds the compaction
///    threshold, returns an empty `ActionCommit` (no-op).
/// 3. Flushes inline entries into child manifest files, grouped by content type
///    (data vs. delete) and partition spec.
/// 4. For manifest refs where the MDV deleted fraction exceeds the compaction
///    threshold, loads the child manifest, filters out deleted entries, writes
///    a new manifest, and replaces the ref.
/// 5. Writes a new root manifest containing only manifest references.
/// 6. Returns an `ActionCommit` with the new snapshot.
pub struct RebalanceRootManifestAction {
    /// Number of inline entries that triggers a flush.
    inline_threshold: usize,
    /// Fraction (0.0..1.0) of MDV-deleted entries that triggers a manifest rewrite.
    mdv_compaction_threshold: f64,
    /// UUID for generating unique file paths in this commit.
    commit_uuid: Uuid,
}

impl RebalanceRootManifestAction {
    /// Create a new action with default thresholds.
    pub fn new() -> Self {
        Self {
            inline_threshold: DEFAULT_INLINE_THRESHOLD,
            mdv_compaction_threshold: DEFAULT_MDV_COMPACTION_THRESHOLD,
            commit_uuid: Uuid::now_v7(),
        }
    }

    /// Override the inline entry count threshold.
    pub fn with_inline_threshold(mut self, threshold: usize) -> Self {
        self.inline_threshold = threshold;
        self
    }

    /// Override the MDV compaction threshold (fraction of deleted entries).
    pub fn with_mdv_compaction_threshold(mut self, threshold: f64) -> Self {
        self.mdv_compaction_threshold = threshold;
        self
    }

    /// Check whether any manifest ref has an MDV whose deleted fraction exceeds
    /// the compaction threshold, given the total entry count from the manifest file.
    fn needs_mdv_compaction(
        &self,
        entries: &[RootManifestEntry],
    ) -> bool {
        for entry in entries {
            if let RootManifestEntry::ManifestRef {
                manifest_file,
                mdv: Some(mdv_bytes),
            } = entry
            {
                if let Ok(mdv) = ManifestDeleteVector::deserialize(mdv_bytes) {
                    let total = manifest_file
                        .added_files_count
                        .unwrap_or(0)
                        + manifest_file.existing_files_count.unwrap_or(0);
                    if total > 0 && mdv.deleted_fraction(total) >= self.mdv_compaction_threshold {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Default for RebalanceRootManifestAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for RebalanceRootManifestAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // 1. Verify V4
        if table.metadata().format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "rebalance_root_manifest requires format version V4, found {:?}",
                    table.metadata().format_version()
                ),
            ));
        }

        // 2. Load current root manifest
        let current_snapshot = match table.metadata().current_snapshot() {
            Some(s) => s,
            None => {
                // No snapshot => nothing to rebalance
                return Ok(ActionCommit::new(vec![], vec![]));
            }
        };

        let root_manifest_path = current_snapshot.manifest_list();
        let bytes = table
            .file_io()
            .new_input(root_manifest_path)?
            .read()
            .await?;

        let (rm_metadata, entries) = read_root_manifest(bytes)?;
        let root_manifest = RootManifest::new(rm_metadata.clone(), entries);

        // 3. Check if rebalance is needed
        let inline_count = root_manifest.inline_count();
        let needs_flush = inline_count >= self.inline_threshold;
        let needs_mdv_compact = self.needs_mdv_compaction(root_manifest.entries());

        if !needs_flush && !needs_mdv_compact {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // Prepare for writing
        let snapshot_id =
            SnapshotProducer::generate_unique_snapshot_id_static(table);
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let format_version = table.metadata().format_version();
        let commit_uuid = self.commit_uuid;
        let mut manifest_counter: u64 = 0;

        // Separate entries into manifest refs and inline entries
        let mut new_entries: Vec<RootManifestEntry> = Vec::new();

        // --- Phase A: Process existing manifest refs (MDV compaction) ---
        for entry in root_manifest.entries() {
            if let RootManifestEntry::ManifestRef {
                manifest_file,
                mdv,
            } = entry
            {
                let should_compact = if let Some(mdv_bytes) = mdv {
                    if let Ok(mdv_obj) = ManifestDeleteVector::deserialize(mdv_bytes) {
                        let total = manifest_file
                            .added_files_count
                            .unwrap_or(0)
                            + manifest_file.existing_files_count.unwrap_or(0);
                        total > 0
                            && mdv_obj.deleted_fraction(total)
                                >= self.mdv_compaction_threshold
                    } else {
                        false
                    }
                } else {
                    false
                };

                if should_compact {
                    // Load manifest, filter out deleted entries, write new manifest
                    let mdv_obj = ManifestDeleteVector::deserialize(
                        mdv.as_ref().unwrap(),
                    )?;
                    let manifest = manifest_file
                        .load_manifest(table.file_io())
                        .await?;

                    let spec_id = manifest_file.partition_spec_id;
                    let spec = table
                        .metadata()
                        .partition_spec_by_id(spec_id)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("partition spec {spec_id} not found"),
                            )
                        })?;

                    let manifest_path = format!(
                        "{}/{}/{}-m{}.parquet",
                        table.metadata().location(),
                        META_ROOT_PATH,
                        commit_uuid,
                        manifest_counter,
                    );
                    manifest_counter += 1;

                    let output_file =
                        table.file_io().new_output(&manifest_path)?;
                    let builder = ManifestWriterBuilder::new(
                        output_file,
                        Some(snapshot_id),
                        None,
                        schema.clone(),
                        spec.as_ref().clone(),
                    );
                    let mut writer = match format_version {
                        FormatVersion::V1 => builder.build_v1(),
                        FormatVersion::V2 => builder.build_v2_data(),
                        FormatVersion::V3 | FormatVersion::V4 => {
                            builder.build_v3_data()
                        }
                    };

                    for (idx, entry) in manifest.entries().iter().enumerate() {
                        if !mdv_obj.is_deleted(idx as u32) && entry.is_alive() {
                            let existing = ManifestEntry::builder()
                                .status(ManifestStatus::Existing)
                                .snapshot_id(entry.snapshot_id().unwrap_or(0))
                                .sequence_number(
                                    entry.sequence_number().unwrap_or(0),
                                )
                                .file_sequence_number_opt(
                                    entry.file_sequence_number,
                                )
                                .data_file(entry.data_file().clone())
                                .build();
                            writer.add_entry(existing)?;
                        }
                    }

                    let new_manifest_file =
                        writer.write_manifest_file_parquet().await?;

                    // Add the compacted manifest ref (no MDV needed now)
                    new_entries.push(RootManifestEntry::ManifestRef {
                        manifest_file: new_manifest_file,
                        mdv: None,
                    });
                } else {
                    // Keep the manifest ref as-is
                    new_entries.push(entry.clone());
                }
            }
        }

        // --- Phase B: Flush inline entries into child manifests ---
        if needs_flush {
            // Group inline entries by (content_type, partition_spec_id)
            let mut data_by_spec: HashMap<i32, Vec<&ManifestEntry>> =
                HashMap::new();
            let mut delete_by_spec: HashMap<i32, Vec<&ManifestEntry>> =
                HashMap::new();

            for entry in root_manifest.entries() {
                if let RootManifestEntry::Inline(me) = entry {
                    match me.data_file.content {
                        DataContentType::Data => {
                            data_by_spec
                                .entry(me.data_file.partition_spec_id)
                                .or_default()
                                .push(me);
                        }
                        DataContentType::EqualityDeletes
                        | DataContentType::PositionDeletes => {
                            delete_by_spec
                                .entry(me.data_file.partition_spec_id)
                                .or_default()
                                .push(me);
                        }
                    }
                }
            }

            // Flush data entries by spec
            for (spec_id, entries) in data_by_spec {
                let spec = table
                    .metadata()
                    .partition_spec_by_id(spec_id)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("partition spec {spec_id} not found"),
                        )
                    })?;

                let manifest_path = format!(
                    "{}/{}/{}-m{}.parquet",
                    table.metadata().location(),
                    META_ROOT_PATH,
                    commit_uuid,
                    manifest_counter,
                );
                manifest_counter += 1;

                let output_file =
                    table.file_io().new_output(&manifest_path)?;
                let builder = ManifestWriterBuilder::new(
                    output_file,
                    Some(snapshot_id),
                    None,
                    schema.clone(),
                    spec.as_ref().clone(),
                );
                let mut writer = match format_version {
                    FormatVersion::V1 => builder.build_v1(),
                    FormatVersion::V2 => builder.build_v2_data(),
                    FormatVersion::V3 | FormatVersion::V4 => {
                        builder.build_v3_data()
                    }
                };

                for me in &entries {
                    let manifest_entry = ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(me.snapshot_id.unwrap_or(0))
                        .sequence_number(me.sequence_number.unwrap_or(0))
                        .file_sequence_number_opt(me.file_sequence_number)
                        .data_file(me.data_file.clone())
                        .build();
                    writer.add_entry(manifest_entry)?;
                }

                let manifest_file =
                    writer.write_manifest_file_parquet().await?;
                new_entries.push(RootManifestEntry::ManifestRef {
                    manifest_file,
                    mdv: None,
                });
            }

            // Flush delete entries by spec
            for (spec_id, entries) in delete_by_spec {
                let spec = table
                    .metadata()
                    .partition_spec_by_id(spec_id)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("partition spec {spec_id} not found"),
                        )
                    })?;

                let manifest_path = format!(
                    "{}/{}/{}-m{}.parquet",
                    table.metadata().location(),
                    META_ROOT_PATH,
                    commit_uuid,
                    manifest_counter,
                );
                manifest_counter += 1;

                let output_file =
                    table.file_io().new_output(&manifest_path)?;
                let builder = ManifestWriterBuilder::new(
                    output_file,
                    Some(snapshot_id),
                    None,
                    schema.clone(),
                    spec.as_ref().clone(),
                );
                let mut writer = match format_version {
                    FormatVersion::V1 => builder.build_v1(),
                    FormatVersion::V2 => builder.build_v2_deletes(),
                    FormatVersion::V3 | FormatVersion::V4 => {
                        builder.build_v3_deletes()
                    }
                };

                for me in &entries {
                    let manifest_entry = ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(me.snapshot_id.unwrap_or(0))
                        .sequence_number(me.sequence_number.unwrap_or(0))
                        .file_sequence_number_opt(me.file_sequence_number)
                        .data_file(me.data_file.clone())
                        .build();
                    writer.add_entry(manifest_entry)?;
                }

                let manifest_file =
                    writer.write_manifest_file_parquet().await?;
                new_entries.push(RootManifestEntry::ManifestRef {
                    manifest_file,
                    mdv: None,
                });
            }
        } else {
            // No flush needed — carry inline entries forward as-is
            for entry in root_manifest.entries() {
                if let RootManifestEntry::Inline(_) = entry {
                    new_entries.push(entry.clone());
                }
            }
        }

        // --- Phase C: Write new root manifest ---
        let partition_type = table
            .metadata()
            .default_partition_spec()
            .partition_type(table.metadata().current_schema())?;

        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: table.metadata().default_partition_spec().clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
        };

        let new_root_manifest_path = format!(
            "{}/{}/root-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            snapshot_id,
            commit_uuid,
        );

        let root_bytes =
            write_root_manifest(&new_entries, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_manifest_path)?
            .write(root_bytes.into())
            .await?;

        // --- Phase D: Build snapshot and ActionCommit ---
        let inline_flushed = if needs_flush { inline_count } else { 0 };
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "rebalance-inline-flushed".to_string(),
                    inline_flushed.to_string(),
                ),
                (
                    "rebalance-entries-after".to_string(),
                    new_entries.len().to_string(),
                ),
            ]),
        };

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let first_row_id = table.metadata().next_row_id();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(new_root_manifest_path.clone())
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

        Ok(
            ActionCommit::new(updates, requirements)
                .with_manifest_paths(vec![new_root_manifest_path]),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thresholds() {
        let action = RebalanceRootManifestAction::new();
        assert_eq!(action.inline_threshold, DEFAULT_INLINE_THRESHOLD);
        assert!(
            (action.mdv_compaction_threshold - DEFAULT_MDV_COMPACTION_THRESHOLD).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_custom_thresholds() {
        let action = RebalanceRootManifestAction::new()
            .with_inline_threshold(500)
            .with_mdv_compaction_threshold(0.5);
        assert_eq!(action.inline_threshold, 500);
        assert!((action.mdv_compaction_threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_needs_mdv_compaction_no_mdv() {
        let action = RebalanceRootManifestAction::new();
        let mf = ManifestFile {
            manifest_path: "s3://bucket/m0.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: None,
        }];
        assert!(!action.needs_mdv_compaction(&entries));
    }

    #[test]
    fn test_needs_mdv_compaction_below_threshold() {
        let action = RebalanceRootManifestAction::new()
            .with_mdv_compaction_threshold(0.5);
        let mf = ManifestFile {
            manifest_path: "s3://bucket/m0.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        // Mark 2 out of 10 as deleted (20% < 50% threshold)
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        mdv.mark_deleted(1);
        let mdv_bytes = mdv.serialize().unwrap();

        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: Some(mdv_bytes),
        }];
        assert!(!action.needs_mdv_compaction(&entries));
    }

    #[test]
    fn test_needs_mdv_compaction_above_threshold() {
        let action = RebalanceRootManifestAction::new()
            .with_mdv_compaction_threshold(0.3);
        let mf = ManifestFile {
            manifest_path: "s3://bucket/m0.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        // Mark 4 out of 10 as deleted (40% >= 30% threshold)
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        mdv.mark_deleted(1);
        mdv.mark_deleted(2);
        mdv.mark_deleted(3);
        let mdv_bytes = mdv.serialize().unwrap();

        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: Some(mdv_bytes),
        }];
        assert!(action.needs_mdv_compaction(&entries));
    }
}
