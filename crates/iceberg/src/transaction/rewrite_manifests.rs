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

//! Manifest compaction: rewrites the manifest list to merge many small manifests
//! into fewer large ones.  This is a **metadata-only operation** — no data files
//! are read or written.
//!
//! Two-phase design for concurrent-write resilience:
//!
//! **Phase 1 (slow, done once):** Read all manifests, write compacted manifests
//! to S3. The compacted manifest files are immutable once written.
//!
//! **Phase 2 (fast, retryable):** Load fresh table, diff the manifest list to
//! find new manifests added since Phase 1 started. Build final manifest list =
//! compacted manifests + new manifests. Commit.
//!
//! On commit conflict (snapshot changed), only Phase 2 is redone.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFileFormat, FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestEntry, ManifestFile,
    ManifestListWriter, ManifestStatus, ManifestWriterBuilder, Operation, Snapshot,
    SnapshotReference, SnapshotRetention, Summary,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::{Catalog, Error, ErrorKind, TableCommit, TableRequirement, TableUpdate};

/// Action that compacts (rewrites) manifest files without touching data files.
///
/// Uses a two-phase approach: Phase 1 rewrites manifests (slow, done once),
/// Phase 2 merges with any new manifests and commits (fast, retryable).
/// This makes it safe to use while concurrent writers (e.g., streaming ingest)
/// are appending data.
pub struct RewriteManifestsAction {
    /// Target number of entries per output manifest.
    target_entries_per_manifest: usize,
}

impl RewriteManifestsAction {
    /// Create a new `RewriteManifestsAction` with default settings.
    pub fn new() -> Self {
        Self {
            target_entries_per_manifest: 1000,
        }
    }

    /// Set the target number of entries per output manifest.
    pub fn target_entries_per_manifest(mut self, n: usize) -> Self {
        self.target_entries_per_manifest = n;
        self
    }

    /// Execute manifest compaction directly against the catalog.
    ///
    /// This bypasses `Transaction::commit()` to implement the two-phase
    /// approach with fast retries. Phase 1 (rewriting manifests) is done
    /// once; Phase 2 (merging + committing) retries on conflict.
    pub async fn execute(self, catalog: &dyn Catalog, table: &Table) -> Result<Table> {
        // ── Phase 1: Rewrite manifests (slow, done once) ─────────────
        let starting_snapshot = table.metadata().current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Cannot rewrite manifests: table has no current snapshot",
            )
        })?;

        let starting_manifest_list = starting_snapshot
            .load_manifest_list(table.file_io(), &table.metadata_ref())
            .await?;

        // Record which manifest paths we're compacting
        let starting_manifest_paths: HashSet<String> = starting_manifest_list
            .entries()
            .iter()
            .map(|mf| mf.manifest_path.clone())
            .collect();

        let old_manifest_count = starting_manifest_list.entries().len();

        // Separate data manifests from delete manifests
        let mut data_manifests: Vec<&ManifestFile> = Vec::new();
        let mut delete_manifests: Vec<ManifestFile> = Vec::new();

        for mf in starting_manifest_list.entries() {
            match mf.content {
                ManifestContentType::Data => data_manifests.push(mf),
                ManifestContentType::Deletes => delete_manifests.push(mf.clone()),
            }
        }

        // Load all alive data entries, grouped by partition_spec_id.
        // Entries from different specs have different partition tuple shapes
        // and must be written to separate manifests with matching specs.
        let mut entries_by_spec: HashMap<i32, Vec<ManifestEntry>> = HashMap::new();
        for mf in &data_manifests {
            let manifest = mf.load_manifest(table.file_io()).await?;
            for entry in manifest.entries() {
                if entry.is_alive() {
                    entries_by_spec
                        .entry(entry.data_file().partition_spec_id)
                        .or_default()
                        .push(entry.as_ref().clone());
                }
            }
        }

        // Write compacted manifest files to S3, one set per spec_id.
        // Generate snapshot_id upfront so manifests carry a valid ID
        // (ManifestListWriter rejects manifests with unassigned snapshot_id in V2).
        let commit_uuid = Uuid::now_v7();
        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let format_version = table.metadata().format_version();
        let schema = table.metadata().current_schema().clone();

        let mut compacted_data_manifests: Vec<ManifestFile> = Vec::new();
        let mut manifest_counter: u64 = 0;

        for (spec_id, entries) in &entries_by_spec {
            let spec = table
                .metadata()
                .partition_spec_by_id(*spec_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("partition spec {spec_id} not found in table metadata"),
                    )
                })?;

            for chunk in entries.chunks(self.target_entries_per_manifest) {
                let manifest_path = format!(
                    "{}/metadata/{}-m{}.{}",
                    table.metadata().location(),
                    commit_uuid,
                    manifest_counter,
                    DataFileFormat::Avro,
                );
                manifest_counter += 1;

                let output_file = table.file_io().new_output(&manifest_path)?;
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
                    FormatVersion::V3 => builder.build_v3_data(),
                };

                for entry in chunk {
                    let existing = ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(entry.snapshot_id().unwrap_or(0))
                        .sequence_number(entry.sequence_number().unwrap_or(0))
                        .file_sequence_number_opt(entry.file_sequence_number)
                        .data_file(entry.data_file().clone())
                        .build();
                    writer.add_entry(existing)?;
                }

                let manifest_file = writer.write_manifest_file().await?;
                compacted_data_manifests.push(manifest_file);
            }
        }

        // Phase 1 complete — compacted manifests are on S3 and immutable.
        let new_manifest_count = compacted_data_manifests.len();

        // ── Phase 2: Merge with new manifests and commit (fast, retryable) ──
        let max_retries = 5;
        let mut current_table = table.clone();

        for attempt in 0..max_retries {
            if attempt > 0 {
                // Reload table to get latest snapshot
                current_table = catalog.load_table(table.identifier()).await?;
            }

            let fresh_snapshot =
                current_table
                    .metadata()
                    .current_snapshot()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            "Table has no current snapshot during merge phase",
                        )
                    })?;

            let fresh_manifest_list = fresh_snapshot
                .load_manifest_list(
                    current_table.file_io(),
                    &current_table.metadata_ref(),
                )
                .await?;

            // Find new manifests added since Phase 1 started
            let mut new_manifests: Vec<ManifestFile> = Vec::new();
            for mf in fresh_manifest_list.entries() {
                if !starting_manifest_paths.contains(&mf.manifest_path) {
                    new_manifests.push(mf.clone());
                }
            }

            // Reuse the snapshot_id generated in Phase 1 (manifests already carry it)
            let _ = snapshot_id;
            let next_seq_num = current_table.metadata().next_sequence_number();
            let manifest_list_path = format!(
                "{}/metadata/snap-{}-0-{}.{}",
                current_table.metadata().location(),
                snapshot_id,
                commit_uuid,
                DataFileFormat::Avro,
            );

            let mut manifest_list_writer = match format_version {
                FormatVersion::V1 => ManifestListWriter::v1(
                    current_table
                        .file_io()
                        .new_output(manifest_list_path.clone())?,
                    snapshot_id,
                    current_table.metadata().current_snapshot_id(),
                ),
                FormatVersion::V2 => ManifestListWriter::v2(
                    current_table
                        .file_io()
                        .new_output(manifest_list_path.clone())?,
                    snapshot_id,
                    current_table.metadata().current_snapshot_id(),
                    next_seq_num,
                ),
                FormatVersion::V3 => ManifestListWriter::v3(
                    current_table
                        .file_io()
                        .new_output(manifest_list_path.clone())?,
                    snapshot_id,
                    current_table.metadata().current_snapshot_id(),
                    next_seq_num,
                    None,
                ),
            };

            let all_manifests = compacted_data_manifests
                .iter()
                .cloned()
                .chain(new_manifests.into_iter())
                .chain(delete_manifests.iter().cloned());
            manifest_list_writer.add_manifests(all_manifests)?;
            manifest_list_writer.close().await?;

            // Build snapshot
            let summary = Summary {
                operation: Operation::Replace,
                additional_properties: HashMap::from([
                    (
                        "rewritten-data-manifests-count".to_string(),
                        data_manifests.len().to_string(),
                    ),
                    (
                        "new-data-manifests-count".to_string(),
                        new_manifest_count.to_string(),
                    ),
                    (
                        "new-manifests-since-rewrite".to_string(),
                        fresh_manifest_list
                            .entries()
                            .len()
                            .saturating_sub(old_manifest_count)
                            .to_string(),
                    ),
                    ("attempt".to_string(), attempt.to_string()),
                ]),
            };

            let new_snapshot = Snapshot::builder()
                .with_manifest_list(manifest_list_path)
                .with_snapshot_id(snapshot_id)
                .with_parent_snapshot_id(current_table.metadata().current_snapshot_id())
                .with_sequence_number(next_seq_num)
                .with_summary(summary)
                .with_schema_id(current_table.metadata().current_schema_id())
                .with_timestamp_ms(chrono::Utc::now().timestamp_millis())
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
                    uuid: current_table.metadata().uuid(),
                },
                TableRequirement::RefSnapshotIdMatch {
                    r#ref: MAIN_BRANCH.to_string(),
                    snapshot_id: current_table.metadata().current_snapshot_id(),
                },
            ];

            let table_commit = TableCommit::builder()
                .ident(current_table.identifier().to_owned())
                .updates(updates)
                .requirements(requirements)
                .build();

            match catalog.update_table(table_commit).await {
                Ok(updated_table) => return Ok(updated_table),
                Err(e) if e.retryable() && attempt < max_retries - 1 => {
                    // Snapshot changed — retry Phase 2 only (fast)
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        Err(Error::new(
            ErrorKind::Unexpected,
            format!(
                "Manifest compaction failed after {max_retries} attempts \
                 (concurrent writes too frequent)"
            ),
        ))
    }
}

// Keep TransactionAction impl for compatibility, but it doesn't benefit
// from the two-phase approach. Use execute() directly for concurrent-write
// resilience.
#[async_trait]
impl TransactionAction for RewriteManifestsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Fallback: single-phase approach via Transaction framework.
        // For the merge-aware two-phase approach, use execute() directly.
        let current_snapshot = table.metadata().current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Cannot rewrite manifests: table has no current snapshot",
            )
        })?;

        let manifest_list = current_snapshot
            .load_manifest_list(table.file_io(), &table.metadata_ref())
            .await?;

        let old_manifest_count = manifest_list.entries().len();

        let mut data_manifests: Vec<&ManifestFile> = Vec::new();
        let mut delete_manifests: Vec<ManifestFile> = Vec::new();
        for mf in manifest_list.entries() {
            match mf.content {
                ManifestContentType::Data => data_manifests.push(mf),
                ManifestContentType::Deletes => delete_manifests.push(mf.clone()),
            }
        }

        let mut entries_by_spec: HashMap<i32, Vec<ManifestEntry>> = HashMap::new();
        for mf in &data_manifests {
            let manifest = mf.load_manifest(table.file_io()).await?;
            for entry in manifest.entries() {
                if entry.is_alive() {
                    entries_by_spec
                        .entry(entry.data_file().partition_spec_id)
                        .or_default()
                        .push(entry.as_ref().clone());
                }
            }
        }

        let commit_uuid = Uuid::now_v7();
        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let format_version = table.metadata().format_version();
        let schema = table.metadata().current_schema().clone();

        let mut new_data_manifests: Vec<ManifestFile> = Vec::new();
        let mut manifest_counter: u64 = 0;

        for (spec_id, entries) in &entries_by_spec {
            let spec = table
                .metadata()
                .partition_spec_by_id(*spec_id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("partition spec {spec_id} not found in table metadata"),
                    )
                })?;

            for chunk in entries.chunks(self.target_entries_per_manifest) {
                let manifest_path = format!(
                    "{}/metadata/{}-m{}.{}",
                    table.metadata().location(),
                    commit_uuid,
                    manifest_counter,
                    DataFileFormat::Avro,
                );
                manifest_counter += 1;

                let output_file = table.file_io().new_output(&manifest_path)?;
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
                    FormatVersion::V3 => builder.build_v3_data(),
                };

                for entry in chunk {
                    let existing = ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(entry.snapshot_id().unwrap_or(0))
                        .sequence_number(entry.sequence_number().unwrap_or(0))
                        .file_sequence_number_opt(entry.file_sequence_number)
                        .data_file(entry.data_file().clone())
                        .build();
                    writer.add_entry(existing)?;
                }

                let manifest_file = writer.write_manifest_file().await?;
                new_data_manifests.push(manifest_file);
            }
        }

        let next_seq_num = table.metadata().next_sequence_number();
        let manifest_list_path = format!(
            "{}/metadata/snap-{}-0-{}.{}",
            table.metadata().location(),
            snapshot_id,
            commit_uuid,
            DataFileFormat::Avro,
        );

        let mut manifest_list_writer = match format_version {
            FormatVersion::V1 => ManifestListWriter::v1(
                table.file_io().new_output(manifest_list_path.clone())?,
                snapshot_id,
                table.metadata().current_snapshot_id(),
            ),
            FormatVersion::V2 => ManifestListWriter::v2(
                table.file_io().new_output(manifest_list_path.clone())?,
                snapshot_id,
                table.metadata().current_snapshot_id(),
                next_seq_num,
            ),
            FormatVersion::V3 => ManifestListWriter::v3(
                table.file_io().new_output(manifest_list_path.clone())?,
                snapshot_id,
                table.metadata().current_snapshot_id(),
                next_seq_num,
                None,
            ),
        };

        let all_manifests = new_data_manifests
            .into_iter()
            .chain(delete_manifests.into_iter());
        manifest_list_writer.add_manifests(all_manifests)?;
        manifest_list_writer.close().await?;

        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "rewritten-data-manifests-count".to_string(),
                    data_manifests.len().to_string(),
                ),
                (
                    "new-data-manifests-count".to_string(),
                    manifest_counter.to_string(),
                ),
                (
                    "total-original-manifests".to_string(),
                    old_manifest_count.to_string(),
                ),
            ]),
        };

        let new_snapshot = Snapshot::builder()
            .with_manifest_list(manifest_list_path)
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(table.metadata().current_schema_id())
            .with_timestamp_ms(chrono::Utc::now().timestamp_millis())
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

        Ok(ActionCommit::new(updates, requirements))
    }
}
