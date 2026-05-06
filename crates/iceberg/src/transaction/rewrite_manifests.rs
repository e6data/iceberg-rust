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

use std::collections::HashMap;
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
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Action that compacts (rewrites) manifest files without touching data files.
///
/// Many small manifests are merged into fewer, larger manifests containing
/// `target_entries_per_manifest` alive entries each.  Delete manifests are
/// kept as-is.
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
}

#[async_trait]
impl TransactionAction for RewriteManifestsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // ── 1. Load current snapshot and its manifest list ──────────────
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

        // ── 2. Separate data manifests from delete manifests ───────────
        let mut data_manifest_files: Vec<&ManifestFile> = Vec::new();
        let mut delete_manifest_files: Vec<ManifestFile> = Vec::new();

        for mf in manifest_list.entries() {
            match mf.content {
                ManifestContentType::Data => data_manifest_files.push(mf),
                ManifestContentType::Deletes => delete_manifest_files.push(mf.clone()),
            }
        }

        // ── 3. Load all alive data entries from data manifests ─────────
        let mut alive_entries: Vec<ManifestEntry> = Vec::new();

        for mf in &data_manifest_files {
            let manifest = mf.load_manifest(table.file_io()).await?;
            for entry in manifest.entries() {
                if entry.is_alive() {
                    // Clone the entry — we will write it as Existing into the
                    // new manifest, preserving its original metadata.
                    alive_entries.push(entry.as_ref().clone());
                }
            }
        }

        // ── 4. Write new, larger manifest files ────────────────────────
        let commit_uuid = Uuid::now_v7();
        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let format_version = table.metadata().format_version();
        let schema = table.metadata().current_schema().clone();
        let partition_spec = table.metadata().default_partition_spec().as_ref().clone();

        let mut new_data_manifests: Vec<ManifestFile> = Vec::new();
        let mut created_manifest_paths: Vec<String> = Vec::new();
        let mut manifest_counter: u64 = 0;

        for chunk in alive_entries.chunks(self.target_entries_per_manifest) {
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
                None, // key_metadata
                schema.clone(),
                partition_spec.clone(),
            );

            let mut writer = match format_version {
                FormatVersion::V1 => builder.build_v1(),
                FormatVersion::V2 => builder.build_v2_data(),
                FormatVersion::V3 => builder.build_v3_data(),
            };

            for entry in chunk {
                // Write every alive entry as Existing, preserving its
                // original snapshot_id, sequence_number and
                // file_sequence_number.
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
            created_manifest_paths.push(manifest_file.manifest_path.clone());
            new_data_manifests.push(manifest_file);
        }

        // ── 5. Build the manifest list ─────────────────────────────────
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
                None, // first_row_id — not assigning new rows
            ),
        };

        // Add rewritten data manifests, then keep delete manifests as-is.
        let all_manifests = new_data_manifests
            .into_iter()
            .chain(delete_manifest_files.into_iter());
        manifest_list_writer.add_manifests(all_manifests)?;
        manifest_list_writer.close().await?;

        // ── 6. Build the snapshot ──────────────────────────────────────
        let new_manifest_count = manifest_counter as usize;
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "rewritten-data-manifests-count".to_string(),
                    data_manifest_files.len().to_string(),
                ),
                (
                    "new-data-manifests-count".to_string(),
                    new_manifest_count.to_string(),
                ),
                (
                    "total-original-manifests".to_string(),
                    old_manifest_count.to_string(),
                ),
            ]),
        };

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(manifest_list_path)
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(table.metadata().current_schema_id())
            .with_timestamp_ms(commit_ts)
            .build();

        // ── 7. Return the ActionCommit ─────────────────────────────────
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

        Ok(ActionCommit::new(updates, requirements).with_manifest_paths(created_manifest_paths))
    }
}
