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

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFile, DataFileFormat, FormatVersion, ManifestEntry, ManifestFile, ManifestStatus,
    ManifestWriterBuilder, Operation,
};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

static REWRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Action to replace data files in a table (overwrite operation).
pub struct ReplaceDataFilesAction {
    files_to_delete: Vec<DataFile>,
    files_to_add: Vec<DataFile>,
    delete_manifests: Vec<String>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    validate_from_snapshot_id: Option<i64>,
    data_sequence_number: Option<i64>,
    added_delete_files: Vec<DataFile>,
}

impl ReplaceDataFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            files_to_delete: Vec::new(),
            files_to_add: Vec::new(),
            delete_manifests: Vec::new(),
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            validate_from_snapshot_id: None,
            data_sequence_number: None,
            added_delete_files: Vec::new(),
        }
    }

    /// Set the data files to delete.
    pub fn delete_files(mut self, files: Vec<DataFile>) -> Self {
        self.files_to_delete = files;
        self
    }

    /// Set the data files to add.
    pub fn add_files(mut self, files: Vec<DataFile>) -> Self {
        self.files_to_add = files;
        self
    }

    /// Set manifest paths that should be completely dropped.
    pub fn delete_manifests(mut self, manifests: Vec<String>) -> Self {
        self.delete_manifests = manifests;
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set key metadata for manifest files.
    pub fn set_key_metadata(mut self, key_metadata: Vec<u8>) -> Self {
        self.key_metadata = Some(key_metadata);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(mut self, snapshot_properties: HashMap<String, String>) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }

    /// Set the snapshot ID to validate from.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.validate_from_snapshot_id = Some(snapshot_id);
        self
    }

    /// Set the data sequence number for delete files.
    pub fn data_sequence_number(mut self, seq_num: i64) -> Self {
        self.data_sequence_number = Some(seq_num);
        self
    }

    /// Set the delete files to add (e.g. DVs).
    pub fn add_delete_files(mut self, files: Vec<DataFile>) -> Self {
        self.added_delete_files = files;
        self
    }
}

#[async_trait]
impl TransactionAction for ReplaceDataFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.files_to_add.clone(),
            self.added_delete_files.clone(),
        )
        .with_removed_data_files(self.files_to_delete.clone())
        .with_data_sequence_number(self.data_sequence_number);

        // Validate added data files if any
        if !self.files_to_add.is_empty() {
            snapshot_producer.validate_added_data_files()?;
        }

        let operation = ReplaceOperation {
            files_to_delete: self
                .files_to_delete
                .iter()
                .map(|f| f.file_path.clone())
                .collect(),
            delete_manifests: self.delete_manifests.iter().cloned().collect(),
            commit_uuid: self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            key_metadata: self.key_metadata.clone(),
        };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

struct ReplaceOperation {
    files_to_delete: HashSet<String>,
    delete_manifests: HashSet<String>,
    commit_uuid: Uuid,
    key_metadata: Option<Vec<u8>>,
}

impl SnapshotProduceOperation for ReplaceOperation {
    fn operation(&self) -> Operation {
        Operation::Overwrite
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        let mut result_manifests: Vec<ManifestFile> = Vec::new();
        let mut remaining_to_delete: HashSet<String> = self.files_to_delete.clone();

        for manifest_entry in manifest_list.entries() {
            // Fast-path: drop manifests that are in the delete_manifests set
            if self
                .delete_manifests
                .contains(&manifest_entry.manifest_path)
            {
                continue;
            }

            // Skip manifests with no active files
            if !manifest_entry.has_added_files() && !manifest_entry.has_existing_files() {
                continue;
            }

            // Load the manifest to check if any of its entries need to be deleted
            let manifest = manifest_entry
                .load_manifest(snapshot_produce.table.file_io())
                .await?;

            // Check if this manifest contains any files we need to delete
            let entries = manifest.entries();
            let has_deletes = entries
                .iter()
                .any(|e| e.is_alive() && remaining_to_delete.contains(e.file_path()));

            if !has_deletes {
                // Keep the manifest as-is
                result_manifests.push(manifest_entry.clone());
                continue;
            }

            // Check if ALL alive entries should be deleted
            let alive_entries: Vec<_> = entries.iter().filter(|e| e.is_alive()).collect();
            let all_deleted = alive_entries
                .iter()
                .all(|e| remaining_to_delete.contains(e.file_path()));

            if all_deleted {
                // Mark files as found and drop the entire manifest
                for entry in &alive_entries {
                    remaining_to_delete.remove(entry.file_path());
                }
                continue;
            }

            // Mixed manifest: rewrite it keeping only non-deleted entries
            let counter = REWRITE_COUNTER.fetch_add(1, Ordering::SeqCst);
            let new_manifest_path = format!(
                "{}/metadata/{}-m-rewrite-{}.{}",
                snapshot_produce.table.metadata().location(),
                self.commit_uuid,
                counter,
                DataFileFormat::Avro
            );
            let output_file = snapshot_produce
                .table
                .file_io()
                .new_output(&new_manifest_path)?;

            // The rewritten manifest is part of the NEW commit's manifest list,
            // so its `added_snapshot_id` must be the new snapshot's ID — not
            // None (which would default to UNASSIGNED_SNAPSHOT_ID = -1 and fail
            // the sequence-number assignment check in ManifestListWriter).
            // Per-entry snapshot IDs are preserved separately (Existing entries
            // keep their original snapshot_id, see `entry.snapshot_id()` below).
            let builder = ManifestWriterBuilder::new(
                output_file,
                Some(snapshot_produce.snapshot_id()),
                self.key_metadata.clone(),
                snapshot_produce.table.metadata().current_schema().clone(),
                snapshot_produce
                    .table
                    .metadata()
                    .default_partition_spec()
                    .as_ref()
                    .clone(),
            );

            let mut writer = match snapshot_produce.table.metadata().format_version() {
                FormatVersion::V1 => builder.build_v1(),
                FormatVersion::V2 => builder.build_v2_data(),
                FormatVersion::V3 => builder.build_v3_data(),
            };

            for entry in entries {
                if entry.is_alive() && remaining_to_delete.contains(entry.file_path()) {
                    remaining_to_delete.remove(entry.file_path());
                    // Mark as deleted
                    let deleted = ManifestEntry::builder()
                        .status(ManifestStatus::Deleted)
                        .snapshot_id(entry.snapshot_id().unwrap_or(0))
                        .data_file(entry.data_file().clone())
                        .build();
                    writer.add_entry(deleted)?;
                } else {
                    // Keep entry as existing
                    let existing = ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(entry.snapshot_id().unwrap_or(0))
                        .data_file(entry.data_file().clone())
                        .build();
                    writer.add_entry(existing)?;
                }
            }

            let new_manifest_file = writer.write_manifest_file().await?;
            result_manifests.push(new_manifest_file);
        }

        // Validate that all files_to_delete were found
        if !remaining_to_delete.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Could not find the following files to delete: {}",
                    remaining_to_delete
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }

        Ok(result_manifests)
    }
}
