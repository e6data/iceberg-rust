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

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFile, FormatVersion, ManifestEntry, ManifestFile,
    ManifestStatus, ManifestWriterBuilder, Operation,
};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// Action for atomically replacing data files in a table (compaction primitive).
pub struct ReplaceDataFilesAction {
    commit_uuid: Option<Uuid>,
    snapshot_properties: HashMap<String, String>,
    files_to_delete: Vec<DataFile>,
    files_to_add: Vec<DataFile>,
    delete_manifest_paths: Vec<String>,
}

impl ReplaceDataFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            snapshot_properties: HashMap::default(),
            files_to_delete: vec![],
            files_to_add: vec![],
            delete_manifest_paths: vec![],
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

    /// Set manifest paths to drop entirely (fast-path for manifest-aware compaction).
    pub fn delete_manifests(mut self, paths: Vec<String>) -> Self {
        self.delete_manifest_paths = paths;
        self
    }

    /// Set commit UUID for the snapshot.
    pub fn set_commit_uuid(mut self, commit_uuid: Uuid) -> Self {
        self.commit_uuid = Some(commit_uuid);
        self
    }

    /// Set snapshot summary properties.
    pub fn set_snapshot_properties(
        mut self,
        snapshot_properties: HashMap<String, String>,
    ) -> Self {
        self.snapshot_properties = snapshot_properties;
        self
    }
}

#[async_trait]
impl TransactionAction for ReplaceDataFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            None,
            self.snapshot_properties.clone(),
            self.files_to_add.clone(),
            vec![],
        )
        .with_removed_data_files(self.files_to_delete.clone());

        snapshot_producer
            .commit(
                ReplaceOperation {
                    files_to_delete: self.files_to_delete.clone(),
                    delete_manifest_paths: self.delete_manifest_paths.clone(),
                },
                DefaultManifestProcess,
            )
            .await
    }
}

struct ReplaceOperation {
    files_to_delete: Vec<DataFile>,
    delete_manifest_paths: Vec<String>,
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

        let files_to_delete: HashSet<&str> = self
            .files_to_delete
            .iter()
            .map(|f| f.file_path.as_str())
            .collect();

        let delete_manifest_set: HashSet<&str> = self
            .delete_manifest_paths
            .iter()
            .map(|p| p.as_str())
            .collect();

        let mut result_manifests: Vec<ManifestFile> = Vec::new();
        let mut found_files: HashSet<String> = HashSet::new();

        for manifest_entry in manifest_list.entries() {
            // Fast path: drop entire manifests by path
            if !delete_manifest_set.is_empty()
                && delete_manifest_set.contains(manifest_entry.manifest_path.as_str())
            {
                // Load manifest to track which files were found
                let manifest = manifest_entry
                    .load_manifest(snapshot_produce.table.file_io())
                    .await?;
                for entry in manifest.entries() {
                    if entry.is_alive() && files_to_delete.contains(entry.file_path()) {
                        found_files.insert(entry.file_path().to_string());
                    }
                }
                // Drop this manifest entirely - don't add to result
                continue;
            }

            // Load manifest to check if it contains any files to delete
            let manifest = manifest_entry
                .load_manifest(snapshot_produce.table.file_io())
                .await?;

            let mut has_deletes = false;
            let mut all_deleted = true;

            for entry in manifest.entries() {
                if entry.is_alive() {
                    if files_to_delete.contains(entry.file_path()) {
                        has_deletes = true;
                        found_files.insert(entry.file_path().to_string());
                    } else {
                        all_deleted = false;
                    }
                } else {
                    // Already deleted entries don't count as "alive"
                    all_deleted = false;
                }
            }

            if !has_deletes {
                // No files to delete in this manifest, keep as-is
                result_manifests.push(manifest_entry.clone());
            } else if all_deleted {
                // All alive entries are being deleted, drop the manifest
                continue;
            } else {
                // Mixed manifest: rewrite with only surviving entries
                let new_manifest_path = format!(
                    "{}/metadata/{}-m-rewrite.avro",
                    snapshot_produce.table.metadata().location(),
                    Uuid::now_v7(),
                );
                let output_file = snapshot_produce
                    .table
                    .file_io()
                    .new_output(new_manifest_path)?;

                let builder = ManifestWriterBuilder::new(
                    output_file,
                    Some(
                        snapshot_produce
                            .table
                            .metadata()
                            .current_snapshot_id()
                            .unwrap_or(0),
                    ),
                    None,
                    manifest.metadata().schema().clone(),
                    manifest.metadata().partition_spec().clone(),
                );

                let mut writer = match snapshot_produce.table.metadata().format_version() {
                    FormatVersion::V1 => builder.build_v1(),
                    FormatVersion::V2 => builder.build_v2_data(),
                    FormatVersion::V3 => builder.build_v3_data(),
                };

                for entry in manifest.entries() {
                    if entry.is_alive() && !files_to_delete.contains(entry.file_path()) {
                        // Keep this entry as Existing
                        let existing_entry = ManifestEntry::builder()
                            .status(ManifestStatus::Existing)
                            .snapshot_id(entry.snapshot_id().unwrap_or(0))
                            .sequence_number(entry.sequence_number().unwrap_or(0))
                            .file_sequence_number(entry.file_sequence_number.unwrap_or(0))
                            .data_file(entry.data_file().clone())
                            .build();
                        writer.add_entry(existing_entry)?;
                    }
                }

                let rewritten_manifest = writer.write_manifest_file().await?;
                result_manifests.push(rewritten_manifest);
            }
        }

        // Validate all files to delete were found
        if found_files.len() != files_to_delete.len() {
            let missing: Vec<String> = files_to_delete
                .iter()
                .filter(|f| !found_files.contains(**f))
                .map(|f| f.to_string())
                .collect();
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot replace data files: the following files were not found in the table: {}",
                    missing.join(", ")
                ),
            ));
        }

        Ok(result_manifests)
    }
}
