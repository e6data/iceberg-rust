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
use crate::spec::{DataFile, ManifestContentType, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind};

/// `ReplaceDataFilesAction` is a transaction action for atomically replacing data files
/// in a table. This is the core primitive for compaction: old data files are removed and
/// new (compacted) data files are added in a single atomic snapshot.
///
/// Phase 1 constraint: Manifests that contain a mix of deleted and surviving entries
/// are rejected. Each manifest must contain ONLY entries that are all deleted or all surviving.
pub struct ReplaceDataFilesAction {
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    files_to_delete: Vec<DataFile>,
    files_to_add: Vec<DataFile>,
    validate_from_snapshot_id: Option<i64>,
    data_sequence_number: Option<i64>,
    /// Known manifest paths that contain ONLY files being deleted.
    /// When provided, `existing_manifest()` drops these manifests by path
    /// without loading them from S3 — eliminating the manifest-read bottleneck.
    /// Falls back to the slow path (loading all manifests) if empty.
    delete_manifest_paths: HashSet<String>,
}

impl ReplaceDataFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            files_to_delete: vec![],
            files_to_add: vec![],
            validate_from_snapshot_id: None,
            data_sequence_number: None,
            delete_manifest_paths: HashSet::new(),
        }
    }

    /// Provide known manifest paths that contain ONLY files being deleted.
    ///
    /// When the writer (e.g. Laminar's IcebergSink) tracks which manifests
    /// its files landed in, it can pass those paths here to skip the
    /// expensive manifest-loading step in `existing_manifest()`.
    ///
    /// Each manifest in this set will be dropped from the manifest list
    /// without reading its contents from S3.
    pub fn delete_manifests(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.delete_manifest_paths.extend(paths);
        self
    }

    /// Set the data files to delete (the old files being replaced).
    pub fn delete_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.files_to_delete.extend(files);
        self
    }

    /// Set the data files to add (the new compacted files).
    pub fn add_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.files_to_add.extend(files);
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

    /// Set the snapshot ID to validate from. If set, the action will verify that the
    /// files to delete exist in this snapshot.
    pub fn validate_from_snapshot(mut self, snapshot_id: i64) -> Self {
        self.validate_from_snapshot_id = Some(snapshot_id);
        self
    }

    /// Set the data sequence number for the new files being added.
    pub fn data_sequence_number(mut self, seq_num: i64) -> Self {
        self.data_sequence_number = Some(seq_num);
        self
    }
}

#[async_trait]
impl TransactionAction for ReplaceDataFilesAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if self.files_to_delete.is_empty() && self.files_to_add.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Replace data files action requires at least one file to add or delete",
            ));
        }

        let mut snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.files_to_add.clone(),
            vec![],
        )
        .with_removed_data_files(self.files_to_delete.clone());

        if let Some(seq_num) = self.data_sequence_number {
            snapshot_producer = snapshot_producer.with_data_sequence_number(seq_num);
        }

        // Validate added files
        if !self.files_to_add.is_empty() {
            snapshot_producer.validate_added_data_files(&self.files_to_add)?;
        }

        let replace_op = ReplaceOperation {
            files_to_delete: self.files_to_delete.clone(),
            delete_manifest_paths: self.delete_manifest_paths.clone(),
        };

        snapshot_producer
            .commit(replace_op, DefaultManifestProcess)
            .await
    }
}

/// The operation implementation for replace-data-files.
/// Loads existing manifests, marks entries matching `files_to_delete` as DELETED,
/// and filters out manifests that contain only deleted entries.
struct ReplaceOperation {
    files_to_delete: Vec<DataFile>,
    /// Fast path: known manifest paths to drop without loading from S3.
    delete_manifest_paths: HashSet<String>,
}

impl SnapshotProduceOperation for ReplaceOperation {
    fn operation(&self) -> Operation {
        Operation::Overwrite
    }

    async fn delete_entries(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestEntry>> {
        // Not used in this implementation; deletion is handled in existing_manifest.
        Ok(vec![])
    }

    async fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> Result<Vec<ManifestFile>> {
        let Some(snapshot) = snapshot_produce.table.metadata().current_snapshot() else {
            if !self.files_to_delete.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot delete files from a table with no snapshots",
                ));
            }
            return Ok(vec![]);
        };

        let manifest_list = snapshot
            .load_manifest_list(
                snapshot_produce.table.file_io(),
                &snapshot_produce.table.metadata_ref(),
            )
            .await?;

        let delete_set: HashSet<&str> =
            self.files_to_delete.iter().map(|f| f.file_path()).collect();

        if delete_set.is_empty() {
            // No files to delete, just return all existing manifests
            return Ok(manifest_list
                .entries()
                .iter()
                .filter(|entry| entry.has_added_files() || entry.has_existing_files())
                .cloned()
                .collect());
        }

        let mut result_manifests: Vec<ManifestFile> = Vec::new();
        let mut found_files: HashSet<String> = HashSet::new();

        // Fast path: if the caller provided known manifest paths to drop,
        // skip loading manifests from S3 entirely. Just filter by path.
        if !self.delete_manifest_paths.is_empty() {
            for manifest_entry in manifest_list.entries() {
                if self
                    .delete_manifest_paths
                    .contains(&manifest_entry.manifest_path)
                {
                    // Drop this manifest — caller guarantees it contains only deleted files
                    continue;
                }
                // Keep all other manifests (data + delete manifests)
                if manifest_entry.has_added_files() || manifest_entry.has_existing_files() {
                    result_manifests.push(manifest_entry.clone());
                }
            }
            return Ok(result_manifests);
        }

        // Slow path: load each manifest to determine which ones to drop
        for manifest_entry in manifest_list.entries() {
            // Only process data manifests for deletion
            if manifest_entry.content != ManifestContentType::Data {
                // Keep delete manifests as-is
                if manifest_entry.has_added_files() || manifest_entry.has_existing_files() {
                    result_manifests.push(manifest_entry.clone());
                }
                continue;
            }

            let manifest = manifest_entry
                .load_manifest(snapshot_produce.table.file_io())
                .await?;

            let entries = manifest.entries();
            let alive_entries: Vec<_> = entries.iter().filter(|e| e.is_alive()).collect();

            if alive_entries.is_empty() {
                // No alive entries, skip this manifest
                continue;
            }

            // Count how many alive entries match files_to_delete
            let deleted_count = alive_entries
                .iter()
                .filter(|e| delete_set.contains(e.file_path()))
                .count();

            if deleted_count == 0 {
                // No entries in this manifest are being deleted, keep it as-is
                result_manifests.push(manifest_entry.clone());
            } else if deleted_count == alive_entries.len() {
                // ALL alive entries are being deleted, drop this entire manifest
                for entry in &alive_entries {
                    found_files.insert(entry.file_path().to_string());
                }
            } else {
                // Phase 1 constraint: mixed manifests (some deleted, some surviving) are rejected
                return Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    format!(
                        "Manifest {} contains both deleted and surviving entries. \
                         Phase 1 requires manifests to be fully replaced or fully kept. \
                         Found {} entries to delete out of {} alive entries.",
                        manifest_entry.manifest_path,
                        deleted_count,
                        alive_entries.len()
                    ),
                ));
            }
        }

        // Validate that all files_to_delete were found
        let missing_files: Vec<&str> = delete_set
            .iter()
            .filter(|f| !found_files.contains(**f))
            .copied()
            .collect();

        if !missing_files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot replace files that are not found in table: {}",
                    missing_files.join(", ")
                ),
            ));
        }

        Ok(result_manifests)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Literal, Struct};
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::transaction::{Transaction, TransactionAction};

    #[tokio::test]
    async fn test_empty_replace_data_files_action() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);
        let action = tx
            .replace_data_files()
            .delete_files(vec![])
            .add_files(vec![]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_replace_data_files_add_only() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/new_compacted.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(200)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(300))]))
            .build()
            .unwrap();

        let action = tx.replace_data_files().add_files(vec![data_file]);
        let result = Arc::new(action).commit(&table).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_replace_data_files_builder_chain() {
        let uuid = uuid::Uuid::new_v4();
        let metadata = vec![1, 2, 3];
        let mut props = HashMap::new();
        props.insert("compaction".to_string(), "true".to_string());

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/data.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(10)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .unwrap();

        let action = Transaction::new(&make_v2_minimal_table())
            .replace_data_files()
            .set_commit_uuid(uuid)
            .set_key_metadata(metadata)
            .set_snapshot_properties(props)
            .validate_from_snapshot(42)
            .data_sequence_number(5)
            .add_files(vec![data_file]);

        assert!(action.commit_uuid.is_some());
        assert!(action.key_metadata.is_some());
        assert!(!action.snapshot_properties.is_empty());
        assert_eq!(action.validate_from_snapshot_id, Some(42));
        assert_eq!(action.data_sequence_number, Some(5));
    }
}
