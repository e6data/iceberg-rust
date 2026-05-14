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

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{DataFile, ManifestEntry, ManifestFile, Operation};
use crate::table::Table;
use crate::transaction::snapshot::{
    DefaultManifestProcess, SnapshotProduceOperation, SnapshotProducer,
};
use crate::transaction::{ActionCommit, TransactionAction};

/// RowDeltaAction commits row-level changes to a table.
///
/// It can add data files and delete files in one snapshot. This is used for
/// merge-on-read style changes such as updates and deletes.
pub struct RowDeltaAction {
    check_duplicate: bool,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    added_delete_files: Vec<DataFile>,
}

impl RowDeltaAction {
    pub(crate) fn new() -> Self {
        Self {
            check_duplicate: true,
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            added_data_files: vec![],
            added_delete_files: vec![],
        }
    }

    /// Set whether to check duplicate data files.
    pub fn with_check_duplicate(mut self, v: bool) -> Self {
        self.check_duplicate = v;
        self
    }

    /// Add data files to the table.
    pub fn add_rows(mut self, data_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_data_files.extend(data_files);
        self
    }

    /// Add equality or position delete files to the table.
    pub fn add_deletes(mut self, delete_files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added_delete_files.extend(delete_files);
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
}

#[async_trait]
impl TransactionAction for RowDeltaAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.added_data_files.clone(),
            self.added_delete_files.clone(),
        );

        if !self.added_data_files.is_empty() {
            snapshot_producer.validate_added_data_files()?;
        }

        if !self.added_delete_files.is_empty() {
            snapshot_producer.validate_added_delete_files()?;
        }

        if self.check_duplicate && !self.added_data_files.is_empty() {
            snapshot_producer.validate_duplicate_files().await?;
        }

        snapshot_producer
            .commit(RowDeltaOperation::new(&self), DefaultManifestProcess)
            .await
    }
}

struct RowDeltaOperation {
    has_data_files: bool,
    has_delete_files: bool,
}

impl RowDeltaOperation {
    fn new(action: &RowDeltaAction) -> Self {
        Self {
            has_data_files: !action.added_data_files.is_empty(),
            has_delete_files: !action.added_delete_files.is_empty(),
        }
    }
}

impl SnapshotProduceOperation for RowDeltaOperation {
    fn operation(&self) -> Operation {
        match (self.has_data_files, self.has_delete_files) {
            (true, false) => Operation::Append,
            (false, true) => Operation::Delete,
            (true, true) => Operation::Overwrite,
            (false, false) => Operation::Append,
        }
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

        Ok(manifest_list
            .entries()
            .iter()
            .filter(|entry| entry.has_added_files() || entry.has_existing_files())
            .cloned()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::TableUpdate;
    use crate::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, Literal, ManifestContentType, Operation,
        Struct,
    };
    use crate::transaction::tests::make_v2_minimal_table;
    use crate::transaction::{Transaction, TransactionAction};

    #[tokio::test]
    async fn test_row_delta_with_only_data_files_is_append() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/data.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap();

        let mut action_commit = Arc::new(tx.row_delta().add_rows(vec![data_file]))
            .commit(&table)
            .await
            .unwrap();

        let updates = action_commit.take_updates();
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot update");
        };

        assert_eq!(new_snapshot.summary().operation, Operation::Append);
    }

    #[tokio::test]
    async fn test_row_delta_with_only_delete_files_is_delete() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/delete.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap();

        let mut action_commit = Arc::new(tx.row_delta().add_deletes(vec![delete_file]))
            .commit(&table)
            .await
            .unwrap();

        let updates = action_commit.take_updates();
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot update");
        };

        assert_eq!(new_snapshot.summary().operation, Operation::Delete);
    }

    #[tokio::test]
    async fn test_row_delta_with_data_and_delete_files_writes_both_manifest_types() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let data_file = DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path("test/data.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(10)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap();

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/delete.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap();

        let mut action_commit = Arc::new(
            tx.row_delta()
                .add_rows(vec![data_file])
                .add_deletes(vec![delete_file]),
        )
        .commit(&table)
        .await
        .unwrap();

        let updates = action_commit.take_updates();
        let new_snapshot = if let TableUpdate::AddSnapshot { snapshot } = &updates[0] {
            snapshot
        } else {
            panic!("expected AddSnapshot update");
        };

        assert_eq!(new_snapshot.summary().operation, Operation::Overwrite);

        let manifest_list = new_snapshot
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();

        assert_eq!(manifest_list.entries().len(), 2);
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|entry| entry.content == ManifestContentType::Data)
        );
        assert!(
            manifest_list
                .entries()
                .iter()
                .any(|entry| entry.content == ManifestContentType::Deletes)
        );
    }

    #[tokio::test]
    async fn test_row_delta_rejects_equality_delete_without_equality_ids() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::EqualityDeletes)
            .file_path("test/delete.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .build()
            .unwrap();

        let result = Arc::new(tx.row_delta().add_deletes(vec![delete_file]))
            .commit(&table)
            .await;

        match result {
            Ok(_) => panic!("expected equality delete validation to fail"),
            Err(err) => assert!(
                err.to_string()
                    .contains("Equality delete file must have equality_ids")
            ),
        }
    }

    #[tokio::test]
    async fn test_row_delta_rejects_position_delete_with_equality_ids() {
        let table = make_v2_minimal_table();
        let tx = Transaction::new(&table);

        let delete_file = DataFileBuilder::default()
            .content(DataContentType::PositionDeletes)
            .file_path("test/delete.parquet".to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(50)
            .record_count(3)
            .partition_spec_id(table.metadata().default_partition_spec_id())
            .partition(Struct::from_iter([Some(Literal::long(100))]))
            .equality_ids(Some(vec![1]))
            .build()
            .unwrap();

        let result = Arc::new(tx.row_delta().add_deletes(vec![delete_file]))
            .commit(&table)
            .await;

        match result {
            Ok(_) => panic!("expected position delete validation to fail"),
            Err(err) => assert!(
                err.to_string()
                    .contains("Position delete file should not have equality_ids")
            ),
        }
    }
}
