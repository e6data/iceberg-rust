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

//! This module contains transaction api.
//!
//! The transaction API enables changes to be made to an existing table.
//!
//! Note that this may also have side effects, such as producing new manifest
//! files.
//!
//! Below is a basic example using the "fast-append" action:
//!
//! ```ignore
//! use iceberg::transaction::{ApplyTransactionAction, Transaction};
//! use iceberg::Catalog;
//!
//! // Create a transaction.
//! let tx = Transaction::new(my_table);
//!
//! // Create a `FastAppendAction` which will not rewrite or append
//! // to existing metadata. This will create a new manifest.
//! let action = tx.fast_append().add_data_files(my_data_files);
//!
//! // Apply the fast-append action to the given transaction, returning
//! // the newly updated `Transaction`.
//! let tx = action.apply(tx).unwrap();
//!
//!
//! // End the transaction by committing to an `iceberg::Catalog`
//! // implementation. This will cause a table update to occur.
//! let table = tx
//!     .commit(&some_catalog_impl)
//!     .await
//!     .unwrap();
//! ```

/// The `ApplyTransactionAction` trait provides an `apply` method
/// that allows users to apply a transaction action to a `Transaction`.
mod action;

pub use action::*;
mod append;
mod cold_paths;
mod compact_cold_tier;
mod drop_cold_buckets;
mod graduate_buckets;
mod replace_data_files;
mod rebalance_root_manifest;
mod rewrite_manifests;
mod snapshot;
pub use compact_cold_tier::CompactColdTierAction;
pub use drop_cold_buckets::DropColdBucketsAction;
pub use graduate_buckets::GraduateBucketsAction;
pub use rebalance_root_manifest::RebalanceRootManifestAction;
pub use snapshot::generate_unique_snapshot_id;
mod sort_order;
mod update_location;
mod update_properties;
mod update_schema;
mod update_spec;
mod update_statistics;
mod upgrade_format_version;

use std::sync::Arc;
use std::time::Duration;

use backon::{BackoffBuilder, ExponentialBackoff, ExponentialBuilder, RetryableWithContext};

use crate::error::Result;
use crate::spec::TableProperties;
use crate::table::Table;
use crate::transaction::action::BoxedTransactionAction;
use crate::transaction::append::FastAppendAction;
use crate::transaction::replace_data_files::ReplaceDataFilesAction;
use crate::transaction::rewrite_manifests::RewriteManifestsAction;
use crate::transaction::sort_order::ReplaceSortOrderAction;
use crate::transaction::update_location::UpdateLocationAction;
use crate::transaction::update_properties::UpdatePropertiesAction;
use crate::transaction::update_schema::UpdateSchemaAction;
use crate::transaction::update_spec::UpdateSpecAction;
use crate::transaction::update_statistics::UpdateStatisticsAction;
use crate::transaction::upgrade_format_version::UpgradeFormatVersionAction;
use crate::{Catalog, Error, ErrorKind, TableCommit, TableRequirement, TableUpdate};

/// Table transaction.
#[derive(Clone)]
pub struct Transaction {
    table: Table,
    actions: Vec<BoxedTransactionAction>,
    first_attempt: bool,
    created_manifest_paths: Vec<String>,
    /// When true, OCC commit failures are not retried. Set automatically by
    /// `apply` for any action whose `TransactionAction::disables_retry()` returns
    /// true — currently ReplaceDataFiles, because retrying with a stale delete-file
    /// list produces duplicate or resurrected data.
    disable_retry: bool,
}

impl Transaction {
    /// Creates a new transaction.
    pub fn new(table: &Table) -> Self {
        Self {
            table: table.clone(),
            actions: vec![],
            first_attempt: true,
            created_manifest_paths: Vec::new(),
            disable_retry: false,
        }
    }

    fn update_table_metadata(table: Table, updates: &[TableUpdate]) -> Result<Table> {
        let mut metadata_builder = table.metadata().clone().into_builder(None);
        for update in updates {
            metadata_builder = update.clone().apply(metadata_builder)?;
        }

        Ok(table.with_metadata(Arc::new(metadata_builder.build()?.metadata)))
    }

    /// Applies an [`ActionCommit`] to the given [`Table`], returning a new [`Table`] with updated metadata.
    /// Also appends any derived [`TableUpdate`]s and [`TableRequirement`]s to the provided vectors.
    fn apply(
        table: Table,
        mut action_commit: ActionCommit,
        existing_updates: &mut Vec<TableUpdate>,
        existing_requirements: &mut Vec<TableRequirement>,
    ) -> Result<Table> {
        let updates = action_commit.take_updates();
        let requirements = action_commit.take_requirements();

        for requirement in &requirements {
            requirement.check(Some(table.metadata()))?;
        }

        let updated_table = Self::update_table_metadata(table, &updates)?;

        existing_updates.extend(updates);
        existing_requirements.extend(requirements);

        Ok(updated_table)
    }

    /// Sets table to a new version.
    pub fn upgrade_table_version(&self) -> UpgradeFormatVersionAction {
        UpgradeFormatVersionAction::new()
    }

    /// Update table's property.
    pub fn update_table_properties(&self) -> UpdatePropertiesAction {
        UpdatePropertiesAction::new()
    }

    /// Creates a fast append action.
    pub fn fast_append(&self) -> FastAppendAction {
        FastAppendAction::new()
    }

    /// Creates replace sort order action.
    pub fn replace_sort_order(&self) -> ReplaceSortOrderAction {
        ReplaceSortOrderAction::new()
    }

    /// Set the location of table
    pub fn update_location(&self) -> UpdateLocationAction {
        UpdateLocationAction::new()
    }

    /// Update the statistics of table
    pub fn update_statistics(&self) -> UpdateStatisticsAction {
        UpdateStatisticsAction::new()
    }

    /// Creates a replace data files action.
    pub fn replace_data_files(&self) -> ReplaceDataFilesAction {
        ReplaceDataFilesAction::new()
    }

    /// Creates a rewrite-manifests action that compacts many small manifests
    /// into fewer large ones. This is a metadata-only operation.
    pub fn rewrite_manifests(&self) -> RewriteManifestsAction {
        RewriteManifestsAction::new()
    }

    /// Creates a rebalance action for V4 root manifests. Flushes inline
    /// entries into child manifest files and compacts MDV-heavy manifest refs.
    pub fn rebalance_root_manifest(&self) -> RebalanceRootManifestAction {
        RebalanceRootManifestAction::new()
    }

    /// Creates an action that graduates closed live nodes (and any closed inline
    /// files) out of the hot root manifest into the cold bucket-index. A node
    /// graduates when its newest event time (max of `ts_field_id`) is below
    /// `cutoff_micros`. Partition-spec-agnostic; the caller computes
    /// `cutoff_micros = now − bucket_window`.
    pub fn graduate_buckets(
        &self,
        ts_field_id: i32,
        cutoff_micros: i64,
    ) -> GraduateBucketsAction {
        GraduateBucketsAction::new(ts_field_id, cutoff_micros)
    }

    /// Creates an action that compacts the cold tier: within the bucket-index's
    /// leaf manifests, replaces the given `removed` data files (already merged
    /// into the `added` files by the caller, e.g. tessellate's streaming concat)
    /// and re-clusters the affected leaves. Off the hot commit path.
    pub fn compact_cold_tier(&self) -> CompactColdTierAction {
        CompactColdTierAction::new()
    }

    /// Creates a time-based retention action: drops cold leaves whose data is
    /// entirely older than `cutoff_micros` (max value for `ts_field_id` < cutoff)
    /// from the bucket-index and repoints the root. Partition-spec-agnostic —
    /// keys on the leaf's data timestamp stats, not on partitioning. Metadata-
    /// only, off the hot path; dropped leaves/files are reclaimed by orphan GC.
    pub fn drop_cold_buckets(&self, ts_field_id: i32, cutoff_micros: i64) -> DropColdBucketsAction {
        DropColdBucketsAction::new(ts_field_id, cutoff_micros)
    }

    /// Creates a schema-evolution action limited to additive changes
    /// (`add_column`). For renames/drops/promotions use a future fuller
    /// UpdateSchema implementation; this exists so a streaming sink can
    /// auto-extend the table when an inbound batch carries a new field.
    pub fn update_schema(&self) -> UpdateSchemaAction {
        UpdateSchemaAction::new()
    }

    /// Creates a partition-spec evolution action limited to additive
    /// changes (`add_field`). For removals/renames/field-id changes use a
    /// future fuller UpdatePartitionSpec implementation; this exists so a
    /// streaming sink can start partitioning by an additional column
    /// without rewriting existing data (Iceberg supports two specs
    /// coexisting and existing files keep their original spec id).
    pub fn update_spec(&self) -> UpdateSpecAction {
        UpdateSpecAction::new()
    }

    /// Commit transaction.
    pub async fn commit(self, catalog: &dyn Catalog) -> Result<Table> {
        self.commit_with_manifest_paths(catalog)
            .await
            .map(|(table, _)| table)
    }

    /// Commit transaction and return both the updated table and the manifest paths created.
    pub async fn commit_with_manifest_paths(
        self,
        catalog: &dyn Catalog,
    ) -> Result<(Table, Vec<String>)> {
        if self.actions.is_empty() {
            // nothing to commit
            return Ok((self.table, Vec::new()));
        }

        let table_props =
            TableProperties::try_from(self.table.metadata().properties()).map_err(|e| {
                Error::new(ErrorKind::DataInvalid, "Invalid table properties").with_source(e)
            })?;

        let backoff = Self::build_backoff(table_props)?;
        let tx = self;

        let disable_retry = tx.disable_retry;
        let (tx, result) = (|mut tx: Transaction| async {
            let result = tx.do_commit(catalog).await;
            (tx, result)
        })
        .retry(backoff)
        .sleep(tokio::time::sleep)
        .context(tx)
        .when(|e| e.retryable() && !disable_retry)
        .await;

        let table = result?;
        Ok((table, tx.created_manifest_paths))
    }

    fn build_backoff(props: TableProperties) -> Result<ExponentialBackoff> {
        Ok(ExponentialBuilder::new()
            .with_min_delay(Duration::from_millis(props.commit_min_retry_wait_ms))
            .with_max_delay(Duration::from_millis(props.commit_max_retry_wait_ms))
            .with_total_delay(Some(Duration::from_millis(
                props.commit_total_retry_timeout_ms,
            )))
            .with_max_times(props.commit_num_retries)
            .with_factor(2.0)
            .build())
    }

    async fn do_commit(&mut self, catalog: &dyn Catalog) -> Result<Table> {
        let attempt_started = std::time::Instant::now();
        let is_retry = !self.first_attempt;
        if self.first_attempt {
            self.first_attempt = false;
        } else {
            let refreshed = catalog.load_table(self.table.identifier()).await?;

            if self.table.metadata() != refreshed.metadata()
                || self.table.metadata_location() != refreshed.metadata_location()
            {
                // current base is stale, use refreshed as base and re-apply transaction actions
                self.table = refreshed.clone();
            }
        }
        let refresh_ms = attempt_started.elapsed().as_millis() as u64;

        let mut current_table = self.table.clone();
        let mut existing_updates: Vec<TableUpdate> = vec![];
        let mut existing_requirements: Vec<TableRequirement> = vec![];
        let mut all_manifest_paths: Vec<String> = Vec::new();

        let actions_started = std::time::Instant::now();
        let mut action_ms: Vec<u64> = Vec::with_capacity(self.actions.len());
        let mut action_names: Vec<&'static str> = Vec::with_capacity(self.actions.len());
        for action in &self.actions {
            let action_started = std::time::Instant::now();
            let action_name = action.action_name();
            let mut action_commit = Arc::clone(action).commit(&current_table).await?;
            let this_action_ms = action_started.elapsed().as_millis() as u64;
            action_ms.push(this_action_ms);
            action_names.push(action_name);
            // Name the expensive one directly. `per_action_ms` is positional,
            // so a 36s entry could not previously be attributed to an action
            // without inference — and inference sent instrumentation to the
            // wrong place three times. 2s is well above the ~100ms p50 and
            // low enough to catch everything that matters.
            if this_action_ms > 2_000 {
                log::info!(
                    "slow action: name={} ms={} table={} action_idx={} of {}",
                    action_name,
                    this_action_ms,
                    self.table.identifier(),
                    action_ms.len() - 1,
                    self.actions.len()
                );
            }
            all_manifest_paths.extend(action_commit.take_manifest_paths());
            // apply action commit to current_table
            current_table = Self::apply(
                current_table,
                action_commit,
                &mut existing_updates,
                &mut existing_requirements,
            )?;
        }
        let actions_elapsed_ms = actions_started.elapsed().as_millis() as u64;

        self.created_manifest_paths = all_manifest_paths;

        let table_commit = TableCommit::builder()
            .ident(self.table.identifier().to_owned())
            .updates(existing_updates)
            .requirements(existing_requirements)
            .build();

        let update_started = std::time::Instant::now();
        let result = catalog.update_table(table_commit).await;
        log::info!(
            "commit sub-steps: table={} retry={} refresh_ms={} actions_ms={} per_action_ms={:?} action_names={:?} update_table_ms={} total_ms={} ok={}",
            self.table.identifier(),
            is_retry,
            refresh_ms,
            actions_elapsed_ms,
            action_ms,
            action_names,
            update_started.elapsed().as_millis() as u64,
            attempt_started.elapsed().as_millis() as u64,
            result.is_ok(),
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::BufReader;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    use crate::catalog::MockCatalog;
    use crate::io::FileIOBuilder;
    use crate::spec::TableMetadata;
    use crate::table::Table;
    use crate::transaction::{ApplyTransactionAction, Transaction};
    use crate::{Catalog, Error, ErrorKind, TableCreation, TableIdent};

    pub fn make_v1_table() -> Table {
        let file = File::open(format!(
            "{}/testdata/table_metadata/{}",
            env!("CARGO_MANIFEST_DIR"),
            "TableMetadataV1Valid.json"
        ))
        .unwrap();
        let reader = BufReader::new(file);
        let resp = serde_json::from_reader::<_, TableMetadata>(reader).unwrap();

        Table::builder()
            .metadata(resp)
            .metadata_location("s3://bucket/test/location/metadata/v1.json".to_string())
            .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
            .file_io(FileIOBuilder::new("memory").build().unwrap())
            .build()
            .unwrap()
    }

    pub fn make_v2_table() -> Table {
        let file = File::open(format!(
            "{}/testdata/table_metadata/{}",
            env!("CARGO_MANIFEST_DIR"),
            "TableMetadataV2Valid.json"
        ))
        .unwrap();
        let reader = BufReader::new(file);
        let resp = serde_json::from_reader::<_, TableMetadata>(reader).unwrap();

        Table::builder()
            .metadata(resp)
            .metadata_location("s3://bucket/test/location/metadata/v1.json".to_string())
            .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
            .file_io(FileIOBuilder::new("memory").build().unwrap())
            .build()
            .unwrap()
    }

    pub fn make_v2_minimal_table() -> Table {
        let file = File::open(format!(
            "{}/testdata/table_metadata/{}",
            env!("CARGO_MANIFEST_DIR"),
            "TableMetadataV2ValidMinimal.json"
        ))
        .unwrap();
        let reader = BufReader::new(file);
        let resp = serde_json::from_reader::<_, TableMetadata>(reader).unwrap();

        Table::builder()
            .metadata(resp)
            .metadata_location("s3://bucket/test/location/metadata/v1.json".to_string())
            .identifier(TableIdent::from_strs(["ns1", "test1"]).unwrap())
            .file_io(FileIOBuilder::new("memory").build().unwrap())
            .build()
            .unwrap()
    }

    pub(crate) async fn make_v3_minimal_table_in_catalog(catalog: &impl Catalog) -> Table {
        let table_ident =
            TableIdent::from_strs([format!("ns1-{}", uuid::Uuid::new_v4()), "test1".to_string()])
                .unwrap();

        catalog
            .create_namespace(table_ident.namespace(), HashMap::new())
            .await
            .unwrap();

        let file = File::open(format!(
            "{}/testdata/table_metadata/{}",
            env!("CARGO_MANIFEST_DIR"),
            "TableMetadataV3ValidMinimal.json"
        ))
        .unwrap();
        let reader = BufReader::new(file);
        let base_metadata = serde_json::from_reader::<_, TableMetadata>(reader).unwrap();

        let table_creation = TableCreation::builder()
            .schema((**base_metadata.current_schema()).clone())
            .partition_spec((**base_metadata.default_partition_spec()).clone())
            .sort_order((**base_metadata.default_sort_order()).clone())
            .name(table_ident.name().to_string())
            .format_version(crate::spec::FormatVersion::V3)
            .build();

        catalog
            .create_table(table_ident.namespace(), table_creation)
            .await
            .unwrap()
    }

    /// Helper function to create a test table with retry properties
    pub(super) fn setup_test_table(num_retries: &str) -> Table {
        let table = make_v2_table();

        // Set retry properties
        let mut props = HashMap::new();
        props.insert("commit.retry.min-wait-ms".to_string(), "10".to_string());
        props.insert("commit.retry.max-wait-ms".to_string(), "100".to_string());
        props.insert(
            "commit.retry.total-timeout-ms".to_string(),
            "1000".to_string(),
        );
        props.insert(
            "commit.retry.num-retries".to_string(),
            num_retries.to_string(),
        );

        // Update table properties
        let metadata = table
            .metadata()
            .clone()
            .into_builder(None)
            .set_properties(props)
            .unwrap()
            .build()
            .unwrap()
            .metadata;

        table.with_metadata(Arc::new(metadata))
    }

    /// Helper function to create a transaction with a simple update action
    fn create_test_transaction(table: &Table) -> Transaction {
        let tx = Transaction::new(table);
        tx.update_table_properties()
            .set("test.key".to_string(), "test.value".to_string())
            .apply(tx)
            .unwrap()
    }

    /// Helper function to set up a mock catalog with retryable errors
    fn setup_mock_catalog_with_retryable_errors(
        success_after_attempts: Option<u32>,
        expected_calls: usize,
    ) -> MockCatalog {
        let mut mock_catalog = MockCatalog::new();

        mock_catalog
            .expect_load_table()
            .returning_st(|_| Box::pin(async move { Ok(make_v2_table()) }));

        let attempts = AtomicU32::new(0);
        mock_catalog
            .expect_update_table()
            .times(expected_calls)
            .returning_st(move |_| {
                if let Some(success_after_attempts) = success_after_attempts {
                    attempts.fetch_add(1, Ordering::SeqCst);
                    if attempts.load(Ordering::SeqCst) <= success_after_attempts {
                        Box::pin(async move {
                            Err(
                                Error::new(ErrorKind::CatalogCommitConflicts, "Commit conflict")
                                    .with_retryable(true),
                            )
                        })
                    } else {
                        Box::pin(async move { Ok(make_v2_table()) })
                    }
                } else {
                    // Always fail with retryable error
                    Box::pin(async move {
                        Err(
                            Error::new(ErrorKind::CatalogCommitConflicts, "Commit conflict")
                                .with_retryable(true),
                        )
                    })
                }
            });

        mock_catalog
    }

    /// Helper function to set up a mock catalog with non-retryable error
    fn setup_mock_catalog_with_non_retryable_error() -> MockCatalog {
        let mut mock_catalog = MockCatalog::new();

        mock_catalog
            .expect_load_table()
            .returning_st(|_| Box::pin(async move { Ok(make_v2_table()) }));

        mock_catalog
            .expect_update_table()
            .times(1) // Should only be called once since error is not retryable
            .returning_st(move |_| {
                Box::pin(async move {
                    Err(Error::new(ErrorKind::Unexpected, "Non-retryable error")
                        .with_retryable(false))
                })
            });

        mock_catalog
    }

    #[tokio::test]
    async fn test_commit_retryable_error() {
        // Create a test table with retry properties
        let table = setup_test_table("3");

        // Create a transaction with a simple update action
        let tx = create_test_transaction(&table);

        // Create a mock catalog that fails twice then succeeds
        let mock_catalog = setup_mock_catalog_with_retryable_errors(Some(2), 3);

        // Commit the transaction
        let result = tx.commit(&mock_catalog).await;

        // Verify the result
        assert!(result.is_ok(), "Transaction should eventually succeed");
    }

    #[tokio::test]
    async fn test_commit_non_retryable_error() {
        // Create a test table with retry properties
        let table = setup_test_table("3");

        // Create a transaction with a simple update action
        let tx = create_test_transaction(&table);

        // Create a mock catalog that fails with non-retryable error
        let mock_catalog = setup_mock_catalog_with_non_retryable_error();

        // Commit the transaction
        let result = tx.commit(&mock_catalog).await;

        // Verify the result
        assert!(result.is_err(), "Transaction should fail immediately");
        if let Err(err) = result {
            assert_eq!(err.kind(), ErrorKind::Unexpected);
            assert_eq!(err.message(), "Non-retryable error");
            assert!(!err.retryable(), "Error should not be retryable");
        }
    }

    #[tokio::test]
    async fn test_commit_max_retries_exceeded() {
        // Create a test table with retry properties (only allow 2 retries)
        let table = setup_test_table("2");

        // Create a transaction with a simple update action
        let tx = create_test_transaction(&table);

        // Create a mock catalog that always fails with retryable error
        let mock_catalog = setup_mock_catalog_with_retryable_errors(None, 3); // Initial attempt + 2 retries = 3 total attempts

        // Commit the transaction
        let result = tx.commit(&mock_catalog).await;

        // Verify the result
        assert!(result.is_err(), "Transaction should fail after max retries");
        if let Err(err) = result {
            assert_eq!(err.kind(), ErrorKind::CatalogCommitConflicts);
            assert_eq!(err.message(), "Commit conflict");
            assert!(err.retryable(), "Error should be retryable");
        }
    }
}

#[cfg(test)]
mod test_row_lineage {
    use crate::memory::tests::new_memory_catalog;
    use crate::spec::{
        DataContentType, DataFile, DataFileBuilder, DataFileFormat, Literal, Struct,
    };
    use crate::error::Result;
    use crate::table::Table;
    use crate::transaction::tests::make_v3_minimal_table_in_catalog;
    use crate::transaction::{ApplyTransactionAction, Transaction};

    #[tokio::test]
    async fn test_fast_append_with_row_lineage() {
        // Helper function to create a data file with specified number of rows
        fn file_with_rows(record_count: u64) -> DataFile {
            DataFileBuilder::default()
                .content(DataContentType::Data)
                .file_path(format!("test/{record_count}.parquet"))
                .file_format(DataFileFormat::Parquet)
                .file_size_in_bytes(100)
                .record_count(record_count)
                .partition(Struct::from_iter([Some(Literal::long(0))]))
                .partition_spec_id(0)
                .build()
                .unwrap()
        }
        let catalog = new_memory_catalog().await;

        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Check initial state - next_row_id should be 0
        assert_eq!(table.metadata().next_row_id(), 0);

        // First fast append with 30 rows
        let tx = Transaction::new(&table);
        let data_file_30 = file_with_rows(30);
        let action = tx.fast_append().add_data_files(vec![data_file_30]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Check snapshot and table state after first append
        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.first_row_id(), Some(0));
        assert_eq!(table.metadata().next_row_id(), 30);

        // Check written manifest for first_row_id
        let manifest_list = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();

        assert_eq!(manifest_list.entries().len(), 1);
        let manifest_file = &manifest_list.entries()[0];
        assert_eq!(manifest_file.first_row_id, Some(0));

        // Second fast append with 17 and 11 rows
        let tx = Transaction::new(&table);
        let data_file_17 = file_with_rows(17);
        let data_file_11 = file_with_rows(11);
        let action = tx
            .fast_append()
            .add_data_files(vec![data_file_17, data_file_11]);
        let tx = action.apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Check snapshot and table state after second append
        let snapshot = table.metadata().current_snapshot().unwrap();
        assert_eq!(snapshot.first_row_id(), Some(30));
        assert_eq!(table.metadata().next_row_id(), 30 + 17 + 11);

        // Check written manifest for first_row_id
        let manifest_list = table
            .metadata()
            .current_snapshot()
            .unwrap()
            .load_manifest_list(table.file_io(), table.metadata())
            .await
            .unwrap();
        assert_eq!(manifest_list.entries().len(), 2);
        let manifest_file = &manifest_list.entries()[1];
        assert_eq!(manifest_file.first_row_id, Some(30));
    }

    // ========================================================================
    // M4 (DST) — invariants against the REAL commit path.
    //
    // The e6-observability-bench/dst model tiers prove the commit/GC/OCC
    // *algorithms*. This runs a storage invariant against the actual iceberg-rust
    // transaction + manifest-rewrite + read-back code, driven fully in-process
    // (MemoryCatalog + memory FileIO) so it's deterministic and CI-cheap. First
    // invariant: `replace_data_files` (the merge-on-write compaction path, whose
    // stale-delete-list retry is the M2 incident) must, after commit, leave the
    // live data-file set as (old − deleted + added) — no deleted file lingering,
    // no duplicate path.
    // ========================================================================

    fn dst_data_file(path: &str, rows: u64) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(rows)
            .partition(Struct::from_iter([Some(Literal::long(0))]))
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    /// The LIVE data files of the current snapshot, via the REAL scan planner —
    /// which computes true liveness (accounting for cross-manifest deletes), i.e.
    /// exactly what the executor sees. Returns the tasks so callers can check both
    /// the distinct path set and the raw count (a duplicate would show as count >
    /// distinct).
    async fn dst_live_tasks(table: &crate::table::Table) -> Vec<String> {
        use futures::TryStreamExt;
        let scan = table.scan().select_all().build().unwrap();
        let tasks: Vec<_> = scan
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        tasks.into_iter().map(|t| t.data_file_path).collect()
    }

    async fn dst_live_paths(table: &crate::table::Table) -> std::collections::BTreeSet<String> {
        dst_live_tasks(table).await.into_iter().collect()
    }
    async fn dst_live_count(table: &crate::table::Table) -> usize {
        dst_live_tasks(table).await.len()
    }

    /// Live data-file OBJECTS from the current manifests (as stored). Real
    /// compaction passes THESE to `delete_files` (it reads them back from a
    /// scan/manifest), not hand-built ones — the delete matches the manifest's own
    /// `DataFile` representation.
    async fn dst_live_data_files(table: &crate::table::Table) -> Vec<DataFile> {
        let mut out = Vec::new();
        if let Some(snap) = table.metadata().current_snapshot() {
            let mlist = snap
                .load_manifest_list(table.file_io(), table.metadata())
                .await
                .unwrap();
            for mf in mlist.entries() {
                let manifest = mf.load_manifest(table.file_io()).await.unwrap();
                for entry in manifest.entries() {
                    if entry.is_alive() {
                        out.push(entry.data_file().clone());
                    }
                }
            }
        }
        out
    }

    /// First M4 real-code invariant: `fast_append` is additive and the executor's
    /// scan planner reads back EXACTLY the committed live set — no loss, no
    /// duplication — driven through the REAL commit + manifest + scan code
    /// (MemoryCatalog + in-memory FileIO, fully deterministic).
    #[tokio::test]
    async fn dst_fast_append_additive_via_real_scan() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let a = dst_data_file("test/A.parquet", 10);
        let b = dst_data_file("test/B.parquet", 20);
        let c = dst_data_file("test/C.parquet", 30);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a, b, c])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let expect3: std::collections::BTreeSet<String> =
            ["test/A.parquet", "test/B.parquet", "test/C.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(dst_live_paths(&table).await, expect3);
        assert_eq!(dst_live_count(&table).await, 3, "no duplicate live files");

        let d = dst_data_file("test/D.parquet", 5);
        let e = dst_data_file("test/E.parquet", 7);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![d, e])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let expect5: std::collections::BTreeSet<String> = [
            "test/A.parquet",
            "test/B.parquet",
            "test/C.parquet",
            "test/D.parquet",
            "test/E.parquet",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(dst_live_paths(&table).await, expect5);
        assert_eq!(dst_live_count(&table).await, 5, "append additive: no dup, no loss");
    }

    /// M4 real-code — OCC conflict + RETRY, the "no lost committed data" invariant
    /// against the actual optimistic-concurrency + retry loop. Two fast_appends read
    /// the SAME base version; one commits (advancing the catalog), the other's commit
    /// then conflicts (`CatalogCommitConflicts`) and must RETRY — reload the won
    /// version and re-apply — landing BOTH appends. If the retry re-applied against
    /// its stale base instead, the first append would be lost; this asserts it isn't.
    #[tokio::test]
    async fn dst_occ_conflict_fast_append_no_loss() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        // Both transactions are built from the SAME freshly-created base table.
        let x = dst_data_file("test/X.parquet", 1);
        let y = dst_data_file("test/Y.parquet", 1);
        let tx1 = Transaction::new(&table);
        let tx1 = tx1.fast_append().add_data_files(vec![x]).apply(tx1).unwrap();
        let tx2 = Transaction::new(&table);
        let tx2 = tx2.fast_append().add_data_files(vec![y]).apply(tx2).unwrap();

        // tx1 wins; tx2 is now stale → conflict → real retry loop → must still land.
        let _v1 = tx1.commit(&catalog).await.unwrap();
        let v2 = tx2.commit(&catalog).await.unwrap();

        let live = dst_live_paths(&v2).await;
        assert!(
            live.contains("test/X.parquet"),
            "OCC retry LOST the winning commit's append X (retry re-applied a stale base): {live:?}"
        );
        assert!(
            live.contains("test/Y.parquet"),
            "OCC retry dropped its own append Y: {live:?}"
        );
    }

    /// M4 real-code — expiring an ancestor snapshot must not break the CURRENT
    /// snapshot's reachability (a corollary of "no lost committed data": data a
    /// live snapshot references stays readable through snapshot expiry). Drives the
    /// real `remove_snapshots` metadata builder + the real scan planner.
    #[tokio::test]
    async fn dst_expire_old_snapshot_keeps_current_reachable() {
        let catalog = new_memory_catalog().await;
        let table = make_v3_minimal_table_in_catalog(&catalog).await;

        let a = dst_data_file("test/A.parquet", 10);
        let tx = Transaction::new(&table);
        let tx = tx.fast_append().add_data_files(vec![a]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let old_snap = table.metadata().current_snapshot().unwrap().snapshot_id();

        let b = dst_data_file("test/B.parquet", 20);
        let tx = Transaction::new(&table);
        let tx = tx.fast_append().add_data_files(vec![b]).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let before = dst_live_paths(&table).await;
        assert!(
            before.contains("test/A.parquet") && before.contains("test/B.parquet"),
            "precondition: current sees both appends"
        );

        // Expire the OLD (ancestor) snapshot via the real metadata builder.
        let expired_meta = table
            .metadata()
            .clone()
            .into_builder(None)
            .remove_snapshots(&[old_snap])
            .build()
            .unwrap()
            .metadata;
        let expired = table.with_metadata(std::sync::Arc::new(expired_meta));

        let after = dst_live_paths(&expired).await;
        assert!(
            after.contains("test/A.parquet") && after.contains("test/B.parquet"),
            "expiring an ancestor snapshot broke current-snapshot reachability: {after:?}"
        );
    }

    /// M4 real-code — a SEEDED DST (not a fixed scenario) over random real append
    /// schedules. Each step appends 1..=3 data files through the real commit path;
    /// after every commit the real scan planner must return EXACTLY the cumulative
    /// appended set — no loss, no duplication — across the growing manifest list
    /// (where a manifest-merge bug would surface). Finally, expiring ALL ancestor
    /// snapshots must preserve reachability. Deterministic + replayable by seed.
    #[tokio::test]
    async fn dst_seeded_real_append_reachability() {
        fn mix(s: &mut u64) -> u64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for seed in 0..12u64 {
            let mut rng = seed ^ 0xDEAD_BEEF;
            let catalog = new_memory_catalog().await;
            let mut table = make_v3_minimal_table_in_catalog(&catalog).await;
            let mut expected: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

            for step in 0..12u64 {
                let k = 1 + mix(&mut rng) % 3;
                let mut files = Vec::new();
                for i in 0..k {
                    let path = format!("dst/{seed}/{step}/{i}.parquet");
                    expected.insert(path.clone());
                    files.push(dst_data_file(&path, 1 + mix(&mut rng) % 100));
                }
                let tx = Transaction::new(&table);
                let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
                table = tx.commit(&catalog).await.unwrap();

                assert_eq!(
                    dst_live_paths(&table).await,
                    expected,
                    "seed={seed} step={step}: scan != cumulative appends (loss or spurious rows)"
                );
                assert_eq!(
                    dst_live_count(&table).await,
                    expected.len(),
                    "seed={seed} step={step}: duplicate live files after append"
                );
            }

            // Expire ALL ancestor snapshots — the current snapshot must stay fully
            // reachable (no committed data lost to expiry).
            let current = table.metadata().current_snapshot_id().unwrap();
            let ancestors: Vec<i64> = table
                .metadata()
                .snapshots()
                .map(|s| s.snapshot_id())
                .filter(|id| *id != current)
                .collect();
            if !ancestors.is_empty() {
                let meta = table
                    .metadata()
                    .clone()
                    .into_builder(None)
                    .remove_snapshots(&ancestors)
                    .build()
                    .unwrap()
                    .metadata;
                let expired = table.with_metadata(std::sync::Arc::new(meta));
                assert_eq!(
                    dst_live_paths(&expired).await,
                    expected,
                    "seed={seed}: expiring all ancestors broke reachability of committed data"
                );
            }
        }
    }

    /// M4 real-code — SEEDED DST over MIXED append+replace schedules on a V4 table
    /// (the production format where `replace_data_files` operates on the root
    /// manifest). An oracle tracks the exact live set of hand-built DataFiles; each
    /// step either appends 1..=3 new files or compacts 1..=2 live files into one
    /// merged file via the REAL `replace_data_files` path. After EVERY commit the real
    /// scan planner must return exactly the oracle's live set — no loss, no duplicate,
    /// no resurrected deleted file — across a growing/shrinking manifest. This is the
    /// seeded generalization of `dst_v4_replace_data_files_removes_deleted_files`.
    /// Deterministic + replayable by seed.
    #[tokio::test]
    async fn dst_v4_seeded_append_replace_reachability() {
        fn mix(s: &mut u64) -> u64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for seed in 0..10u64 {
            let mut rng = seed ^ 0x5EED_1CE5;
            let catalog = new_memory_catalog().await;
            let mut table = dst_make_v4_table_in_catalog(&catalog).await;
            // Oracle: path -> the exact DataFile we appended (needed to delete later,
            // since manifest-walk can't read V4 root manifests).
            let mut live: std::collections::BTreeMap<String, DataFile> =
                std::collections::BTreeMap::new();
            let mut ctr = 0u64;

            for step in 0..16u64 {
                let do_replace = live.len() >= 2 && mix(&mut rng) % 100 < 45;
                if do_replace {
                    // Compact 1..=2 live files into one merged file.
                    let keys: Vec<String> = live.keys().cloned().collect();
                    let ndel = 1 + (mix(&mut rng) % 2) as usize;
                    let mut del_keys: Vec<String> = Vec::new();
                    for _ in 0..ndel {
                        let idx = (mix(&mut rng) as usize) % keys.len();
                        let k = keys[idx].clone();
                        if !del_keys.contains(&k) {
                            del_keys.push(k);
                        }
                    }
                    let del: Vec<DataFile> = del_keys.iter().map(|k| live[k].clone()).collect();
                    ctr += 1;
                    let merged_path = format!("dst/{seed}/m{ctr}.parquet");
                    let rows = 1 + mix(&mut rng) % 100;
                    let merged = dst_data_file(&merged_path, rows);
                    let snap = table.metadata().current_snapshot().unwrap().snapshot_id();
                    let tx = Transaction::new(&table)
                        .replace_data_files()
                        .delete_files(del)
                        .add_files(vec![merged.clone()])
                        .validate_from_snapshot(snap)
                        .apply(Transaction::new(&table))
                        .unwrap();
                    table = tx.commit(&catalog).await.unwrap();
                    for k in &del_keys {
                        live.remove(k);
                    }
                    live.insert(merged_path, merged);
                } else {
                    // Append 1..=3 new files.
                    let k = 1 + mix(&mut rng) % 3;
                    let mut files = Vec::new();
                    for _ in 0..k {
                        ctr += 1;
                        let path = format!("dst/{seed}/a{ctr}.parquet");
                        let rows = 1 + mix(&mut rng) % 100;
                        let df = dst_data_file(&path, rows);
                        files.push(df.clone());
                        live.insert(path, df);
                    }
                    let tx = Transaction::new(&table);
                    let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
                    table = tx.commit(&catalog).await.unwrap();
                }

                let expected: std::collections::BTreeSet<String> = live.keys().cloned().collect();
                assert_eq!(
                    dst_live_paths(&table).await,
                    expected,
                    "seed={seed} step={step}: scan != oracle live set (append/replace loss or dup)"
                );
                assert_eq!(
                    dst_live_count(&table).await,
                    expected.len(),
                    "seed={seed} step={step}: duplicate live data-file entries"
                );
            }
        }
    }

    /// Create a V4 table in the catalog (reusing the v3-minimal schema/spec). V4
    /// `fast_append` writes into the ROOT manifest — which is exactly what
    /// `replace_data_files` operates on, so deletes take effect (unlike a v3 table).
    async fn dst_make_v4_table_in_catalog(catalog: &impl crate::Catalog) -> crate::table::Table {
        use std::collections::HashMap;
        let ident = crate::TableIdent::from_strs([
            format!("dstv4-{}", uuid::Uuid::new_v4()),
            "t".to_string(),
        ])
        .unwrap();
        catalog
            .create_namespace(ident.namespace(), HashMap::new())
            .await
            .unwrap();
        let file = std::fs::File::open(format!(
            "{}/testdata/table_metadata/TableMetadataV3ValidMinimal.json",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        let base: crate::spec::TableMetadata =
            serde_json::from_reader(std::io::BufReader::new(file)).unwrap();
        // Fast retry backoff so transient-fault tests don't spend real wall-time
        // sleeping between the commit retry loop's attempts.
        let props = HashMap::from([
            ("commit.retry.min-wait-ms".to_string(), "1".to_string()),
            ("commit.retry.max-wait-ms".to_string(), "2".to_string()),
        ]);
        let creation = crate::TableCreation::builder()
            .schema((**base.current_schema()).clone())
            .partition_spec((**base.default_partition_spec()).clone())
            .sort_order((**base.default_sort_order()).clone())
            .name(ident.name().to_string())
            .properties(props)
            .format_version(crate::spec::FormatVersion::V4)
            .build();
        catalog
            .create_table(ident.namespace(), creation)
            .await
            .unwrap()
    }

    /// M4 real-code — the replace/stale-delete invariant against a V4 table (the
    /// production format). `replace_data_files` removing A,B and adding D must leave
    /// the live set as {C, D} via the real scan planner — no deleted file lingering,
    /// no duplicate. This is the direct real-code counterpart of the `occ.rs` model
    /// and the `disable_retry` mitigation.
    #[tokio::test]
    async fn dst_v4_replace_data_files_removes_deleted_files() {
        let catalog = new_memory_catalog().await;
        let table = dst_make_v4_table_in_catalog(&catalog).await;
        assert_eq!(table.metadata().format_version(), crate::spec::FormatVersion::V4);

        let a = dst_data_file("test/A.parquet", 10);
        let b = dst_data_file("test/B.parquet", 20);
        let c = dst_data_file("test/C.parquet", 30);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a.clone(), b.clone(), c])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let expect3: std::collections::BTreeSet<String> =
            ["test/A.parquet", "test/B.parquet", "test/C.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(dst_live_paths(&table).await, expect3);

        // Delete A,B by their (path-bearing) DataFile — replace_data_files matches
        // deletes by path, and on V4 it operates on the root manifest where the
        // appends landed.
        let del: Vec<DataFile> = vec![a, b];
        let snap_id = table.metadata().current_snapshot().unwrap().snapshot_id();
        let d = dst_data_file("test/D.parquet", 30);
        let tx = Transaction::new(&table);
        let tx = tx
            .replace_data_files()
            .delete_files(del)
            .add_files(vec![d])
            .validate_from_snapshot(snap_id)
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let expect_after: std::collections::BTreeSet<String> =
            ["test/C.parquet", "test/D.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(
            dst_live_paths(&table).await,
            expect_after,
            "V4 replace_data_files must remove deleted files, add the new one, keep the rest"
        );
        assert_eq!(
            dst_live_count(&table).await,
            2,
            "no duplicate live files after replace"
        );
    }

    /// M4 real-code — the MARQUEE M2 incident test. Two `replace_data_files`
    /// (compaction) transactions plan against the SAME base snapshot, each deleting a
    /// pair that OVERLAPS on B and adding a merged file. One commits first; the second
    /// is now stale. This is the direct real-code counterpart of `occ.rs`: the danger
    /// is that the loser's retry reuses its stale delete-list (RetryMode::Naive →
    /// re-deletes an already-gone file / resurrects a merged one → duplicate data).
    ///
    /// MITIGATION: `ReplaceDataFilesAction::disables_retry()` returns `true`, so
    /// `apply` sets `Transaction::disable_retry` and the retry guard
    /// (`e.retryable() && !disable_retry`) refuses to retry the loser — it fails fast
    /// and the caller must re-plan against the fresh table. This test pins the safety
    /// property either way: after both transactions resolve, the authoritative
    /// committed state must have NO duplicate and NO resurrected (winner-deleted)
    /// file. The `Err` arm (fail-fast, current behavior) leaves the winner's state
    /// untouched; the `Ok` arm (were retry re-enabled) would require the retry to
    /// re-derive against the fresh base — either way, no corruption.
    #[tokio::test]
    async fn dst_v4_conflicting_replace_no_dup_no_resurrect() {
        let catalog = new_memory_catalog().await;
        let table = dst_make_v4_table_in_catalog(&catalog).await;
        let ident = table.identifier().clone();

        // Seed five files: A,B,C,D,E.
        let a = dst_data_file("test/A.parquet", 10);
        let b = dst_data_file("test/B.parquet", 20);
        let c = dst_data_file("test/C.parquet", 30);
        let d = dst_data_file("test/D.parquet", 40);
        let e = dst_data_file("test/E.parquet", 50);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a.clone(), b.clone(), c.clone(), d.clone(), e])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let base_snap = table.metadata().current_snapshot().unwrap().snapshot_id();
        assert_eq!(dst_live_count(&table).await, 5);

        // Two compactions from the SAME base, deletes OVERLAPPING on B.
        let f = dst_data_file("test/F.parquet", 30); // tx1 merges A+B
        let h = dst_data_file("test/H.parquet", 50); // tx2 merges B+C
        let tx1 = Transaction::new(&table)
            .replace_data_files()
            .delete_files(vec![a, b.clone()])
            .add_files(vec![f])
            .validate_from_snapshot(base_snap)
            .apply(Transaction::new(&table))
            .unwrap();
        let tx2 = Transaction::new(&table)
            .replace_data_files()
            .delete_files(vec![b, c])
            .add_files(vec![h])
            .validate_from_snapshot(base_snap)
            .apply(Transaction::new(&table))
            .unwrap();

        // tx1 wins → {C,D,E,F}.
        let _table = tx1.commit(&catalog).await.unwrap();
        // tx2 is stale (B already gone). Whether it fails or retries, capture the
        // authoritative committed state from the catalog.
        let res = tx2.commit(&catalog).await;
        let final_tbl = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
        let live = dst_live_paths(&final_tbl).await;

        // SAFETY INVARIANTS — must hold whether tx2 failed (were disable_retry wired)
        // or retried and re-derived against the fresh base:
        // 1. No duplicate / resurrected file: live entry count == distinct paths.
        assert_eq!(
            dst_live_count(&final_tbl).await,
            live.len(),
            "duplicate live data-file entries after conflicting replace: {live:?}"
        );
        // 2. The winner's deletes (A, B) must never reappear.
        assert!(
            !live.contains("test/A.parquet") && !live.contains("test/B.parquet"),
            "winner-deleted files resurrected by the stale replace: {live:?}"
        );
        // 3. The winner's merge output F is present.
        assert!(
            live.contains("test/F.parquet"),
            "winner's merge output F must be live: {live:?}"
        );
        // Document which path the real code took (the knob is unwired → it retries).
        match res {
            // Retried + re-derived: B was already gone, so tx2 legitimately deleted
            // only C and added H against the fresh base → {D,E,F,H}.
            Ok(_) => assert!(
                live.contains("test/H.parquet") && !live.contains("test/C.parquet"),
                "retry re-derived unsoundly (stale plan): {live:?}"
            ),
            // Fail-fast (disable_retry wired): winner's state stands.
            Err(_) => assert!(
                live.contains("test/C.parquet") && !live.contains("test/H.parquet"),
                "failed replace must leave the winner's state untouched: {live:?}"
            ),
        }
    }

    // ---------------------------------------------------------------------------
    // M3 real-code FAULT INJECTION.
    //
    // A `Catalog` decorator that injects register (`update_table`) failures. In the
    // real commit path, an action first WRITES its data/manifest files through FileIO
    // and only then REGISTERS the new snapshot via the catalog. Failing the register
    // models the "crash between write and register" incident: the files land in the
    // store as orphans, but the snapshot must NOT become reachable, and the table must
    // stay exactly at its prior state (commit atomicity). Injected errors are
    // non-retryable, so they surface as a failed commit rather than being retried away.
    // ---------------------------------------------------------------------------

    #[derive(Debug)]
    struct FaultyCatalog<C: crate::Catalog> {
        inner: C,
        fail_updates: std::sync::atomic::AtomicUsize,
        fail_updates_retryable: std::sync::atomic::AtomicUsize,
        update_attempts: std::sync::atomic::AtomicUsize,
    }

    impl<C: crate::Catalog> FaultyCatalog<C> {
        fn new(inner: C) -> Self {
            Self {
                inner,
                fail_updates: std::sync::atomic::AtomicUsize::new(0),
                fail_updates_retryable: std::sync::atomic::AtomicUsize::new(0),
                update_attempts: std::sync::atomic::AtomicUsize::new(0),
            }
        }
        /// Fail the next `n` `update_table` (register) calls with a NON-retryable
        /// error, then behave normally (crash between write and register).
        fn fail_next_updates(&self, n: usize) {
            self.fail_updates
                .store(n, std::sync::atomic::Ordering::SeqCst);
        }
        /// Fail the next `n` `update_table` calls with a RETRYABLE error, then behave
        /// normally (transient object-store/catalog flakiness the retry loop absorbs).
        fn fail_next_updates_retryable(&self, n: usize) {
            self.fail_updates_retryable
                .store(n, std::sync::atomic::Ordering::SeqCst);
        }
        /// Total `update_table` calls observed (each commit attempt is one call).
        fn attempts(&self) -> usize {
            self.update_attempts
                .load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl<C: crate::Catalog> crate::Catalog for FaultyCatalog<C> {
        async fn list_namespaces(
            &self,
            parent: Option<&crate::NamespaceIdent>,
        ) -> Result<Vec<crate::NamespaceIdent>> {
            self.inner.list_namespaces(parent).await
        }
        async fn create_namespace(
            &self,
            ns: &crate::NamespaceIdent,
            props: std::collections::HashMap<String, String>,
        ) -> Result<crate::Namespace> {
            self.inner.create_namespace(ns, props).await
        }
        async fn get_namespace(&self, ns: &crate::NamespaceIdent) -> Result<crate::Namespace> {
            self.inner.get_namespace(ns).await
        }
        async fn namespace_exists(&self, ns: &crate::NamespaceIdent) -> Result<bool> {
            self.inner.namespace_exists(ns).await
        }
        async fn update_namespace(
            &self,
            ns: &crate::NamespaceIdent,
            props: std::collections::HashMap<String, String>,
        ) -> Result<()> {
            self.inner.update_namespace(ns, props).await
        }
        async fn drop_namespace(&self, ns: &crate::NamespaceIdent) -> Result<()> {
            self.inner.drop_namespace(ns).await
        }
        async fn list_tables(
            &self,
            ns: &crate::NamespaceIdent,
        ) -> Result<Vec<crate::TableIdent>> {
            self.inner.list_tables(ns).await
        }
        async fn create_table(
            &self,
            ns: &crate::NamespaceIdent,
            creation: crate::TableCreation,
        ) -> Result<Table> {
            self.inner.create_table(ns, creation).await
        }
        async fn load_table(&self, t: &crate::TableIdent) -> Result<Table> {
            self.inner.load_table(t).await
        }
        async fn drop_table(&self, t: &crate::TableIdent) -> Result<()> {
            self.inner.drop_table(t).await
        }
        async fn table_exists(&self, t: &crate::TableIdent) -> Result<bool> {
            self.inner.table_exists(t).await
        }
        async fn rename_table(
            &self,
            src: &crate::TableIdent,
            dst: &crate::TableIdent,
        ) -> Result<()> {
            self.inner.rename_table(src, dst).await
        }
        async fn register_table(
            &self,
            t: &crate::TableIdent,
            metadata_location: String,
        ) -> Result<Table> {
            self.inner.register_table(t, metadata_location).await
        }
        async fn update_table(&self, commit: crate::TableCommit) -> Result<Table> {
            self.update_attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let remaining = self.fail_updates.load(std::sync::atomic::Ordering::SeqCst);
            if remaining > 0 {
                self.fail_updates
                    .store(remaining - 1, std::sync::atomic::Ordering::SeqCst);
                return Err(crate::Error::new(
                    crate::ErrorKind::Unexpected,
                    "injected fault: store/catalog unavailable during register \
                     (crash between write and register)",
                ));
            }
            let retryable = self
                .fail_updates_retryable
                .load(std::sync::atomic::Ordering::SeqCst);
            if retryable > 0 {
                self.fail_updates_retryable
                    .store(retryable - 1, std::sync::atomic::Ordering::SeqCst);
                return Err(crate::Error::new(
                    crate::ErrorKind::Unexpected,
                    "injected transient fault: store/catalog temporarily unavailable",
                )
                .with_retryable(true));
            }
            self.inner.update_table(commit).await
        }
    }

    /// M3 real-code — a register fault must be ATOMIC and RECOVERABLE. The data/
    /// manifest writes for the appended file land, but the catalog register fails.
    /// The commit must fail, the authoritative table must be unchanged (the new file
    /// never becomes reachable — it is an orphan for GC, cf. the M1 model), and a
    /// fresh retry after the fault clears must land the data with no loss or dup.
    #[tokio::test]
    async fn dst_fault_commit_register_failure_is_atomic() {
        let catalog = FaultyCatalog::new(new_memory_catalog().await);
        let table = dst_make_v4_table_in_catalog(&catalog).await;
        let ident = table.identifier().clone();

        // Clean commit of A, B.
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/A.parquet", 1), dst_data_file("test/B.parquet", 2)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let ab: std::collections::BTreeSet<String> = ["test/A.parquet", "test/B.parquet"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(dst_live_paths(&table).await, ab);

        // Inject the register fault, then attempt to append C.
        catalog.fail_next_updates(1);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/C.parquet", 3)])
            .apply(tx)
            .unwrap();
        let res = tx.commit(&catalog).await;
        assert!(
            res.is_err(),
            "a failed register must surface as a failed commit, not silent success"
        );

        // ATOMICITY: the authoritative table is unchanged — C never became reachable.
        let reloaded = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
        assert_eq!(
            dst_live_paths(&reloaded).await,
            ab,
            "failed commit advanced the table / left a phantom row"
        );

        // RECOVERY: the fault cleared; a fresh commit of C lands with no loss/dup
        // (the orphan from the crashed attempt does not interfere).
        let tx = Transaction::new(&reloaded);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/C.parquet", 3)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let abc: std::collections::BTreeSet<String> =
            ["test/A.parquet", "test/B.parquet", "test/C.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(
            dst_live_paths(&table).await,
            abc,
            "recovery commit after a register fault lost or duplicated data"
        );
        assert_eq!(dst_live_count(&table).await, 3);
    }

    /// M3 real-code — a register fault during a REPLACE (compaction) must not delete
    /// or corrupt data. The replace writes its merged file and plans the deletes, but
    /// the register fails: the table must stay exactly at {A,B,C} (no half-applied
    /// compaction), and a clean re-run must then produce {C,D}.
    #[tokio::test]
    async fn dst_fault_replace_register_failure_preserves_data() {
        let catalog = FaultyCatalog::new(new_memory_catalog().await);
        let table = dst_make_v4_table_in_catalog(&catalog).await;
        let ident = table.identifier().clone();

        let a = dst_data_file("test/A.parquet", 1);
        let b = dst_data_file("test/B.parquet", 2);
        let c = dst_data_file("test/C.parquet", 3);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a.clone(), b.clone(), c])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let abc: std::collections::BTreeSet<String> =
            ["test/A.parquet", "test/B.parquet", "test/C.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(dst_live_paths(&table).await, abc);

        // Replace A,B -> D, but the register faults.
        catalog.fail_next_updates(1);
        let snap = table.metadata().current_snapshot().unwrap().snapshot_id();
        let res = Transaction::new(&table)
            .replace_data_files()
            .delete_files(vec![a.clone(), b.clone()])
            .add_files(vec![dst_data_file("test/D.parquet", 3)])
            .validate_from_snapshot(snap)
            .apply(Transaction::new(&table))
            .unwrap()
            .commit(&catalog)
            .await;
        assert!(res.is_err(), "a faulted replace register must fail the commit");

        // ATOMICITY: nothing deleted, nothing added.
        let reloaded = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
        assert_eq!(
            dst_live_paths(&reloaded).await,
            abc,
            "a failed replace half-applied: deleted data or advanced the table"
        );

        // RECOVERY: clean replace now yields {C,D}.
        let snap = reloaded.metadata().current_snapshot().unwrap().snapshot_id();
        let table = Transaction::new(&reloaded)
            .replace_data_files()
            .delete_files(vec![a, b])
            .add_files(vec![dst_data_file("test/D.parquet", 3)])
            .validate_from_snapshot(snap)
            .apply(Transaction::new(&reloaded))
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();
        let cd: std::collections::BTreeSet<String> = ["test/C.parquet", "test/D.parquet"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(dst_live_paths(&table).await, cd);
        assert_eq!(dst_live_count(&table).await, 2);
    }

    /// M3 real-code — SEEDED fault schedule. Over a random append schedule, randomly
    /// crash the register before some commits. The invariant across the whole run:
    /// the reachable set equals EXACTLY the set of successfully-committed files —
    /// never losing a committed append, never making a crashed (unregistered) write
    /// reachable. Deterministic + replayable by seed. This is the real-code
    /// counterpart of the M1 over-deletion/orphan sim.
    #[tokio::test]
    async fn dst_fault_seeded_commit_crash_no_loss_no_phantom() {
        fn mix(s: &mut u64) -> u64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for seed in 0..10u64 {
            let mut rng = seed ^ 0xFA01_7ED5;
            let catalog = FaultyCatalog::new(new_memory_catalog().await);
            let mut table = dst_make_v4_table_in_catalog(&catalog).await;
            let ident = table.identifier().clone();
            let mut oracle: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let mut ctr = 0u64;

            for step in 0..16u64 {
                let crash = mix(&mut rng) % 100 < 35;
                if crash {
                    catalog.fail_next_updates(1);
                }
                let k = 1 + mix(&mut rng) % 3;
                let mut files = Vec::new();
                let mut paths = Vec::new();
                for _ in 0..k {
                    ctr += 1;
                    let p = format!("dst/{seed}/f{ctr}.parquet");
                    files.push(dst_data_file(&p, 1 + mix(&mut rng) % 50));
                    paths.push(p);
                }
                let tx = Transaction::new(&table);
                let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
                let res = tx.commit(&catalog).await;

                if crash {
                    assert!(
                        res.is_err(),
                        "seed={seed} step={step}: injected register crash didn't fail the commit"
                    );
                    // The crashed writes stay unregistered → reload the prior state.
                    table = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
                } else {
                    table = res.expect("clean commit failed unexpectedly");
                    for p in paths {
                        oracle.insert(p);
                    }
                }

                assert_eq!(
                    dst_live_paths(&table).await,
                    oracle,
                    "seed={seed} step={step}: reachable set != successfully-committed set \
                     (lost a commit or a crashed write became reachable)"
                );
                assert_eq!(
                    dst_live_count(&table).await,
                    oracle.len(),
                    "seed={seed} step={step}: duplicate reachable files"
                );
            }
        }
    }

    /// M3 real-code — a TRANSIENT (retryable) register fault must be absorbed by the
    /// real commit retry loop. Two retryable faults are injected before an append;
    /// the production `backon` loop (default 4 retries) must re-run the commit and
    /// land the data. Proves the real retry path recovers from flaky object-store /
    /// catalog register calls — the complement of the non-retryable crash test.
    #[tokio::test]
    async fn dst_fault_transient_retryable_append_recovers() {
        let catalog = FaultyCatalog::new(new_memory_catalog().await);
        let table = dst_make_v4_table_in_catalog(&catalog).await;

        catalog.fail_next_updates_retryable(2);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/A.parquet", 1)])
            .apply(tx)
            .unwrap();
        let table = tx
            .commit(&catalog)
            .await
            .expect("retry loop must absorb transient register faults and land the append");

        let a: std::collections::BTreeSet<String> =
            ["test/A.parquet"].iter().map(|s| s.to_string()).collect();
        assert_eq!(dst_live_paths(&table).await, a);
        // 2 faulted attempts + 1 that succeeded = at least 3 register calls.
        assert!(
            catalog.attempts() >= 3,
            "expected the commit to retry through the injected faults, saw {} attempt(s)",
            catalog.attempts()
        );
    }

    /// M3 real-code — the sharp interaction of the `disable_retry` wiring with a
    /// RETRYABLE fault. A replace sets `disable_retry`, so even a retryable register
    /// error must NOT be retried: the commit fails on the first attempt and the table
    /// is untouched. This is what stops a retry from re-applying the stale delete-list
    /// (the M2 duplicate incident) even when the failure looks transient.
    #[tokio::test]
    async fn dst_fault_transient_retryable_replace_not_retried() {
        let catalog = FaultyCatalog::new(new_memory_catalog().await);
        let table = dst_make_v4_table_in_catalog(&catalog).await;

        let a = dst_data_file("test/A.parquet", 1);
        let b = dst_data_file("test/B.parquet", 2);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a.clone(), b.clone()])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let attempts_before = catalog.attempts();

        // A single retryable fault would be absorbed by a fast_append; a replace must
        // refuse to retry and fail immediately.
        catalog.fail_next_updates_retryable(1);
        let snap = table.metadata().current_snapshot().unwrap().snapshot_id();
        let res = Transaction::new(&table)
            .replace_data_files()
            .delete_files(vec![a])
            .add_files(vec![dst_data_file("test/C.parquet", 1)])
            .validate_from_snapshot(snap)
            .apply(Transaction::new(&table))
            .unwrap()
            .commit(&catalog)
            .await;
        assert!(
            res.is_err(),
            "disable_retry must suppress retry even for a retryable error on a replace"
        );
        assert_eq!(
            catalog.attempts() - attempts_before,
            1,
            "replace must make exactly one register attempt (no retry), saw {}",
            catalog.attempts() - attempts_before
        );

        // Table untouched: still {A, B}.
        let ab: std::collections::BTreeSet<String> = ["test/A.parquet", "test/B.parquet"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(dst_live_paths(&table).await, ab);
    }

    /// M3 real-code — SEEDED transient-fault schedule. Before each append, inject
    /// 0..=2 retryable register faults; every commit must still land (the retry loop
    /// absorbs them) and the reachable set must equal the cumulative appended set —
    /// no loss, no duplicate — under a random storm of transient flakiness.
    /// Deterministic + replayable by seed.
    #[tokio::test]
    async fn dst_fault_seeded_transient_no_loss() {
        fn mix(s: &mut u64) -> u64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for seed in 0..8u64 {
            let mut rng = seed ^ 0x7A11_5EED;
            let catalog = FaultyCatalog::new(new_memory_catalog().await);
            let mut table = dst_make_v4_table_in_catalog(&catalog).await;
            let mut oracle: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let mut ctr = 0u64;

            for _step in 0..12u64 {
                let faults = mix(&mut rng) % 3; // 0, 1, or 2 transient faults
                if faults > 0 {
                    catalog.fail_next_updates_retryable(faults as usize);
                }
                let k = 1 + mix(&mut rng) % 3;
                let mut files = Vec::new();
                for _ in 0..k {
                    ctr += 1;
                    let p = format!("dst/{seed}/t{ctr}.parquet");
                    files.push(dst_data_file(&p, 1 + mix(&mut rng) % 50));
                    oracle.insert(p);
                }
                let tx = Transaction::new(&table);
                let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
                table = tx
                    .commit(&catalog)
                    .await
                    .expect("append must survive transient faults via the retry loop");

                assert_eq!(
                    dst_live_paths(&table).await,
                    oracle,
                    "seed={seed}: reachable set != cumulative appends under transient faults"
                );
                assert_eq!(
                    dst_live_count(&table).await,
                    oracle.len(),
                    "seed={seed}: duplicate reachable files under transient faults"
                );
            }
        }
    }

    /// M3 real-code — CLOCK SKEW. History is ordered by SEQUENCE NUMBER (skew-immune),
    /// and snapshot timestamps carry a 1-minute tolerance, so a backwards/skewed wall
    /// clock can neither reorder history nor slip a stale snapshot in. (OCC conflict
    /// detection is snapshot-id based — RefSnapshotIdMatch — and is exercised by the
    /// conflicting-replace tests.) This drives the real MetadataBuilder::add_snapshot
    /// checks with synthetic snapshots whose (sequence, timestamp) we control.
    #[tokio::test]
    async fn dst_clockskew_ordering_and_tolerance() {
        const ONE_MINUTE_MS: i64 = 60_000;
        let catalog = new_memory_catalog().await;
        let table = dst_make_v4_table_in_catalog(&catalog).await;
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/A.parquet", 1)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let cur = table.metadata().current_snapshot().unwrap().clone();
        let next_row_id = table.metadata().next_row_id();
        let schema_id = cur.schema_id().unwrap_or(0);
        let mk = |id_delta: i64, seq: i64, ts: i64| -> crate::spec::Snapshot {
            crate::spec::Snapshot::builder()
                .with_snapshot_id(cur.snapshot_id() + id_delta)
                .with_parent_snapshot_id(Some(cur.snapshot_id()))
                .with_sequence_number(seq)
                .with_timestamp_ms(ts)
                .with_manifest_list(cur.manifest_list().to_string())
                .with_summary(cur.summary().clone())
                .with_schema_id(schema_id)
                .with_row_range(next_row_id, 0u64)
                .build()
        };

        // 1. Gross backwards skew (5 min) with a valid higher sequence number —
        //    REJECTED by the timestamp tolerance guard.
        let big_skew = mk(1, cur.sequence_number() + 1, cur.timestamp_ms() - 5 * ONE_MINUTE_MS);
        assert!(
            table
                .metadata()
                .clone()
                .into_builder(None)
                .add_snapshot(big_skew)
                .is_err(),
            "a snapshot timestamped >1min before the last must be rejected (backwards clock)"
        );

        // 2. Small skew (10s) within tolerance — ACCEPTED (concurrent machines drift).
        let small_skew = mk(2, cur.sequence_number() + 1, cur.timestamp_ms() - 10_000);
        assert!(
            table
                .metadata()
                .clone()
                .into_builder(None)
                .add_snapshot(small_skew)
                .is_ok(),
            "small (<1min) clock skew must be tolerated"
        );

        // 3. Non-increasing sequence number (even with a fine timestamp) — REJECTED.
        //    Ordering is by sequence number, not wall clock: the skew-immune invariant.
        let stale_seq = mk(3, cur.sequence_number(), cur.timestamp_ms() + 1_000);
        assert!(
            table
                .metadata()
                .clone()
                .into_builder(None)
                .add_snapshot(stale_seq)
                .is_err(),
            "a non-increasing sequence number must be rejected (ordering is by sequence, not clock)"
        );
    }

    // ---------------------------------------------------------------------------
    // M3 real-code BYTE-LEVEL fault injection (below FileIO).
    //
    // These use a byte-level opendal fault layer wrapping an in-memory store, so the
    // fault hits the RAW read/write/delete the commit and scan paths perform — a
    // manifest write that fails mid-commit, a manifest read that fails during scan
    // planning. This is strictly deeper than the catalog-level FaultyCatalog, which
    // can only fail the register step (a fault that has already survived every store
    // write). Errors from opendal map to non-retryable iceberg errors, so a byte
    // fault fails the commit outright.
    // ---------------------------------------------------------------------------

    /// Fallible variant of `dst_live_paths` — a scan whose storage reads may fault.
    async fn dst_try_live_paths(
        table: &crate::table::Table,
    ) -> Result<std::collections::BTreeSet<String>> {
        use futures::TryStreamExt;
        let scan = table.scan().select_all().build()?;
        let tasks: Vec<_> = scan.plan_files().await?.try_collect().await?;
        Ok(tasks.into_iter().map(|t| t.data_file_path).collect())
    }

    /// Build a V4 table over an in-memory store wrapped with a byte-level fault layer.
    async fn dst_make_v4_faulty(
        ctrl: std::sync::Arc<crate::io::fault_layer::FaultController>,
    ) -> (crate::memory::MemoryCatalog, Table) {
        let file_io = crate::io::FileIO::memory_with_faults(ctrl);
        let catalog = crate::memory::MemoryCatalog::new_with_file_io("memory://dst", file_io);
        let table = dst_make_v4_table_in_catalog(&catalog).await;
        (catalog, table)
    }

    /// M3 real-code — a BYTE-LEVEL write fault (a manifest write failing mid-commit)
    /// must be atomic and recoverable: the commit fails, the table is unchanged (no
    /// half-written snapshot becomes reachable), and a clean retry lands the data.
    /// Deeper than the catalog test — this fails the store write, before the register.
    #[tokio::test]
    async fn dst_bytefault_write_failure_is_atomic() {
        let ctrl = crate::io::fault_layer::FaultController::new();
        let (catalog, table) = dst_make_v4_faulty(ctrl.clone()).await;
        let ident = table.identifier().clone();

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/A.parquet", 1), dst_data_file("test/B.parquet", 2)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let ab: std::collections::BTreeSet<String> = ["test/A.parquet", "test/B.parquet"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(dst_live_paths(&table).await, ab);

        // Fail the next raw write — the manifest write inside the commit.
        ctrl.fail_next_writes(1);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/C.parquet", 3)])
            .apply(tx)
            .unwrap();
        assert!(
            tx.commit(&catalog).await.is_err(),
            "a failed manifest write must fail the commit"
        );

        // ATOMICITY: the authoritative table is unchanged.
        let reloaded = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
        assert_eq!(
            dst_live_paths(&reloaded).await,
            ab,
            "a failed write advanced the table / left a phantom row"
        );

        // RECOVERY: with the store healthy, the commit of C lands with no loss/dup.
        let tx = Transaction::new(&reloaded);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/C.parquet", 3)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let abc: std::collections::BTreeSet<String> =
            ["test/A.parquet", "test/B.parquet", "test/C.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(dst_live_paths(&table).await, abc);
        assert_eq!(dst_live_count(&table).await, 3);
    }

    /// M3 real-code — a BYTE-LEVEL read fault during scan planning must surface as a
    /// scan ERROR, never as silently fewer rows. A storage read that fails is the
    /// most dangerous fault class: if the planner swallowed it, a transient S3 blip
    /// would look like data loss. The committed data is intact; only the read faults.
    #[tokio::test]
    async fn dst_bytefault_read_failure_scan_errors_not_silent() {
        let ctrl = crate::io::fault_layer::FaultController::new();
        let (catalog, table) = dst_make_v4_faulty(ctrl.clone()).await;
        let ident = table.identifier().clone();

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![
                dst_data_file("test/A.parquet", 1),
                dst_data_file("test/B.parquet", 2),
                dst_data_file("test/C.parquet", 3),
            ])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        assert_eq!(dst_live_count(&table).await, 3);

        // Fresh load → cold cache, so the scan reads manifests from the store.
        let fresh = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
        // Fail the first manifest read the scan performs.
        ctrl.fail_next_reads(1);
        assert!(
            dst_try_live_paths(&fresh).await.is_err(),
            "a manifest read fault must surface as a scan error, never silent data loss"
        );
        assert!(ctrl.reads() >= 1, "the scan must have attempted a manifest read");
    }

    /// M3 real-code — SEEDED byte-level write-crash schedule. Before each append,
    /// randomly fail the manifest write. Every faulted commit must fail; the
    /// reachable set must always equal exactly the set of successfully-committed
    /// files — no lost commit, no half-written snapshot made reachable. Deterministic.
    #[tokio::test]
    async fn dst_bytefault_seeded_write_crash_no_loss() {
        fn mix(s: &mut u64) -> u64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        for seed in 0..8u64 {
            let mut rng = seed ^ 0xB17E_FA01;
            let ctrl = crate::io::fault_layer::FaultController::new();
            let (catalog, mut table) = dst_make_v4_faulty(ctrl.clone()).await;
            let ident = table.identifier().clone();
            let mut oracle: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
            let mut ctr = 0u64;

            for step in 0..12u64 {
                let crash = mix(&mut rng) % 100 < 35;
                if crash {
                    ctrl.fail_next_writes(1);
                }
                let k = 1 + mix(&mut rng) % 3;
                let mut files = Vec::new();
                let mut paths = Vec::new();
                for _ in 0..k {
                    ctr += 1;
                    let p = format!("dst/{seed}/b{ctr}.parquet");
                    files.push(dst_data_file(&p, 1 + mix(&mut rng) % 50));
                    paths.push(p);
                }
                let tx = Transaction::new(&table);
                let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
                let res = tx.commit(&catalog).await;

                if crash {
                    assert!(
                        res.is_err(),
                        "seed={seed} step={step}: injected write crash didn't fail the commit"
                    );
                    table = crate::Catalog::load_table(&catalog, &ident).await.unwrap();
                } else {
                    table = res.expect("clean commit failed unexpectedly");
                    for p in paths {
                        oracle.insert(p);
                    }
                }

                assert_eq!(
                    dst_live_paths(&table).await,
                    oracle,
                    "seed={seed} step={step}: reachable set != successfully-committed set"
                );
                assert_eq!(
                    dst_live_count(&table).await,
                    oracle.len(),
                    "seed={seed} step={step}: duplicate reachable files"
                );
            }
        }
    }

    /// M3 real-code — NETWORK PARTITION. While the store is unreachable, EVERY read
    /// and write fails: a commit cannot land and a cold scan cannot read. The table
    /// must be unchanged through the partition, and once the partition HEALS, both
    /// commit and scan recover — no data lost, no phantom introduced.
    #[tokio::test]
    async fn dst_bytefault_partition_then_heal() {
        let ctrl = crate::io::fault_layer::FaultController::new();
        let (catalog, table) = dst_make_v4_faulty(ctrl.clone()).await;
        let ident = table.identifier().clone();

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/A.parquet", 1), dst_data_file("test/B.parquet", 2)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let ab: std::collections::BTreeSet<String> = ["test/A.parquet", "test/B.parquet"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(dst_live_paths(&table).await, ab);

        // Fresh (cold-cache) handle so the scan below reads manifests from the store.
        let fresh = crate::Catalog::load_table(&catalog, &ident).await.unwrap();

        // PARTITION: the store is unreachable.
        ctrl.partition();
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/C.parquet", 3)])
            .apply(tx)
            .unwrap();
        assert!(
            tx.commit(&catalog).await.is_err(),
            "a commit during a store partition must fail"
        );
        assert!(
            dst_try_live_paths(&fresh).await.is_err(),
            "a cold scan during a store partition must fail, not read stale/empty"
        );

        // HEAL: the store is reachable again.
        ctrl.heal();
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![dst_data_file("test/C.parquet", 3)])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();
        let abc: std::collections::BTreeSet<String> =
            ["test/A.parquet", "test/B.parquet", "test/C.parquet"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        assert_eq!(
            dst_live_paths(&table).await,
            abc,
            "commit did not recover after the partition healed"
        );
        // Reads recover too (the healed scan succeeds).
        assert!(
            dst_try_live_paths(&fresh).await.is_ok(),
            "scan did not recover after the partition healed"
        );
    }

    /// M3 real-code — LATENCY. Injecting per-operation latency below FileIO must not
    /// change results: a full append + compaction under added store latency yields
    /// exactly the same live set. Guards against timing-dependent races in the
    /// commit/scan path. (Latency is timing, not a fault; correctness is the assertion.)
    #[tokio::test]
    async fn dst_bytefault_latency_preserves_correctness() {
        let ctrl = crate::io::fault_layer::FaultController::new();
        let (catalog, table) = dst_make_v4_faulty(ctrl.clone()).await;
        ctrl.set_latency_ms(2);

        let a = dst_data_file("test/A.parquet", 1);
        let b = dst_data_file("test/B.parquet", 2);
        let c = dst_data_file("test/C.parquet", 3);
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![a.clone(), b.clone(), c])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Compact A,B -> D under latency.
        let snap = table.metadata().current_snapshot().unwrap().snapshot_id();
        let table = Transaction::new(&table)
            .replace_data_files()
            .delete_files(vec![a, b])
            .add_files(vec![dst_data_file("test/D.parquet", 3)])
            .validate_from_snapshot(snap)
            .apply(Transaction::new(&table))
            .unwrap()
            .commit(&catalog)
            .await
            .unwrap();

        let cd: std::collections::BTreeSet<String> = ["test/C.parquet", "test/D.parquet"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(dst_live_paths(&table).await, cd);
        assert_eq!(dst_live_count(&table).await, 2);
        // Sanity: operations actually flowed through the latency-injecting layer.
        assert!(
            ctrl.reads() > 0 && ctrl.writes() > 0,
            "expected reads and writes to traverse the fault layer"
        );
    }
}
