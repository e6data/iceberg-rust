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
use std::future::Future;
use std::ops::RangeFrom;

use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFile, DataFileFormat, FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestEntry,
    ManifestFile, ManifestListWriter, ManifestWriter, ManifestWriterBuilder, Operation, Snapshot,
    SnapshotReference, SnapshotRetention, SnapshotSummaryCollector, Struct, StructType, Summary,
    TableProperties, update_snapshot_summaries,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Extract a string key from the first partition field value for grouping.
/// Returns "__null__" for null first fields or "__empty__" for unpartitioned.
fn first_partition_value_key(partition: &Struct) -> String {
    let fields = partition.fields();
    if fields.is_empty() {
        return "__empty__".to_string();
    }
    match &fields[0] {
        Some(literal) => format!("{literal:?}"),
        None => "__null__".to_string(),
    }
}

/// A trait that defines how different table operations produce new snapshots.
///
/// `SnapshotProduceOperation` is used by [`SnapshotProducer`] to customize snapshot creation
/// based on the type of operation being performed (e.g., `Append`, `Overwrite`, `Delete`, etc.).
/// Each operation type implements this trait to specify:
/// - Which operation type to record in the snapshot summary
/// - Which existing manifest files should be included in the new snapshot
/// - Which manifest entries should be marked as deleted
///
/// # When it accomplishes
///
/// This trait is used during the snapshot creation process in [`SnapshotProducer::commit()`]:
///
/// 1. **Operation Type Recording**: The `operation()` method determines which operation type
///    (e.g., `Operation::Append`, `Operation::Overwrite`) is recorded in the snapshot summary.
///    This metadata helps track what kind of change was made to the table.
///
/// 2. **Manifest File Selection**: The `existing_manifest()` method determines which existing
///    manifest files from the current snapshot should be carried forward to the new snapshot.
///    For example:
///    - An `Append` operation typically includes all existing manifests plus new ones
///    - An `Overwrite` operation might exclude manifests for partitions being overwritten
///
/// 3. **Delete Entry Processing**: The `delete_entries()` method is intended for future delete
///    operations to specify which manifest entries should be marked as deleted.
pub(crate) trait SnapshotProduceOperation: Send + Sync {
    /// Returns the operation type that will be recorded in the snapshot summary.
    ///
    /// This determines what kind of operation is being performed (e.g., `Append`, `Overwrite`),
    /// which is stored in the snapshot metadata for tracking and auditing purposes.
    fn operation(&self) -> Operation;

    /// Returns manifest entries that should be marked as deleted in the new snapshot.
    #[allow(unused)]
    fn delete_entries(
        &self,
        snapshot_produce: &SnapshotProducer,
    ) -> impl Future<Output = Result<Vec<ManifestEntry>>> + Send;

    /// Returns existing manifest files that should be included in the new snapshot.
    ///
    /// This method determines which manifest files from the current snapshot should be
    /// carried forward to the new snapshot. The selection depends on the operation type:
    ///
    /// - **Append operations**: Typically include all existing manifests
    /// - **Overwrite operations**: May exclude manifests for partitions being overwritten
    /// - **Delete operations**: May exclude manifests for partitions being deleted
    fn existing_manifest(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
    ) -> impl Future<Output = Result<Vec<ManifestFile>>> + Send;
}

pub(crate) struct DefaultManifestProcess;

impl ManifestProcess for DefaultManifestProcess {
    fn process_manifests(
        &self,
        _snapshot_produce: &SnapshotProducer<'_>,
        manifests: Vec<ManifestFile>,
    ) -> Vec<ManifestFile> {
        manifests
    }
}

pub(crate) trait ManifestProcess: Send + Sync {
    fn process_manifests(
        &self,
        snapshot_produce: &SnapshotProducer<'_>,
        manifests: Vec<ManifestFile>,
    ) -> Vec<ManifestFile>;
}

pub(crate) struct SnapshotProducer<'a> {
    pub(crate) table: &'a Table,
    snapshot_id: i64,
    commit_uuid: Uuid,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    added_data_files: Vec<DataFile>,
    removed_data_files: Vec<DataFile>,
    data_sequence_number: Option<i64>,
    added_delete_files: Vec<DataFile>,
    // A counter used to generate unique manifest file names.
    // It starts from 0 and increments for each new manifest file.
    // Note: This counter is limited to the range of (0..u64::MAX).
    manifest_counter: RangeFrom<u64>,
}

impl<'a> SnapshotProducer<'a> {
    pub(crate) fn new(
        table: &'a Table,
        commit_uuid: Uuid,
        key_metadata: Option<Vec<u8>>,
        snapshot_properties: HashMap<String, String>,
        added_data_files: Vec<DataFile>,
        added_delete_files: Vec<DataFile>,
    ) -> Self {
        Self {
            table,
            snapshot_id: Self::generate_unique_snapshot_id(table),
            commit_uuid,
            key_metadata,
            snapshot_properties,
            added_data_files,
            removed_data_files: Vec::new(),
            data_sequence_number: None,
            added_delete_files,
            manifest_counter: (0..),
        }
    }

    pub(crate) fn with_removed_data_files(mut self, files: Vec<DataFile>) -> Self {
        self.removed_data_files = files;
        self
    }

    pub(crate) fn with_data_sequence_number(mut self, seq_num: Option<i64>) -> Self {
        self.data_sequence_number = seq_num;
        self
    }

    pub(crate) fn validate_added_data_files(&self) -> Result<()> {
        for data_file in &self.added_data_files {
            if data_file.content_type() != crate::spec::DataContentType::Data {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Only data content type is allowed for fast append",
                ));
            }
            // Validate partition value against the file's own spec (not necessarily
            // the default). Compaction may produce files with older spec IDs when
            // merging historical data written before a partition evolution.
            if let Some(spec) = self
                .table
                .metadata()
                .partition_spec_by_id(data_file.partition_spec_id)
            {
                let partition_type = spec
                    .partition_type(self.table.metadata().current_schema())
                    .map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("invalid partition spec: {e}"),
                        )
                    })?;
                Self::validate_partition_value(data_file.partition(), &partition_type)?;
            }
            // If spec not found, skip validation (the commit will succeed and
            // Lakekeeper validates at the catalog level).
        }

        Ok(())
    }

    pub(crate) async fn validate_duplicate_files(&self) -> Result<()> {
        let new_files: HashSet<&str> = self
            .added_data_files
            .iter()
            .map(|df| df.file_path.as_str())
            .collect();

        let mut referenced_files = Vec::new();
        if let Some(current_snapshot) = self.table.metadata().current_snapshot() {
            let manifest_list = current_snapshot
                .load_manifest_list(self.table.file_io(), &self.table.metadata_ref())
                .await?;
            for manifest_list_entry in manifest_list.entries() {
                let manifest = manifest_list_entry
                    .load_manifest(self.table.file_io())
                    .await?;
                for entry in manifest.entries() {
                    let file_path = entry.file_path();
                    if new_files.contains(file_path) && entry.is_alive() {
                        referenced_files.push(file_path.to_string());
                    }
                }
            }
        }

        if !referenced_files.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot add files that are already referenced by table, files: {}",
                    referenced_files.join(", ")
                ),
            ));
        }

        Ok(())
    }

    pub(crate) fn generate_unique_snapshot_id_static(table: &Table) -> i64 {
        Self::generate_unique_snapshot_id(table)
    }

    fn generate_unique_snapshot_id(table: &Table) -> i64 {
        let generate_random_id = || -> i64 {
            let (lhs, rhs) = Uuid::new_v4().as_u64_pair();
            let snapshot_id = (lhs ^ rhs) as i64;
            if snapshot_id < 0 {
                -snapshot_id
            } else {
                snapshot_id
            }
        };
        let mut snapshot_id = generate_random_id();

        while table
            .metadata()
            .snapshots()
            .any(|s| s.snapshot_id() == snapshot_id)
        {
            snapshot_id = generate_random_id();
        }
        snapshot_id
    }

    /// Returns the snapshot ID this producer is committing to.
    /// Used by callers (e.g. `ReplaceDataFilesAction`) when they rewrite
    /// existing manifests inline and need to attribute the rewritten
    /// manifest to the new commit.
    pub(crate) fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    /// Whether this table should use Parquet manifests.
    ///
    /// Whether to write manifests as Parquet (default for V2+) or Avro.
    ///
    /// Parquet manifests enable columnar projection during query planning.
    /// Default: Parquet for V2+ tables. Set `write.parquet.metadata-codec = avro`
    /// to opt out for ecosystem compatibility (Spark, Trino, Flink).
    fn use_parquet_manifests(&self) -> bool {
        let prop = self
            .table
            .metadata()
            .properties()
            .get("write.parquet.metadata-codec");
        match prop.map(|v| v.as_str()) {
            Some(v) if v.eq_ignore_ascii_case("avro") => false,
            _ => matches!(
                self.table.metadata().format_version(),
                FormatVersion::V2 | FormatVersion::V3
            ),
        }
    }

    fn new_manifest_writer(&mut self, content: ManifestContentType) -> Result<ManifestWriter> {
        let ext = if self.use_parquet_manifests() { "parquet" } else { "avro" };
        let new_manifest_path = format!(
            "{}/{}/{}-m{}.{}",
            self.table.metadata().location(),
            META_ROOT_PATH,
            self.commit_uuid,
            self.manifest_counter.next().unwrap(),
            ext
        );
        let output_file = self.table.file_io().new_output(new_manifest_path)?;
        let builder = ManifestWriterBuilder::new(
            output_file,
            Some(self.snapshot_id),
            self.key_metadata.clone(),
            self.table.metadata().current_schema().clone(),
            self.table
                .metadata()
                .default_partition_spec()
                .as_ref()
                .clone(),
        );
        match self.table.metadata().format_version() {
            FormatVersion::V1 => Ok(builder.build_v1()),
            FormatVersion::V2 => match content {
                ManifestContentType::Data => Ok(builder.build_v2_data()),
                ManifestContentType::Deletes => Ok(builder.build_v2_deletes()),
            },
            FormatVersion::V3 => match content {
                ManifestContentType::Data => Ok(builder.build_v3_data()),
                ManifestContentType::Deletes => Ok(builder.build_v3_deletes()),
            },
        }
    }

    // Check if the partition value is compatible with the partition type.
    fn validate_partition_value(
        partition_value: &Struct,
        partition_type: &StructType,
    ) -> Result<()> {
        if partition_value.fields().len() != partition_type.fields().len() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Partition value is not compatible with partition type",
            ));
        }

        for (value, field) in partition_value.fields().iter().zip(partition_type.fields()) {
            let field = field.field_type.as_primitive_type().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Partition field should only be primitive type.",
                )
            })?;
            if let Some(value) = value
                && !field.compatible(&value.as_primitive_literal().unwrap())
            {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Partition value is not compatible partition type",
                ));
            }
        }
        Ok(())
    }

    /// Try to merge new data files into the most recent existing data manifest
    /// if it's below the entry count threshold. Returns true if merged.
    ///
    /// This reduces manifest proliferation from frequent micro-batch commits
    /// (e.g., 30-second OTel ingestion cycles). Instead of creating 2,880
    /// manifests/day, small manifests get merged inline.
    async fn try_merge_into_existing(
        &mut self,
        existing_manifests: &mut Vec<ManifestFile>,
        min_count: usize,
    ) -> Result<bool> {
        // Find the most recent data manifest that's small enough to merge into
        let merge_candidate_idx = existing_manifests.iter().rposition(|mf| {
            mf.content == ManifestContentType::Data
                && mf.added_files_count.unwrap_or(0) + mf.existing_files_count.unwrap_or(0)
                    < min_count as u32
        });

        let Some(idx) = merge_candidate_idx else {
            return Ok(false);
        };

        // Load the existing manifest entries
        let candidate = &existing_manifests[idx];
        let manifest = candidate.load_manifest(self.table.file_io()).await?;
        let (existing_entries, _metadata) = manifest.into_parts();

        // Create a new merged manifest with existing + new entries
        let mut writer = self.new_manifest_writer(ManifestContentType::Data)?;

        // Re-add existing entries
        for entry_ref in &existing_entries {
            let entry = entry_ref.as_ref();
            if entry.is_alive() {
                writer.add_existing_file(
                    entry.data_file.clone(),
                    entry.snapshot_id.unwrap_or(0),
                    entry.sequence_number.unwrap_or(0),
                    entry.file_sequence_number,
                )?;
            }
        }

        // Add new entries
        let added_data_files = std::mem::take(&mut self.added_data_files);
        let snapshot_id = self.snapshot_id;
        let format_version = self.table.metadata().format_version();
        for data_file in added_data_files {
            let builder = ManifestEntry::builder()
                .status(crate::spec::ManifestStatus::Added)
                .data_file(data_file);
            let entry = if format_version == FormatVersion::V1 {
                builder.snapshot_id(snapshot_id).build()
            } else {
                builder.build()
            };
            writer.add_entry(entry)?;
        }

        // Write merged manifest
        let merged_manifest = if self.use_parquet_manifests() {
            writer.write_manifest_file_parquet().await?
        } else {
            writer.write_manifest_file().await?
        };

        // Replace the old manifest with the merged one
        existing_manifests[idx] = merged_manifest;

        Ok(true)
    }

    /// Write manifest files for added data files, grouped by first partition value.
    ///
    /// When `write.manifest.partition-scoped=true`, files are grouped by the first
    /// partition field value and each group gets its own manifest. This produces tight
    /// partition summaries (lower_bound == upper_bound for the grouping field), enabling
    /// the manifest evaluator to skip 98%+ of manifests during query planning.
    ///
    /// Without this property (default), all files go into a single manifest (original behavior).
    async fn write_added_manifests(&mut self) -> Result<Vec<ManifestFile>> {
        let added_data_files = std::mem::take(&mut self.added_data_files);
        if added_data_files.is_empty() {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "No added data files found when write an added manifest file",
            ));
        }

        // Partition-scoped manifests: default true for partitioned tables.
        // Each tenant gets its own manifest with tight partition bounds,
        // enabling 90%+ manifest-level pruning.
        // Set write.manifest.partition-scoped=false to opt out.
        let partition_scoped = {
            let prop = self
                .table
                .metadata()
                .properties()
                .get("write.manifest.partition-scoped");
            match prop.map(|v| v.as_str()) {
                Some(v) if v.eq_ignore_ascii_case("false") => false,
                _ => true,
            }
        };

        if !partition_scoped {
            // Original behavior: single manifest for all files
            let manifest = self.write_single_manifest(added_data_files).await?;
            return Ok(vec![manifest]);
        }

        // Group files by first partition field value
        let mut groups: HashMap<String, Vec<DataFile>> = HashMap::new();
        for data_file in added_data_files {
            let key = first_partition_value_key(&data_file.partition);
            groups.entry(key).or_default().push(data_file);
        }

        let mut manifests = Vec::with_capacity(groups.len());
        for (_key, files) in groups {
            let manifest = self.write_single_manifest(files).await?;
            manifests.push(manifest);
        }

        Ok(manifests)
    }

    async fn write_single_manifest(&mut self, data_files: Vec<DataFile>) -> Result<ManifestFile> {
        let snapshot_id = self.snapshot_id;
        let format_version = self.table.metadata().format_version();
        let mut writer = self.new_manifest_writer(ManifestContentType::Data)?;

        for data_file in data_files {
            let builder = ManifestEntry::builder()
                .status(crate::spec::ManifestStatus::Added)
                .data_file(data_file);
            let entry = if format_version == FormatVersion::V1 {
                builder.snapshot_id(snapshot_id).build()
            } else {
                builder.build()
            };
            writer.add_entry(entry)?;
        }

        if self.use_parquet_manifests() {
            writer.write_manifest_file_parquet().await
        } else {
            writer.write_manifest_file().await
        }
    }

    // Write manifest file for added delete files and return the ManifestFile for ManifestList.
    async fn write_added_delete_manifest(&mut self) -> Result<ManifestFile> {
        let added_delete_files = std::mem::take(&mut self.added_delete_files);
        if added_delete_files.is_empty() {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "No added delete files found when writing a delete manifest file",
            ));
        }

        let snapshot_id = self.snapshot_id;
        let format_version = self.table.metadata().format_version();
        let data_sequence_number = self.data_sequence_number;
        let manifest_entries: Vec<ManifestEntry> = added_delete_files
            .into_iter()
            .map(|data_file| {
                if format_version == FormatVersion::V1 {
                    if let Some(seq) = data_sequence_number {
                        ManifestEntry::builder()
                            .status(crate::spec::ManifestStatus::Added)
                            .snapshot_id(snapshot_id)
                            .sequence_number(seq)
                            .data_file(data_file)
                            .build()
                    } else {
                        ManifestEntry::builder()
                            .status(crate::spec::ManifestStatus::Added)
                            .snapshot_id(snapshot_id)
                            .data_file(data_file)
                            .build()
                    }
                } else if let Some(seq) = data_sequence_number {
                    ManifestEntry::builder()
                        .status(crate::spec::ManifestStatus::Added)
                        .sequence_number(seq)
                        .data_file(data_file)
                        .build()
                } else {
                    ManifestEntry::builder()
                        .status(crate::spec::ManifestStatus::Added)
                        .data_file(data_file)
                        .build()
                }
            })
            .collect();
        let mut writer = self.new_manifest_writer(ManifestContentType::Deletes)?;
        for entry in manifest_entries {
            writer.add_entry(entry)?;
        }
        if self.use_parquet_manifests() {
            writer.write_manifest_file_parquet().await
        } else {
            writer.write_manifest_file().await
        }
    }

    async fn manifest_file<OP: SnapshotProduceOperation, MP: ManifestProcess>(
        &mut self,
        snapshot_produce_operation: &OP,
        manifest_process: &MP,
    ) -> Result<Vec<ManifestFile>> {
        // Assert current snapshot producer contains new content to add to new snapshot.
        //
        // TODO: Allowing snapshot property setup with no added data files is a workaround.
        // We should clean it up after all necessary actions are supported.
        // For details, please refer to https://github.com/apache/iceberg-rust/issues/1548
        if self.added_data_files.is_empty()
            && self.snapshot_properties.is_empty()
            && self.removed_data_files.is_empty()
        {
            return Err(Error::new(
                ErrorKind::PreconditionFailed,
                "No added data files, removed data files, or added snapshot properties found when write a manifest file",
            ));
        }

        let existing_manifests = snapshot_produce_operation.existing_manifest(self).await?;
        let mut manifest_files = existing_manifests;

        // Process added entries.
        // When manifest merging is enabled, merge new entries into the most
        // recent small manifest instead of creating a new one every commit.
        if !self.added_data_files.is_empty() {
            let merge_enabled = self
                .table
                .metadata()
                .properties()
                .get("commit.manifest-merge.enabled")
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let min_count = self
                .table
                .metadata()
                .properties()
                .get("commit.manifest.min-count-to-merge")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(100);

            let merged = if merge_enabled {
                self.try_merge_into_existing(&mut manifest_files, min_count)
                    .await?
            } else {
                false
            };

            if !merged {
                let added_manifests = self.write_added_manifests().await?;
                manifest_files.extend(added_manifests);
            }
        }

        // Process added delete files.
        if !self.added_delete_files.is_empty() {
            let delete_manifest = self.write_added_delete_manifest().await?;
            manifest_files.push(delete_manifest);
        }

        let manifest_files = manifest_process.process_manifests(self, manifest_files);
        Ok(manifest_files)
    }

    // Returns a `Summary` of the current snapshot
    fn summary<OP: SnapshotProduceOperation>(
        &self,
        snapshot_produce_operation: &OP,
    ) -> Result<Summary> {
        let mut summary_collector = SnapshotSummaryCollector::default();
        let table_metadata = self.table.metadata_ref();

        let partition_summary_limit = if let Some(limit) = table_metadata
            .properties()
            .get(TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT)
        {
            if let Ok(limit) = limit.parse::<u64>() {
                limit
            } else {
                TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT_DEFAULT
            }
        } else {
            TableProperties::PROPERTY_WRITE_PARTITION_SUMMARY_LIMIT_DEFAULT
        };

        summary_collector.set_partition_summary_limit(partition_summary_limit);

        for data_file in &self.added_data_files {
            summary_collector.add_file(
                data_file,
                table_metadata.current_schema().clone(),
                table_metadata.default_partition_spec().clone(),
            );
        }

        for data_file in &self.removed_data_files {
            summary_collector.remove_file(
                data_file,
                table_metadata.current_schema().clone(),
                table_metadata.default_partition_spec().clone(),
            );
        }

        for data_file in &self.added_delete_files {
            summary_collector.add_file(
                data_file,
                table_metadata.current_schema().clone(),
                table_metadata.default_partition_spec().clone(),
            );
        }

        let previous_snapshot = table_metadata
            .snapshot_by_id(self.snapshot_id)
            .and_then(|snapshot| snapshot.parent_snapshot_id())
            .and_then(|parent_id| table_metadata.snapshot_by_id(parent_id));

        let mut additional_properties = summary_collector.build();
        additional_properties.extend(self.snapshot_properties.clone());

        let summary = Summary {
            operation: snapshot_produce_operation.operation(),
            additional_properties,
        };

        update_snapshot_summaries(
            summary,
            previous_snapshot.map(|s| s.summary()),
            snapshot_produce_operation.operation() == Operation::Overwrite,
        )
    }

    fn generate_manifest_list_file_path(&self, attempt: i64) -> String {
        format!(
            "{}/{}/snap-{}-{}-{}.{}",
            self.table.metadata().location(),
            META_ROOT_PATH,
            self.snapshot_id,
            attempt,
            self.commit_uuid,
            DataFileFormat::Avro
        )
    }

    /// Finished building the action and return the [`ActionCommit`] to the transaction.
    pub(crate) async fn commit<OP: SnapshotProduceOperation, MP: ManifestProcess>(
        mut self,
        snapshot_produce_operation: OP,
        process: MP,
    ) -> Result<ActionCommit> {
        let manifest_list_path = self.generate_manifest_list_file_path(0);
        let next_seq_num = self.table.metadata().next_sequence_number();
        let first_row_id = self.table.metadata().next_row_id();
        let mut manifest_list_writer = match self.table.metadata().format_version() {
            FormatVersion::V1 => ManifestListWriter::v1(
                self.table
                    .file_io()
                    .new_output(manifest_list_path.clone())?,
                self.snapshot_id,
                self.table.metadata().current_snapshot_id(),
            ),
            FormatVersion::V2 => ManifestListWriter::v2(
                self.table
                    .file_io()
                    .new_output(manifest_list_path.clone())?,
                self.snapshot_id,
                self.table.metadata().current_snapshot_id(),
                next_seq_num,
            ),
            FormatVersion::V3 => ManifestListWriter::v3(
                self.table
                    .file_io()
                    .new_output(manifest_list_path.clone())?,
                self.snapshot_id,
                self.table.metadata().current_snapshot_id(),
                next_seq_num,
                Some(first_row_id),
            ),
        };

        // Calling self.summary() before self.manifest_file() is important because self.added_data_files
        // will be set to an empty vec after self.manifest_file() returns, resulting in an empty summary
        // being generated.
        let summary = self.summary(&snapshot_produce_operation).map_err(|err| {
            Error::new(ErrorKind::Unexpected, "Failed to create snapshot summary.").with_source(err)
        })?;

        let new_manifests = self
            .manifest_file(&snapshot_produce_operation, &process)
            .await?;

        let created_manifest_paths: Vec<String> = new_manifests
            .iter()
            .map(|m| m.manifest_path.clone())
            .collect();

        manifest_list_writer.add_manifests(new_manifests.into_iter())?;
        let writer_next_row_id = manifest_list_writer.next_row_id();
        manifest_list_writer.close().await?;

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(manifest_list_path)
            .with_snapshot_id(self.snapshot_id)
            .with_parent_snapshot_id(self.table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(self.table.metadata().current_schema_id())
            .with_timestamp_ms(commit_ts);

        let new_snapshot = if let Some(writer_next_row_id) = writer_next_row_id {
            let assigned_rows = writer_next_row_id - self.table.metadata().next_row_id();
            new_snapshot
                .with_row_range(first_row_id, assigned_rows)
                .build()
        } else {
            new_snapshot.build()
        };

        let updates = vec![
            TableUpdate::AddSnapshot {
                snapshot: new_snapshot,
            },
            TableUpdate::SetSnapshotRef {
                ref_name: MAIN_BRANCH.to_string(),
                reference: SnapshotReference::new(
                    self.snapshot_id,
                    SnapshotRetention::branch(None, None, None),
                ),
            },
        ];

        let requirements = vec![
            TableRequirement::UuidMatch {
                uuid: self.table.metadata().uuid(),
            },
            TableRequirement::RefSnapshotIdMatch {
                r#ref: MAIN_BRANCH.to_string(),
                snapshot_id: self.table.metadata().current_snapshot_id(),
            },
        ];

        Ok(ActionCommit::new(updates, requirements).with_manifest_paths(created_manifest_paths))
    }
}
