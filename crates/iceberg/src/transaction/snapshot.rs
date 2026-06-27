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
#[allow(unused_imports)]
use std::collections::BTreeSet;
use std::future::Future;
use std::ops::RangeFrom;

use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFile, DataFileFormat, FormatVersion, MAIN_BRANCH, ManifestContentType, ManifestEntry,
    ManifestFile, ManifestListWriter, ManifestStatus, ManifestWriter, ManifestWriterBuilder,
    Operation, Snapshot, SnapshotReference, SnapshotRetention, SnapshotSummaryCollector, Struct,
    StructType, Summary, TableProperties, update_snapshot_summaries,
};
use crate::spec::root_manifest::{
    RootManifestEntry, RootManifestMetadata, read_root_manifest, reconstruct_root,
    write_root_manifest,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Generate a snapshot_id that is guaranteed not to collide with any existing
/// snapshot on `table`. Use this when you need to know the snapshot_id of a
/// commit *before* it runs — for example, to attach `StatisticsFile` entries
/// to the same snapshot the commit will create.
///
/// Pair with [`FastAppendAction::with_snapshot_id`](super::append::FastAppendAction::with_snapshot_id)
/// to ensure the action commits under the pre-allocated id rather than
/// generating a fresh one. Without that pairing, the action will still
/// allocate its own id and the values will diverge.
pub fn generate_unique_snapshot_id(table: &Table) -> i64 {
    SnapshotProducer::generate_unique_snapshot_id_static(table)
}

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
    /// Cached root manifest entries with the snapshot_id they were built for.
    cached_root_entries: Option<(Option<i64>, Vec<RootManifestEntry>)>,
    file_to_manifest_index: Option<HashMap<String, String>>,
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
            cached_root_entries: None,
            file_to_manifest_index: None,
        }
    }

    pub(crate) fn with_removed_data_files(mut self, files: Vec<DataFile>) -> Self {
        self.removed_data_files = files;
        self
    }

    /// Override the auto-generated snapshot_id with a caller-provided one.
    /// Used by actions (e.g. `FastAppendAction.with_snapshot_id`) that need
    /// the snapshot_id to be knowable *before* the commit runs — for example,
    /// when registering `StatisticsFile` entries that reference this snapshot
    /// within the same transaction.
    pub(crate) fn with_snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.snapshot_id = snapshot_id;
        self
    }

    pub(crate) fn with_data_sequence_number(mut self, seq_num: Option<i64>) -> Self {
        self.data_sequence_number = seq_num;
        self
    }

    pub(crate) fn with_cached_root_entries(mut self, snapshot_id: Option<i64>, entries: Vec<RootManifestEntry>) -> Self {
        self.cached_root_entries = Some((snapshot_id, entries));
        self
    }

    pub(crate) fn with_file_to_manifest_index(mut self, index: HashMap<String, String>) -> Self {
        self.file_to_manifest_index = Some(index);
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
            // Also check V4 inline entries (not backed by manifest files)
            for entry in manifest_list.inline_entries() {
                let file_path = entry.file_path();
                if new_files.contains(file_path) && entry.is_alive() {
                    referenced_files.push(file_path.to_string());
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
    /// Whether to write manifests as Parquet (opt-in) or Avro (default).
    ///
    /// Parquet manifests enable columnar projection during query planning
    /// but are NOT part of the ratified Iceberg spec. Standard readers
    /// (Spark, Trino, Flink, DuckDB) cannot read them.
    ///
    /// Opt-in via table property `write.parquet.metadata-codec = parquet`.
    /// Default: Avro (ecosystem compatible).
    fn use_parquet_manifests(&self) -> bool {
        self.table
            .metadata()
            .properties()
            .get("write.parquet.metadata-codec")
            .map(|v| v.eq_ignore_ascii_case("parquet"))
            .unwrap_or(false)
    }

    fn new_manifest_writer(&mut self, content: ManifestContentType) -> Result<ManifestWriter> {
        let ext = if self.use_parquet_manifests() {
            "parquet"
        } else {
            "avro"
        };
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
            FormatVersion::V3 | FormatVersion::V4 => match content {
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

        let partition_scoped = self
            .table
            .metadata()
            .properties()
            .get("write.manifest.partition-scoped")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

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

        // Process added entries — always write new manifests (FastAppend).
        // Manifest consolidation happens post-commit in merge_manifests_if_needed().
        if !self.added_data_files.is_empty() {
            let added_manifests = self.write_added_manifests().await?;
            manifest_files.extend(added_manifests);
        }

        // Process added delete files.
        if !self.added_delete_files.is_empty() {
            let delete_manifest = self.write_added_delete_manifest().await?;
            manifest_files.push(delete_manifest);
        }

        let manifest_files = manifest_process.process_manifests(self, manifest_files);

        // MergeAppend: consolidate small manifests at commit time.
        // Matches Java SDK's MergingSnapshotProducer.mergeManifests() behavior:
        // - Groups manifests by partition spec
        // - Merges manifests below target size into target-sized bins
        // - Only triggers when total count exceeds min-count-to-merge
        // - Entries in merged manifests change status from Added to Existing
        //
        // This eliminates the need for external manifest rewriting (Tessellate)
        // and keeps the manifest count bounded regardless of commit frequency.
        let manifest_files = self.merge_manifests_if_needed(manifest_files).await?;

        Ok(manifest_files)
    }

    /// Consolidate small manifests into target-sized manifests at commit time.
    ///
    /// Algorithm (matching Java's MergingSnapshotProducer):
    /// 1. Check if total manifest count exceeds `commit.manifest.min-count-to-merge`
    /// 2. Group manifests by partition_spec_id
    /// 3. Within each group, sort by manifest_length (smallest first)
    /// 4. Bin-pack small manifests into bins of `commit.manifest.target-size-bytes`
    /// 5. For each bin with >1 manifest: read entries, write merged manifest
    /// 6. Entries change status: Added → Existing (they're no longer "new")
    ///
    /// Cost: O(small_manifests) S3 reads + O(bins) S3 writes per merge cycle.
    /// Amortized per commit: ~9ms (merge triggers every ~8 commits).
    async fn merge_manifests_if_needed(
        &mut self,
        manifests: Vec<ManifestFile>,
    ) -> Result<Vec<ManifestFile>> {
        let merge_enabled = self
            .table
            .metadata()
            .properties()
            .get("commit.manifest-merge.enabled")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        if !merge_enabled {
            return Ok(manifests);
        }

        let min_count: usize = self
            .table
            .metadata()
            .properties()
            .get("commit.manifest.min-count-to-merge")
            .and_then(|v| v.parse().ok())
            .unwrap_or(100);

        let target_size: i64 = self
            .table
            .metadata()
            .properties()
            .get("commit.manifest.target-size-bytes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8 * 1024 * 1024); // 8MB default, same as Java

        if manifests.len() <= min_count {
            return Ok(manifests);
        }

        // Separate manifests into mergeable (small, data) and keep-as-is
        let mut to_keep: Vec<ManifestFile> = Vec::new();
        let mut to_merge: Vec<ManifestFile> = Vec::new();

        for mf in manifests {
            if mf.content == ManifestContentType::Deletes {
                // Don't merge delete manifests — they have different semantics
                to_keep.push(mf);
            } else if mf.manifest_length >= target_size {
                // Already at or above target size — don't touch
                to_keep.push(mf);
            } else {
                to_merge.push(mf);
            }
        }

        if to_merge.len() <= 1 {
            // Nothing to merge — 0 or 1 small manifests
            to_keep.extend(to_merge);
            return Ok(to_keep);
        }

        // Sort by size (smallest first) for optimal bin packing
        to_merge.sort_by_key(|mf| mf.manifest_length);

        // Bin-pack: group small manifests into bins of ~target_size
        let mut bins: Vec<Vec<ManifestFile>> = Vec::new();
        let mut current_bin: Vec<ManifestFile> = Vec::new();
        let mut current_bin_size: i64 = 0;

        for mf in to_merge {
            if current_bin_size + mf.manifest_length > target_size && !current_bin.is_empty() {
                bins.push(std::mem::take(&mut current_bin));
                current_bin_size = 0;
            }
            current_bin_size += mf.manifest_length;
            current_bin.push(mf);
        }
        if !current_bin.is_empty() {
            bins.push(current_bin);
        }

        // Merge each bin with >1 manifest into a single manifest
        let total_small = bins.iter().map(|b| b.len()).sum::<usize>();
        let mut merged_manifests: Vec<ManifestFile> = Vec::with_capacity(bins.len());

        for bin in bins {
            if bin.len() == 1 {
                // Single manifest in bin — keep as-is
                merged_manifests.push(bin.into_iter().next().unwrap());
                continue;
            }

            // Read all manifests in this bin, collect their entries
            let mut all_entries: Vec<ManifestEntry> = Vec::new();
            for mf in &bin {
                let manifest = mf.load_manifest(self.table.file_io()).await?;
                for entry_ref in manifest.entries() {
                    let mut entry = entry_ref.as_ref().clone();
                    // Change status: Added → Existing (entries are no longer new
                    // after being merged into a consolidated manifest)
                    if entry.status == crate::spec::ManifestStatus::Added {
                        entry.status = crate::spec::ManifestStatus::Existing;
                        // Existing entries must have sequence numbers
                        if entry.sequence_number.is_none() {
                            entry.sequence_number = Some(mf.sequence_number);
                        }
                        if entry.file_sequence_number.is_none() {
                            entry.file_sequence_number = Some(mf.sequence_number);
                        }
                    }
                    all_entries.push(entry);
                }
            }

            // Write merged manifest
            let mut writer = self.new_manifest_writer(ManifestContentType::Data)?;
            for entry in all_entries {
                writer.add_existing_file(
                    entry.data_file,
                    entry.snapshot_id.unwrap_or(0),
                    entry.sequence_number.unwrap_or(0),
                    entry.file_sequence_number,
                )?;
            }

            let merged = if self.use_parquet_manifests() {
                writer.write_manifest_file_parquet().await?
            } else {
                writer.write_manifest_file().await?
            };

            log::info!(
                "manifest merge: {} manifests → 1 ({} entries, {} bytes)",
                bin.len(),
                merged.added_files_count.unwrap_or(0) + merged.existing_files_count.unwrap_or(0),
                merged.manifest_length,
            );

            merged_manifests.push(merged);
        }

        log::info!(
            "manifest merge complete: {} small manifests → {} bins ({} kept as-is)",
            total_small,
            merged_manifests.len(),
            to_keep.len(),
        );

        to_keep.extend(merged_manifests);
        Ok(to_keep)
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

        // The previous snapshot is the current snapshot in the metadata —
        // self.snapshot_id is the NEW snapshot being created (not yet in metadata).
        let previous_snapshot = table_metadata.current_snapshot();

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
        // V4 uses root manifest (single-file commit) instead of manifest list.
        // Dispatch goes through `effective_format_version()` so this branch
        // also fires for tables that declare V3 to a strict catalog (e.g.
        // Lakekeeper pre-V4) but carry the `e6.actual-format-version=4`
        // property -- see `crate::table::E6_ACTUAL_FORMAT_VERSION_KEY`.
        if self.table.effective_format_version() == FormatVersion::V4 {
            return self.commit_v4(snapshot_produce_operation, process).await;
        }

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
            FormatVersion::V3 | FormatVersion::V4 => ManifestListWriter::v3(
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

    /// V4 commit path: writes a single root manifest instead of manifest files + manifest list.
    ///
    /// For append operations, new data files are inlined directly in the root manifest.
    /// Existing manifest references from the current snapshot are carried forward unchanged.
    /// This reduces commit overhead from 3+ S3 PUTs to 1 PUT + 1 CAS.
    async fn commit_v4<OP: SnapshotProduceOperation, MP: ManifestProcess>(
        &mut self,
        snapshot_produce_operation: OP,
        _process: MP,
    ) -> Result<ActionCommit> {
        let next_seq_num = self.table.metadata().next_sequence_number();

        // Generate summary before draining added_data_files
        let mut summary = self.summary(&snapshot_produce_operation).map_err(|err| {
            Error::new(ErrorKind::Unexpected, "Failed to create snapshot summary.").with_source(err)
        })?;

        // Load existing root manifest entries from current snapshot (or use cache).
        //
        // Cache validation: the cached entries carry a snapshot_id from when they were
        // built. If the table's current_snapshot_id has moved (concurrent commit won
        // the CAS race and our retry refreshed the table metadata), the cache is stale
        // and must be discarded. This prevents silently dropping concurrent writes.
        let cached = self.cached_root_entries.take();
        let current_snapshot_id = self.table.metadata().current_snapshot_id();
        let cache_valid = cached.as_ref().map_or(false, |(cached_sid, _)| {
            // Cache carries the snapshot_id it was built for. If the table has
            // moved to a different snapshot (concurrent commit won CAS race),
            // the cache is stale and must be discarded.
            *cached_sid == current_snapshot_id
        });
        // Load the previous entries AND the previous bucket-index pointer. The
        // pointer MUST be carried forward: the hot commit only rewrites the live
        // tier, so dropping it here would orphan the cold bucket-index (tiered
        // layout). Tuple: (entries, carried bucket_index_path).
        let (mut entries, carried_bucket_index_path, current_chain_depth): (
            Vec<RootManifestEntry>,
            Option<String>,
            u32,
        ) = if let Some((_, cached)) = cached.filter(|_| cache_valid) {
                // Cache path (laminar never populates the cache). The cache does
                // not carry the pointer, so recover it from the current
                // snapshot's root metadata rather than risk orphaning the cold
                // tier.
                let (path, depth) = match self.table.metadata().current_snapshot() {
                    Some(s) => {
                        let b = self.table.file_io().new_input(s.manifest_list())?.read().await?;
                        read_root_manifest(b)
                            .ok()
                            .map(|(m, _)| (m.bucket_index_path, m.chain_depth))
                            .unwrap_or((None, 0))
                    }
                    None => (None, 0),
                };
                (cached, path, depth)
            } else if let Some(current_snapshot) =
                self.table.metadata().current_snapshot()
            {
                let manifest_list_path = current_snapshot.manifest_list();
                let bytes = self
                    .table
                    .file_io()
                    .new_input(manifest_list_path)?
                    .read()
                    .await?;

                // Try reading as root manifest; fall back to manifest list for upgrade path
                match read_root_manifest(bytes.clone()) {
                    Ok((prev_meta, existing_entries)) => {
                        (existing_entries, prev_meta.bucket_index_path, prev_meta.chain_depth)
                    }
                    Err(_) => {
                        // Upgrading from V3: convert manifest list entries to manifest refs
                        let manifest_list = crate::spec::ManifestList::parse_with_version(
                            &bytes,
                            FormatVersion::V3,
                        )?;
                        let entries = manifest_list
                            .entries()
                            .iter()
                            .map(|mf| RootManifestEntry::ManifestRef {
                                manifest_file: mf.clone(),
                                mdv: None,
                            })
                            .collect();
                        (entries, None, 0)
                    }
                }
            } else {
                (vec![], None, 0)
            };

        // --- Incremental (log-structured) root ---------------------------------
        // When enabled, a plain append commit writes a DELTA: a root carrying only
        // THIS commit's new refs plus a `prev_root_path` pointer back to the prior
        // root. The full live set is reconstructed by walking the chain on read, so
        // the per-commit root write is O(this commit), not O(#live refs). The chain
        // is collapsed back to a base when it gets too deep, when there are removals
        // (which must rewrite carried refs), or by any lifecycle op. Gated → default
        // off, so flat/tiered behavior is unchanged.
        let incremental = self
            .table
            .metadata()
            .properties()
            .get("root-manifest.incremental")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        const MAX_CHAIN: u32 = 64;
        let current_root_path: Option<String> = self
            .table
            .metadata()
            .current_snapshot()
            .map(|s| s.manifest_list().to_string());
        let do_delta = incremental
            && current_root_path.is_some()
            && self.removed_data_files.is_empty()
            && current_chain_depth < MAX_CHAIN;
        if do_delta {
            // Delta carries only this commit's new refs; discard the carried head.
            entries.clear();
        } else if incremental && current_chain_depth > 0 {
            // Collapse to a base: the head read above gave only the head delta's
            // entries, so reconstruct the full live set from the chain.
            if let Some(p) = &current_root_path {
                let (_, full) = reconstruct_root(self.table.file_io(), p).await?;
                entries = full;
            }
        }

        // Handle file removals (compaction / overwrite operations)
        let removed_data_files = std::mem::take(&mut self.removed_data_files);
        if !removed_data_files.is_empty() {
            let paths_to_remove: HashSet<String> = removed_data_files
                .iter()
                .map(|df| df.file_path.clone())
                .collect();

            // Remove matching inline entries directly
            entries.retain(|entry| match entry {
                RootManifestEntry::Inline(me) => !paths_to_remove.contains(&me.data_file.file_path),
                RootManifestEntry::ManifestRef { .. } => true,
            });

            // For manifest refs: build MDVs by scanning child manifests for removed paths.
            // Load each referenced manifest, find row indices of files being removed,
            // and create/merge MDV bitmaps.

            // If we have an index, only scan manifests that contain removed files
            let manifests_to_scan: HashSet<String> = if let Some(ref index) = self.file_to_manifest_index {
                paths_to_remove.iter()
                    .filter_map(|p| index.get(p).cloned())
                    .collect()
            } else {
                // No index — scan all manifest refs (existing behavior)
                entries.iter()
                    .filter_map(|e| match e {
                        RootManifestEntry::ManifestRef { manifest_file, .. } => Some(manifest_file.manifest_path.clone()),
                        _ => None,
                    })
                    .collect()
            };

            let mut total_mdv_deleted_files: u64 = 0;
            let mut total_mdv_deleted_rows: u64 = 0;

            for entry in entries.iter_mut() {
                if let RootManifestEntry::ManifestRef {
                    manifest_file,
                    mdv,
                } = entry
                {
                    if !manifests_to_scan.contains(&manifest_file.manifest_path) {
                        continue;  // Skip manifests not in the index
                    }

                    let manifest = manifest_file
                        .load_manifest(self.table.file_io())
                        .await?;

                    let mut new_mdv = match mdv.as_ref() {
                        Some(existing_bytes) => {
                            use crate::spec::root_manifest::ManifestDeleteVector;
                            ManifestDeleteVector::deserialize(existing_bytes)?
                        }
                        None => {
                            use crate::spec::root_manifest::ManifestDeleteVector;
                            ManifestDeleteVector::new()
                        }
                    };

                    let mut found_any = false;
                    for (idx, manifest_entry) in manifest.entries().iter().enumerate() {
                        if paths_to_remove.contains(&manifest_entry.data_file.file_path) {
                            new_mdv.mark_deleted(idx as u32);
                            found_any = true;
                            total_mdv_deleted_files += 1;
                            total_mdv_deleted_rows += manifest_entry.data_file.record_count as u64;
                        }
                    }

                    if found_any {
                        *mdv = Some(new_mdv.serialize()?);
                    }
                }
            }

            if total_mdv_deleted_files > 0 {
                summary.additional_properties.insert(
                    "deleted-data-files".to_string(),
                    total_mdv_deleted_files.to_string(),
                );
                summary.additional_properties.insert(
                    "deleted-records".to_string(),
                    total_mdv_deleted_rows.to_string(),
                );
            }
        }

        // Add new data files as inline entries
        let added_data_files = std::mem::take(&mut self.added_data_files);
        for df in added_data_files {
            entries.push(RootManifestEntry::Inline(ManifestEntry {
                status: ManifestStatus::Added,
                snapshot_id: Some(self.snapshot_id),
                sequence_number: Some(next_seq_num),
                file_sequence_number: Some(next_seq_num),
                data_file: df,
            }));
        }

        // Add new delete files as inline entries
        let added_delete_files = std::mem::take(&mut self.added_delete_files);
        for df in added_delete_files {
            entries.push(RootManifestEntry::Inline(ManifestEntry {
                status: ManifestStatus::Added,
                snapshot_id: Some(self.snapshot_id),
                sequence_number: Some(next_seq_num),
                file_sequence_number: Some(next_seq_num),
                data_file: df,
            }));
        }

        // Adaptive inline→child flush: when inline count exceeds threshold,
        // flush inline entries to a child manifest and replace with a manifest ref.
        // This eliminates the need for separate table maintenance.
        let partition_type = self
            .table
            .metadata()
            .default_partition_spec()
            .partition_type(self.table.metadata().current_schema())?;

        let inline_threshold = self
            .table
            .metadata()
            .properties()
            .get("root-manifest.inline-threshold")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(100);

        let inline_count = entries
            .iter()
            .filter(|e| matches!(e, RootManifestEntry::Inline(_)))
            .count();

        // Tiered layout: APPEND-ONLY live nodes (V4 "one-file commit" principle).
        // Each commit flushes its just-added inline files into a fresh child
        // manifest ("node") and keeps only the node *reference* in the root, so
        // the per-commit root rewrite is O(#live nodes), never O(#live files) —
        // metadata growth proportional to the operation, not the table. Closed
        // live nodes are relocated to the cold bucket-index by `graduate_buckets`;
        // small live nodes are merged by compaction. (Non-tiered tables keep the
        // legacy threshold-based flush.)
        let tiered = self
            .table
            .metadata()
            .properties()
            .get("tiered-metadata.enabled")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        if (tiered && inline_count > 0) || (!tiered && inline_count > inline_threshold) {
            // Split inline entries by content type (data vs delete)
            let mut data_entries: Vec<ManifestEntry> = Vec::new();
            let mut delete_entries: Vec<ManifestEntry> = Vec::new();
            entries.retain(|e| match e {
                RootManifestEntry::Inline(me) => {
                    match me.data_file.content {
                        crate::spec::DataContentType::Data => data_entries.push(me.clone()),
                        _ => delete_entries.push(me.clone()),
                    }
                    false // remove from entries
                }
                RootManifestEntry::ManifestRef { .. } => true, // keep
            });

            // Flush data entries to a child data manifest
            if !data_entries.is_empty() {
                let path = format!(
                    "{}/{}/{}-m{}.parquet",
                    self.table.metadata().location(),
                    META_ROOT_PATH,
                    self.commit_uuid,
                    self.manifest_counter.next().unwrap_or(0),
                );
                let mut writer = ManifestWriterBuilder::new(
                    self.table.file_io().new_output(&path)?,
                    Some(self.snapshot_id),
                    self.key_metadata.clone(),
                    self.table.metadata().current_schema().clone(),
                    self.table.metadata().default_partition_spec().as_ref().clone(),
                )
                .build_v3_data();

                for entry in &data_entries {
                    writer.add_entry(entry.clone())?;
                }
                // V4 child manifests are Parquet -- the path was templated
                // with the `.parquet` extension just above, so the underlying
                // bytes MUST be Parquet to match. Using the Avro writer
                // (`write_manifest_file`) here previously produced
                // .parquet-named files with Avro magic, which any reader
                // dispatched by extension (tessellate's live-set scan, the
                // iceberg-rust scan path) fails on with
                // "Invalid Parquet file. Corrupt footer".
                // `RebalanceRootManifestAction` already takes the Parquet
                // path; this lines commit_v4 up with that convention.
                entries.push(RootManifestEntry::ManifestRef {
                    manifest_file: writer.write_manifest_file_parquet().await?,
                    mdv: None,
                });
            }

            // Flush delete entries to a separate child delete manifest
            if !delete_entries.is_empty() {
                let path = format!(
                    "{}/{}/{}-m{}.parquet",
                    self.table.metadata().location(),
                    META_ROOT_PATH,
                    self.commit_uuid,
                    self.manifest_counter.next().unwrap_or(0),
                );
                let mut writer = ManifestWriterBuilder::new(
                    self.table.file_io().new_output(&path)?,
                    Some(self.snapshot_id),
                    self.key_metadata.clone(),
                    self.table.metadata().current_schema().clone(),
                    self.table.metadata().default_partition_spec().as_ref().clone(),
                )
                .build_v3_deletes();

                for entry in &delete_entries {
                    writer.add_entry(entry.clone())?;
                }
                // See the matching note on the data-entry flush above:
                // .parquet path => Parquet bytes. Avro here silently writes
                // bytes that fail to read.
                entries.push(RootManifestEntry::ManifestRef {
                    manifest_file: writer.write_manifest_file_parquet().await?,
                    mdv: None,
                });
            }
        }

        // Merge small manifest refs to keep ref count bounded.
        // Only merge refs without MDVs — MDV bitmaps reference row indices
        // in the original manifest, so merging would invalidate them.
        // Skipped on a delta write: a delta holds only this commit's new ref(s);
        // merging carried refs is a base/collapse concern.
        if !do_delta {
            let mergeable_refs: Vec<ManifestFile> = entries
                .iter()
                .filter_map(|e| match e {
                    RootManifestEntry::ManifestRef { manifest_file, mdv } if mdv.is_none() => {
                        Some(manifest_file.clone())
                    }
                    _ => None,
                })
                .collect();

            let merged = self.merge_manifests_if_needed(mergeable_refs).await?;

            // Replace mergeable refs with merged result
            entries.retain(|e| match e {
                RootManifestEntry::ManifestRef { mdv, .. } => mdv.is_some(),
                RootManifestEntry::Inline(_) => true,
            });
            for mf in merged {
                entries.push(RootManifestEntry::ManifestRef {
                    manifest_file: mf,
                    mdv: None,
                });
            }
        }

        // Build root manifest metadata
        let rm_metadata = RootManifestMetadata {
            schema: self.table.metadata().current_schema().clone(),
            schema_id: self.table.metadata().current_schema_id(),
            partition_spec: self.table.metadata().default_partition_spec().clone(),
            format_version: FormatVersion::V4,
            snapshot_id: self.snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: self.table.metadata().current_snapshot_id(),
            // Carry the cold bucket-index pointer forward unchanged — the hot
            // commit only rewrites the live tier (see entry-load above). Without
            // this, the first commit after a bucket-close would orphan the cold
            // tier.
            bucket_index_path: carried_bucket_index_path,
            // Delta → point back at the prior root and bump chain depth; base →
            // None / 0. The reader walks `prev_root_path` to reconstruct.
            prev_root_path: if do_delta { current_root_path.clone() } else { None },
            chain_depth: if do_delta { current_chain_depth + 1 } else { 0 },
        };

        // Write root manifest as single Parquet file
        let root_manifest_path = format!(
            "{}/{}/root-{}-{}.parquet",
            self.table.metadata().location(),
            META_ROOT_PATH,
            self.snapshot_id,
            self.commit_uuid,
        );

        let bytes = write_root_manifest(&entries, &rm_metadata, &partition_type)?;
        self.table
            .file_io()
            .new_output(&root_manifest_path)?
            .write(bytes.into())
            .await?;

        // Compute row lineage for V4 (same as V3 path)
        let first_row_id = self.table.metadata().next_row_id();
        let added_rows: u64 = entries
            .iter()
            .filter_map(|e| match e {
                RootManifestEntry::Inline(me) if me.snapshot_id == Some(self.snapshot_id) => {
                    Some(me.data_file.record_count)
                }
                _ => None,
            })
            .sum();

        // Create snapshot — reuses manifest_list field for root manifest path
        let commit_ts = chrono::Utc::now().timestamp_millis();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(root_manifest_path.clone())
            .with_snapshot_id(self.snapshot_id)
            .with_parent_snapshot_id(self.table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(self.table.metadata().current_schema_id())
            .with_timestamp_ms(commit_ts)
            .with_row_range(first_row_id, added_rows)
            .build();

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

        Ok(ActionCommit::new(updates, requirements)
            .with_manifest_paths(vec![root_manifest_path])
            .with_root_manifest_entries(Some(self.snapshot_id), entries))
    }
}

#[cfg(test)]
mod test_v4_commit {
    use crate::catalog::memory::tests::new_memory_catalog;
    use crate::catalog::{Catalog, NamespaceIdent, TableCreation};
    use crate::spec::{
        DataContentType, DataFile, DataFileFormat, FormatVersion, NestedField, PrimitiveType,
        Schema, Struct, Type,
    };
    use crate::transaction::Transaction;
    use crate::transaction::action::ApplyTransactionAction;
    use futures::TryStreamExt;
    use std::collections::{HashMap, HashSet};

    fn test_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            ])
            .build()
            .unwrap()
    }

    fn test_data_file(path: &str) -> DataFile {
        DataFile {
            content: DataContentType::Data,
            file_path: path.to_string(),
            file_format: DataFileFormat::Parquet,
            partition: Struct::empty(),
            record_count: 100,
            file_size_in_bytes: 1024,
            column_sizes: HashMap::new(),
            value_counts: HashMap::new(),
            null_value_counts: HashMap::new(),
            nan_value_counts: HashMap::new(),
            lower_bounds: HashMap::new(),
            upper_bounds: HashMap::new(),
            key_metadata: None,
            split_offsets: None,
            equality_ids: None,
            sort_order_id: None,
            partition_spec_id: 0,
            first_row_id: None,
            content_offset: None,
            content_size_in_bytes: None,
            referenced_data_file: None,
        }
    }

    #[tokio::test]
    async fn test_v4_fast_append_writes_root_manifest() {
        let catalog = new_memory_catalog().await;
        let ns = NamespaceIdent::new("test_ns".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .unwrap();

        let table = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("v4table".to_string())
                    .schema(test_schema())
                    .format_version(FormatVersion::V4)
                    .build(),
            )
            .await
            .unwrap();

        assert_eq!(table.metadata().format_version(), FormatVersion::V4);

        let files = vec![
            test_data_file("s3://bucket/data/a.parquet"),
            test_data_file("s3://bucket/data/b.parquet"),
        ];

        let tx = Transaction::new(&table);
        let tx = tx.fast_append().add_data_files(files).apply(tx).unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Verify snapshot was created
        let snapshot = table
            .metadata()
            .current_snapshot()
            .expect("should have snapshot");
        // Root manifest path should contain "root-"
        assert!(
            snapshot.manifest_list().contains("root-"),
            "manifest path should be a root manifest: {}",
            snapshot.manifest_list()
        );
    }

    #[tokio::test]
    async fn test_v4_second_append_accumulates_inlines() {
        let catalog = new_memory_catalog().await;
        let ns = NamespaceIdent::new("test_ns2".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .unwrap();

        let table = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("v4table2".to_string())
                    .schema(test_schema())
                    .format_version(FormatVersion::V4)
                    .build(),
            )
            .await
            .unwrap();

        // First append
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![test_data_file("s3://bucket/1.parquet")])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // Verify table is still V4 after first commit
        assert_eq!(table.metadata().format_version(), FormatVersion::V4,
            "table format version should be V4 after first commit");

        let snap1 = table.metadata().current_snapshot().unwrap();
        let snap1_id = snap1.snapshot_id();

        // Second append
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![test_data_file("s3://bucket/2.parquet")])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        let snap2 = table.metadata().current_snapshot().unwrap();
        assert_ne!(snap1_id, snap2.snapshot_id());
        assert!(snap2.manifest_list().contains("root-"));
    }

    #[tokio::test]
    async fn test_v4_scan_returns_inline_entries() {
        let catalog = new_memory_catalog().await;
        let ns = NamespaceIdent::new("test_scan".into());
        catalog
            .create_namespace(&ns, HashMap::new())
            .await
            .unwrap();

        let table = catalog
            .create_table(
                &ns,
                TableCreation::builder()
                    .name("v4scan".to_string())
                    .schema(test_schema())
                    .format_version(FormatVersion::V4)
                    .build(),
            )
            .await
            .unwrap();

        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .with_check_duplicate(false)
            .add_data_files(vec![
                test_data_file("s3://bucket/data/a.parquet"),
                test_data_file("s3://bucket/data/b.parquet"),
            ])
            .apply(tx)
            .unwrap();
        let table = tx.commit(&catalog).await.unwrap();

        // plan_files should return 2 tasks from inline entries
        let tasks: Vec<_> = table
            .scan()
            .build()
            .unwrap()
            .plan_files()
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(tasks.len(), 2);
        let paths: HashSet<&str> = tasks.iter().map(|t| t.data_file_path.as_str()).collect();
        assert!(paths.contains("s3://bucket/data/a.parquet"));
        assert!(paths.contains("s3://bucket/data/b.parquet"));
    }
}
