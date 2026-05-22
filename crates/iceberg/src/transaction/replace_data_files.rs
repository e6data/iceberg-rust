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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::{
    DataFile, DataFileFormat, Datum, FieldSummary, FormatVersion, ManifestEntry, ManifestFile,
    ManifestStatus, ManifestWriterBuilder, Operation, StructType,
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
    /// Cached manifest result from the first commit attempt.
    /// On retry, reuse this to skip re-reading all manifests from S3.
    cached_manifests: Arc<Mutex<Option<Vec<ManifestFile>>>>,
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
            cached_manifests: Arc::new(Mutex::new(None)),
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

    /// Hint manifest paths that the caller believes are fully covered by
    /// `files_to_delete`. **Currently ignored.**
    ///
    /// Originally an optimization to skip loading manifests we knew would be
    /// fully dropped. Removed because it was unsafe whenever a manifest
    /// contained any file outside `files_to_delete` (which is the common case
    /// when upstream batches many partitions per commit). The slow path
    /// detects fully-deletable manifests on its own without risking the
    /// silent loss of co-resident files.
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
            data_files_to_delete: self.files_to_delete.clone(),
            commit_uuid: self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            key_metadata: self.key_metadata.clone(),
            cached_manifests: Arc::clone(&self.cached_manifests),
        };

        snapshot_producer
            .commit(operation, DefaultManifestProcess)
            .await
    }
}

/// Build a `Vec<HashSet<Vec<u8>>>` where index `i` holds the set of
/// byte-encoded partition values at position `i` across all files to delete.
///
/// The byte encoding matches the one used by the manifest writer when it
/// creates `FieldSummary` bounds (`Datum::to_bytes`), so a direct byte
/// comparison against `FieldSummary::lower_bound` / `upper_bound` is valid.
fn build_target_partition_bytes(
    files_to_delete: &[DataFile],
    partition_type: &StructType,
) -> Vec<HashSet<Vec<u8>>> {
    let num_fields = partition_type.fields().len();
    let mut result: Vec<HashSet<Vec<u8>>> = vec![HashSet::new(); num_fields];

    let field_types: Vec<_> = partition_type
        .fields()
        .iter()
        .filter_map(|f| f.field_type.as_primitive_type().cloned())
        .collect();

    // If the partition spec has non-primitive fields we can't encode them,
    // return empty sets (which will disable filtering for those positions).
    if field_types.len() != num_fields {
        return result;
    }

    for file in files_to_delete {
        // Skip files whose partition field count doesn't match this spec.
        if file.partition().fields().len() != num_fields {
            continue;
        }
        for (i, field_val) in file.partition().iter().enumerate() {
            if let Some(literal) = field_val {
                if let Some(prim) = literal.as_primitive_literal() {
                    let datum = Datum::new(field_types[i].clone(), prim);
                    if let Ok(bytes) = datum.to_bytes() {
                        result[i].insert(bytes.to_vec());
                    }
                }
            }
        }
    }

    result
}

/// Returns `false` when the manifest's partition summaries prove that it
/// cannot contain any of the target files (i.e., can be safely skipped).
///
/// Only filters on partition fields where both `lower_bound` and
/// `upper_bound` are present and equal (single-value summary). When a
/// range or missing bounds are encountered the field is conservatively
/// treated as a possible match.
fn manifest_could_contain_target_files(
    partitions: &[FieldSummary],
    target_partition_bytes: &[HashSet<Vec<u8>>],
) -> bool {
    for (i, summary) in partitions.iter().enumerate() {
        let targets = match target_partition_bytes.get(i) {
            Some(t) if !t.is_empty() => t,
            _ => continue, // no targets for this field position — skip
        };

        match (&summary.lower_bound, &summary.upper_bound) {
            (Some(lower), Some(upper)) if lower == upper => {
                // Single value in this partition field across the whole manifest.
                // If that value is not among the target partition bytes we can
                // rule out this manifest entirely.
                let bound_bytes: &[u8] = lower.as_ref();
                if !targets.iter().any(|t| t.as_slice() == bound_bytes) {
                    return false;
                }
            }
            _ => {
                // Range or missing bounds — can't safely filter, keep manifest.
                continue;
            }
        }
    }

    true // all fields passed or couldn't be filtered
}

struct ReplaceOperation {
    files_to_delete: HashSet<String>,
    /// Full DataFile objects for computing partition-level filters.
    data_files_to_delete: Vec<DataFile>,
    commit_uuid: Uuid,
    key_metadata: Option<Vec<u8>>,
    /// Shared cache for manifest computation results. Populated on the first
    /// commit attempt and reused on retries to avoid re-reading manifests
    /// from S3.
    cached_manifests: Arc<Mutex<Option<Vec<ManifestFile>>>>,
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
        // On retry, reuse the cached manifest result to avoid re-reading all
        // manifests from S3. The manifest content hasn't changed between
        // retries — only the snapshot ref may have advanced.
        //
        // Rewritten manifests in the cache carry the OLD snapshot's ID in
        // `added_snapshot_id`. The ManifestListWriter requires unassigned-
        // sequence manifests to match the CURRENT snapshot ID. Fix up the
        // IDs so the retry's ManifestListWriter accepts them.
        {
            let cache = self.cached_manifests.lock().unwrap();
            if let Some(ref cached) = *cache {
                let current_snap_id = snapshot_produce.snapshot_id();
                let fixed: Vec<ManifestFile> = cached
                    .iter()
                    .map(|mf| {
                        let mut m = mf.clone();
                        if m.sequence_number == crate::spec::UNASSIGNED_SEQUENCE_NUMBER {
                            m.added_snapshot_id = current_snap_id;
                        }
                        m
                    })
                    .collect();
                return Ok(fixed);
            }
        }

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

        // Pre-compute byte-encoded partition values from the files being
        // deleted, keyed by partition spec id. Used below to skip manifests
        // whose FieldSummary bounds prove they cannot contain any target file.
        let metadata = snapshot_produce.table.metadata();
        let schema = metadata.current_schema();
        let mut partition_bytes_by_spec: HashMap<i32, Vec<HashSet<Vec<u8>>>> = HashMap::new();

        for manifest_entry in manifest_list.entries() {
            // NOTE: The previous fast-path here used `delete_manifests` to drop
            // an entire manifest_entry without inspecting its contents. This is
            // unsafe whenever a single manifest file holds entries for files
            // that are NOT in `files_to_delete` — which is the common case for
            // upstream Laminar's FastAppend, where every commit cycle batches
            // ~14 data files across many partitions into a single manifest.
            // Compaction targets one partition's slice but the manifest path
            // is shared with files for other partitions/tenants; dropping the
            // whole manifest then loses all of them from the snapshot.
            //
            // Always go through the slow path below — it loads the manifest
            // and correctly drops only the files in `files_to_delete`, while
            // preserving the rest as Existing entries. Cost: one extra S3
            // GET per manifest in the snapshot (which is what the slow path
            // already does anyway). The `delete_manifests` field is retained
            // on the action for backwards-compat but is no longer consulted.

            // Skip manifests with no active files
            if !manifest_entry.has_added_files() && !manifest_entry.has_existing_files() {
                continue;
            }

            // Short-circuit: if all files to delete have been found,
            // keep remaining manifests as-is without loading them from S3.
            if remaining_to_delete.is_empty() {
                result_manifests.push(manifest_entry.clone());
                continue;
            }

            // Skip manifests whose partition summaries prove they cannot
            // contain any of the target files. This avoids an S3 GET for
            // manifests that are clearly for a different partition value
            // (common after compaction when lower_bound == upper_bound).
            if let Some(ref summaries) = manifest_entry.partitions {
                let spec_id = manifest_entry.partition_spec_id;
                let target_bytes = partition_bytes_by_spec.entry(spec_id).or_insert_with(|| {
                    metadata
                        .partition_spec_by_id(spec_id)
                        .and_then(|spec| spec.partition_type(schema).ok())
                        .map(|pt| build_target_partition_bytes(&self.data_files_to_delete, &pt))
                        .unwrap_or_default()
                });

                if !target_bytes.is_empty()
                    && !manifest_could_contain_target_files(summaries, target_bytes)
                {
                    result_manifests.push(manifest_entry.clone());
                    continue;
                }
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
            let use_parquet = {
                let prop = snapshot_produce
                    .table
                    .metadata()
                    .properties()
                    .get("write.parquet.metadata-codec");
                match prop.map(|v| v.as_str()) {
                    Some(v) if v.eq_ignore_ascii_case("avro") => false,
                    _ => matches!(
                        snapshot_produce.table.metadata().format_version(),
                        FormatVersion::V2 | FormatVersion::V3
                    ),
                }
            };
            let ext = if use_parquet { "parquet" } else { "avro" };
            let new_manifest_path = format!(
                "{}/metadata/{}-m-rewrite-{}.{}",
                snapshot_produce.table.metadata().location(),
                self.commit_uuid,
                counter,
                ext
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

            let new_manifest_file = if use_parquet {
                writer.write_manifest_file_parquet().await?
            } else {
                writer.write_manifest_file().await?
            };
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

        // Cache the result for future retries.
        {
            let mut cache = self.cached_manifests.lock().unwrap();
            *cache = Some(result_manifests.clone());
        }

        Ok(result_manifests)
    }
}
