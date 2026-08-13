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
    /// Arc-wrapped for cheap sharing across the commit path (see 2026-07-31
    /// heap profile: `DataFile::clone` from `spawn_root_manifest_tasks` was
    /// holding ~1.15 GB across in-flight commits; Vec::clone was deep-cloning
    /// every DataFile including its 6 internal HashMaps). Arc lets us hand
    /// the same underlying Vec to the SnapshotProducer's removed_data_files
    /// slot (via `with_removed_data_files`) and to the ReplaceOperation's
    /// filter cache without a Vec::clone. `with_removed_data_files` still
    /// materialises owned Vec for now — Step B (Arc<Vec> through
    /// SnapshotProducer) collapses that last clone in a follow-up.
    files_to_delete: Arc<Vec<DataFile>>,
    files_to_add: Vec<DataFile>,
    delete_manifests: Vec<String>,
    commit_uuid: Option<Uuid>,
    key_metadata: Option<Vec<u8>>,
    snapshot_properties: HashMap<String, String>,
    validate_from_snapshot_id: Option<i64>,
    data_sequence_number: Option<i64>,
    added_delete_files: Vec<DataFile>,
    /// Cached manifest result from the first commit attempt, tagged with
    /// the snapshot_id it was built from. On retry, reuse only if the
    /// current snapshot matches; otherwise discard and rebuild from the
    /// new snapshot's manifest list.
    cached_manifests: Arc<Mutex<Option<(Option<i64>, Vec<ManifestFile>)>>>,
    /// Caller-provided override for the new snapshot's id. Mirrors
    /// `FastAppendAction.with_snapshot_id`. Set via
    /// [`Self::with_snapshot_id`]; pair with
    /// [`crate::transaction::generate_unique_snapshot_id`] to pre-allocate
    /// an id the caller can also use in `StatisticsFile` entries within
    /// the same transaction (e.g. for compaction carry-forward of
    /// per-snapshot Puffin stats).
    snapshot_id_override: Option<i64>,
    file_to_manifest_index: Option<HashMap<String, String>>,
    cached_root_entries: Option<(Option<i64>, Vec<crate::spec::root_manifest::RootManifestEntry>)>,
}

impl ReplaceDataFilesAction {
    pub(crate) fn new() -> Self {
        Self {
            files_to_delete: Arc::new(Vec::new()),
            files_to_add: Vec::new(),
            delete_manifests: Vec::new(),
            commit_uuid: None,
            key_metadata: None,
            snapshot_properties: HashMap::default(),
            validate_from_snapshot_id: None,
            data_sequence_number: None,
            added_delete_files: Vec::new(),
            cached_manifests: Arc::new(Mutex::new(None)),
            snapshot_id_override: None,
            file_to_manifest_index: None,
            cached_root_entries: None,
        }
    }

    /// Set the data files to delete.
    pub fn delete_files(mut self, files: Vec<DataFile>) -> Self {
        self.files_to_delete = Arc::new(files);
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

    /// Pre-allocate the new snapshot's id, overriding the random id that
    /// `commit()` would otherwise generate. Mirror of
    /// [`super::append::FastAppendAction::with_snapshot_id`]; same use case
    /// (referencing the snapshot_id elsewhere in the same transaction —
    /// most commonly to attach carry-forward `StatisticsFile` entries to
    /// the new snapshot in a compaction commit so per-snapshot Puffin
    /// stats don't orphan).
    ///
    /// Pair with [`crate::transaction::generate_unique_snapshot_id`] to
    /// generate the id before the action is built.
    pub fn with_snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.snapshot_id_override = Some(snapshot_id);
        self
    }

    /// Provide a mapping from data file path to manifest path for targeted MDV scanning.
    /// When set, only manifests known to contain removed files will be loaded during V4 commit.
    pub fn with_file_manifest_index(mut self, index: HashMap<String, String>) -> Self {
        self.file_to_manifest_index = Some(index);
        self
    }

    /// Pass cached root manifest entries to avoid re-reading from S3 on
    /// consecutive commits within the same transaction.
    pub fn with_cached_root_entries(mut self, snapshot_id: Option<i64>, entries: Vec<crate::spec::root_manifest::RootManifestEntry>) -> Self {
        self.cached_root_entries = Some((snapshot_id, entries));
        self
    }
}

#[async_trait]
impl TransactionAction for ReplaceDataFilesAction {
    /// A retry after a commit conflict could re-apply this action's delete-file list
    /// against a table where those files are already gone (or add its merged file a
    /// second time), duplicating or resurrecting data. Fail the transaction fast
    /// instead so the caller re-plans the compaction against the fresh snapshot.
    fn disables_retry(&self) -> bool {
        true
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let mut snapshot_producer = SnapshotProducer::new(
            table,
            self.commit_uuid.unwrap_or_else(Uuid::now_v7),
            self.key_metadata.clone(),
            self.snapshot_properties.clone(),
            self.files_to_add.clone(),
            self.added_delete_files.clone(),
        )
        // Vec::clone here still deep-clones every DataFile. Step B
        // (SnapshotProducer.with_removed_data_files accepts Arc<Vec>) will
        // collapse this into an Arc::clone. Not shipped yet: the change
        // touches ~15 sites in snapshot.rs across added/removed_data_files
        // and added_delete_files and can't be safely rushed.
        .with_removed_data_files((*self.files_to_delete).clone())
        .with_data_sequence_number(self.data_sequence_number);

        if let Some(id) = self.snapshot_id_override {
            snapshot_producer = snapshot_producer.with_snapshot_id(id);
        }
        if let Some((snapshot_id, ref cached)) = self.cached_root_entries {
            snapshot_producer = snapshot_producer.with_cached_root_entries(snapshot_id, cached.clone());
        }
        if let Some(ref index) = self.file_to_manifest_index {
            snapshot_producer = snapshot_producer.with_file_to_manifest_index(index.clone());
        }

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
            // Arc::clone — no deep DataFile copy. Was the largest per-commit
            // deep clone in the 2026-07-31 heap profile (DataFile::clone
            // dominant self-bytes allocator via spawn_root_manifest_tasks).
            data_files_to_delete: Arc::clone(&self.files_to_delete),
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
/// Diagnostic companion to [`manifest_could_contain_target_files`]: is this
/// manifest's summary provably disjoint from the targets on any partition
/// field, using RANGE comparison rather than the single-value equality the
/// pruning function requires?
///
/// Pure measurement — nothing branches on this. It exists to size the prize
/// before changing behaviour: a manifest counted here was loaded from S3 even
/// though its own bounds show it cannot hold a target file. For the
/// observability tables the interesting field is `timestamp_hour`, where a
/// cold-tier leaf sits entirely below the graduation cutoff while every target
/// is at or above it.
///
/// Byte-comparing bounds is sound for the transforms in use here
/// (identity strings, hour-as-int) because their serialised forms are
/// order-preserving. Deliberately conservative: any field it cannot reason
/// about contributes nothing rather than a false "disjoint".
fn summary_disjoint_from_targets(
    partitions: &[FieldSummary],
    target_partition_bytes: &[HashSet<Vec<u8>>],
) -> bool {
    for (i, summary) in partitions.iter().enumerate() {
        let targets = match target_partition_bytes.get(i) {
            Some(t) if !t.is_empty() => t,
            _ => continue,
        };
        let (Some(lower), Some(upper)) = (&summary.lower_bound, &summary.upper_bound) else {
            continue;
        };
        let (lo, hi): (&[u8], &[u8]) = (lower.as_ref(), upper.as_ref());
        // Disjoint iff EVERY target falls outside [lo, hi]. Length-differing
        // encodings are skipped rather than guessed at.
        let all_outside = targets.iter().all(|t| {
            let t: &[u8] = t.as_slice();
            if t.len() != lo.len() || t.len() != hi.len() {
                return false;
            }
            t < lo || t > hi
        });
        if all_outside {
            return true;
        }
    }
    false
}

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
    /// Full DataFile objects for computing partition-level filters. Arc so
    /// this is a cheap handle-share from `ReplaceDataFilesAction` — see the
    /// heap-profile note on that field.
    data_files_to_delete: Arc<Vec<DataFile>>,
    commit_uuid: Uuid,
    key_metadata: Option<Vec<u8>>,
    /// Shared cache for manifest computation results, tagged with the
    /// snapshot_id they were built from. Populated on the first commit
    /// attempt and reused on retries only if the snapshot hasn't changed
    /// (i.e., no concurrent commit occurred). When a concurrent commit
    /// advances the snapshot, the cache is invalidated and manifests are
    /// rebuilt from the new snapshot's manifest list.
    cached_manifests: Arc<Mutex<Option<(Option<i64>, Vec<ManifestFile>)>>>,
}

impl SnapshotProduceOperation for ReplaceOperation {
    fn operation(&self) -> Operation {
        // Replace, NOT Overwrite: per the Iceberg spec, `Replace` is "data/delete
        // files added and removed WITHOUT changing table data" (compaction, format
        // change, relocation) — exactly what ReplaceDataFiles does. `Overwrite` is a
        // logical whole-table overwrite. Returning Overwrite made every compaction
        // hit truncate_table_summary (reset TOTAL_*, deleted-* = prev cumulative).
        Operation::Replace
    }

    // ReplaceDataFiles is a partial compaction, never a full-table truncate, so it
    // inherits the default `truncates_full_table() == false`.

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
        // On retry, reuse the cached manifest result ONLY if the base
        // snapshot hasn't changed (no concurrent commit occurred). When a
        // concurrent commit advances the snapshot, the manifest list is
        // different — reusing stale cached manifests would silently drop
        // the concurrent commit's new/rewritten manifests, causing data loss.
        //
        // When the cache IS valid (same snapshot, e.g., retry due to a
        // transient network error), rewritten manifests carry the OLD
        // snapshot's ID in `added_snapshot_id`. Fix up the IDs so the
        // retry's ManifestListWriter accepts them.
        {
            let cache = self.cached_manifests.lock().unwrap();
            if let Some((cached_snapshot_id, ref cached)) = *cache {
                let current_snapshot_id = snapshot_produce
                    .table
                    .metadata()
                    .current_snapshot()
                    .map(|s| s.snapshot_id());
                if cached_snapshot_id == current_snapshot_id {
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
                // Cache is stale — snapshot changed due to a concurrent
                // commit. Fall through to rebuild from the new snapshot.
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

        // ── Walk instrumentation: measurement only, no behaviour change. ──
        //
        // This loop is the body of the `actions_ms` that dominates tessellate's
        // tick — measured 96.6s across 26 commits on `logs` (p50 3.35s each),
        // while the root it produces is only ~16 KB and the catalog round-trip
        // is ~0 ms. So the cost is here, in walking and loading manifests.
        //
        // The open question these settle: tessellate commits target ONE closed
        // hour and ONE partition, and cold-tier bucket-index leaves hold hours
        // older than the graduation cutoff, so they cannot contain the target
        // files. Yet nothing in this loop is hour-aware —
        // `manifest_could_contain_target_files` prunes only when a field's
        // lower_bound == upper_bound, so a leaf spanning ANY range is loaded no
        // matter how far its hours sit from the target.
        //
        // If `loaded` is dominated by entries whose hour range is disjoint from
        // the target, the fix is hour-RANGE pruning off the leaf summaries (the
        // same LVI Phase 6 and the pre-graduate check already use). If `loaded`
        // turns out small and the time is elsewhere, the hypothesis is wrong
        // and these numbers say so — which is the point of measuring first.
        let walk_start = std::time::Instant::now();
        let mut n_total: usize = 0;
        let mut n_skip_inactive: usize = 0;
        let mut n_skip_short_circuit: usize = 0;
        let mut n_skip_summary: usize = 0;
        let mut n_loaded: usize = 0;
        let mut load_micros: u128 = 0;
        // Hour-disjointness tally: of the manifests we actually paid an S3 GET
        // for, how many had a timestamp_hour range that does not overlap the
        // targets? Those are the provably-wasted loads that hour-range pruning
        // would eliminate. Counted from the SAME FieldSummary bounds the
        // pruning function reads, so no extra I/O.
        let mut n_loaded_hour_disjoint: usize = 0;

        for manifest_entry in manifest_list.entries() {
            n_total += 1;
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
                n_skip_inactive += 1;
                continue;
            }

            // Short-circuit: if all files to delete have been found,
            // keep remaining manifests as-is without loading them from S3.
            if remaining_to_delete.is_empty() {
                n_skip_short_circuit += 1;
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
                        .map(|pt| build_target_partition_bytes(self.data_files_to_delete.as_slice(), &pt))
                        .unwrap_or_default()
                });

                if !target_bytes.is_empty()
                    && !manifest_could_contain_target_files(summaries, target_bytes)
                {
                    n_skip_summary += 1;
                    result_manifests.push(manifest_entry.clone());
                    continue;
                }

                // About to pay an S3 GET. Record whether this manifest's
                // partition-field ranges are DISJOINT from the targets on any
                // field — i.e. a load that hour-range pruning would have
                // avoided. Uses only the summary bounds already in hand.
                if !target_bytes.is_empty()
                    && summary_disjoint_from_targets(summaries, target_bytes)
                {
                    n_loaded_hour_disjoint += 1;
                }
            }

            // Load the manifest to check if any of its entries need to be deleted
            n_loaded += 1;
            let load_start = std::time::Instant::now();
            let manifest = manifest_entry
                .load_manifest(snapshot_produce.table.file_io())
                .await?;
            load_micros += load_start.elapsed().as_micros();

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
                FormatVersion::V3 | FormatVersion::V4 => builder.build_v3_data(),
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

        // One line per commit. `loaded` is the S3-GET count that
        // `manifest_could_contain_target_files` could not prune;
        // `loaded_hour_disjoint` is the subset provably unable to hold a
        // target, i.e. the waste that hour-RANGE pruning would remove.
        // `load_ms` vs `walk_ms` separates I/O from CPU so a slow walk is not
        // misread as slow S3.
        log::info!(
            "replace_data_files: manifest walk entries_total={} skipped_inactive={} \
             skipped_short_circuit={} skipped_by_summary={} loaded={} \
             loaded_hour_disjoint={} load_ms={} walk_ms={} targets={}",
            n_total,
            n_skip_inactive,
            n_skip_short_circuit,
            n_skip_summary,
            n_loaded,
            n_loaded_hour_disjoint,
            (load_micros / 1000) as u64,
            walk_start.elapsed().as_millis() as u64,
            self.files_to_delete.len()
        );

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

        // Cache the result tagged with the current snapshot_id. On retry,
        // the cache is only reused if the snapshot hasn't changed.
        {
            let current_snapshot_id = snapshot_produce
                .table
                .metadata()
                .current_snapshot()
                .map(|s| s.snapshot_id());
            let mut cache = self.cached_manifests.lock().unwrap();
            *cache = Some((current_snapshot_id, result_manifests.clone()));
        }

        Ok(result_manifests)
    }
}

#[cfg(test)]
mod walk_pruning_tests {
    use std::collections::HashSet;

    use serde_bytes::ByteBuf;

    use super::{manifest_could_contain_target_files, summary_disjoint_from_targets};
    use crate::spec::FieldSummary;

    fn summary(lo: &[u8], hi: &[u8]) -> FieldSummary {
        FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(ByteBuf::from(lo.to_vec())),
            upper_bound: Some(ByteBuf::from(hi.to_vec())),
        }
    }

    fn targets(vals: &[&[u8]]) -> Vec<HashSet<Vec<u8>>> {
        vec![vals.iter().map(|v| v.to_vec()).collect()]
    }

    /// The gap this instrumentation exists to size.
    ///
    /// A cold-tier leaf covering hours 100-200 cannot hold a file from hour
    /// 500, and its own summary proves it. But the pruning function only fires
    /// on `lower == upper`, so it says "keep" and the caller pays an S3 GET.
    /// The diagnostic sees the range and reports the load as wasted.
    #[test]
    fn range_leaf_is_loaded_despite_being_provably_disjoint() {
        let s = vec![summary(&[0, 100], &[0, 200])];
        let t = targets(&[&[1, 244]]); // well above the upper bound
        assert!(
            manifest_could_contain_target_files(&s, &t),
            "pruning keeps it — only single-value bounds are prunable"
        );
        assert!(
            summary_disjoint_from_targets(&s, &t),
            "but the range proves it cannot contain the target"
        );
    }

    /// Overlap must never be reported as disjoint — that is the direction that
    /// would lose data if this ever drove behaviour.
    #[test]
    fn overlapping_range_is_not_disjoint() {
        let s = vec![summary(&[0, 100], &[0, 200])];
        assert!(!summary_disjoint_from_targets(&s, &targets(&[&[0, 150]])));
        assert!(!summary_disjoint_from_targets(&s, &targets(&[&[0, 100]]))); // on the lower edge
        assert!(!summary_disjoint_from_targets(&s, &targets(&[&[0, 200]]))); // on the upper edge
    }

    /// Disjoint only when EVERY target is outside; one inside is enough to
    /// require the load.
    #[test]
    fn any_target_inside_range_means_not_disjoint() {
        let s = vec![summary(&[0, 100], &[0, 200])];
        assert!(!summary_disjoint_from_targets(
            &s,
            &targets(&[&[0, 150], &[1, 244]])
        ));
        assert!(summary_disjoint_from_targets(
            &s,
            &targets(&[&[0, 5], &[1, 244]])
        ));
    }

    /// Single-value bounds: the pruning function already handles these, and
    /// the diagnostic must agree rather than double-count them as waste.
    #[test]
    fn single_value_bounds_agree_with_pruning() {
        let s = vec![summary(&[0, 7], &[0, 7])];
        assert!(!manifest_could_contain_target_files(
            &s,
            &targets(&[&[0, 9]])
        ));
        assert!(summary_disjoint_from_targets(&s, &targets(&[&[0, 9]])));
        assert!(manifest_could_contain_target_files(
            &s,
            &targets(&[&[0, 7]])
        ));
        assert!(!summary_disjoint_from_targets(&s, &targets(&[&[0, 7]])));
    }

    /// Conservative on anything it cannot reason about: differing encoded
    /// widths, absent bounds, and empty targets all yield "not disjoint".
    #[test]
    fn unreasonable_input_is_never_called_disjoint() {
        // width mismatch
        let s = vec![summary(&[0, 100], &[0, 200])];
        assert!(!summary_disjoint_from_targets(&s, &targets(&[&[5]])));
        // missing bounds
        let none = vec![FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: None,
            upper_bound: None,
        }];
        assert!(!summary_disjoint_from_targets(&none, &targets(&[&[0, 9]])));
        // no targets for the field
        assert!(!summary_disjoint_from_targets(&s, &[HashSet::new()]));
        assert!(!summary_disjoint_from_targets(&s, &[]));
    }

    /// Multi-field: disjointness on ANY field is sufficient, which mirrors the
    /// observability spec where timestamp_hour discriminates but the leading
    /// signallake_tenant field has cardinality ~1 and never does.
    #[test]
    fn disjoint_on_any_field_is_enough() {
        let s = vec![summary(&[1], &[1]), summary(&[0, 100], &[0, 200])];
        let t = vec![
            [vec![1u8]].into_iter().collect::<HashSet<_>>(), // matches field 0
            [vec![1u8, 244]].into_iter().collect::<HashSet<_>>(), // disjoint on field 1
        ];
        assert!(summary_disjoint_from_targets(&s, &t));
    }
}
