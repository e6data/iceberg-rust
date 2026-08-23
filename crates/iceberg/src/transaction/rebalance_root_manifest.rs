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

//! Background rebalancing for V4 root manifests.
//!
//! Over time, fast-append commits accumulate inline entries in the root manifest.
//! When the inline count exceeds a threshold, this action flushes them into child
//! manifest files grouped by partition spec, replacing the inline entries with
//! manifest references. It also compacts child manifests whose manifest delete
//! vector (MDV) deleted fraction exceeds a threshold, rewriting them without the
//! deleted entries.
//!
//! This is a metadata-only operation — no data files are read or written.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::Mutex;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::root_manifest::{
    ManifestDeleteVector, RootManifest, RootManifestEntry, RootManifestMetadata,
    reconstruct_root, write_root_manifest,
};
use crate::spec::{
    DataContentType, FormatVersion, ManifestContentType, ManifestEntry, ManifestFile,
    ManifestStatus, ManifestWriterBuilder, Operation, PartitionSpec, SchemaRef, Snapshot,
    SnapshotReference, SnapshotRetention, Struct, Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::ActionCommit;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Default number of inline entries before triggering a flush to child manifests.
const DEFAULT_INLINE_THRESHOLD: usize = 1000;

/// Default fraction of MDV-deleted entries in a child manifest before rewriting it.
const DEFAULT_MDV_COMPACTION_THRESHOLD: f64 = 0.3;

const META_ROOT_PATH: &str = "metadata";

/// Whether a child manifest's partition summary spans more than one value on
/// any field (`lower_bound != upper_bound`). Such a manifest cannot be skipped
/// by an equality partition predicate during scan planning. An unpartitioned
/// table (no summary) is never wide.
fn manifest_file_is_wide(mf: &ManifestFile) -> bool {
    match &mf.partitions {
        None => false,
        Some(fields) => fields.iter().any(|f| match (&f.lower_bound, &f.upper_bound) {
            (Some(lo), Some(hi)) => lo != hi,
            _ => false,
        }),
    }
}

/// Build an Existing-status ManifestEntry from a source entry. Extracted so the
/// streaming write paths and the legacy `write_entries_clustered` bulk path share
/// one canonical conversion. `data_file.clone()` is the unavoidable per-entry
/// allocation (DataFile owns its column stats + partition tuple).
fn build_existing_entry(e: &ManifestEntry) -> ManifestEntry {
    ManifestEntry::builder()
        .status(ManifestStatus::Existing)
        .snapshot_id(e.snapshot_id().unwrap_or(0))
        .sequence_number(e.sequence_number().unwrap_or(0))
        .file_sequence_number_opt(e.file_sequence_number)
        .data_file(e.data_file().clone())
        .build()
}

/// Write ONE manifest file from a streaming iterator of entries.
///
/// Peak memory bounded by the ManifestWriter's internal row-group buffer, NOT by
/// the total input size — replaces the `Vec<ManifestEntry>` allocation the
/// bulk-collect callers used to hold (`survivors: Vec<...>` in the rebalance
/// hot path was ~50 MB per rewritten manifest on sri-olly).
///
/// Caller owns the `manifest_counter` and post-increments it after this returns.
/// `commit_uuid` + `manifest_id` build the deterministic output path.
#[allow(clippy::too_many_arguments)]
async fn write_manifest_from_iter(
    table: &Table,
    schema: &SchemaRef,
    spec: &PartitionSpec,
    format_version: FormatVersion,
    snapshot_id: i64,
    commit_uuid: Uuid,
    manifest_id: u64,
    is_delete: bool,
    entries: impl IntoIterator<Item = ManifestEntry>,
) -> Result<ManifestFile> {
    let manifest_path = format!(
        "{}/{}/{}-m{}.parquet",
        table.metadata().location(),
        META_ROOT_PATH,
        commit_uuid,
        manifest_id,
    );
    let output_file = table.file_io().new_output(&manifest_path)?;
    let builder = ManifestWriterBuilder::new(
        output_file,
        Some(snapshot_id),
        None,
        schema.clone(),
        spec.clone(),
    );
    let mut writer = match (format_version, is_delete) {
        (FormatVersion::V1, _) => builder.build_v1(),
        (FormatVersion::V2, false) => builder.build_v2_data(),
        (FormatVersion::V2, true) => builder.build_v2_deletes(),
        (_, false) => builder.build_v3_data(),
        (_, true) => builder.build_v3_deletes(),
    };
    for e in entries {
        writer.add_entry(e)?;
    }
    writer.write_manifest_file_parquet().await
}

/// Write `entries` (all sharing one content type and partition spec) into child
/// manifest(s). When `partition_scoped`, one manifest is written per distinct
/// partition tuple — producing tight (single-partition) summaries the planner
/// can skip on. Otherwise a single manifest is written (legacy behavior).
///
/// Bulk API retained for `graduate_buckets` and other callers that already
/// materialize a full Vec. New code paths in `RebalanceRootManifestAction`
/// use `write_manifest_from_iter` directly to avoid the intermediate Vec.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn write_entries_clustered(
    table: &Table,
    schema: &SchemaRef,
    spec: &PartitionSpec,
    format_version: FormatVersion,
    snapshot_id: i64,
    commit_uuid: Uuid,
    manifest_counter: &mut u64,
    is_delete: bool,
    entries: Vec<ManifestEntry>,
    partition_scoped: bool,
) -> Result<Vec<ManifestFile>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let groups: Vec<Vec<ManifestEntry>> = if partition_scoped {
        let mut by_part: HashMap<Struct, Vec<ManifestEntry>> = HashMap::new();
        for e in entries {
            by_part
                .entry(e.data_file.partition.clone())
                .or_default()
                .push(e);
        }
        by_part.into_values().collect()
    } else {
        vec![entries]
    };

    let mut out = Vec::with_capacity(groups.len());
    for group in groups {
        let manifest_path = format!(
            "{}/{}/{}-m{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            commit_uuid,
            *manifest_counter,
        );
        *manifest_counter += 1;

        let output_file = table.file_io().new_output(&manifest_path)?;
        let builder = ManifestWriterBuilder::new(
            output_file,
            Some(snapshot_id),
            None,
            schema.clone(),
            spec.clone(),
        );
        let mut writer = match (format_version, is_delete) {
            (FormatVersion::V1, _) => builder.build_v1(),
            (FormatVersion::V2, false) => builder.build_v2_data(),
            (FormatVersion::V2, true) => builder.build_v2_deletes(),
            (_, false) => builder.build_v3_data(),
            (_, true) => builder.build_v3_deletes(),
        };
        for e in group {
            writer.add_entry(e)?;
        }
        out.push(writer.write_manifest_file_parquet().await?);
    }
    Ok(out)
}

/// Action that rebalances a V4 root manifest by flushing accumulated inline
/// entries into child manifest files and compacting MDV-heavy manifest refs.
///
/// Use via `Transaction::rebalance_root_manifest()` or apply directly.
///
/// # When to run
///
/// Schedule periodically (e.g. every N commits or on a timer) when the table
/// uses V4 format and streaming ingest accumulates many inline entries.
///
/// # What it does
///
/// 1. Loads the current root manifest from the table's current snapshot.
/// 2. If `inline_count < inline_threshold` AND no MDV exceeds the compaction
///    threshold, returns an empty `ActionCommit` (no-op).
/// 3. Flushes inline entries into child manifest files, grouped by content type
///    (data vs. delete) and partition spec.
/// 4. For manifest refs where the MDV deleted fraction exceeds the compaction
///    threshold, loads the child manifest, filters out deleted entries, writes
///    a new manifest, and replaces the ref.
/// 5. Writes a new root manifest containing only manifest references.
/// 6. Returns an `ActionCommit` with the new snapshot.
pub struct RebalanceRootManifestAction {
    /// Number of inline entries that triggers a flush.
    inline_threshold: usize,
    /// Fraction (0.0..1.0) of MDV-deleted entries that triggers a manifest rewrite.
    mdv_compaction_threshold: f64,
    /// When true, child manifests are written one-per-partition-value (the full
    /// partition tuple), and any existing "partition-wide" manifest (whose
    /// partition summary spans more than one value) is rewritten split by
    /// partition. This produces tight summaries (lower_bound == upper_bound on
    /// every field) so the scan planner can skip manifests by partition prune.
    partition_scoped: bool,
    /// Optional cap on the number of child manifests rewritten per commit
    /// invocation. Task #485 (partition-bounded rebalance):
    ///
    /// Prior behavior: one atomic commit rewrites ALL manifests-needing-rewrite
    /// (MDV over threshold OR partition-wide when partition-scoped). Under
    /// backon retry, the WHOLE working set (loaded child manifests, cloned
    /// DataFiles, HashMap<Struct, Vec<usize>> indices, output ManifestWriters)
    /// is pinned across attempts. On sri-olly this held ~2.9 GB of live heap
    /// even with Fix A (streaming reads) + Fix C-write (chunked parquet write).
    ///
    /// New behavior when `Some(n)`: Phase A rewrites at most `n` manifests
    /// per commit, then Phase C writes a new root with the rewritten refs
    /// plus the untouched refs carried forward as-is. The caller can invoke
    /// commit() in a loop; each invocation makes bounded progress and commits
    /// atomically. Retry state pins only `n` manifests' working set, not
    /// the total. The remaining manifests carry forward unchanged (kept as
    /// their existing ManifestRef) and get processed in subsequent invocations.
    ///
    /// Phase B (inline flush) still runs to completion within a single
    /// invocation — inline entries are bounded by `inline_threshold` × entry
    /// size (typically < 50 MB) so splitting them isn't worth the extra
    /// commit count.
    ///
    /// When `None` (default), preserves the original one-shot behavior for
    /// callers that need it (e.g. small tables where extra commit round-trips
    /// dominate the wall clock).
    max_manifests_per_commit: Option<usize>,
    /// Rewritten Phase-A outputs, memoised by (source manifest path, source
    /// MDV bytes), so a CAS retry reuses the S3 work instead of redoing it.
    ///
    /// Rebalance consolidates CLOSED-hour leaves. That work is disjoint from
    /// laminar's hot appends, and the outputs are fresh UUID-addressed S3
    /// objects — the old manifests are not removed until the root swap — so a
    /// delayed or lost CAS does not invalidate them. Only the small root
    /// write has to be redone. Same reasoning `compact_cold_tier` documents
    /// for its `PreparedCompaction`: "UUID-addressed and safe to reuse across
    /// CAS retries. The only per-retry work is writing the small delta root."
    ///
    /// The key includes the source MDV: if a concurrent writer tombstoned
    /// more rows in that leaf, the cached rewrite is stale and must be
    /// recomputed. Keyed rather than all-or-nothing so one changed leaf does
    /// not discard the other N-1 rewrites.
    ///
    /// This is what `max_manifests_per_commit` was compensating for — its own
    /// doc says it bounds "retry state ... not the total". With reuse, the cap
    /// bounds only the FIRST pass.
    rewrite_cache: Arc<Mutex<HashMap<(String, Option<Vec<u8>>), Vec<ManifestFile>>>>,
    /// Optional (hour_field_idx, max_hour) filter — skip rewriting root entries
    /// whose partition hour is strictly greater than `max_hour`. Used by
    /// callers that need to leave the CURRENT hour's entries alone to avoid
    /// racing an active append stream on CAS: laminar's per-checkpoint appends
    /// touch the same root entries the rebalance would rewrite; skipping the
    /// hot hour eliminates that specific OCC conflict source without giving up
    /// rebalancing of closed hours.
    ///
    /// When set, entries whose hour cannot be extracted (invalid idx, non-int
    /// literal, missing) are conservatively KEPT (not filtered out) — the wrong
    /// safe default is "rebalance too aggressively", not "silently include".
    /// When None (default), no filtering — original behaviour.
    hour_filter: Option<(usize, i64)>,
    /// UUID for generating unique file paths in this commit.
    commit_uuid: Uuid,
}

impl RebalanceRootManifestAction {
    /// Create a new action with default thresholds.
    pub fn new() -> Self {
        Self {
            inline_threshold: DEFAULT_INLINE_THRESHOLD,
            mdv_compaction_threshold: DEFAULT_MDV_COMPACTION_THRESHOLD,
            partition_scoped: false,
            max_manifests_per_commit: None,
            rewrite_cache: Arc::new(Mutex::new(HashMap::new())),
            hour_filter: None,
            commit_uuid: Uuid::now_v7(),
        }
    }

    /// Skip rewriting root entries whose partition hour is strictly greater
    /// than `max_hour`. See the field doc on `hour_filter` for rationale.
    /// `hour_field_idx` is the 0-based index of the Hour-transform partition
    /// field in the table's default partition spec (caller looks it up once
    /// via `metadata().default_partition_spec().fields().iter().position(...)`).
    /// Passing `max_hour = i64::MAX` is equivalent to not setting the filter.
    pub fn with_hour_filter(mut self, hour_field_idx: usize, max_hour: i64) -> Self {
        self.hour_filter = Some((hour_field_idx, max_hour));
        self
    }

    /// Set the per-commit rewrite cap (task #485). See the field doc on
    /// `max_manifests_per_commit` for the rationale — this bounds Phase A's
    /// working set to N manifests per commit invocation. Callers invoking
    /// commit() in a loop can drain the full rebalance in `total / N`
    /// commits, each with retry-state pinned to only N manifests' peak.
    pub fn with_max_manifests_per_commit(mut self, n: usize) -> Self {
        self.max_manifests_per_commit = if n > 0 { Some(n) } else { None };
        self
    }

    /// Override the inline entry count threshold.
    pub fn with_inline_threshold(mut self, threshold: usize) -> Self {
        self.inline_threshold = threshold;
        self
    }

    /// Override the MDV compaction threshold (fraction of deleted entries).
    pub fn with_mdv_compaction_threshold(mut self, threshold: f64) -> Self {
        self.mdv_compaction_threshold = threshold;
        self
    }

    /// Enable partition-scoped (one-manifest-per-partition) clustering and
    /// reclustering of existing partition-wide manifests.
    pub fn with_partition_scoped(mut self, enabled: bool) -> Self {
        self.partition_scoped = enabled;
        self
    }

    /// Decode a manifest-file's partition-summary upper hour bound. Used when
    /// the caller sets `hour_filter` — mirror of tessellate's own
    /// `leaf_summary_hour`. Returns None when the field isn't present or the
    /// bound bytes don't decode as a 32-bit little-endian int (the Hour
    /// transform's on-disk shape). None → conservatively KEEP for filter
    /// purposes (safer default; see field doc).
    fn manifest_summary_hour(mf: &ManifestFile, idx: usize) -> Option<i64> {
        let fs = mf.partitions.as_ref()?.get(idx)?;
        let b = fs.upper_bound.as_ref()?;
        (b.len() >= 4).then(|| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
    }

    /// True when `hour_filter` is set and the entry's hour is strictly greater
    /// than `max_hour` (i.e. the caller told us "don't touch this — active
    /// writers are appending here"). Entries with an undecodable hour are
    /// treated as in-range (not filtered) so we never silently skip closed
    /// hours that happen to have a wonky partition summary.
    fn entry_above_hour_filter(&self, entry: &RootManifestEntry) -> bool {
        let Some((idx, max_hour)) = self.hour_filter else {
            return false;
        };
        let h = match entry {
            RootManifestEntry::ManifestRef { manifest_file, .. } => {
                Self::manifest_summary_hour(manifest_file, idx)
            }
            RootManifestEntry::Inline(me) => match me.data_file.partition.fields().get(idx) {
                Some(Some(crate::spec::Literal::Primitive(
                    crate::spec::PrimitiveLiteral::Int(v),
                ))) => Some(*v as i64),
                _ => None,
            },
        };
        match h {
            Some(hour) => hour > max_hour,
            None => false, // conservative: keep in-range when we can't decode
        }
    }

    /// Whether any manifest ref is "partition-wide" — its partition summary
    /// spans more than one value (some field's lower_bound != upper_bound) and
    /// so cannot be skipped by an equality partition predicate. Only meaningful
    /// when `partition_scoped` is enabled.
    fn needs_recluster(&self, entries: &[RootManifestEntry]) -> bool {
        self.partition_scoped
            && entries.iter().any(|e| match e {
                RootManifestEntry::ManifestRef { manifest_file, .. } => {
                    manifest_file_is_wide(manifest_file)
                }
                RootManifestEntry::Inline(_) => false,
            })
    }

    /// Check whether any manifest ref has an MDV whose deleted fraction exceeds
    /// the compaction threshold, given the total entry count from the manifest file.
    fn needs_mdv_compaction(
        &self,
        entries: &[RootManifestEntry],
    ) -> bool {
        for entry in entries {
            if let RootManifestEntry::ManifestRef {
                manifest_file,
                mdv: Some(mdv_bytes),
            } = entry
            {
                if let Ok(mdv) = ManifestDeleteVector::deserialize(mdv_bytes) {
                    let total = manifest_file
                        .added_files_count
                        .unwrap_or(0)
                        + manifest_file.existing_files_count.unwrap_or(0);
                    if total > 0 && mdv.deleted_fraction(total) >= self.mdv_compaction_threshold {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Default for RebalanceRootManifestAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for RebalanceRootManifestAction {
    fn action_name(&self) -> &'static str {
        "rebalance_root_manifest"
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // 1. Verify V4. We accept both:
        //    a) tables that declare V4 to the catalog directly (in-process
        //       / file-system catalogs / a V4-aware REST catalog), and
        //    b) tables that declare V3 but carry the e6 opt-in property
        //       (`e6.actual-format-version=4`) -- the path for V3-only
        //       catalogs like Lakekeeper pre-V4. See
        //       `crate::table::E6_ACTUAL_FORMAT_VERSION_KEY`.
        // The `effective_format_version` accessor is the single source of
        // truth for behaviour dispatch.
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "rebalance_root_manifest requires format version V4 \
                     (effective={:?}, declared={:?})",
                    table.effective_format_version(),
                    table.metadata().format_version()
                ),
            ));
        }

        // 2. Load current root manifest
        let current_snapshot = match table.metadata().current_snapshot() {
            Some(s) => s,
            None => {
                // No snapshot => nothing to rebalance
                return Ok(ActionCommit::new(vec![], vec![]));
            }
        };

        let root_manifest_path = current_snapshot.manifest_list();
        let (rm_metadata, entries) =
            reconstruct_root(table.file_io(), root_manifest_path).await?;
        let root_manifest = RootManifest::new(rm_metadata.clone(), entries);

        // 3. Check if rebalance is needed.
        // Read the SAME properties that the foreground commit path (`commit_v4`)
        // uses for its flush triggers, across ALL THREE dimensions (inline count,
        // estimated inline bytes, total entry count), so the rebalancer's flush
        // branch stays alive even when an operator relies on the byte or entry
        // triggers instead of inline count. Without this the branch was
        // effectively dead: commit_v4 flushes at its defaults (100 for count,
        // 8388608 for bytes, 1000 for entries), so the inline count alone rarely
        // reached this action's hardcoded default (1000). The builder override
        // (`with_inline_threshold`) still wins when the property is absent.
        let inline_threshold = table
            .metadata()
            .properties()
            .get("root-manifest.inline-threshold")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(self.inline_threshold);
        let flush_bytes = table
            .metadata()
            .properties()
            .get("root-manifest.flush-bytes")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8388608);
        let flush_entries = table
            .metadata()
            .properties()
            .get("root-manifest.flush-entries")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1000);
        let inline_count = root_manifest.inline_count();
        let inline_bytes = root_manifest
            .entries()
            .iter()
            .filter_map(|entry| match entry {
                RootManifestEntry::Inline(me) => Some(256 + me.data_file.file_path.len()),
                RootManifestEntry::ManifestRef { .. } => None,
            })
            .sum::<usize>();
        let total_entries = root_manifest.entries().len();
        let needs_flush = inline_count >= inline_threshold
            || inline_bytes >= flush_bytes
            || total_entries >= flush_entries;
        let needs_mdv_compact = self.needs_mdv_compaction(root_manifest.entries());
        let needs_recluster = self.needs_recluster(root_manifest.entries());
        // A backlog of carried path tombstones is itself work worth a commit.
        // Without this the sweep below could never run on a table where nothing
        // else needs rebalancing, and the set would grow forever — which is how
        // sri-olly reached 430k paths / ~108 MB of a 132 MB root.
        let sweep_min_paths = table
            .metadata()
            .properties()
            .get("root-manifest.tombstone-sweep-min-paths")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1000);
        let needs_tombstone_sweep = rm_metadata.removed_paths.len() >= sweep_min_paths;

        if !needs_flush && !needs_mdv_compact && !needs_recluster && !needs_tombstone_sweep {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // Prepare for writing
        let snapshot_id =
            SnapshotProducer::generate_unique_snapshot_id_static(table);
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let format_version = table.metadata().format_version();
        let commit_uuid = self.commit_uuid;
        let mut manifest_counter: u64 = 0;

        // Separate entries into manifest refs and inline entries
        let mut new_entries: Vec<RootManifestEntry> = Vec::new();
        let mut cache_hits: usize = 0;

        // Task #485: track how many manifests Phase A has actually rewritten
        // this invocation. When `max_manifests_per_commit` is set, we stop
        // rewriting after `cap` and fall through to carry the remaining
        // manifest refs forward as-is. Caller loops commit() until the action
        // returns a no-op (all rewrites done) — each commit is atomic and
        // pins only up to `cap` manifests' working set in the backon retry
        // future, instead of the whole rebalance's.
        let mut rewrites_done: usize = 0;

        // --- Phase A: Process existing manifest refs ---
        // A ref is rewritten when its MDV crosses the compaction threshold OR
        // (when partition-scoped) its partition summary is wide. In both cases we
        // load the child manifest, drop any MDV-deleted entries, and re-emit the
        // survivors — split one-manifest-per-partition when partition-scoped, so
        // wide manifests become tight (skippable) ones.
        for entry in root_manifest.entries() {
            if let RootManifestEntry::ManifestRef {
                manifest_file,
                mdv,
            } = entry
            {
                let mdv_obj = match mdv {
                    Some(bytes) => Some(ManifestDeleteVector::deserialize(bytes)?),
                    None => None,
                };
                let over_mdv_threshold = mdv_obj.as_ref().is_some_and(|m| {
                    let total = manifest_file.added_files_count.unwrap_or(0)
                        + manifest_file.existing_files_count.unwrap_or(0);
                    total > 0 && m.deleted_fraction(total) >= self.mdv_compaction_threshold
                });
                let is_wide = self.partition_scoped && manifest_file_is_wide(manifest_file);

                if !over_mdv_threshold && !is_wide {
                    // Keep the manifest ref (and any MDV) as-is.
                    new_entries.push(entry.clone());
                    continue;
                }

                // Hot-hour skip: if the caller set a max_hour and this manifest's
                // summary hour is above it, carry forward as-is. Same shape as the
                // `!over_mdv_threshold && !is_wide` early-return above — the
                // manifest keeps its MDV (if any) and its shape (partition-wide
                // or not). Point of the filter is exactly to avoid rewriting
                // entries an active writer is racing us on.
                if self.entry_above_hour_filter(entry) {
                    new_entries.push(entry.clone());
                    continue;
                }

                // Task #485 cap: if we've already rewritten `max_manifests_per_commit`
                // entries this invocation, carry the rest forward unchanged. The
                // caller's next commit() invocation loads the fresh table view
                // (post this commit) and picks up where we left off — the still-
                // needs-rewrite refs are unchanged, so `needs_mdv_compaction` /
                // `needs_recluster` at the top of commit() will re-trigger for
                // them and they'll be processed then.
                if let Some(cap) = self.max_manifests_per_commit {
                    if rewrites_done >= cap {
                        new_entries.push(entry.clone());
                        continue;
                    }
                }
                rewrites_done += 1;

                // Load the child manifest and stream surviving entries (alive,
                // not MDV-deleted) into new manifest(s). Rewriting drops the MDV
                // entirely.
                //
                // Streaming pattern (replaces the `let survivors: Vec<...>` +
                // bulk `write_entries_clustered` pair). Two write shapes:
                //
                //   * non-partition-scoped: build one filter/map iterator over
                //     `manifest.entries()` and feed it directly to a single
                //     writer via `write_manifest_from_iter`. Peak memory = one
                //     row-group buffer inside the writer (was ~50 MB per
                //     source manifest for the fully-collected Vec on sri-olly).
                //
                //   * partition-scoped: two logical passes over the SAME source
                //     Manifest (already in RAM):
                //       pass 1 groups source indices by partition tuple
                //         into HashMap<Struct, Vec<usize>> (tiny — 8 bytes
                //         per surviving entry vs ~1-5 KB in the old Vec of
                //         cloned ManifestEntry),
                //       pass 2 writes one manifest per partition serially,
                //         each pulling from source entries by index.
                //     Only one output writer is live at a time, so peak =
                //     one row-group buffer + the small indices map.
                // Reuse a prior rewrite of this exact (source manifest, MDV)
                // if we already produced one in an earlier CAS attempt. The
                // outputs are fresh UUID-addressed S3 objects and the source
                // is a closed-hour leaf, so nothing about them goes stale when
                // a commit loses the race — only the root write must be redone.
                let cache_key = (manifest_file.manifest_path.clone(), mdv.clone());
                if let Some(cached) = self.rewrite_cache.lock().await.get(&cache_key) {
                    for mf in cached.iter().cloned() {
                        new_entries.push(RootManifestEntry::ManifestRef {
                            manifest_file: mf,
                            mdv: None,
                        });
                    }
                    rewrites_done += 1;
                    cache_hits += 1;
                    continue;
                }
                let mut produced: Vec<ManifestFile> = Vec::new();

                let manifest = manifest_file.load_manifest(table.file_io()).await?;
                let spec_id = manifest_file.partition_spec_id;
                let spec = table
                    .metadata()
                    .partition_spec_by_id(spec_id)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("partition spec {spec_id} not found"),
                        )
                    })?;
                let is_delete = manifest_file.content == ManifestContentType::Deletes;

                let is_survivor = |idx: usize, e: &ManifestEntry| -> bool {
                    e.is_alive()
                        && !mdv_obj
                            .as_ref()
                            .is_some_and(|m| m.is_deleted(idx as u32))
                };

                if self.partition_scoped {
                    // Pass 1: index only — HashMap<Struct, Vec<usize>>.
                    let mut by_part_indices: HashMap<Struct, Vec<usize>> =
                        HashMap::new();
                    for (idx, e) in manifest.entries().iter().enumerate() {
                        if !is_survivor(idx, e) {
                            continue;
                        }
                        by_part_indices
                            .entry(e.data_file().partition.clone())
                            .or_default()
                            .push(idx);
                    }
                    // Pass 2: one manifest per partition, streamed.
                    for (_partition, indices) in by_part_indices {
                        let entries_iter = indices
                            .into_iter()
                            .map(|idx| build_existing_entry(&manifest.entries()[idx]));
                        let mf = write_manifest_from_iter(
                            table,
                            &schema,
                            spec.as_ref(),
                            format_version,
                            snapshot_id,
                            commit_uuid,
                            manifest_counter,
                            is_delete,
                            entries_iter,
                        )
                        .await?;
                        manifest_counter += 1;
                        produced.push(mf.clone());
                        new_entries.push(RootManifestEntry::ManifestRef {
                            manifest_file: mf,
                            mdv: None,
                        });
                    }
                } else {
                    // Non-partition-scoped: single streaming pass into one writer.
                    let entries_iter =
                        manifest.entries().iter().enumerate().filter_map(|(idx, e)| {
                            if is_survivor(idx, e) {
                                Some(build_existing_entry(e))
                            } else {
                                None
                            }
                        });
                    let mf = write_manifest_from_iter(
                        table,
                        &schema,
                        spec.as_ref(),
                        format_version,
                        snapshot_id,
                        commit_uuid,
                        manifest_counter,
                        is_delete,
                        entries_iter,
                    )
                    .await?;
                    manifest_counter += 1;
                    produced.push(mf.clone());
                    new_entries.push(RootManifestEntry::ManifestRef {
                        manifest_file: mf,
                        mdv: None,
                    });
                }
                self.rewrite_cache.lock().await.insert(cache_key, produced);
            }
        }

        // Diagnostics for the cap decision. `cache_hits` is the number of
        // Phase-A rewrites served from a previous CAS attempt instead of
        // re-reading and re-writing the source manifest. This is the number to
        // watch before raising TESSELLATE_V2_REBALANCE_MAX_MANIFESTS_PER_COMMIT:
        // the cap exists to bound RETRY redo, so if hits are high the redo cost
        // is gone and the cap can rise. If `retries` stay ~0 the cache will
        // rarely be exercised, and the cap can rise for a different reason —
        // there was never much redo to bound.
        log::info!(
            "rebalance phase A: rewrites_done={} cache_hits={} cap={} entries_in_root={}",
            rewrites_done,
            cache_hits,
            self.max_manifests_per_commit
                .map(|c| c.to_string())
                .unwrap_or_else(|| "none".to_string()),
            root_manifest.entries().len()
        );

        // --- Phase B: Flush inline entries into child manifests ---
        if needs_flush {
            // Group inline entries by (content_type, partition_spec_id) using
            // ONLY source indices — HashMap<(is_delete, spec_id), Vec<usize>>
            // is a few bytes per entry vs the old
            // HashMap<i32, Vec<ManifestEntry>> which held a full ManifestEntry
            // (with cloned DataFile, ~1-5 KB each). For partition-scoped writes
            // we build a second-level index inside the flush loop so each
            // partition's manifest streams from source indices one at a time.
            //
            // Hot-hour skip: same filter as Phase A above — inline entries whose
            // partition hour is above `hour_filter.max_hour` are NOT grouped
            // here; instead they fall through to the trailing carry-forward
            // block that copies them into `new_entries` unchanged. Rationale
            // identical: don't race active append writers on the current hour.
            let mut inline_by_group: HashMap<(bool, i32), Vec<usize>> = HashMap::new();
            for (idx, entry) in root_manifest.entries().iter().enumerate() {
                if let RootManifestEntry::Inline(me) = entry {
                    if self.entry_above_hour_filter(entry) {
                        // Skipped here → carried forward by the trailing loop.
                        continue;
                    }
                    let is_delete = matches!(
                        me.data_file.content,
                        DataContentType::EqualityDeletes | DataContentType::PositionDeletes,
                    );
                    inline_by_group
                        .entry((is_delete, me.data_file.partition_spec_id))
                        .or_default()
                        .push(idx);
                }
            }

            for ((is_delete, spec_id), indices) in inline_by_group {
                let spec = table
                    .metadata()
                    .partition_spec_by_id(spec_id)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("partition spec {spec_id} not found"),
                        )
                    })?;

                // Convert a root-manifest inline `me` at source index `idx`
                // into an Existing ManifestEntry. Kept as a closure so both
                // partition-scoped and single-manifest paths use the same
                // conversion.
                let make_entry = |idx: usize| -> ManifestEntry {
                    let me = match &root_manifest.entries()[idx] {
                        RootManifestEntry::Inline(me) => me,
                        // Filtered above — this arm is unreachable in
                        // practice, but return a zeroed placeholder rather
                        // than panic (the index list is authoritative).
                        _ => unreachable!("inline_by_group only holds Inline indices"),
                    };
                    ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(me.snapshot_id.unwrap_or(0))
                        .sequence_number(me.sequence_number.unwrap_or(0))
                        .file_sequence_number_opt(me.file_sequence_number)
                        .data_file(me.data_file.clone())
                        .build()
                };

                if self.partition_scoped {
                    // Pass 1: partition-tuple index over the inline slice —
                    // HashMap<Struct, Vec<usize>> where value is the SAME
                    // source index space used above (indices into
                    // `root_manifest.entries()`).
                    let mut by_part_indices: HashMap<Struct, Vec<usize>> =
                        HashMap::new();
                    for idx in &indices {
                        let part = match &root_manifest.entries()[*idx] {
                            RootManifestEntry::Inline(me) => me.data_file.partition.clone(),
                            _ => unreachable!(),
                        };
                        by_part_indices.entry(part).or_default().push(*idx);
                    }
                    for (_partition, part_indices) in by_part_indices {
                        let entries_iter = part_indices.into_iter().map(make_entry);
                        let mf = write_manifest_from_iter(
                            table,
                            &schema,
                            spec.as_ref(),
                            format_version,
                            snapshot_id,
                            commit_uuid,
                            manifest_counter,
                            is_delete,
                            entries_iter,
                        )
                        .await?;
                        manifest_counter += 1;
                        new_entries.push(RootManifestEntry::ManifestRef {
                            manifest_file: mf,
                            mdv: None,
                        });
                    }
                } else {
                    // Single manifest for this (is_delete, spec_id) — stream
                    // from indices directly.
                    let entries_iter = indices.into_iter().map(make_entry);
                    let mf = write_manifest_from_iter(
                        table,
                        &schema,
                        spec.as_ref(),
                        format_version,
                        snapshot_id,
                        commit_uuid,
                        manifest_counter,
                        is_delete,
                        entries_iter,
                    )
                    .await?;
                    manifest_counter += 1;
                    new_entries.push(RootManifestEntry::ManifestRef {
                        manifest_file: mf,
                        mdv: None,
                    });
                }
            }

            // Carry-forward any inline entries the hot-hour filter caused us
            // to skip during grouping above. Without this, filtered inline
            // entries silently vanish from the rewritten root (data loss).
            // No-op when hour_filter is None (nothing was skipped).
            if self.hour_filter.is_some() {
                for entry in root_manifest.entries() {
                    if let RootManifestEntry::Inline(_) = entry
                        && self.entry_above_hour_filter(entry)
                    {
                        new_entries.push(entry.clone());
                    }
                }
            }
        } else {
            // No flush needed — carry inline entries forward as-is
            for entry in root_manifest.entries() {
                if let RootManifestEntry::Inline(_) = entry {
                    new_entries.push(entry.clone());
                }
            }
        }

        // --- Phase C: Write new root manifest ---
        let partition_type = table
            .metadata()
            .default_partition_spec()
            .partition_type(table.metadata().current_schema())?;

        // Sweep the carried path tombstones before writing the root.
        //
        // This used to be `rm_metadata.removed_paths.clone()` — an unconditional
        // carry-forward. Combined with `is_survivor` consulting only
        // `e.is_alive()` and the MDV (never `removed_paths`), and with removals
        // on an incremental table never becoming MDVs, nothing here ever retired
        // a tombstone: the set grew monotonically to 430k paths / ~108 MB of a
        // 132 MB root on sri-olly.
        //
        // The sweep also has to live HERE, not only in laminar's collapse. Once
        // the root shrank enough for this action to start winning its CAS again,
        // it began writing a fresh base (`chain_depth: 0`) every ~10 min, which
        // resets laminar's chain so it never reaches MAX_CHAIN and its collapse —
        // and therefore its sweep — never fires. Whoever writes the base sweeps.
        //
        // Cost gate. A sweep reads EVERY manifest in the tree, so it is only
        // worth doing against a real backlog — `needs_tombstone_sweep` above uses
        // the same threshold to decide whether a backlog alone justifies a
        // commit. Gating only on non-empty (the first cut) meant that once the
        // backlog was drained, every rebalance triggered for some OTHER reason
        // still paid a full ~4.3k-manifest scan to retire a handful of paths —
        // seconds of S3 reads inside the CAS window, repeated per attempt.
        //
        // Fail-open: a sweep error must not fail the rebalance. Worst case we
        // carry the tombstones forward exactly as before.
        let mut swept_removed = rm_metadata.removed_paths.clone();
        if swept_removed.len() >= sweep_min_paths {
            let max_manifests = table
                .metadata()
                .properties()
                .get("root-manifest.tombstone-materialize-max-manifests")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8192);
            let before = swept_removed.len();
            let cold_owned = crate::transaction::cold_paths::resolve_cold_presence(
                table.file_io(),
                rm_metadata.bucket_index_path.as_deref(),
            )
            .await;
            match crate::transaction::snapshot::materialize_carried_tombstones(
                table.file_io(),
                &mut new_entries,
                &mut swept_removed,
                cold_owned.as_ref(),
                max_manifests,
            )
            .await
            {
                Ok((retired, scanned)) => log::info!(
                    "rebalance: swept carried tombstones: before={} retired={} \
                     remaining={} manifests_scanned={}",
                    before,
                    retired,
                    swept_removed.len(),
                    scanned
                ),
                Err(e) => {
                    log::warn!("rebalance: tombstone sweep failed, carrying forward: {e:#}");
                    swept_removed = rm_metadata.removed_paths.clone();
                }
            }
        }

        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: table.metadata().default_partition_spec().clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            // Carry the cold bucket-index pointer forward unchanged — rebalance
            // only rewrites live refs/MDV, never the tiered cold layer.
            bucket_index_path: rm_metadata.bucket_index_path.clone(),
            prev_root_path: None,
            chain_depth: 0,
            node_level: 0,
            removed_paths: swept_removed,
        };

        let new_root_manifest_path = format!(
            "{}/{}/root-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            snapshot_id,
            commit_uuid,
        );

        let root_bytes =
            write_root_manifest(&new_entries, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_manifest_path)?
            .write(root_bytes.into())
            .await?;

        // --- Phase D: Build snapshot and ActionCommit ---
        let inline_flushed = if needs_flush { inline_count } else { 0 };
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "rebalance-inline-flushed".to_string(),
                    inline_flushed.to_string(),
                ),
                (
                    "rebalance-entries-after".to_string(),
                    new_entries.len().to_string(),
                ),
            ]),
        };

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let first_row_id = table.metadata().next_row_id();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(new_root_manifest_path.clone())
            .with_snapshot_id(snapshot_id)
            .with_parent_snapshot_id(table.metadata().current_snapshot_id())
            .with_sequence_number(next_seq_num)
            .with_summary(summary)
            .with_schema_id(table.metadata().current_schema_id())
            .with_timestamp_ms(commit_ts)
            .with_row_range(first_row_id, 0)
            .build();

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

        Ok(
            ActionCommit::new(updates, requirements)
                .with_manifest_paths(vec![new_root_manifest_path]),
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_bytes::ByteBuf;

    use super::*;
    use crate::spec::FieldSummary;

    fn mf_with_partitions(partitions: Option<Vec<FieldSummary>>) -> ManifestFile {
        ManifestFile {
            manifest_path: "s3://bucket/m.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(1),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions,
            key_metadata: None,
            first_row_id: None,
        }
    }

    fn fsummary(lo: &[u8], hi: &[u8]) -> FieldSummary {
        FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(ByteBuf::from(lo.to_vec())),
            upper_bound: Some(ByteBuf::from(hi.to_vec())),
        }
    }

    fn manifest_ref(mf: ManifestFile) -> RootManifestEntry {
        RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: None,
        }
    }

    #[test]
    fn hour_filter_manifest_ref_skips_current_hour() {
        // hour_field_idx = 0. Encoding: hour bucket as i32 LE bytes.
        let hour_hot: i32 = 496200;
        let hour_cold: i32 = 496195;
        let max_hour: i64 = 496199; // strictly-greater is skipped

        let hot_mf = mf_with_partitions(Some(vec![fsummary(
            &hour_hot.to_le_bytes(),
            &hour_hot.to_le_bytes(),
        )]));
        let cold_mf = mf_with_partitions(Some(vec![fsummary(
            &hour_cold.to_le_bytes(),
            &hour_cold.to_le_bytes(),
        )]));

        let action = RebalanceRootManifestAction::new().with_hour_filter(0, max_hour);
        assert!(action.entry_above_hour_filter(&manifest_ref(hot_mf)));
        assert!(!action.entry_above_hour_filter(&manifest_ref(cold_mf)));

        // no filter set → nothing above (both keep default = false)
        let unfiltered = RebalanceRootManifestAction::new();
        let hot_mf2 = mf_with_partitions(Some(vec![fsummary(
            &hour_hot.to_le_bytes(),
            &hour_hot.to_le_bytes(),
        )]));
        assert!(!unfiltered.entry_above_hour_filter(&manifest_ref(hot_mf2)));

        // undecodable hour (no partitions) → conservatively KEEP (not above filter)
        let no_parts = mf_with_partitions(None);
        assert!(!action.entry_above_hour_filter(&manifest_ref(no_parts)));

        // undecodable hour (upper_bound too short) → conservatively KEEP
        let short_bound = mf_with_partitions(Some(vec![fsummary(&[0u8], &[0u8])]));
        assert!(!action.entry_above_hour_filter(&manifest_ref(short_bound)));
    }

    #[test]
    fn partition_scoped_builder_default_off() {
        assert!(!RebalanceRootManifestAction::new().partition_scoped);
        assert!(
            RebalanceRootManifestAction::new()
                .with_partition_scoped(true)
                .partition_scoped
        );
    }

    #[test]
    fn manifest_wide_detection() {
        // Unpartitioned table -> never wide.
        assert!(!manifest_file_is_wide(&mf_with_partitions(None)));
        // Single partition value (lower == upper) -> tight.
        assert!(!manifest_file_is_wide(&mf_with_partitions(Some(vec![fsummary(
            &[0, 0, 0, 1],
            &[0, 0, 0, 1]
        )]))));
        // Range (lower != upper) -> wide.
        assert!(manifest_file_is_wide(&mf_with_partitions(Some(vec![fsummary(
            &[0, 0, 0, 1],
            &[0, 0, 0, 9]
        )]))));
        // Multi-field, second field wide -> wide.
        assert!(manifest_file_is_wide(&mf_with_partitions(Some(vec![
            fsummary(&[1], &[1]),
            fsummary(&[2], &[5]),
        ]))));
    }

    #[test]
    fn needs_recluster_requires_flag_and_wide() {
        let wide = vec![manifest_ref(mf_with_partitions(Some(vec![fsummary(
            &[0, 0, 0, 1],
            &[0, 0, 0, 9],
        )])))];
        let tight = vec![manifest_ref(mf_with_partitions(Some(vec![fsummary(
            &[0, 0, 0, 1],
            &[0, 0, 0, 1],
        )])))];

        // Flag off -> never needs recluster, even with wide manifests.
        assert!(!RebalanceRootManifestAction::new().needs_recluster(&wide));

        // Flag on -> wide triggers; tight does not.
        let scoped = RebalanceRootManifestAction::new().with_partition_scoped(true);
        assert!(scoped.needs_recluster(&wide));
        assert!(!scoped.needs_recluster(&tight));
    }

    #[test]
    fn test_default_thresholds() {
        let action = RebalanceRootManifestAction::new();
        assert_eq!(action.inline_threshold, DEFAULT_INLINE_THRESHOLD);
        assert!(
            (action.mdv_compaction_threshold - DEFAULT_MDV_COMPACTION_THRESHOLD).abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn test_custom_thresholds() {
        let action = RebalanceRootManifestAction::new()
            .with_inline_threshold(500)
            .with_mdv_compaction_threshold(0.5);
        assert_eq!(action.inline_threshold, 500);
        assert!((action.mdv_compaction_threshold - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_needs_mdv_compaction_no_mdv() {
        let action = RebalanceRootManifestAction::new();
        let mf = ManifestFile {
            manifest_path: "s3://bucket/m0.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: None,
        }];
        assert!(!action.needs_mdv_compaction(&entries));
    }

    #[test]
    fn test_needs_mdv_compaction_below_threshold() {
        let action = RebalanceRootManifestAction::new()
            .with_mdv_compaction_threshold(0.5);
        let mf = ManifestFile {
            manifest_path: "s3://bucket/m0.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        // Mark 2 out of 10 as deleted (20% < 50% threshold)
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        mdv.mark_deleted(1);
        let mdv_bytes = mdv.serialize().unwrap();

        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: Some(mdv_bytes),
        }];
        assert!(!action.needs_mdv_compaction(&entries));
    }

    #[test]
    fn test_needs_mdv_compaction_above_threshold() {
        let action = RebalanceRootManifestAction::new()
            .with_mdv_compaction_threshold(0.3);
        let mf = ManifestFile {
            manifest_path: "s3://bucket/m0.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        };
        // Mark 4 out of 10 as deleted (40% >= 30% threshold)
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        mdv.mark_deleted(1);
        mdv.mark_deleted(2);
        mdv.mark_deleted(3);
        let mdv_bytes = mdv.serialize().unwrap();

        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: mf,
            mdv: Some(mdv_bytes),
        }];
        assert!(action.needs_mdv_compaction(&entries));
    }
}
