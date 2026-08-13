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

//! Graduate closed live nodes into the cold bucket-index (tiered V4 metadata).
//!
//! With append-only live nodes (the V4 "one-file commit" model), each commit
//! writes its new files into a fresh child manifest ("node") and keeps only the
//! node *reference* in the root. This action relocates the **closed** live nodes
//! — those whose newest event time has fallen behind the cutoff — out of the hot
//! root and into the cold bucket-index. The move is reference-only: a closed
//! node is immutable, so graduating it just moves its `ManifestFile` ref from
//! the root into the bucket-index (no data read, no re-write).
//!
//! For robustness it also handles any residual **inline** entries (the older
//! "live = inline" shape): closed inline files are materialized into new
//! partition-tight cold leaves.
//!
//! The close decision keys on **time** — the max value of `ts_field_id` across a
//! node's data files — so it is fully partition-spec-agnostic. The caller
//! computes `cutoff_micros = now − bucket_window`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{iter as stream_iter, StreamExt};
use uuid::Uuid;

/// Concurrency for the parallel `load_manifest` fallback in the collapse-fold
/// classify + TTL-prune passes. On sri-olly the fold saw ~271 cold leaves +
/// ~143 hot refs where `ref_max_event_micros` couldn't answer from the
/// partition summary (because `tiered-metadata.timestamp-field=ingestion_time`
/// isn't a partition field). Sequential await drove ~18s of critical-path
/// wall on every no-op fold (271 + 143 × ~45ms S3 GET). Parallel-fetching
/// bounded by this constant brings that back into the single-digit hundreds
/// of ms. Matches merge_worker's fanout; well within S3's per-client budget.
const FOLD_MANIFEST_FETCH_CONCURRENCY: usize = 32;

/// Bound the TTL-prune auto-heal batch. When a sidecar is fully poisoned
/// (older writers cached `Some(None)` for every leaf on tables whose
/// retention ts-field ≠ partition source — e.g. logs on `ingestion_time`),
/// the poison-tolerance fix (2026-07-21) routes EVERY leaf into the parallel
/// prefetch on the first fold. On sri-olly that's thousands of manifest
/// loads on the commit critical path, blowing the 30s commit timeout →
/// commits drop → sidecar never rewrites → deadlock. Capping the per-fold
/// prefetch means each fold heals a slice (coldest-first, so TTL drops
/// oldest data first) and the sidecar warms incrementally over ~ceil(N/cap)
/// folds. At 32-way concurrency + ~50ms/leaf, this ceiling costs ~400ms of
/// wall in the worst case — comfortably inside the commit budget.
const MAX_TTL_PREFETCH_LEAVES: usize = 256;

/// Suffix appended to a bucket-index path to derive its max-ts sidecar path.
/// The sidecar caches, for the leaves that particular bucket-index references,
/// the newest event time each leaf holds under a given `ts_field_id` — the
/// exact value TTL prune and classification need. Sidecar present + covering
/// = zero S3 GETs on the fold's ttl_prune pass; sidecar missing/stale = fall
/// back to the parallel-prefetch loop (which stays exactly as before).
const BUCKET_INDEX_MAXTS_SIDECAR_SUFFIX: &str = ".max-ts.json";

/// JSON shape of the max-ts sidecar. Reader validates `ts_field_id` and
/// `format_version` before trusting the map; a mismatch (e.g. someone changed
/// `tiered-metadata.timestamp-field`) invalidates the whole sidecar and we
/// fall back to loading manifests. Entries not present in the map on read
/// still fall back on a per-leaf basis.
#[derive(serde::Serialize, serde::Deserialize)]
struct MaxTsSidecar {
    /// Sidecar format version; bump if the on-disk shape changes.
    format_version: u32,
    /// The ts_field_id whose max values are cached. Mismatches invalidate.
    ts_field_id: i32,
    /// `{leaf.manifest_path → newest ts_field_id upper-bound in micros}`.
    /// Only REAL values are written by current writers; a leaf we can't resolve
    /// is left ABSENT (never cached as `None`) so the next fold re-derives it.
    /// The type stays `Option<i64>` for wire-compat with older sidecars that DID
    /// cache negatives — the reader treats any such `None` as a miss (auto-heal),
    /// never as an authoritative "no timestamp".
    entries: HashMap<String, Option<i64>>,
}
const MAXTS_SIDECAR_FORMAT_V1: u32 = 1;

fn bucket_index_maxts_sidecar_path(bucket_index_path: &str) -> String {
    format!("{bucket_index_path}{BUCKET_INDEX_MAXTS_SIDECAR_SUFFIX}")
}

/// Load a max-ts sidecar for `bucket_index_path`. `None` on any failure
/// (missing, unreadable, corrupt, wrong ts_field_id, wrong format version)
/// — the caller falls back to loading manifests, so all failure modes are
/// non-fatal.
async fn load_maxts_sidecar(
    file_io: &crate::io::FileIO,
    bucket_index_path: &str,
    ts_field_id: i32,
) -> Option<HashMap<String, Option<i64>>> {
    let path = bucket_index_maxts_sidecar_path(bucket_index_path);
    let input = file_io.new_input(&path).ok()?;
    let bytes = input.read().await.ok()?;
    let sidecar: MaxTsSidecar = serde_json::from_slice(&bytes).ok()?;
    if sidecar.format_version != MAXTS_SIDECAR_FORMAT_V1 {
        return None;
    }
    if sidecar.ts_field_id != ts_field_id {
        return None;
    }
    Some(sidecar.entries)
}

/// Write a max-ts sidecar for `bucket_index_path`. Non-fatal on failure —
/// we log and continue, since the sidecar is a pure cache and the next fold
/// will just fall back to loading manifests until the write eventually
/// succeeds.
async fn write_maxts_sidecar(
    file_io: &crate::io::FileIO,
    bucket_index_path: &str,
    ts_field_id: i32,
    entries: HashMap<String, Option<i64>>,
) -> crate::error::Result<()> {
    let sidecar = MaxTsSidecar {
        format_version: MAXTS_SIDECAR_FORMAT_V1,
        ts_field_id,
        entries,
    };
    let bytes = serde_json::to_vec(&sidecar).map_err(|e| {
        crate::error::Error::new(
            crate::error::ErrorKind::Unexpected,
            format!("serialize max-ts sidecar: {e}"),
        )
    })?;
    let path = bucket_index_maxts_sidecar_path(bucket_index_path);
    let output = file_io.new_output(&path)?;
    output.write(bytes.into()).await
}

use tokio::sync::Mutex;

use super::rebalance_root_manifest::write_entries_clustered;
use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    read_root_manifest, reconstruct_root, write_root_manifest, RootManifestEntry,
    RootManifestMetadata,
};
use crate::spec::{
    DataFile, FormatVersion, ManifestEntry, ManifestFile, ManifestStatus, Operation,
    PartitionSpec, PrimitiveLiteral, Snapshot, SnapshotReference, SnapshotRetention, Summary,
    Transform, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";
/// Chain-depth cap for the incremental delta path. Mirrors `commit_v4`'s
/// constant at `snapshot.rs:1349` and the identical cap in `compact_cold_tier`
/// — beyond MAX_CHAIN we fall back to a flat-base collapse.
const MAX_CHAIN: u32 = 64;

/// Cached prep from a prior `commit()` attempt on the same
/// [`GraduateBucketsAction`] instance. Same retry-cheap invariant as
/// `compact_cold_tier`'s `PreparedCompaction`: the heavy S3-resident outputs
/// (the new bucket-index parquet + new cold-tier leaves inside it) are
/// UUID-addressed and safe to reuse across CAS retries. The only per-retry
/// work is writing the small delta root that points at them.
///
/// Under sri-olly's Forbid-concurrency CronJob only ONE tessellate instance
/// runs at a time, so between prep and retry the only thing that can change
/// is laminar's hot appends — which don't touch the cold tier. The
/// bucket_index_path fingerprint check is therefore essentially always
/// satisfied. If it isn't (rare cross-tick collision), the cached prep is
/// invalidated (its S3 outputs become orphans, reclaimed by tombstone GC)
/// and re-prep runs.
struct PreparedGraduation {
    /// Reconstructed hot-root entries that stay hot after graduation. Cached
    /// so the flat-base fallback (chain_depth >= MAX_CHAIN) doesn't need a
    /// re-reconstruct.
    kept: Vec<RootManifestEntry>,
    /// Freshly-written bucket-index containing existing cold leaves +
    /// newly-graduated ones. Path is unique per (snapshot_id, commit_uuid).
    bucket_index_path: String,
    /// Bucket-index path the head root pointed at when prep ran. Fast-path
    /// validity: if the current head still points here, no other cold-tier
    /// writer landed in between and the cached prep is valid.
    prep_bucket_index_path: Option<String>,
    /// Merged ancestor tombstones from prep-time `reconstruct_root`; needed
    /// only for the flat-base fallback (delta path emits empty `removed_paths`
    /// so ancestor tombstones stay live via chain walk).
    ancestor_removed_paths: Vec<String>,
    /// Snapshot id allocated once at prep and reused across retries so leaves
    /// have consistent lineage regardless of which retry wins the CAS.
    snapshot_id: i64,
    /// Summary counters from the fold — carried into the new snapshot's
    /// Summary regardless of which retry ultimately wins.
    nodes_moved: usize,
    inline_leaves: usize,
    cold_leaves_total: usize,
}

/// Action that relocates closed live nodes (and any closed inline files) into
/// the cold bucket-index. Use via `Transaction::graduate_buckets(ts_field_id,
/// cutoff_micros)`.
pub struct GraduateBucketsAction {
    ts_field_id: i32,
    cutoff_micros: i64,
    commit_uuid: Uuid,
    /// Cached prep for retry cheapness. `None` before the first `commit()`
    /// call, populated after prep runs. `tokio::sync::Mutex` because the
    /// Transaction commit loop uses `Arc<Self>` and holds the lock across
    /// async I/O.
    prepared: Mutex<Option<PreparedGraduation>>,
}

impl GraduateBucketsAction {
    /// Create with the event-time field id and the close cutoff (micros). A node
    /// (or inline file) graduates when its max value for `ts_field_id` is below
    /// `cutoff_micros`.
    pub fn new(ts_field_id: i32, cutoff_micros: i64) -> Self {
        Self {
            ts_field_id,
            cutoff_micros,
            commit_uuid: Uuid::now_v7(),
            prepared: Mutex::new(None),
        }
    }
}

/// Max value of the `ts_field_id` upper-bound statistic across `files` (newest
/// event time). `None` if no file carries the stat. Pure.
fn max_ts_of(files: &[DataFile], ts_field_id: i32) -> Option<i64> {
    files
        .iter()
        .filter_map(
            |f| match f.upper_bounds().get(&ts_field_id).map(|d| d.literal()) {
                Some(PrimitiveLiteral::Long(v)) => Some(*v),
                _ => None,
            },
        )
        .max()
}

/// Max event time of a single data file for `ts_field_id`, if present. Pure.
fn file_max_ts(df: &DataFile, ts_field_id: i32) -> Option<i64> {
    match df.upper_bounds().get(&ts_field_id).map(|d| d.literal()) {
        Some(PrimitiveLiteral::Long(v)) => Some(*v),
        _ => None,
    }
}

/// Resolve a max-ts sidecar lookup into a trustworthy value. A hit exists ONLY
/// when the entry holds a REAL value (`Some(Some)`). A present-but-`None` entry
/// — poison written by an older writer that cached a non-answer — is treated as
/// a MISS so the caller re-derives the max-ts (partition summary / manifest
/// load) instead of trusting it. Absent (`None`) is likewise a miss. Pure.
///
/// Without this, an all-`None` sidecar permanently suppresses TTL for any table
/// whose retention ts-field isn't the partition source (e.g. logs on
/// `ingestion_time`, where `ref_max_event_micros` is structurally always None).
fn sidecar_hit(entry: Option<&Option<i64>>) -> Option<i64> {
    match entry {
        Some(Some(v)) => Some(*v),
        _ => None,
    }
}

/// Push a dropped data file AND its per-file puffin sidecar into the tombstone
/// set. A data file `X.parquet` carries a stats sidecar at `X.parquet.stats`
/// (see merge_puffin); it orphans when the data file is dropped, so tombstone
/// both. A file with no sidecar → the extra path is an idempotent no-op delete.
fn push_data_file_and_sidecar(paths: &mut Vec<String>, data_file_path: &str) {
    paths.push(format!("{data_file_path}.stats"));
    paths.push(data_file_path.to_string());
}

fn le_i32(b: &[u8]) -> Option<i64> {
    (b.len() >= 4).then(|| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64)
}
fn le_i64(b: &[u8]) -> Option<i64> {
    (b.len() >= 8).then(|| i64::from_le_bytes(b[..8].try_into().unwrap()))
}

/// Exclusive upper bound (micros) of event time covered by a partition
/// upper-bound value (Iceberg little-endian single-value encoding), for the
/// contiguous, monotonic time transforms. `None` for transforms whose bucket
/// doesn't map to a simple time range (month/year variable width, bucket,
/// truncate, …) — the caller then falls back to loading the manifest. Pure.
fn transform_upper_micros(transform: &Transform, bytes: &[u8]) -> Option<i64> {
    const HOUR_US: i64 = 3_600 * 1_000_000;
    const DAY_US: i64 = 86_400 * 1_000_000;
    match transform {
        // hour/day: value is the bucket ordinal from the epoch; the newest
        // possible event is the END of that bucket (exclusive).
        Transform::Hour => Some((le_i32(bytes)? + 1) * HOUR_US),
        Transform::Day => Some((le_i32(bytes)? + 1) * DAY_US),
        // identity on a timestamp column: the bound IS the max event time.
        Transform::Identity => le_i64(bytes),
        _ => None,
    }
}

/// Newest event time (micros, exclusive upper bound) a manifest can hold,
/// derived from its PARTITION SUMMARY for the `ts_field_id`-derived partition
/// field — WITHOUT loading the manifest. `None` when the timestamp isn't
/// partitioned by a supported time transform (caller falls back to the manifest).
/// This is what keeps graduation's classification O(refs) cheap reads instead of
/// O(refs) full manifest loads. Pure.
fn ref_max_event_micros(mf: &ManifestFile, spec: &PartitionSpec, ts_field_id: i32) -> Option<i64> {
    let parts = mf.partitions.as_ref()?;
    for (idx, pf) in spec.fields().iter().enumerate() {
        if pf.source_id != ts_field_id {
            continue;
        }
        let ub = parts.get(idx)?.upper_bound.as_ref()?;
        if let Some(m) = transform_upper_micros(&pf.transform, ub) {
            return Some(m);
        }
    }
    None
}

/// Outcome of a graduation fold when something was actually moved to cold.
pub(crate) struct FoldOutcome {
    /// Path of the freshly-written bucket-index (existing cold leaves + newly
    /// closed nodes/inline). The caller sets this as the new root's
    /// `bucket_index_path`.
    pub bucket_index_path: String,
    pub nodes_moved: usize,
    pub inline_leaves: usize,
    pub cold_leaves_total: usize,
    /// TTL-drop (Type 1 retention): S3 object paths that this fold removed from
    /// the tree because their newest `ts_field_id` value fell below the retention
    /// cutoff — the data files (and the manifest/leaf files that held them). The
    /// caller surfaces these to the sole tombstone writer (laminar) so they are
    /// reclaimed after grace. Empty unless a retention cutoff was supplied.
    pub ttl_dropped_paths: Vec<String>,
}

/// Per-node graduation decision, accounting for incremental delta path-tombstones
/// (`removed_paths`). A closed node is only safe to move to cold "by reference"
/// when none of its files were merged away — otherwise the cold tier keeps a
/// reference to a file the compaction GC will delete (the 2026-07-16 data-loss
/// bug). Pure + unit-tested; the caller performs the actual manifest I/O.
enum GraduatedNodePlan {
    /// No pending removal touches the node — graduate by cheap reference.
    ByReference,
    /// Every alive file was removed — drop the node (nothing enters cold).
    Skip,
    /// Some files removed — materialize a clean leaf from these kept entries.
    Materialize(Vec<ManifestEntry>),
}

/// Classify a graduating node's manifest entries against the pending removals.
/// Incremental deletes are delta path-tombstones (not MDVs), so a node can look
/// "clean" (`mdv == None`) yet still list merged-away files — this catches them.
fn plan_graduated_node(
    entries: &[Arc<ManifestEntry>],
    removed_paths: &HashSet<String>,
) -> GraduatedNodePlan {
    let alive: Vec<&Arc<ManifestEntry>> = entries.iter().filter(|e| e.is_alive()).collect();
    let has_removed = alive
        .iter()
        .any(|e| removed_paths.contains(e.data_file().file_path()));
    if !has_removed {
        return GraduatedNodePlan::ByReference;
    }
    let kept: Vec<ManifestEntry> = alive
        .iter()
        .filter(|e| !removed_paths.contains(e.data_file().file_path()))
        .map(|e| {
            ManifestEntry::builder()
                .status(ManifestStatus::Existing)
                .data_file(e.data_file().clone())
                .build()
        })
        .collect();
    if kept.is_empty() {
        GraduatedNodePlan::Skip
    } else {
        GraduatedNodePlan::Materialize(kept)
    }
}

/// Fold the "closed" entries of a reconstructed live set (those whose max
/// `ts_field_id` event time is below `cutoff_micros`) into the cold bucket-index,
/// returning the entries that stay hot (`kept`) and — if anything was moved — a
/// [`FoldOutcome`] with the updated bucket-index path.
///
/// This is the durable core of graduation: it reads any existing bucket-index at
/// `carried_bucket_index_path`, relocates closed clean nodes by reference,
/// materializes closed inline files into partition-tight cold leaves, and writes
/// a new bucket-index — but it does NOT write a root. The caller folds this into
/// its own root commit (a collapse, or a standalone base), so graduation is
/// atomic with the root rewrite and cannot be orphaned by a concurrent append.
/// Partition-spec-agnostic (keys purely on the time cutoff).
///
/// Returns `(kept, None)` when nothing is closed (caller keeps its carried
/// pointer and its entries unchanged). `manifest_counter`/`commit_uuid` are
/// threaded from the caller so cold-leaf manifests share the caller's namespace.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn fold_closed_into_bucket_index(
    table: &Table,
    entries: Vec<RootManifestEntry>,
    // Ref-resident pending removals (incremental delta path-tombstones from a
    // merge's `replace_data_files`). On incremental tables these are NOT recorded
    // as MDVs, so the `mdv.is_some()` dirty-check below can't see them. A
    // graduating node that still lists any of these is MATERIALIZED (its leaf
    // rewritten without them) before it enters cold — otherwise graduation would
    // carry merged-away, soon-GC-deleted files into the cold tier, leaving
    // dangling references that 404 after grace (the 2026-07-16 data-loss bug).
    removed_paths: &HashSet<String>,
    carried_bucket_index_path: Option<&str>,
    cutoff_micros: i64,
    ts_field_id: i32,
    max_graduate: Option<usize>,
    // TTL (Type 1 retention): entries/leaves whose newest `ts_field_id` value is
    // below this cutoff are DROPPED from the tree (not graduated, not kept) and
    // their object paths returned in `FoldOutcome::ttl_dropped_paths`. `None`
    // disables TTL (graduation-only fold). `max_ttl_drop_files` bounds how many
    // paths one fold may drop so the caller's snapshot summary stays small — the
    // rest drop on later collapses (eventually consistent).
    retention_cutoff_micros: Option<i64>,
    max_ttl_drop_files: usize,
    snapshot_id: i64,
    commit_uuid: Uuid,
    manifest_counter: &mut u64,
) -> Result<(Vec<RootManifestEntry>, Option<FoldOutcome>)> {
    let schema = table.metadata().current_schema().clone();
    let format_version = table.metadata().format_version();
    let spec = table.metadata().default_partition_spec().clone();
    let partition_type = spec.partition_type(&schema)?;
    let next_seq_num = table.metadata().next_sequence_number();

    // Sub-phase timers so `tiered collapse-fold sub-phase` can attribute the
    // fold wall-clock across (bi_load / ttl_prune / classify / write_bucket_index).
    // On sri-olly graduated=0/ttl_dropped=0 folds still spend ~6.5s per commit;
    // these break out where. `fallback_loads` counts the leaves/refs where
    // `ref_max_event_micros` returned None and we had to S3-read the manifest to
    // recover the timestamp — hot signal for "timestamp field isn't a partition
    // field" cases where the fast path is effectively bypassed on every entry.
    let fold_started = std::time::Instant::now();
    let mut ttl_prune_fallback_loads: usize = 0;
    let mut classify_fallback_loads: usize = 0;
    // Max-ts sidecar telemetry: how many TTL-prune leaves got their max_ts
    // from the persisted sidecar (0 S3 GETs) vs had to fall through to the
    // parallel-prefetch (~9ms/leaf). On a warm sidecar the fold should be
    // dominated by bi_load + write, with ttl_prune ~ same as bi_load.
    let mut sidecar_hits: usize = 0;
    let mut sidecar_misses: usize = 0;
    // Fresh max_ts values computed this fold — the new sidecar payload.
    // Populated regardless of outcome, so a no-op fold can also write the
    // sidecar the first time to warm it.
    let mut computed_max_ts: HashMap<String, Option<i64>> = HashMap::new();

    // Accumulates TTL-dropped object paths (data files + the manifest/leaf files
    // that referenced them), bounded by `max_ttl_drop_files`. `ttl_budget_left`
    // returns whether another leaf/entry may still be dropped this fold.
    let mut ttl_dropped_paths: Vec<String> = Vec::new();
    let ttl_budget_left =
        |dropped: &Vec<String>| retention_cutoff_micros.is_some() && dropped.len() < max_ttl_drop_files;

    // Existing cold leaves (graduated nodes get appended to these).
    let bi_load_start = std::time::Instant::now();
    let mut cold_leaves: Vec<ManifestFile> = match carried_bucket_index_path {
        Some(path) => {
            let b = table.file_io().new_input(path)?.read().await?;
            read_bucket_index(b)?.leaves().to_vec()
        }
        None => Vec::new(),
    };
    let bi_load_ms = bi_load_start.elapsed().as_millis() as u64;

    // Load the max-ts sidecar (if any) BEFORE the TTL-prune loop, so the
    // hot loop can hit it directly. Missing sidecar or a hit for a leaf
    // that's since been graduated-in-then-out remains fine — the caller
    // just falls through to the parallel fetch.
    let sidecar_max_ts: HashMap<String, Option<i64>> = match carried_bucket_index_path {
        Some(path) => load_maxts_sidecar(table.file_io(), path, ts_field_id)
            .await
            .unwrap_or_default(),
        None => HashMap::new(),
    };

    // ── TTL prune of the cold tier (steady-state path). Data ages into cold via
    // graduation long before it hits retention (retention ≫ bucket-window), so
    // expired data almost always lives here. Drop each cold leaf whose newest
    // `ts_field_id` value is below the retention cutoff; tombstone its data files
    // and the leaf manifest itself. Bounded by the file budget. ──
    let ttl_prune_start = std::time::Instant::now();
    if let Some(retention_cutoff) = retention_cutoff_micros {
        let cold_total = cold_leaves.len();
        let mut with_ts_count = 0usize;
        let mut none_ts_count = 0usize;
        let mut newest_max_ts: Option<i64> = None;
        let mut oldest_max_ts: Option<i64> = None;
        let mut surviving: Vec<ManifestFile> = Vec::with_capacity(cold_leaves.len());

        // Pre-fetch pass: parallel-load every cold leaf whose max_ts can't be
        // answered from the partition summary OR the max-ts sidecar. Before
        // the parallel fix this was one sequential `.await` per leaf inside
        // the reduce loop; before the sidecar it was O(all cold leaves) S3
        // GETs even in steady state. buffer_unordered lets S3 answer 32 at
        // once for whatever the sidecar didn't cover; the resulting hashmap
        // is a synchronous lookup in the same reduce loop.
        let prefetch: HashMap<String, Option<i64>> = {
            let mut needs_fetch: Vec<ManifestFile> = cold_leaves
                .iter()
                .filter(|leaf| {
                    // Order matters — cheapest checks first:
                    //   sidecar hit → skip;
                    //   partition summary fast path → skip;
                    //   otherwise fall into the parallel prefetch.
                    // Only a REAL cached value is a hit; a cached `None` falls
                    // through to re-derive (see `sidecar_hit`).
                    if sidecar_hit(sidecar_max_ts.get(&leaf.manifest_path)).is_some() {
                        sidecar_hits += 1;
                        return false;
                    }
                    if ref_max_event_micros(leaf, &spec, ts_field_id).is_some() {
                        return false;
                    }
                    sidecar_misses += 1;
                    true
                })
                .cloned()
                .collect();
            ttl_prune_fallback_loads = needs_fetch.len();
            // Cap the auto-heal batch. Coldest-first (oldest sequence_number)
            // so leaves most likely to be TTL-expired heal first — retention
            // starts dropping the oldest data immediately, even mid-heal. On
            // subsequent folds the sidecar has real values for the coldest
            // slice, so needs_fetch shrinks each cycle and the whole sidecar
            // converges within a handful of folds.
            if needs_fetch.len() > MAX_TTL_PREFETCH_LEAVES {
                needs_fetch.sort_by_key(|l| l.sequence_number);
                needs_fetch.truncate(MAX_TTL_PREFETCH_LEAVES);
            }
            let file_io = table.file_io();
            let mut map: HashMap<String, Option<i64>> = HashMap::with_capacity(needs_fetch.len());
            let mut s = stream_iter(needs_fetch.into_iter().map(|leaf| {
                let file_io = file_io.clone();
                async move {
                    let mx_res: Result<Option<i64>> = match leaf.load_manifest(&file_io).await {
                        Ok(manifest) => {
                            let files: Vec<DataFile> = manifest
                                .entries()
                                .iter()
                                .filter(|e| e.is_alive())
                                .map(|e| e.data_file().clone())
                                .collect();
                            Ok(max_ts_of(&files, ts_field_id))
                        }
                        Err(e) => Err(e),
                    };
                    (leaf.manifest_path.clone(), mx_res)
                }
            }))
            .buffer_unordered(FOLD_MANIFEST_FETCH_CONCURRENCY);
            while let Some((path, mx_res)) = s.next().await {
                map.insert(path, mx_res?);
            }
            map
        };

        // Coldest-first so the budget reclaims the oldest data before newer.
        let mut with_ts: Vec<(ManifestFile, Option<i64>)> = Vec::with_capacity(cold_leaves.len());
        for leaf in std::mem::take(&mut cold_leaves) {
            // Resolution order matches the filter above: sidecar → partition
            // summary → prefetched load. Record whichever wins into
            // `computed_max_ts` so the next fold can just hit the sidecar
            // for this leaf. Leaves resolved via prefetch are the only ones
            // that "cost" this fold; sidecar hits are pure in-memory.
            let mx = if let Some(v) = sidecar_hit(sidecar_max_ts.get(&leaf.manifest_path)) {
                Some(v)
            } else if let Some(m) = ref_max_event_micros(&leaf, &spec, ts_field_id) {
                Some(m)
            } else {
                prefetch.get(&leaf.manifest_path).copied().flatten()
            };
            computed_max_ts.insert(leaf.manifest_path.clone(), mx);
            match mx {
                Some(m) => {
                    with_ts_count += 1;
                    newest_max_ts = Some(newest_max_ts.map_or(m, |n| n.max(m)));
                    oldest_max_ts = Some(oldest_max_ts.map_or(m, |o| o.min(m)));
                }
                None => none_ts_count += 1,
            }
            with_ts.push((leaf, mx));
        }
        with_ts.sort_by_key(|(_, mx)| mx.unwrap_or(i64::MAX));
        let mut dropped_leaves = 0usize;
        for (leaf, mx) in with_ts {
            let expired = mx.map(|m| m < retention_cutoff).unwrap_or(false);
            if expired && ttl_budget_left(&ttl_dropped_paths) {
                let manifest = leaf.load_manifest(table.file_io()).await?;
                for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                    push_data_file_and_sidecar(&mut ttl_dropped_paths, e.data_file().file_path());
                }
                ttl_dropped_paths.push(leaf.manifest_path.clone());
                dropped_leaves += 1;
            } else {
                surviving.push(leaf);
            }
        }
        // Diagnostic: makes the TTL prune self-explaining. If dropped=0 despite
        // old data, this line disambiguates the cause: ts_field_id resolution
        // (none_ts high), poisoned bounds (oldest_max_ts recent/future), or an
        // already-pruned cold tier (cold_total low).
        log::info!(
            "ttl cold-leaf prune: ts_field_id={} cutoff_us={} cold_leaves={} with_ts={} none_ts={} oldest_max_ts={:?} newest_max_ts={:?} dropped_leaves={} dropped_paths={}",
            ts_field_id, retention_cutoff, cold_total, with_ts_count, none_ts_count,
            oldest_max_ts, newest_max_ts, dropped_leaves, ttl_dropped_paths.len()
        );
        cold_leaves = surviving;
    }
    let ttl_prune_ms = ttl_prune_start.elapsed().as_millis() as u64;

    // Partition into: closed live nodes (→ cold by reference), closed inline
    // files (→ materialize as cold leaves), and kept (stay hot). `closed_refs`
    // carries each node's newest event time so we can graduate the COLDEST
    // first when the per-collapse cap trims the batch.
    let classify_start = std::time::Instant::now();
    let entries_total = entries.len();
    let mut kept: Vec<RootManifestEntry> = Vec::new();
    let mut closed_refs: Vec<(ManifestFile, i64)> = Vec::new();
    let mut closed_inline_files: Vec<DataFile> = Vec::new();

    // Pre-fetch pass (mirrors the TTL-prune prefetch above): parallel-load
    // manifests for hot ManifestRef entries whose max_ts can't be answered
    // from the partition summary. Same rationale — sri-olly had 143/153 hot
    // entries falling back, driving 5.6s of classify wall on a no-op fold.
    // Inline entries need no S3 and MDV-carrying refs are kept unconditionally,
    // so we filter to the exact set that needs a manifest load.
    let classify_prefetch: HashMap<String, Option<i64>> = {
        let needs_fetch: Vec<ManifestFile> = entries
            .iter()
            .filter_map(|e| match e {
                RootManifestEntry::ManifestRef { manifest_file, mdv } if mdv.is_none() => {
                    if ref_max_event_micros(manifest_file, &spec, ts_field_id).is_none() {
                        Some(manifest_file.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        classify_fallback_loads = needs_fetch.len();
        let file_io = table.file_io();
        let mut map: HashMap<String, Option<i64>> = HashMap::with_capacity(needs_fetch.len());
        let mut s = stream_iter(needs_fetch.into_iter().map(|mf| {
            let file_io = file_io.clone();
            async move {
                let mx_res: Result<Option<i64>> = match mf.load_manifest(&file_io).await {
                    Ok(manifest) => {
                        let files: Vec<DataFile> = manifest
                            .entries()
                            .iter()
                            .filter(|e| e.is_alive())
                            .map(|e| e.data_file().clone())
                            .collect();
                        Ok(max_ts_of(&files, ts_field_id))
                    }
                    Err(e) => Err(e),
                };
                (mf.manifest_path.clone(), mx_res)
            }
        }))
        .buffer_unordered(FOLD_MANIFEST_FETCH_CONCURRENCY);
        while let Some((path, mx_res)) = s.next().await {
            map.insert(path, mx_res?);
        }
        map
    };

    for entry in entries {
        match entry {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                // Only graduate clean nodes; an MDV-carrying node has pending
                // deletes and is left for the rebalance/compaction path.
                if mdv.is_some() {
                    kept.push(RootManifestEntry::ManifestRef { manifest_file, mdv });
                    continue;
                }
                // Fast path: newest event time straight from the partition
                // summary (no manifest read). Fall back to the prefetch
                // hashmap (parallel-loaded upstream) when the ts field isn't
                // partitioned by a time transform.
                let max_micros = match ref_max_event_micros(&manifest_file, &spec, ts_field_id) {
                    Some(m) => Some(m),
                    None => classify_prefetch
                        .get(&manifest_file.manifest_path)
                        .copied()
                        .flatten(),
                };
                // Expired past retention (edge case for hot data: normally it
                // graduates to cold long before this). Drop + tombstone rather
                // than graduate. `retention_cutoff < cutoff_micros`, so an expired
                // ref is always also "closed".
                let expired = matches!((retention_cutoff_micros, max_micros),
                    (Some(rc), Some(mx)) if mx < rc);
                if expired && ttl_budget_left(&ttl_dropped_paths) {
                    let manifest = manifest_file.load_manifest(table.file_io()).await?;
                    for e in manifest.entries().iter().filter(|e| e.is_alive()) {
                        push_data_file_and_sidecar(&mut ttl_dropped_paths, e.data_file().file_path());
                    }
                    ttl_dropped_paths.push(manifest_file.manifest_path.clone());
                    continue;
                }
                match max_micros {
                    Some(mx) if mx < cutoff_micros => closed_refs.push((manifest_file, mx)),
                    _ => kept.push(RootManifestEntry::ManifestRef { manifest_file, mdv }),
                }
            }
            RootManifestEntry::Inline(me) => {
                let mx = file_max_ts(&me.data_file, ts_field_id);
                let expired = matches!((retention_cutoff_micros, mx),
                    (Some(rc), Some(m)) if m < rc);
                if expired && ttl_budget_left(&ttl_dropped_paths) {
                    push_data_file_and_sidecar(&mut ttl_dropped_paths, me.data_file.file_path());
                    continue;
                }
                let closed = mx.map(|m| m < cutoff_micros).unwrap_or(false);
                if closed {
                    closed_inline_files.push(me.data_file.clone());
                } else {
                    kept.push(RootManifestEntry::Inline(me));
                }
            }
        }
    }

    // Bound the graduation per collapse. On a long-history table the first
    // graduation can otherwise move most of the table in one commit; cap it and
    // graduate the COLDEST refs first — the rest stay hot and graduate on later
    // collapses (eventually consistent, bounded per-commit work).
    if let Some(cap) = max_graduate {
        if closed_refs.len() > cap {
            closed_refs.sort_by_key(|(_, mx)| *mx);
            for (mf, _) in closed_refs.split_off(cap) {
                kept.push(RootManifestEntry::ManifestRef {
                    manifest_file: mf,
                    mdv: None,
                });
            }
        }
    }
    let graduated_nodes: Vec<ManifestFile> = closed_refs.into_iter().map(|(mf, _)| mf).collect();

    let classify_ms = classify_start.elapsed().as_millis() as u64;

    // Nothing to do only if no graduation AND no TTL prune happened. A TTL-only
    // fold (cold leaves dropped, nothing graduated) still must persist the
    // pruned bucket-index and return the dropped paths.
    if graduated_nodes.is_empty()
        && closed_inline_files.is_empty()
        && ttl_dropped_paths.is_empty()
    {
        // Opportunistically warm the max-ts sidecar for the CURRENT
        // bucket-index if it's missing / incomplete. Steady-state folds
        // don't rewrite the bucket-index so the sidecar would otherwise
        // never populate; this covers the first fold post-deploy and any
        // fold where the earlier write failed. Skipped when the sidecar
        // already covers every cold leaf (no work needed).
        let mut sidecar_write_ms: u64 = 0;
        if let Some(carried) = carried_bucket_index_path {
            // A leaf is "covered" only by a REAL cached value; a `None` entry is
            // poison and must be rewritten with a real value if we now have one.
            let covers_all = !cold_leaves.is_empty()
                && cold_leaves.iter().all(|leaf| {
                    sidecar_hit(sidecar_max_ts.get(&leaf.manifest_path)).is_some()
                });
            if !covers_all {
                let mut new_sidecar: HashMap<String, Option<i64>> =
                    HashMap::with_capacity(cold_leaves.len());
                for leaf in &cold_leaves {
                    // Only persist a REAL max-ts. Caching a `None` poisons the
                    // sidecar (the reader would trust it and never re-derive); an
                    // absent leaf is correctly re-fetched by the next fold.
                    if let Some(v) = computed_max_ts
                        .get(&leaf.manifest_path)
                        .copied()
                        .flatten()
                        .or_else(|| ref_max_event_micros(leaf, &spec, ts_field_id))
                    {
                        new_sidecar.insert(leaf.manifest_path.clone(), Some(v));
                    }
                }
                let write_start = std::time::Instant::now();
                if let Err(e) =
                    write_maxts_sidecar(table.file_io(), carried, ts_field_id, new_sidecar).await
                {
                    log::warn!(
                        "tiered collapse-fold: opportunistic max-ts sidecar warm failed (non-fatal): {e}"
                    );
                }
                sidecar_write_ms = write_start.elapsed().as_millis() as u64;
            }
        }
        log::info!(
            "tiered collapse-fold sub-phase: total_ms={} bi_load_ms={} ttl_prune_ms={} classify_ms={} write_ms={} entries={} cold_leaves={} ttl_fallback_loads={} classify_fallback_loads={} sidecar_hits={} sidecar_misses={} outcome=none",
            fold_started.elapsed().as_millis() as u64,
            bi_load_ms,
            ttl_prune_ms,
            classify_ms,
            sidecar_write_ms,
            entries_total,
            cold_leaves.len(),
            ttl_prune_fallback_loads,
            classify_fallback_loads,
            sidecar_hits,
            sidecar_misses,
        );
        return Ok((kept, None));
    }
    let nodes_moved = graduated_nodes.len();

    // Materialize any closed inline files into new partition-tight cold leaves.
    let mut inline_leaves = 0usize;
    if !closed_inline_files.is_empty() {
        let grad_entries: Vec<ManifestEntry> = closed_inline_files
            .into_iter()
            .map(|df| {
                ManifestEntry::builder()
                    .status(ManifestStatus::Existing)
                    .data_file(df)
                    .build()
            })
            .collect();
        let new_leaves = write_entries_clustered(
            table,
            &schema,
            spec.as_ref(),
            format_version,
            snapshot_id,
            commit_uuid,
            manifest_counter,
            false,
            grad_entries,
            true,
        )
        .await?;
        inline_leaves = new_leaves.len();
        cold_leaves.extend(new_leaves);
    }
    // Move graduated live nodes into cold. The "a closed node is immutable"
    // assumption only holds when nothing has been removed from it. On incremental
    // tables a merge's deletes are delta path-tombstones (`removed_paths`), not
    // MDVs, so a node whose small files were merged away still lists them and is
    // NOT caught by the `mdv.is_some()` check above. Moving it by reference would
    // carry those merged-away (soon-GC-deleted) files into the cold tier ⇒
    // dangling refs ⇒ 404 after grace. So: a graduating node with any pending
    // removal is MATERIALIZED here (leaf rewritten keeping only non-removed alive
    // entries). Nodes with no pending removal still move by cheap reference.
    if removed_paths.is_empty() {
        cold_leaves.extend(graduated_nodes);
    } else {
        for mf in graduated_nodes {
            // Best-effort: a load failure (e.g. an already-corrupt/absent
            // manifest) must not fail the whole collapse commit — fall back to
            // moving by reference (preserving prior behavior for that node).
            let manifest = match mf.load_manifest(table.file_io()).await {
                Ok(m) => m,
                Err(e) => {
                    log::warn!(
                        "graduation: could not load {} to materialize removals, moving by reference: {e}",
                        mf.manifest_path
                    );
                    cold_leaves.push(mf);
                    continue;
                }
            };
            match plan_graduated_node(manifest.entries(), removed_paths) {
                // Clean node → move by reference (fast path preserved).
                GraduatedNodePlan::ByReference => cold_leaves.push(mf),
                // Every file merged away → fully orphaned manifest; drop it (its
                // data files are already tombstoned by the merge). The stale
                // manifest object leaks; the reachability backstop reclaims it.
                GraduatedNodePlan::Skip => {}
                // Mixed → materialize a clean cold leaf without the removed files.
                GraduatedNodePlan::Materialize(kept) => {
                    let rewritten = write_entries_clustered(
                        table,
                        &schema,
                        spec.as_ref(),
                        format_version,
                        snapshot_id,
                        commit_uuid,
                        manifest_counter,
                        false,
                        kept,
                        true,
                    )
                    .await?;
                    cold_leaves.extend(rewritten);
                }
            }
        }
    }

    let bucket_index_path = format!(
        "{}/{}/bucket-index-{}-{}.parquet",
        table.metadata().location(),
        META_ROOT_PATH,
        snapshot_id,
        commit_uuid,
    );
    let bi_metadata = RootManifestMetadata {
        schema: schema.clone(),
        schema_id: table.metadata().current_schema_id(),
        partition_spec: spec.clone(),
        format_version: FormatVersion::V4,
        snapshot_id,
        sequence_number: next_seq_num,
        parent_snapshot_id: table.metadata().current_snapshot_id(),
        bucket_index_path: None,
        prev_root_path: None,
        chain_depth: 0,
        node_level: 0,
        removed_paths: Vec::new(),
    };
    let write_start = std::time::Instant::now();
    let bi_bytes = write_bucket_index(&cold_leaves, &bi_metadata, &partition_type)?;
    table
        .file_io()
        .new_output(&bucket_index_path)?
        .write(bi_bytes.into())
        .await?;
    // Refresh the sidecar to cover the new bucket-index. `computed_max_ts` is
    // keyed by ORIGINAL cold-leaf paths (those we ran the ttl-prune loop
    // over); leaves that were dropped by ttl are naturally excluded from the
    // new cold_leaves so we don't leak stale entries. Fold-in the values for
    // newly graduated leaves too, computed from their partition summaries
    // when available. Non-fatal on write failure — next fold will just miss
    // the sidecar and fall back to the prefetch path.
    let mut new_sidecar: HashMap<String, Option<i64>> = HashMap::with_capacity(cold_leaves.len());
    for leaf in &cold_leaves {
        // Only persist a REAL max-ts (see the ttl-prune reader): a cached `None`
        // is a non-answer that would poison the next fold, so skip it and let the
        // reader re-derive that leaf.
        if let Some(v) = computed_max_ts
            .get(&leaf.manifest_path)
            .copied()
            .flatten()
            .or_else(|| ref_max_event_micros(leaf, &spec, ts_field_id))
        {
            new_sidecar.insert(leaf.manifest_path.clone(), Some(v));
        }
    }
    if let Err(e) =
        write_maxts_sidecar(table.file_io(), &bucket_index_path, ts_field_id, new_sidecar).await
    {
        log::warn!(
            "tiered collapse-fold: max-ts sidecar write failed (non-fatal): {e}"
        );
    }
    let write_ms = write_start.elapsed().as_millis() as u64;

    let cold_leaves_total = cold_leaves.len();
    log::info!(
        "tiered collapse-fold sub-phase: total_ms={} bi_load_ms={} ttl_prune_ms={} classify_ms={} write_ms={} entries={} cold_leaves={} nodes_moved={} inline_leaves={} ttl_fallback_loads={} classify_fallback_loads={} sidecar_hits={} sidecar_misses={} outcome=changed",
        fold_started.elapsed().as_millis() as u64,
        bi_load_ms,
        ttl_prune_ms,
        classify_ms,
        write_ms,
        entries_total,
        cold_leaves_total,
        nodes_moved,
        inline_leaves,
        ttl_prune_fallback_loads,
        classify_fallback_loads,
        sidecar_hits,
        sidecar_misses,
    );
    Ok((
        kept,
        Some(FoldOutcome {
            bucket_index_path,
            nodes_moved,
            inline_leaves,
            cold_leaves_total,
            ttl_dropped_paths,
        }),
    ))
}

impl GraduateBucketsAction {
    /// Heavy read + write path (equivalent of phases 1-5 in the
    /// compact_cold_tier breakdown): reconstruct root, fold closed entries
    /// into the cold bucket-index (which reads existing cold leaves +
    /// writes new leaves + writes new bucket-index parquet). Runs at most
    /// once per action instance (result cached in `self.prepared`) unless
    /// invalidated by a concurrent cold-tier writer.
    async fn prepare(&self, table: &Table) -> Result<Option<PreparedGraduation>> {
        let current_snapshot = table.metadata().current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "graduate_buckets: table has no current snapshot",
            )
        })?;

        let root_path = current_snapshot.manifest_list();
        let (rm_metadata, entries) = reconstruct_root(table.file_io(), root_path).await?;
        let prep_bucket_index_path = rm_metadata.bucket_index_path.clone();

        // Allocate ONCE at prep — reused across retries so the leaf manifests
        // (which embed snapshot_id) keep consistent lineage.
        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let commit_uuid = self.commit_uuid;
        let mut manifest_counter: u64 = 0;

        // Fold closed entries into the cold bucket-index. Returns the entries
        // that stay hot plus the updated bucket-index; None ⇒ nothing closed
        // this pass.
        let removed_set: HashSet<String> =
            rm_metadata.removed_paths.iter().cloned().collect();
        let (kept, fold) = fold_closed_into_bucket_index(
            table,
            entries,
            &removed_set,
            rm_metadata.bucket_index_path.as_deref(),
            self.cutoff_micros,
            self.ts_field_id,
            None, // manual/maintenance use: no per-collapse cap
            None, // TTL retention drop is driven only by the commit_v4 collapse
            0,
            snapshot_id,
            commit_uuid,
            &mut manifest_counter,
        )
        .await?;
        let fold = match fold {
            Some(f) => f,
            None => return Ok(None),
        };

        Ok(Some(PreparedGraduation {
            kept,
            bucket_index_path: fold.bucket_index_path,
            prep_bucket_index_path,
            ancestor_removed_paths: rm_metadata.removed_paths,
            snapshot_id,
            nodes_moved: fold.nodes_moved,
            inline_leaves: fold.inline_leaves,
            cold_leaves_total: fold.cold_leaves_total,
        }))
    }

    /// Phase 6 equivalent: write the new (delta or flat) root and build the
    /// ActionCommit with fresh `RefSnapshotIdMatch` for the current table
    /// head. Runs on every `commit()` attempt. Cost: ~500 ms on the delta
    /// path (metadata-only parquet write), or flat-base serialize+PUT when
    /// `chain_depth >= MAX_CHAIN`.
    async fn finalize(
        &self,
        table: &Table,
        prep: &PreparedGraduation,
    ) -> Result<ActionCommit> {
        let current_snapshot = table.metadata().current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "graduate_buckets: table has no current snapshot at finalize",
            )
        })?;
        let root_path = current_snapshot.manifest_list();
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(&schema)?;

        // Read HEAD root metadata only (no chain walk) — needed for
        // chain_depth (delta-vs-flat decision). `read_root_manifest` reads
        // just the one parquet file at `root_path`, not the whole chain.
        let head_bytes = table.file_io().new_input(root_path)?.read().await?;
        let (head_meta, _head_entries) = read_root_manifest(head_bytes)?;

        // Delta vs flat-base root. On a `root-manifest.incremental` table,
        // emit the swap as a small delta (`prev_root_path=Some(current)`,
        // `chain_depth+=1`, empty entries slice) so `actions_ms` drops from
        // the flat-root serialize + reconstruct-collapse cost to a few-KB
        // metadata-only write. This is what stops the retry loop from losing
        // every CAS race vs. laminar's hot-append cadence — the mirror of
        // the compact_cold_tier fix in commits c7c8ffc + 6d64deb.
        //
        // At `chain_depth == MAX_CHAIN`, fall through to the flat-base
        // branch — same collapse semantics commit_v4 uses at the cap. When
        // the table lacks the incremental property (default off), the flat
        // branch is preserved for parity with pre-delta behavior.
        let incremental = table
            .metadata()
            .properties()
            .get("root-manifest.incremental")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let do_delta = incremental && head_meta.chain_depth < MAX_CHAIN;

        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id: prep.snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: Some(prep.bucket_index_path.clone()),
            prev_root_path: if do_delta {
                Some(root_path.to_string())
            } else {
                None
            },
            chain_depth: if do_delta {
                head_meta.chain_depth + 1
            } else {
                0
            },
            node_level: 0,
            // Delta: no new tombstones from this action (graduation only moves
            // refs into the cold tier, doesn't remove data files). Ancestor
            // tombstones live on prior roots and are unioned by
            // reconstruct_root — do NOT copy them onto the delta.
            //
            // Flat-base fallback: carry forward the merged ancestor set from
            // prep, same as pre-fix behavior. (Slight staleness under
            // prep-reuse — ancestor tombstones added by concurrent writers
            // between prep and this collapse won't be captured — but the
            // fast-path only activates when bucket_index_path is unchanged,
            // which under laminar-only writes matches the tombstone-add
            // invariant too.)
            removed_paths: if do_delta {
                Vec::new()
            } else {
                prep.ancestor_removed_paths.clone()
            },
        };
        // Fresh root path per attempt (retries write their own delta root;
        // failed attempts leave orphans reclaimed later by tombstone GC).
        let new_root_path = format!(
            "{}/{}/root-{}-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            prep.snapshot_id,
            self.commit_uuid,
            Uuid::now_v7(),
        );
        let entries_to_write: &[RootManifestEntry] = if do_delta { &[] } else { &prep.kept };
        let root_bytes = write_root_manifest(entries_to_write, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_path)?
            .write(root_bytes.into())
            .await?;

        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "graduate-nodes-moved".to_string(),
                    prep.nodes_moved.to_string(),
                ),
                (
                    "graduate-inline-leaves".to_string(),
                    prep.inline_leaves.to_string(),
                ),
                (
                    "graduate-cold-leaves-total".to_string(),
                    prep.cold_leaves_total.to_string(),
                ),
            ]),
        };

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let first_row_id = table.metadata().next_row_id();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(new_root_path.clone())
            .with_snapshot_id(prep.snapshot_id)
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
                    prep.snapshot_id,
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

        Ok(ActionCommit::new(updates, requirements)
            .with_manifest_paths(vec![new_root_path, prep.bucket_index_path.clone()]))
    }
}

#[async_trait]
impl TransactionAction for GraduateBucketsAction {
    fn action_name(&self) -> &'static str {
        "graduate_buckets"
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "graduate_buckets requires format version V4 (effective={:?})",
                    table.effective_format_version()
                ),
            ));
        }
        if table.metadata().current_snapshot().is_none() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // Fast-path check: if we have a cached prep AND the table's HEAD
        // still points at the same cold bucket-index (i.e., no other
        // cold-tier writer landed between prep and this retry), we can skip
        // phases 1-5 entirely and only re-emit the small delta root. That's
        // what makes retries win the OCC race — the critical section drops
        // from ~20 s (full prep) to ~500 ms (root write only).
        //
        // The check is a single S3 GET of the HEAD root parquet (no chain
        // walk) — cheap even when it invalidates.
        let mut guard = self.prepared.lock().await;

        // Reusable only if the cold bucket-index pointer hasn't shifted
        // AND we're still on the delta-writable side of the chain-depth cap
        // — MAX_CHAIN triggers a flat-base rewrite that needs a fresh
        // `kept` reconstruct; cached prep's `kept` is as-of-prep-time.
        let can_reuse = if let Some(ref prep) = *guard {
            let current_snapshot = table.metadata().current_snapshot().unwrap();
            let head_bytes = table
                .file_io()
                .new_input(current_snapshot.manifest_list())?
                .read()
                .await?;
            let (head_meta, _) = read_root_manifest(head_bytes)?;
            head_meta.bucket_index_path == prep.prep_bucket_index_path
                && head_meta.chain_depth < MAX_CHAIN
        } else {
            false
        };

        if can_reuse {
            let prep = guard.as_ref().unwrap();
            return self.finalize(table, prep).await;
        }

        // Slow path: full prep. Cache the result before finalizing so
        // subsequent retries (of this action instance) take the fast path.
        let prep = match self.prepare(table).await? {
            Some(p) => p,
            // Nothing to graduate this pass — emit an empty ActionCommit;
            // don't cache (nothing to cache).
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };
        let ac = self.finalize(table, &prep).await?;
        *guard = Some(prep);
        Ok(ac)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use super::*;
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Datum, Struct};

    fn df(path: &str, ts_max: Option<(i32, i64)>) -> DataFile {
        let mut b = DataFileBuilder::default();
        b.content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty());
        if let Some((fid, micros)) = ts_max {
            b.upper_bounds(HashMap::from([(fid, Datum::timestamp_micros(micros))]));
        }
        b.build().unwrap()
    }

    fn alive_entry(path: &str) -> Arc<ManifestEntry> {
        Arc::new(
            ManifestEntry::builder()
                .status(ManifestStatus::Existing)
                .data_file(df(path, None))
                .build(),
        )
    }

    fn removed(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|p| p.to_string()).collect()
    }

    // No pending removal touches the node → cheap by-reference graduation.
    #[test]
    fn plan_by_reference_when_nothing_removed() {
        let entries = vec![alive_entry("a"), alive_entry("b")];
        // empty removal set, and a non-matching removal set, both → ByReference
        assert!(matches!(
            plan_graduated_node(&entries, &removed(&[])),
            GraduatedNodePlan::ByReference
        ));
        assert!(matches!(
            plan_graduated_node(&entries, &removed(&["x"])),
            GraduatedNodePlan::ByReference
        ));
    }

    // Some files merged away → materialize a leaf with ONLY the survivors. This is
    // the core 2026-07-16 fix: the removed file must never reach the cold tier.
    #[test]
    fn plan_materializes_without_removed_files() {
        let entries = vec![alive_entry("a"), alive_entry("b"), alive_entry("c")];
        match plan_graduated_node(&entries, &removed(&["b"])) {
            GraduatedNodePlan::Materialize(kept) => {
                let paths: Vec<&str> = kept.iter().map(|e| e.data_file().file_path()).collect();
                assert_eq!(paths, vec!["a", "c"], "removed file 'b' must be dropped");
            }
            other => panic!("expected Materialize, got {:?}", plan_name(&other)),
        }
    }

    // Every alive file removed → drop the node entirely (nothing enters cold).
    #[test]
    fn plan_skips_when_all_removed() {
        let entries = vec![alive_entry("a"), alive_entry("b")];
        assert!(matches!(
            plan_graduated_node(&entries, &removed(&["a", "b"])),
            GraduatedNodePlan::Skip
        ));
    }

    fn plan_name(p: &GraduatedNodePlan) -> &'static str {
        match p {
            GraduatedNodePlan::ByReference => "ByReference",
            GraduatedNodePlan::Skip => "Skip",
            GraduatedNodePlan::Materialize(_) => "Materialize",
        }
    }

    // The TTL-prune poison-tolerance fix: only a REAL cached value counts as a
    // sidecar hit. A present-but-`None` entry (poison from an older writer) and
    // an absent entry both resolve to a MISS so the caller re-derives — this is
    // what lets an all-`None` logs sidecar self-heal instead of suppressing TTL
    // forever.
    #[test]
    fn sidecar_hit_only_trusts_real_values() {
        let real = Some(900i64);
        let poison = None::<i64>;
        assert_eq!(sidecar_hit(Some(&real)), Some(900), "real value → hit");
        assert_eq!(sidecar_hit(Some(&poison)), None, "cached None → miss (poison)");
        assert_eq!(sidecar_hit(None), None, "absent → miss");
    }

    #[test]
    fn max_ts_takes_newest() {
        let fid = 5;
        let files = vec![
            df("a", Some((fid, 100))),
            df("b", Some((fid, 900))),
            df("c", Some((fid, 500))),
        ];
        assert_eq!(max_ts_of(&files, fid), Some(900));
    }

    #[test]
    fn max_ts_none_without_stat() {
        assert_eq!(max_ts_of(&[df("a", None)], 5), None);
    }

    #[test]
    fn file_max_ts_reads_field() {
        assert_eq!(file_max_ts(&df("a", Some((5, 42))), 5), Some(42));
        assert_eq!(file_max_ts(&df("a", Some((5, 42))), 9), None);
    }

    #[test]
    fn transform_upper_micros_decodes_time_buckets() {
        const H: i64 = 3_600 * 1_000_000;
        const D: i64 = 86_400 * 1_000_000;
        // hour ordinal 10 → exclusive upper = END of hour 10 = hour 11 start.
        assert_eq!(
            transform_upper_micros(&Transform::Hour, &10i32.to_le_bytes()),
            Some(11 * H)
        );
        // day ordinal 3 → end of day 3 = day 4 start.
        assert_eq!(
            transform_upper_micros(&Transform::Day, &3i32.to_le_bytes()),
            Some(4 * D)
        );
        // identity on a timestamp column: bound IS the max event time (micros).
        assert_eq!(
            transform_upper_micros(&Transform::Identity, &1_700_000_000_000_000i64.to_le_bytes()),
            Some(1_700_000_000_000_000)
        );
        // variable-width / non-time transforms fall back (None → load manifest).
        assert_eq!(
            transform_upper_micros(&Transform::Month, &10i32.to_le_bytes()),
            None
        );
        // truncated/garbage bytes → None (safe fallback), never a bogus bound.
        assert_eq!(transform_upper_micros(&Transform::Hour, &[1u8, 2]), None);
        assert_eq!(transform_upper_micros(&Transform::Identity, &[1u8, 2, 3, 4]), None);
    }
}
