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

//! Compact the **cold tier** of a tiered V4 table (tessellate's job).
//!
//! The hot path keeps the live bucket inline in the root; `graduate_buckets`
//! moves closed buckets into immutable leaf manifests under the bucket-index.
//! Over time a cold partition accumulates many small *data files* (one per live
//! rotation), which makes cold scans slow. The fix is data-file compaction —
//! but the cold files live under the bucket-index, not in the root, so the hot
//! `replace_data_files` (which only touches the root's entries) can't reach
//! them.
//!
//! This action is the cold-tier equivalent, run by the compaction service
//! (tessellate), **off the hot commit path**. The caller does the actual data
//! merge (read N small parquet → write 1 big parquet via streaming concat) and
//! passes the result here:
//!
//!   - `removed`: file paths of the small cold files that were merged away,
//!   - `added`:   the merged replacement `DataFile`s (already written to S3).
//!
//! The action rewrites only the **affected** leaf manifests (those containing a
//! removed file) — dropping the removed entries and re-clustering the survivors
//! plus the merged files per partition — carries unaffected leaves forward by
//! reference, writes a new bucket-index, and points the root at it. The live
//! tier (root inline + refs) is untouched.
//!
//! v1 loads every leaf manifest to find the affected ones. That is read-only
//! and runs at the compaction-service cadence (not per commit); a future
//! optimization can target leaves by partition or a file→leaf index.
//!
//! # Prep vs. CAS split
//!
//! The heavy read + write work (reconstruct_root chain-walk, serial cold-leaf
//! manifest load, cluster+write new leaves + bucket-index) is captured in a
//! `PreparedCompaction` cached inside the action. On the first `commit()`
//! attempt we run full prep — that produces S3 files addressed by
//! UUID-in-path (immutable, reusable). On subsequent attempts (Transaction's
//! CAS retry loop refreshes the table and calls `commit()` again), we skip
//! prep entirely if the current head root's `bucket_index_path` still equals
//! the one prep saw. Only the small delta root gets rewritten, referencing
//! the current head — a ~500 ms critical section instead of the ~20 s of
//! phases 1-5. This is what actually lets `compact_cold_tier` win the CAS
//! race against laminar's ~15-30 s hot-append cadence — fix #1's delta root
//! made the write cheap, this split makes the retry cheap.
//!
//! Under sri-olly's Forbid-concurrency CronJob, only ONE tessellate instance
//! runs at a time, so between prep and retry the ONLY thing that can change
//! is laminar's hot appends — which don't touch the cold tier. The
//! bucket_index_path fingerprint check is therefore essentially always
//! satisfied. If it ever isn't (e.g., another cold-tier writer landed in
//! between — a rare cross-tick collision), we discard the cached prep (its
//! S3 files become orphans, reclaimed by tombstone GC eventually) and
//! re-prep from scratch.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{iter as stream_iter, StreamExt};
use tokio::sync::Mutex;
use uuid::Uuid;

use super::rebalance_root_manifest::write_entries_clustered;
use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    read_root_manifest, reconstruct_root, write_root_manifest, RootManifestEntry,
    RootManifestMetadata,
};
use crate::spec::{
    DataFile, Datum, FormatVersion, Literal, ManifestEntry, ManifestFile, ManifestStatus,
    Operation, Snapshot, SnapshotReference, SnapshotRetention, Struct, StructType, Summary, Type,
    MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::{max_chain_depth, SnapshotProducer};
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Cached prep from a prior `commit()` attempt on the same
/// [`CompactColdTierAction`] instance. See the module doc for the retry-cheap
/// invariant. All heavy S3-resident outputs (new leaves, new bucket-index)
/// are UUID-addressed and safe to reuse across retries; the only per-retry
/// work is writing the small delta root that points at them.
struct PreparedCompaction {
    /// The freshly-written cold bucket-index (or `None` if cold tier is now
    /// empty). Path is unique per (snapshot_id, commit_uuid), so subsequent
    /// retries reuse the same file.
    new_bucket_index_path: Option<String>,
    /// The bucket-index path the table's head root pointed at when prep ran.
    /// Fast-path validity: if the current head still points here, no other
    /// cold-tier writer landed in between and the cached prep is valid.
    prep_bucket_index_path: Option<String>,
    /// Snapshot id allocated once at prep time and reused across retries so
    /// the leaf manifests we wrote (which embed this id) have consistent
    /// lineage regardless of which retry ultimately wins the CAS.
    snapshot_id: i64,
    /// Root entries reconstructed at prep time — needed only for the flat-base
    /// fallback path (when `chain_depth` hits `MAX_CHAIN`). Cheap to hold via
    /// `Arc` so `commit()` retries don't pay a `Vec::clone` on the fast path.
    root_entries: Arc<Vec<RootManifestEntry>>,
    /// Merged ancestor tombstones from prep-time `reconstruct_root`, needed
    /// only for the flat-base fallback path (delta path emits empty
    /// `removed_paths` — see fix #1).
    ancestor_removed_paths: Arc<Vec<String>>,
    /// Set when prep found at least one affected leaf. Cached so retries can
    /// short-circuit the "no-op" branch without redoing the leaf scan.
    any_affected: bool,
}

/// Action that compacts the cold tier: replaces `removed` data files (already
/// merged into `added` by the caller) within the bucket-index's leaf manifests.
///
/// Use via `Transaction::compact_cold_tier()`.
pub struct CompactColdTierAction {
    removed: HashSet<String>,
    added: Vec<DataFile>,
    commit_uuid: Uuid,
    snapshot_id_override: Option<i64>,
    /// Cached prep for retry cheapness. `None` before the first `commit()`
    /// call, populated after prep runs. `tokio::sync::Mutex` because the
    /// Transaction commit loop uses `Arc<Self>` and holds the lock across
    /// async I/O.
    prepared: Mutex<Option<PreparedCompaction>>,
}

impl CompactColdTierAction {
    /// Create an empty cold-compaction action (no removals or additions yet).
    pub fn new() -> Self {
        Self {
            removed: HashSet::new(),
            added: Vec::new(),
            commit_uuid: Uuid::now_v7(),
            snapshot_id_override: None,
            prepared: Mutex::new(None),
        }
    }

    /// Mark cold data-file paths to remove (the small files merged away).
    pub fn remove_files(mut self, paths: impl IntoIterator<Item = String>) -> Self {
        self.removed.extend(paths);
        self
    }

    /// Add the merged replacement data files (already written to storage).
    pub fn add_files(mut self, files: impl IntoIterator<Item = DataFile>) -> Self {
        self.added.extend(files);
        self
    }

    /// Pre-allocate the new snapshot's id, overriding the random id that
    /// `commit()` would otherwise generate. Mirror of
    /// [`super::replace_data_files::ReplaceDataFilesAction::with_snapshot_id`].
    /// Same use case: referencing the snapshot_id elsewhere in the same
    /// transaction — most commonly to attach per-output-file `StatisticsFile`
    /// entries (per-file `.parquet.stats` puffin sidecars) to the new snapshot
    /// so the executor's `metadata.statistics_for_snapshot(current)` surfaces
    /// them. Without this, laminar's cold-redirect of a graduation-crossed
    /// merge cannot register the merged output's carry-forward sidecar in
    /// the same commit and the executor falls back to path-convention
    /// (`<parquet>.stats`) lookup for that one merged output.
    pub fn with_snapshot_id(mut self, snapshot_id: i64) -> Self {
        self.snapshot_id_override = Some(snapshot_id);
        self
    }
}

impl Default for CompactColdTierAction {
    fn default() -> Self {
        Self::new()
    }
}

/// Filter a leaf's data files against the removal set. Returns the survivors and
/// whether any file was removed (i.e. this leaf must be rewritten). Pure.
fn apply_removals(files: Vec<DataFile>, removed: &HashSet<String>) -> (Vec<DataFile>, bool) {
    let mut survivors = Vec::with_capacity(files.len());
    let mut affected = false;
    for f in files {
        if removed.contains(&f.file_path) {
            affected = true;
        } else {
            survivors.push(f);
        }
    }
    (survivors, affected)
}

/// Concurrency for the cold-leaf load that survives the partition prune.
/// Matches the fold's fanout in `graduate_buckets`; well inside S3's
/// per-client budget.
const COLD_LEAF_FETCH_CONCURRENCY: usize = 32;

/// Could `leaf` hold a data file belonging to any partition in `targets`?
///
/// Answers "no" ONLY when the leaf's partition summary is TIGHT — `lower ==
/// upper` on every field, which is exactly what partition-scoped cold leaves
/// are — and the value it is pinned to matches no target. Everything else
/// (no targets, absent summary, arity mismatch, a wide field, a value we
/// can't encode) answers "yes" and the leaf is loaded as before.
///
/// Comparison is byte equality against `Datum::to_bytes`, the same encoding
/// the manifest writer used to build the summary. Equality only — no ordering
/// — because byte order does not track value order for every primitive type,
/// and a range test would be silently wrong for signed integers. Pure.
/// Why a leaf survived the prune. The two reasons are operationally very
/// different and must not be conflated: `Matched` is the prune working as
/// intended, `Unprunable` is a leaf whose summary is too wide to rule out —
/// which on a tiered table means it was moved into cold by reference from a
/// hot manifest that was never partition-tight to begin with. Reporting only
/// the total made 191 unprunable leaves read as 191 leaves in one partition.
#[derive(PartialEq, Clone, Copy)]
pub(crate) enum LeafKeepReason {
    Matched,
    Unprunable,
    Pruned,
}

fn leaf_keep_reason(
    leaf: &ManifestFile,
    targets: &HashSet<Struct>,
    partition_type: &StructType,
) -> LeafKeepReason {
    if targets.is_empty() {
        return LeafKeepReason::Unprunable;
    }
    let Some(summary) = leaf.partitions.as_ref() else {
        return LeafKeepReason::Unprunable;
    };
    let fields = partition_type.fields();
    if summary.len() != fields.len() {
        return LeafKeepReason::Unprunable;
    }
    // Every field must be tight, else we cannot rule the leaf out at all.
    let mut pinned: Vec<&[u8]> = Vec::with_capacity(summary.len());
    for fs in summary.iter() {
        match (&fs.lower_bound, &fs.upper_bound) {
            (Some(lo), Some(hi)) if lo == hi => pinned.push(lo.as_ref()),
            _ => return LeafKeepReason::Unprunable,
        }
    }
    let matched = targets.iter().any(|target| {
        let values = target.fields();
        if values.len() != fields.len() {
            return true;
        }
        for (idx, field) in fields.iter().enumerate() {
            let Some(Literal::Primitive(p)) = &values[idx] else {
                // Null or nested partition value — not encodable as a bound
                // here, so don't claim a mismatch.
                return true;
            };
            let Type::Primitive(pt) = &field.field_type.as_ref() else {
                return true;
            };
            let Ok(bytes) = Datum::new(pt.clone(), p.clone()).to_bytes() else {
                return true;
            };
            if bytes.as_ref() != pinned[idx] {
                return false;
            }
        }
        true
    });
    if matched {
        LeafKeepReason::Matched
    } else {
        LeafKeepReason::Pruned
    }
}

/// Is this leaf's partition summary TIGHT — `lower == upper` on every field,
/// pinning it to exactly one partition?
///
/// Tightness is what makes a leaf prunable at all, independent of any target.
/// A wide leaf can never be ruled out, so it is loaded on every compaction of
/// every partition forever. Cold leaves written by `write_entries_clustered`
/// are tight by construction; leaves that arrived by REFERENCE from a hot
/// manifest are only as tight as that manifest was, and laminar's flush groups
/// on `[timestamp_hour]` alone by default — hour-tight, tenant-wide. Pure.
pub(crate) fn leaf_is_tight(leaf: &ManifestFile, partition_type: &StructType) -> bool {
    let Some(summary) = leaf.partitions.as_ref() else {
        return false;
    };
    if summary.len() != partition_type.fields().len() {
        return false;
    }
    summary.iter().all(|fs| match (&fs.lower_bound, &fs.upper_bound) {
        (Some(lo), Some(hi)) => lo == hi,
        _ => false,
    })
}

/// Back-compat wrapper: a leaf is loaded unless it is provably Pruned.
fn leaf_may_hold_partitions(
    leaf: &ManifestFile,
    targets: &HashSet<Struct>,
    partition_type: &StructType,
) -> bool {
    leaf_keep_reason(leaf, targets, partition_type) != LeafKeepReason::Pruned
}

fn existing_entry(df: DataFile) -> ManifestEntry {
    ManifestEntry::builder()
        .status(ManifestStatus::Existing)
        .data_file(df)
        .build()
}

impl CompactColdTierAction {
    /// The heavy read + write path (phases 1-5 in the actions_ms breakdown):
    /// reconstruct root, read bucket-index, load every affected cold leaf,
    /// apply removals, cluster survivors + additions into new leaves, write
    /// new leaves + new bucket-index to S3. Runs at most once per action
    /// instance (result cached in `self.prepared`) unless invalidated by a
    /// concurrent cold-tier writer.
    async fn prepare(&self, table: &Table) -> Result<Option<PreparedCompaction>> {
        let current_snapshot = table.metadata().current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "compact_cold_tier: table has no current snapshot",
            )
        })?;

        // Load the root and the cold bucket-index it points to.
        let root_path = current_snapshot.manifest_list();
        let (rm_metadata, root_entries) = reconstruct_root(table.file_io(), root_path).await?;
        let prep_bucket_index_path = rm_metadata.bucket_index_path.clone();

        let leaves: Vec<ManifestFile> = match &rm_metadata.bucket_index_path {
            Some(path) => {
                let b = table.file_io().new_input(path)?.read().await?;
                read_bucket_index(b)?.leaves().to_vec()
            }
            None => Vec::new(),
        };

        // Partition leaves into unaffected (carry forward) vs affected (rewrite),
        // collecting survivors from the affected ones.
        let mut kept_leaf_refs: Vec<RootManifestEntry> = Vec::new();
        let mut survivors: Vec<ManifestEntry> = Vec::new();
        let mut any_affected = false;

        // V4 incremental commits (laminar merge-on-write, etc.) tombstone
        // removed files in `RootManifestMetadata.removed_paths` rather than
        // rewriting the child manifest. Cold bucket-index leaves are MDV-blind
        // by design, so the ONLY way to see those deletes at this tier is via
        // this set. Without it, a compaction pass would resurrect the deleted
        // files into a new leaf (and hit 404s if the tombstone deleter has
        // already reclaimed them past grace).
        let removed_paths_set: HashSet<String> =
            rm_metadata.removed_paths.iter().cloned().collect();

        // Needed by the partition prune below as well as the rewrite further
        // down, so resolve it once here.
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(table.metadata().current_schema())?;

        // Narrow the walk before doing any I/O.
        //
        // Prep only loads a leaf to answer ONE question: does it hold any path
        // in `self.removed`? Leaves that answer "no" are re-emitted as the very
        // `ManifestFile` ref that `read_bucket_index` already returned — so
        // fetching them was pure waste. Serial + unfiltered, that made prep
        // scale with the whole cold tier: on sri-olly 6,698 metrics_1m leaves ×
        // ~26 ms/GET = the measured 175 s of a 294 s tick (2,021 logs leaves →
        // 61 s, 611 metrics → 17-23 s; the model fits all three).
        //
        // A compaction never crosses partitions — its merged outputs live in
        // the same partitions as the inputs it removes — and cold leaves are
        // written partition-tight, so a leaf whose summary pins it to some
        // OTHER partition provably cannot hold a removed path. That is decided
        // from the bucket-index alone, with no S3 read. Whatever survives the
        // prune is then loaded in parallel, so the conservative fallbacks below
        // stay fast too.
        //
        // Conservative by construction: the prune skips a leaf only when the
        // summary PROVES it can't match. No target partitions (pure-removal
        // caller), an absent summary, or a wide one ⇒ load it.
        // Whole-index shape, derived from partition summaries only — no loads.
        //
        // `distinct_summary_partitions` collapses to None if ANY leaf is
        // non-tight, which on the one table that needed the answer told us
        // nothing. Report the population instead: how many leaves are
        // structurally unprunable (wide), and how many distinct partitions the
        // tight ones span. `index_wide` is the number that decides whether the
        // fix belongs at the WRITE side (make hot manifests partition-tight so
        // by-reference graduation yields tight leaves) or at the cold tier.
        let index_tight: Vec<ManifestFile> = leaves
            .iter()
            .filter(|l| leaf_is_tight(l, &partition_type))
            .cloned()
            .collect();
        let index_wide = leaves.len() - index_tight.len();
        let index_partitions =
            crate::transaction::graduate_buckets::distinct_summary_partitions(&index_tight);

        let target_partitions: HashSet<Struct> =
            self.added.iter().map(|df| df.partition.clone()).collect();
        let mut loaded_matched = 0usize;
        let mut loaded_unprunable = 0usize;
        let (candidates, pruned): (Vec<ManifestFile>, Vec<ManifestFile>) = leaves
            .into_iter()
            .partition(|leaf| {
                match leaf_keep_reason(leaf, &target_partitions, &partition_type) {
                    LeafKeepReason::Matched => {
                        loaded_matched += 1;
                        true
                    }
                    LeafKeepReason::Unprunable => {
                        loaded_unprunable += 1;
                        true
                    }
                    LeafKeepReason::Pruned => false,
                }
            });
        let pruned_count = pruned.len();
        // A pruned leaf is unaffected by definition — carry it forward as-is.
        for leaf in pruned {
            kept_leaf_refs.push(RootManifestEntry::ManifestRef {
                manifest_file: leaf,
                mdv: None,
            });
        }

        let scan_start = std::time::Instant::now();
        let candidate_count = candidates.len();
        let file_io = table.file_io();
        // `buffered`, not `_unordered`: leaf order drives the resulting
        // bucket-index, and a deterministic index is worth the head-of-line
        // wait at this concurrency.
        let mut loaded = stream_iter(candidates.into_iter().map(|leaf| {
            let file_io = file_io.clone();
            async move {
                let res = leaf.load_manifest(&file_io).await;
                (leaf, res)
            }
        }))
        .buffered(COLD_LEAF_FETCH_CONCURRENCY);

        // Diagnostic for the leaf-count investigation: sri-olly logs carries
        // 4,280 cold leaves against ~50 live data files (~85 leaves per live
        // file), and the two candidate explanations need different fixes.
        // Either those leaves hold nothing live — in which case the append path
        // must validate before adding, since graduation's existing
        // `GraduatedNodePlan::Skip` only runs when `removed_paths` is non-empty
        // — or they do hold data, and the fix is merging same-partition leaves.
        // `empty_leaves` separates them, riding the load we already do.
        let mut empty_leaves: usize = 0;
        let mut entries_seen: usize = 0;
        let mut alive_entries: usize = 0;
        while let Some((leaf, manifest)) = loaded.next().await {
            let manifest = manifest?;
            let files: Vec<DataFile> = manifest
                .entries()
                .iter()
                .filter(|e| e.is_alive_and_kept(&removed_paths_set))
                .map(|e| e.data_file().clone())
                .collect();
            entries_seen += manifest.entries().len();
            alive_entries += files.len();
            if files.is_empty() {
                empty_leaves += 1;
            }
            let (leaf_survivors, affected) = apply_removals(files, &self.removed);
            if affected {
                any_affected = true;
                survivors.extend(leaf_survivors.into_iter().map(existing_entry));
            } else {
                // Unaffected — keep the leaf as-is (no rewrite, no re-read cost).
                kept_leaf_refs.push(RootManifestEntry::ManifestRef {
                    manifest_file: leaf,
                    mdv: None,
                });
            }
        }
        log::info!(
            "compact_cold_tier leaf scan: leaves={} pruned_by_partition={} loaded={} \
             loaded_matched={} loaded_unprunable={} \
             empty_leaves={} entries_seen={} alive_entries={} \
             index_wide={} index_partitions_tight={:?} \
             target_partitions={} scan_ms={}",
            pruned_count + candidate_count,
            pruned_count,
            candidate_count,
            loaded_matched,
            loaded_unprunable,
            empty_leaves,
            entries_seen,
            alive_entries,
            index_wide,
            index_partitions,
            target_partitions.len(),
            scan_start.elapsed().as_millis() as u64,
        );

        // If nothing matched the removal set and there's nothing to add, no-op.
        if !any_affected && self.added.is_empty() {
            return Ok(None);
        }

        // Non-empty removal set but no matches ⇒ the paths aren't in the cold
        // tier (still hot, or already reclaimed). Adding `self.added` in that
        // state would duplicate rows: the "inputs" stay live in hot manifests
        // while the compacted "outputs" also land in a new cold leaf, and both
        // are visible to readers. Refuse. Callers (laminar's cold-redirect of a
        // graduation-crossed merge, tessellate's cold-compact) should treat this
        // as "not in the right state" and drop the outputs as orphan.
        if !any_affected && !self.removed.is_empty() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "compact_cold_tier: {} removal path(s) matched no cold-leaf entries — refusing to add {} file(s) alone (would duplicate rows if inputs are still hot)",
                    self.removed.len(),
                    self.added.len(),
                ),
            ));
        }

        // Re-cluster the affected survivors + the merged additions into new,
        // partition-tight leaf manifests. snapshot_id is allocated ONCE here
        // and cached in PreparedCompaction so leaf-manifest lineage stays
        // consistent across retries (the leaves embed this id; regenerating
        // per retry would produce contradictory lineage on winning attempts).
        let snapshot_id = self
            .snapshot_id_override
            .unwrap_or_else(|| SnapshotProducer::generate_unique_snapshot_id_static(table));
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let format_version = table.metadata().format_version();
        let commit_uuid = self.commit_uuid;
        let mut manifest_counter: u64 = 0;

        let mut rewrite_entries = survivors;
        rewrite_entries.extend(self.added.iter().cloned().map(existing_entry));

        let new_leaves = write_entries_clustered(
            table,
            &schema,
            spec.as_ref(),
            format_version,
            snapshot_id,
            commit_uuid,
            &mut manifest_counter,
            false,
            rewrite_entries,
            true, // partition-tight cold leaves
        )
        .await?;

        // New bucket-index = unaffected leaf refs + the freshly written leaves.
        let mut all_leaves: Vec<ManifestFile> = kept_leaf_refs
            .iter()
            .filter_map(|e| match e {
                RootManifestEntry::ManifestRef { manifest_file, .. } => Some(manifest_file.clone()),
                _ => None,
            })
            .collect();
        all_leaves.extend(new_leaves);

        let new_bucket_index_path = if all_leaves.is_empty() {
            None
        } else {
            let path = format!(
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
            let bi_bytes = write_bucket_index(&all_leaves, &bi_metadata, &partition_type)?;
            table
                .file_io()
                .new_output(&path)?
                .write(bi_bytes.into())
                .await?;
            Some(path)
        };

        Ok(Some(PreparedCompaction {
            new_bucket_index_path,
            prep_bucket_index_path,
            snapshot_id,
            root_entries: Arc::new(root_entries),
            ancestor_removed_paths: Arc::new(rm_metadata.removed_paths),
            any_affected,
        }))
    }

    /// Phase 6 in the actions_ms breakdown: write the new (delta or flat)
    /// root and build the ActionCommit with fresh `RefSnapshotIdMatch` for
    /// the current table head. Runs on every `commit()` attempt (both the
    /// slow-path first attempt and the fast-path retries). Cost: ~500 ms
    /// on the delta path (metadata-only parquet write), or the flat-base
    /// serialize+PUT cost when `chain_depth >= MAX_CHAIN`.
    async fn finalize(
        &self,
        table: &Table,
        prep: &PreparedCompaction,
    ) -> Result<ActionCommit> {
        let current_snapshot = table.metadata().current_snapshot().ok_or_else(|| {
            Error::new(
                ErrorKind::Unexpected,
                "compact_cold_tier: table has no current snapshot at finalize",
            )
        })?;
        let root_path = current_snapshot.manifest_list();
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(table.metadata().current_schema())?;

        // Read HEAD root metadata only (no chain walk) — needed for
        // chain_depth (delta-vs-flat decision) and (on flat-base fallback)
        // for the current entry set. `read_root_manifest` reads just the one
        // parquet file at `root_path`, not the whole chain — cheap (~200 ms).
        let head_bytes = table.file_io().new_input(root_path)?.read().await?;
        let (head_meta, _head_entries) = read_root_manifest(head_bytes)?;

        // Delta vs flat-base root. On a `root-manifest.incremental` table,
        // emit the swap as a small delta (`prev_root_path=Some(current)`,
        // `chain_depth+=1`, empty entries slice) so `actions_ms` drops from
        // the ~10-15s flat-root serialize + reconstruct-collapse cost to a
        // few-KB metadata-only write. reconstruct_root walks the chain via
        // prev_root_path and treats HEAD meta as authoritative for
        // `bucket_index_path`, so the cold-pointer swap is picked up.
        // Removals are materialized (rewritten leaves don't contain them),
        // so NO new tombstone is added; ancestor tombstones live on prior
        // roots and are unioned by reconstruct_root — do NOT copy them onto
        // the delta (that would duplicate and block eventual GC).
        //
        // At `chain_depth == MAX_CHAIN` (mirrors commit_v4's cap at
        // snapshot.rs:1349), fall through to the flat-base branch — same
        // collapse semantics commit_v4 uses on its 1-in-MAX_CHAIN commit.
        // When the table lacks the incremental property (default off), the
        // flat branch is preserved unchanged for parity with pre-delta
        // behavior.
        //
        // Flat-base fallback correctness under prep-reuse: on the flat
        // branch we ALWAYS use `prep.root_entries` (captured at prep time)
        // — which is stale if laminar has committed since prep. That's
        // fine because the outer commit-cache invalidates prep whenever
        // laminar's write changed `bucket_index_path`, which is the ONLY
        // way the cold-relevant entry set can change (laminar hot appends
        // add inline entries, not ManifestRefs — inline entries on the
        // cached root are stale but harmless because the CAS'd new root
        // is a NEW snapshot whose inline entries are what the writer
        // intends; readers reconstruct the merged view via the chain and
        // laminar's inline entries live on THEIR ancestor roots, not
        // ours). The strictly-correct alternative (re-reconstruct root on
        // MAX_CHAIN fallback) is deferred as a small follow-up; it saves
        // rare stale-entry inclusion on collapse, not steady-state perf.
        let incremental = table
            .metadata()
            .properties()
            .get("root-manifest.incremental")
            .map(|v| v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let do_delta = incremental && head_meta.chain_depth < max_chain_depth();

        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id: prep.snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: prep.new_bucket_index_path.clone(),
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
            removed_paths: if do_delta {
                Vec::new()
            } else {
                (*prep.ancestor_removed_paths).clone()
            },
        };
        // Root path includes commit_uuid + attempt-fresh nanoid to keep
        // each attempt's root parquet distinct on S3 — a failed retry
        // leaves an orphan, reclaimed later by tombstone GC.
        let new_root_path = format!(
            "{}/{}/root-{}-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            prep.snapshot_id,
            self.commit_uuid,
            Uuid::now_v7(),
        );
        let entries_to_write: &[RootManifestEntry] = if do_delta {
            &[]
        } else {
            prep.root_entries.as_slice()
        };
        let root_bytes = write_root_manifest(entries_to_write, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_path)?
            .write(root_bytes.into())
            .await?;

        let leaves_after = if prep.new_bucket_index_path.is_some() {
            // Not cheap to recompute without a bucket-index read; report
            // the counts the caller passed in as a rough approximation for
            // the summary. Callers that need precision should use the
            // per-commit event stream instead.
            self.added.len() + prep.root_entries.len()
        } else {
            0
        };
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                (
                    "cold-compact-removed".to_string(),
                    self.removed.len().to_string(),
                ),
                (
                    "cold-compact-added".to_string(),
                    self.added.len().to_string(),
                ),
                (
                    "cold-compact-leaves-after".to_string(),
                    leaves_after.to_string(),
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

        let mut manifest_paths = vec![new_root_path];
        if let Some(p) = &prep.new_bucket_index_path {
            manifest_paths.push(p.clone());
        }
        // Suppress warning: `any_affected` is stored on prep only for the
        // "distinguish real work from no-op" semantic; the finalize path
        // itself doesn't branch on it (a `None` return from prepare already
        // handled the no-op case, and the refuse-if-not-affected guard fires
        // inside prepare too).
        let _ = prep.any_affected;
        Ok(ActionCommit::new(updates, requirements).with_manifest_paths(manifest_paths))
    }
}

#[async_trait]
impl TransactionAction for CompactColdTierAction {
    fn action_name(&self) -> &'static str {
        "compact_cold_tier"
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "compact_cold_tier requires format version V4 (effective={:?})",
                    table.effective_format_version()
                ),
            ));
        }
        if self.removed.is_empty() && self.added.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
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
        // walk) — cheap even when it invalidates. Held under the mutex so a
        // parallel commit attempt (should never happen in practice — the
        // Transaction commit loop is serial per action instance — but the
        // mutex makes it safe regardless) can't race the prep write.
        let mut guard = self.prepared.lock().await;

        // Determine if we can reuse cached prep. Only reusable if the cold
        // bucket-index pointer hasn't shifted (bucket_index_path unchanged)
        // AND we're still on the delta-writable side of the chain-depth cap
        // — MAX_CHAIN triggers a flat-base rewrite that needs a fresh
        // root_entries reconstruct, which cached prep can't provide (its
        // entries are as-of-prep-time; see the flat-base fallback comment
        // in `finalize`).
        let can_reuse = if let Some(ref prep) = *guard {
            let current_snapshot = table.metadata().current_snapshot().unwrap();
            let head_bytes = table
                .file_io()
                .new_input(current_snapshot.manifest_list())?
                .read()
                .await?;
            let (head_meta, _) = read_root_manifest(head_bytes)?;
            head_meta.bucket_index_path == prep.prep_bucket_index_path
                && head_meta.chain_depth < max_chain_depth()
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
            // No-op — prepare returned early (nothing to do). Emit an empty
            // ActionCommit; don't cache (nothing to cache).
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };
        let ac = self.finalize(table, &prep).await?;
        *guard = Some(prep);
        Ok(ac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{DataContentType, DataFileBuilder, DataFileFormat, Struct};

    fn df(path: &str) -> DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition_spec_id(0)
            .partition(Struct::empty())
            .build()
            .unwrap()
    }

    #[test]
    fn apply_removals_filters_and_flags() {
        let files = vec![df("keep-1"), df("merged-away"), df("keep-2")];
        let removed: HashSet<String> = ["merged-away".to_string()].into_iter().collect();
        let (survivors, affected) = apply_removals(files, &removed);
        assert!(affected);
        let paths: Vec<&str> = survivors.iter().map(|f| f.file_path.as_str()).collect();
        assert_eq!(paths, vec!["keep-1", "keep-2"]);
    }

    #[test]
    fn apply_removals_unaffected_leaf() {
        let files = vec![df("a"), df("b")];
        let removed: HashSet<String> = ["x".to_string()].into_iter().collect();
        let (survivors, affected) = apply_removals(files, &removed);
        assert!(!affected);
        assert_eq!(survivors.len(), 2);
    }

    fn leaf_with(partitions: Option<Vec<crate::spec::FieldSummary>>) -> ManifestFile {
        ManifestFile {
            manifest_path: "s3://b/leaf.parquet".to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: crate::spec::ManifestContentType::Data,
            sequence_number: 5,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions,
            key_metadata: None,
            first_row_id: None,
        }
    }

    /// Single Long partition field, matching the `hour`-style specs in use.
    fn long_partition_type() -> StructType {
        StructType::new(vec![crate::spec::NestedField::required(
            1000,
            "p",
            Type::Primitive(crate::spec::PrimitiveType::Long),
        )
        .into()])
    }

    fn long_partition(v: i64) -> Struct {
        Struct::from_iter([Some(Literal::Primitive(crate::spec::PrimitiveLiteral::Long(
            v,
        )))])
    }

    fn tight_summary(v: i64) -> Option<Vec<crate::spec::FieldSummary>> {
        let bytes = Datum::long(v).to_bytes().unwrap();
        Some(vec![crate::spec::FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(bytes.clone()),
            upper_bound: Some(bytes),
        }])
    }

    // The prune is what stops prep from scaling with the whole cold tier
    // (6,698 leaves × ~26ms/GET = 175s on sri-olly). A tight leaf pinned to a
    // partition none of the targets match provably holds no removed path.
    #[test]
    fn tight_leaf_in_another_partition_is_pruned() {
        let pt = long_partition_type();
        let targets: HashSet<Struct> = [long_partition(7)].into_iter().collect();
        assert!(!leaf_may_hold_partitions(
            &leaf_with(tight_summary(9)),
            &targets,
            &pt
        ));
        assert!(leaf_may_hold_partitions(
            &leaf_with(tight_summary(7)),
            &targets,
            &pt
        ));
    }

    // Every way of NOT knowing must load the leaf. Pruning on a guess here
    // silently drops data files out of the rewrite.
    #[test]
    fn prune_is_conservative_when_it_cannot_prove_a_mismatch() {
        let pt = long_partition_type();
        let targets: HashSet<Struct> = [long_partition(7)].into_iter().collect();

        // No targets at all (pure-removal caller) — nothing to prune against.
        assert!(leaf_may_hold_partitions(
            &leaf_with(tight_summary(9)),
            &HashSet::new(),
            &pt
        ));
        // No summary.
        assert!(leaf_may_hold_partitions(&leaf_with(None), &targets, &pt));
        // Summary arity disagrees with the spec.
        assert!(leaf_may_hold_partitions(
            &leaf_with(Some(vec![])),
            &targets,
            &pt
        ));
        // Wide field (lower != upper) — the leaf spans values, so it may match.
        let wide = Some(vec![crate::spec::FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(Datum::long(1).to_bytes().unwrap()),
            upper_bound: Some(Datum::long(99).to_bytes().unwrap()),
        }]);
        assert!(leaf_may_hold_partitions(&leaf_with(wide), &targets, &pt));
        // Missing bound.
        let half = Some(vec![crate::spec::FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(Datum::long(7).to_bytes().unwrap()),
            upper_bound: None,
        }]);
        assert!(leaf_may_hold_partitions(&leaf_with(half), &targets, &pt));
    }

    // Tightness is the property that makes a leaf prunable at all, and it is
    // independent of any target — a wide leaf is loaded on every compaction of
    // every partition, forever.
    #[test]
    fn tightness_is_independent_of_any_target() {
        let pt = long_partition_type();
        assert!(leaf_is_tight(&leaf_with(tight_summary(7)), &pt));
        assert!(!leaf_is_tight(&leaf_with(None), &pt));
        assert!(!leaf_is_tight(&leaf_with(Some(vec![])), &pt));
        let wide = Some(vec![crate::spec::FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(Datum::long(1).to_bytes().unwrap()),
            upper_bound: Some(Datum::long(99).to_bytes().unwrap()),
        }]);
        assert!(!leaf_is_tight(&leaf_with(wide), &pt));
        // A missing bound is not tight either — absence is not equality.
        let half = Some(vec![crate::spec::FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(Datum::long(7).to_bytes().unwrap()),
            upper_bound: None,
        }]);
        assert!(!leaf_is_tight(&leaf_with(half), &pt));
    }

    // The two keep-reasons must stay distinguishable. Conflating them made
    // 191 unprunable leaves read as "191 leaves in one partition", which is a
    // completely different diagnosis with a completely different fix.
    #[test]
    fn keep_reason_separates_matched_from_unprunable() {
        let pt = long_partition_type();
        let targets: HashSet<Struct> = [long_partition(7)].into_iter().collect();

        assert!(matches!(
            leaf_keep_reason(&leaf_with(tight_summary(7)), &targets, &pt),
            LeafKeepReason::Matched
        ));
        assert!(matches!(
            leaf_keep_reason(&leaf_with(tight_summary(9)), &targets, &pt),
            LeafKeepReason::Pruned
        ));
        // Wide summary: kept, but NOT because it matched — this is the case
        // that inflates `loaded` on a table whose cold leaves arrived by
        // reference from non-partition-tight hot manifests.
        let wide = Some(vec![crate::spec::FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(Datum::long(1).to_bytes().unwrap()),
            upper_bound: Some(Datum::long(99).to_bytes().unwrap()),
        }]);
        assert!(matches!(
            leaf_keep_reason(&leaf_with(wide), &targets, &pt),
            LeafKeepReason::Unprunable
        ));
        assert!(matches!(
            leaf_keep_reason(&leaf_with(None), &targets, &pt),
            LeafKeepReason::Unprunable
        ));
        // No targets at all is also "cannot decide", not "matched".
        assert!(matches!(
            leaf_keep_reason(&leaf_with(tight_summary(7)), &HashSet::new(), &pt),
            LeafKeepReason::Unprunable
        ));
    }

    // Any ONE matching target keeps the leaf — a batched swap spans several
    // partitions and each leaf only has to match one of them.
    #[test]
    fn leaf_matching_any_target_partition_is_kept() {
        let pt = long_partition_type();
        let targets: HashSet<Struct> = [long_partition(1), long_partition(7), long_partition(42)]
            .into_iter()
            .collect();
        assert!(leaf_may_hold_partitions(
            &leaf_with(tight_summary(42)),
            &targets,
            &pt
        ));
        assert!(!leaf_may_hold_partitions(
            &leaf_with(tight_summary(43)),
            &targets,
            &pt
        ));
    }

    #[test]
    fn builder_accumulates() {
        let a = CompactColdTierAction::new()
            .remove_files(["p1".to_string(), "p2".to_string()])
            .add_files([df("merged")]);
        assert_eq!(a.removed.len(), 2);
        assert_eq!(a.added.len(), 1);
    }
}
