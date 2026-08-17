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

//! Cold-presence sidecar: "is this data file listed anywhere in the cold tier?"
//!
//! # Why this exists
//!
//! Retiring a carried path-tombstone is only safe when no cold leaf still lists
//! the file — otherwise the tombstone is the sole thing suppressing it and
//! retiring it **resurrects deleted data**. `materialize_carried_tombstones`
//! answered that by loading every cold leaf manifest, which on sri-olly meant
//! ~7,300 manifest loads costing ~29 s, on laminar's ingest commit path, roughly
//! six times an hour. Worse, it is O(cold tier) for work that is inherently
//! O(tombstones), and it bails out entirely above a manifest cap — so past that
//! size it silently stops retiring anything at all.
//!
//! This module answers the same question from one small object.
//!
//! # Why a Bloom filter is the right structure
//!
//! The error mode lines up exactly with the safety requirement. A Bloom filter
//! has **no false negatives**, so "definitely absent" is authoritative — and
//! absent is the only direction retirement needs:
//!
//! | lookup says          | action           | risk                              |
//! |----------------------|------------------|-----------------------------------|
//! | definitely NOT present | safe to retire | none — no false negatives         |
//! | maybe present          | veto retirement| conservative; lingers one cycle   |
//!
//! A false positive costs one tombstone one extra cycle. A false negative would
//! cost data. The structure cannot produce the latter.
//!
//! # The invariant, and why staleness fails closed
//!
//! Correctness rests on the filter being a **superset** of the real cold path
//! set. Inserts are exact; removals (TTL dropping leaves) are simply skipped,
//! which keeps it a superset. But a writer that adds to cold and *forgets* to
//! insert would create a false negative, and a false negative resurrects data.
//!
//! That must not depend on remembering. The sidecar records the leaf set it was
//! built against (`covered_leaf_count` + `covered_leaf_digest`), both derivable
//! from the bucket index the caller already holds. [`load_cold_paths_sidecar`]
//! recomputes them and returns `None` on any mismatch — the caller then skips
//! cold-veto retirement entirely, which is exactly today's conservative
//! behaviour. **Forgetting to update becomes a performance bug, never a
//! correctness one.**

use std::collections::HashMap;

use serde_derive::{Deserialize, Serialize};

use crate::spec::ManifestFile;

/// Suffix appended to a bucket-index path to locate its cold-paths sidecar.
/// Mirrors `BUCKET_INDEX_MAXTS_SIDECAR_SUFFIX`.
pub(crate) const COLD_PATHS_SIDECAR_SUFFIX: &str = ".cold-paths.json";

const COLD_PATHS_FORMAT_V1: u32 = 1;

/// Target false-positive probability. 1% means ~1% of retirable tombstones wait
/// one more pass — bounded, self-correcting, and cheap: at 1% the filter costs
/// ~9.6 bits per path (~600 KB per 500 K files, ~5 MB at the 4.3 M-file
/// 1,000-tenant × 6-month target).
const TARGET_FPP: f64 = 0.01;

/// Floor on the bit count so a tiny or empty cold tier still produces a usable
/// filter rather than a degenerate one.
const MIN_BITS: u64 = 1_024;

/// Ceiling on the bit count (~16 MB). Past this the filter saturates and false
/// positives climb, which only makes retirement more conservative — never
/// unsafe. Prevents an unbounded allocation on a pathological tier.
const MAX_BITS: u64 = 128 * 1024 * 1024;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Second basis for double hashing. Must differ from `FNV_OFFSET` so the two
/// hashes are independent.
const FNV_OFFSET_ALT: u64 = 0x9e37_79b9_7f4a_7c15;

/// Deterministic FNV-1a. Explicitly NOT `DefaultHasher`, whose output is not
/// guaranteed stable across Rust releases — a sidecar written by one build must
/// be readable by another, or the filter silently starts missing paths.
fn fnv1a(data: &[u8], basis: u64) -> u64 {
    let mut h = basis;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Bloom filter over cold-tier data-file paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ColdPathFilter {
    bits: Vec<u8>,
    n_bits: u64,
    n_hashes: u32,
    n_items: u64,
}

impl ColdPathFilter {
    /// Size a filter for `expected_items` at [`TARGET_FPP`].
    ///
    /// `m = -n·ln(p) / (ln2)²`, `k = (m/n)·ln2` — the standard optimum.
    pub(crate) fn with_capacity(expected_items: usize) -> Self {
        let n = expected_items.max(1) as f64;
        let ln2 = std::f64::consts::LN_2;
        let raw_bits = (-n * TARGET_FPP.ln() / (ln2 * ln2)).ceil();
        let n_bits = (raw_bits as u64).clamp(MIN_BITS, MAX_BITS);
        let n_hashes = (((n_bits as f64 / n) * ln2).round() as u32).clamp(1, 16);
        Self {
            bits: vec![0u8; n_bits.div_ceil(8) as usize],
            n_bits,
            n_hashes,
            n_items: 0,
        }
    }

    /// Bit positions for `path`, via double hashing: `h1 + i·h2`.
    fn positions(&self, path: &str) -> impl Iterator<Item = u64> + '_ {
        let b = path.as_bytes();
        let h1 = fnv1a(b, FNV_OFFSET);
        // Force odd so successive probes stride the whole filter rather than
        // collapsing onto a subset when h2 shares factors with n_bits.
        let h2 = fnv1a(b, FNV_OFFSET_ALT) | 1;
        let n_bits = self.n_bits;
        (0..self.n_hashes as u64).map(move |i| h1.wrapping_add(i.wrapping_mul(h2)) % n_bits)
    }

    pub(crate) fn insert(&mut self, path: &str) {
        for pos in self.positions(path).collect::<Vec<_>>() {
            self.bits[(pos / 8) as usize] |= 1u8 << (pos % 8);
        }
        self.n_items += 1;
    }

    /// `false` means the path is **definitely absent** — the only answer that
    /// authorises retirement. `true` means "maybe present": veto.
    pub(crate) fn maybe_contains(&self, path: &str) -> bool {
        self.positions(path)
            .all(|pos| self.bits[(pos / 8) as usize] & (1u8 << (pos % 8)) != 0)
    }

    pub(crate) fn len(&self) -> u64 {
        self.n_items
    }
}

/// Stable digest over the leaf set a sidecar was built from.
///
/// Order-independent (leaves are XOR-folded), so a bucket-index rewrite that
/// only reorders leaves does not spuriously invalidate the sidecar — while any
/// added or removed leaf does.
pub(crate) fn coverage_digest(leaves: &[ManifestFile]) -> u64 {
    leaves
        .iter()
        .map(|l| fnv1a(l.manifest_path.as_bytes(), FNV_OFFSET))
        .fold(0u64, |acc, h| acc ^ h)
}

#[derive(Serialize, Deserialize)]
struct ColdPathsSidecar {
    format_version: u32,
    n_bits: u64,
    n_hashes: u32,
    n_items: u64,
    /// Leaf count the filter was built against — checked on load.
    covered_leaf_count: u64,
    /// Order-independent digest of those leaves' paths — checked on load.
    covered_leaf_digest: u64,
    /// The bitset. `serde_bytes` keeps this compact rather than a JSON array of
    /// integers.
    #[serde(with = "serde_bytes")]
    bits: Vec<u8>,
}

fn cold_paths_sidecar_path(bucket_index_path: &str) -> String {
    format!("{bucket_index_path}{COLD_PATHS_SIDECAR_SUFFIX}")
}

/// Load the cold-paths filter for `bucket_index_path`, validated against the
/// leaf set it must cover.
///
/// Returns `None` — meaning "do not retire on cold-absence" — for every failure
/// mode: missing, unreadable, corrupt, wrong format version, or a leaf set that
/// does not match what the filter was built from. Every one of those degrades to
/// today's conservative behaviour rather than risking a false negative.
pub(crate) async fn load_cold_paths_sidecar(
    file_io: &crate::io::FileIO,
    bucket_index_path: &str,
    current_leaves: &[ManifestFile],
) -> Option<ColdPathFilter> {
    let path = cold_paths_sidecar_path(bucket_index_path);
    let bytes = file_io.new_input(&path).ok()?.read().await.ok()?;
    let sidecar: ColdPathsSidecar = serde_json::from_slice(&bytes).ok()?;

    if sidecar.format_version != COLD_PATHS_FORMAT_V1 {
        return None;
    }
    // Coverage guard — the fail-closed half of the invariant.
    if sidecar.covered_leaf_count != current_leaves.len() as u64
        || sidecar.covered_leaf_digest != coverage_digest(current_leaves)
    {
        return None;
    }
    // Structural sanity: a truncated bitset would silently read as zeros, which
    // is a false negative — the one error this must never produce.
    if sidecar.n_bits == 0 || sidecar.bits.len() != sidecar.n_bits.div_ceil(8) as usize {
        return None;
    }

    Some(ColdPathFilter {
        bits: sidecar.bits,
        n_bits: sidecar.n_bits,
        n_hashes: sidecar.n_hashes,
        n_items: sidecar.n_items,
    })
}

/// Write the cold-paths sidecar for `bucket_index_path`, stamped with the leaf
/// set it covers. Callers treat failure as non-fatal: a missing sidecar just
/// means the next sweep skips cold-absence retirement.
pub(crate) async fn write_cold_paths_sidecar(
    file_io: &crate::io::FileIO,
    bucket_index_path: &str,
    filter: &ColdPathFilter,
    covered_leaves: &[ManifestFile],
) -> crate::error::Result<()> {
    let sidecar = ColdPathsSidecar {
        format_version: COLD_PATHS_FORMAT_V1,
        n_bits: filter.n_bits,
        n_hashes: filter.n_hashes,
        n_items: filter.n_items,
        covered_leaf_count: covered_leaves.len() as u64,
        covered_leaf_digest: coverage_digest(covered_leaves),
        bits: filter.bits.clone(),
    };
    let bytes = serde_json::to_vec(&sidecar).map_err(|e| {
        crate::error::Error::new(
            crate::error::ErrorKind::Unexpected,
            format!("serialize cold-paths sidecar: {e}"),
        )
    })?;
    file_io
        .new_output(&cold_paths_sidecar_path(bucket_index_path))?
        .write(bytes.into())
        .await
}

/// Build a filter from the data-file paths a caller already holds, keyed by leaf.
/// Used for a full rebuild (tessellate, which loads leaves during compaction
/// anyway) and by tests.
pub(crate) fn build_filter(paths_by_leaf: &HashMap<String, Vec<String>>) -> ColdPathFilter {
    let total: usize = paths_by_leaf.values().map(|v| v.len()).sum();
    let mut f = ColdPathFilter::with_capacity(total);
    for paths in paths_by_leaf.values() {
        for p in paths {
            f.insert(p);
        }
    }
    f
}

/// What the cold tier can tell us about a path, and — critically — what it
/// cannot.
///
/// The three cases are genuinely different and collapsing them is unsafe:
/// a table with no cold tier can retire freely, while a table whose sidecar is
/// missing or stale knows *nothing* and must retire nothing.
pub(crate) enum ColdPresence<'a> {
    /// No bucket index — the cold tier is genuinely empty, so nothing can be
    /// vetoed and every unmatched tombstone is safe to retire.
    Empty,
    /// A sidecar validated against the current leaf set.
    Filter(&'a ColdPathFilter),
    /// A cold tier exists but we could not establish its contents (sidecar
    /// missing, corrupt, or built against a different leaf set). Vetoes
    /// everything — the fail-closed case.
    Unknown,
}

impl ColdPresence<'_> {
    /// `true` vetoes retirement. `Unknown` answers `true` for every path, which
    /// is what makes an absent sidecar degrade to "retire nothing" instead of
    /// "retire everything".
    pub(crate) fn may_contain(&self, path: &str) -> bool {
        match self {
            Self::Empty => false,
            Self::Filter(f) => f.maybe_contains(path),
            Self::Unknown => true,
        }
    }

    /// Whether cold-absence retirement can happen at all this pass — used for
    /// telemetry, so a silently-degraded sweep is visible rather than looking
    /// like a healthy one that found nothing.
    pub(crate) fn is_authoritative(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Owned form of [`ColdPresence`], so a caller can resolve it once and borrow it
/// for the sweep.
pub(crate) enum ColdPresenceOwned {
    Empty,
    Filter(ColdPathFilter),
    Unknown,
}

impl ColdPresenceOwned {
    pub(crate) fn as_ref(&self) -> ColdPresence<'_> {
        match self {
            Self::Empty => ColdPresence::Empty,
            Self::Filter(f) => ColdPresence::Filter(f),
            Self::Unknown => ColdPresence::Unknown,
        }
    }

    /// Short label for telemetry, so a degraded sweep is distinguishable from a
    /// healthy one that simply found nothing to retire.
    pub(crate) fn describe(&self) -> &'static str {
        match self {
            Self::Empty => "no-cold-tier",
            Self::Filter(_) => "sidecar",
            Self::Unknown => "unknown-veto-all",
        }
    }
}

/// Resolve what we can say about the cold tier, reading only the bucket index
/// (one small object) plus its sidecar.
///
/// Never loads leaf manifests — that is the whole point. Any failure resolves to
/// [`ColdPresenceOwned::Unknown`], which vetoes all cold-absence retirement.
pub(crate) async fn resolve_cold_presence(
    file_io: &crate::io::FileIO,
    bucket_index_path: Option<&str>,
) -> ColdPresenceOwned {
    let Some(bp) = bucket_index_path else {
        // No bucket index at all: the cold tier is genuinely empty.
        return ColdPresenceOwned::Empty;
    };
    let Ok(input) = file_io.new_input(bp) else {
        return ColdPresenceOwned::Unknown;
    };
    let Ok(bytes) = input.read().await else {
        return ColdPresenceOwned::Unknown;
    };
    let Ok(index) = crate::spec::bucket_index::read_bucket_index(bytes) else {
        return ColdPresenceOwned::Unknown;
    };
    let leaves = index.leaves();
    if leaves.is_empty() {
        return ColdPresenceOwned::Empty;
    }
    match load_cold_paths_sidecar(file_io, bp, leaves).await {
        Some(f) => ColdPresenceOwned::Filter(f),
        None => ColdPresenceOwned::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{ManifestContentType, ManifestFile};

    fn leaf(path: &str) -> ManifestFile {
        ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: 1,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 1,
            min_sequence_number: 1,
            added_snapshot_id: 1,
            added_files_count: Some(1),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        }
    }

    /// THE safety property. A false negative retires a tombstone whose file is
    /// still listed in cold, resurrecting deleted data. Everything else in this
    /// module is a performance detail; this one is correctness.
    #[test]
    fn never_reports_absent_for_a_path_it_holds() {
        let paths: Vec<String> = (0..5_000)
            .map(|i| format!("s3://b/data/compacted-{i:08}.parquet"))
            .collect();
        let mut f = ColdPathFilter::with_capacity(paths.len());
        for p in &paths {
            f.insert(p);
        }
        for p in &paths {
            assert!(
                f.maybe_contains(p),
                "false negative on {p} — this would resurrect deleted data"
            );
        }
        assert_eq!(f.len(), 5_000);
    }

    /// False positives are permitted but must stay near the target, or nothing
    /// ever retires and the sweep silently becomes a no-op again.
    #[test]
    fn false_positive_rate_stays_near_target() {
        let n = 20_000;
        let mut f = ColdPathFilter::with_capacity(n);
        for i in 0..n {
            f.insert(&format!("s3://b/data/present-{i:08}.parquet"));
        }
        let probes = 20_000;
        let fp = (0..probes)
            .filter(|i| f.maybe_contains(&format!("s3://b/data/absent-{i:08}.parquet")))
            .count();
        let rate = fp as f64 / probes as f64;
        assert!(
            rate < TARGET_FPP * 3.0,
            "fpp {rate:.4} is far above the {TARGET_FPP} target ({fp}/{probes})"
        );
    }

    #[test]
    fn empty_filter_reports_everything_absent() {
        let f = ColdPathFilter::with_capacity(0);
        assert!(!f.maybe_contains("s3://b/data/anything.parquet"));
        assert_eq!(f.len(), 0);
    }

    /// The coverage guard must fire on any leaf added or removed — that is what
    /// turns "a writer forgot to insert" into a skipped retirement instead of a
    /// resurrection.
    #[test]
    fn coverage_digest_detects_added_and_removed_leaves() {
        let a = vec![leaf("s3://b/m/a-m0.parquet"), leaf("s3://b/m/b-m0.parquet")];
        let mut added = a.clone();
        added.push(leaf("s3://b/m/c-m0.parquet"));
        let removed = vec![a[0].clone()];

        assert_ne!(coverage_digest(&a), coverage_digest(&added), "added leaf");
        assert_ne!(
            coverage_digest(&a),
            coverage_digest(&removed),
            "removed leaf"
        );
    }

    /// Reordering is not a change — a bucket-index rewrite that only re-clusters
    /// leaves must not invalidate an otherwise-correct sidecar.
    #[test]
    fn coverage_digest_is_order_independent() {
        let a = vec![
            leaf("s3://b/m/a-m0.parquet"),
            leaf("s3://b/m/b-m0.parquet"),
            leaf("s3://b/m/c-m0.parquet"),
        ];
        let mut b = a.clone();
        b.reverse();
        assert_eq!(coverage_digest(&a), coverage_digest(&b));
    }

    /// Hashing must be stable across builds: a sidecar written by one binary is
    /// read by another. `DefaultHasher` would silently break this.
    #[test]
    fn hashing_is_deterministic() {
        let p = "s3://b/data/compacted-00000042.parquet";
        assert_eq!(
            fnv1a(p.as_bytes(), FNV_OFFSET),
            fnv1a(p.as_bytes(), FNV_OFFSET)
        );
        assert_ne!(
            fnv1a(p.as_bytes(), FNV_OFFSET),
            fnv1a(p.as_bytes(), FNV_OFFSET_ALT),
            "the two bases must give independent hashes"
        );
    }

    /// The fail-closed case. An absent or stale sidecar must veto everything,
    /// because "I don't know what cold holds" and "cold holds nothing" have
    /// opposite consequences.
    #[test]
    fn unknown_cold_presence_vetoes_everything() {
        let unknown = ColdPresence::Unknown;
        assert!(unknown.may_contain("s3://b/data/anything.parquet"));
        assert!(!unknown.is_authoritative());

        let empty = ColdPresence::Empty;
        assert!(!empty.may_contain("s3://b/data/anything.parquet"));
        assert!(empty.is_authoritative());
    }

    #[test]
    fn filter_presence_answers_from_the_bloom() {
        let mut f = ColdPathFilter::with_capacity(100);
        f.insert("s3://b/data/held.parquet");
        let p = ColdPresence::Filter(&f);
        assert!(
            p.may_contain("s3://b/data/held.parquet"),
            "held path vetoes"
        );
        assert!(!p.may_contain("s3://b/data/never-inserted.parquet"));
        assert!(p.is_authoritative());
    }

    #[test]
    fn build_filter_covers_every_path() {
        let mut m = HashMap::new();
        m.insert("leaf-a".to_string(), vec![
            "s3://b/data/1.parquet".to_string(),
            "s3://b/data/2.parquet".to_string(),
        ]);
        m.insert("leaf-b".to_string(), vec![
            "s3://b/data/3.parquet".to_string()
        ]);
        let f = build_filter(&m);
        assert_eq!(f.len(), 3);
        for p in [
            "s3://b/data/1.parquet",
            "s3://b/data/2.parquet",
            "s3://b/data/3.parquet",
        ] {
            assert!(f.maybe_contains(p));
        }
    }

    /// Sizing must scale with the input rather than pinning to the floor, or the
    /// filter saturates at scale and vetoes everything.
    #[test]
    fn sizing_scales_and_stays_within_bounds() {
        let small = ColdPathFilter::with_capacity(10);
        assert_eq!(small.n_bits, MIN_BITS, "tiny inputs take the floor");

        let big = ColdPathFilter::with_capacity(500_000);
        assert!(big.n_bits > small.n_bits);
        assert!(big.n_bits <= MAX_BITS);
        // ~9.6 bits/item at 1% ⇒ ~600 KB for 500 K paths.
        assert!(
            big.bits.len() < 1_000_000,
            "500 K paths should cost well under 1 MB, got {}",
            big.bits.len()
        );
        assert!((1..=16).contains(&big.n_hashes));

        let huge = ColdPathFilter::with_capacity(1_000_000_000);
        assert_eq!(huge.n_bits, MAX_BITS, "clamped rather than unbounded");
    }
}
