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

//! Time-based retention for the cold tier of a tiered V4 table.
//!
//! Old data lives in immutable leaf manifests under the bucket-index. Retention
//! drops the leaves whose data is entirely older than a cutoff, trims the
//! bucket-index, and repoints the root — **never touching the root's live
//! entries or any live data file**. The dropped leaves + their data files become
//! unreferenced and are reclaimed by a separate orphan-GC pass.
//!
//! The cutoff is **time**, which is universal — it does NOT depend on how the
//! table is partitioned. A leaf's event-time is read from its data files'
//! statistics for a configured timestamp field (`ts_field_id`), so retention is
//! fully partition-spec-agnostic: a table partitioned by tenant, by region, by
//! day, or not at all all retain identically.
//!
//! A leaf is KEPT when its max event-time is at/after the cutoff (it still holds
//! data within the window); leaves whose max event-time has fallen behind the
//! cutoff are dropped. A leaf lacking the timestamp statistic is kept
//! (conservative — never drop data we can't time-bound).
//!
//! Cost: O(bucket-index leaves) manifest reads per pass (to read each leaf's
//! timestamp stat) — read-only and run at the retention cadence, never on the
//! hot path. (A future optimization caches each leaf's ts bound in the
//! bucket-index row, making this O(1).)

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;
use crate::spec::bucket_index::{read_bucket_index, write_bucket_index};
use crate::spec::root_manifest::{
    reconstruct_root, write_root_manifest, RootManifestMetadata,
};
use crate::spec::{
    DataFile, FormatVersion, ManifestFile, Operation, PrimitiveLiteral, Snapshot,
    SnapshotReference, SnapshotRetention, Summary, MAIN_BRANCH,
};
use crate::table::Table;
use crate::transaction::action::TransactionAction;
use crate::transaction::snapshot::SnapshotProducer;
use crate::transaction::ActionCommit;
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

const META_ROOT_PATH: &str = "metadata";

/// Keep-vs-drop for one cold leaf.
///
/// Extracted so the sidecar fast path and the manifest fallback provably
/// apply the SAME rule — that equivalence is the whole correctness claim of
/// consulting the sidecar, and inlining it twice would let the two drift.
///
/// `None` (no max-ts derivable) keeps the leaf: dropping data because a stat
/// was missing would be unrecoverable, whereas keeping it costs one more TTL
/// pass.
fn keep_leaf(max_ts: Option<i64>, cutoff_micros: i64) -> bool {
    match max_ts {
        Some(max_ts) => max_ts >= cutoff_micros,
        None => true,
    }
}

/// Action that drops cold leaves whose data is entirely older than the cutoff
/// (time-based retention), trims the bucket-index, and repoints the root.
///
/// Use via `Transaction::drop_cold_buckets(ts_field_id, cutoff_micros)`.
pub struct DropColdBucketsAction {
    ts_field_id: i32,
    cutoff_micros: i64,
    commit_uuid: Uuid,
}

impl DropColdBucketsAction {
    /// Create with the timestamp field id and the retention cutoff (micros).
    /// Leaves whose max value for `ts_field_id` is `< cutoff_micros` are dropped.
    pub fn new(ts_field_id: i32, cutoff_micros: i64) -> Self {
        Self {
            ts_field_id,
            cutoff_micros,
            commit_uuid: Uuid::now_v7(),
        }
    }
}

/// Max value of the `ts_field_id` upper-bound statistic across `files` (event
/// time of the newest row). `None` if no file carries the stat. Pure.
fn max_ts_of(files: &[DataFile], ts_field_id: i32) -> Option<i64> {
    files
        .iter()
        .filter_map(|f| match f.upper_bounds().get(&ts_field_id).map(|d| d.literal()) {
            Some(PrimitiveLiteral::Long(v)) => Some(*v),
            _ => None,
        })
        .max()
}

#[async_trait]
impl TransactionAction for DropColdBucketsAction {
    fn action_name(&self) -> &'static str {
        "drop_cold_buckets"
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if table.effective_format_version() != FormatVersion::V4 {
            return Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!(
                    "drop_cold_buckets requires format version V4 (effective={:?})",
                    table.effective_format_version()
                ),
            ));
        }

        let current_snapshot = match table.metadata().current_snapshot() {
            Some(s) => s,
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        let root_path = current_snapshot.manifest_list();
        let (rm_metadata, root_entries) = reconstruct_root(table.file_io(), root_path).await?;

        let leaves: Vec<ManifestFile> = match &rm_metadata.bucket_index_path {
            Some(path) => {
                let b = table.file_io().new_input(path)?.read().await?;
                read_bucket_index(b)?.leaves().to_vec()
            }
            None => return Ok(ActionCommit::new(vec![], vec![])),
        };

        // Keep a leaf when its newest event-time is at/after the cutoff (or it
        // lacks the stat — conservative). Reading the leaf's data-file stats
        // makes the decision partition-agnostic.
        let mut kept: Vec<ManifestFile> = Vec::new();
        let mut dropped: Vec<ManifestFile> = Vec::new();
        // THE COST. This loads EVERY cold leaf manifest from S3, one at a
        // time, on every TTL run — just to read one number per leaf (max
        // event-time) and decide keep-vs-drop. Nothing is batched, nothing is
        // concurrent, and there is no cache consulted.
        //
        // Measured 2026-08-13: drop_cold_buckets was the single most expensive
        // action in the tick — 98.1s across 3 invocations (45.4s logs, 37.2s
        // metrics_1m, 15.5s metrics) out of 203.1s total actions_ms. Leaf
        // counts at the time were 1282 / 788 / 550, which at ~30-45ms per
        // sequential S3 GET reproduces those timings almost exactly.
        //
        // That makes the cost strictly O(cold leaves) per TTL run, which is
        // why it grows as leaves accumulate and why laminar — which never runs
        // this action — is unaffected on the same tables.
        //
        // Note graduate_buckets already maintains a max-ts SIDECAR for exactly
        // this decision (see its sidecar_hits/sidecar_misses counters). This
        // path does not consult it.
        let scan_start = std::time::Instant::now();
        let n_leaves = leaves.len();
        let mut load_micros: u128 = 0;
        let mut sidecar_hits: usize = 0;
        let mut sidecar_misses: usize = 0;

        // Consult the max-ts sidecar that graduate_buckets already maintains
        // for exactly this keep-vs-drop decision. One read replaces up to N
        // sequential per-leaf manifest GETs. A miss (absent, stale
        // ts_field_id, or a cached `None` "poison" value) falls through to the
        // manifest load below, so behaviour is identical — only the I/O
        // changes. `sidecar_hit` is what enforces that a cached non-answer is
        // NOT trusted; without it an all-None sidecar would permanently
        // suppress TTL on tables whose retention field is not the partition
        // source (e.g. logs on ingestion_time).
        let maxts_sidecar = match &rm_metadata.bucket_index_path {
            Some(bip) => {
                crate::transaction::graduate_buckets::load_maxts_sidecar(
                    table.file_io(),
                    bip,
                    self.ts_field_id,
                )
                .await
            }
            None => None,
        };

        for leaf in leaves {
            // Sidecar first; only pay the S3 GET on a miss.
            let cached = maxts_sidecar
                .as_ref()
                .and_then(|m| {
                    crate::transaction::graduate_buckets::sidecar_hit(m.get(&leaf.manifest_path))
                });
            let keep = if let Some(max_ts) = cached {
                sidecar_hits += 1;
                keep_leaf(Some(max_ts), self.cutoff_micros)
            } else {
                sidecar_misses += 1;
                let load_start = std::time::Instant::now();
                let manifest = leaf.load_manifest(table.file_io()).await?;
                load_micros += load_start.elapsed().as_micros();
                let files: Vec<DataFile> = manifest
                    .entries()
                    .iter()
                    .filter(|e| e.is_alive())
                    .map(|e| e.data_file().clone())
                    .collect();
                keep_leaf(max_ts_of(&files, self.ts_field_id), self.cutoff_micros)
            };
            if keep {
                kept.push(leaf);
            } else {
                dropped.push(leaf);
            }
        }

        log::info!(
            "drop_cold_buckets scan: leaves={} kept={} dropped={} sidecar_hits={} \
             sidecar_misses={} manifest_loads={} load_ms={} scan_ms={}",
            n_leaves,
            kept.len(),
            dropped.len(),
            sidecar_hits,
            sidecar_misses,
            sidecar_misses,
            (load_micros / 1000) as u64,
            scan_start.elapsed().as_millis() as u64
        );

        if dropped.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }
        let dropped_count = dropped.len();

        let snapshot_id = SnapshotProducer::generate_unique_snapshot_id_static(table);
        let next_seq_num = table.metadata().next_sequence_number();
        let schema = table.metadata().current_schema().clone();
        let spec = table.metadata().default_partition_spec().clone();
        let partition_type = spec.partition_type(table.metadata().current_schema())?;
        let commit_uuid = self.commit_uuid;

        let new_bucket_index_path = if kept.is_empty() {
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
            let bi_bytes = write_bucket_index(&kept, &bi_metadata, &partition_type)?;
            table
                .file_io()
                .new_output(&path)?
                .write(bi_bytes.into())
                .await?;
            Some(path)
        };

        let new_rm_metadata = RootManifestMetadata {
            schema: schema.clone(),
            schema_id: table.metadata().current_schema_id(),
            partition_spec: spec.clone(),
            format_version: FormatVersion::V4,
            snapshot_id,
            sequence_number: next_seq_num,
            parent_snapshot_id: table.metadata().current_snapshot_id(),
            bucket_index_path: new_bucket_index_path.clone(),
            prev_root_path: None,
            chain_depth: 0,
            node_level: 0,
            removed_paths: rm_metadata.removed_paths.clone(),
        };
        let new_root_path = format!(
            "{}/{}/root-{}-{}.parquet",
            table.metadata().location(),
            META_ROOT_PATH,
            snapshot_id,
            commit_uuid,
        );
        let root_bytes = write_root_manifest(&root_entries, &new_rm_metadata, &partition_type)?;
        table
            .file_io()
            .new_output(&new_root_path)?
            .write(root_bytes.into())
            .await?;

        let dropped_paths = dropped
            .iter()
            .map(|m| m.manifest_path.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let summary = Summary {
            operation: Operation::Replace,
            additional_properties: HashMap::from([
                ("retention-leaves-dropped".to_string(), dropped_count.to_string()),
                ("retention-leaves-kept".to_string(), kept.len().to_string()),
                ("retention-cutoff-micros".to_string(), self.cutoff_micros.to_string()),
                ("retention-dropped-leaf-paths".to_string(), dropped_paths),
            ]),
        };

        let commit_ts = chrono::Utc::now().timestamp_millis();
        let first_row_id = table.metadata().next_row_id();
        let new_snapshot = Snapshot::builder()
            .with_manifest_list(new_root_path.clone())
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

        let mut manifest_paths = vec![new_root_path];
        if let Some(p) = new_bucket_index_path {
            manifest_paths.push(p);
        }
        Ok(ActionCommit::new(updates, requirements).with_manifest_paths(manifest_paths))
    }
}

#[cfg(test)]
mod tests {
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
        let files = vec![df("a", None), df("b", None)];
        assert_eq!(max_ts_of(&files, 5), None);
    }

    #[test]
    fn max_ts_ignores_other_fields() {
        // Only field 5 carries a stat; asking for field 9 → None.
        let files = vec![df("a", Some((5, 100)))];
        assert_eq!(max_ts_of(&files, 9), None);
    }
}

#[cfg(test)]
mod keep_leaf_tests {
    use super::keep_leaf;

    /// The sidecar fast path and the manifest fallback must agree for every
    /// input — that equivalence is what makes consulting the sidecar safe.
    #[test]
    fn cutoff_boundary_is_inclusive_keep() {
        assert!(keep_leaf(Some(100), 100), "at cutoff must be KEPT");
        assert!(keep_leaf(Some(101), 100));
        assert!(!keep_leaf(Some(99), 100), "strictly older is dropped");
    }

    /// A missing max-ts keeps the leaf. Dropping on absent data would be
    /// unrecoverable; keeping costs one more TTL pass. This is also what makes
    /// a sidecar MISS safe: it falls through to the manifest load, and if that
    /// also yields nothing the leaf survives.
    #[test]
    fn absent_max_ts_keeps() {
        assert!(keep_leaf(None, 100));
        assert!(keep_leaf(None, i64::MAX));
    }

    /// Extremes must not wrap or panic.
    #[test]
    fn extremes_are_sane() {
        assert!(keep_leaf(Some(i64::MAX), 0));
        assert!(!keep_leaf(Some(i64::MIN), 0));
    }
}
