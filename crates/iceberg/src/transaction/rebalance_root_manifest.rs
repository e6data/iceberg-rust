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

/// Write `entries` (all sharing one content type and partition spec) into child
/// manifest(s). When `partition_scoped`, one manifest is written per distinct
/// partition tuple — producing tight (single-partition) summaries the planner
/// can skip on. Otherwise a single manifest is written (legacy behavior).
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
            commit_uuid: Uuid::now_v7(),
        }
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

        // 3. Check if rebalance is needed
        let inline_count = root_manifest.inline_count();
        let needs_flush = inline_count >= self.inline_threshold;
        let needs_mdv_compact = self.needs_mdv_compaction(root_manifest.entries());
        let needs_recluster = self.needs_recluster(root_manifest.entries());

        if !needs_flush && !needs_mdv_compact && !needs_recluster {
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

                // Load the child manifest and collect surviving entries (alive,
                // not MDV-deleted). Rewriting drops the MDV entirely.
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

                let survivors: Vec<ManifestEntry> = manifest
                    .entries()
                    .iter()
                    .enumerate()
                    .filter(|(idx, e)| {
                        e.is_alive()
                            && !mdv_obj
                                .as_ref()
                                .is_some_and(|m| m.is_deleted(*idx as u32))
                    })
                    .map(|(_, e)| {
                        ManifestEntry::builder()
                            .status(ManifestStatus::Existing)
                            .snapshot_id(e.snapshot_id().unwrap_or(0))
                            .sequence_number(e.sequence_number().unwrap_or(0))
                            .file_sequence_number_opt(e.file_sequence_number)
                            .data_file(e.data_file().clone())
                            .build()
                    })
                    .collect();

                let new_manifests = write_entries_clustered(
                    table,
                    &schema,
                    spec.as_ref(),
                    format_version,
                    snapshot_id,
                    commit_uuid,
                    &mut manifest_counter,
                    is_delete,
                    survivors,
                    self.partition_scoped,
                )
                .await?;
                for mf in new_manifests {
                    new_entries.push(RootManifestEntry::ManifestRef {
                        manifest_file: mf,
                        mdv: None,
                    });
                }
            }
        }

        // --- Phase B: Flush inline entries into child manifests ---
        if needs_flush {
            // Build Existing entries grouped by (content_type, partition_spec_id);
            // write_entries_clustered then splits each group per partition tuple
            // when partition-scoped.
            let mut data_by_spec: HashMap<i32, Vec<ManifestEntry>> = HashMap::new();
            let mut delete_by_spec: HashMap<i32, Vec<ManifestEntry>> = HashMap::new();

            for entry in root_manifest.entries() {
                if let RootManifestEntry::Inline(me) = entry {
                    let existing = ManifestEntry::builder()
                        .status(ManifestStatus::Existing)
                        .snapshot_id(me.snapshot_id.unwrap_or(0))
                        .sequence_number(me.sequence_number.unwrap_or(0))
                        .file_sequence_number_opt(me.file_sequence_number)
                        .data_file(me.data_file.clone())
                        .build();
                    match me.data_file.content {
                        DataContentType::Data => {
                            data_by_spec
                                .entry(me.data_file.partition_spec_id)
                                .or_default()
                                .push(existing);
                        }
                        DataContentType::EqualityDeletes
                        | DataContentType::PositionDeletes => {
                            delete_by_spec
                                .entry(me.data_file.partition_spec_id)
                                .or_default()
                                .push(existing);
                        }
                    }
                }
            }

            for (is_delete, by_spec) in
                [(false, data_by_spec), (true, delete_by_spec)]
            {
                for (spec_id, group) in by_spec {
                    let spec = table
                        .metadata()
                        .partition_spec_by_id(spec_id)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("partition spec {spec_id} not found"),
                            )
                        })?;
                    let manifests = write_entries_clustered(
                        table,
                        &schema,
                        spec.as_ref(),
                        format_version,
                        snapshot_id,
                        commit_uuid,
                        &mut manifest_counter,
                        is_delete,
                        group,
                        self.partition_scoped,
                    )
                    .await?;
                    for mf in manifests {
                        new_entries.push(RootManifestEntry::ManifestRef {
                            manifest_file: mf,
                            mdv: None,
                        });
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
