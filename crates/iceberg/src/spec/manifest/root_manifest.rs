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

//! Root manifest format for Iceberg v4.
//!
//! A root manifest is a single Parquet file that replaces the manifest list in v4.
//! It can contain both manifest references (pointing to child manifest files) and
//! inline data/delete file entries, enabling single-file commits for small writes.
//!
//! The file uses a two-section layout with two Parquet row groups:
//! - Row group 0: manifest references (17 columns)
//! - Row group 1: inline entries (21 columns, same schema as parquet_manifest.rs)
//!
//! File-level metadata includes `root-manifest-layout=two-section` to identify
//! the format.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use roaring::RoaringBitmap;

use arrow_array::builder::{BinaryBuilder, Int32Builder, Int64Builder, StringBuilder};
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::parquet_manifest::{
    col_binary_opt, col_i32_opt, col_i64_opt,
    col_str_opt, nullable_i64, read_binary_opt,
    manifest_arrow_schema, manifest_entries_to_record_batch,
    parse_bounds_map_json, parse_i64_map_json, parse_partition_json,
};
use super::{ManifestEntry, ManifestStatus};
use crate::error::Result;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestFile,
    PartitionSpec, SchemaRef, StructType, Transform,
};
use crate::{Error, ErrorKind};

// ============================================================================
// Types
// ============================================================================

/// A single entry in the root manifest file.
#[derive(Debug, Clone)]
pub enum RootManifestEntry {
    /// Reference to a child manifest file with optional delete vector.
    ManifestRef {
        /// The referenced manifest file metadata.
        manifest_file: ManifestFile,
        /// Serialized roaring bitmap marking deleted row indices in the child manifest.
        mdv: Option<Vec<u8>>,
    },
    /// Inline data or delete file entry (same shape as ManifestEntry).
    Inline(ManifestEntry),
}

/// Magic prefix marking a *guarded* MDV envelope (staleness guard present).
///
/// Safe to use as a discriminator because a roaring bitmap serialization always
/// begins with the cookie byte `0x3A` or `0x3B`, never `0x4D` (`'M'`). So a
/// buffer starting with this magic is unambiguously a guarded envelope, and a
/// legacy raw-roaring blob (written before guards existed) never collides.
const MDV_MAGIC: &[u8; 4] = b"MDV1";

/// Staleness guard for an MDV: a snapshot of the child manifest the positional
/// bitmap was computed against.
///
/// The MDV is a *positional* soft-delete — its bitmap holds row indices into a
/// specific child manifest, valid only if that manifest still has the exact same
/// entry order and count at scan time. This guard records the child manifest's
/// entry count plus an order-sensitive checksum of its entry file paths so a
/// stale MDV (child manifest rewritten/reordered under it) is *detected* instead
/// of silently soft-deleting the wrong rows.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MdvGuard {
    entry_count: u32,
    checksum: u64,
}

/// A manifest delete vector: marks specific row indices in a child manifest as
/// logically deleted without rewriting the manifest file.
///
/// Used during compaction (replace_data_files) on V4 tables to soft-delete
/// entries in child manifests. The bitmap is serialized as roaring bitmap bytes
/// and stored in the `mdv_bitmap` column of root manifest reference entries.
///
/// Optionally carries a staleness [`MdvGuard`]: when present it is serialized in
/// a versioned envelope (magic prefix) so readers can validate the bitmap is
/// still positionally aligned with the child manifest before applying it.
#[derive(Debug, Clone)]
pub struct ManifestDeleteVector {
    bitmap: RoaringBitmap,
    guard: Option<MdvGuard>,
}

impl ManifestDeleteVector {
    /// Create an empty MDV.
    pub fn new() -> Self {
        Self {
            bitmap: RoaringBitmap::new(),
            guard: None,
        }
    }

    /// Mark a row index as deleted.
    pub fn mark_deleted(&mut self, row_index: u32) {
        self.bitmap.insert(row_index);
    }

    /// Check if a row index is deleted.
    pub fn is_deleted(&self, row_index: u32) -> bool {
        self.bitmap.contains(row_index)
    }

    /// Number of deleted entries.
    pub fn deleted_count(&self) -> u64 {
        self.bitmap.len()
    }

    /// Check if the MDV is empty (no deletions).
    pub fn is_empty(&self) -> bool {
        self.bitmap.is_empty()
    }

    /// Serialize to bytes for storage in root manifest.
    ///
    /// Without a guard this emits the legacy format — raw roaring bitmap bytes,
    /// byte-for-byte identical to older writers, so old readers keep working.
    /// With a guard it emits a versioned envelope:
    /// `MDV_MAGIC (4) | entry_count u32 LE (4) | checksum u64 LE (8) | roaring bytes`.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        match &self.guard {
            None => {
                let mut buf = Vec::new();
                self.bitmap.serialize_into(&mut buf).map_err(|e| {
                    Error::new(ErrorKind::Unexpected, format!("MDV serialize failed: {e}"))
                })?;
                Ok(buf)
            }
            Some(g) => {
                let mut buf = Vec::with_capacity(16);
                buf.extend_from_slice(MDV_MAGIC);
                buf.extend_from_slice(&g.entry_count.to_le_bytes());
                buf.extend_from_slice(&g.checksum.to_le_bytes());
                self.bitmap.serialize_into(&mut buf).map_err(|e| {
                    Error::new(ErrorKind::Unexpected, format!("MDV serialize failed: {e}"))
                })?;
                Ok(buf)
            }
        }
    }

    /// Deserialize from bytes read from root manifest.
    ///
    /// A buffer beginning with [`MDV_MAGIC`] is a guarded envelope (guard parsed
    /// from the 16-byte header, bitmap from the remainder). Any other buffer is a
    /// legacy raw-roaring blob and yields `guard == None` (a no-op guard).
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        if bytes.len() >= MDV_MAGIC.len() && &bytes[..MDV_MAGIC.len()] == MDV_MAGIC {
            if bytes.len() < 16 {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "MDV envelope too short",
                ));
            }
            let entry_count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
            let checksum = u64::from_le_bytes([
                bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14],
                bytes[15],
            ]);
            let bitmap = RoaringBitmap::deserialize_from(&bytes[16..]).map_err(|e| {
                Error::new(ErrorKind::DataInvalid, format!("MDV deserialize failed: {e}"))
            })?;
            Ok(Self {
                bitmap,
                guard: Some(MdvGuard {
                    entry_count,
                    checksum,
                }),
            })
        } else {
            let bitmap = RoaringBitmap::deserialize_from(bytes).map_err(|e| {
                Error::new(ErrorKind::DataInvalid, format!("MDV deserialize failed: {e}"))
            })?;
            Ok(Self {
                bitmap,
                guard: None,
            })
        }
    }

    /// Merge another MDV into this one (union of deleted indices).
    pub fn merge(&mut self, other: &ManifestDeleteVector) {
        self.bitmap |= &other.bitmap;
    }

    /// Fraction of entries deleted (for compaction threshold checks).
    pub fn deleted_fraction(&self, total_entries: u32) -> f64 {
        if total_entries == 0 {
            return 0.0;
        }
        self.bitmap.len() as f64 / total_entries as f64
    }

    /// Attach (or replace) the staleness guard: the child manifest entry count
    /// and the order-sensitive checksum of its entry file paths this bitmap was
    /// computed against. Call before `serialize` at MDV build time.
    pub fn set_guard(&mut self, entry_count: u32, checksum: u64) {
        self.guard = Some(MdvGuard {
            entry_count,
            checksum,
        });
    }

    /// Order-sensitive FNV-1a 64-bit checksum over a sequence of file paths.
    ///
    /// Each path is length-delimited (its byte length hashed as 8 little-endian
    /// bytes) before its bytes, so `["ab","c"]` and `["a","bc"]` hash
    /// differently. Stable across processes (no `Hash` randomization), which the
    /// guard needs since it is written and validated in different runs.
    pub fn compute_checksum<'a, I: IntoIterator<Item = &'a str>>(paths: I) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x00000100000001b3;
        let mut hash = FNV_OFFSET;
        let fnv = |byte: u8, h: &mut u64| {
            *h ^= byte as u64;
            *h = h.wrapping_mul(FNV_PRIME);
        };
        for path in paths {
            let len = path.len() as u64;
            for b in len.to_le_bytes() {
                fnv(b, &mut hash);
            }
            for b in path.as_bytes() {
                fnv(*b, &mut hash);
            }
        }
        hash
    }

    /// Validate this MDV is still positionally aligned with the child manifest.
    ///
    /// A guardless (legacy) MDV validates as `Ok` — it predates guards, so we
    /// preserve the prior (unchecked) behavior rather than reject it. A guarded
    /// MDV validates only when the recorded entry count and checksum both match
    /// the manifest observed at scan time; otherwise it is stale and applying its
    /// positional bitmap would delete the wrong rows, so we error.
    pub fn validate_against(&self, entry_count: u32, checksum: u64) -> Result<()> {
        match &self.guard {
            None => Ok(()),
            Some(g) => {
                if g.entry_count == entry_count && g.checksum == checksum {
                    Ok(())
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "stale manifest delete vector: guard(count={}, checksum={:#x}) != \
                             manifest(count={}, checksum={:#x})",
                            g.entry_count, g.checksum, entry_count, checksum
                        ),
                    ))
                }
            }
        }
    }
}

impl Default for ManifestDeleteVector {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata stored in root manifest Parquet file-level key-value metadata.
#[derive(Debug, Clone)]
pub struct RootManifestMetadata {
    /// Table schema at the time the root manifest was written.
    pub schema: SchemaRef,
    /// Schema ID.
    pub schema_id: i32,
    /// Partition spec used for inline entries.
    pub partition_spec: Arc<PartitionSpec>,
    /// Format version of the table.
    pub format_version: FormatVersion,
    /// Snapshot ID this root manifest belongs to.
    pub snapshot_id: i64,
    /// Sequence number of the snapshot.
    pub sequence_number: i64,
    /// Parent snapshot ID, if any.
    pub parent_snapshot_id: Option<i64>,
    /// Tiered-metadata pointer: path to the cold **bucket-index** (super-manifest)
    /// listing closed leaf manifests, if this table uses the tiered layout.
    /// `None` for the flat layout. The hot commit path carries this forward
    /// unchanged; only the bucket-close (cold) path updates it. Encoded as the
    /// `bucket-index-path` file-level key; absent key decodes to `None`, so the
    /// change is backward-compatible with pre-tiering root manifests.
    pub bucket_index_path: Option<String>,
    /// Incremental (log-structured) root: pointer to the PREVIOUS root in the
    /// delta chain. `None` ⇒ this root is a **base** (a full entry list);
    /// `Some` ⇒ this root is a **delta** whose `entries` are only the refs ADDED
    /// by its commit, and the full live set is reconstructed by walking
    /// `prev_root_path` back to the base. Encoded as `prev-root-path`; absent
    /// decodes to `None` (a flat/base root) — backward compatible.
    pub prev_root_path: Option<String>,
    /// Number of deltas since the last base (0 for a base). Lets the writer cap
    /// chain length without walking it. Encoded as `chain-depth`.
    pub chain_depth: u32,
    /// Balanced-tree level of this node. `0` = a leaf/flat node whose
    /// `ManifestRef` entries point at real leaf manifests (or are inline data).
    /// `>0` = an INTERIOR node whose `ManifestRef` entries point at child *nodes*
    /// (other root-manifest files) one level down — the reader recurses on them.
    /// The collapse builds a balanced fan-out tree (root level L → … → leaves at
    /// 0) when the live set exceeds the fan-out; small sets stay a single level-0
    /// node (a flat base). Encoded as `node-level`; absent ⇒ 0 (backward compat).
    pub node_level: u32,
    /// Incremental removals: data-file paths this node TOMBSTONES. Lets an
    /// incremental commit that removes files (e.g. laminar's merge-on-write:
    /// remove the small inputs, add the merged output) stay an O(1) delta
    /// instead of collapsing — the delta records the removed paths here, and the
    /// reader (`reconstruct_root` → scan) excludes any data file whose path is in
    /// the union of `removed_paths` down the chain. Newline-joined under the
    /// `removed-paths` key; absent ⇒ empty (backward compatible). Cleared at
    /// collapse for files materialized away; ref-resident removals persist as a
    /// tombstone until the ref is rewritten.
    pub removed_paths: Vec<String>,
}

/// The root manifest: replaces ManifestList in v4.
#[derive(Debug, Clone)]
pub struct RootManifest {
    entries: Vec<RootManifestEntry>,
    metadata: RootManifestMetadata,
}

impl RootManifest {
    /// Create a new root manifest.
    pub fn new(metadata: RootManifestMetadata, entries: Vec<RootManifestEntry>) -> Self {
        Self { entries, metadata }
    }

    /// Get entries slice.
    pub fn entries(&self) -> &[RootManifestEntry] {
        &self.entries
    }

    /// Consume the root manifest and return its entries.
    pub fn into_entries(self) -> Vec<RootManifestEntry> {
        self.entries
    }

    /// Get metadata.
    pub fn metadata(&self) -> &RootManifestMetadata {
        &self.metadata
    }

    /// Iterator over manifest references and their optional delete vectors.
    pub fn manifest_refs(&self) -> impl Iterator<Item = (&ManifestFile, Option<&[u8]>)> {
        self.entries.iter().filter_map(|e| match e {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                Some((manifest_file, mdv.as_deref()))
            }
            RootManifestEntry::Inline(_) => None,
        })
    }

    /// Iterator over inline entries (both data and delete).
    pub fn inline_entries(&self) -> impl Iterator<Item = &ManifestEntry> {
        self.entries.iter().filter_map(|e| match e {
            RootManifestEntry::Inline(entry) => Some(entry),
            RootManifestEntry::ManifestRef { .. } => None,
        })
    }

    /// Count of inline entries.
    pub fn inline_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, RootManifestEntry::Inline(_)))
            .count()
    }

    /// Apply file removals to the root manifest entries.
    ///
    /// For inline entries: removes entries whose file_path is in `paths_to_remove`.
    /// For manifest ref entries: the caller must build MDVs separately by scanning
    /// child manifests to find row indices matching the removed paths.
    ///
    /// Returns the entries with inline removals applied. Manifest refs are unchanged.
    pub fn remove_inline_files(&mut self, paths_to_remove: &HashSet<String>) {
        self.entries.retain(|entry| {
            match entry {
                RootManifestEntry::Inline(me) => {
                    !paths_to_remove.contains(&me.data_file.file_path)
                }
                RootManifestEntry::ManifestRef { .. } => true,
            }
        });
    }

}

// ============================================================================
// Arrow Schema for manifest refs (row group 0)
// ============================================================================

/// Build the Arrow schema for manifest ref entries (row group 0).
///
/// 17 columns covering manifest file metadata plus MDV bitmap.
fn manifest_ref_arrow_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("manifest_path", DataType::Utf8, true),
        Field::new("manifest_length", DataType::Int64, true),
        Field::new("manifest_content", DataType::Int32, true),
        Field::new("manifest_seq_number", DataType::Int64, true),
        Field::new("manifest_min_seq_number", DataType::Int64, true),
        Field::new("manifest_added_snapshot", DataType::Int64, true),
        Field::new("manifest_added_files", DataType::Int32, true),
        Field::new("manifest_existing_files", DataType::Int32, true),
        Field::new("manifest_deleted_files", DataType::Int32, true),
        Field::new("manifest_added_rows", DataType::Int64, true),
        Field::new("manifest_existing_rows", DataType::Int64, true),
        Field::new("manifest_deleted_rows", DataType::Int64, true),
        Field::new("manifest_partitions_json", DataType::Binary, true),
        Field::new("manifest_key_metadata", DataType::Binary, true),
        Field::new("manifest_first_row_id", DataType::Int64, true),
        Field::new("mdv_bitmap", DataType::Binary, true),
        Field::new("partition_spec_id", DataType::Int32, true),
    ])
}

// ============================================================================
// Metadata encoding / decoding
// ============================================================================

fn encode_root_manifest_metadata(metadata: &RootManifestMetadata) -> HashMap<String, String> {
    let mut kv = HashMap::new();
    kv.insert(
        "schema".to_string(),
        serde_json::to_string(metadata.schema.as_ref()).unwrap_or_default(),
    );
    kv.insert("schema-id".to_string(), metadata.schema_id.to_string());
    kv.insert(
        "partition-spec".to_string(),
        serde_json::to_string(&metadata.partition_spec.fields()).unwrap_or_default(),
    );
    kv.insert(
        "partition-spec-id".to_string(),
        metadata.partition_spec.spec_id().to_string(),
    );
    kv.insert(
        "format-version".to_string(),
        (metadata.format_version as u8).to_string(),
    );
    kv.insert("snapshot-id".to_string(), metadata.snapshot_id.to_string());
    kv.insert(
        "sequence-number".to_string(),
        metadata.sequence_number.to_string(),
    );
    if let Some(parent) = metadata.parent_snapshot_id {
        kv.insert("parent-snapshot-id".to_string(), parent.to_string());
    }
    if let Some(path) = &metadata.bucket_index_path {
        kv.insert("bucket-index-path".to_string(), path.clone());
    }
    if let Some(path) = &metadata.prev_root_path {
        kv.insert("prev-root-path".to_string(), path.clone());
    }
    if metadata.chain_depth > 0 {
        kv.insert("chain-depth".to_string(), metadata.chain_depth.to_string());
    }
    if metadata.node_level > 0 {
        kv.insert("node-level".to_string(), metadata.node_level.to_string());
    }
    if !metadata.removed_paths.is_empty() {
        kv.insert("removed-paths".to_string(), metadata.removed_paths.join("\n"));
    }
    kv.insert("root-manifest".to_string(), "true".to_string());
    kv.insert(
        "root-manifest-layout".to_string(),
        "two-section".to_string(),
    );
    // refs-count and inlines-count are set by the writer after separating entries
    kv
}

fn decode_root_manifest_metadata(
    meta: &HashMap<String, String>,
) -> Result<RootManifestMetadata> {
    let schema = Arc::new({
        let s = meta.get("schema").ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "schema is required in root manifest metadata",
            )
        })?;
        serde_json::from_str::<crate::spec::Schema>(s).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to parse schema in root manifest metadata")
                .with_source(e)
        })?
    });

    let schema_id: i32 = meta
        .get("schema-id")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to parse schema-id").with_source(e)
        })?
        .unwrap_or(0);

    let partition_spec = {
        let fields_str = meta.get("partition-spec").ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "partition-spec is required in root manifest metadata",
            )
        })?;
        let fields: Vec<crate::spec::PartitionField> =
            serde_json::from_str(fields_str).map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Failed to parse partition-spec in root manifest metadata",
                )
                .with_source(e)
            })?;
        let spec_id: i32 = meta
            .get("partition-spec-id")
            .map(|s| s.parse())
            .transpose()
            .map_err(|e| {
                Error::new(ErrorKind::DataInvalid, "Failed to parse partition-spec-id")
                    .with_source(e)
            })?
            .unwrap_or(0);
        Arc::new(
            PartitionSpec::builder(schema.clone())
                .with_spec_id(spec_id)
                .add_unbound_fields(fields.into_iter().map(|f| f.into_unbound()))?
                .build()?,
        )
    };

    let format_version = meta
        .get("format-version")
        .map(|s| {
            serde_json::from_str::<FormatVersion>(s).map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Failed to parse format-version in root manifest metadata",
                )
                .with_source(e)
            })
        })
        .transpose()?
        .unwrap_or(FormatVersion::V2);

    let snapshot_id: i64 = meta
        .get("snapshot-id")
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "snapshot-id is required in root manifest metadata",
            )
        })?
        .parse()
        .map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to parse snapshot-id").with_source(e)
        })?;

    let sequence_number: i64 = meta
        .get("sequence-number")
        .ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "sequence-number is required in root manifest metadata",
            )
        })?
        .parse()
        .map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to parse sequence-number").with_source(e)
        })?;

    let parent_snapshot_id: Option<i64> = meta
        .get("parent-snapshot-id")
        .map(|s| {
            s.parse().map_err(|e| {
                Error::new(ErrorKind::DataInvalid, "Failed to parse parent-snapshot-id")
                    .with_source(e)
            })
        })
        .transpose()?;

    // Absent key => flat (non-tiered) layout. Backward-compatible with
    // root manifests written before the tiered-metadata change.
    let bucket_index_path: Option<String> = meta.get("bucket-index-path").cloned();
    // Absent => base/flat root (not part of a delta chain).
    let prev_root_path: Option<String> = meta.get("prev-root-path").cloned();
    let chain_depth: u32 = meta
        .get("chain-depth")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let node_level: u32 = meta
        .get("node-level")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let removed_paths: Vec<String> = meta
        .get("removed-paths")
        .map(|s| s.split('\n').map(|p| p.to_string()).collect())
        .unwrap_or_default();

    Ok(RootManifestMetadata {
        schema,
        schema_id,
        partition_spec,
        format_version,
        snapshot_id,
        sequence_number,
        parent_snapshot_id,
        bucket_index_path,
        prev_root_path,
        chain_depth,
        node_level,
        removed_paths,
    })
}

// ============================================================================
// Writer: RootManifestEntry -> Parquet (two-section layout)
// ============================================================================

/// Write root manifest entries to a Parquet-format byte buffer.
///
/// Uses a two-section layout:
/// - Row group 0: manifest refs (17 columns)
/// - Row group 1: inline entries (21 columns, same schema as parquet_manifest)
///
/// Both row groups are always written, even if empty (0 rows).
pub fn write_root_manifest(
    entries: &[RootManifestEntry],
    metadata: &RootManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<u8>> {
    // Separate entries into refs and inlines
    let mut refs: Vec<(&ManifestFile, Option<&[u8]>)> = Vec::new();
    let mut inlines: Vec<&ManifestEntry> = Vec::new();

    for entry in entries {
        match entry {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                refs.push((manifest_file, mdv.as_deref()));
            }
            RootManifestEntry::Inline(me) => {
                inlines.push(me);
            }
        }
    }

    // Build the combined (superset) schema. Parquet requires all row groups to
    // share the same schema, so we use a union of ref + inline columns (all
    // nullable). Row group 0 populates only ref columns; row group 1 populates
    // only inline columns. The entry type is determined by row group index.
    //
    // Empty batches (0 rows) don't produce row groups in Parquet, so we track
    // counts in file-level metadata so the reader knows the layout.
    let mut kv_metadata = encode_root_manifest_metadata(metadata);
    kv_metadata.insert("refs-count".to_string(), refs.len().to_string());
    kv_metadata.insert("inlines-count".to_string(), inlines.len().to_string());

    let refs_schema = Arc::new(manifest_ref_arrow_schema().with_metadata(kv_metadata));
    let combined_schema = build_combined_schema(&refs_schema);

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, combined_schema.clone(), Some(props)).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to create parquet writer: {e}"),
        )
    })?;

    // Row group 0: manifest refs (only written if non-empty)
    if !refs.is_empty() {
        let rg0_batch = build_combined_batch_for_refs(&refs, &combined_schema, metadata)?;
        writer.write(&rg0_batch).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to write refs batch: {e}"),
            )
        })?;
        if !inlines.is_empty() {
            // Flush to force a new row group boundary before inlines
            writer.flush().map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!("Failed to flush refs row group: {e}"),
                )
            })?;
        }
    }

    // Row group 1 (or 0 if no refs): inline entries (only written if non-empty)
    if !inlines.is_empty() {
        let rg1_batch = build_combined_batch_for_inlines(&inlines, &combined_schema, partition_type, metadata)?;
        writer.write(&rg1_batch).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to write inlines batch: {e}"),
            )
        })?;
    }

    writer.close().map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to close writer: {e}"),
        )
    })?;

    Ok(buf)
}

/// Build the combined (superset) schema for the two-section layout.
///
/// Contains all manifest ref columns + all inline entry columns (no entry_type).
/// All columns are nullable since each row group only populates its own subset.
fn build_combined_schema(refs_schema: &Arc<ArrowSchema>) -> Arc<ArrowSchema> {
    // Manifest ref columns (from refs_schema, preserving metadata)
    let mut fields: Vec<Field> = refs_schema
        .fields()
        .iter()
        .map(|f| {
            // Ensure all fields are nullable
            Field::new(f.name(), f.data_type().clone(), true)
        })
        .collect();

    // Inline entry columns from manifest_arrow_schema
    // Skip partition_spec_id since it's already in refs schema
    let inline_schema = manifest_arrow_schema();
    for field in inline_schema.fields() {
        if field.name() == "partition_spec_id" {
            continue; // Already present from refs schema
        }
        fields.push(Field::new(field.name(), field.data_type().clone(), true));
    }

    let metadata = refs_schema.metadata().clone();
    Arc::new(ArrowSchema::new(fields).with_metadata(metadata))
}

/// Build a combined RecordBatch for row group 0 (manifest refs).
///
/// Ref columns are populated, inline columns are all null.
fn build_combined_batch_for_refs(
    refs: &[(&ManifestFile, Option<&[u8]>)],
    schema: &Arc<ArrowSchema>,
    _metadata: &RootManifestMetadata,
) -> Result<RecordBatch> {
    let n = refs.len();

    // Manifest ref columns
    let mut manifest_path = StringBuilder::with_capacity(n, n * 100);
    let mut manifest_length = Int64Builder::with_capacity(n);
    let mut manifest_content = Int32Builder::with_capacity(n);
    let mut manifest_seq_number = Int64Builder::with_capacity(n);
    let mut manifest_min_seq_number = Int64Builder::with_capacity(n);
    let mut manifest_added_snapshot = Int64Builder::with_capacity(n);
    let mut manifest_added_files = Int32Builder::with_capacity(n);
    let mut manifest_existing_files = Int32Builder::with_capacity(n);
    let mut manifest_deleted_files = Int32Builder::with_capacity(n);
    let mut manifest_added_rows = Int64Builder::with_capacity(n);
    let mut manifest_existing_rows = Int64Builder::with_capacity(n);
    let mut manifest_deleted_rows = Int64Builder::with_capacity(n);
    let mut manifest_partitions_json = BinaryBuilder::with_capacity(n, n * 128);
    let mut manifest_key_metadata = BinaryBuilder::with_capacity(n, n * 16);
    let mut manifest_first_row_id = Int64Builder::with_capacity(n);
    let mut mdv_bitmap = BinaryBuilder::with_capacity(n, n * 64);
    let mut part_spec_id = Int32Builder::with_capacity(n);

    for (mf, mdv) in refs {
        manifest_path.append_value(&mf.manifest_path);
        manifest_length.append_value(mf.manifest_length);
        manifest_content.append_value(mf.content as i32);
        manifest_seq_number.append_value(mf.sequence_number);
        manifest_min_seq_number.append_value(mf.min_sequence_number);
        manifest_added_snapshot.append_value(mf.added_snapshot_id);
        match mf.added_files_count {
            Some(v) => manifest_added_files.append_value(v as i32),
            None => manifest_added_files.append_null(),
        }
        match mf.existing_files_count {
            Some(v) => manifest_existing_files.append_value(v as i32),
            None => manifest_existing_files.append_null(),
        }
        match mf.deleted_files_count {
            Some(v) => manifest_deleted_files.append_value(v as i32),
            None => manifest_deleted_files.append_null(),
        }
        match mf.added_rows_count {
            Some(v) => manifest_added_rows.append_value(v as i64),
            None => manifest_added_rows.append_null(),
        }
        match mf.existing_rows_count {
            Some(v) => manifest_existing_rows.append_value(v as i64),
            None => manifest_existing_rows.append_null(),
        }
        match mf.deleted_rows_count {
            Some(v) => manifest_deleted_rows.append_value(v as i64),
            None => manifest_deleted_rows.append_null(),
        }
        match &mf.partitions {
            Some(parts) => {
                let json = serde_json::to_vec(parts).unwrap_or_default();
                manifest_partitions_json.append_value(&json);
            }
            None => manifest_partitions_json.append_null(),
        }
        match &mf.key_metadata {
            Some(km) => manifest_key_metadata.append_value(km.as_slice()),
            None => manifest_key_metadata.append_null(),
        }
        match mf.first_row_id {
            Some(v) => manifest_first_row_id.append_value(v as i64),
            None => manifest_first_row_id.append_null(),
        }
        match mdv {
            Some(bm) => mdv_bitmap.append_value(bm),
            None => mdv_bitmap.append_null(),
        }
        part_spec_id.append_value(mf.partition_spec_id);
    }

    // Build null columns for all inline fields
    let inline_schema = manifest_arrow_schema();
    let mut null_inline_columns: Vec<ArrayRef> = Vec::new();
    for field in inline_schema.fields() {
        if field.name() == "partition_spec_id" {
            continue;
        }
        null_inline_columns.push(make_null_array(field.data_type(), n));
    }

    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(manifest_path.finish()),
        Arc::new(manifest_length.finish()),
        Arc::new(manifest_content.finish()),
        Arc::new(manifest_seq_number.finish()),
        Arc::new(manifest_min_seq_number.finish()),
        Arc::new(manifest_added_snapshot.finish()),
        Arc::new(manifest_added_files.finish()),
        Arc::new(manifest_existing_files.finish()),
        Arc::new(manifest_deleted_files.finish()),
        Arc::new(manifest_added_rows.finish()),
        Arc::new(manifest_existing_rows.finish()),
        Arc::new(manifest_deleted_rows.finish()),
        Arc::new(manifest_partitions_json.finish()),
        Arc::new(manifest_key_metadata.finish()),
        Arc::new(manifest_first_row_id.finish()),
        Arc::new(mdv_bitmap.finish()),
        Arc::new(part_spec_id.finish()),
    ];
    columns.extend(null_inline_columns);

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to build refs RecordBatch: {e}")))
}

/// Build a combined RecordBatch for row group 1 (inline entries).
///
/// Inline columns are populated, ref columns are all null.
fn build_combined_batch_for_inlines<E: std::borrow::Borrow<ManifestEntry>>(
    entries: &[E],
    schema: &Arc<ArrowSchema>,
    partition_type: &StructType,
    metadata: &RootManifestMetadata,
) -> Result<RecordBatch> {
    let n = entries.len();

    // Build null columns for all ref fields (17 columns from manifest_ref_arrow_schema)
    let ref_schema = manifest_ref_arrow_schema();
    let mut null_ref_columns: Vec<ArrayRef> = Vec::new();
    for field in ref_schema.fields() {
        null_ref_columns.push(make_null_array(field.data_type(), n));
    }

    // Build inline columns using manifest_entries_to_record_batch
    let inline_schema = Arc::new(manifest_arrow_schema());
    let inline_batch = manifest_entries_to_record_batch(
        entries,
        &inline_schema,
        partition_type,
        metadata.format_version,
    )?;

    // Combine: ref null columns + inline columns (skip partition_spec_id from inline since it's in ref null columns)
    let mut columns: Vec<ArrayRef> = null_ref_columns;

    let inline_schema_tmp = manifest_arrow_schema();
    let inline_field_names: Vec<&str> = inline_schema_tmp
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();

    for (idx, name) in inline_field_names.iter().enumerate() {
        if *name == "partition_spec_id" {
            // partition_spec_id is already covered by the ref schema null column;
            // we need to replace that null column with the actual inline values
            // Find the index of partition_spec_id in ref_schema
            let ref_psi_idx = ref_schema
                .fields()
                .iter()
                .position(|f| f.name() == "partition_spec_id")
                .unwrap();
            columns[ref_psi_idx] = inline_batch.column(idx).clone();
            continue;
        }
        columns.push(inline_batch.column(idx).clone());
    }

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to build inlines RecordBatch: {e}")))
}

/// Create a null array of the given data type and length.
fn make_null_array(data_type: &DataType, n: usize) -> ArrayRef {
    match data_type {
        DataType::Int32 => {
            let mut b = Int32Builder::with_capacity(n);
            for _ in 0..n {
                b.append_null();
            }
            Arc::new(b.finish())
        }
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(n);
            for _ in 0..n {
                b.append_null();
            }
            Arc::new(b.finish())
        }
        DataType::Utf8 => {
            let mut b = StringBuilder::with_capacity(n, 0);
            for _ in 0..n {
                b.append_null();
            }
            Arc::new(b.finish())
        }
        DataType::Binary => {
            let mut b = BinaryBuilder::with_capacity(n, 0);
            for _ in 0..n {
                b.append_null();
            }
            Arc::new(b.finish())
        }
        _ => {
            // Fallback: use arrow's null array
            Arc::new(arrow_array::new_null_array(data_type, n))
        }
    }
}

// ============================================================================
// Reader: Parquet -> RootManifestEntry (two-section layout)
// ============================================================================

/// Read root manifest from Parquet bytes.
///
/// Expects a two-section layout:
/// - Row group 0: manifest refs
/// - Row group 1: inline entries
///
/// The layout is identified by `root-manifest-layout=two-section` in file metadata.
pub fn read_root_manifest(
    bytes: Bytes,
) -> Result<(RootManifestMetadata, Vec<RootManifestEntry>)> {
    let reader_builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to open root manifest: {e}"),
            )
        })?;

    let arrow_schema = reader_builder.schema();
    let arrow_meta = arrow_schema.metadata();

    // Decode metadata
    let metadata = decode_root_manifest_metadata(arrow_meta)?;
    let partition_type = metadata.partition_spec.partition_type(&metadata.schema)?;

    let parquet_meta = reader_builder.metadata().clone();
    let num_row_groups = parquet_meta.num_row_groups();

    // Determine layout from metadata counts.
    // refs-count and inlines-count tell us how many entries of each type exist.
    // The writer only creates row groups for non-empty sections:
    // - Both non-empty: RG0=refs, RG1=inlines (2 row groups)
    // - Only refs: RG0=refs (1 row group)
    // - Only inlines: RG0=inlines (1 row group)
    // - Both empty: 0 row groups
    let refs_count: usize = arrow_meta
        .get("refs-count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let inlines_count: usize = arrow_meta
        .get("inlines-count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let has_refs = refs_count > 0;
    let has_inlines = inlines_count > 0;

    let mut entries = Vec::new();

    // Determine row group indices
    let refs_rg: Option<usize> = if has_refs && num_row_groups > 0 {
        Some(0)
    } else {
        None
    };
    let inlines_rg: Option<usize> = if has_inlines {
        if has_refs && num_row_groups > 1 {
            Some(1) // RG1 when refs occupy RG0
        } else if !has_refs && num_row_groups > 0 {
            Some(0) // RG0 when no refs
        } else {
            None
        }
    } else {
        None
    };

    // Read manifest refs from their row group
    if let Some(rg_idx) = refs_rg {
        let rg_reader = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to open root manifest for refs: {e}"),
                )
            })?
            .with_row_groups(vec![rg_idx])
            .build()
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to build refs reader: {e}"),
                )
            })?;

        for batch_result in rg_reader {
            let batch = batch_result.map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to read refs batch: {e}"),
                )
            })?;
            let ref_entries = record_batch_to_manifest_refs(&batch, &metadata)?;
            entries.extend(ref_entries);
        }
    }

    // Read inline entries from their row group
    if let Some(rg_idx) = inlines_rg {
        let rg_reader = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to open root manifest for inlines: {e}"),
                )
            })?
            .with_row_groups(vec![rg_idx])
            .build()
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to build inlines reader: {e}"),
                )
            })?;

        for batch_result in rg_reader {
            let batch = batch_result.map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to read inlines batch: {e}"),
                )
            })?;
            let inline_entries = record_batch_to_inline_entries(&batch, &metadata, &partition_type)?;
            entries.extend(inline_entries);
        }
    }

    Ok((metadata, entries))
}

/// Predicate deciding whether a reconstructed *data-file* entry is materialized.
/// Lets a caller prune the live set to a partition/time band while walking the
/// chain + tree, instead of building the full set and filtering after — the
/// difference between O(whole table) and O(band) peak memory.
///
/// `None` = keep everything (the default). Every caller EXCEPT the closed-hour
/// consolidation sweep passes `None`: graduation's collapse-fold, cold-tier
/// compaction, drop-cold-buckets, rebalance, and the read path all require the
/// complete live set for correctness. Only the sweep — which acts solely on the
/// closed-but-not-graduated hour band — supplies a filter. The predicate applies
/// to `Inline` (materialized data) entries only; `ManifestRef` entries pass
/// through unchanged (interior tree-node refs carry no partition summary yet, so
/// a subtree can't be skipped wholesale here; the caller prunes any refs it then
/// loads).
pub type EntryFilter<'a> = &'a (dyn Fn(&DataFile) -> bool + Sync);

/// `true` if the entry survives the filter. `ManifestRef`s always survive
/// (pass-through); `Inline` data entries are tested against the predicate.
fn entry_passes(entry: &RootManifestEntry, keep: Option<EntryFilter<'_>>) -> bool {
    match keep {
        None => true,
        Some(k) => match entry {
            RootManifestEntry::Inline(me) => k(me.data_file()),
            RootManifestEntry::ManifestRef { .. } => true,
        },
    }
}

/// Reconstruct the full entry set of an incremental (chained) root by walking
/// `prev_root_path` from `head_path` back to the base, unioning each delta's
/// entries. Returns the entries plus the HEAD metadata (its `bucket_index_path`
/// etc. are authoritative — carried on the most recent root).
///
/// The live tier is append-only: a delta's entries are only the refs ADDED by
/// its commit, and refs leave the root only via lifecycle ops that write a fresh
/// base. So the live set is simply the union down the chain — no removal
/// bookkeeping. For a base/flat root (`prev_root_path == None`) this is one read.
///
/// `MAX_CHAIN_WALK` bounds the walk defensively against a corrupt/cyclic chain.
///
/// The returned metadata's `removed_paths` holds the tombstones that still apply
/// to files inside child manifest **refs** (inline tombstones are materialized
/// away here). Callers that write a new base carry this forward; the scan reads
/// it to skip those data files. So `meta.removed_paths` on the result is the
/// authoritative ref-tombstone set, NOT the head node's raw field.
///
/// This is the full-live-set reconstruct every caller but the closed-hour sweep
/// uses; see [`reconstruct_root_filtered`] for the band-pruned variant.
pub async fn reconstruct_root(
    file_io: &crate::io::FileIO,
    head_path: &str,
) -> Result<(RootManifestMetadata, Vec<RootManifestEntry>)> {
    reconstruct_root_filtered(file_io, head_path, None).await
}

/// Like [`reconstruct_root`] but prunes materialized data entries by `keep` as
/// it walks — leaf entries failing the predicate are never accumulated, so peak
/// memory scales with the kept band, not the whole tiered tree. See
/// [`EntryFilter`]. `keep = None` is identical to [`reconstruct_root`].
pub async fn reconstruct_root_filtered(
    file_io: &crate::io::FileIO,
    head_path: &str,
    keep: Option<EntryFilter<'_>>,
) -> Result<(RootManifestMetadata, Vec<RootManifestEntry>)> {
    const MAX_CHAIN_WALK: usize = 100_000;
    let mut entries: Vec<RootManifestEntry> = Vec::new();
    let mut removed: HashSet<String> = HashSet::new();
    let mut head_meta: Option<RootManifestMetadata> = None;
    let mut path = head_path.to_string();
    for _ in 0..MAX_CHAIN_WALK {
        let bytes = file_io.new_input(&path)?.read().await?;
        let (meta, mut these) = read_root_manifest(bytes)?;
        if head_meta.is_none() {
            head_meta = Some(meta.clone());
        }
        // Union the tombstones recorded down the chain. Collected in FULL even
        // when filtering (they are cheap path strings, and a tombstone for a
        // kept file must still drop it in finalize).
        for p in &meta.removed_paths {
            removed.insert(p.clone());
        }
        if meta.node_level > 0 {
            // Reached a balanced-tree base (the bottom of the L0 chain): its
            // entries are child-node refs, not data. Traverse the subtree to
            // gather the real leaf entries (filtered per `keep`), then stop (a
            // tree base has no prev).
            collect_subtree_entries(file_io, these, &mut entries, keep).await?;
            return Ok(finalize_reconstruct(
                head_meta.expect("read at least one root"),
                entries,
                removed,
            ));
        }
        // L0 delta: drop out-of-band inline entries before accumulating.
        these.retain(|e| entry_passes(e, keep));
        entries.append(&mut these);
        match &meta.prev_root_path {
            Some(prev) => path = prev.clone(),
            None => {
                return Ok(finalize_reconstruct(
                    head_meta.expect("read at least one root"),
                    entries,
                    removed,
                ));
            }
        }
    }
    Err(Error::new(
        ErrorKind::DataInvalid,
        "reconstruct_root: chain exceeded MAX_CHAIN_WALK (corrupt/cyclic prev-root-path?)",
    ))
}

/// Collect every root-manifest object path in the L0 delta chain from `head_path`
/// back to (and including) the base — i.e. every root file a collapse replaces
/// and thus orphans. Reads only each root's metadata (`prev_root_path`); no entry
/// or subtree materialization, so it's a cheap sequence of small-object reads.
/// Used by the collapse path to tombstone orphaned root deltas (metadata GC).
pub async fn chain_root_paths(
    file_io: &crate::io::FileIO,
    head_path: &str,
) -> Result<Vec<String>> {
    const MAX_CHAIN_WALK: usize = 100_000;
    let mut paths: Vec<String> = Vec::new();
    let mut path = head_path.to_string();
    for _ in 0..MAX_CHAIN_WALK {
        let bytes = file_io.new_input(&path)?.read().await?;
        let (meta, _) = read_root_manifest(bytes)?;
        paths.push(path.clone());
        // node_level>0 is a balanced-tree base (no prev). An L0 delta with a
        // prev continues the chain; the base (prev=None) ends it.
        match meta.prev_root_path {
            Some(prev) if meta.node_level == 0 => path = prev,
            _ => return Ok(paths),
        }
    }
    Err(Error::new(
        ErrorKind::DataInvalid,
        "chain_root_paths: chain exceeded MAX_CHAIN_WALK (corrupt/cyclic prev-root-path?)",
    ))
}

/// Apply the chain's tombstones to the reconstructed entries: drop any INLINE
/// entry whose data file was removed, and any leaf REF whose manifest path was
/// removed (materializing both removals), then set `meta.removed_paths` to the
/// removals that matched neither — these reference files living inside child
/// manifest refs, so they remain a tombstone the scan must apply when it reads
/// those manifests. Pure.
fn finalize_reconstruct(
    mut meta: RootManifestMetadata,
    entries: Vec<RootManifestEntry>,
    removed: HashSet<String>,
) -> (RootManifestMetadata, Vec<RootManifestEntry>) {
    if removed.is_empty() {
        meta.removed_paths = Vec::new();
        return (meta, entries);
    }
    let mut matched: HashSet<String> = HashSet::new();
    let kept: Vec<RootManifestEntry> = entries
        .into_iter()
        .filter(|e| match e {
            RootManifestEntry::Inline(me) => {
                if removed.contains(&me.data_file.file_path) {
                    matched.insert(me.data_file.file_path.clone());
                    false
                } else {
                    true
                }
            }
            // A leaf ref is tombstoned by its MANIFEST path. Graduation moves a
            // ref into the cold bucket-index and records its path here; without
            // this arm the ref survives in the base root beneath the delta and
            // is re-graduated on every later tick, appending a duplicate to the
            // bucket-index each time. Measured on sri-olly 2026-08-16:
            // 230,456 index rows for 18,053 distinct leaves (92.2% duplicates),
            // one leaf repeated 74x, growing ~3,938 rows/tick unbounded.
            //
            // Widening the match is inert for every other consumer: they all
            // compare `removed_paths` against DATA-file paths, which never
            // collide with manifest paths, and nothing physically deletes off
            // this set (reclaim runs off the JSONL tombstone ledger).
            RootManifestEntry::ManifestRef { manifest_file, .. } => {
                if removed.contains(&manifest_file.manifest_path) {
                    matched.insert(manifest_file.manifest_path.clone());
                    false
                } else {
                    true
                }
            }
        })
        .collect();
    let mut for_refs: Vec<String> = removed.difference(&matched).cloned().collect();
    for_refs.sort();
    meta.removed_paths = for_refs;
    (meta, kept)
}

/// Recursively gather the leaf-level entries under a set of interior-node refs.
/// Each ref's `manifest_path` is a child node; a child at `node_level == 0` is a
/// leaf (its entries are real manifest refs / inline data → collected), a child
/// at `node_level > 0` is interior (recurse).
fn collect_subtree_entries<'a>(
    file_io: &'a crate::io::FileIO,
    interior_entries: Vec<RootManifestEntry>,
    out: &'a mut Vec<RootManifestEntry>,
    keep: Option<EntryFilter<'a>>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
    Box::pin(async move {
        for entry in interior_entries {
            let RootManifestEntry::ManifestRef { manifest_file, .. } = entry else {
                // Interior nodes hold only node refs; ignore stray inline defensively.
                continue;
            };
            let bytes = file_io.new_input(&manifest_file.manifest_path)?.read().await?;
            let (child_meta, mut child_entries) = read_root_manifest(bytes)?;
            if child_meta.node_level == 0 {
                // Leaf: filter its data entries to the band before accumulating.
                // The leaf is read in full (one at a time — bounded transient),
                // but only kept entries persist in `out`, so the accumulated set
                // is O(band), not O(whole tree).
                child_entries.retain(|e| entry_passes(e, keep));
                out.extend(child_entries);
            } else {
                collect_subtree_entries(file_io, child_entries, out, keep).await?;
            }
        }
        Ok(())
    })
}

/// Reorder `entries` so all entries of the same partition tuple are contiguous.
/// [`build_balanced_tree`] then packs each leaf node from a contiguous slice, so
/// every leaf comes out **partition-tight** (its partition summary spans one —
/// or, only at a chunk boundary, a couple of — partitions) instead of the
/// arbitrary wide spread that fan-out-order chunking of the reconstructed set
/// produces.
///
/// This is the manifest-layer compaction the standalone rebalance did, folded
/// into the (single-writer, race-free) collapse: the reader prunes leaves by
/// partition summary and graduation folds fewer wide manifests — with no extra
/// commit, no second writer, and no dependency on lakekeeper being linearizable
/// (the standalone rebalance's data-loss trap).
///
/// Safe: the reconstructed live set is order-independent (a set), so reordering
/// changes only which leaf a data file lands in, never which files are present.
///
/// Two entry shapes are clustered independently:
/// - **Inline** data files → grouped by their full partition tuple (a
///   HashMap group per partition), the same as before.
/// - **`ManifestRef`** entries → sorted by their time-transform partition
///   **bucket** (the `Hour`/`Day` ordinal read from the manifest's partition
///   summary, no manifest load). On a **tiered** table the reconstructed live
///   set is entirely manifest refs (not inline files), so this is the branch
///   that actually tightens its leaves: same-hour refs become contiguous, so
///   each leaf chunk spans one — or, only at a chunk boundary, two adjacent —
///   time buckets instead of an arbitrary 4-hour spread. The reader then prunes
///   whole leaves by time range and graduation folds time-contiguous leaves.
///   A stable sort preserves any within-hour partition locality already present
///   in the reconstructed order. Tables with no time-transform partition field
///   leave the refs in reconstruction order (no worse than before).
fn cluster_entries_by_partition(
    entries: Vec<RootManifestEntry>,
    spec: &PartitionSpec,
) -> Vec<RootManifestEntry> {
    let mut by_part: HashMap<crate::spec::Struct, Vec<RootManifestEntry>> = HashMap::new();
    let mut refs: Vec<RootManifestEntry> = Vec::new();
    for e in entries {
        match &e {
            RootManifestEntry::Inline(me) => {
                let p = me.data_file.partition.clone();
                by_part.entry(p).or_default().push(e);
            }
            RootManifestEntry::ManifestRef { .. } => refs.push(e),
        }
    }
    // Sort manifest refs by their time-transform bucket so same-hour refs land in
    // the same leaf. Falls back to reconstruction order when the table has no
    // monotonic time-transform partition field.
    if let Some(idx) = spec
        .fields()
        .iter()
        .position(|pf| matches!(pf.transform, Transform::Hour | Transform::Day))
    {
        refs.sort_by_key(|e| match e {
            RootManifestEntry::ManifestRef { manifest_file, .. } => {
                ref_time_bucket(manifest_file, idx)
            }
            _ => i64::MAX,
        });
    }
    let mut out: Vec<RootManifestEntry> =
        Vec::with_capacity(refs.len() + by_part.values().map(Vec::len).sum::<usize>());
    for group in by_part.into_values() {
        out.extend(group);
    }
    out.extend(refs);
    out
}

/// Time-transform bucket ordinal (`Hour`/`Day` value) of a manifest ref, read
/// from its partition summary lower bound at field index `idx` (Iceberg
/// little-endian single-value encoding). Refs missing the summary sort last so
/// they don't split a well-formed run. Pure.
fn ref_time_bucket(mf: &ManifestFile, idx: usize) -> i64 {
    mf.partitions
        .as_ref()
        .and_then(|parts| parts.get(idx))
        .and_then(|fs| fs.lower_bound.as_ref())
        .and_then(|b| (b.len() >= 4).then(|| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64))
        .unwrap_or(i64::MAX)
}

/// Build a balanced fan-out tree over `entries` and return the path of its root
/// node (already written, along with every interior/leaf node). The root carries
/// the template's `bucket_index_path`, `node_level = tree height`, and
/// `prev_root_path = None` (it is a collapse base, the bottom of the L0 chain).
///
/// When `entries.len() <= fanout` this writes a single level-0 node (a flat base
/// — identical to the pre-tree collapse), so small live sets pay nothing. Above
/// the fan-out the entries are **partition-clustered** (see
/// [`cluster_entries_by_partition`]) and chunked bottom-up: partition-tight
/// level-0 leaf nodes, then interior levels of node refs, until one root remains.
///
/// Interior node refs carry `partitions: None` (conservative) for now — the
/// bounds aggregation that lets the reader PRUNE whole subtrees is the paired
/// read-side increment; today `reconstruct_root` still visits every leaf, but
/// the leaves themselves are now partition-tight so the read side prunes them.
#[allow(clippy::too_many_arguments)]
pub async fn build_balanced_tree(
    file_io: &crate::io::FileIO,
    location: &str,
    template: &RootManifestMetadata,
    partition_type: &crate::spec::StructType,
    commit_uuid: uuid::Uuid,
    entries: Vec<RootManifestEntry>,
    fanout: usize,
) -> Result<String> {
    let fanout = fanout.max(2);
    let root_path = format!(
        "{}/metadata/root-{}-{}.parquet",
        location, template.snapshot_id, commit_uuid
    );

    // Small set → a single flat level-0 base (no tree). Carries the pointer.
    if entries.len() <= fanout {
        let meta = RootManifestMetadata {
            node_level: 0,
            prev_root_path: None,
            chain_depth: 0,
            ..template.clone()
        };
        let bytes = write_root_manifest(&entries, &meta, partition_type)?;
        file_io.new_output(&root_path)?.write(bytes.into()).await?;
        return Ok(root_path);
    }

    // Cluster by partition so each leaf chunk is partition-/time-tight (see
    // cluster_entries_by_partition). This is the hot-side manifest compaction,
    // done for free during the collapse's base rewrite. On tiered tables the
    // entries are manifest refs, so this clusters them by their time bucket.
    let entries = cluster_entries_by_partition(entries, &template.partition_spec);

    let mut counter: u64 = 0;
    let node_ref = |path: &str| RootManifestEntry::ManifestRef {
        manifest_file: ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: 0,
            partition_spec_id: template.partition_spec.spec_id(),
            content: ManifestContentType::Data,
            sequence_number: template.sequence_number,
            min_sequence_number: template.sequence_number,
            added_snapshot_id: template.snapshot_id,
            added_files_count: None,
            existing_files_count: None,
            deleted_files_count: None,
            added_rows_count: None,
            existing_rows_count: None,
            deleted_rows_count: None,
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        },
        mdv: None,
    };

    // Level 0: pack the real entries into leaf nodes.
    let mut child_refs: Vec<RootManifestEntry> = Vec::new();
    for chunk in entries.chunks(fanout) {
        let path = format!(
            "{}/metadata/tree-{}-{}-n{}.parquet",
            location, template.snapshot_id, commit_uuid, counter
        );
        counter += 1;
        let meta = RootManifestMetadata {
            node_level: 0,
            prev_root_path: None,
            chain_depth: 0,
            bucket_index_path: None,
            ..template.clone()
        };
        let bytes = write_root_manifest(chunk, &meta, partition_type)?;
        file_io.new_output(&path)?.write(bytes.into()).await?;
        child_refs.push(node_ref(&path));
    }

    // Interior levels until a single node remains.
    let mut level: u32 = 1;
    while child_refs.len() > fanout {
        let mut parents: Vec<RootManifestEntry> = Vec::new();
        for chunk in child_refs.chunks(fanout) {
            let path = format!(
                "{}/metadata/tree-{}-{}-n{}.parquet",
                location, template.snapshot_id, commit_uuid, counter
            );
            counter += 1;
            let meta = RootManifestMetadata {
                node_level: level,
                prev_root_path: None,
                chain_depth: 0,
                bucket_index_path: None,
                ..template.clone()
            };
            let bytes = write_root_manifest(chunk, &meta, partition_type)?;
            file_io.new_output(&path)?.write(bytes.into()).await?;
            parents.push(node_ref(&path));
        }
        child_refs = parents;
        level += 1;
    }

    // Root node: the remaining ≤fanout child refs; carries the pointer + height.
    let root_meta = RootManifestMetadata {
        node_level: level,
        prev_root_path: None,
        chain_depth: 0,
        ..template.clone()
    };
    let bytes = write_root_manifest(&child_refs, &root_meta, partition_type)?;
    file_io.new_output(&root_path)?.write(bytes.into()).await?;
    Ok(root_path)
}

/// Parse manifest ref entries from a RecordBatch (row group 0).
fn record_batch_to_manifest_refs(
    batch: &RecordBatch,
    metadata: &RootManifestMetadata,
) -> Result<Vec<RootManifestEntry>> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let manifest_path_arr = col_str_opt(batch, "manifest_path");
    let manifest_length_arr = col_i64_opt(batch, "manifest_length");
    let manifest_content_arr = col_i32_opt(batch, "manifest_content");
    let manifest_seq_number_arr = col_i64_opt(batch, "manifest_seq_number");
    let manifest_min_seq_number_arr = col_i64_opt(batch, "manifest_min_seq_number");
    let manifest_added_snapshot_arr = col_i64_opt(batch, "manifest_added_snapshot");
    let manifest_added_files_arr = col_i32_opt(batch, "manifest_added_files");
    let manifest_existing_files_arr = col_i32_opt(batch, "manifest_existing_files");
    let manifest_deleted_files_arr = col_i32_opt(batch, "manifest_deleted_files");
    let manifest_added_rows_arr = col_i64_opt(batch, "manifest_added_rows");
    let manifest_existing_rows_arr = col_i64_opt(batch, "manifest_existing_rows");
    let manifest_deleted_rows_arr = col_i64_opt(batch, "manifest_deleted_rows");
    let manifest_partitions_json_arr = col_binary_opt(batch, "manifest_partitions_json");
    let manifest_key_metadata_arr = col_binary_opt(batch, "manifest_key_metadata");
    let manifest_first_row_id_arr = col_i64_opt(batch, "manifest_first_row_id");
    let mdv_bitmap_arr = col_binary_opt(batch, "mdv_bitmap");
    let part_spec_id_arr = col_i32_opt(batch, "partition_spec_id");

    for i in 0..n {
        let mf_path = manifest_path_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .unwrap_or_default();
        let mf_length = nullable_i64(manifest_length_arr, i).unwrap_or(0);
        let mf_content: ManifestContentType = manifest_content_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i))
                }
            })
            .unwrap_or(0)
            .try_into()?;
        let mf_seq = nullable_i64(manifest_seq_number_arr, i).unwrap_or(0);
        let mf_min_seq = nullable_i64(manifest_min_seq_number_arr, i).unwrap_or(0);
        let mf_added_snap = nullable_i64(manifest_added_snapshot_arr, i).unwrap_or(0);

        let mf_added_files = manifest_added_files_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i) as u32)
                }
            });
        let mf_existing_files = manifest_existing_files_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i) as u32)
                }
            });
        let mf_deleted_files = manifest_deleted_files_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i) as u32)
                }
            });
        let mf_added_rows = nullable_i64(manifest_added_rows_arr, i).map(|v| v as u64);
        let mf_existing_rows =
            nullable_i64(manifest_existing_rows_arr, i).map(|v| v as u64);
        let mf_deleted_rows =
            nullable_i64(manifest_deleted_rows_arr, i).map(|v| v as u64);

        let partitions: Option<Vec<crate::spec::FieldSummary>> =
            read_binary_opt(manifest_partitions_json_arr, i)
                .and_then(|b| serde_json::from_slice(b).ok());

        let mf_key_metadata =
            read_binary_opt(manifest_key_metadata_arr, i).map(|b| b.to_vec());

        let mf_first_row_id =
            nullable_i64(manifest_first_row_id_arr, i).map(|v| v as u64);

        let mdv = read_binary_opt(mdv_bitmap_arr, i).map(|b| b.to_vec());

        let spec_id = part_spec_id_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i))
                }
            })
            .unwrap_or(metadata.partition_spec.spec_id());

        let manifest_file = ManifestFile {
            manifest_path: mf_path,
            manifest_length: mf_length,
            partition_spec_id: spec_id,
            content: mf_content,
            sequence_number: mf_seq,
            min_sequence_number: mf_min_seq,
            added_snapshot_id: mf_added_snap,
            added_files_count: mf_added_files,
            existing_files_count: mf_existing_files,
            deleted_files_count: mf_deleted_files,
            added_rows_count: mf_added_rows,
            existing_rows_count: mf_existing_rows,
            deleted_rows_count: mf_deleted_rows,
            partitions,
            key_metadata: mf_key_metadata,
            first_row_id: mf_first_row_id,
        };

        entries.push(RootManifestEntry::ManifestRef { manifest_file, mdv });
    }

    Ok(entries)
}

/// Parse inline entries from a RecordBatch (row group 1).
///
/// Uses the same column layout as parquet_manifest entries but wraps them
/// as RootManifestEntry::Inline.
fn record_batch_to_inline_entries(
    batch: &RecordBatch,
    metadata: &RootManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<RootManifestEntry>> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let status_arr = col_i32_opt(batch, "status");
    let snapshot_id_arr = col_i64_opt(batch, "snapshot_id");
    let seq_num_arr = col_i64_opt(batch, "sequence_number");
    let file_seq_arr = col_i64_opt(batch, "file_sequence_number");
    let content_arr = col_i32_opt(batch, "content");
    let file_path_arr = col_str_opt(batch, "file_path");
    let file_format_arr = col_str_opt(batch, "file_format");
    let partition_json_arr = col_str_opt(batch, "partition_json");
    let record_count_arr = col_i64_opt(batch, "record_count");
    let file_size_arr = col_i64_opt(batch, "file_size_in_bytes");
    let column_sizes_arr = col_binary_opt(batch, "column_sizes_json");
    let value_counts_arr = col_binary_opt(batch, "value_counts_json");
    let null_value_counts_arr = col_binary_opt(batch, "null_value_counts_json");
    let nan_value_counts_arr = col_binary_opt(batch, "nan_value_counts_json");
    let lower_bounds_arr = col_binary_opt(batch, "lower_bounds_json");
    let upper_bounds_arr = col_binary_opt(batch, "upper_bounds_json");
    let key_metadata_arr = col_binary_opt(batch, "key_metadata");
    let split_offsets_arr = col_binary_opt(batch, "split_offsets_json");
    let equality_ids_arr = col_binary_opt(batch, "equality_ids_json");
    let sort_order_id_arr = col_i32_opt(batch, "sort_order_id");
    let part_spec_id_arr = col_i32_opt(batch, "partition_spec_id");

    for i in 0..n {
        let status_val: ManifestStatus = status_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i))
                }
            })
            .unwrap_or(0)
            .try_into()?;
        let snap_id = nullable_i64(snapshot_id_arr, i);
        let seq_number = nullable_i64(seq_num_arr, i);
        let file_seq_number = nullable_i64(file_seq_arr, i);

        let content_type: DataContentType = content_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i))
                }
            })
            .unwrap_or(0)
            .try_into()?;

        let fpath = file_path_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i).to_string())
                }
            })
            .unwrap_or_default();

        let fformat_raw = file_format_arr
            .and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i))
                }
            })
            .unwrap_or("PARQUET");
        let fformat: DataFileFormat = fformat_raw.parse().map_err(|e| {
            let path = file_path_arr
                .and_then(|a| if Array::is_null(a, i) { None } else { Some(a.value(i)) })
                .unwrap_or("<none>");
            Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "root inline entry {i}: bad file_format {fformat_raw:?} (path={path}): {e}"
                ),
            )
        })?;

        let partition = parse_partition_json(
            partition_json_arr.and_then(|a| {
                if Array::is_null(a, i) {
                    None
                } else {
                    Some(a.value(i))
                }
            }),
            partition_type,
        );

        let rec_count = nullable_i64(record_count_arr, i).unwrap_or(0) as u64;
        let f_size = nullable_i64(file_size_arr, i).unwrap_or(0) as u64;

        let column_sizes = parse_i64_map_json(read_binary_opt(column_sizes_arr, i));
        let value_counts = parse_i64_map_json(read_binary_opt(value_counts_arr, i));
        let null_value_counts =
            parse_i64_map_json(read_binary_opt(null_value_counts_arr, i));
        let nan_value_counts =
            parse_i64_map_json(read_binary_opt(nan_value_counts_arr, i));
        let lower_bounds =
            parse_bounds_map_json(read_binary_opt(lower_bounds_arr, i), &metadata.schema);
        let upper_bounds =
            parse_bounds_map_json(read_binary_opt(upper_bounds_arr, i), &metadata.schema);

        let key_meta = read_binary_opt(key_metadata_arr, i).map(|b| b.to_vec());
        let split_offs: Option<Vec<i64>> = read_binary_opt(split_offsets_arr, i)
            .and_then(|b| serde_json::from_slice(b).ok());
        let eq_ids: Option<Vec<i32>> = read_binary_opt(equality_ids_arr, i)
            .and_then(|b| serde_json::from_slice(b).ok());
        let sort_id = sort_order_id_arr.and_then(|a| {
            if Array::is_null(a, i) {
                None
            } else {
                Some(a.value(i))
            }
        });
        let spec_id = part_spec_id_arr
            .map(|a| {
                if Array::is_null(a, i) {
                    metadata.partition_spec.spec_id()
                } else {
                    a.value(i)
                }
            })
            .unwrap_or(metadata.partition_spec.spec_id());

        let data_file = DataFile {
            content: content_type,
            file_path: fpath,
            file_format: fformat,
            partition,
            record_count: rec_count,
            file_size_in_bytes: f_size,
            column_sizes,
            value_counts,
            null_value_counts,
            nan_value_counts,
            lower_bounds,
            upper_bounds,
            key_metadata: key_meta,
            split_offsets: split_offs,
            equality_ids: eq_ids,
            sort_order_id: sort_id,
            partition_spec_id: spec_id,
            first_row_id: None,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        };

        entries.push(RootManifestEntry::Inline(ManifestEntry {
            status: status_val,
            snapshot_id: snap_id,
            sequence_number: seq_number,
            file_sequence_number: file_seq_number,
            data_file,
        }));
    }

    Ok(entries)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::spec::{
        DataContentType, DataFile, DataFileFormat, Datum, FormatVersion,
        ManifestContentType, ManifestFile, NestedField, PartitionSpec, PrimitiveType, Schema,
        Struct, Type, UnboundPartitionField,
    };

    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Long),
                    )),
                    Arc::new(NestedField::optional(
                        2,
                        "name",
                        Type::Primitive(PrimitiveType::String),
                    )),
                    Arc::new(NestedField::optional(
                        3,
                        "ts",
                        Type::Primitive(PrimitiveType::Timestamptz),
                    )),
                ])
                .build()
                .unwrap(),
        )
    }

    fn test_partition_spec(schema: &SchemaRef) -> PartitionSpec {
        PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id".to_string(),
                transform: crate::spec::Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap()
    }

    fn test_metadata(schema: &SchemaRef, partition_spec: &PartitionSpec) -> RootManifestMetadata {
        RootManifestMetadata {
            schema: schema.clone(),
            schema_id: 0,
            partition_spec: Arc::new(partition_spec.clone()),
            format_version: FormatVersion::V2,
            snapshot_id: 100,
            sequence_number: 5,
            parent_snapshot_id: Some(99),
            bucket_index_path: None,
            prev_root_path: None,
            chain_depth: 0,
            node_level: 0,
            removed_paths: Vec::new(),
        }
    }

    fn test_manifest_file(path: &str) -> ManifestFile {
        ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 5,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(20),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(5000),
            deleted_rows_count: Some(0),
            partitions: None,
            key_metadata: None,
            first_row_id: None,
        }
    }

    fn test_inline_entry(path: &str, record_count: u64) -> ManifestEntry {
        ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: Some(100),
            sequence_number: Some(5),
            file_sequence_number: Some(5),
            data_file: DataFile {
                content: DataContentType::Data,
                file_path: path.to_string(),
                file_format: DataFileFormat::Parquet,
                partition: Struct::empty(),
                record_count,
                file_size_in_bytes: 50000,
                column_sizes: HashMap::from([(1, 2000), (2, 3000)]),
                value_counts: HashMap::from([(1, record_count), (2, record_count)]),
                null_value_counts: HashMap::from([(2, 50)]),
                nan_value_counts: HashMap::new(),
                lower_bounds: HashMap::from([(1, Datum::long(0))]),
                upper_bounds: HashMap::from([(1, Datum::long(999))]),
                key_metadata: None,
                split_offsets: Some(vec![4, 1000]),
                equality_ids: None,
                sort_order_id: Some(0),
                partition_spec_id: 0,
                first_row_id: None,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
            },
        }
    }

    #[test]
    fn cluster_entries_by_partition_groups_same_partition_contiguously() {
        use crate::spec::{Literal, Struct};
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let mk = |path: &str, p: i64| {
            let mut e = test_inline_entry(path, 1);
            e.data_file.partition = Struct::from_iter([Some(Literal::long(p))]);
            RootManifestEntry::Inline(e)
        };
        // Interleaved partitions A(1) B(2) A(1) C(3) B(2) A(1) + one ManifestRef.
        let entries = vec![
            mk("f0", 1),
            mk("f1", 2),
            mk("f2", 1),
            mk("f3", 3),
            mk("f4", 2),
            mk("f5", 1),
        ];
        let out = cluster_entries_by_partition(entries, &spec);
        assert_eq!(out.len(), 6, "no entries lost or added");
        let parts: Vec<Struct> = out
            .iter()
            .map(|e| match e {
                RootManifestEntry::Inline(me) => me.data_file.partition.clone(),
                RootManifestEntry::ManifestRef { .. } => unreachable!("all inline here"),
            })
            .collect();
        // Every distinct partition must form exactly ONE contiguous run.
        let mut runs: Vec<Struct> = Vec::new();
        for p in &parts {
            if runs.last() != Some(p) {
                assert!(
                    !runs.contains(p),
                    "partition appears in a second, non-contiguous run"
                );
                runs.push(p.clone());
            }
        }
        assert_eq!(runs.len(), 3, "three distinct partitions, three contiguous runs");
    }

    #[test]
    fn cluster_entries_by_partition_sorts_manifest_refs_by_time_bucket() {
        use crate::spec::{ByteBuf, FieldSummary};

        // Partition spec: hour(ts). Field index 0 is the Hour transform, so the
        // clustering sorts refs by that summary's lower bound.
        let schema = test_schema();
        let spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_unbound_field(UnboundPartitionField {
                source_id: 3,
                field_id: None,
                name: "ts_hour".to_string(),
                transform: crate::spec::Transform::Hour,
            })
            .unwrap()
            .build()
            .unwrap();

        // A manifest ref whose partition summary reports `hour` at field 0.
        let mk_ref = |path: &str, hour: i32| {
            let mut mf = test_manifest_file(path);
            mf.partitions = Some(vec![FieldSummary {
                contains_null: false,
                contains_nan: Some(false),
                lower_bound: Some(ByteBuf::from(hour.to_le_bytes().to_vec())),
                upper_bound: Some(ByteBuf::from(hour.to_le_bytes().to_vec())),
            }]);
            RootManifestEntry::ManifestRef {
                manifest_file: mf,
                mdv: None,
            }
        };

        // Interleaved hours 495533, 495484, 495533, 495486, 495484 (the exact
        // 4-hour-mixed spread seen live). After clustering they must come out
        // sorted, so each leaf chunk covers one contiguous hour run.
        let entries = vec![
            mk_ref("m0", 495533),
            mk_ref("m1", 495484),
            mk_ref("m2", 495533),
            mk_ref("m3", 495486),
            mk_ref("m4", 495484),
        ];
        let out = cluster_entries_by_partition(entries, &spec);
        assert_eq!(out.len(), 5, "no refs lost or added");
        let hours: Vec<i64> = out
            .iter()
            .map(|e| match e {
                RootManifestEntry::ManifestRef { manifest_file, .. } => {
                    ref_time_bucket(manifest_file, 0)
                }
                RootManifestEntry::Inline(_) => unreachable!("all refs here"),
            })
            .collect();
        assert_eq!(
            hours,
            vec![495484, 495484, 495486, 495533, 495533],
            "refs sorted by hour bucket => same-hour refs contiguous"
        );
    }

    /// A leaf REF must be suppressible by tombstone, exactly like an inline
    /// entry. Before the fix `finalize_reconstruct` matched tombstones against
    /// inline data-file paths only and returned every `ManifestRef` verbatim,
    /// so graduation had NO way to retire a ref: its delta writes no entries,
    /// the base root beneath kept listing the ref, and each later chain walk
    /// re-offered it to be graduated (and duplicated into the bucket-index)
    /// again. sri-olly 2026-08-16: 230,456 index rows for 18,053 distinct
    /// leaves, one repeated 74x, +3,938/tick and unbounded.
    #[test]
    fn finalize_reconstruct_tombstones_leaf_refs_not_only_inline() {
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let graduated = "s3://b/leaf-graduated-m0.parquet";
        let still_hot = "s3://b/leaf-hot-m0.parquet";

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file(graduated),
                mdv: None,
            },
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file(still_hot),
                mdv: None,
            },
            RootManifestEntry::Inline(test_inline_entry("s3://b/d-live.parquet", 1)),
        ];
        // Graduation stamped the moved ref's MANIFEST path as a tombstone.
        let removed: HashSet<String> = [graduated.to_string()].into_iter().collect();

        let (meta_out, kept) = finalize_reconstruct(test_metadata(&schema, &spec), entries, removed);

        let kept_refs: Vec<&str> = kept
            .iter()
            .filter_map(|e| match e {
                RootManifestEntry::ManifestRef { manifest_file, .. } => {
                    Some(manifest_file.manifest_path.as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            kept_refs,
            vec![still_hot],
            "the graduated ref must be suppressed; the hot ref must survive"
        );
        assert_eq!(
            kept.iter()
                .filter(|e| matches!(e, RootManifestEntry::Inline(_)))
                .count(),
            1,
            "an untombstoned inline entry is untouched"
        );
        // Matched tombstones are materialized, so they must not also be carried
        // forward as scan tombstones (that is what grew removed_paths unbounded).
        assert!(
            !meta_out.removed_paths.contains(&graduated.to_string()),
            "a ref tombstone that matched is retired, not carried forward"
        );
    }

    /// The tombstone must not be dropped when it matches nothing in THIS walk —
    /// it has to keep suppressing the ref on later reconstructs, since the base
    /// root still lists it until a flat collapse rewrites the chain.
    #[test]
    fn finalize_reconstruct_carries_unmatched_ref_tombstone_forward() {
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let absent = "s3://b/leaf-not-in-this-walk-m0.parquet";

        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: test_manifest_file("s3://b/leaf-other-m0.parquet"),
            mdv: None,
        }];
        let removed: HashSet<String> = [absent.to_string()].into_iter().collect();

        let (meta_out, kept) = finalize_reconstruct(test_metadata(&schema, &spec), entries, removed);

        assert_eq!(kept.len(), 1, "unrelated ref survives");
        assert!(
            meta_out.removed_paths.contains(&absent.to_string()),
            "an unmatched tombstone stays live for later walks"
        );
    }

    #[test]
    fn chain_metadata_round_trips() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        // Base root: absent prev/chain decode to None / 0.
        let base = test_metadata(&schema, &partition_spec);
        let bytes = write_root_manifest(&[], &base, &partition_type).unwrap();
        let (read, _) = read_root_manifest(Bytes::from(bytes)).unwrap();
        assert_eq!(read.prev_root_path, None);
        assert_eq!(read.chain_depth, 0);

        // Delta root: prev pointer + chain depth survive the round trip.
        let mut delta = test_metadata(&schema, &partition_spec);
        delta.prev_root_path = Some("s3://bucket/metadata/root-prev.parquet".to_string());
        delta.chain_depth = 7;
        let bytes = write_root_manifest(&[], &delta, &partition_type).unwrap();
        let (read, _) = read_root_manifest(Bytes::from(bytes)).unwrap();
        assert_eq!(
            read.prev_root_path.as_deref(),
            Some("s3://bucket/metadata/root-prev.parquet")
        );
        assert_eq!(read.chain_depth, 7);
    }

    #[tokio::test]
    async fn balanced_tree_round_trips_multilevel() {
        use std::collections::HashSet;

        use crate::io::FileIOBuilder;

        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let partition_type = spec.partition_type(&schema).unwrap();
        let template = test_metadata(&schema, &spec);
        let file_io = FileIOBuilder::new("memory").build().unwrap();
        let location = "memory:///tbl";

        // 50 entries at fan-out 4 → 13 leaf nodes → 4 interior → 1 root = 3 levels,
        // so the collector must recurse two interior levels to reach the leaves.
        let n = 50usize;
        let entries: Vec<RootManifestEntry> = (0..n)
            .map(|i| {
                RootManifestEntry::Inline(test_inline_entry(
                    &format!("memory:///tbl/data/f{i}.parquet"),
                    1,
                ))
            })
            .collect();
        let want: HashSet<String> = (0..n)
            .map(|i| format!("memory:///tbl/data/f{i}.parquet"))
            .collect();

        let root_path = build_balanced_tree(
            &file_io,
            location,
            &template,
            &partition_type,
            uuid::Uuid::nil(),
            entries,
            4,
        )
        .await
        .unwrap();

        // The root is a multi-level interior node.
        let bytes = file_io.new_input(&root_path).unwrap().read().await.unwrap();
        let (root_meta, _) = read_root_manifest(bytes).unwrap();
        assert!(
            root_meta.node_level >= 2,
            "fan-out 4 over 50 entries → ≥3 levels, got node_level {}",
            root_meta.node_level
        );

        // reconstruct_root traverses the whole tree and returns exactly the leaves.
        let (_, got) = reconstruct_root(&file_io, &root_path).await.unwrap();
        let got_paths: HashSet<String> = got
            .iter()
            .map(|e| match e {
                RootManifestEntry::Inline(me) => me.data_file.file_path.clone(),
                RootManifestEntry::ManifestRef { manifest_file, .. } => {
                    manifest_file.manifest_path.clone()
                }
            })
            .collect();
        assert_eq!(
            got_paths, want,
            "tree reconstruct must return every leaf exactly once"
        );
    }

    #[tokio::test]
    async fn reconstruct_root_filtered_prunes_leaves_through_tree() {
        use std::collections::HashSet;

        use crate::io::FileIOBuilder;

        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let partition_type = spec.partition_type(&schema).unwrap();
        let template = test_metadata(&schema, &spec);
        let file_io = FileIOBuilder::new("memory").build().unwrap();
        let location = "memory:///tbl";

        // 50 entries → multi-level tree; record_count = i is the discriminator so
        // the filter must prune at the LEAF level while recursing interior nodes.
        let n = 50u64;
        let entries: Vec<RootManifestEntry> = (0..n)
            .map(|i| {
                RootManifestEntry::Inline(test_inline_entry(
                    &format!("memory:///tbl/data/f{i}.parquet"),
                    i,
                ))
            })
            .collect();
        let root_path = build_balanced_tree(
            &file_io,
            location,
            &template,
            &partition_type,
            uuid::Uuid::nil(),
            entries,
            4,
        )
        .await
        .unwrap();

        // Keep only even record_count.
        let keep = |df: &DataFile| df.record_count % 2 == 0;
        let (_, got) = reconstruct_root_filtered(&file_io, &root_path, Some(&keep))
            .await
            .unwrap();
        let got_counts: HashSet<u64> = got
            .iter()
            .map(|e| match e {
                RootManifestEntry::Inline(me) => me.data_file.record_count,
                RootManifestEntry::ManifestRef { .. } => unreachable!("leaves are inline"),
            })
            .collect();
        let want: HashSet<u64> = (0..n).filter(|i| i % 2 == 0).collect();
        assert_eq!(
            got_counts, want,
            "filtered reconstruct returns ONLY kept leaves, pruned through the tree"
        );
        assert_eq!(got.len(), 25, "exactly half of 50 kept");

        // `None` == unfiltered == every leaf (parity with reconstruct_root).
        let (_, all) = reconstruct_root_filtered(&file_io, &root_path, None)
            .await
            .unwrap();
        assert_eq!(all.len(), 50, "None filter keeps everything (unpruned)");
    }

    // Regression for the tessellate readiness-check wedge (2026-08-07 sri-olly):
    // callers that mixed `read_root_manifest` (head-only) with a walk built by
    // `reconstruct_root` (full chain) were checking each data path against ONLY
    // the head delta's `removed_paths` — ancestor-delta tombstones leaked through
    // and got counted as live files. The invariant our fix relies on is that
    // `reconstruct_root`'s returned meta.removed_paths carries every tombstone
    // from the chain that didn't materialize as an inline entry (i.e. still needs
    // to filter ManifestRef children). This test locks that in end-to-end:
    // three chained deltas each remove a distinct manifest-ref-owned path, and
    // walking the head must surface all three (sorted) in the returned meta.
    #[tokio::test]
    async fn reconstruct_root_accumulates_removed_paths_across_delta_chain() {
        use std::collections::HashSet;

        use crate::io::FileIOBuilder;

        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let file_io = FileIOBuilder::new("memory").build().unwrap();

        // Three chained root manifests, each tombstoning one path. No inline
        // entries → nothing materializes → all three tombstones stay in
        // meta.removed_paths for the manifest-ref filter to apply downstream.
        let paths = ["memory:///root-base.parquet", "memory:///root-d1.parquet", "memory:///root-d2.parquet"];
        let tombstones = ["memory:///data/base_tomb.parquet", "memory:///data/d1_tomb.parquet", "memory:///data/d2_tomb.parquet"];

        for (i, path) in paths.iter().enumerate() {
            let mut meta = test_metadata(&schema, &partition_spec);
            meta.prev_root_path = if i == 0 { None } else { Some(paths[i - 1].to_string()) };
            meta.chain_depth = i as u32;
            meta.removed_paths = vec![tombstones[i].to_string()];
            let bytes = write_root_manifest(&[], &meta, &partition_type).unwrap();
            file_io.new_output(*path).unwrap().write(bytes.into()).await.unwrap();
        }

        let (merged_meta, entries) = reconstruct_root(&file_io, paths[2]).await.unwrap();
        assert!(entries.is_empty(), "no entries were written across the chain");

        let got: HashSet<String> = merged_meta.removed_paths.into_iter().collect();
        let want: HashSet<String> = tombstones.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            got, want,
            "reconstruct_root must union removed_paths across every ancestor delta so downstream ManifestRef filtering catches tombstones added by prior commits"
        );

        // Contrast: head-only `read_root_manifest` sees ONLY delta2's set.
        let head_bytes = file_io.new_input(paths[2]).unwrap().read().await.unwrap();
        let (head_meta, _) = read_root_manifest(head_bytes).unwrap();
        assert_eq!(
            head_meta.removed_paths.len(),
            1,
            "head-only read must expose the bug this test guards against: it misses ancestor tombstones"
        );
        assert_eq!(head_meta.removed_paths[0], tombstones[2]);
    }

    // Regression for the attribute_index wedge: a collapsed balanced-tree base has
    // `node_level > 0` AND `prev_root_path == None`. The raw single read
    // (`read_root_manifest`) returns the base's DIRECT entries — refs to INTERIOR
    // nodes, not data manifests. `load_manifest_list` must therefore route such a
    // base through `reconstruct_root` (which recurses to the leaves), not the raw
    // read; otherwise the interior node leaks out as a data ManifestFile and the
    // data-manifest reader misreads it (empty file_format). This asserts the two
    // conditions the fix keys on and that reconstruct yields only real data leaves.
    #[tokio::test]
    async fn tree_base_has_no_prev_and_positive_node_level_reconstruct_flattens() {
        use crate::io::FileIOBuilder;
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let partition_type = spec.partition_type(&schema).unwrap();
        let template = test_metadata(&schema, &spec);
        let file_io = FileIOBuilder::new("memory").build().unwrap();
        let location = "memory:///tbl";

        let n = 50usize; // fan-out 4 over 50 → multi-level interior tree
        let entries: Vec<RootManifestEntry> = (0..n)
            .map(|i| RootManifestEntry::Inline(test_inline_entry(&format!("memory:///tbl/data/f{i}.parquet"), 1)))
            .collect();
        let root_path = build_balanced_tree(&file_io, location, &template, &partition_type, uuid::Uuid::nil(), entries, 4)
            .await
            .unwrap();

        // The collapsed tree base: node_level>0 AND no prev pointer — the exact combo
        // that the buggy `else` branch mis-handled.
        let bytes = file_io.new_input(&root_path).unwrap().read().await.unwrap();
        let (base_meta, base_direct) = read_root_manifest(bytes).unwrap();
        assert!(base_meta.node_level > 0, "tree base must have node_level>0");
        assert!(base_meta.prev_root_path.is_none(), "a collapsed base has no prev pointer");

        // Raw read surfaces INTERIOR-node refs (the leak): every direct entry is a ref
        // whose target is itself a node (node_level>0), never a data manifest.
        let mut saw_interior_ref = false;
        for e in &base_direct {
            if let RootManifestEntry::ManifestRef { manifest_file, .. } = e {
                let cb = file_io.new_input(&manifest_file.manifest_path).unwrap().read().await.unwrap();
                let (child_meta, _) = read_root_manifest(cb).unwrap();
                if child_meta.node_level > 0 {
                    saw_interior_ref = true;
                }
            }
        }
        assert!(saw_interior_ref, "raw read of a multi-level tree base must expose interior-node refs");

        // The fix: reconstruct_root flattens to exactly the data leaves — no interior refs.
        let (_, flat) = reconstruct_root(&file_io, &root_path).await.unwrap();
        assert_eq!(flat.len(), n, "reconstruct must return every data leaf");
        for e in &flat {
            if let RootManifestEntry::ManifestRef { manifest_file, .. } = e {
                let cb = file_io.new_input(&manifest_file.manifest_path).unwrap().read().await.unwrap();
                let (child_meta, _) = read_root_manifest(cb).unwrap();
                assert_eq!(child_meta.node_level, 0, "reconstruct must not surface interior nodes");
            }
        }
    }

    #[test]
    fn round_trip_mixed_entries() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
                mdv: None,
            },
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/file1.parquet", 1000)),
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m1.avro"),
                mdv: None,
            },
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/file2.parquet", 500)),
        ];

        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();
        assert!(!bytes.is_empty());

        let (read_meta, read_entries) = read_root_manifest(Bytes::from(bytes)).unwrap();

        // Verify metadata
        assert_eq!(read_meta.snapshot_id, 100);
        assert_eq!(read_meta.sequence_number, 5);
        assert_eq!(read_meta.parent_snapshot_id, Some(99));
        assert_eq!(read_meta.schema_id, 0);

        // With two-section layout, refs come first (row group 0), then inlines (row group 1)
        assert_eq!(read_entries.len(), 4);

        // First two entries: ManifestRefs (from row group 0)
        match &read_entries[0] {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                assert_eq!(manifest_file.manifest_path, "s3://bucket/metadata/m0.avro");
                assert_eq!(manifest_file.manifest_length, 4096);
                assert_eq!(manifest_file.added_files_count, Some(10));
                assert!(mdv.is_none());
            }
            _ => panic!("Expected ManifestRef"),
        }

        match &read_entries[1] {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                assert_eq!(manifest_file.manifest_path, "s3://bucket/metadata/m1.avro");
                assert!(mdv.is_none());
            }
            _ => panic!("Expected ManifestRef"),
        }

        // Last two entries: Inlines (from row group 1)
        match &read_entries[2] {
            RootManifestEntry::Inline(me) => {
                assert_eq!(me.data_file.file_path, "s3://bucket/data/file1.parquet");
                assert_eq!(me.data_file.record_count, 1000);
                assert_eq!(me.status, ManifestStatus::Added);
                assert_eq!(me.snapshot_id, Some(100));
                assert_eq!(me.data_file.column_sizes.get(&1), Some(&2000));
                assert_eq!(me.data_file.lower_bounds.get(&1), Some(&Datum::long(0)));
                assert_eq!(me.data_file.upper_bounds.get(&1), Some(&Datum::long(999)));
            }
            _ => panic!("Expected Inline"),
        }

        match &read_entries[3] {
            RootManifestEntry::Inline(me) => {
                assert_eq!(me.data_file.file_path, "s3://bucket/data/file2.parquet");
                assert_eq!(me.data_file.record_count, 500);
            }
            _ => panic!("Expected Inline"),
        }
    }

    #[test]
    fn inline_only() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/a.parquet", 100)),
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/b.parquet", 200)),
        ];

        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_root_manifest(Bytes::from(bytes)).unwrap();

        assert_eq!(read_entries.len(), 2);
        for e in &read_entries {
            assert!(matches!(e, RootManifestEntry::Inline(_)));
        }
    }

    #[test]
    fn refs_only() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
                mdv: None,
            },
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m1.avro"),
                mdv: None,
            },
        ];

        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_root_manifest(Bytes::from(bytes)).unwrap();

        assert_eq!(read_entries.len(), 2);
        for e in &read_entries {
            assert!(matches!(e, RootManifestEntry::ManifestRef { .. }));
        }
    }

    #[test]
    fn empty_root_manifest() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        let entries: Vec<RootManifestEntry> = vec![];
        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();
        let (read_meta, read_entries) = read_root_manifest(Bytes::from(bytes)).unwrap();

        assert_eq!(read_entries.len(), 0);
        assert_eq!(read_meta.snapshot_id, 100);
    }

    #[test]
    fn mdv_bitmap_round_trip() {
        use roaring::RoaringBitmap;

        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        // Create a roaring bitmap marking rows 0, 5, 10 as deleted
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(0);
        bitmap.insert(5);
        bitmap.insert(10);
        let mut bitmap_bytes = Vec::new();
        bitmap.serialize_into(&mut bitmap_bytes).unwrap();

        let entries = vec![RootManifestEntry::ManifestRef {
            manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
            mdv: Some(bitmap_bytes.clone()),
        }];

        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_root_manifest(Bytes::from(bytes)).unwrap();

        assert_eq!(read_entries.len(), 1);
        match &read_entries[0] {
            RootManifestEntry::ManifestRef { mdv, .. } => {
                let mdv_bytes = mdv.as_ref().expect("MDV should be present");
                let read_bitmap =
                    RoaringBitmap::deserialize_from(mdv_bytes.as_slice()).unwrap();
                assert!(read_bitmap.contains(0));
                assert!(read_bitmap.contains(5));
                assert!(read_bitmap.contains(10));
                assert!(!read_bitmap.contains(1));
                assert_eq!(read_bitmap.len(), 3);
            }
            _ => panic!("Expected ManifestRef"),
        }
    }

    #[test]
    fn root_manifest_accessor_methods() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
                mdv: None,
            },
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/a.parquet", 100)),
        ];

        let rm = RootManifest::new(metadata.clone(), entries);

        assert_eq!(rm.entries().len(), 2);
        assert_eq!(rm.metadata().snapshot_id, 100);
        assert_eq!(rm.inline_count(), 1);
        assert_eq!(rm.manifest_refs().count(), 1);
        assert_eq!(rm.inline_entries().count(), 1);

        let entries = rm.into_entries();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn round_trip_preserves_mdv() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        mdv.mark_deleted(3);
        let mdv_bytes = mdv.serialize().unwrap();

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
                mdv: Some(mdv_bytes),
            },
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m1.avro"),
                mdv: None,
            },
        ];

        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_root_manifest(Bytes::from(bytes)).unwrap();

        assert_eq!(read_entries.len(), 2);

        match &read_entries[0] {
            RootManifestEntry::ManifestRef {
                mdv: Some(bytes), ..
            } => {
                let read_mdv = ManifestDeleteVector::deserialize(bytes).unwrap();
                assert!(read_mdv.is_deleted(0));
                assert!(!read_mdv.is_deleted(1));
                assert!(!read_mdv.is_deleted(2));
                assert!(read_mdv.is_deleted(3));
                assert_eq!(read_mdv.deleted_count(), 2);
            }
            _ => panic!("Expected ManifestRef with MDV"),
        }

        match &read_entries[1] {
            RootManifestEntry::ManifestRef { mdv: None, .. } => {}
            _ => panic!("Expected ManifestRef without MDV"),
        }
    }

    #[test]
    fn mdv_guard_round_trip_preserves_guard_and_bitmap() {
        // A guarded MDV serializes into the versioned envelope and deserializes
        // back with the guard intact, so a later scan can validate positional
        // alignment. Fix #5.
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(1);
        mdv.mark_deleted(4);
        let checksum = ManifestDeleteVector::compute_checksum(
            ["s3://b/a.parquet", "s3://b/b.parquet", "s3://b/c.parquet"],
        );
        mdv.set_guard(3, checksum);

        let bytes = mdv.serialize().unwrap();
        // Envelope, not raw roaring: begins with the magic prefix.
        assert_eq!(&bytes[..4], MDV_MAGIC);

        let read = ManifestDeleteVector::deserialize(&bytes).unwrap();
        assert!(read.is_deleted(1));
        assert!(read.is_deleted(4));
        assert!(!read.is_deleted(0));
        assert_eq!(read.deleted_count(), 2);
        // Guard validates against the same manifest snapshot.
        read.validate_against(3, checksum).unwrap();
    }

    #[test]
    fn mdv_legacy_guardless_is_backward_compatible() {
        // A guardless MDV serializes to raw roaring bytes (no magic prefix), so
        // old readers keep working, and validate_against is a no-op (Ok).
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        mdv.mark_deleted(2);
        let bytes = mdv.serialize().unwrap();
        assert_ne!(&bytes[..MDV_MAGIC.len().min(bytes.len())], MDV_MAGIC);

        let read = ManifestDeleteVector::deserialize(&bytes).unwrap();
        assert!(read.is_deleted(0));
        assert!(read.is_deleted(2));
        // No guard => validates against ANY manifest snapshot.
        read.validate_against(999, 0xdead_beef).unwrap();
    }

    #[test]
    fn mdv_guard_detects_stale_manifest() {
        // A guarded MDV validated against a mismatched entry count OR checksum is
        // stale (the child manifest was rewritten/reordered under it) and errors,
        // instead of silently soft-deleting the wrong rows.
        let mut mdv = ManifestDeleteVector::new();
        mdv.mark_deleted(0);
        let checksum = ManifestDeleteVector::compute_checksum(["s3://b/a.parquet"]);
        mdv.set_guard(1, checksum);

        // Mismatched entry count.
        assert!(mdv.validate_against(2, checksum).is_err());
        // Mismatched checksum.
        assert!(mdv.validate_against(1, checksum ^ 1).is_err());
        // Exact match is Ok.
        mdv.validate_against(1, checksum).unwrap();
    }

    #[test]
    fn mdv_checksum_is_order_sensitive_and_delimited() {
        // Length-delimiting each path makes the checksum sensitive to path
        // boundaries as well as order: concatenation collisions cannot occur.
        let a = ManifestDeleteVector::compute_checksum(["ab", "c"]);
        let b = ManifestDeleteVector::compute_checksum(["a", "bc"]);
        assert_ne!(a, b, "boundary shift must change the checksum");

        let fwd = ManifestDeleteVector::compute_checksum(["x", "y"]);
        let rev = ManifestDeleteVector::compute_checksum(["y", "x"]);
        assert_ne!(fwd, rev, "reordering must change the checksum");

        // Deterministic across calls (no Hash randomization).
        assert_eq!(
            ManifestDeleteVector::compute_checksum(["p", "q", "r"]),
            ManifestDeleteVector::compute_checksum(["p", "q", "r"]),
        );
    }

    #[test]
    fn mdv_envelope_too_short_errors() {
        // A buffer that starts with the magic but is truncated below the 16-byte
        // header is rejected rather than mis-parsed.
        let mut bytes = MDV_MAGIC.to_vec();
        bytes.extend_from_slice(&[0u8; 4]); // 8 bytes total, < 16
        assert!(ManifestDeleteVector::deserialize(&bytes).is_err());
    }

    #[test]
    fn mdv_comprehensive() {
        let mut mdv = ManifestDeleteVector::new();
        assert!(mdv.is_empty());
        assert_eq!(mdv.deleted_count(), 0);
        assert_eq!(mdv.deleted_fraction(100), 0.0);

        // Mark some rows
        mdv.mark_deleted(0);
        mdv.mark_deleted(5);
        mdv.mark_deleted(10);
        assert!(!mdv.is_empty());
        assert_eq!(mdv.deleted_count(), 3);
        assert!(mdv.is_deleted(0));
        assert!(mdv.is_deleted(5));
        assert!(mdv.is_deleted(10));
        assert!(!mdv.is_deleted(1));
        assert!(!mdv.is_deleted(99));
        assert!((mdv.deleted_fraction(10) - 0.3).abs() < f64::EPSILON);

        // Serialize round-trip
        let bytes = mdv.serialize().unwrap();
        let mdv2 = ManifestDeleteVector::deserialize(&bytes).unwrap();
        assert_eq!(mdv2.deleted_count(), 3);
        assert!(mdv2.is_deleted(0));
        assert!(mdv2.is_deleted(5));
        assert!(mdv2.is_deleted(10));

        // Merge
        let mut mdv3 = ManifestDeleteVector::new();
        mdv3.mark_deleted(20);
        mdv3.mark_deleted(30);
        mdv3.merge(&mdv2);
        assert_eq!(mdv3.deleted_count(), 5);
        assert!(mdv3.is_deleted(0));
        assert!(mdv3.is_deleted(20));
        assert!(mdv3.is_deleted(30));

        // Idempotent insert
        mdv3.mark_deleted(0);
        assert_eq!(mdv3.deleted_count(), 5);
    }

    #[test]
    fn remove_inline_files() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/keep.parquet", 100)),
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/remove.parquet", 200)),
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/manifest-1.avro"),
                mdv: None,
            },
        ];

        let mut rm = RootManifest::new(metadata, entries);
        assert_eq!(rm.inline_count(), 2);
        assert_eq!(rm.entries().len(), 3);

        let mut to_remove = std::collections::HashSet::new();
        to_remove.insert("s3://bucket/remove.parquet".to_string());
        rm.remove_inline_files(&to_remove);

        assert_eq!(rm.inline_count(), 1);
        assert_eq!(rm.entries().len(), 2); // 1 inline + 1 ref
        // Verify the kept inline entry
        let kept = rm.inline_entries().next().unwrap();
        assert_eq!(kept.data_file.file_path, "s3://bucket/keep.parquet");
    }

    #[test]
    fn two_section_layout_metadata() {
        // Verify that the layout metadata is present in the written file
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
                mdv: None,
            },
        ];

        let bytes = write_root_manifest(&entries, &metadata, &partition_type).unwrap();

        // Read the parquet metadata directly to check for layout marker
        let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(&bytes)).unwrap();
        let arrow_meta = reader.schema().metadata().clone();
        assert_eq!(arrow_meta.get("root-manifest-layout").map(|s| s.as_str()), Some("two-section"));
        assert_eq!(arrow_meta.get("root-manifest").map(|s| s.as_str()), Some("true"));
        assert_eq!(arrow_meta.get("refs-count").map(|s| s.as_str()), Some("1"));
        assert_eq!(arrow_meta.get("inlines-count").map(|s| s.as_str()), Some("0"));

        // Only 1 row group (refs only, empty inlines don't get a row group)
        assert_eq!(reader.metadata().num_row_groups(), 1);
    }
}
