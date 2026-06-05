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
    append_map_json, append_opt_json, col_binary_opt, col_i32, col_i32_opt, col_i64_opt,
    col_str_opt, nullable_i64, parse_bounds_map_json, parse_i64_map_json,
    parse_partition_json, read_binary_opt, serialize_bounds_map, serialize_i64_map,
    serialize_partition_json,
};
use super::{ManifestEntry, ManifestStatus};
use crate::error::Result;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, FormatVersion, ManifestContentType, ManifestFile,
    ManifestList, PartitionSpec, SchemaRef, StructType,
};
use crate::{Error, ErrorKind};

// ============================================================================
// Types
// ============================================================================

/// Discriminator for root manifest entry types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RootEntryType {
    /// Reference to a child manifest file.
    ManifestRef = 0,
    /// Inline data file entry.
    InlineData = 1,
    /// Inline delete file entry.
    InlineDelete = 2,
}

impl TryFrom<i32> for RootEntryType {
    type Error = Error;

    fn try_from(v: i32) -> Result<Self> {
        match v {
            0 => Ok(RootEntryType::ManifestRef),
            1 => Ok(RootEntryType::InlineData),
            2 => Ok(RootEntryType::InlineDelete),
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("invalid root entry type: {v}"),
            )),
        }
    }
}

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

/// A manifest delete vector: marks specific row indices in a child manifest as
/// logically deleted without rewriting the manifest file.
///
/// Used during compaction (replace_data_files) on V4 tables to soft-delete
/// entries in child manifests. The bitmap is serialized as roaring bitmap bytes
/// and stored in the `mdv_bitmap` column of root manifest reference entries.
#[derive(Debug, Clone)]
pub struct ManifestDeleteVector {
    bitmap: RoaringBitmap,
}

impl ManifestDeleteVector {
    /// Create an empty MDV.
    pub fn new() -> Self {
        Self {
            bitmap: RoaringBitmap::new(),
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
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.bitmap
            .serialize_into(&mut buf)
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("MDV serialize failed: {e}")))?;
        Ok(buf)
    }

    /// Deserialize from bytes read from root manifest.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        let bitmap = RoaringBitmap::deserialize_from(bytes)
            .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("MDV deserialize failed: {e}")))?;
        Ok(Self { bitmap })
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

    /// Compatibility shim: convert root manifest to a ManifestList.
    ///
    /// ManifestRef entries map directly to ManifestFile entries.
    /// Inline entries are grouped by content type into synthetic ManifestFile entries
    /// with empty manifest_path and correct counts. The scan code integration for
    /// loading inline entries will be handled in Phase 2.
    pub fn to_manifest_list(&self) -> ManifestList {
        let mut manifest_files: Vec<ManifestFile> = Vec::new();

        // Collect manifest refs directly
        for entry in &self.entries {
            if let RootManifestEntry::ManifestRef { manifest_file, .. } = entry {
                manifest_files.push(manifest_file.clone());
            }
        }

        // Group inline entries by content type
        let mut inline_data_count: u32 = 0;
        let mut inline_data_rows: u64 = 0;
        let mut inline_delete_count: u32 = 0;
        let mut inline_delete_rows: u64 = 0;
        let mut has_inline_data = false;
        let mut has_inline_delete = false;

        for entry in &self.entries {
            if let RootManifestEntry::Inline(me) = entry {
                match me.data_file.content {
                    DataContentType::Data => {
                        has_inline_data = true;
                        inline_data_count += 1;
                        inline_data_rows += me.data_file.record_count;
                    }
                    DataContentType::EqualityDeletes
                    | DataContentType::PositionDeletes => {
                        has_inline_delete = true;
                        inline_delete_count += 1;
                        inline_delete_rows += me.data_file.record_count;
                    }
                }
            }
        }

        if has_inline_data {
            manifest_files.push(ManifestFile {
                manifest_path: String::new(),
                manifest_length: 0,
                partition_spec_id: self.metadata.partition_spec.spec_id(),
                content: ManifestContentType::Data,
                sequence_number: self.metadata.sequence_number,
                min_sequence_number: self.metadata.sequence_number,
                added_snapshot_id: self.metadata.snapshot_id,
                added_files_count: Some(inline_data_count),
                existing_files_count: Some(0),
                deleted_files_count: Some(0),
                added_rows_count: Some(inline_data_rows),
                existing_rows_count: Some(0),
                deleted_rows_count: Some(0),
                partitions: None,
                key_metadata: None,
                first_row_id: None,
            });
        }

        if has_inline_delete {
            manifest_files.push(ManifestFile {
                manifest_path: String::new(),
                manifest_length: 0,
                partition_spec_id: self.metadata.partition_spec.spec_id(),
                content: ManifestContentType::Deletes,
                sequence_number: self.metadata.sequence_number,
                min_sequence_number: self.metadata.sequence_number,
                added_snapshot_id: self.metadata.snapshot_id,
                added_files_count: Some(inline_delete_count),
                existing_files_count: Some(0),
                deleted_files_count: Some(0),
                added_rows_count: Some(inline_delete_rows),
                existing_rows_count: Some(0),
                deleted_rows_count: Some(0),
                partitions: None,
                key_metadata: None,
                first_row_id: None,
            });
        }

        ManifestList::new(manifest_files)
    }
}

// ============================================================================
// Arrow Schema
// ============================================================================

/// Build the combined Arrow schema for root manifest entries.
///
/// Contains the entry_type discriminator, manifest ref columns, inline entry columns,
/// and mdv_bitmap column. All columns are nullable except entry_type.
pub fn root_manifest_arrow_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        // Discriminator
        Field::new("entry_type", DataType::Int32, false),
        // Manifest ref columns
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
        // Inline entry columns
        Field::new("status", DataType::Int32, true),
        Field::new("snapshot_id", DataType::Int64, true),
        Field::new("sequence_number", DataType::Int64, true),
        Field::new("file_sequence_number", DataType::Int64, true),
        Field::new("content", DataType::Int32, true),
        Field::new("file_path", DataType::Utf8, true),
        Field::new("file_format", DataType::Utf8, true),
        Field::new("partition_json", DataType::Utf8, true),
        Field::new("record_count", DataType::Int64, true),
        Field::new("file_size_in_bytes", DataType::Int64, true),
        Field::new("column_sizes_json", DataType::Binary, true),
        Field::new("value_counts_json", DataType::Binary, true),
        Field::new("null_value_counts_json", DataType::Binary, true),
        Field::new("nan_value_counts_json", DataType::Binary, true),
        Field::new("lower_bounds_json", DataType::Binary, true),
        Field::new("upper_bounds_json", DataType::Binary, true),
        Field::new("key_metadata", DataType::Binary, true),
        Field::new("split_offsets_json", DataType::Binary, true),
        Field::new("equality_ids_json", DataType::Binary, true),
        Field::new("sort_order_id", DataType::Int32, true),
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
    kv.insert("root-manifest".to_string(), "true".to_string());
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

    Ok(RootManifestMetadata {
        schema,
        schema_id,
        partition_spec,
        format_version,
        snapshot_id,
        sequence_number,
        parent_snapshot_id,
    })
}

// ============================================================================
// Writer: RootManifestEntry -> Parquet
// ============================================================================

/// Write root manifest entries to a Parquet-format byte buffer.
pub fn write_root_manifest(
    entries: &[RootManifestEntry],
    metadata: &RootManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<u8>> {
    let kv_metadata = encode_root_manifest_metadata(metadata);
    let schema = Arc::new(root_manifest_arrow_schema().with_metadata(kv_metadata));
    let batch = root_manifest_entries_to_record_batch(entries, &schema, partition_type)?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to create parquet writer: {e}"),
        )
    })?;

    writer.write(&batch).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to write batch: {e}"),
        )
    })?;
    writer.close().map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            format!("Failed to close writer: {e}"),
        )
    })?;

    Ok(buf)
}

fn root_manifest_entries_to_record_batch(
    entries: &[RootManifestEntry],
    schema: &Arc<ArrowSchema>,
    partition_type: &StructType,
) -> Result<RecordBatch> {
    let n = entries.len();

    // Discriminator
    let mut entry_type = Int32Builder::with_capacity(n);

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

    // Inline entry columns
    let mut status = Int32Builder::with_capacity(n);
    let mut snapshot_id = Int64Builder::with_capacity(n);
    let mut seq_num = Int64Builder::with_capacity(n);
    let mut file_seq = Int64Builder::with_capacity(n);
    let mut content = Int32Builder::with_capacity(n);
    let mut file_path = StringBuilder::with_capacity(n, n * 100);
    let mut file_format = StringBuilder::with_capacity(n, n * 8);
    let mut partition_json = StringBuilder::with_capacity(n, n * 64);
    let mut record_count = Int64Builder::with_capacity(n);
    let mut file_size = Int64Builder::with_capacity(n);
    let mut column_sizes_json = BinaryBuilder::with_capacity(n, n * 128);
    let mut value_counts_json = BinaryBuilder::with_capacity(n, n * 128);
    let mut null_value_counts_json = BinaryBuilder::with_capacity(n, n * 128);
    let mut nan_value_counts_json = BinaryBuilder::with_capacity(n, n * 64);
    let mut lower_bounds_json = BinaryBuilder::with_capacity(n, n * 256);
    let mut upper_bounds_json = BinaryBuilder::with_capacity(n, n * 256);
    let mut key_metadata_b = BinaryBuilder::with_capacity(n, n * 16);
    let mut split_offsets_json = BinaryBuilder::with_capacity(n, n * 64);
    let mut equality_ids_json = BinaryBuilder::with_capacity(n, n * 32);
    let mut sort_order_id = Int32Builder::with_capacity(n);
    let mut part_spec_id = Int32Builder::with_capacity(n);

    for entry in entries {
        match entry {
            RootManifestEntry::ManifestRef { manifest_file: mf, mdv } => {
                entry_type.append_value(RootEntryType::ManifestRef as i32);

                // Fill manifest ref columns
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
                // Partitions as JSON
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
                    Some(bm) => mdv_bitmap.append_value(bm.as_slice()),
                    None => mdv_bitmap.append_null(),
                }

                // Null out inline columns
                status.append_null();
                snapshot_id.append_null();
                seq_num.append_null();
                file_seq.append_null();
                content.append_null();
                file_path.append_null();
                file_format.append_null();
                partition_json.append_null();
                record_count.append_null();
                file_size.append_null();
                column_sizes_json.append_null();
                value_counts_json.append_null();
                null_value_counts_json.append_null();
                nan_value_counts_json.append_null();
                lower_bounds_json.append_null();
                upper_bounds_json.append_null();
                key_metadata_b.append_null();
                split_offsets_json.append_null();
                equality_ids_json.append_null();
                sort_order_id.append_null();
                part_spec_id.append_null();
            }
            RootManifestEntry::Inline(me) => {
                let df = &me.data_file;
                let etype = match df.content {
                    DataContentType::Data => RootEntryType::InlineData,
                    DataContentType::EqualityDeletes | DataContentType::PositionDeletes => {
                        RootEntryType::InlineDelete
                    }
                };
                entry_type.append_value(etype as i32);

                // Null out manifest ref columns
                manifest_path.append_null();
                manifest_length.append_null();
                manifest_content.append_null();
                manifest_seq_number.append_null();
                manifest_min_seq_number.append_null();
                manifest_added_snapshot.append_null();
                manifest_added_files.append_null();
                manifest_existing_files.append_null();
                manifest_deleted_files.append_null();
                manifest_added_rows.append_null();
                manifest_existing_rows.append_null();
                manifest_deleted_rows.append_null();
                manifest_partitions_json.append_null();
                manifest_key_metadata.append_null();
                manifest_first_row_id.append_null();
                mdv_bitmap.append_null();

                // Fill inline columns
                status.append_value(me.status as i32);
                match me.snapshot_id {
                    Some(v) => snapshot_id.append_value(v),
                    None => snapshot_id.append_null(),
                }
                match me.sequence_number {
                    Some(v) => seq_num.append_value(v),
                    None => seq_num.append_null(),
                }
                match me.file_sequence_number {
                    Some(v) => file_seq.append_value(v),
                    None => file_seq.append_null(),
                }
                content.append_value(df.content as i32);
                file_path.append_value(&df.file_path);
                file_format.append_value(df.file_format.to_string().to_ascii_uppercase());

                let part_str = serialize_partition_json(&df.partition, partition_type);
                partition_json.append_value(&part_str);

                record_count.append_value(df.record_count as i64);
                file_size.append_value(df.file_size_in_bytes as i64);

                append_map_json(&mut column_sizes_json, &serialize_i64_map(&df.column_sizes));
                append_map_json(&mut value_counts_json, &serialize_i64_map(&df.value_counts));
                append_map_json(
                    &mut null_value_counts_json,
                    &serialize_i64_map(&df.null_value_counts),
                );
                append_map_json(
                    &mut nan_value_counts_json,
                    &serialize_i64_map(&df.nan_value_counts),
                );
                append_map_json(&mut lower_bounds_json, &serialize_bounds_map(&df.lower_bounds));
                append_map_json(&mut upper_bounds_json, &serialize_bounds_map(&df.upper_bounds));

                match &df.key_metadata {
                    Some(km) => key_metadata_b.append_value(km.as_slice()),
                    None => key_metadata_b.append_null(),
                }
                append_opt_json(&mut split_offsets_json, &df.split_offsets);
                append_opt_json(&mut equality_ids_json, &df.equality_ids);
                match df.sort_order_id {
                    Some(v) => sort_order_id.append_value(v),
                    None => sort_order_id.append_null(),
                }
                part_spec_id.append_value(df.partition_spec_id);
            }
        }
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(entry_type.finish()),
        // Manifest ref columns
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
        // Inline entry columns
        Arc::new(status.finish()),
        Arc::new(snapshot_id.finish()),
        Arc::new(seq_num.finish()),
        Arc::new(file_seq.finish()),
        Arc::new(content.finish()),
        Arc::new(file_path.finish()),
        Arc::new(file_format.finish()),
        Arc::new(partition_json.finish()),
        Arc::new(record_count.finish()),
        Arc::new(file_size.finish()),
        Arc::new(column_sizes_json.finish()),
        Arc::new(value_counts_json.finish()),
        Arc::new(null_value_counts_json.finish()),
        Arc::new(nan_value_counts_json.finish()),
        Arc::new(lower_bounds_json.finish()),
        Arc::new(upper_bounds_json.finish()),
        Arc::new(key_metadata_b.finish()),
        Arc::new(split_offsets_json.finish()),
        Arc::new(equality_ids_json.finish()),
        Arc::new(sort_order_id.finish()),
        Arc::new(part_spec_id.finish()),
    ];

    RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to build RecordBatch: {e}")))
}

// ============================================================================
// Reader: Parquet -> RootManifestEntry
// ============================================================================

/// Read root manifest from Parquet bytes.
pub fn read_root_manifest(
    bytes: &[u8],
) -> Result<(RootManifestMetadata, Vec<RootManifestEntry>)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(bytes))
        .map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to open root manifest: {e}"),
            )
        })?;

    let arrow_schema = reader.schema();
    let arrow_meta = arrow_schema.metadata();

    // Convert to HashMap<String, String> for our decoder
    let metadata = decode_root_manifest_metadata(arrow_meta)?;

    let partition_type = metadata.partition_spec.partition_type(&metadata.schema)?;

    let batch_reader = reader.build().map_err(|e| {
        Error::new(
            ErrorKind::DataInvalid,
            format!("Failed to build reader: {e}"),
        )
    })?;

    let mut entries = Vec::new();
    for batch_result in batch_reader {
        let batch = batch_result.map_err(|e| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Failed to read batch: {e}"),
            )
        })?;
        let batch_entries =
            record_batch_to_root_manifest_entries(&batch, &metadata, &partition_type)?;
        entries.extend(batch_entries);
    }

    Ok((metadata, entries))
}

fn record_batch_to_root_manifest_entries(
    batch: &RecordBatch,
    metadata: &RootManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<RootManifestEntry>> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let entry_type_arr = col_i32(batch, "entry_type")?;

    // Manifest ref columns
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

    // Inline entry columns
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
        let etype: RootEntryType = entry_type_arr.value(i).try_into()?;

        match etype {
            RootEntryType::ManifestRef => {
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

                let manifest_file = ManifestFile {
                    manifest_path: mf_path,
                    manifest_length: mf_length,
                    partition_spec_id: metadata.partition_spec.spec_id(),
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
            RootEntryType::InlineData | RootEntryType::InlineDelete => {
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

                let fformat: DataFileFormat = file_format_arr
                    .and_then(|a| {
                        if Array::is_null(a, i) {
                            None
                        } else {
                            Some(a.value(i))
                        }
                    })
                    .unwrap_or("PARQUET")
                    .parse()?;

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
        }
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

        let (read_meta, read_entries) = read_root_manifest(&bytes).unwrap();

        // Verify metadata
        assert_eq!(read_meta.snapshot_id, 100);
        assert_eq!(read_meta.sequence_number, 5);
        assert_eq!(read_meta.parent_snapshot_id, Some(99));
        assert_eq!(read_meta.schema_id, 0);

        // Verify entry count and types
        assert_eq!(read_entries.len(), 4);

        // First entry: ManifestRef
        match &read_entries[0] {
            RootManifestEntry::ManifestRef { manifest_file, mdv } => {
                assert_eq!(manifest_file.manifest_path, "s3://bucket/metadata/m0.avro");
                assert_eq!(manifest_file.manifest_length, 4096);
                assert_eq!(manifest_file.added_files_count, Some(10));
                assert!(mdv.is_none());
            }
            _ => panic!("Expected ManifestRef"),
        }

        // Second entry: Inline
        match &read_entries[1] {
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

        // Fourth entry: Inline
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
        let (_, read_entries) = read_root_manifest(&bytes).unwrap();

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
        let (_, read_entries) = read_root_manifest(&bytes).unwrap();

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
        let (read_meta, read_entries) = read_root_manifest(&bytes).unwrap();

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
        let (_, read_entries) = read_root_manifest(&bytes).unwrap();

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
    fn to_manifest_list_counts() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let metadata = test_metadata(&schema, &partition_spec);

        let entries = vec![
            RootManifestEntry::ManifestRef {
                manifest_file: test_manifest_file("s3://bucket/metadata/m0.avro"),
                mdv: None,
            },
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/a.parquet", 100)),
            RootManifestEntry::Inline(test_inline_entry("s3://bucket/data/b.parquet", 200)),
        ];

        let rm = RootManifest::new(metadata, entries);

        // Verify inline_count
        assert_eq!(rm.inline_count(), 2);
        assert_eq!(rm.manifest_refs().count(), 1);

        let ml = rm.to_manifest_list();
        let ml_entries = ml.entries();

        // Should have 1 real manifest ref + 1 synthetic for inline data
        assert_eq!(ml_entries.len(), 2);

        // First is the real manifest ref
        assert_eq!(ml_entries[0].manifest_path, "s3://bucket/metadata/m0.avro");

        // Second is synthetic for inline data entries
        assert_eq!(ml_entries[1].manifest_path, "");
        assert_eq!(ml_entries[1].content, ManifestContentType::Data);
        assert_eq!(ml_entries[1].added_files_count, Some(2));
        assert_eq!(ml_entries[1].added_rows_count, Some(300));
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
}
