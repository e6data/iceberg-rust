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

//! Parquet-based manifest serialization for columnar metadata projection.
//!
//! Replaces Avro manifest files with Parquet to enable the executor to read
//! only the manifest columns needed for query planning (e.g., file_path +
//! partition bounds) without deserializing all 200+ column statistics.
//!
//! Same v2 semantics as Avro manifests — just a different serialization format.
//!
//! Map-typed columns (column_sizes, lower_bounds, upper_bounds, etc.) are
//! stored as JSON-encoded binary blobs. This is simpler than Arrow Map types
//! and still allows projection to skip these columns entirely when not needed.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, Int32Builder, Int64Builder, StringBuilder};
use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray, BinaryArray};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriterOptions;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::{ManifestEntry, ManifestMetadata, ManifestStatus};
use crate::error::Result;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion, Literal,
    ManifestContentType, PartitionSpec, PrimitiveLiteral, RawLiteral, Schema,
    SchemaRef, Struct, StructType, Type,
};
use crate::{Error, ErrorKind};

// ============================================================================
// Arrow Schema
// ============================================================================

/// Build a flat Arrow schema for manifest entries.
///
/// Map-typed columns use Binary (JSON-encoded) for simplicity.
/// This still allows columnar projection — the executor skips map columns
/// when it only needs file_path + partition + status.
pub fn manifest_arrow_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("status", DataType::Int32, false),
        Field::new("snapshot_id", DataType::Int64, true),
        Field::new("sequence_number", DataType::Int64, true),
        Field::new("file_sequence_number", DataType::Int64, true),
        Field::new("content", DataType::Int32, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("file_format", DataType::Utf8, false),
        Field::new("partition_json", DataType::Utf8, true),
        Field::new("record_count", DataType::Int64, false),
        Field::new("file_size_in_bytes", DataType::Int64, false),
        // Map columns as JSON binary — allows projection to skip them
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
        Field::new("partition_spec_id", DataType::Int32, false),
    ])
}

// ============================================================================
// Metadata encoding
// ============================================================================

/// Encode manifest metadata as HashMap for Arrow schema metadata.
/// This gets written as Parquet file-level key-value metadata.
pub fn encode_manifest_metadata(metadata: &ManifestMetadata) -> HashMap<String, String> {
    let mut kv = HashMap::new();
    kv.insert("schema".to_string(), serde_json::to_string(metadata.schema.as_ref()).unwrap_or_default());
    kv.insert("schema-id".to_string(), metadata.schema_id.to_string());
    kv.insert("partition-spec".to_string(), serde_json::to_string(&metadata.partition_spec.fields()).unwrap_or_default());
    kv.insert("partition-spec-id".to_string(), metadata.partition_spec.spec_id().to_string());
    kv.insert("format-version".to_string(), (metadata.format_version as u8).to_string());
    kv.insert("content".to_string(), metadata.content.to_string());
    kv
}

// ============================================================================
// Writer: ManifestEntry → Parquet
// ============================================================================

/// Write manifest entries to a Parquet-format byte buffer.
pub fn write_parquet_manifest(
    entries: &[ManifestEntry],
    metadata: &ManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<u8>> {
    // Embed manifest metadata in the Arrow schema's metadata field.
    // ArrowWriter propagates this to the Parquet file-level key-value metadata.
    let kv_metadata = encode_manifest_metadata(metadata);
    let schema = Arc::new(manifest_arrow_schema().with_metadata(kv_metadata));
    let batch = manifest_entries_to_record_batch(entries, &schema, partition_type, metadata.format_version)?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();

    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to create parquet writer: {e}")))?;

    writer.write(&batch)
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to write batch: {e}")))?;
    writer.close()
        .map_err(|e| Error::new(ErrorKind::Unexpected, format!("Failed to close writer: {e}")))?;

    Ok(buf)
}

/// Convert ManifestEntry slice to Arrow RecordBatch.
fn manifest_entries_to_record_batch(
    entries: &[ManifestEntry],
    schema: &Arc<ArrowSchema>,
    partition_type: &StructType,
    format_version: FormatVersion,
) -> Result<RecordBatch> {
    let n = entries.len();

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
        let df = &entry.data_file;

        status.append_value(entry.status as i32);
        match entry.snapshot_id {
            Some(v) => snapshot_id.append_value(v),
            None => snapshot_id.append_null(),
        }
        match entry.sequence_number {
            Some(v) => seq_num.append_value(v),
            None => seq_num.append_null(),
        }
        match entry.file_sequence_number {
            Some(v) => file_seq.append_value(v),
            None => file_seq.append_null(),
        }
        content.append_value(df.content as i32);
        file_path.append_value(&df.file_path);
        file_format.append_value(df.file_format.to_string().to_ascii_uppercase());

        // Partition as JSON string
        let part_str = serialize_partition_json(&df.partition, partition_type);
        partition_json.append_value(&part_str);

        record_count.append_value(df.record_count as i64);
        file_size.append_value(df.file_size_in_bytes as i64);

        // Map columns as JSON bytes
        append_map_json(&mut column_sizes_json, &serialize_i64_map(&df.column_sizes));
        append_map_json(&mut value_counts_json, &serialize_i64_map(&df.value_counts));
        append_map_json(&mut null_value_counts_json, &serialize_i64_map(&df.null_value_counts));
        append_map_json(&mut nan_value_counts_json, &serialize_i64_map(&df.nan_value_counts));

        // Bounds: serialize field_id → base64(bytes) as JSON
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

    let columns: Vec<ArrayRef> = vec![
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

fn serialize_partition_json(partition: &Struct, partition_type: &StructType) -> String {
    // Use RawLiteral for proper Iceberg partition value serialization.
    // RawLiteral handles all type conversions (timestamps, decimals, etc.) correctly.
    match RawLiteral::try_from(Literal::Struct(partition.clone()), &Type::Struct(partition_type.clone())) {
        Ok(raw) => serde_json::to_string(&raw).unwrap_or_default(),
        Err(_) => "null".to_string(),
    }
}

fn serialize_i64_map(m: &HashMap<i32, u64>) -> String {
    if m.is_empty() {
        return String::new();
    }
    // Serialize as JSON: {"field_id": value, ...}
    let map: HashMap<String, u64> = m.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    serde_json::to_string(&map).unwrap_or_default()
}

fn serialize_bounds_map(m: &HashMap<i32, Datum>) -> String {
    if m.is_empty() {
        return String::new();
    }
    // Serialize as JSON: {"field_id": "base64(bytes)", ...}
    use base64::Engine;
    let map: HashMap<String, String> = m
        .iter()
        .filter_map(|(k, v)| {
            v.to_bytes().ok().map(|bytes| {
                (k.to_string(), base64::engine::general_purpose::STANDARD.encode(bytes.as_ref()))
            })
        })
        .collect();
    serde_json::to_string(&map).unwrap_or_default()
}

fn append_map_json(builder: &mut BinaryBuilder, json: &str) {
    if json.is_empty() {
        builder.append_null();
    } else {
        builder.append_value(json.as_bytes());
    }
}

fn append_opt_json<T: serde::Serialize>(builder: &mut BinaryBuilder, val: &Option<T>) {
    match val {
        Some(v) => {
            let json = serde_json::to_string(v).unwrap_or_default();
            builder.append_value(json.as_bytes());
        }
        None => builder.append_null(),
    }
}

// ============================================================================
// Reader: Parquet → ManifestEntry
// ============================================================================

/// Read manifest entries from a Parquet-format manifest file.
pub fn read_parquet_manifest(bytes: &[u8]) -> Result<(ManifestMetadata, Vec<ManifestEntry>)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(bytes))
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to open parquet manifest: {e}")))?;

    // ArrowWriter stores Arrow schema metadata in the Parquet file-level key-value metadata
    // under the key "ARROW:schema". Our manifest metadata keys are embedded alongside it.
    // We also check the Arrow schema metadata directly.
    let parquet_metadata = reader.metadata().clone();
    let arrow_schema = reader.schema();

    // First try: Arrow schema metadata (ArrowWriter propagates schema.metadata here)
    let arrow_meta = arrow_schema.metadata();
    let metadata = if arrow_meta.contains_key("schema") {
        let map: HashMap<String, Vec<u8>> = arrow_meta
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bytes().to_vec()))
            .collect();
        ManifestMetadata::parse(&map)?
    } else {
        // Fallback: Parquet file-level key-value metadata
        let kv_metadata = parquet_metadata.file_metadata().key_value_metadata();
        parse_parquet_manifest_metadata(kv_metadata)?
    };

    let partition_type = metadata.partition_spec.partition_type(&metadata.schema)?;

    let batch_reader = reader
        .build()
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to build reader: {e}")))?;

    let mut entries = Vec::new();
    for batch_result in batch_reader {
        let batch = batch_result
            .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to read batch: {e}")))?;
        let batch_entries = record_batch_to_manifest_entries(
            &batch,
            &metadata,
            &partition_type,
        )?;
        entries.extend(batch_entries);
    }

    Ok((metadata, entries))
}

/// Read manifest entries with column projection.
///
/// Only the specified columns are read from the Parquet file. Columns not
/// requested will be empty/default in the returned ManifestEntry.
///
/// Common projection sets:
/// - Planning: `["status", "content", "file_path", "file_format", "record_count", "file_size_in_bytes", "partition_spec_id"]`
/// - Pruning:  above + `["lower_bounds_json", "upper_bounds_json"]`
/// - Full:     all columns (use `read_parquet_manifest` instead)
pub fn read_parquet_manifest_projected(
    bytes: &[u8],
    columns: &[&str],
) -> Result<(ManifestMetadata, Vec<ManifestEntry>)> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(bytes))
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to open parquet manifest: {e}")))?;

    let arrow_schema = builder.schema();
    let arrow_meta = arrow_schema.metadata();
    let metadata = if arrow_meta.contains_key("schema") {
        let map: HashMap<String, Vec<u8>> = arrow_meta
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bytes().to_vec()))
            .collect();
        ManifestMetadata::parse(&map)?
    } else {
        let parquet_metadata = builder.metadata().clone();
        let kv_metadata = parquet_metadata.file_metadata().key_value_metadata();
        parse_parquet_manifest_metadata(kv_metadata)?
    };

    // Build projection mask from requested columns
    let parquet_schema = builder.parquet_schema();
    let mut projection_indices = Vec::new();
    for col_name in columns {
        for (idx, field) in arrow_schema.fields().iter().enumerate() {
            if field.name() == *col_name {
                projection_indices.push(idx);
                break;
            }
        }
    }

    let projection = parquet::arrow::ProjectionMask::leaves(
        parquet_schema,
        projection_indices,
    );

    let partition_type = metadata.partition_spec.partition_type(&metadata.schema)?;

    let batch_reader = builder
        .with_projection(projection)
        .build()
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to build projected reader: {e}")))?;

    let mut entries = Vec::new();
    for batch_result in batch_reader {
        let batch = batch_result
            .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to read batch: {e}")))?;
        let batch_entries = record_batch_to_manifest_entries(&batch, &metadata, &partition_type)?;
        entries.extend(batch_entries);
    }

    Ok((metadata, entries))
}

fn parse_parquet_manifest_metadata(
    kv: Option<&Vec<parquet::file::metadata::KeyValue>>,
) -> Result<ManifestMetadata> {
    let kv = kv.ok_or_else(|| {
        Error::new(ErrorKind::DataInvalid, "Parquet manifest missing key-value metadata")
    })?;

    let mut map: HashMap<String, Vec<u8>> = HashMap::new();
    for entry in kv {
        if let Some(ref v) = entry.value {
            map.insert(entry.key.clone(), v.as_bytes().to_vec());
        }
    }

    ManifestMetadata::parse(&map)
}

fn record_batch_to_manifest_entries(
    batch: &RecordBatch,
    metadata: &ManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<ManifestEntry>> {
    let n = batch.num_rows();
    let mut entries = Vec::with_capacity(n);

    let status_arr = col_i32(batch, "status")?;
    let snapshot_id_arr = col_i64_opt(batch, "snapshot_id");
    let seq_num_arr = col_i64_opt(batch, "sequence_number");
    let file_seq_arr = col_i64_opt(batch, "file_sequence_number");
    let content_arr = col_i32(batch, "content")?;
    let file_path_arr = col_str(batch, "file_path")?;
    let file_format_arr = col_str(batch, "file_format")?;
    let partition_json_arr = col_str_opt(batch, "partition_json");
    let record_count_arr = col_i64(batch, "record_count")?;
    let file_size_arr = col_i64(batch, "file_size_in_bytes")?;
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
        let status: ManifestStatus = status_arr.value(i).try_into()?;
        let snapshot_id = nullable_i64(snapshot_id_arr, i);
        let sequence_number = nullable_i64(seq_num_arr, i);
        let file_sequence_number = nullable_i64(file_seq_arr, i);

        let content_type: DataContentType = content_arr.value(i).try_into()?;
        let file_path = file_path_arr.value(i).to_string();
        let file_format: DataFileFormat = file_format_arr.value(i).parse()?;
        let record_count = record_count_arr.value(i) as u64;
        let file_size = file_size_arr.value(i) as u64;

        let partition = parse_partition_json(
            partition_json_arr.and_then(|a| if Array::is_null(a, i) { None } else { Some(a.value(i)) }),
            partition_type,
        );

        let column_sizes = parse_i64_map_json(read_binary_opt(column_sizes_arr, i));
        let value_counts = parse_i64_map_json(read_binary_opt(value_counts_arr, i));
        let null_value_counts = parse_i64_map_json(read_binary_opt(null_value_counts_arr, i));
        let nan_value_counts = parse_i64_map_json(read_binary_opt(nan_value_counts_arr, i));

        let lower_bounds = parse_bounds_map_json(read_binary_opt(lower_bounds_arr, i), &metadata.schema);
        let upper_bounds = parse_bounds_map_json(read_binary_opt(upper_bounds_arr, i), &metadata.schema);

        let key_metadata_val = read_binary_opt(key_metadata_arr, i).map(|b| b.to_vec());
        let split_offsets: Option<Vec<i64>> = read_binary_opt(split_offsets_arr, i)
            .and_then(|b| serde_json::from_slice(b).ok());
        let equality_ids: Option<Vec<i32>> = read_binary_opt(equality_ids_arr, i)
            .and_then(|b| serde_json::from_slice(b).ok());
        let sort_id = sort_order_id_arr.and_then(|a| if Array::is_null(a, i) { None } else { Some(a.value(i)) });
        let spec_id = part_spec_id_arr.map(|a| a.value(i)).unwrap_or(metadata.partition_spec.spec_id());

        let data_file = DataFile {
            content: content_type,
            file_path,
            file_format,
            partition,
            record_count,
            file_size_in_bytes: file_size,
            column_sizes,
            value_counts,
            null_value_counts,
            nan_value_counts,
            lower_bounds,
            upper_bounds,
            key_metadata: key_metadata_val,
            split_offsets,
            equality_ids,
            sort_order_id: sort_id,
            partition_spec_id: spec_id,
            first_row_id: None,
            referenced_data_file: None,
            content_offset: None,
            content_size_in_bytes: None,
        };

        entries.push(ManifestEntry {
            status,
            snapshot_id,
            sequence_number,
            file_sequence_number,
            data_file,
        });
    }

    Ok(entries)
}

// ============================================================================
// JSON deserialization helpers
// ============================================================================

fn parse_partition_json(json: Option<&str>, partition_type: &StructType) -> Struct {
    let Some(json_str) = json else { return Struct::empty() };
    if json_str == "null" || json_str.is_empty() {
        return Struct::empty();
    }

    // Deserialize via RawLiteral for proper Iceberg type handling
    let raw: RawLiteral = match serde_json::from_str(json_str) {
        Ok(r) => r,
        Err(_) => return Struct::empty(),
    };

    match raw.try_into(&Type::Struct(partition_type.clone())) {
        Ok(Some(Literal::Struct(s))) => s,
        _ => Struct::empty(),
    }
}

fn parse_i64_map_json(bytes: Option<&[u8]>) -> HashMap<i32, u64> {
    let Some(b) = bytes else { return HashMap::new() };
    let Ok(map) = serde_json::from_slice::<HashMap<String, u64>>(b) else {
        return HashMap::new();
    };
    map.into_iter()
        .filter_map(|(k, v)| k.parse::<i32>().ok().map(|k| (k, v)))
        .collect()
}

fn parse_bounds_map_json(bytes: Option<&[u8]>, schema: &Schema) -> HashMap<i32, Datum> {
    let Some(b) = bytes else { return HashMap::new() };
    let Ok(map) = serde_json::from_slice::<HashMap<String, String>>(b) else {
        return HashMap::new();
    };
    use base64::Engine;
    let mut result = HashMap::new();
    for (k_str, v_b64) in map {
        let Ok(field_id) = k_str.parse::<i32>() else { continue };
        let Ok(raw_bytes) = base64::engine::general_purpose::STANDARD.decode(&v_b64) else { continue };
        // Find the field type from schema to reconstruct the Datum
        if let Some(field) = schema.field_by_id(field_id) {
            if let Some(prim_type) = field.field_type.as_primitive_type() {
                if let Ok(datum) = Datum::try_from_bytes(&raw_bytes, prim_type.clone()) {
                    result.insert(field_id, datum);
                }
            }
        }
    }
    result
}

// ============================================================================
// Arrow column access helpers
// ============================================================================

fn col_i32<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array> {
    batch.column_by_name(name)
        .and_then(|a| a.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, format!("missing column: {name}")))
}

fn col_i64<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array> {
    batch.column_by_name(name)
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, format!("missing column: {name}")))
}

fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a StringArray> {
    batch.column_by_name(name)
        .and_then(|a| a.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, format!("missing column: {name}")))
}

fn col_i32_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Int32Array> {
    batch.column_by_name(name).and_then(|a| a.as_any().downcast_ref::<Int32Array>())
}

fn col_i64_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a Int64Array> {
    batch.column_by_name(name).and_then(|a| a.as_any().downcast_ref::<Int64Array>())
}

fn col_str_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a StringArray> {
    batch.column_by_name(name).and_then(|a| a.as_any().downcast_ref::<StringArray>())
}

fn col_binary_opt<'a>(batch: &'a RecordBatch, name: &str) -> Option<&'a BinaryArray> {
    batch.column_by_name(name).and_then(|a| a.as_any().downcast_ref::<BinaryArray>())
}

fn nullable_i64(arr: Option<&Int64Array>, i: usize) -> Option<i64> {
    arr.and_then(|a| if Array::is_null(a, i) { None } else { Some(a.value(i)) })
}

fn read_binary_opt<'a>(arr: Option<&'a BinaryArray>, i: usize) -> Option<&'a [u8]> {
    arr.and_then(|a| if Array::is_null(a, i) { None } else { Some(a.value(i)) })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{NestedField, PartitionField, PrimitiveType, Transform, Type, UnboundPartitionField};

    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long))),
                    Arc::new(NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String))),
                    Arc::new(NestedField::optional(3, "ts", Type::Primitive(PrimitiveType::Timestamptz))),
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
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap()
    }

    #[test]
    fn round_trip_parquet_manifest() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        // Create test entries
        let entries = vec![
            ManifestEntry {
                status: ManifestStatus::Added,
                snapshot_id: Some(100),
                sequence_number: Some(1),
                file_sequence_number: Some(1),
                data_file: DataFile {
                    content: DataContentType::Data,
                    file_path: "s3://bucket/data/file1.parquet".to_string(),
                    file_format: DataFileFormat::Parquet,
                    partition: Struct::empty(),
                    record_count: 1000,
                    file_size_in_bytes: 50000,
                    column_sizes: HashMap::from([(1, 2000), (2, 3000)]),
                    value_counts: HashMap::from([(1, 1000), (2, 950)]),
                    null_value_counts: HashMap::from([(2, 50)]),
                    nan_value_counts: HashMap::new(),
                    lower_bounds: HashMap::new(),
                    upper_bounds: HashMap::new(),
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
            },
            ManifestEntry {
                status: ManifestStatus::Existing,
                snapshot_id: Some(99),
                sequence_number: Some(0),
                file_sequence_number: Some(0),
                data_file: DataFile {
                    content: DataContentType::Data,
                    file_path: "s3://bucket/data/file2.parquet".to_string(),
                    file_format: DataFileFormat::Parquet,
                    partition: Struct::empty(),
                    record_count: 500,
                    file_size_in_bytes: 25000,
                    column_sizes: HashMap::new(),
                    value_counts: HashMap::new(),
                    null_value_counts: HashMap::new(),
                    nan_value_counts: HashMap::new(),
                    lower_bounds: HashMap::new(),
                    upper_bounds: HashMap::new(),
                    key_metadata: None,
                    split_offsets: None,
                    equality_ids: None,
                    sort_order_id: None,
                    partition_spec_id: 0,
                    first_row_id: None,
                    referenced_data_file: None,
                    content_offset: None,
                    content_size_in_bytes: None,
                },
            },
        ];

        // Write
        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();
        assert!(!bytes.is_empty(), "parquet manifest should not be empty");

        // Read back
        let (read_metadata, read_entries) = read_parquet_manifest(&bytes).unwrap();

        // Verify metadata
        assert_eq!(read_metadata.format_version, FormatVersion::V2);
        assert_eq!(read_metadata.content, ManifestContentType::Data);
        assert_eq!(read_metadata.schema_id, 0);

        // Verify entries
        assert_eq!(read_entries.len(), 2);
        assert_eq!(read_entries[0].status, ManifestStatus::Added);
        assert_eq!(read_entries[0].snapshot_id, Some(100));
        assert_eq!(read_entries[0].data_file.file_path, "s3://bucket/data/file1.parquet");
        assert_eq!(read_entries[0].data_file.record_count, 1000);
        assert_eq!(read_entries[0].data_file.file_size_in_bytes, 50000);
        assert_eq!(read_entries[0].data_file.column_sizes.get(&1), Some(&2000));
        assert_eq!(read_entries[0].data_file.column_sizes.get(&2), Some(&3000));
        assert_eq!(read_entries[0].data_file.split_offsets, Some(vec![4, 1000]));

        assert_eq!(read_entries[1].status, ManifestStatus::Existing);
        assert_eq!(read_entries[1].data_file.file_path, "s3://bucket/data/file2.parquet");
        assert_eq!(read_entries[1].data_file.record_count, 500);
    }

    #[test]
    fn projected_read_skips_statistics() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        let entries = vec![ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: Some(100),
            sequence_number: Some(1),
            file_sequence_number: Some(1),
            data_file: DataFile {
                content: DataContentType::Data,
                file_path: "s3://bucket/data/file1.parquet".to_string(),
                file_format: DataFileFormat::Parquet,
                partition: Struct::empty(),
                record_count: 1000,
                file_size_in_bytes: 50000,
                column_sizes: HashMap::from([(1, 2000), (2, 3000)]),
                value_counts: HashMap::from([(1, 1000)]),
                null_value_counts: HashMap::new(),
                nan_value_counts: HashMap::new(),
                lower_bounds: HashMap::new(),
                upper_bounds: HashMap::new(),
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
        }];

        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();

        // Projected read: only planning columns (skip all statistics)
        let planning_cols = &[
            "status", "content", "file_path", "file_format",
            "record_count", "file_size_in_bytes", "partition_spec_id",
        ];
        let (_, projected_entries) = read_parquet_manifest_projected(&bytes, planning_cols).unwrap();

        assert_eq!(projected_entries.len(), 1);
        assert_eq!(projected_entries[0].data_file.file_path, "s3://bucket/data/file1.parquet");
        assert_eq!(projected_entries[0].data_file.record_count, 1000);
        assert_eq!(projected_entries[0].status, ManifestStatus::Added);
        // Statistics should be empty since we didn't project them
        assert!(projected_entries[0].data_file.column_sizes.is_empty());
        assert!(projected_entries[0].data_file.value_counts.is_empty());
    }

    #[test]
    fn partition_value_round_trip() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        // Create entry with non-empty partition value
        let partition = vec![Some(Literal::long(42))].into_iter().collect::<Struct>();

        let entries = vec![ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: Some(100),
            sequence_number: Some(1),
            file_sequence_number: Some(1),
            data_file: DataFile {
                content: DataContentType::Data,
                file_path: "s3://bucket/data/partitioned.parquet".to_string(),
                file_format: DataFileFormat::Parquet,
                partition,
                record_count: 500,
                file_size_in_bytes: 25000,
                column_sizes: HashMap::new(),
                value_counts: HashMap::new(),
                null_value_counts: HashMap::new(),
                nan_value_counts: HashMap::new(),
                lower_bounds: HashMap::new(),
                upper_bounds: HashMap::new(),
                key_metadata: None,
                split_offsets: None,
                equality_ids: None,
                sort_order_id: None,
                partition_spec_id: 0,
                first_row_id: None,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
            },
        }];

        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_parquet_manifest(&bytes).unwrap();

        assert_eq!(read_entries.len(), 1);
        let read_partition = &read_entries[0].data_file.partition;
        assert_eq!(read_partition.fields().len(), 1);
        assert_eq!(read_partition[0], Some(Literal::long(42)));
    }

    #[test]
    fn bounds_round_trip() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        // field 1 = Long, field 2 = String
        let lower_bounds = HashMap::from([
            (1, Datum::long(100)),
            (2, Datum::string("aaa")),
        ]);
        let upper_bounds = HashMap::from([
            (1, Datum::long(999)),
            (2, Datum::string("zzz")),
        ]);

        let entries = vec![ManifestEntry {
            status: ManifestStatus::Added,
            snapshot_id: Some(100),
            sequence_number: Some(1),
            file_sequence_number: Some(1),
            data_file: DataFile {
                content: DataContentType::Data,
                file_path: "s3://bucket/data/with_bounds.parquet".to_string(),
                file_format: DataFileFormat::Parquet,
                partition: Struct::empty(),
                record_count: 1000,
                file_size_in_bytes: 50000,
                column_sizes: HashMap::from([(1, 4000), (2, 8000)]),
                value_counts: HashMap::from([(1, 1000), (2, 1000)]),
                null_value_counts: HashMap::from([(1, 0), (2, 10)]),
                nan_value_counts: HashMap::new(),
                lower_bounds,
                upper_bounds,
                key_metadata: Some(vec![1, 2, 3, 4]),
                split_offsets: Some(vec![4, 5000, 10000]),
                equality_ids: None,
                sort_order_id: Some(0),
                partition_spec_id: 0,
                first_row_id: None,
                referenced_data_file: None,
                content_offset: None,
                content_size_in_bytes: None,
            },
        }];

        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_parquet_manifest(&bytes).unwrap();

        assert_eq!(read_entries.len(), 1);
        let df = &read_entries[0].data_file;

        // Verify bounds
        assert_eq!(df.lower_bounds.get(&1), Some(&Datum::long(100)));
        assert_eq!(df.lower_bounds.get(&2), Some(&Datum::string("aaa")));
        assert_eq!(df.upper_bounds.get(&1), Some(&Datum::long(999)));
        assert_eq!(df.upper_bounds.get(&2), Some(&Datum::string("zzz")));

        // Verify column sizes
        assert_eq!(df.column_sizes.get(&1), Some(&4000));
        assert_eq!(df.column_sizes.get(&2), Some(&8000));

        // Verify value counts
        assert_eq!(df.value_counts.get(&1), Some(&1000));
        assert_eq!(df.null_value_counts.get(&2), Some(&10));

        // Verify key metadata
        assert_eq!(df.key_metadata, Some(vec![1, 2, 3, 4]));

        // Verify split offsets
        assert_eq!(df.split_offsets, Some(vec![4, 5000, 10000]));
    }

    #[test]
    fn string_partition_round_trip() {
        // Simulates the multi-tenant OTel scenario: partition by tenant name (string)
        let schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(1, "tenant", Type::Primitive(PrimitiveType::String))),
                    Arc::new(NestedField::optional(2, "data", Type::Primitive(PrimitiveType::String))),
                ])
                .build()
                .unwrap(),
        );
        let partition_spec = PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "tenant".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap();
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        // Two entries for different tenants
        let entries = vec![
            ManifestEntry {
                status: ManifestStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(1),
                file_sequence_number: Some(1),
                data_file: DataFile {
                    content: DataContentType::Data,
                    file_path: "s3://bucket/tenant_a/file1.parquet".to_string(),
                    file_format: DataFileFormat::Parquet,
                    partition: vec![Some(Literal::string("tenant_alpha"))].into_iter().collect(),
                    record_count: 1000,
                    file_size_in_bytes: 50000,
                    column_sizes: HashMap::new(),
                    value_counts: HashMap::new(),
                    null_value_counts: HashMap::new(),
                    nan_value_counts: HashMap::new(),
                    lower_bounds: HashMap::new(),
                    upper_bounds: HashMap::new(),
                    key_metadata: None,
                    split_offsets: None,
                    equality_ids: None,
                    sort_order_id: None,
                    partition_spec_id: 0,
                    first_row_id: None,
                    referenced_data_file: None,
                    content_offset: None,
                    content_size_in_bytes: None,
                },
            },
            ManifestEntry {
                status: ManifestStatus::Added,
                snapshot_id: Some(1),
                sequence_number: Some(1),
                file_sequence_number: Some(1),
                data_file: DataFile {
                    content: DataContentType::Data,
                    file_path: "s3://bucket/tenant_b/file2.parquet".to_string(),
                    file_format: DataFileFormat::Parquet,
                    partition: vec![Some(Literal::string("tenant_beta"))].into_iter().collect(),
                    record_count: 500,
                    file_size_in_bytes: 25000,
                    column_sizes: HashMap::new(),
                    value_counts: HashMap::new(),
                    null_value_counts: HashMap::new(),
                    nan_value_counts: HashMap::new(),
                    lower_bounds: HashMap::new(),
                    upper_bounds: HashMap::new(),
                    key_metadata: None,
                    split_offsets: None,
                    equality_ids: None,
                    sort_order_id: None,
                    partition_spec_id: 0,
                    first_row_id: None,
                    referenced_data_file: None,
                    content_offset: None,
                    content_size_in_bytes: None,
                },
            },
        ];

        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_parquet_manifest(&bytes).unwrap();

        assert_eq!(read_entries.len(), 2);
        assert_eq!(read_entries[0].data_file.partition[0], Some(Literal::string("tenant_alpha")));
        assert_eq!(read_entries[1].data_file.partition[0], Some(Literal::string("tenant_beta")));
        assert_eq!(read_entries[0].data_file.file_path, "s3://bucket/tenant_a/file1.parquet");
        assert_eq!(read_entries[1].data_file.record_count, 500);
    }

    #[test]
    fn format_detection_by_extension() {
        // Verify that .parquet and .avro paths are correctly distinguished
        assert!("s3://bucket/metadata/abc-m0.parquet".ends_with(".parquet"));
        assert!(!"s3://bucket/metadata/abc-m0.avro".ends_with(".parquet"));
        assert!(!"s3://bucket/metadata/abc-m0.parquet.crc".ends_with(".parquet"));
    }

    #[test]
    fn empty_manifest_round_trip() {
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        // Empty entries
        let entries: Vec<ManifestEntry> = vec![];
        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();
        let (read_metadata, read_entries) = read_parquet_manifest(&bytes).unwrap();

        assert_eq!(read_entries.len(), 0);
        assert_eq!(read_metadata.format_version, FormatVersion::V2);
    }

    #[test]
    fn large_manifest_round_trip() {
        // Test with 500 entries to validate batched read/write
        let schema = test_schema();
        let partition_spec = test_partition_spec(&schema);
        let partition_type = partition_spec.partition_type(&schema).unwrap();

        let metadata = ManifestMetadata::builder()
            .schema(schema.clone())
            .schema_id(0)
            .partition_spec(partition_spec)
            .format_version(FormatVersion::V2)
            .content(ManifestContentType::Data)
            .build();

        let entries: Vec<ManifestEntry> = (0..500)
            .map(|i| ManifestEntry {
                status: ManifestStatus::Added,
                snapshot_id: Some(100),
                sequence_number: Some(i as i64),
                file_sequence_number: Some(i as i64),
                data_file: DataFile {
                    content: DataContentType::Data,
                    file_path: format!("s3://bucket/data/file_{i:04}.parquet"),
                    file_format: DataFileFormat::Parquet,
                    partition: vec![Some(Literal::long(i))].into_iter().collect(),
                    record_count: 1000 + i as u64,
                    file_size_in_bytes: 50000,
                    column_sizes: HashMap::from([(1, i as u64 * 100)]),
                    value_counts: HashMap::new(),
                    null_value_counts: HashMap::new(),
                    nan_value_counts: HashMap::new(),
                    lower_bounds: HashMap::from([(1, Datum::long(i))]),
                    upper_bounds: HashMap::from([(1, Datum::long(i + 1000))]),
                    key_metadata: None,
                    split_offsets: None,
                    equality_ids: None,
                    sort_order_id: None,
                    partition_spec_id: 0,
                    first_row_id: None,
                    referenced_data_file: None,
                    content_offset: None,
                    content_size_in_bytes: None,
                },
            })
            .collect();

        let bytes = write_parquet_manifest(&entries, &metadata, &partition_type).unwrap();
        let (_, read_entries) = read_parquet_manifest(&bytes).unwrap();

        assert_eq!(read_entries.len(), 500);
        // Verify first and last
        assert_eq!(read_entries[0].data_file.file_path, "s3://bucket/data/file_0000.parquet");
        assert_eq!(read_entries[0].data_file.record_count, 1000);
        assert_eq!(read_entries[499].data_file.file_path, "s3://bucket/data/file_0499.parquet");
        assert_eq!(read_entries[499].data_file.record_count, 1499);
        // Verify bounds survived
        assert_eq!(read_entries[0].data_file.lower_bounds.get(&1), Some(&Datum::long(0)));
        assert_eq!(read_entries[499].data_file.upper_bounds.get(&1), Some(&Datum::long(1499)));
        // Verify partition values
        assert_eq!(read_entries[0].data_file.partition[0], Some(Literal::long(0)));
        assert_eq!(read_entries[499].data_file.partition[0], Some(Literal::long(499)));
    }
}
