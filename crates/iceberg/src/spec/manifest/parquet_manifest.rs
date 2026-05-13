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

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::builder::{
    BinaryBuilder, Int32Builder, Int64Builder, ListBuilder, StringBuilder,
    StructBuilder,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use super::{ManifestEntry, ManifestMetadata, ManifestStatus};
use crate::error::Result;
use crate::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, FormatVersion,
    ManifestContentType, PartitionSpec, PrimitiveLiteral, Schema, SchemaRef,
    Struct, StructType,
};
use crate::{Error, ErrorKind};

// ============================================================================
// Arrow Schema Construction
// ============================================================================

/// Build an Arrow schema for manifest entries.
///
/// Fields are flattened (no nested data_file struct) for optimal projection.
/// The partition field is a struct whose shape depends on the partition spec.
pub fn manifest_arrow_schema(partition_type: &StructType) -> ArrowSchema {
    let partition_fields: Vec<Arc<Field>> = partition_type
        .fields()
        .iter()
        .map(|f| {
            Arc::new(Field::new(
                f.name.as_str(),
                iceberg_to_arrow_type(&f.field_type),
                !f.required,
            ))
        })
        .collect();

    let partition_struct = if partition_fields.is_empty() {
        // Unpartitioned: use a nullable empty struct
        DataType::Struct(Vec::<Arc<Field>>::new().into())
    } else {
        DataType::Struct(partition_fields.into())
    };

    // Map entries: key=i32 (field_id), value=i64 or binary
    let map_i64_type = DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("key", DataType::Int32, false)),
                    Arc::new(Field::new("value", DataType::Int64, true)),
                ]
                .into(),
            ),
            false,
        )),
        false,
    );

    let map_binary_type = DataType::Map(
        Arc::new(Field::new(
            "entries",
            DataType::Struct(
                vec![
                    Arc::new(Field::new("key", DataType::Int32, false)),
                    Arc::new(Field::new("value", DataType::Binary, true)),
                ]
                .into(),
            ),
            false,
        )),
        false,
    );

    ArrowSchema::new(vec![
        Field::new("status", DataType::Int32, false),
        Field::new("snapshot_id", DataType::Int64, true),
        Field::new("sequence_number", DataType::Int64, true),
        Field::new("file_sequence_number", DataType::Int64, true),
        Field::new("content", DataType::Int32, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("file_format", DataType::Utf8, false),
        Field::new("partition", partition_struct, false),
        Field::new("record_count", DataType::Int64, false),
        Field::new("file_size_in_bytes", DataType::Int64, false),
        Field::new("column_sizes", map_i64_type.clone(), true),
        Field::new("value_counts", map_i64_type.clone(), true),
        Field::new("null_value_counts", map_i64_type.clone(), true),
        Field::new("nan_value_counts", map_i64_type, true),
        Field::new("lower_bounds", map_binary_type.clone(), true),
        Field::new("upper_bounds", map_binary_type, true),
        Field::new("key_metadata", DataType::Binary, true),
        Field::new(
            "split_offsets",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
            true,
        ),
        Field::new(
            "equality_ids",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
        Field::new("sort_order_id", DataType::Int32, true),
    ])
}

/// Convert an Iceberg primitive type to Arrow DataType.
fn iceberg_to_arrow_type(ty: &crate::spec::Type) -> DataType {
    use crate::spec::{PrimitiveType, Type};
    match ty {
        Type::Primitive(p) => match p {
            PrimitiveType::Boolean => DataType::Boolean,
            PrimitiveType::Int => DataType::Int32,
            PrimitiveType::Long => DataType::Int64,
            PrimitiveType::Float => DataType::Float32,
            PrimitiveType::Double => DataType::Float64,
            PrimitiveType::Date => DataType::Date32,
            PrimitiveType::Time => DataType::Time64(arrow_schema::TimeUnit::Microsecond),
            PrimitiveType::Timestamp => {
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None)
            }
            PrimitiveType::Timestamptz => DataType::Timestamp(
                arrow_schema::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            PrimitiveType::TimestampNs => {
                DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None)
            }
            PrimitiveType::TimestamptzNs => DataType::Timestamp(
                arrow_schema::TimeUnit::Nanosecond,
                Some("UTC".into()),
            ),
            PrimitiveType::String => DataType::Utf8,
            PrimitiveType::Uuid => DataType::FixedSizeBinary(16),
            PrimitiveType::Fixed(len) => DataType::FixedSizeBinary(*len as i32),
            PrimitiveType::Binary => DataType::Binary,
            PrimitiveType::Decimal { precision, scale } => {
                DataType::Decimal128(*precision as u8, *scale as i8)
            }
        },
        Type::Struct(s) => {
            let fields: Vec<Arc<Field>> = s
                .fields()
                .iter()
                .map(|f| {
                    Arc::new(Field::new(
                        f.name.as_str(),
                        iceberg_to_arrow_type(&f.field_type),
                        !f.required,
                    ))
                })
                .collect();
            DataType::Struct(fields.into())
        }
        Type::List(l) => DataType::List(Arc::new(Field::new(
            "item",
            iceberg_to_arrow_type(&l.element_field.field_type),
            !l.element_field.required,
        ))),
        Type::Map(m) => DataType::Map(
            Arc::new(Field::new(
                "entries",
                DataType::Struct(
                    vec![
                        Arc::new(Field::new(
                            "key",
                            iceberg_to_arrow_type(&m.key_field.field_type),
                            false,
                        )),
                        Arc::new(Field::new(
                            "value",
                            iceberg_to_arrow_type(&m.value_field.field_type),
                            !m.value_field.required,
                        )),
                    ]
                    .into(),
                ),
                false,
            )),
            false,
        ),
    }
}

// ============================================================================
// Manifest Metadata (Parquet file key-value metadata)
// ============================================================================

/// Encode manifest metadata as Parquet file key-value metadata.
pub fn encode_manifest_metadata(metadata: &ManifestMetadata) -> HashMap<String, String> {
    let mut kv = HashMap::new();
    kv.insert(
        "schema".to_string(),
        serde_json::to_string(&metadata.schema.as_ref()).unwrap_or_default(),
    );
    kv.insert(
        "schema-id".to_string(),
        metadata.schema_id.to_string(),
    );
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
        match metadata.format_version {
            FormatVersion::V1 => "1",
            FormatVersion::V2 => "2",
            FormatVersion::V3 => "3",
        }
        .to_string(),
    );
    kv.insert(
        "content".to_string(),
        match metadata.content {
            ManifestContentType::Data => "data",
            ManifestContentType::Deletes => "deletes",
        }
        .to_string(),
    );
    kv.insert("manifest-format".to_string(), "parquet".to_string());
    kv
}

// ============================================================================
// Parquet Manifest Reader
// ============================================================================

/// Read manifest entries from a Parquet-format manifest file.
///
/// This reads the full manifest (all columns). For projected reads,
/// use `read_parquet_manifest_projected()`.
pub fn read_parquet_manifest(
    bytes: &[u8],
) -> Result<(ManifestMetadata, Vec<ManifestEntry>)> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::copy_from_slice(bytes))
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to open parquet manifest: {e}")))?;

    // Parse metadata from Parquet file key-value metadata
    let parquet_metadata = reader.metadata();
    let kv_metadata = parquet_metadata.file_metadata().key_value_metadata();
    let metadata = parse_parquet_manifest_metadata(kv_metadata)?;

    let partition_type = metadata.partition_spec.partition_type(&metadata.schema)?;

    // Build reader (all columns)
    let batch_reader = reader
        .build()
        .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to build parquet reader: {e}")))?;

    let mut entries = Vec::new();
    for batch_result in batch_reader {
        let batch = batch_result
            .map_err(|e| Error::new(ErrorKind::DataInvalid, format!("Failed to read batch: {e}")))?;
        let batch_entries = record_batch_to_manifest_entries(
            &batch,
            metadata.partition_spec.spec_id(),
            &partition_type,
            &metadata.schema,
        )?;
        entries.extend(batch_entries);
    }

    Ok((metadata, entries))
}

/// Parse ManifestMetadata from Parquet key-value metadata.
///
/// Converts Parquet KeyValue pairs to HashMap<String, Vec<u8>> to reuse
/// the existing ManifestMetadata::parse() which expects Avro user metadata format.
fn parse_parquet_manifest_metadata(
    kv: Option<&Vec<parquet::file::metadata::KeyValue>>,
) -> Result<ManifestMetadata> {
    let kv = kv.ok_or_else(|| {
        Error::new(
            ErrorKind::DataInvalid,
            "Parquet manifest missing key-value metadata",
        )
    })?;

    let mut map: HashMap<String, Vec<u8>> = HashMap::new();
    for entry in kv {
        if let Some(ref v) = entry.value {
            map.insert(entry.key.clone(), v.as_bytes().to_vec());
        }
    }

    ManifestMetadata::parse(&map)
}

/// Convert a RecordBatch of manifest entries to ManifestEntry structs.
///
/// This is the core conversion from Arrow columnar representation to
/// the Iceberg manifest entry model.
fn record_batch_to_manifest_entries(
    batch: &RecordBatch,
    partition_spec_id: i32,
    _partition_type: &StructType,
    _schema: &Schema,
) -> Result<Vec<ManifestEntry>> {
    use arrow_array::*;

    let num_rows = batch.num_rows();
    let mut entries = Vec::with_capacity(num_rows);

    // Extract column arrays
    let status_arr = batch
        .column_by_name("status")
        .and_then(|a| a.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "missing status column"))?;

    let snapshot_id_arr = batch
        .column_by_name("snapshot_id")
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>());

    let seq_num_arr = batch
        .column_by_name("sequence_number")
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>());

    let file_seq_arr = batch
        .column_by_name("file_sequence_number")
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>());

    let content_arr = batch
        .column_by_name("content")
        .and_then(|a| a.as_any().downcast_ref::<Int32Array>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "missing content column"))?;

    let file_path_arr = batch
        .column_by_name("file_path")
        .and_then(|a| a.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "missing file_path column"))?;

    let file_format_arr = batch
        .column_by_name("file_format")
        .and_then(|a| a.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "missing file_format column"))?;

    let record_count_arr = batch
        .column_by_name("record_count")
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "missing record_count column"))?;

    let file_size_arr = batch
        .column_by_name("file_size_in_bytes")
        .and_then(|a| a.as_any().downcast_ref::<Int64Array>())
        .ok_or_else(|| Error::new(ErrorKind::DataInvalid, "missing file_size_in_bytes column"))?;

    for i in 0..num_rows {
        let status: ManifestStatus = status_arr.value(i).try_into()?;

        let snapshot_id = snapshot_id_arr
            .and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) });

        let sequence_number = seq_num_arr
            .and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) });

        let file_sequence_number = file_seq_arr
            .and_then(|a| if a.is_null(i) { None } else { Some(a.value(i)) });

        let content: DataContentType = content_arr.value(i).try_into()?;
        let file_path = file_path_arr.value(i).to_string();
        let file_format: DataFileFormat = file_format_arr.value(i).parse()?;
        let record_count = record_count_arr.value(i) as u64;
        let file_size = file_size_arr.value(i) as u64;

        // TODO: Parse partition, column_sizes, bounds from batch
        // For now, create minimal DataFile with core fields
        let data_file = DataFile {
            content,
            file_path,
            file_format,
            partition: Struct::empty(),
            record_count,
            file_size_in_bytes: file_size,
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
            partition_spec_id,
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
