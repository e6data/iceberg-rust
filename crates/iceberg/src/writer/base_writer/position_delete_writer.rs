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

//! This module provides `PositionDeleteFileWriter`.
//!
//! Position delete files contain (file_path, pos) pairs that identify
//! specific rows to delete from data files.

use std::collections::HashMap;

use arrow_array::RecordBatch;

use crate::spec::{DataContentType, DataFile, PartitionKey, Struct};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Builder for `PositionDeleteFileWriter`.
#[derive(Clone, Debug)]
pub struct PositionDeleteFileWriterBuilder<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    inner: RollingFileWriterBuilder<B, L, F>,
}

impl<B, L, F> PositionDeleteFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Create a new `PositionDeleteFileWriterBuilder` using a `RollingFileWriterBuilder`.
    ///
    /// # Arguments
    /// * `inner_builder` - The underlying rolling file writer builder
    pub fn new(inner_builder: RollingFileWriterBuilder<B, L, F>) -> Self {
        Self {
            inner: inner_builder,
        }
    }

    /// Build the `PositionDeleteFileWriter`.
    pub fn build(self) -> PositionDeleteFileWriter<B, L, F> {
        PositionDeleteFileWriter {
            inner_builder: self.inner,
            writers: HashMap::new(),
            partition_keys: HashMap::new(),
            closed: false,
        }
    }
}

/// A writer for position delete files.
///
/// Position delete files identify specific rows to delete by their
/// file path and row position. The input RecordBatch must have the
/// position delete schema:
/// - `file_path` (Utf8): Path of the data file containing the row
/// - `pos` (Int64): 0-indexed row position within the data file
///
/// This writer supports writing to multiple partitions. Each write call
/// includes the partition for that batch, and the writer maintains separate
/// files per partition (fanout pattern).
///
/// # Example
/// ```rust,ignore
/// use iceberg::writer::base_writer::position_delete_writer::PositionDeleteFileWriterBuilder;
///
/// // Create the writer - no partition needed at construction
/// let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder)
///     .build()
///     .await?;
///
/// // Write position deletes with partition info
/// let batch = RecordBatch::try_new(
///     position_delete_schema,
///     vec![
///         Arc::new(StringArray::from(vec!["s3://bucket/data/file.parquet"])),
///         Arc::new(Int64Array::from(vec![42])),
///     ],
/// )?;
/// writer.write(Some(&partition_key), batch).await?;
///
/// // Can write to different partitions - writer manages multiple files
/// writer.write(Some(&other_partition), other_batch).await?;
/// writer.write(None, unpartitioned_batch).await?;  // unpartitioned
///
/// // Close and get all delete files (one per partition)
/// let delete_files = writer.close().await?;
/// ```
pub struct PositionDeleteFileWriter<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
    /// Builder to create new RollingFileWriters on demand
    inner_builder: RollingFileWriterBuilder<B, L, F>,
    /// Map from partition data to the writer for that partition.
    /// Key is Option<Struct>: None for unpartitioned, Some(data) for partitioned.
    writers: HashMap<Option<Struct>, RollingFileWriter<B, L, F>>,
    /// Store the full PartitionKey for each partition (needed for metadata on close)
    partition_keys: HashMap<Option<Struct>, PartitionKey>,
    /// Whether the writer has been closed
    closed: bool,
}

impl<B, L, F> PositionDeleteFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Write a batch of position deletes for a specific partition.
    ///
    /// The batch must have the position delete schema with columns:
    /// - `file_path` (Utf8)
    /// - `pos` (Int64)
    ///
    /// # Arguments
    /// * `partition` - The partition these deletes belong to. Use `None` for unpartitioned tables.
    /// * `batch` - The RecordBatch containing file_path and pos columns.
    ///
    /// # Fanout Behavior
    /// This writer maintains separate files per partition. You can write to any partition
    /// at any time - the writer will route to the correct file. All files are closed
    /// when `close()` is called.
    pub async fn write(
        &mut self,
        partition: Option<&PartitionKey>,
        batch: RecordBatch,
    ) -> Result<()> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete writer has been closed.",
            ));
        }

        // Use partition data as the key (None for unpartitioned)
        let key = partition.map(|pk| pk.data().clone());

        // Get or create writer for this partition
        if !self.writers.contains_key(&key) {
            // Create new writer for this partition
            let new_writer = self.inner_builder.clone().build();
            self.writers.insert(key.clone(), new_writer);

            // Store the partition key for metadata when closing
            if let Some(pk) = partition {
                self.partition_keys.insert(key.clone(), pk.clone());
            }
        }

        // Write to the partition's writer
        let writer = self.writers.get_mut(&key).unwrap();
        let partition_key = partition.cloned();
        writer.write(&partition_key, &batch).await
    }

    /// Close the writer and return all resulting delete files.
    ///
    /// Each returned `DataFile` will have:
    /// - `content` set to `DataContentType::PositionDeletes`
    /// - Partition info set based on which partition the file belongs to
    ///
    /// # Returns
    /// A Vec of DataFiles, one or more per partition (depending on file size rolling).
    pub async fn close(&mut self) -> Result<Vec<DataFile>> {
        if self.closed {
            return Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete writer has been closed.",
            ));
        }

        self.closed = true;
        let mut all_files = Vec::new();

        // Close all writers and collect results
        for (partition_data, writer) in self.writers.drain() {
            let builders = writer.close().await?;

            for mut builder in builders {
                builder.content(DataContentType::PositionDeletes);

                // Set partition info if this was a partitioned write
                if let Some(ref data) = partition_data {
                    if let Some(pk) = self.partition_keys.get(&partition_data) {
                        builder.partition(data.clone());
                        builder.partition_spec_id(pk.spec().spec_id());
                    }
                }

                all_files.push(builder.build().map_err(|e| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Failed to build position delete file: {}", e),
                    )
                })?);
            }
        }

        Ok(all_files)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::arrow::arrow_reader::ArrowReaderMetadata;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::io::FileIOBuilder;
    use crate::spec::{
        DELETE_FILE_PATH_FIELD_ID, DELETE_FILE_POS_FIELD_ID, DataContentType, DataFileFormat,
        NestedField, PrimitiveType, Schema, Type,
    };
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;

    /// Create Arrow schema for position deletes with proper field IDs
    fn create_position_delete_arrow_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                DELETE_FILE_PATH_FIELD_ID.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                DELETE_FILE_POS_FIELD_ID.to_string(),
            )])),
        ])
    }

    /// Create Iceberg schema for position deletes
    fn create_position_delete_iceberg_schema() -> Schema {
        Schema::builder()
            .with_fields(vec![
                Arc::new(NestedField::required(
                    DELETE_FILE_PATH_FIELD_ID,
                    "file_path",
                    Type::Primitive(PrimitiveType::String),
                )),
                Arc::new(NestedField::required(
                    DELETE_FILE_POS_FIELD_ID,
                    "pos",
                    Type::Primitive(PrimitiveType::Long),
                )),
            ])
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_position_delete_writer() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIOBuilder::new_fs_io().build()?;
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        // Use position delete schema
        let schema = Arc::new(create_position_delete_iceberg_schema());
        let parquet_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);

        let rolling_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        // No partition at construction - partition is per-write now
        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder).build();

        // Create a batch of position deletes
        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec![
                "s3://bucket/data/file1.parquet",
                "s3://bucket/data/file2.parquet",
                "s3://bucket/data/file1.parquet",
            ])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ])?;

        // Write with no partition (unpartitioned table)
        writer.write(None, batch).await?;
        let delete_files = writer.close().await?;

        // Verify results
        assert_eq!(delete_files.len(), 1);
        let delete_file = &delete_files[0];

        // Check content type is PositionDeletes
        assert_eq!(delete_file.content, DataContentType::PositionDeletes);

        // Check record count
        assert_eq!(delete_file.record_count, 3);

        // Check file format
        assert_eq!(delete_file.file_format, DataFileFormat::Parquet);

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_field_ids() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIOBuilder::new_fs_io().build()?;
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let schema = Arc::new(create_position_delete_iceberg_schema());
        let parquet_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);

        let rolling_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder).build();

        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec!["s3://bucket/data/file1.parquet"])),
            Arc::new(Int64Array::from(vec![42])),
        ])?;

        writer.write(None, batch).await?;
        let delete_files = writer.close().await?;

        // Read back the Parquet file and verify field IDs
        let input_file = file_io.new_input(delete_files[0].file_path.clone())?;
        let input_content = input_file.read().await?;

        let parquet_reader = ArrowReaderMetadata::load(
            &input_content,
            parquet::arrow::arrow_reader::ArrowReaderOptions::default(),
        )?;

        let field_ids: Vec<i32> = parquet_reader
            .parquet_schema()
            .columns()
            .iter()
            .map(|col| col.self_type().get_basic_info().id())
            .collect();

        // Verify correct field IDs are written to Parquet
        assert_eq!(field_ids, vec![
            DELETE_FILE_PATH_FIELD_ID,
            DELETE_FILE_POS_FIELD_ID
        ]);

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_closed_error() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIOBuilder::new_fs_io().build()?;
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let schema = Arc::new(create_position_delete_iceberg_schema());
        let parquet_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);

        let rolling_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder).build();

        // Close the writer (even without writing anything)
        let _ = writer.close().await?;

        // Try to write after close - should fail
        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec!["s3://bucket/data/file1.parquet"])),
            Arc::new(Int64Array::from(vec![42])),
        ])?;

        let result = writer.write(None, batch).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().message().contains("closed"));

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_multiple_partitions() -> Result<()> {
        use crate::spec::{Literal, PartitionSpec, Struct, Transform};

        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIOBuilder::new_fs_io().build()?;
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);

        let schema = Arc::new(create_position_delete_iceberg_schema());
        let parquet_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema.clone());

        let rolling_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder).build();

        // Create partition spec and keys for testing
        // Using a simple identity partition on an "id" field
        let table_schema = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Int),
                    )),
                    Arc::new(NestedField::required(
                        2,
                        "data",
                        Type::Primitive(PrimitiveType::String),
                    )),
                ])
                .build()
                .unwrap(),
        );

        let partition_spec = PartitionSpec::builder(table_schema.clone())
            .add_partition_field("id", "id", Transform::Identity)
            .unwrap()
            .build()
            .unwrap();

        // Create two partition keys
        let partition1 = PartitionKey::new(
            partition_spec.clone(),
            table_schema.clone(),
            Struct::from_iter([Some(Literal::int(1))]),
        );
        let partition2 = PartitionKey::new(
            partition_spec.clone(),
            table_schema.clone(),
            Struct::from_iter([Some(Literal::int(2))]),
        );

        let arrow_schema = Arc::new(create_position_delete_arrow_schema());

        // Write deletes for partition 1
        let batch1 = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(StringArray::from(vec!["s3://bucket/id=1/file1.parquet"])),
            Arc::new(Int64Array::from(vec![10])),
        ])?;
        writer.write(Some(&partition1), batch1).await?;

        // Write deletes for partition 2
        let batch2 = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(StringArray::from(vec!["s3://bucket/id=2/file2.parquet"])),
            Arc::new(Int64Array::from(vec![20])),
        ])?;
        writer.write(Some(&partition2), batch2).await?;

        // Write more deletes for partition 1 (fanout - can go back!)
        let batch3 = RecordBatch::try_new(arrow_schema.clone(), vec![
            Arc::new(StringArray::from(vec!["s3://bucket/id=1/file1.parquet"])),
            Arc::new(Int64Array::from(vec![30])),
        ])?;
        writer.write(Some(&partition1), batch3).await?;

        // Close and verify
        let delete_files = writer.close().await?;

        // Should have 2 delete files (one per partition)
        assert_eq!(delete_files.len(), 2);

        // All should be position deletes
        for df in &delete_files {
            assert_eq!(df.content, DataContentType::PositionDeletes);
        }

        // Check total record count (1 + 1 + 1 = 3, but split across 2 files)
        let total_records: u64 = delete_files.iter().map(|f| f.record_count).sum();
        assert_eq!(total_records, 3);

        // Verify partition 1 file has 2 records (batch1 + batch3)
        // Verify partition 2 file has 1 record (batch2)
        let partition1_records: u64 = delete_files
            .iter()
            .filter(|f| f.record_count == 2)
            .map(|f| f.record_count)
            .sum();
        let partition2_records: u64 = delete_files
            .iter()
            .filter(|f| f.record_count == 1)
            .map(|f| f.record_count)
            .sum();
        assert_eq!(partition1_records, 2);
        assert_eq!(partition2_records, 1);

        Ok(())
    }
}
