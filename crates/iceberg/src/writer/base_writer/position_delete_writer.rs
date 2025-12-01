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

use arrow_array::RecordBatch;

use crate::spec::{DataContentType, DataFile, PartitionKey};
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Builder for `PositionDeleteFileWriter`.
#[derive(Clone, Debug)]
pub struct PositionDeleteFileWriterBuilder<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator>
{
    inner: RollingFileWriterBuilder<B, L, F>,
    partition_key: Option<PartitionKey>,
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
    /// * `partition_key` - Optional partition key for partitioned tables
    pub fn new(
        inner_builder: RollingFileWriterBuilder<B, L, F>,
        partition_key: Option<PartitionKey>,
    ) -> Self {
        Self {
            inner: inner_builder,
            partition_key,
        }
    }
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriterBuilder for PositionDeleteFileWriterBuilder<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    type R = PositionDeleteFileWriter<B, L, F>;

    async fn build(self) -> Result<Self::R> {
        Ok(PositionDeleteFileWriter {
            inner: Some(self.inner.clone().build()),
            partition_key: self.partition_key,
        })
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
/// # Example
/// ```rust,ignore
/// use iceberg::writer::base_writer::position_delete_writer::PositionDeleteFileWriterBuilder;
///
/// // Create the writer
/// let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder, None)
///     .build()
///     .await?;
///
/// // Write position deletes
/// let batch = RecordBatch::try_new(
///     position_delete_schema,
///     vec![
///         Arc::new(StringArray::from(vec!["s3://bucket/data/file.parquet"])),
///         Arc::new(Int64Array::from(vec![42])),
///     ],
/// )?;
/// writer.write(batch).await?;
///
/// // Close and get the delete files
/// let delete_files = writer.close().await?;
/// ```
#[derive(Debug)]
pub struct PositionDeleteFileWriter<B: FileWriterBuilder, L: LocationGenerator, F: FileNameGenerator>
{
    inner: Option<RollingFileWriter<B, L, F>>,
    partition_key: Option<PartitionKey>,
}

#[async_trait::async_trait]
impl<B, L, F> IcebergWriter for PositionDeleteFileWriter<B, L, F>
where
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
{
    /// Write a batch of position deletes.
    ///
    /// The batch must have the position delete schema with columns:
    /// - `file_path` (Utf8)
    /// - `pos` (Int64)
    async fn write(&mut self, batch: RecordBatch) -> Result<()> {
        if let Some(writer) = self.inner.as_mut() {
            writer.write(&self.partition_key, &batch).await
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete writer has been closed.",
            ))
        }
    }

    /// Close the writer and return the resulting delete files.
    ///
    /// Each returned `DataFile` will have:
    /// - `content` set to `DataContentType::PositionDeletes`
    /// - Partition info set if a partition key was provided
    async fn close(&mut self) -> Result<Vec<DataFile>> {
        if let Some(writer) = self.inner.take() {
            writer
                .close()
                .await?
                .into_iter()
                .map(|mut res| {
                    res.content(DataContentType::PositionDeletes);
                    if let Some(pk) = self.partition_key.as_ref() {
                        res.partition(pk.data().clone());
                        res.partition_spec_id(pk.spec().spec_id());
                    }
                    res.build().map_err(|e| {
                        Error::new(
                            ErrorKind::DataInvalid,
                            format!("Failed to build position delete file: {}", e),
                        )
                    })
                })
                .collect()
        } else {
            Err(Error::new(
                ErrorKind::Unexpected,
                "Position delete writer has been closed.",
            ))
        }
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
        DataContentType, DataFileFormat, NestedField, PrimitiveType, Schema, Type,
        DELETE_FILE_PATH_FIELD_ID, DELETE_FILE_POS_FIELD_ID,
    };
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;

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

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder, None)
            .build()
            .await?;

        // Create a batch of position deletes
        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(StringArray::from(vec![
                    "s3://bucket/data/file1.parquet",
                    "s3://bucket/data/file2.parquet",
                    "s3://bucket/data/file1.parquet",
                ])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )?;

        writer.write(batch).await?;
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

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder, None)
            .build()
            .await?;

        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(StringArray::from(vec!["s3://bucket/data/file1.parquet"])),
                Arc::new(Int64Array::from(vec![42])),
            ],
        )?;

        writer.write(batch).await?;
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
        assert_eq!(
            field_ids,
            vec![DELETE_FILE_PATH_FIELD_ID, DELETE_FILE_POS_FIELD_ID]
        );

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

        let mut writer = PositionDeleteFileWriterBuilder::new(rolling_builder, None)
            .build()
            .await?;

        // Close the writer
        let _ = writer.close().await?;

        // Try to write after close - should fail
        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(StringArray::from(vec!["s3://bucket/data/file1.parquet"])),
                Arc::new(Int64Array::from(vec![42])),
            ],
        )?;

        let result = writer.write(batch).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().message().contains("closed"));

        Ok(())
    }
}
