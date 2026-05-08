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

use arrow_array::RecordBatch;

use crate::spec::{DataContentType, DataFile, PartitionKey};
use crate::writer::file_writer::FileWriterBuilder;
use crate::writer::file_writer::location_generator::{FileNameGenerator, LocationGenerator};
use crate::writer::file_writer::rolling_writer::{RollingFileWriter, RollingFileWriterBuilder};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Builder for `PositionDeleteFileWriter`.
#[derive(Debug)]
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
    pub fn new(inner: RollingFileWriterBuilder<B, L, F>) -> Self {
        Self { inner }
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

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        Ok(PositionDeleteFileWriter {
            inner: Some(self.inner.build()),
            partition_key,
        })
    }
}

/// A writer for position delete files within one spec/partition.
///
/// The input `RecordBatch` must use the position delete schema:
/// - `file_path` (Utf8): path of the data file containing the row
/// - `pos` (Int64): zero-based row position within that data file
#[derive(Debug)]
pub struct PositionDeleteFileWriter<
    B: FileWriterBuilder,
    L: LocationGenerator,
    F: FileNameGenerator,
> {
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
                            format!("Failed to build position delete file: {e}"),
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
mod test {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use parquet::arrow::PARQUET_FIELD_ID_META_KEY;
    use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use crate::Result;
    use crate::io::FileIO;
    use crate::metadata_columns::{
        RESERVED_FIELD_ID_DELETE_FILE_PATH, RESERVED_FIELD_ID_DELETE_FILE_POS,
    };
    use crate::spec::{
        DataFileFormat, Literal, NestedField, PartitionKey, PartitionSpec, PrimitiveType, Schema,
        Struct, Transform, Type, position_delete_schema,
    };
    use crate::writer::base_writer::position_delete_writer::PositionDeleteFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
    use crate::writer::{IcebergWriter, IcebergWriterBuilder};

    fn create_position_delete_arrow_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("file_path", DataType::Utf8, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                RESERVED_FIELD_ID_DELETE_FILE_PATH.to_string(),
            )])),
            Field::new("pos", DataType::Int64, false).with_metadata(HashMap::from([(
                PARQUET_FIELD_ID_META_KEY.to_string(),
                RESERVED_FIELD_ID_DELETE_FILE_POS.to_string(),
            )])),
        ])
    }

    #[tokio::test]
    async fn test_position_delete_writer() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let parquet_builder = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(position_delete_schema()),
        );
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io.clone(),
            location_gen,
            file_name_gen,
        );
        let mut position_delete_writer =
            PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
                .build(None)
                .await
                .unwrap();

        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec![
                "s3://bucket/data/file1.parquet",
                "s3://bucket/data/file2.parquet",
                "s3://bucket/data/file1.parquet",
            ])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ])?;
        position_delete_writer.write(batch).await?;

        let delete_files = position_delete_writer.close().await.unwrap();
        assert_eq!(delete_files.len(), 1);

        let delete_file = &delete_files[0];
        assert_eq!(delete_file.file_format, DataFileFormat::Parquet);
        assert_eq!(
            delete_file.content,
            crate::spec::DataContentType::PositionDeletes
        );
        assert_eq!(delete_file.record_count, 3);
        assert_eq!(delete_file.partition, Struct::empty());

        let input_file = file_io.new_input(delete_file.file_path.clone())?;
        let input_content = input_file.read().await?;
        let parquet_reader =
            ArrowReaderMetadata::load(&input_content, ArrowReaderOptions::default())
                .expect("Failed to load Parquet metadata");
        let field_ids: Vec<i32> = parquet_reader
            .parquet_schema()
            .columns()
            .iter()
            .map(|col| col.self_type().get_basic_info().id())
            .collect();

        assert_eq!(field_ids, vec![
            RESERVED_FIELD_ID_DELETE_FILE_PATH,
            RESERVED_FIELD_ID_DELETE_FILE_POS
        ]);

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_closed_error() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let parquet_builder = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(position_delete_schema()),
        );
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io,
            location_gen,
            file_name_gen,
        );
        let mut position_delete_writer =
            PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
                .build(None)
                .await
                .unwrap();

        let _ = position_delete_writer.close().await?;
        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec!["s3://bucket/data/file1.parquet"])),
            Arc::new(Int64Array::from(vec![42])),
        ])?;

        let result = position_delete_writer.write(batch).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().message().contains("closed"));

        Ok(())
    }

    #[tokio::test]
    async fn test_position_delete_writer_with_partition() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen = DefaultFileNameGenerator::new(
            "test_partitioned".to_string(),
            None,
            DataFileFormat::Parquet,
        );
        let parquet_builder = ParquetWriterBuilder::new(
            WriterProperties::builder().build(),
            Arc::new(position_delete_schema()),
        );
        let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            file_io,
            location_gen,
            file_name_gen,
        );

        let table_schema = Arc::new(
            Schema::builder()
                .with_schema_id(5)
                .with_fields(vec![
                    NestedField::required(5, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(6, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()?,
        );
        let partition_value = Struct::from_iter([Some(Literal::int(1))]);
        let partition_spec = PartitionSpec::builder(table_schema.clone())
            .with_spec_id(7)
            .add_partition_field("id", "id", Transform::Identity)?
            .build()?;
        let partition_key = PartitionKey::new(
            partition_spec.clone(),
            table_schema.clone(),
            partition_value,
        );

        let mut position_delete_writer =
            PositionDeleteFileWriterBuilder::new(rolling_writer_builder)
                .build(Some(partition_key))
                .await
                .unwrap();

        let arrow_schema = Arc::new(create_position_delete_arrow_schema());
        let batch = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(StringArray::from(vec![
                "s3://bucket/data/id=1/file.parquet",
            ])),
            Arc::new(Int64Array::from(vec![10])),
        ])?;
        position_delete_writer.write(batch).await?;

        let delete_files = position_delete_writer.close().await.unwrap();
        assert_eq!(delete_files.len(), 1);
        assert_eq!(
            delete_files[0].content,
            crate::spec::DataContentType::PositionDeletes
        );
        assert_eq!(
            delete_files[0].partition,
            Struct::from_iter([Some(Literal::int(1))])
        );
        assert_eq!(delete_files[0].partition_spec_id, 7);

        Ok(())
    }
}
