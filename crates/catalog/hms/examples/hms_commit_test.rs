// Proper Iceberg write and read example using HMS catalog
// Run with: cargo run -p iceberg-catalog-hms --example hms_commit_test
//
// Prerequisites:
//   cd crates/catalog/hms/testdata/hms_catalog
//   docker-compose up -d

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringArray};
use futures::TryStreamExt;
use iceberg::io::{
    S3_ACCESS_KEY_ID, S3_DISABLE_CONFIG_LOAD, S3_DISABLE_EC2_METADATA, S3_ENDPOINT,
    S3_PATH_STYLE_ACCESS, S3_REGION, S3_SECRET_ACCESS_KEY,
};
use iceberg::spec::{DataFileFormat, NestedField, PrimitiveType, Schema, Type};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use iceberg_catalog_hms::{
    HmsCatalogBuilder, HMS_CATALOG_PROP_THRIFT_TRANSPORT, HMS_CATALOG_PROP_URI,
    HMS_CATALOG_PROP_WAREHOUSE, THRIFT_TRANSPORT_BUFFERED,
};
use parquet::file::properties::WriterProperties;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure catalog - adjust these if your containers use different addresses
    let hms_addr = std::env::var("HMS_ADDR").unwrap_or_else(|_| "127.0.0.1:9083".to_string());
    let minio_addr =
        std::env::var("MINIO_ADDR").unwrap_or_else(|_| "http://127.0.0.1:9000".to_string());

    println!("=== Iceberg HMS Catalog Write/Read Example ===\n");
    println!("HMS Address: {}", hms_addr);
    println!("MinIO Address: {}", minio_addr);

    let props = HashMap::from([
        (HMS_CATALOG_PROP_URI.to_string(), hms_addr),
        (
            HMS_CATALOG_PROP_THRIFT_TRANSPORT.to_string(),
            THRIFT_TRANSPORT_BUFFERED.to_string(),
        ),
        (
            HMS_CATALOG_PROP_WAREHOUSE.to_string(),
            "s3a://warehouse/hive".to_string(),
        ),
        (S3_ENDPOINT.to_string(), minio_addr),
        (S3_ACCESS_KEY_ID.to_string(), "admin".to_string()),
        (S3_SECRET_ACCESS_KEY.to_string(), "password".to_string()),
        (S3_REGION.to_string(), "us-east-1".to_string()),
        // MinIO requires path-style access and disabling AWS SDK credential lookups
        (S3_PATH_STYLE_ACCESS.to_string(), "true".to_string()),
        (S3_DISABLE_EC2_METADATA.to_string(), "true".to_string()),
        (S3_DISABLE_CONFIG_LOAD.to_string(), "true".to_string()),
    ]);

    // =========================================================================
    // Step 1: Create HMS Catalog
    // =========================================================================
    println!("\n[Step 1] Creating HMS catalog...");
    let catalog = HmsCatalogBuilder::default()
        .load("hms_test", props)
        .await?;
    println!("  Catalog created successfully");

    // =========================================================================
    // Step 2: Create Namespace (Database)
    // =========================================================================
    let ns_name = format!(
        "demo_db_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    );
    let namespace = NamespaceIdent::new(ns_name.clone());

    println!("\n[Step 2] Creating namespace '{}'...", ns_name);
    catalog
        .create_namespace(&namespace, HashMap::new())
        .await?;
    println!("  Namespace created");

    // =========================================================================
    // Step 3: Create Iceberg Table with Schema
    // =========================================================================
    let schema = Schema::builder()
        .with_schema_id(0)
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()?;

    let table_creation = TableCreation::builder()
        .name("users".to_string())
        .schema(schema)
        .location(format!("s3a://warehouse/hive/{}/users", ns_name))
        .build();

    println!("\n[Step 3] Creating table 'users'...");
    let table = catalog.create_table(&namespace, table_creation).await?;
    println!("  Table created");
    println!("  Location: {}", table.metadata().location());
    println!(
        "  Metadata: {:?}",
        table.metadata_location().unwrap_or("N/A")
    );

    // =========================================================================
    // Step 4: Write First Batch of Data
    // =========================================================================
    println!("\n[Step 4] Writing first batch of data...");

    let arrow_schema: Arc<arrow_schema::Schema> =
        Arc::new(table.metadata().current_schema().as_ref().try_into()?);

    // Create first batch: 3 users
    let batch1 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])) as ArrayRef,
        ],
    )?;

    println!("  Data to write:");
    println!("    id | name");
    println!("    ---|-------");
    println!("     1 | Alice");
    println!("     2 | Bob");
    println!("     3 | Charlie");

    let data_files = write_data(&table, batch1.clone(), "batch1").await?;
    println!("  Wrote {} data file(s)", data_files.len());
    for df in &data_files {
        println!("    - {}", df.file_path());
    }

    // =========================================================================
    // Step 5: Commit First Transaction
    // =========================================================================
    println!("\n[Step 5] Committing first transaction...");

    let tx = Transaction::new(&table);
    let append_action = tx.fast_append().add_data_files(data_files);
    let tx = append_action.apply(tx)?;
    let table = tx.commit(&catalog).await?;

    let snapshot = table.metadata().current_snapshot().unwrap();
    println!("  Committed snapshot: {}", snapshot.snapshot_id());
    println!("  Operation: {:?}", snapshot.summary().operation);

    // =========================================================================
    // Step 6: Read Data Back (Table Scan)
    // =========================================================================
    println!("\n[Step 6] Reading data from table...");

    let scan = table.scan().select_all().build()?;

    let stream = scan.to_arrow().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;

    println!("  Read {} batch(es)", batches.len());
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    println!("  Total rows: {}", total_rows);

    // Print the data
    println!("\n  Data read from table:");
    println!("    id | name");
    println!("    ---|-------");
    for batch in &batches {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            println!("    {:>2} | {}", ids.value(i), names.value(i));
        }
    }

    // Verify data matches
    assert_eq!(total_rows, 3);
    assert_eq!(batches[0], batch1);
    println!("  Data verification passed!");

    // =========================================================================
    // Step 7: Write Second Batch of Data
    // =========================================================================
    println!("\n[Step 7] Writing second batch of data...");

    let batch2 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])) as ArrayRef,
            Arc::new(StringArray::from(vec!["Diana", "Eve"])) as ArrayRef,
        ],
    )?;

    println!("  Data to write:");
    println!("    id | name");
    println!("    ---|-------");
    println!("     4 | Diana");
    println!("     5 | Eve");

    let data_files2 = write_data(&table, batch2, "batch2").await?;
    println!("  Wrote {} data file(s)", data_files2.len());

    // =========================================================================
    // Step 8: Commit Second Transaction
    // =========================================================================
    println!("\n[Step 8] Committing second transaction...");

    let tx2 = Transaction::new(&table);
    let append_action2 = tx2.fast_append().add_data_files(data_files2);
    let tx2 = append_action2.apply(tx2)?;
    let table = tx2.commit(&catalog).await?;

    let snapshot2 = table.metadata().current_snapshot().unwrap();
    println!("  Committed snapshot: {}", snapshot2.snapshot_id());
    println!(
        "  Total snapshots: {}",
        table.metadata().snapshots().count()
    );

    // =========================================================================
    // Step 9: Read All Data After Second Commit
    // =========================================================================
    println!("\n[Step 9] Reading all data after second commit...");

    let scan2 = table.scan().select_all().build()?;
    let stream2 = scan2.to_arrow().await?;
    let batches2: Vec<RecordBatch> = stream2.try_collect().await?;

    let total_rows2: usize = batches2.iter().map(|b| b.num_rows()).sum();
    println!("  Total rows: {}", total_rows2);

    println!("\n  All data in table:");
    println!("    id | name");
    println!("    ---|-------");
    for batch in &batches2 {
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let names = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            println!("    {:>2} | {}", ids.value(i), names.value(i));
        }
    }

    assert_eq!(total_rows2, 5);
    println!("  Data verification passed!");

    // =========================================================================
    // Step 10: Reload Table from Catalog and Verify
    // =========================================================================
    println!("\n[Step 10] Reloading table from catalog...");

    let reloaded_table = catalog.load_table(table.identifier()).await?;
    println!(
        "  Snapshots in reloaded table: {}",
        reloaded_table.metadata().snapshots().count()
    );

    let scan3 = reloaded_table.scan().select_all().build()?;
    let stream3 = scan3.to_arrow().await?;
    let batches3: Vec<RecordBatch> = stream3.try_collect().await?;
    let total_rows3: usize = batches3.iter().map(|b| b.num_rows()).sum();
    println!("  Rows after reload: {}", total_rows3);

    assert_eq!(total_rows3, 5);
    println!("  Persistence verification passed!");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n========================================");
    println!("All tests passed!");
    println!("========================================");
    println!("\nSummary:");
    println!("  - Created namespace: {}", ns_name);
    println!("  - Created table: users");
    println!("  - Wrote 2 batches of data (5 total rows)");
    println!("  - Created 2 snapshots");
    println!("  - Successfully read data back via table scan");
    println!("  - Verified data persistence across catalog reload");

    Ok(())
}

/// Helper function to write a RecordBatch to the table and return DataFiles
async fn write_data(
    table: &iceberg::table::Table,
    batch: RecordBatch,
    prefix: &str,
) -> Result<Vec<iceberg::spec::DataFile>, Box<dyn std::error::Error>> {
    let location_generator = DefaultLocationGenerator::new(table.metadata().clone())?;
    let file_name_generator =
        DefaultFileNameGenerator::new(prefix.to_string(), None, DataFileFormat::Parquet);

    let parquet_writer_builder = ParquetWriterBuilder::new(
        WriterProperties::default(),
        table.metadata().current_schema().clone(),
    );

    let rolling_writer_builder = RollingFileWriterBuilder::new_with_default_file_size(
        parquet_writer_builder,
        table.file_io().clone(),
        location_generator,
        file_name_generator,
    );

    let data_file_writer_builder = DataFileWriterBuilder::new(rolling_writer_builder, None);
    let mut writer = data_file_writer_builder.build().await?;

    writer.write(batch).await?;
    let data_files = writer.close().await?;

    Ok(data_files)
}
