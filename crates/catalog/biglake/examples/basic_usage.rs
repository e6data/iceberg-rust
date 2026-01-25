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

//! Basic usage example for BigLake Catalog

use std::collections::HashMap;

use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use iceberg::{Catalog, NamespaceIdent, TableCreation};
use iceberg_catalog_biglake::{
    BigLakeCatalogBuilder, BIGLAKE_CATALOG_ID, BIGLAKE_LOCATION, BIGLAKE_PROJECT_ID,
    BIGLAKE_WAREHOUSE,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ========================================
    // 1. CREATE CATALOG
    // ========================================

    println!("Creating BigLake catalog...");

    let catalog = BigLakeCatalogBuilder::default()
        .load(
            "my_catalog",
            HashMap::from([
                // Required: GCP project ID
                (BIGLAKE_PROJECT_ID.to_string(), "my-gcp-project".to_string()),
                // Required: GCP location/region
                (BIGLAKE_LOCATION.to_string(), "us-central1".to_string()),
                // Required: BigLake catalog ID
                (BIGLAKE_CATALOG_ID.to_string(), "my_biglake_catalog".to_string()),
                // Required: GCS warehouse path
                (BIGLAKE_WAREHOUSE.to_string(), "gs://my-bucket/warehouse".to_string()),
            ]),
        )
        .await?;

    println!("✓ Catalog created");

    // ========================================
    // 2. NAMESPACE OPERATIONS
    // ========================================

    // List all namespaces
    println!("\nListing namespaces...");
    let namespaces = catalog.list_namespaces(None).await?;
    println!("  Found {} namespaces", namespaces.len());

    // Create a namespace
    let namespace = NamespaceIdent::new("my_database".to_string());
    println!("\nCreating namespace '{}'...", namespace);

    let properties = HashMap::from([
        ("owner".to_string(), "data_team".to_string()),
        ("description".to_string(), "My database".to_string()),
    ]);

    catalog.create_namespace(&namespace, properties).await?;
    println!("✓ Namespace created");

    // Check if namespace exists
    let exists = catalog.namespace_exists(&namespace).await?;
    println!("  Namespace exists: {}", exists);

    // Get namespace properties
    let ns_info = catalog.get_namespace(&namespace).await?;
    println!("  Properties: {:?}", ns_info.properties());

    // ========================================
    // 3. TABLE OPERATIONS
    // ========================================

    // Define a schema
    let schema = Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
            NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
        ])
        .build()?;

    // Create a table
    println!("\nCreating table 'users'...");

    let table_creation = TableCreation::builder()
        .name("users".to_string())
        .schema(schema)
        .properties(HashMap::from([
            ("owner".to_string(), "data_team".to_string()),
        ]))
        .build();

    let table = catalog.create_table(&namespace, table_creation).await?;
    println!("✓ Table created");
    println!("  Location: {}", table.metadata().location());
    println!("  Schema: {:?}", table.metadata().current_schema());

    // List tables in namespace
    println!("\nListing tables in namespace...");
    let tables = catalog.list_tables(&namespace).await?;
    for table_ident in &tables {
        println!("  - {}", table_ident.name());
    }

    // Load a table
    println!("\nLoading table...");
    let loaded_table = catalog.load_table(tables.first().unwrap()).await?;
    println!("✓ Table loaded");
    println!("  Metadata location: {}", loaded_table.metadata_location_result()?);

    // Check if table exists
    let table_exists = catalog.table_exists(tables.first().unwrap()).await?;
    println!("  Table exists: {}", table_exists);

    // ========================================
    // 4. CLEANUP (optional)
    // ========================================

    // Drop table
    println!("\nDropping table...");
    catalog.drop_table(tables.first().unwrap()).await?;
    println!("✓ Table dropped");

    // Drop namespace
    println!("\nDropping namespace...");
    catalog.drop_namespace(&namespace).await?;
    println!("✓ Namespace dropped");

    Ok(())
}
