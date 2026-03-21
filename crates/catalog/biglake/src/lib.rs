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

//! Apache Iceberg BigLake Catalog implementation.
//!
//! This crate provides a BigLake catalog that wraps the REST catalog with
//! automatic GCP token management. It handles:
//!
//! - Fetching GCP tokens from the metadata server (GKE Workload Identity)
//! - Automatic token refresh before expiry
//! - Retry on auth errors (401/403)
//! - Required BigLake headers (x-goog-user-project, X-Iceberg-Access-Delegation)
//!
//! # Authentication
//!
//! The catalog fetches tokens from the GCP metadata server, which works on:
//! - GKE pods (with Workload Identity)
//! - GCE VMs
//! - Cloud Run
//! - Cloud Functions
//!
//! For local development, you can set `GCE_METADATA_HOST` environment variable
//! to point to a local metadata server emulator.
//!
//! # Storage Access
//!
//! GCS access is handled automatically by OpenDAL using Application Default
//! Credentials (ADC). Enable the `storage-gcs` feature on the `iceberg` crate.
//!
//! # Example
//!
//! ```rust,no_run
//! use std::collections::HashMap;
//!
//! use iceberg::{Catalog, CatalogBuilder};
//! use iceberg_catalog_biglake::{
//!     BIGLAKE_CATALOG_ID, BIGLAKE_PROJECT_ID, BIGLAKE_WAREHOUSE, BigLakeCatalogBuilder,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let catalog = BigLakeCatalogBuilder::default()
//!         .load(
//!             "biglake",
//!             HashMap::from([
//!                 (BIGLAKE_PROJECT_ID.to_string(), "my-project".to_string()),
//!                 (BIGLAKE_CATALOG_ID.to_string(), "my-catalog".to_string()),
//!                 (
//!                     BIGLAKE_WAREHOUSE.to_string(),
//!                     "gs://my-bucket/warehouse".to_string(),
//!                 ),
//!             ]),
//!         )
//!         .await?;
//!
//!     // Use like any other Catalog
//!     let namespaces = catalog.list_namespaces(None).await?;
//!     println!("Namespaces: {:?}", namespaces);
//!
//!     Ok(())
//! }
//! ```

mod catalog;
mod token;

pub use catalog::{BigLakeCatalog, BigLakeCatalogBuilder, BigLakeCatalogConfig};

/// Property key for GCP project ID (required)
pub const BIGLAKE_PROJECT_ID: &str = "biglake.project-id";

/// Property key for BigLake catalog ID (required)
pub const BIGLAKE_CATALOG_ID: &str = "biglake.catalog-id";

/// Property key for warehouse location (required, must start with gs://)
pub const BIGLAKE_WAREHOUSE: &str = "warehouse";

/// Property key for service account name for metadata server (default: "default")
pub const BIGLAKE_SERVICE_ACCOUNT: &str = "biglake.service-account";

/// Property key for BigLake REST endpoint (optional)
pub const BIGLAKE_URI: &str = "biglake.uri";

/// Default BigLake REST endpoint
pub const DEFAULT_BIGLAKE_URI: &str = "https://biglake.googleapis.com/iceberg/v1/restcatalog";
