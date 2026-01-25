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

//! Apache Iceberg BigLake Catalog implementation using native gRPC.
//!
//! This crate provides a BigLake catalog that uses the `google-cloud-biglake-v1`
//! gRPC client directly for all catalog operations. It follows the same pattern
//! as the Glue catalog - a native cloud catalog implementation that:
//!
//! - Uses the BigLake gRPC client (`IcebergCatalogService`) for all catalog operations
//! - Stores table metadata on GCS via FileIO (not through the catalog API)
//! - Uses vended credentials from `load_iceberg_table_credentials` for storage access
//!
//! # Authentication
//!
//! Authentication is handled automatically by the `google-cloud-biglake-v1` SDK using
//! Google Cloud Application Default Credentials (ADC). The SDK supports:
//!
//! - Environment variables (GOOGLE_APPLICATION_CREDENTIALS)
//! - gcloud CLI credentials
//! - GCE/GKE metadata server
//! - Workload Identity
//!
//! # Example
//!
//! ```ignore
//! use std::collections::HashMap;
//! use iceberg::CatalogBuilder;
//! use iceberg_catalog_biglake::{
//!     BigLakeCatalogBuilder,
//!     BIGLAKE_PROJECT_ID, BIGLAKE_LOCATION, BIGLAKE_CATALOG_ID, BIGLAKE_WAREHOUSE,
//! };
//!
//! let catalog = BigLakeCatalogBuilder::default()
//!     .load("my-catalog", HashMap::from([
//!         (BIGLAKE_PROJECT_ID.to_string(), "my-project".to_string()),
//!         (BIGLAKE_LOCATION.to_string(), "us-central1".to_string()),
//!         (BIGLAKE_CATALOG_ID.to_string(), "my-catalog".to_string()),
//!         (BIGLAKE_WAREHOUSE.to_string(), "gs://my-bucket/warehouse".to_string()),
//!     ]))
//!     .await?;
//! ```

mod catalog;
mod error;

pub use catalog::{BigLakeCatalog, BigLakeCatalogBuilder, BigLakeCatalogConfig};

// Configuration property keys
/// GCP project ID (required)
pub const BIGLAKE_PROJECT_ID: &str = "biglake.project-id";
/// GCP region/location (required, e.g., "us-central1")
pub const BIGLAKE_LOCATION: &str = "biglake.location";
/// BigLake catalog ID (required)
pub const BIGLAKE_CATALOG_ID: &str = "biglake.catalog-id";
/// GCS warehouse path (required, must start with gs://)
pub const BIGLAKE_WAREHOUSE: &str = "warehouse";
/// Billing/quota project for X-Goog-User-Project header (optional, defaults to project-id)
pub const BIGLAKE_USER_PROJECT: &str = "biglake.user-project";

// GCS configuration keys (for FileIO)
/// Google Cloud Storage token for vended credentials
pub const GCS_TOKEN: &str = "gcs.oauth2.token";
