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
//! This crate provides a BigLake catalog that wraps the Iceberg REST catalog
//! with Google Cloud authentication and BigLake-specific configuration.

mod catalog;
mod token;
mod utils;

pub use catalog::{BigLakeCatalog, BigLakeCatalogBuilder, BigLakeCatalogConfig};
pub use token::GcpTokenProvider;

// Configuration property keys
/// GCP project ID (required)
pub const BIGLAKE_PROJECT_ID: &str = "biglake.project-id";
/// GCP region/location (required, e.g., "us-central1")
pub const BIGLAKE_LOCATION: &str = "biglake.location";
/// BigLake catalog name (required)
pub const BIGLAKE_CATALOG_NAME: &str = "biglake.catalog-name";
/// GCS warehouse path (required, must start with gs://)
pub const BIGLAKE_WAREHOUSE: &str = "warehouse";
/// Billing/quota project for X-Goog-User-Project header (optional, defaults to project-id)
pub const BIGLAKE_USER_PROJECT: &str = "biglake.user-project";
/// BigLake REST endpoint (optional, defaults to production)
pub const BIGLAKE_URI: &str = "biglake.uri";
/// Explicit GCP credentials JSON, base64 encoded (optional, uses ADC if not set)
pub const GCP_CREDENTIALS_JSON: &str = "gcp.credentials-json";

/// Default BigLake REST catalog endpoint
pub const DEFAULT_BIGLAKE_URI: &str = "https://biglake.googleapis.com/iceberg/v1/restcatalog";
