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

//! BigLake catalog implementation.

use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, Namespace, NamespaceIdent, Result, TableCommit,
    TableCreation, TableIdent,
};
use iceberg_catalog_rest::{RestCatalog, RestCatalogBuilder};

use crate::token::GcpTokenProvider;
use crate::utils::{build_catalog_prefix, validate_namespace};
use crate::{
    BIGLAKE_CATALOG_NAME, BIGLAKE_LOCATION, BIGLAKE_PROJECT_ID, BIGLAKE_URI, BIGLAKE_USER_PROJECT,
    BIGLAKE_WAREHOUSE, DEFAULT_BIGLAKE_URI, GCP_CREDENTIALS_JSON,
};

/// Configuration for BigLake catalog.
#[derive(Debug, Clone)]
pub struct BigLakeCatalogConfig {
    /// GCP project ID
    pub project_id: String,
    /// GCP region/location (e.g., "us-central1")
    pub location: String,
    /// BigLake catalog name
    pub catalog_name: String,
    /// GCS warehouse path (must start with gs://)
    pub warehouse: String,
    /// Billing/quota project for X-Goog-User-Project header
    pub user_project: Option<String>,
    /// BigLake REST endpoint
    pub uri: String,
    /// Original properties passed to the builder
    pub props: HashMap<String, String>,
}

/// Builder for BigLake catalog.
#[derive(Debug, Default)]
pub struct BigLakeCatalogBuilder {
    config: Option<BigLakeCatalogConfig>,
}

impl CatalogBuilder for BigLakeCatalogBuilder {
    type C = BigLakeCatalog;

    fn load(
        self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl Future<Output = Result<Self::C>> + Send {
        let name = name.into();
        async move {
            // Extract required properties
            let project_id = props.get(BIGLAKE_PROJECT_ID).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Missing required property: {BIGLAKE_PROJECT_ID}"),
                )
            })?;

            let location = props.get(BIGLAKE_LOCATION).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Missing required property: {BIGLAKE_LOCATION}"),
                )
            })?;

            let catalog_name = props.get(BIGLAKE_CATALOG_NAME).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Missing required property: {BIGLAKE_CATALOG_NAME}"),
                )
            })?;

            let warehouse = props.get(BIGLAKE_WAREHOUSE).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Missing required property: {BIGLAKE_WAREHOUSE}"),
                )
            })?;

            // Validate warehouse is GCS path
            if !warehouse.starts_with("gs://") {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("Warehouse must be a GCS path (gs://...), got: {warehouse}"),
                ));
            }

            let user_project = props.get(BIGLAKE_USER_PROJECT).cloned();
            let uri = props
                .get(BIGLAKE_URI)
                .cloned()
                .unwrap_or_else(|| DEFAULT_BIGLAKE_URI.to_string());

            let credentials_json = props.get(GCP_CREDENTIALS_JSON).map(|s| s.as_str());

            // Create GCP token provider
            let token_provider =
                Arc::new(GcpTokenProvider::with_credentials(credentials_json).await?);

            // Build REST catalog config with BigLake settings
            let prefix = build_catalog_prefix(&project_id, &location, &catalog_name);
            let billing_project = user_project.as_ref().unwrap_or(&project_id);

            let mut rest_props = props.clone();
            rest_props.insert("uri".to_string(), uri.clone());
            rest_props.insert("prefix".to_string(), prefix);
            rest_props.insert("warehouse".to_string(), warehouse.clone());
            rest_props.insert(
                "header.x-goog-user-project".to_string(),
                billing_project.clone(),
            );
            rest_props.insert(
                "header.X-Iceberg-Access-Delegation".to_string(),
                "vended-credentials".to_string(),
            );
            // Set GCS user project for storage operations
            rest_props.insert("gcs.user-project".to_string(), billing_project.clone());

            // Build inner REST catalog with token provider
            let inner = RestCatalogBuilder::default()
                .with_token_provider(token_provider.clone())
                .load(name, rest_props)
                .await?;

            Ok(BigLakeCatalog {
                config: BigLakeCatalogConfig {
                    project_id,
                    location,
                    catalog_name,
                    warehouse,
                    user_project,
                    uri,
                    props,
                },
                inner,
                _token_provider: token_provider,
            })
        }
    }
}

/// BigLake catalog implementation.
///
/// Wraps the Iceberg REST catalog with Google Cloud authentication and
/// BigLake-specific configuration.
#[derive(Debug)]
pub struct BigLakeCatalog {
    config: BigLakeCatalogConfig,
    inner: RestCatalog,
    // Keep token provider alive for the lifetime of the catalog
    _token_provider: Arc<GcpTokenProvider>,
}

impl BigLakeCatalog {
    /// Returns the catalog configuration.
    pub fn config(&self) -> &BigLakeCatalogConfig {
        &self.config
    }
}

#[async_trait]
impl Catalog for BigLakeCatalog {
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        if let Some(parent) = parent {
            validate_namespace(parent)?;
        }
        self.inner.list_namespaces(parent).await
    }

    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        validate_namespace(namespace)?;
        self.inner.create_namespace(namespace, properties).await
    }

    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        validate_namespace(namespace)?;
        self.inner.get_namespace(namespace).await
    }

    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        validate_namespace(namespace)?;
        self.inner.namespace_exists(namespace).await
    }

    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        validate_namespace(namespace)?;
        self.inner.update_namespace(namespace, properties).await
    }

    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        validate_namespace(namespace)?;
        self.inner.drop_namespace(namespace).await
    }

    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        validate_namespace(namespace)?;
        self.inner.list_tables(namespace).await
    }

    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<Table> {
        validate_namespace(namespace)?;
        self.inner.create_table(namespace, creation).await
    }

    async fn load_table(&self, table: &TableIdent) -> Result<Table> {
        validate_namespace(&table.namespace)?;
        self.inner.load_table(table).await
    }

    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        validate_namespace(&table.namespace)?;
        self.inner.drop_table(table).await
    }

    async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
        validate_namespace(&table.namespace)?;
        self.inner.table_exists(table).await
    }

    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        validate_namespace(&src.namespace)?;
        validate_namespace(&dest.namespace)?;
        self.inner.rename_table(src, dest).await
    }

    async fn update_table(&self, commit: TableCommit) -> Result<Table> {
        validate_namespace(&commit.identifier().namespace)?;
        self.inner.update_table(commit).await
    }

    async fn register_table(
        &self,
        identifier: &TableIdent,
        metadata_location: String,
    ) -> Result<Table> {
        validate_namespace(&identifier.namespace)?;
        self.inner
            .register_table(identifier, metadata_location)
            .await
    }
}
