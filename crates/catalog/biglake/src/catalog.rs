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

//! BigLake Catalog implementation wrapping REST catalog with GCP auth.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, Namespace, NamespaceIdent, Result, TableCommit,
    TableCreation, TableIdent,
};
use iceberg_catalog_rest::RestCatalogBuilder;
use tokio::sync::RwLock;

use crate::token::{fetch_gcp_token, Token};
use crate::{
    BIGLAKE_CATALOG_ID, BIGLAKE_PROJECT_ID, BIGLAKE_SERVICE_ACCOUNT, BIGLAKE_URI, BIGLAKE_WAREHOUSE,
    DEFAULT_BIGLAKE_URI,
};

/// BigLake Catalog configuration.
#[derive(Debug, Clone)]
pub struct BigLakeCatalogConfig {
    /// Catalog name
    pub name: String,
    /// GCP project ID
    pub project_id: String,
    /// BigLake catalog ID
    pub catalog_id: String,
    /// GCS warehouse path (gs://...)
    pub warehouse: String,
    /// BigLake REST endpoint (optional)
    pub uri: String,
    /// Service account name for metadata server (default: "default")
    pub service_account: String,
    /// Additional properties passed to REST catalog
    pub props: HashMap<String, String>,
}

impl Default for BigLakeCatalogConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            project_id: String::new(),
            catalog_id: String::new(),
            warehouse: String::new(),
            uri: DEFAULT_BIGLAKE_URI.to_string(),
            service_account: "default".to_string(),
            props: HashMap::new(),
        }
    }
}

/// Builder for [`BigLakeCatalog`].
#[derive(Debug, Default)]
pub struct BigLakeCatalogBuilder {
    config: BigLakeCatalogConfig,
}

impl CatalogBuilder for BigLakeCatalogBuilder {
    type C = BigLakeCatalog;

    fn load(
        mut self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl std::future::Future<Output = Result<Self::C>> + Send {
        self.config.name = name.into();

        // Extract known properties
        if let Some(v) = props.get(BIGLAKE_PROJECT_ID) {
            self.config.project_id = v.clone();
        }
        if let Some(v) = props.get(BIGLAKE_CATALOG_ID) {
            self.config.catalog_id = v.clone();
        }
        if let Some(v) = props.get(BIGLAKE_WAREHOUSE) {
            self.config.warehouse = v.clone();
        }
        if let Some(v) = props.get(BIGLAKE_URI) {
            self.config.uri = v.clone();
        }
        if let Some(v) = props.get(BIGLAKE_SERVICE_ACCOUNT) {
            self.config.service_account = v.clone();
        }

        // Collect remaining properties (for REST catalog passthrough)
        self.config.props = props
            .into_iter()
            .filter(|(k, _)| {
                k != BIGLAKE_PROJECT_ID
                    && k != BIGLAKE_CATALOG_ID
                    && k != BIGLAKE_WAREHOUSE
                    && k != BIGLAKE_URI
                    && k != BIGLAKE_SERVICE_ACCOUNT
            })
            .collect();

        async move {
            // Validate required fields
            if self.config.project_id.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} is required", BIGLAKE_PROJECT_ID),
                ));
            }
            if self.config.catalog_id.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} is required", BIGLAKE_CATALOG_ID),
                ));
            }
            if self.config.warehouse.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} is required", BIGLAKE_WAREHOUSE),
                ));
            }
            if !self.config.warehouse.starts_with("gs://") {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} must start with gs://", BIGLAKE_WAREHOUSE),
                ));
            }

            BigLakeCatalog::new(self.config).await
        }
    }
}

/// Internal state holding the REST catalog and token.
struct CatalogState {
    catalog: iceberg_catalog_rest::RestCatalog,
    token: Token,
}

/// BigLake Catalog - wraps REST catalog with GCP token management.
///
/// This catalog automatically:
/// - Fetches GCP tokens from the metadata server
/// - Refreshes tokens before they expire
/// - Retries operations on auth errors (401/403)
/// - Adds required BigLake headers (x-goog-user-project, X-Iceberg-Access-Delegation)
pub struct BigLakeCatalog {
    config: BigLakeCatalogConfig,
    state: Arc<RwLock<CatalogState>>,
}

impl Debug for BigLakeCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BigLakeCatalog")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl BigLakeCatalog {
    /// Create a new BigLake catalog.
    pub async fn new(config: BigLakeCatalogConfig) -> Result<Self> {
        // Fetch initial token
        let token = fetch_gcp_token(&config.service_account).await?;

        // Build initial REST catalog
        let catalog = Self::build_rest_catalog(&config, &token.access_token).await?;

        tracing::info!(
            project_id = %config.project_id,
            catalog_id = %config.catalog_id,
            uri = %config.uri,
            "Created BigLake catalog"
        );

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(CatalogState { catalog, token })),
        })
    }

    /// Build a REST catalog with the given token.
    async fn build_rest_catalog(
        config: &BigLakeCatalogConfig,
        token: &str,
    ) -> Result<iceberg_catalog_rest::RestCatalog> {
        let mut props = config.props.clone();

        // Set the URI
        props.insert("uri".to_string(), config.uri.clone());

        // Set warehouse
        props.insert("warehouse".to_string(), config.warehouse.clone());

        // Add token for authentication
        props.insert("token".to_string(), token.to_string());

        // Required headers for BigLake REST API
        props.insert(
            "header.x-goog-user-project".to_string(),
            config.project_id.clone(),
        );
        props.insert(
            "header.X-Iceberg-Access-Delegation".to_string(),
            "vended-credentials".to_string(),
        );

        // REST catalog prefix for BigLake: projects/{project}/catalogs/{catalog}
        props.insert(
            "prefix".to_string(),
            format!(
                "projects/{}/catalogs/{}",
                config.project_id, config.catalog_id
            ),
        );

        RestCatalogBuilder::default()
            .load(&config.name, props)
            .await
    }

    /// Ensure the catalog token is valid, refreshing if needed.
    async fn ensure_valid_token(&self) -> Result<()> {
        let needs_refresh = {
            let state = self.state.read().await;
            state.token.is_expiring()
        };

        if needs_refresh {
            self.refresh_token().await?;
        }

        Ok(())
    }

    /// Force refresh the token and rebuild the REST catalog.
    async fn refresh_token(&self) -> Result<()> {
        tracing::info!(
            service_account = %self.config.service_account,
            "Refreshing GCP access token"
        );

        let new_token = fetch_gcp_token(&self.config.service_account).await?;
        let new_catalog = Self::build_rest_catalog(&self.config, &new_token.access_token).await?;

        let mut state = self.state.write().await;
        state.token = new_token;
        state.catalog = new_catalog;

        tracing::info!("Successfully refreshed BigLake catalog token");
        Ok(())
    }

    /// Force refresh the token.
    ///
    /// Call this before `create_table` or `update_table` if you want to ensure
    /// a fresh token, since those operations cannot be automatically retried
    /// on auth errors (the input types don't implement Clone).
    ///
    /// For most operations this is not needed - the catalog automatically
    /// refreshes tokens before they expire and retries on auth errors.
    pub async fn force_refresh_token(&self) -> Result<()> {
        self.refresh_token().await
    }

    /// Check if an error is an authentication error.
    fn is_auth_error(err: &Error) -> bool {
        let msg = err.to_string().to_lowercase();
        msg.contains("401")
            || msg.contains("403")
            || msg.contains("unauthorized")
            || msg.contains("forbidden")
            || msg.contains("permission denied")
            || msg.contains("unauthenticated")
    }
}

#[async_trait]
impl Catalog for BigLakeCatalog {
    /// List namespaces.
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.list_namespaces(parent).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.list_namespaces(parent).await
            }
            Err(e) => Err(e),
        }
    }

    /// Create a namespace.
    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.create_namespace(namespace, properties.clone()).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.create_namespace(namespace, properties).await
            }
            Err(e) => Err(e),
        }
    }

    /// Get a namespace.
    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.get_namespace(namespace).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.get_namespace(namespace).await
            }
            Err(e) => Err(e),
        }
    }

    /// Check if a namespace exists.
    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.namespace_exists(namespace).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.namespace_exists(namespace).await
            }
            Err(e) => Err(e),
        }
    }

    /// Update namespace properties.
    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.update_namespace(namespace, properties.clone()).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.update_namespace(namespace, properties).await
            }
            Err(e) => Err(e),
        }
    }

    /// Drop a namespace.
    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.drop_namespace(namespace).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.drop_namespace(namespace).await
            }
            Err(e) => Err(e),
        }
    }

    /// List tables in a namespace.
    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.list_tables(namespace).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.list_tables(namespace).await
            }
            Err(e) => Err(e),
        }
    }

    /// Create a table.
    ///
    /// Note: If you get an auth error, call `force_refresh_token()` before retrying.
    /// This operation cannot auto-retry because `TableCreation` doesn't implement Clone.
    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<Table> {
        self.ensure_valid_token().await?;
        let result = {
            let state = self.state.read().await;
            state.catalog.create_table(namespace, creation).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(
                    error = %e,
                    "Got auth error on create_table, refreshing token. Caller must retry."
                );
                self.refresh_token().await?;
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Load a table.
    async fn load_table(&self, table: &TableIdent) -> Result<Table> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.load_table(table).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.load_table(table).await
            }
            Err(e) => Err(e),
        }
    }

    /// Drop a table.
    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.drop_table(table).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.drop_table(table).await
            }
            Err(e) => Err(e),
        }
    }

    /// Check if a table exists.
    async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.table_exists(table).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.table_exists(table).await
            }
            Err(e) => Err(e),
        }
    }

    /// Rename a table.
    async fn rename_table(&self, src: &TableIdent, dest: &TableIdent) -> Result<()> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state.catalog.rename_table(src, dest).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.rename_table(src, dest).await
            }
            Err(e) => Err(e),
        }
    }

    /// Update a table (commit).
    ///
    /// Note: If you get an auth error, call `force_refresh_token()` before retrying.
    /// This operation cannot auto-retry because `TableCommit` doesn't implement Clone.
    async fn update_table(&self, commit: TableCommit) -> Result<Table> {
        self.ensure_valid_token().await?;
        let result = {
            let state = self.state.read().await;
            state.catalog.update_table(commit).await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(
                    error = %e,
                    "Got auth error on update_table, refreshing token. Caller must retry."
                );
                self.refresh_token().await?;
                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    /// Register a table.
    async fn register_table(&self, table: &TableIdent, metadata_location: String) -> Result<Table> {
        self.ensure_valid_token().await?;

        let result = {
            let state = self.state.read().await;
            state
                .catalog
                .register_table(table, metadata_location.clone())
                .await
        };

        match result {
            Ok(v) => Ok(v),
            Err(e) if Self::is_auth_error(&e) => {
                tracing::warn!(error = %e, "Got auth error, refreshing token and retrying");
                self.refresh_token().await?;
                let state = self.state.read().await;
                state.catalog.register_table(table, metadata_location).await
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_auth_error() {
        let err_401 = Error::new(ErrorKind::Unexpected, "HTTP 401 Unauthorized");
        assert!(BigLakeCatalog::is_auth_error(&err_401));

        let err_403 = Error::new(ErrorKind::Unexpected, "HTTP 403 Forbidden");
        assert!(BigLakeCatalog::is_auth_error(&err_403));

        let err_permission =
            Error::new(ErrorKind::Unexpected, "Permission denied to access resource");
        assert!(BigLakeCatalog::is_auth_error(&err_permission));

        let err_other = Error::new(ErrorKind::DataInvalid, "Invalid table name");
        assert!(!BigLakeCatalog::is_auth_error(&err_other));
    }
}
