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

//! BigLake Catalog implementation using native gRPC.

use std::collections::HashMap;
use std::fmt::Debug;

use async_trait::async_trait;
use google_cloud_api::model::HttpBody;
use google_cloud_biglake_v1::client::IcebergCatalogService;
use google_cloud_biglake_v1::model::{IcebergNamespace, IcebergNamespaceUpdate};
use iceberg::io::FileIO;
use iceberg::spec::{TableMetadata, TableMetadataBuilder};
use iceberg::table::Table;
use iceberg::{
    Catalog, CatalogBuilder, Error, ErrorKind, MetadataLocation, Namespace, NamespaceIdent, Result,
    TableCommit, TableCreation, TableIdent,
};

use crate::error::{from_biglake_error, is_not_found};
use crate::{BIGLAKE_CATALOG_ID, BIGLAKE_PROJECT_ID, BIGLAKE_WAREHOUSE};

/// BigLake Catalog configuration.
#[derive(Debug, Clone)]
pub struct BigLakeCatalogConfig {
    /// Catalog name
    pub name: String,
    /// GCP project ID
    pub project_id: String,
    /// BigLake catalog ID
    pub catalog_id: String,
    /// GCS warehouse path
    pub warehouse: String,
    /// User project for billing (optional)
    pub user_project: Option<String>,
    /// Additional properties
    pub props: HashMap<String, String>,
}

impl BigLakeCatalogConfig {
    /// Returns the catalog parent path: projects/{project}/catalogs/{catalog}
    pub fn catalog_parent(&self) -> String {
        format!("projects/{}/catalogs/{}", self.project_id, self.catalog_id)
    }

    /// Returns namespace path for a given namespace
    pub fn namespace_path(&self, namespace: &str) -> String {
        format!("{}/namespaces/{}", self.catalog_parent(), namespace)
    }

    /// Returns table path
    pub fn table_path(&self, namespace: &str, table: &str) -> String {
        format!("{}/tables/{}", self.namespace_path(namespace), table)
    }

    /// Returns the default table location for a table
    pub fn default_table_location(&self, namespace: &str, table: &str) -> String {
        format!("{}/{}/{}", self.warehouse.trim_end_matches('/'), namespace, table)
    }
}

/// Builder for [`BigLakeCatalog`].
#[derive(Debug, Default)]
pub struct BigLakeCatalogBuilder {
    config: Option<BigLakeCatalogConfig>,
}

impl CatalogBuilder for BigLakeCatalogBuilder {
    type C = BigLakeCatalog;

    fn load(
        mut self,
        name: impl Into<String>,
        props: HashMap<String, String>,
    ) -> impl std::future::Future<Output = Result<Self::C>> + Send {
        let name = name.into();

        let project_id = props.get(BIGLAKE_PROJECT_ID).cloned();
        let catalog_id = props.get(BIGLAKE_CATALOG_ID).cloned();
        let warehouse = props.get(BIGLAKE_WAREHOUSE).cloned();
        let user_project = props.get(crate::BIGLAKE_USER_PROJECT).cloned();

        // Collect remaining properties
        let remaining_props: HashMap<String, String> = props
            .into_iter()
            .filter(|(k, _)| {
                k != BIGLAKE_PROJECT_ID
                    && k != BIGLAKE_CATALOG_ID
                    && k != BIGLAKE_WAREHOUSE
                    && k != crate::BIGLAKE_USER_PROJECT
            })
            .collect();

        self.config = Some(BigLakeCatalogConfig {
            name,
            project_id: project_id.unwrap_or_default(),
            catalog_id: catalog_id.unwrap_or_default(),
            warehouse: warehouse.unwrap_or_default(),
            user_project,
            props: remaining_props,
        });

        async move {
            let config = self.config.ok_or_else(|| {
                Error::new(ErrorKind::DataInvalid, "Catalog configuration is required")
            })?;

            // Validate required fields
            if config.project_id.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} is required", BIGLAKE_PROJECT_ID),
                ));
            }
            if config.catalog_id.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} is required", BIGLAKE_CATALOG_ID),
                ));
            }
            if config.warehouse.is_empty() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} is required", BIGLAKE_WAREHOUSE),
                ));
            }
            if !config.warehouse.starts_with("gs://") {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!("{} must start with gs://", BIGLAKE_WAREHOUSE),
                ));
            }

            BigLakeCatalog::new(config).await
        }
    }
}

/// BigLake Catalog using native gRPC.
pub struct BigLakeCatalog {
    config: BigLakeCatalogConfig,
    client: IcebergCatalogService,
    file_io: FileIO,
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
    async fn new(config: BigLakeCatalogConfig) -> Result<Self> {
        // Create the BigLake gRPC client using ADC
        let client = IcebergCatalogService::builder()
            .build()
            .await
            .map_err(|e| Error::new(ErrorKind::Unexpected, "Failed to create BigLake client").with_source(e))?;

        // Create FileIO for the warehouse
        let file_io = FileIO::from_path(&config.warehouse)?
            .with_props(&config.props)
            .build()?;

        Ok(BigLakeCatalog {
            config,
            client,
            file_io,
        })
    }

    /// Get the catalog's FileIO.
    pub fn file_io(&self) -> FileIO {
        self.file_io.clone()
    }

    /// Build FileIO with vended credentials from BigLake.
    async fn build_file_io_with_credentials(
        &self,
        namespace: &str,
        table_name: &str,
    ) -> Result<FileIO> {
        let creds_response = self
            .client
            .load_iceberg_table_credentials()
            .set_name(&self.config.table_path(namespace, table_name))
            .send()
            .await
            .map_err(from_biglake_error)?;

        let mut props = self.config.props.clone();

        // Apply vended credentials from the response
        for cred in &creds_response.storage_credentials {
            // The config map contains keys like "gcs.oauth2.token"
            for (key, value) in &cred.config {
                props.insert(key.clone(), value.clone());
            }
        }

        FileIO::from_path(&self.config.warehouse)?
            .with_props(props)
            .build()
    }

    /// Validate that a namespace is single-level (BigLake limitation).
    fn validate_namespace(namespace: &NamespaceIdent) -> Result<String> {
        let parts: Vec<&String> = namespace.iter().collect();
        if parts.len() != 1 {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "BigLake only supports single-level namespaces",
            ));
        }
        Ok(parts[0].clone())
    }
}

#[async_trait]
impl Catalog for BigLakeCatalog {
    /// List namespaces from BigLake catalog.
    ///
    /// BigLake only supports single-level namespaces, so if parent is Some,
    /// we return an empty list.
    async fn list_namespaces(
        &self,
        parent: Option<&NamespaceIdent>,
    ) -> Result<Vec<NamespaceIdent>> {
        // BigLake doesn't support nested namespaces
        if parent.is_some() {
            return Ok(vec![]);
        }

        let mut namespaces = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_iceberg_namespaces()
                .set_parent(&self.config.catalog_parent());

            if let Some(token) = &page_token {
                request = request.set_page_token(token);
            }

            let response = request.send().await.map_err(from_biglake_error)?;

            // Each namespace in the response is a ListValue (Vec<Value>) containing the namespace parts
            // For single-level namespaces, we take the first element
            for ns in &response.namespaces {
                // ns is a Vec<Value> representing an array like ["namespace_name"]
                if let Some(first) = ns.first() {
                    // Extract the string value using pattern matching
                    if let serde_json::Value::String(name) = first {
                        namespaces.push(NamespaceIdent::new(name.clone()));
                    }
                }
            }

            if response.next_page_token.is_empty() {
                break;
            }
            page_token = Some(response.next_page_token.clone());
        }

        Ok(namespaces)
    }

    /// Create a new namespace in the BigLake catalog.
    async fn create_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<Namespace> {
        let ns_name = Self::validate_namespace(namespace)?;

        let ns = IcebergNamespace::new()
            .set_namespace([&ns_name])
            .set_properties(properties.iter().map(|(k, v)| (k.as_str(), v.as_str())));

        self.client
            .create_iceberg_namespace()
            .set_parent(&self.config.catalog_parent())
            .set_iceberg_namespace(ns)
            .send()
            .await
            .map_err(from_biglake_error)?;

        Ok(Namespace::with_properties(namespace.clone(), properties))
    }

    /// Get a namespace from the BigLake catalog.
    async fn get_namespace(&self, namespace: &NamespaceIdent) -> Result<Namespace> {
        let ns_name = Self::validate_namespace(namespace)?;

        let response = self
            .client
            .get_iceberg_namespace()
            .set_name(&self.config.namespace_path(&ns_name))
            .send()
            .await
            .map_err(from_biglake_error)?;

        Ok(Namespace::with_properties(
            namespace.clone(),
            response.properties.clone(),
        ))
    }

    /// Check if a namespace exists in the BigLake catalog.
    async fn namespace_exists(&self, namespace: &NamespaceIdent) -> Result<bool> {
        let ns_name = Self::validate_namespace(namespace)?;

        let result = self
            .client
            .get_iceberg_namespace()
            .set_name(&self.config.namespace_path(&ns_name))
            .send()
            .await;

        match result {
            Ok(_) => Ok(true),
            Err(e) if is_not_found(&e) => Ok(false),
            Err(e) => Err(from_biglake_error(e)),
        }
    }

    /// Update a namespace in the BigLake catalog.
    async fn update_namespace(
        &self,
        namespace: &NamespaceIdent,
        properties: HashMap<String, String>,
    ) -> Result<()> {
        let ns_name = Self::validate_namespace(namespace)?;

        // Get current namespace to determine what to update
        let current = self.get_namespace(namespace).await?;
        let current_props = current.properties();

        // Compute removals (keys in current but not in new)
        let removals: Vec<&str> = current_props
            .keys()
            .filter(|k| !properties.contains_key(*k))
            .map(|k| k.as_str())
            .collect();

        // Compute updates (keys in new that are different or new)
        let updates: Vec<(&str, &str)> = properties
            .iter()
            .filter(|(k, v)| current_props.get(*k) != Some(*v))
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let ns_update = IcebergNamespaceUpdate::new()
            .set_removals(removals)
            .set_updates(updates);

        self.client
            .update_iceberg_namespace()
            .set_name(&self.config.namespace_path(&ns_name))
            .set_iceberg_namespace_update(ns_update)
            .send()
            .await
            .map_err(from_biglake_error)?;

        Ok(())
    }

    /// Drop a namespace from the BigLake catalog.
    async fn drop_namespace(&self, namespace: &NamespaceIdent) -> Result<()> {
        let ns_name = Self::validate_namespace(namespace)?;

        // Check if namespace is empty
        let tables = self.list_tables(namespace).await?;
        if !tables.is_empty() {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Namespace {} is not empty", ns_name),
            ));
        }

        self.client
            .delete_iceberg_namespace()
            .set_name(&self.config.namespace_path(&ns_name))
            .send()
            .await
            .map_err(from_biglake_error)?;

        Ok(())
    }

    /// List tables in a namespace.
    async fn list_tables(&self, namespace: &NamespaceIdent) -> Result<Vec<TableIdent>> {
        let ns_name = Self::validate_namespace(namespace)?;

        let mut tables = Vec::new();
        let mut page_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_iceberg_table_identifiers()
                .set_parent(&self.config.namespace_path(&ns_name));

            if let Some(token) = &page_token {
                request = request.set_page_token(token);
            }

            let response = request.send().await.map_err(from_biglake_error)?;

            for table_id in &response.identifiers {
                // Extract table name from the identifier
                tables.push(TableIdent::new(namespace.clone(), table_id.name.clone()));
            }

            if response.next_page_token.is_empty() {
                break;
            }
            page_token = Some(response.next_page_token.clone());
        }

        Ok(tables)
    }

    /// Create a new table in the BigLake catalog.
    async fn create_table(
        &self,
        namespace: &NamespaceIdent,
        creation: TableCreation,
    ) -> Result<Table> {
        let ns_name = Self::validate_namespace(namespace)?;
        let table_name = creation.name.clone();

        // Determine table location
        let location = creation
            .location
            .clone()
            .unwrap_or_else(|| self.config.default_table_location(&ns_name, &table_name));

        // Build initial table metadata
        let mut creation_with_location = creation;
        creation_with_location.location = Some(location.clone());

        let metadata = TableMetadataBuilder::from_table_creation(creation_with_location)?
            .build()?
            .metadata;

        // Generate metadata location
        let metadata_location = MetadataLocation::new_with_table_location(&location).to_string();

        // Write metadata to GCS
        metadata
            .write_to(&self.file_io, &metadata_location)
            .await?;

        // Create table in BigLake
        // The request body is the Iceberg REST spec CreateTableRequest format
        let request_body = serde_json::json!({
            "name": table_name,
            "location": location,
            "schema": metadata.current_schema(),
            "properties": metadata.properties(),
        });

        let body_bytes = serde_json::to_vec(&request_body).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to serialize request").with_source(e)
        })?;

        let http_body = HttpBody::new()
            .set_content_type("application/json")
            .set_data(body_bytes);

        self.client
            .create_iceberg_table()
            .set_parent(&self.config.namespace_path(&ns_name))
            .set_http_body(http_body)
            .send()
            .await
            .map_err(from_biglake_error)?;

        // Load and return the created table
        self.load_table(&TableIdent::new(namespace.clone(), table_name))
            .await
    }

    /// Load a table from the BigLake catalog.
    async fn load_table(&self, table: &TableIdent) -> Result<Table> {
        let ns_name = Self::validate_namespace(table.namespace())?;
        let table_name = table.name();

        // Get table from BigLake
        let response = self
            .client
            .get_iceberg_table()
            .set_name(&self.config.table_path(&ns_name, table_name))
            .send()
            .await
            .map_err(from_biglake_error)?;

        // Parse response to get metadata location
        let body: serde_json::Value =
            serde_json::from_slice(&response.data).map_err(|e| {
                Error::new(ErrorKind::DataInvalid, "Failed to parse table response").with_source(e)
            })?;

        let metadata_location = body["metadata-location"]
            .as_str()
            .or_else(|| body["metadataLocation"].as_str())
            .ok_or_else(|| {
                Error::new(ErrorKind::DataInvalid, "Missing metadata-location in response")
            })?;

        // Get vended credentials for GCS access
        let file_io = self
            .build_file_io_with_credentials(&ns_name, table_name)
            .await?;

        // Load actual metadata from GCS
        let metadata = TableMetadata::read_from(&file_io, metadata_location).await?;

        // Build and return Table
        Table::builder()
            .file_io(file_io)
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(table.clone())
            .build()
    }

    /// Drop a table from the BigLake catalog.
    async fn drop_table(&self, table: &TableIdent) -> Result<()> {
        let ns_name = Self::validate_namespace(table.namespace())?;
        let table_name = table.name();

        self.client
            .delete_iceberg_table()
            .set_name(&self.config.table_path(&ns_name, table_name))
            .send()
            .await
            .map_err(from_biglake_error)?;

        Ok(())
    }

    /// Check if a table exists in the BigLake catalog.
    async fn table_exists(&self, table: &TableIdent) -> Result<bool> {
        let ns_name = Self::validate_namespace(table.namespace())?;
        let table_name = table.name();

        let result = self
            .client
            .get_iceberg_table()
            .set_name(&self.config.table_path(&ns_name, table_name))
            .send()
            .await;

        match result {
            Ok(_) => Ok(true),
            Err(e) if is_not_found(&e) => Ok(false),
            Err(e) => Err(from_biglake_error(e)),
        }
    }

    /// Rename a table.
    ///
    /// BigLake API does not support table renaming, so this returns an error.
    async fn rename_table(&self, _src: &TableIdent, _dest: &TableIdent) -> Result<()> {
        Err(Error::new(
            ErrorKind::FeatureUnsupported,
            "BigLake does not support table renaming",
        ))
    }

    /// Register an existing table in the BigLake catalog.
    async fn register_table(&self, table: &TableIdent, metadata_location: String) -> Result<Table> {
        let ns_name = Self::validate_namespace(table.namespace())?;
        let table_name = table.name();

        // Read metadata to validate it exists
        let metadata = TableMetadata::read_from(&self.file_io, &metadata_location).await?;

        // Register table in BigLake
        self.client
            .register_iceberg_table()
            .set_parent(&self.config.namespace_path(&ns_name))
            .set_name(table_name)
            .set_metadata_location(&metadata_location)
            .send()
            .await
            .map_err(from_biglake_error)?;

        // Return the registered table
        Table::builder()
            .file_io(self.file_io())
            .metadata_location(metadata_location)
            .metadata(metadata)
            .identifier(table.clone())
            .build()
    }

    /// Update a table in the BigLake catalog.
    async fn update_table(&self, commit: TableCommit) -> Result<Table> {
        let table_ident = commit.identifier().clone();
        let ns_name = Self::validate_namespace(table_ident.namespace())?;

        // Load current table
        let current_table = self.load_table(&table_ident).await?;
        let _current_metadata_location = current_table.metadata_location_result()?.to_string();

        // Apply updates to get new metadata
        let staged_table = commit.apply(current_table)?;
        let staged_metadata_location = staged_table.metadata_location_result()?;

        // Write new metadata to GCS
        staged_table
            .metadata()
            .write_to(staged_table.file_io(), staged_metadata_location)
            .await?;

        // Update BigLake with new metadata location (CommitTable format)
        let request_body = serde_json::json!({
            "requirements": [],  // BigLake validates requirements server-side
            "updates": [{
                "action": "set-metadata-location",
                "metadata-location": staged_metadata_location,
            }],
        });

        let body_bytes = serde_json::to_vec(&request_body).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Failed to serialize request").with_source(e)
        })?;

        let http_body = HttpBody::new()
            .set_content_type("application/json")
            .set_data(body_bytes);

        self.client
            .update_iceberg_table()
            .set_name(&self.config.table_path(&ns_name, table_ident.name()))
            .set_http_body(http_body)
            .send()
            .await
            .map_err(from_biglake_error)?;

        Ok(staged_table)
    }
}
