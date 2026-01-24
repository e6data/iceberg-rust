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

//! Utility functions for BigLake catalog.

use iceberg::{Error, ErrorKind, NamespaceIdent, Result};

/// Validates that a namespace is single-level (BigLake requirement).
pub fn validate_namespace(namespace: &NamespaceIdent) -> Result<()> {
    if namespace.as_ref().len() != 1 {
        return Err(Error::new(
            ErrorKind::DataInvalid,
            format!(
                "BigLake only supports single-level namespaces, got {} levels: {:?}",
                namespace.as_ref().len(),
                namespace
            ),
        ));
    }
    Ok(())
}

/// Builds the BigLake catalog resource path prefix.
pub fn build_catalog_prefix(project_id: &str, location: &str, catalog_name: &str) -> String {
    format!("projects/{project_id}/locations/{location}/catalogs/{catalog_name}")
}
