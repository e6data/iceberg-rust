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

//! GCP token fetching from metadata server.

use std::time::{Duration, Instant};

use iceberg::{Error, ErrorKind, Result};

/// Default GCP metadata server host
const DEFAULT_METADATA_HOST: &str = "metadata.google.internal";

/// Buffer time before token expiry to trigger refresh (5 minutes)
pub const TOKEN_REFRESH_BUFFER: Duration = Duration::from_secs(300);

/// Response from GCP metadata server token endpoint
#[derive(Debug, serde::Deserialize)]
struct GcpTokenResponse {
    access_token: String,
    expires_in: u64,
    // token_type is always "Bearer" but we don't need it
}

/// Token with expiry information
#[derive(Debug, Clone)]
pub struct Token {
    pub access_token: String,
    pub expires_at: Instant,
}

impl Token {
    /// Check if token is expired or expiring soon
    pub fn is_expiring(&self) -> bool {
        Instant::now() + TOKEN_REFRESH_BUFFER > self.expires_at
    }
}

/// Fetch OAuth2 access token from GCP metadata server.
///
/// Works on GKE pods, GCE VMs, Cloud Run, and Cloud Functions.
/// The metadata server is available at a well-known internal address.
///
/// # Arguments
/// * `service_account` - Service account name (usually "default" for attached SA)
///
/// # Returns
/// * `Ok(Token)` - Successfully fetched token with expiry
/// * `Err(...)` - Network or parsing error
pub async fn fetch_gcp_token(service_account: &str) -> Result<Token> {
    // Allow override of metadata host via env var (useful for testing)
    let metadata_host =
        std::env::var("GCE_METADATA_HOST").unwrap_or_else(|_| DEFAULT_METADATA_HOST.to_string());

    let url = format!(
        "http://{}/computeMetadata/v1/instance/service-accounts/{}/token",
        metadata_host, service_account
    );

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to create HTTP client for GCP metadata",
            )
            .with_source(e)
        })?;

    let response = client
        .get(&url)
        .header("Metadata-Flavor", "Google")
        .send()
        .await
        .map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!(
                    "Failed to reach GCP metadata server at '{}'. \
                     Ensure you're running on GCP (GKE/GCE/Cloud Run) with proper service account",
                    metadata_host
                ),
            )
            .with_source(e)
        })?;

    if !response.status().is_success() {
        return Err(Error::new(
            ErrorKind::Unexpected,
            format!("GCP metadata server returned status {}", response.status()),
        ));
    }

    let token_response: GcpTokenResponse = response.json().await.map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to parse GCP metadata token response",
        )
        .with_source(e)
    })?;

    let expires_at = Instant::now() + Duration::from_secs(token_response.expires_in);

    tracing::debug!(
        expires_in_secs = token_response.expires_in,
        "Fetched GCP access token from metadata server"
    );

    Ok(Token {
        access_token: token_response.access_token,
        expires_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_expiry_check() {
        let token = Token {
            access_token: "test".to_string(),
            expires_at: Instant::now() + Duration::from_secs(60), // expires in 1 min
        };

        // Should be expiring (within 5 min buffer)
        assert!(token.is_expiring());

        let fresh_token = Token {
            access_token: "test".to_string(),
            expires_at: Instant::now() + Duration::from_secs(3600), // expires in 1 hour
        };

        // Should not be expiring
        assert!(!fresh_token.is_expiring());
    }
}
