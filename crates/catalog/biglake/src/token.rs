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

//! GCP token provider for BigLake authentication.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use iceberg::{Error, ErrorKind, Result};
use iceberg_catalog_rest::TokenProvider;
use serde::Deserialize;
use tokio::sync::RwLock;

/// Default GCE metadata server host
const DEFAULT_GCE_METADATA_HOST: &str = "metadata.google.internal";

/// Token response from GCP metadata server or OAuth2 endpoint
#[derive(Debug, Deserialize)]
struct GcpTokenResponse {
    access_token: String,
    #[serde(default)]
    expires_in: u64,
    #[allow(dead_code)]
    #[serde(default)]
    token_type: String,
}

/// Application Default Credentials file format
#[derive(Debug, Deserialize)]
struct AdcCredentials {
    /// For authorized_user type
    client_id: Option<String>,
    client_secret: Option<String>,
    refresh_token: Option<String>,
    /// For service_account type
    client_email: Option<String>,
    private_key: Option<String>,
    /// Credential type
    #[serde(rename = "type")]
    cred_type: Option<String>,
}

/// Cached token with expiry
struct CachedToken {
    access_token: String,
    expires_at: Instant,
}

/// Token provider that uses Google Cloud Application Default Credentials (ADC).
///
/// Supports authentication in this order:
/// 1. Explicit credentials JSON (base64 encoded) via configuration
/// 2. GOOGLE_APPLICATION_CREDENTIALS environment variable
/// 3. gcloud CLI default credentials (~/.config/gcloud/application_default_credentials.json)
/// 4. GCE/GKE metadata server (on Google Cloud compute environments)
pub struct GcpTokenProvider {
    /// Base64-encoded credentials JSON (optional)
    credentials_json: Option<String>,
    /// Cached token with expiry
    cached_token: Arc<RwLock<Option<CachedToken>>>,
    /// HTTP client for token requests
    client: reqwest::Client,
}

impl std::fmt::Debug for GcpTokenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GcpTokenProvider")
            .field("has_credentials", &self.credentials_json.is_some())
            .finish()
    }
}

impl GcpTokenProvider {
    /// Creates a new GcpTokenProvider using Application Default Credentials.
    pub async fn new() -> Result<Self> {
        Self::with_credentials(None).await
    }

    /// Creates a new GcpTokenProvider with optional explicit credentials.
    ///
    /// # Arguments
    /// * `credentials_json` - Optional base64-encoded service account JSON
    pub async fn with_credentials(credentials_json: Option<&str>) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Failed to create HTTP client").with_source(e)
            })?;

        Ok(Self {
            credentials_json: credentials_json.map(String::from),
            cached_token: Arc::new(RwLock::new(None)),
            client,
        })
    }

    /// Fetch a fresh token using the configured credentials.
    async fn fetch_token(&self) -> Result<CachedToken> {
        // Try methods in order of priority
        if let Some(ref creds_b64) = self.credentials_json {
            return self.fetch_token_from_credentials(creds_b64).await;
        }

        // Try GOOGLE_APPLICATION_CREDENTIALS
        if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
            if let Ok(token) = self.fetch_token_from_file(&PathBuf::from(path)).await {
                return Ok(token);
            }
        }

        // Try gcloud ADC file
        if let Some(adc_path) = self.get_gcloud_adc_path() {
            if adc_path.exists() {
                if let Ok(token) = self.fetch_token_from_file(&adc_path).await {
                    return Ok(token);
                }
            }
        }

        // Try metadata server (GCE/GKE/Cloud Run)
        self.fetch_token_from_metadata_server("default").await
    }

    /// Get the path to gcloud's application default credentials file.
    fn get_gcloud_adc_path(&self) -> Option<PathBuf> {
        dirs::home_dir().map(|home| {
            home.join(".config")
                .join("gcloud")
                .join("application_default_credentials.json")
        })
    }

    /// Fetch token from base64-encoded credentials JSON.
    async fn fetch_token_from_credentials(&self, creds_b64: &str) -> Result<CachedToken> {
        use base64::Engine;
        let creds_bytes =
            base64::engine::general_purpose::STANDARD
                .decode(creds_b64)
                .map_err(|e| {
                    Error::new(ErrorKind::DataInvalid, "Invalid base64 credentials").with_source(e)
                })?;

        let creds: AdcCredentials = serde_json::from_slice(&creds_bytes).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Invalid credentials JSON").with_source(e)
        })?;

        self.fetch_token_with_creds(&creds).await
    }

    /// Fetch token from a credentials file.
    async fn fetch_token_from_file(&self, path: &PathBuf) -> Result<CachedToken> {
        let content = tokio::fs::read_to_string(path).await.map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("Failed to read credentials file: {}", path.display()),
            )
            .with_source(e)
        })?;

        let creds: AdcCredentials = serde_json::from_str(&content).map_err(|e| {
            Error::new(ErrorKind::DataInvalid, "Invalid credentials file format").with_source(e)
        })?;

        self.fetch_token_with_creds(&creds).await
    }

    /// Fetch token using parsed credentials.
    async fn fetch_token_with_creds(&self, creds: &AdcCredentials) -> Result<CachedToken> {
        let cred_type = creds.cred_type.as_deref().unwrap_or("authorized_user");

        match cred_type {
            "authorized_user" => self.refresh_user_token(creds).await,
            "service_account" => {
                // For service accounts, we'd need to sign a JWT
                // For now, fall back to metadata server if available
                Err(Error::new(
                    ErrorKind::FeatureUnsupported,
                    "Service account JSON credentials require JWT signing. \
                     Use GOOGLE_APPLICATION_CREDENTIALS with a key file or run on GCP compute.",
                ))
            }
            _ => Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Unsupported credential type: {}", cred_type),
            )),
        }
    }

    /// Refresh an authorized_user token using the OAuth2 refresh flow.
    async fn refresh_user_token(&self, creds: &AdcCredentials) -> Result<CachedToken> {
        let client_id = creds.client_id.as_ref().ok_or_else(|| {
            Error::new(ErrorKind::DataInvalid, "Missing client_id in credentials")
        })?;
        let client_secret = creds.client_secret.as_ref().ok_or_else(|| {
            Error::new(ErrorKind::DataInvalid, "Missing client_secret in credentials")
        })?;
        let refresh_token = creds.refresh_token.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Missing refresh_token in credentials",
            )
        })?;

        let params = [
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("refresh_token", refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];

        let response = self
            .client
            .post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await
            .map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Failed to refresh OAuth2 token").with_source(e)
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("OAuth2 token refresh failed: {} - {}", status, body),
            ));
        }

        let token_response: GcpTokenResponse = response.json().await.map_err(|e| {
            Error::new(ErrorKind::Unexpected, "Failed to parse token response").with_source(e)
        })?;

        let expires_at = if token_response.expires_in > 0 {
            Instant::now() + Duration::from_secs(token_response.expires_in.saturating_sub(60))
        } else {
            // Default to 1 hour if not specified
            Instant::now() + Duration::from_secs(3540)
        };

        Ok(CachedToken {
            access_token: token_response.access_token,
            expires_at,
        })
    }

    /// Fetch token from GCP metadata server.
    ///
    /// Works on GKE pods, GCE VMs, Cloud Run, and Cloud Functions.
    async fn fetch_token_from_metadata_server(&self, service_account: &str) -> Result<CachedToken> {
        let metadata_host = std::env::var("GCE_METADATA_HOST")
            .unwrap_or_else(|_| DEFAULT_GCE_METADATA_HOST.to_string());

        let url = format!(
            "http://{}/computeMetadata/v1/instance/service-accounts/{}/token",
            metadata_host, service_account
        );

        let response = self
            .client
            .get(&url)
            .header("Metadata-Flavor", "Google")
            .send()
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "Failed to reach GCP metadata server at '{}'. \
                         Ensure you're running on GCP (GKE/GCE/Cloud Run) with proper service account, \
                         or set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'",
                        metadata_host
                    ),
                )
                .with_source(e)
            })?;

        if !response.status().is_success() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!(
                    "GCP metadata server returned status {}. \
                     Ensure you're running on GCP or have ADC configured.",
                    response.status()
                ),
            ));
        }

        let token_response: GcpTokenResponse = response.json().await.map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                "Failed to parse GCP metadata token response",
            )
            .with_source(e)
        })?;

        let expires_at = if token_response.expires_in > 0 {
            // Subtract 60 seconds to refresh before actual expiry
            Instant::now() + Duration::from_secs(token_response.expires_in.saturating_sub(60))
        } else {
            // Default to ~1 hour if not specified
            Instant::now() + Duration::from_secs(3540)
        };

        tracing::debug!(
            "Fetched GCP access token (expires in {} seconds)",
            token_response.expires_in
        );

        Ok(CachedToken {
            access_token: token_response.access_token,
            expires_at,
        })
    }
}

#[async_trait]
impl TokenProvider for GcpTokenProvider {
    async fn get_token(&self) -> Result<String> {
        // Check cache first
        {
            let cache = self.cached_token.read().await;
            if let Some(ref cached) = *cache {
                if Instant::now() < cached.expires_at {
                    return Ok(cached.access_token.clone());
                }
            }
        }

        // Fetch fresh token
        let token = self.fetch_token().await?;
        let access_token = token.access_token.clone();

        // Update cache
        *self.cached_token.write().await = Some(token);

        Ok(access_token)
    }

    async fn refresh_token(&self) -> Result<String> {
        // Clear cache and fetch fresh token
        *self.cached_token.write().await = None;
        self.get_token().await
    }
}
