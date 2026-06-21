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

//! Azure Workload-Identity bearer-token injection for opendal-azdls.
//!
//! `opendal-service-azdls` 0.57's `AzdlsBuilder` constructs its credential
//! chain with a `StaticEnv` that contains only the explicitly-set
//! `adls.client-id` / `adls.tenant-id` / `adls.authority-host` properties —
//! it never propagates `AZURE_FEDERATED_TOKEN_FILE` from the OS env, which
//! reqsign's `WorkloadIdentityCredentialProvider` requires. The result on
//! AKS is that opendal's IMDS provider returns a node-VM identity (wrong
//! principal), and ADLS rejects writes with 403
//! `AuthorizationPermissionMismatch`.
//!
//! Rather than patch upstream opendal, this module performs the federated
//! → AAD token exchange ourselves and wraps the operator's `HttpClient` so
//! every outgoing request carries a fresh `Authorization: Bearer <token>`
//! header for the WI managed identity. We hook in via the public
//! `Operator::inner().info().update_http_client(..)` surface, so nothing in
//! the opendal stack needs to change.

use std::sync::Arc;
use std::time::{Duration, Instant};

use http::header::AUTHORIZATION;
use http::{HeaderValue, Request, Response};
use opendal::Buffer;
use opendal::raw::{HttpBody, HttpClient, HttpFetch};
use opendal::{Error as ODError, ErrorKind as ODErrorKind, Result as ODResult};
use serde::Deserialize;
use tokio::sync::Mutex;

use crate::{Error, ErrorKind, Result};

const AAD_SCOPE: &str = "https://storage.azure.com/.default";
const DEFAULT_AUTHORITY_HOST: &str = "https://login.microsoftonline.com";
const REFRESH_SKEW: Duration = Duration::from_secs(120);

/// Environment-derived WI settings the federated exchange needs.
#[derive(Debug, Clone)]
pub(crate) struct WiEnv {
    pub tenant_id: String,
    pub client_id: String,
    pub federated_token_file: String,
    pub authority_host: String,
}

/// Reads `AZURE_*` env vars; returns `None` if any of the three required
/// vars (tenant, client, federated token file) is missing or empty.
pub(crate) fn read_wi_env() -> Option<WiEnv> {
    let tenant_id = non_empty_env("AZURE_TENANT_ID")?;
    let client_id = non_empty_env("AZURE_CLIENT_ID")?;
    let federated_token_file = non_empty_env("AZURE_FEDERATED_TOKEN_FILE")?;
    let authority_host = non_empty_env("AZURE_AUTHORITY_HOST")
        .unwrap_or_else(|| DEFAULT_AUTHORITY_HOST.to_string());
    Some(WiEnv {
        tenant_id,
        client_id,
        federated_token_file,
        authority_host,
    })
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

#[derive(Debug)]
struct CachedToken {
    token: String,
    /// Instant after which we should refresh (= AAD-expiry minus REFRESH_SKEW).
    refresh_after: Instant,
}

/// Caches a WI access token and refreshes on demand using the federated
/// assertion flow.
#[derive(Debug)]
pub(crate) struct WiTokenFetcher {
    env: WiEnv,
    http: reqwest::Client,
    cached: Mutex<Option<CachedToken>>,
}

impl WiTokenFetcher {
    pub(crate) fn new(env: WiEnv) -> Self {
        Self {
            env,
            http: reqwest::Client::new(),
            cached: Mutex::new(None),
        }
    }

    /// Returns a valid bearer token, refreshing the cached one if needed.
    pub(crate) async fn get_token(&self) -> Result<String> {
        let mut guard = self.cached.lock().await;
        if let Some(cached) = guard.as_ref() {
            if Instant::now() < cached.refresh_after {
                return Ok(cached.token.clone());
            }
        }

        let assertion = tokio::fs::read_to_string(&self.env.federated_token_file)
            .await
            .map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    format!(
                        "WI: read AZURE_FEDERATED_TOKEN_FILE {}: {e}",
                        self.env.federated_token_file
                    ),
                )
            })?;
        let assertion = assertion.trim().to_string();

        let url = format!(
            "{}/{}/oauth2/v2.0/token",
            self.env.authority_host.trim_end_matches('/'),
            self.env.tenant_id
        );
        let form = [
            ("client_id", self.env.client_id.as_str()),
            ("scope", AAD_SCOPE),
            ("grant_type", "client_credentials"),
            (
                "client_assertion_type",
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            ),
            ("client_assertion", assertion.as_str()),
        ];
        let resp = self
            .http
            .post(&url)
            .form(&form)
            .send()
            .await
            .map_err(|e| Error::new(ErrorKind::Unexpected, format!("WI: AAD POST {url}: {e}")))?;
        let status = resp.status();
        let body = resp
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable>".to_string());
        if !status.is_success() {
            return Err(Error::new(
                ErrorKind::Unexpected,
                format!("WI: AAD token endpoint returned {status}: {body}"),
            ));
        }
        let tr: TokenResponse = serde_json::from_str(&body).map_err(|e| {
            Error::new(
                ErrorKind::Unexpected,
                format!("WI: parse AAD response: {e}; body={body}"),
            )
        })?;

        let refresh_after = Instant::now() + Duration::from_secs(tr.expires_in) - REFRESH_SKEW;
        let token = tr.access_token;
        *guard = Some(CachedToken {
            token: token.clone(),
            refresh_after,
        });
        Ok(token)
    }
}

/// `HttpFetch` wrapper that injects the WI bearer token on every outgoing
/// request unless the caller already set its own `Authorization`.
pub(crate) struct WiHttpFetch {
    inner: HttpClient,
    fetcher: Arc<WiTokenFetcher>,
}

impl WiHttpFetch {
    pub(crate) fn new(inner: HttpClient, fetcher: Arc<WiTokenFetcher>) -> Self {
        Self { inner, fetcher }
    }
}

impl HttpFetch for WiHttpFetch {
    async fn fetch(&self, mut req: Request<Buffer>) -> ODResult<Response<HttpBody>> {
        if !req.headers().contains_key(AUTHORIZATION) {
            let token = self.fetcher.get_token().await.map_err(|e| {
                ODError::new(ODErrorKind::Unexpected, "WI: fetch bearer token failed")
                    .set_source(e)
            })?;
            let value = HeaderValue::from_str(&format!("Bearer {token}")).map_err(|e| {
                ODError::new(
                    ODErrorKind::Unexpected,
                    "WI: bearer token has invalid header bytes",
                )
                .set_source(e)
            })?;
            req.headers_mut().insert(AUTHORIZATION, value);
        }
        self.inner.fetch(req).await
    }
}

/// Wrap the operator's existing `HttpClient` with WI bearer injection.
///
/// Builds an `HttpClient` whose `HttpFetch` impl prepends an
/// `Authorization: Bearer <token>` to every request before delegating to
/// the wrapped client.
pub(crate) fn wrap_http_client(inner: HttpClient, fetcher: Arc<WiTokenFetcher>) -> HttpClient {
    HttpClient::with(WiHttpFetch::new(inner, fetcher))
}

#[allow(dead_code)]
fn _force_link<B>(_: Request<Buffer>, _: Response<B>) {}
