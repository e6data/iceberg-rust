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

use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::time::{Duration, Instant};

use http::StatusCode;
use iceberg::{Error, ErrorKind, Result};
use reqwest::header::HeaderMap;
use reqwest::{Client, IntoUrl, Method, Request, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use tokio::sync::Mutex;

/// Maximum number of retries when the catalog returns `429 Too Many
/// Requests` or `503 Service Unavailable`.
const RATE_LIMIT_MAX_RETRIES: u32 = 3;
/// Upper bound on a single backoff sleep (honored even if `Retry-After`
/// suggests something larger).
const RATE_LIMIT_BACKOFF_CAP: Duration = Duration::from_secs(30);

/// Refresh a cached OAuth token this long before its server-reported expiry.
/// Buys headroom against clock skew and the mint round-trip; small enough that
/// short-lived tokens still get most of their useful TTL.
const EXPIRY_BUFFER: Duration = Duration::from_secs(60);

use crate::RestCatalogConfig;
use crate::types::{ErrorResponse, TokenResponse};

/// A token value together with its server-reported expiry deadline.
#[derive(Debug, Clone)]
struct CachedToken {
    value: String,
    /// `None` means the token cannot be auto-refreshed — either it was
    /// provided pre-issued via config (no credential to re-mint with) or the
    /// server omitted `expires_in`.
    expires_at: Option<Instant>,
}

impl CachedToken {
    /// Token has a known expiry within `EXPIRY_BUFFER` of now (or already
    /// past). Tokens with no expiry are treated as fresh — they are either
    /// pre-issued or come from servers that don't advertise TTL.
    fn is_expiring(&self) -> bool {
        match self.expires_at {
            Some(exp) => exp.saturating_duration_since(Instant::now()) <= EXPIRY_BUFFER,
            None => false,
        }
    }
}

/// All state needed to mint and cache an OAuth token. Borrows the HTTP
/// transport from `HttpClient` rather than owning it, so there's one
/// authoritative owner of the `reqwest::Client`.
pub(crate) struct TokenState {
    token_endpoint: String,
    credential: Option<(Option<String>, String)>,
    extra_oauth_params: HashMap<String, String>,
    cache: Mutex<Option<CachedToken>>,
}

impl Debug for TokenState {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenState")
            .field("token_endpoint", &self.token_endpoint)
            .finish_non_exhaustive()
    }
}

impl TokenState {
    fn new(
        token_endpoint: String,
        credential: Option<(Option<String>, String)>,
        extra_oauth_params: HashMap<String, String>,
        initial_cache: Option<CachedToken>,
    ) -> Self {
        Self {
            token_endpoint,
            credential,
            extra_oauth_params,
            cache: Mutex::new(initial_cache),
        }
    }

    /// Perform a single `client_credentials` token exchange. Does not mutate
    /// the cache — caller decides whether to store the result.
    async fn mint_once(&self, client: &Client) -> Result<CachedToken> {
        let (client_id, client_secret) = self.credential.as_ref().ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                "Credential must be provided for authentication",
            )
        })?;

        let mut params = HashMap::with_capacity(4);
        params.insert("grant_type", "client_credentials");
        if let Some(client_id) = client_id {
            params.insert("client_id", client_id);
        }
        params.insert("client_secret", client_secret);
        params.extend(
            self.extra_oauth_params
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str())),
        );

        let mut auth_req = client
            .request(Method::POST, &self.token_endpoint)
            .form(&params)
            .build()?;
        auth_req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/x-www-form-urlencoded"),
        );
        let auth_url = auth_req.url().clone();
        let auth_resp = client.execute(auth_req).await?;

        let auth_res: TokenResponse = if auth_resp.status() == StatusCode::OK {
            let text = auth_resp
                .bytes()
                .await
                .map_err(|err| err.with_url(auth_url.clone()))?;
            serde_json::from_slice(&text).map_err(|e| {
                Error::new(
                    ErrorKind::Unexpected,
                    "Failed to parse response from rest catalog server!",
                )
                .with_context("operation", "auth")
                .with_context("url", auth_url.to_string())
                .with_context("json", String::from_utf8_lossy(&text))
                .with_source(e)
            })?
        } else {
            let code = auth_resp.status();
            let text = auth_resp
                .bytes()
                .await
                .map_err(|err| err.with_url(auth_url.clone()))?;
            let e: ErrorResponse = serde_json::from_slice(&text).map_err(|e| {
                Error::new(ErrorKind::Unexpected, "Received unexpected response")
                    .with_context("code", code.to_string())
                    .with_context("operation", "auth")
                    .with_context("url", auth_url.to_string())
                    .with_context("json", String::from_utf8_lossy(&text))
                    .with_source(e)
            })?;
            return Err(Error::from(e));
        };

        let expires_at = auth_res
            .expires_in
            .map(|secs| Instant::now() + Duration::from_secs(secs));
        if expires_at.is_none() {
            tracing::debug!(
                token_endpoint = %self.token_endpoint,
                "oauth token minted without expires_in; will not auto-refresh"
            );
        } else {
            tracing::debug!(
                token_endpoint = %self.token_endpoint,
                expires_in_secs = auth_res.expires_in,
                "oauth token minted"
            );
        }
        Ok(CachedToken {
            value: auth_res.access_token,
            expires_at,
        })
    }

    /// Return a bearer token to use, minting a fresh one if the cache is
    /// empty or expiring. The cache `Mutex` is held across the mint so
    /// concurrent callers serialize and exactly one OAuth round-trip fires
    /// per refresh cycle. Returns `Ok(None)` only when no auth is configured
    /// at all (no credential and no pre-issued token).
    async fn get_or_refresh(&self, client: &Client) -> Result<Option<String>> {
        let mut cache = self.cache.lock().await;

        if let Some(c) = cache.as_ref() {
            if !c.is_expiring() {
                return Ok(Some(c.value.clone()));
            }
        }

        if self.credential.is_none() {
            return Ok(cache.as_ref().map(|c| c.value.clone()));
        }

        let fresh = self.mint_once(client).await?;
        let value = fresh.value.clone();
        *cache = Some(fresh);
        Ok(Some(value))
    }

    async fn invalidate(&self) {
        *self.cache.lock().await = None;
    }
}

pub(crate) struct HttpClient {
    client: Client,
    state: TokenState,
    /// Extra headers to be added to each request.
    extra_headers: HeaderMap,
}

impl Debug for HttpClient {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpClient")
            .field("client", &self.client)
            .field("state", &self.state)
            .field("extra_headers", &self.extra_headers)
            .finish_non_exhaustive()
    }
}

impl HttpClient {
    pub fn new(cfg: &RestCatalogConfig) -> Result<Self> {
        let extra_headers = cfg.extra_headers()?;
        let initial_cache = cfg.token().map(|value| CachedToken {
            value,
            expires_at: None,
        });
        let state = TokenState::new(
            cfg.get_token_endpoint(),
            cfg.credential(),
            cfg.extra_oauth_params(),
            initial_cache,
        );
        Ok(HttpClient {
            client: cfg.client().unwrap_or_default(),
            state,
            extra_headers,
        })
    }

    /// Update the http client with new configuration.
    ///
    /// When the token endpoint, credential, and extra oauth params are all
    /// unchanged and no new pre-issued token is supplied, this reuses the
    /// existing `TokenState` — preserving the cached token across the
    /// `/v1/config` → merged-config transition. Otherwise it builds fresh
    /// state.
    pub async fn update_with(self, cfg: &RestCatalogConfig) -> Result<Self> {
        let new_extra_headers = cfg.extra_headers()?;
        let extra_headers = if new_extra_headers.is_empty() {
            self.extra_headers
        } else {
            new_extra_headers
        };

        let new_endpoint = {
            let endpoint = cfg.get_token_endpoint();
            if endpoint.is_empty() {
                self.state.token_endpoint.clone()
            } else {
                endpoint
            }
        };
        let new_credential = cfg.credential().or_else(|| self.state.credential.clone());
        let new_oauth = {
            let oauth = cfg.extra_oauth_params();
            if oauth.is_empty() {
                self.state.extra_oauth_params.clone()
            } else {
                oauth
            }
        };
        let client = cfg.client().unwrap_or(self.client);
        let new_user_token = cfg.token();

        let unchanged = new_endpoint == self.state.token_endpoint
            && new_credential == self.state.credential
            && new_oauth == self.state.extra_oauth_params;

        if unchanged && new_user_token.is_none() {
            return Ok(HttpClient {
                client,
                state: self.state,
                extra_headers,
            });
        }

        let initial_cache = if let Some(t) = new_user_token {
            Some(CachedToken {
                value: t,
                expires_at: None,
            })
        } else if unchanged {
            self.state.cache.lock().await.clone()
        } else {
            None
        };

        let state = TokenState::new(new_endpoint, new_credential, new_oauth, initial_cache);
        Ok(HttpClient {
            client,
            state,
            extra_headers,
        })
    }

    /// This API is testing only to assert the token.
    #[cfg(test)]
    pub(crate) async fn token(&self) -> Option<String> {
        self.state.get_or_refresh(&self.client).await.ok().flatten()
    }

    /// Invalidate the current token without generating a new one. On the next
    /// request, the client will attempt to mint a new token.
    pub(crate) async fn invalidate_token(&self) -> Result<()> {
        self.state.invalidate().await;
        Ok(())
    }

    /// Invalidate the current token and mint a new one. Mints first so that
    /// if the credential is invalid the current token is left intact.
    pub(crate) async fn regenerate_token(&self) -> Result<()> {
        let fresh = self.state.mint_once(&self.client).await?;
        *self.state.cache.lock().await = Some(fresh);
        Ok(())
    }

    /// Add the bearer token to the request, minting one via OAuth if needed.
    /// No-op when neither a credential nor a pre-issued token is configured.
    /// A pre-issued `token` is used as-is and never refreshed.
    async fn authenticate(&self, req: &mut Request) -> Result<()> {
        let Some(token) = self.state.get_or_refresh(&self.client).await? else {
            return Ok(());
        };

        req.headers_mut().insert(
            http::header::AUTHORIZATION,
            format!("Bearer {token}").parse().map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Invalid token received from catalog server!",
                )
                .with_source(e)
            })?,
        );

        Ok(())
    }

    #[inline]
    pub fn request<U: IntoUrl>(&self, method: Method, url: U) -> RequestBuilder {
        self.client
            .request(method, url)
            .headers(self.extra_headers.clone())
    }

    /// Executes the given `Request` and returns a `Response`.
    pub async fn execute(&self, mut request: Request) -> Result<Response> {
        request.headers_mut().extend(self.extra_headers.clone());
        Ok(self.client.execute(request).await?)
    }

    /// Queries the Iceberg REST catalog with two retry classes:
    /// 401/419 → invalidate token and retry once (safety net for clock
    /// skew and server-side rotation); 429/503 → backoff and retry up to
    /// `RATE_LIMIT_MAX_RETRIES`, honoring `Retry-After` when present.
    pub async fn query_catalog(&self, request: Request) -> Result<Response> {
        let mut auth_retried = false;
        let mut rate_attempt: u32 = 0;

        loop {
            let mut attempt = request.try_clone().ok_or_else(|| {
                Error::new(
                    ErrorKind::Unexpected,
                    "request body cannot be cloned; catalog retry requires clonable bodies",
                )
            })?;

            self.authenticate(&mut attempt).await?;
            let response = self.execute(attempt).await?;
            let status = response.status().as_u16();

            match status {
                401 | 419 if !auth_retried => {
                    tracing::info!(
                        status,
                        "catalog returned auth failure; invalidating token and retrying once"
                    );
                    self.invalidate_token().await?;
                    auth_retried = true;
                    continue;
                }
                429 | 503 if rate_attempt < RATE_LIMIT_MAX_RETRIES => {
                    let wait = parse_retry_after(response.headers())
                        .unwrap_or_else(|| exponential_backoff(rate_attempt))
                        .min(RATE_LIMIT_BACKOFF_CAP);
                    tracing::warn!(
                        status,
                        attempt = rate_attempt + 1,
                        wait_ms = wait.as_millis() as u64,
                        "catalog rate-limited; backing off"
                    );
                    drop(response);
                    tokio::time::sleep(wait).await;
                    rate_attempt += 1;
                    continue;
                }
                _ => return Ok(response),
            }
        }
    }
}

/// Parse an RFC 7231 `Retry-After` integer-seconds header. HTTP-date form is
/// not supported — practically no catalog uses it.
fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(http::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Exponential backoff: 500ms, 1s, 2s, ... Caller bounds `attempt` to
/// `RATE_LIMIT_MAX_RETRIES` and caps the result at `RATE_LIMIT_BACKOFF_CAP`.
fn exponential_backoff(attempt: u32) -> Duration {
    Duration::from_millis(500) * (1u32 << attempt)
}

/// Deserializes a catalog response into the given [`DeserializedOwned`] type.
pub(crate) async fn deserialize_catalog_response<R: DeserializeOwned>(
    response: Response,
) -> Result<R> {
    let bytes = response.bytes().await?;

    serde_json::from_slice::<R>(&bytes).map_err(|e| {
        Error::new(
            ErrorKind::Unexpected,
            "Failed to parse response from rest catalog server",
        )
        .with_context("json", String::from_utf8_lossy(&bytes))
        .with_source(e)
    })
}

/// Deserializes a unexpected catalog response into an error.
pub(crate) async fn deserialize_unexpected_catalog_error(response: Response) -> Error {
    let err = Error::new(
        ErrorKind::Unexpected,
        "Received response with unexpected status code",
    )
    .with_context("status", response.status().to_string())
    .with_context("headers", format!("{:?}", response.headers()));

    let bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(err) => return err.into(),
    };

    if bytes.is_empty() {
        return err;
    }
    err.with_context("json", String::from_utf8_lossy(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_expiring_with_no_expiry_returns_false() {
        let t = CachedToken {
            value: "x".into(),
            expires_at: None,
        };
        assert!(!t.is_expiring());
    }

    #[test]
    fn is_expiring_with_far_future_expiry_returns_false() {
        let t = CachedToken {
            value: "x".into(),
            expires_at: Some(Instant::now() + Duration::from_secs(3600)),
        };
        assert!(!t.is_expiring());
    }

    #[test]
    fn is_expiring_within_buffer_returns_true() {
        let t = CachedToken {
            value: "x".into(),
            expires_at: Some(Instant::now() + Duration::from_secs(30)),
        };
        assert!(t.is_expiring());
    }

    #[test]
    fn is_expiring_already_past_returns_true() {
        let t = CachedToken {
            value: "x".into(),
            expires_at: Some(Instant::now() - Duration::from_secs(10)),
        };
        assert!(t.is_expiring());
    }

    #[test]
    fn exponential_backoff_doubles() {
        assert_eq!(exponential_backoff(0), Duration::from_millis(500));
        assert_eq!(exponential_backoff(1), Duration::from_millis(1000));
        assert_eq!(exponential_backoff(2), Duration::from_millis(2000));
        assert_eq!(exponential_backoff(3), Duration::from_millis(4000));
    }

    #[test]
    fn parse_retry_after_integer_seconds() {
        let mut h = HeaderMap::new();
        h.insert(
            http::header::RETRY_AFTER,
            http::HeaderValue::from_static("12"),
        );
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(12)));
    }

    #[test]
    fn parse_retry_after_absent_or_malformed() {
        let empty = HeaderMap::new();
        assert_eq!(parse_retry_after(&empty), None);

        let mut date_form = HeaderMap::new();
        date_form.insert(
            http::header::RETRY_AFTER,
            http::HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(parse_retry_after(&date_form), None);
    }
}
