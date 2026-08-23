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

//! Single-flight credential provider for the EKS Pod Identity Agent.
//!
//! ## The failure this fixes
//!
//! On sri-olly, tessellate loses ~45% of its tombstone deletes every tick
//! (measured 13,082 errors of 29,341 eligible in one sweep). Every one of
//! those failures is:
//!
//! ```text
//! ECS metadata endpoint returned unexpected status 429 Too Many Requests
//! endpoint: http://169.254.170.23/v1/credentials
//! retryable: false
//! ```
//!
//! Three upstream behaviours combine to produce it:
//!
//! 1. **No single-flight in `reqsign_core::Signer`.** `sign()` locks its
//!    cache, clones, *drops the guard*, then awaits `provide_credential`.
//!    So N concurrent signers that all observe an invalid credential each
//!    fire their own request at the agent. (Note it does NOT hold a
//!    `std::sync::Mutex` across the await — swapping that lock for an async
//!    one changes nothing on its own. What coalesces is *holding* a lock
//!    across the refresh, which is what this wrapper does.)
//! 2. **429 is marked non-retryable** by `reqsign_aws_v4`'s ECS provider:
//!    `500..=599` gets `set_retryable(true)`, the catch-all arm that 429
//!    lands in does not. Verified identical in 3.0.1 and 3.0.2, so there is
//!    no version to upgrade to.
//! 3. **`Signer::sign` has no retry**, so the error propagates to opendal as
//!    `Unexpected (persistent)` and kills the S3 operation outright.
//!
//! ## Why tessellate and not laminar
//!
//! It is cold start, not code. All 4,482 credential errors in a tick land in
//! the **first 3.4 seconds** of the pod's life — before `v2 tick start` at
//! +7.2s. A fresh tessellate pod has an empty credential cache and its very
//! first act is a 32-wide `buffer_unordered` fan-out over the tombstone
//! sweep. All 32 miss simultaneously and all 32 hit the agent. Laminar is a
//! long-lived process that warms its credential once at low concurrency and
//! refreshes rarely, so it never assembles a herd.
//!
//! That is why this wrapper is built around **single-flight**, and why a
//! longer TTL would not have helped: a pod that lives ~5 minutes never gets
//! to use one. (A TTL *extension* is not possible in any case — the agent's
//! `Expiration` is what STS signed; using the credential past it yields 403.)
//!
//! ## What this does
//!
//! Wraps the provider chain in an async mutex that is **held across the
//! refresh**, so concurrent callers collapse into one agent request, and
//! retries a 429 with jittered backoff instead of failing the S3 op.

use std::sync::Arc;
use std::time::Duration;

use reqsign_aws_v4::{Credential, DefaultCredentialProvider};
use reqsign_core::time::Timestamp;
use reqsign_core::{Context, ProvideCredential, Result as ReqsignResult};
use tokio::sync::Mutex;

/// Attempts at fetching a credential before giving up (1 initial + retries).
const MAX_ATTEMPTS: usize = 5;

/// Base for exponential backoff between attempts. With MAX_ATTEMPTS=5 the
/// worst-case added latency is ~50+100+200+400ms ≈ 750ms plus jitter, which
/// is well inside tessellate's 570s deadline and far cheaper than losing the
/// operation.
const BASE_BACKOFF: Duration = Duration::from_millis(50);

/// Upper bound on a single refresh. Because the lock is held across the
/// refresh, a hung agent call would otherwise stall every other signer in
/// the process.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(10);

/// A credential provider that serialises refreshes across all concurrent
/// signers in the process and retries the agent's 429.
///
/// Sits *below* `reqsign_core::Signer`'s own cache: `Signer` only calls into
/// here when its cached credential has expired (or on first use), so the
/// value this adds is coalescing that call, not caching per se.
#[derive(Debug)]
pub(crate) struct SharedCachedCredentialProvider<P = DefaultCredentialProvider> {
    inner: P,
    cached: Arc<Mutex<Option<Credential>>>,
    max_attempts: usize,
    base_backoff: Duration,
}

impl<P> SharedCachedCredentialProvider<P>
where P: ProvideCredential<Credential = Credential>
{
    pub(crate) fn new(inner: P) -> Self {
        Self {
            inner,
            cached: Arc::new(Mutex::new(None)),
            max_attempts: MAX_ATTEMPTS,
            base_backoff: BASE_BACKOFF,
        }
    }

    /// Shrink the backoff so tests don't sleep for real.
    #[cfg(test)]
    fn with_retry_params(mut self, max_attempts: usize, base_backoff: Duration) -> Self {
        self.max_attempts = max_attempts;
        self.base_backoff = base_backoff;
        self
    }
}

/// True when the agent rejected us for rate — the case upstream marks
/// non-retryable. Matched on the rendered error because `reqsign_core::Error`
/// exposes neither the HTTP status nor a typed kind for it; the provider
/// formats both `"429"` and `"Too Many Requests"` into the message and
/// context, so this is stable against either rendering.
fn is_rate_limited(err: &reqsign_core::Error) -> bool {
    let rendered = err.to_string();
    rendered.contains("429") || rendered.contains("Too Many Requests")
}

/// Deterministic-ish jitter without pulling in a RNG dependency: derives a
/// 0..=63ms offset from the low bits of the process-wide attempt counter so
/// concurrent processes on the same node don't re-collide in lockstep.
fn jitter_for(attempt: usize) -> Duration {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    // Mix the attempt in so successive retries of one call differ too.
    Duration::from_millis(((n ^ (attempt as u64).wrapping_mul(31)) & 0x3F) as u64)
}

impl<P> ProvideCredential for SharedCachedCredentialProvider<P>
where P: ProvideCredential<Credential = Credential>
{
    type Credential = Credential;

    async fn provide_credential(&self, ctx: &Context) -> ReqsignResult<Option<Credential>> {
        // Held across the refresh below — this is the single-flight. Callers
        // that arrive during an in-progress refresh block here and then read
        // the fresh value on the fast path rather than issuing their own
        // request at the agent.
        let mut guard = self.cached.lock().await;

        // Someone refreshed while we waited for the lock.
        if let Some(cred) = guard.as_ref() {
            if credential_is_usable(cred) {
                return Ok(Some(cred.clone()));
            }
        }

        let mut last_err: Option<reqsign_core::Error> = None;
        for attempt in 0..self.max_attempts {
            if attempt > 0 {
                let backoff = self.base_backoff * (1u32 << (attempt as u32 - 1));
                tokio::time::sleep(backoff + jitter_for(attempt)).await;
            }

            let fetched =
                match tokio::time::timeout(REFRESH_TIMEOUT, self.inner.provide_credential(ctx))
                    .await
                {
                    Ok(res) => res,
                    Err(_elapsed) => {
                        last_err = Some(reqsign_core::Error::unexpected(
                            "pod-identity credential refresh timed out",
                        ));
                        continue;
                    }
                };

            match fetched {
                Ok(Some(cred)) => {
                    *guard = Some(cred.clone());
                    return Ok(Some(cred));
                }
                // Chain exhausted without an error: nothing to retry.
                Ok(None) => return Ok(None),
                Err(err) => {
                    if is_rate_limited(&err) {
                        last_err = Some(err);
                        continue;
                    }
                    return Err(err);
                }
            }
        }

        Err(last_err.unwrap_or_else(|| {
            reqsign_core::Error::unexpected("pod-identity credential refresh failed")
        }))
    }
}

/// Mirrors `reqsign_aws_v4::Credential::is_valid`, which is private to that
/// crate: non-empty key material, and not inside the 120s pre-expiry buffer.
fn credential_is_usable(cred: &Credential) -> bool {
    if (cred.access_key_id.is_empty() || cred.secret_access_key.is_empty())
        && cred.session_token.is_none()
    {
        return false;
    }
    match cred.expires_in {
        Some(exp) => exp > Timestamp::now() + Duration::from_secs(120),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cred(expires_in: Option<Timestamp>) -> Credential {
        Credential {
            access_key_id: "AKIA".into(),
            secret_access_key: "secret".into(),
            session_token: Some("token".into()),
            expires_in,
        }
    }

    #[test]
    fn rate_limited_matches_the_live_agent_error() {
        // Verbatim shape of the error observed on sri-olly.
        let err = reqsign_core::Error::unexpected(
            "ECS metadata endpoint returned unexpected status 429 Too Many Requests: Too Many Requests",
        );
        assert!(is_rate_limited(&err));
    }

    #[test]
    fn rate_limited_ignores_unrelated_errors() {
        let err = reqsign_core::Error::unexpected("ECS metadata service error: 503");
        assert!(!is_rate_limited(&err));
        let err = reqsign_core::Error::config_invalid("ECS credentials endpoint not found");
        assert!(!is_rate_limited(&err));
    }

    #[test]
    fn credential_without_expiry_is_usable() {
        assert!(credential_is_usable(&cred(None)));
    }

    #[test]
    fn credential_inside_the_120s_buffer_is_not_usable() {
        let soon = Timestamp::now() + Duration::from_secs(60);
        assert!(!credential_is_usable(&cred(Some(soon))));
    }

    #[test]
    fn credential_well_ahead_of_expiry_is_usable() {
        let later = Timestamp::now() + Duration::from_secs(600);
        assert!(credential_is_usable(&cred(Some(later))));
    }

    #[test]
    fn credential_with_empty_key_material_is_not_usable() {
        let mut c = cred(None);
        c.access_key_id = String::new();
        c.secret_access_key = String::new();
        c.session_token = None;
        assert!(!credential_is_usable(&c));
    }

    #[test]
    fn jitter_stays_within_bounds() {
        for attempt in 0..8 {
            assert!(jitter_for(attempt) <= Duration::from_millis(63));
        }
    }

    /// Fake agent: 429s for the first `fail_times` calls, then succeeds.
    /// Counts total calls so coalescing can be asserted.
    #[derive(Debug)]
    struct FlakyAgent {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        fail_times: usize,
        ttl: Option<Duration>,
    }

    impl ProvideCredential for FlakyAgent {
        type Credential = Credential;

        async fn provide_credential(&self, _ctx: &Context) -> ReqsignResult<Option<Credential>> {
            use std::sync::atomic::Ordering;
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                return Err(reqsign_core::Error::unexpected(
                    "ECS metadata endpoint returned unexpected status 429 Too Many Requests",
                ));
            }
            Ok(Some(cred(self.ttl.map(|d| Timestamp::now() + d))))
        }
    }

    fn agent(fail_times: usize) -> (FlakyAgent, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            FlakyAgent {
                calls: calls.clone(),
                fail_times,
                ttl: Some(Duration::from_secs(3600)),
            },
            calls,
        )
    }

    fn count(c: &Arc<std::sync::atomic::AtomicUsize>) -> usize {
        c.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[tokio::test]
    async fn retries_past_a_burst_of_429s() {
        let (inner, calls) = agent(3);
        let p =
            SharedCachedCredentialProvider::new(inner).with_retry_params(5, Duration::from_millis(1));
        let got = p.provide_credential(&Context::new()).await.unwrap();
        assert!(got.is_some(), "should recover after transient 429s");
        assert_eq!(count(&calls), 4);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts_and_reports_the_429() {
        let (inner, calls) = agent(usize::MAX);
        let p =
            SharedCachedCredentialProvider::new(inner).with_retry_params(3, Duration::from_millis(1));
        let err = p.provide_credential(&Context::new()).await.unwrap_err();
        assert!(is_rate_limited(&err), "final error should be the 429");
        assert_eq!(count(&calls), 3);
    }

    /// The core of the fix: the cold-start herd that produced 4,482 errors in
    /// 3.4 s must collapse into ONE request at the agent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_cold_start_callers_make_one_agent_call() {
        let (inner, calls) = agent(0);
        let p = Arc::new(
            SharedCachedCredentialProvider::new(inner).with_retry_params(5, Duration::from_millis(1)),
        );

        let mut handles = Vec::new();
        for _ in 0..32 {
            let p = p.clone();
            handles.push(tokio::spawn(async move {
                p.provide_credential(&Context::new()).await.unwrap()
            }));
        }
        for h in handles {
            assert!(h.await.unwrap().is_some());
        }
        assert_eq!(
            count(&calls),
            1,
            "32 concurrent signers must coalesce into a single agent request"
        );
    }

    #[tokio::test]
    async fn a_credential_inside_the_buffer_is_not_served_from_cache() {
        // TTL inside the 120 s buffer is never usable, so each call must
        // refresh — and must do so exactly once per call, not per caller.
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let inner = FlakyAgent {
            calls: calls.clone(),
            fail_times: 0,
            ttl: Some(Duration::from_secs(30)),
        };
        let p =
            SharedCachedCredentialProvider::new(inner).with_retry_params(5, Duration::from_millis(1));
        p.provide_credential(&Context::new()).await.unwrap();
        p.provide_credential(&Context::new()).await.unwrap();
        assert_eq!(count(&calls), 2);
    }
}
