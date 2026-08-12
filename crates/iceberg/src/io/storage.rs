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

#[cfg(any(
    feature = "storage-s3",
    feature = "storage-gcs",
    feature = "storage-oss",
    feature = "storage-azdls",
))]
use std::sync::Arc;
#[cfg(feature = "storage-s3")]
use std::{collections::HashMap, sync::RwLock};

use opendal::layers::RetryLayer;
#[cfg(feature = "storage-azdls")]
use opendal::services::AzdlsConfig;
#[cfg(feature = "storage-gcs")]
use opendal::services::GcsConfig;
#[cfg(feature = "storage-oss")]
use opendal::services::OssConfig;
#[cfg(feature = "storage-s3")]
use opendal::services::S3Config;
use opendal::Operator;

#[cfg(feature = "storage-azdls")]
use super::AzureStorageScheme;
use super::FileIOBuilder;
#[cfg(feature = "storage-s3")]
use crate::{Error, ErrorKind};

/// The storage carries all supported storage services in iceberg
#[derive(Debug)]
pub(crate) enum Storage {
    #[cfg(feature = "storage-memory")]
    Memory(Operator),
    #[cfg(feature = "storage-fs")]
    LocalFs,
    /// Expects paths of the form `s3[a]://<bucket>/<path>`.
    ///
    /// In opendal 0.55 we used to inject a `CustomAwsCredentialLoader` here
    /// to drive EKS Pod Identity token refresh. opendal 0.57 ships a
    /// `credential_provider_chain` natively (IRSA, EKS Pod Identity, EC2
    /// instance metadata, env vars — all auto-refreshing), so this struct
    /// no longer needs the loader field.
    ///
    /// `operators` caches one built `Operator` per bucket. Rebuilding an
    /// Operator every call re-constructs opendal's provider chain, which
    /// re-probes the pod-identity / IMDS endpoints on cold cache and
    /// defeats the intra-Operator credential TTL. On sri-olly 2026-08-12,
    /// a synchronized metrics_1m tumble minted hundreds of chains inside
    /// one wall-clock ms and drove `169.254.170.23` into 429, cascading
    /// into wedged sink checkpoints. `RwLock` (not `Mutex`) so hot-path
    /// gets are contention-free; misses take the write lock briefly to
    /// insert. Keyed by bucket because a single `Storage::S3` variant
    /// is API-allowed to serve multiple buckets even though laminar's
    /// production path is single-bucket per catalog.
    #[cfg(feature = "storage-s3")]
    S3 {
        /// s3 storage could have `s3://` and `s3a://`.
        /// Storing the scheme string here to return the correct path.
        configured_scheme: String,
        config: Arc<S3Config>,
        operators: Arc<RwLock<HashMap<String, Operator>>>,
    },
    #[cfg(feature = "storage-gcs")]
    Gcs { config: Arc<GcsConfig> },
    #[cfg(feature = "storage-oss")]
    Oss { config: Arc<OssConfig> },
    /// Expects paths of the form
    /// `abfs[s]://<filesystem>@<account>.dfs.<endpoint-suffix>/<path>` or
    /// `wasb[s]://<container>@<account>.blob.<endpoint-suffix>/<path>`.
    #[cfg(feature = "storage-azdls")]
    Azdls {
        /// Because Azdls accepts multiple possible schemes, we store the full
        /// passed scheme here to later validate schemes passed via paths.
        configured_scheme: AzureStorageScheme,
        config: Arc<AzdlsConfig>,
    },
}

impl Storage {
    /// Convert iceberg config to opendal config.
    pub(crate) fn build(file_io_builder: FileIOBuilder) -> crate::Result<Self> {
        let (scheme_str, props, extensions) = file_io_builder.into_parts();
        let _ = (&props, &extensions);
        // opendal 0.57 dropped the top-level `Scheme` enum; we now dispatch
        // directly on the scheme string. Aliases (`s3`/`s3a`,
        // `abfs[s]`/`wasb[s]`) are recognised in the match arms.
        let normalized = Self::normalize_scheme(&scheme_str)?;

        match normalized.as_str() {
            #[cfg(feature = "storage-memory")]
            "memory" => Ok(Self::Memory(super::memory_config_build()?)),
            #[cfg(feature = "storage-fs")]
            "fs" => Ok(Self::LocalFs),
            #[cfg(feature = "storage-s3")]
            "s3" => Ok(Self::S3 {
                configured_scheme: scheme_str,
                config: super::s3_config_parse(props)?.into(),
                operators: Arc::new(RwLock::new(HashMap::new())),
            }),
            #[cfg(feature = "storage-gcs")]
            "gcs" => Ok(Self::Gcs {
                config: super::gcs_config_parse(props)?.into(),
            }),
            #[cfg(feature = "storage-oss")]
            "oss" => Ok(Self::Oss {
                config: super::oss_config_parse(props)?.into(),
            }),
            #[cfg(feature = "storage-azdls")]
            "azdls" => {
                let scheme = scheme_str.parse::<AzureStorageScheme>()?;
                Ok(Self::Azdls {
                    config: super::azdls_config_parse(props)?.into(),
                    configured_scheme: scheme,
                })
            }
            // Update doc on [`FileIO`] when adding new schemes.
            _ => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                format!("Constructing file io from scheme: {normalized} not supported now",),
            )),
        }
    }

    /// Creates operator from path.
    ///
    /// # Arguments
    ///
    /// * path: It should be *absolute* path starting with scheme string used to construct [`FileIO`].
    ///
    /// # Returns
    ///
    /// The return value consists of two parts:
    ///
    /// * An [`opendal::Operator`] instance used to operate on file.
    /// * Relative path to the root uri of [`opendal::Operator`].
    pub(crate) fn create_operator<'a>(
        &self,
        path: &'a impl AsRef<str>,
    ) -> crate::Result<(Operator, &'a str)> {
        let path = path.as_ref();
        let _ = path;
        let (operator, relative_path): (Operator, &str) = match self {
            #[cfg(feature = "storage-memory")]
            Storage::Memory(op) => {
                if let Some(stripped) = path.strip_prefix("memory:/") {
                    Ok::<_, crate::Error>((op.clone(), stripped))
                } else {
                    Ok::<_, crate::Error>((op.clone(), &path[1..]))
                }
            }
            #[cfg(feature = "storage-fs")]
            Storage::LocalFs => {
                let op = super::fs_config_build()?;

                if let Some(stripped) = path.strip_prefix("file:/") {
                    Ok::<_, crate::Error>((op, stripped))
                } else {
                    Ok::<_, crate::Error>((op, &path[1..]))
                }
            }
            #[cfg(feature = "storage-s3")]
            Storage::S3 {
                configured_scheme,
                config,
                operators,
            } => {
                // Extract bucket from `s3://<bucket>/...` (or `s3a://...`)
                // without building a fresh Operator just to read `.info()`.
                // Parsing here matches what `s3_config_build` does internally.
                let bucket = super::s3_bucket_from_path(path)?;

                // Fast path: cache hit — clone the Arc-backed Operator.
                // `Operator` is cheaply cloneable (internal Arc).
                let op = {
                    let guard = operators.read().map_err(|_| {
                        Error::new(
                            ErrorKind::Unexpected,
                            "S3 operator cache RwLock poisoned",
                        )
                    })?;
                    guard.get(&bucket).cloned()
                };

                let op = match op {
                    Some(op) => op,
                    None => {
                        // Slow path: build once, install under write lock.
                        // A second concurrent inserter is possible but
                        // benign — the newer build is functionally
                        // equivalent and gets discarded when we take the
                        // existing entry.
                        let built = super::s3_config_build(config, path)?;
                        let mut guard = operators.write().map_err(|_| {
                            Error::new(
                                ErrorKind::Unexpected,
                                "S3 operator cache RwLock poisoned",
                            )
                        })?;
                        guard.entry(bucket.clone()).or_insert(built).clone()
                    }
                };

                let prefix = format!("{}://{}/", configured_scheme, bucket);
                if path.starts_with(&prefix) {
                    Ok((op, &path[prefix.len()..]))
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid s3 url: {path}, should start with {prefix}"),
                    ))
                }
            }
            #[cfg(feature = "storage-gcs")]
            Storage::Gcs { config } => {
                let operator = super::gcs_config_build(config, path)?;
                let prefix = format!("gs://{}/", operator.info().name());
                if path.starts_with(&prefix) {
                    Ok((operator, &path[prefix.len()..]))
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid gcs url: {path}, should start with {prefix}"),
                    ))
                }
            }
            #[cfg(feature = "storage-oss")]
            Storage::Oss { config } => {
                let op = super::oss_config_build(config, path)?;

                // Check prefix of oss path.
                let prefix = format!("oss://{}/", op.info().name());
                if path.starts_with(&prefix) {
                    Ok((op, &path[prefix.len()..]))
                } else {
                    Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Invalid oss url: {path}, should start with {prefix}"),
                    ))
                }
            }
            #[cfg(feature = "storage-azdls")]
            Storage::Azdls {
                configured_scheme,
                config,
            } => super::azdls_create_operator(path, config, configured_scheme),
            #[cfg(all(
                not(feature = "storage-s3"),
                not(feature = "storage-fs"),
                not(feature = "storage-gcs"),
                not(feature = "storage-oss"),
                not(feature = "storage-azdls"),
            ))]
            _ => Err(Error::new(
                ErrorKind::FeatureUnsupported,
                "No storage service has been enabled",
            )),
        }?;

        // Transient errors are common for object stores; however there's no
        // harm in retrying temporary failures for other storage backends as well.
        let operator = operator.layer(RetryLayer::new());

        // Optionally stack the local NVMe write-through cache (merge-on-write
        // read acceleration). No-op unless LAMINAR_LOCAL_CACHE_ENABLE=1, in
        // which case the returned operator is byte-identical to the above.
        let operator = super::write_through_cache::maybe_wrap(operator);

        Ok((operator, relative_path))
    }

    /// Normalize a user-facing scheme string (e.g. `s3a`, `abfss`) into the
    /// canonical opendal-service identifier (`s3`, `azdls`, ...). Returns
    /// the canonical name on success; an unknown scheme is surfaced as
    /// `FeatureUnsupported` via the caller's match.
    fn normalize_scheme(scheme: &str) -> crate::Result<String> {
        let canon = match scheme {
            "memory" => "memory",
            "file" | "" => "fs",
            "s3" | "s3a" => "s3",
            "gs" | "gcs" => "gcs",
            "oss" => "oss",
            "abfss" | "abfs" | "wasbs" | "wasb" => "azdls",
            other => other,
        };
        Ok(canon.to_string())
    }

    /// Number of distinct buckets currently cached under `Storage::S3`.
    /// Test-only introspection so cache-hit vs cache-insert behavior can
    /// be asserted without going through `.info().name()`.
    #[cfg(all(test, feature = "storage-s3"))]
    fn s3_cache_len(&self) -> Option<usize> {
        match self {
            Storage::S3 { operators, .. } => Some(operators.read().unwrap().len()),
            _ => None,
        }
    }
}

#[cfg(all(test, feature = "storage-s3"))]
mod tests {
    use super::*;
    use crate::io::FileIOBuilder;

    fn build_s3_storage() -> Storage {
        // Minimal config; region + bucket-in-URL are enough to exercise
        // create_operator's cache path without hitting the network.
        let builder = FileIOBuilder::new("s3")
            .with_prop("s3.region", "us-east-1")
            .with_prop("s3.access-key-id", "test-key")
            .with_prop("s3.secret-access-key", "test-secret")
            .with_prop("s3.disable-ec2-metadata", "true");
        Storage::build(builder).expect("storage build")
    }

    #[test]
    fn s3_create_operator_caches_same_bucket() {
        let storage = build_s3_storage();
        assert_eq!(storage.s3_cache_len(), Some(0), "cache starts empty");

        // Two calls with same bucket → one cache entry, second is a hit.
        let _ = storage
            .create_operator(&"s3://my-bucket/data/foo.parquet".to_string())
            .expect("first call");
        assert_eq!(storage.s3_cache_len(), Some(1), "one bucket inserted");

        let _ = storage
            .create_operator(&"s3://my-bucket/data/bar.parquet".to_string())
            .expect("second call same bucket");
        assert_eq!(
            storage.s3_cache_len(),
            Some(1),
            "same bucket must not double-insert (this is the sri-olly 429 fix)"
        );
    }

    #[test]
    fn s3_create_operator_caches_per_bucket() {
        let storage = build_s3_storage();

        for bucket in ["alpha", "beta", "gamma"] {
            let path = format!("s3://{bucket}/x.parquet");
            let _ = storage.create_operator(&path).expect("call");
        }

        assert_eq!(
            storage.s3_cache_len(),
            Some(3),
            "each distinct bucket gets its own cached Operator"
        );
    }

    #[test]
    fn s3_create_operator_accepts_s3a_scheme() {
        // The `s3a` alias goes through the same cache as `s3://`.
        // configured_scheme is `s3a` so the returned relative-path prefix
        // check uses the alias, but the cache key is still the bucket.
        let builder = FileIOBuilder::new("s3a")
            .with_prop("s3.region", "us-east-1")
            .with_prop("s3.access-key-id", "test-key")
            .with_prop("s3.secret-access-key", "test-secret")
            .with_prop("s3.disable-ec2-metadata", "true");
        let storage = Storage::build(builder).unwrap();

        let path = "s3a://bucket-x/dir/f.parquet".to_string();
        let (_op, rel) = storage.create_operator(&path).expect("s3a call");
        assert_eq!(rel, "dir/f.parquet");
        assert_eq!(storage.s3_cache_len(), Some(1));
    }
}
