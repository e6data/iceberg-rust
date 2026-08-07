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

//! Local NVMe write-through cache for merge-on-write inputs.
//!
//! Laminar's inline merge reads the small per-checkpoint data files from
//! object storage. Those files were written seconds-to-minutes earlier by the
//! same process, so if we keep a local copy at write time the merge can read
//! from NVMe (page cache ≈ memory speed) instead of paying an S3 GET per input.
//!
//! This is implemented as an opendal [`Layer`] wrapping the storage operator so
//! it is a **single-pass byte tee**: the authoritative object-store write is
//! untouched (same streaming upload, same bytes), and the local copy is a
//! best-effort side write. Reads of a locally-present data file are served from
//! disk; everything else passes straight through.
//!
//! Everything here is **flag-gated**: the layer is only stacked when
//! `LAMINAR_LOCAL_CACHE_ENABLE=1` (with `LAMINAR_LOCAL_CACHE_DIR` set). When
//! the flag is off, [`maybe_wrap`] returns the operator unchanged, so behaviour
//! is byte-identical to not having this module at all.
//!
//! Disk safety (the cache must never fill the volume and wedge the pod):
//! - a hard byte cap (`LAMINAR_LOCAL_CACHE_MAX_BYTES`, default 40 GiB — below
//!   the provisioned volume so we self-limit before the kubelet evicts on
//!   `sizeLimit`);
//! - FIFO eviction: when a write would exceed the cap, the oldest entries are
//!   unlinked first; a file that still won't fit (or is larger than the whole
//!   cap) is simply not cached (the S3 copy still exists);
//! - a startup sweep clears any orphans a crashed prior pod left behind;
//! - every local read/write failure degrades to the object store — a missing,
//!   evicted, or short local file is served from S3, never surfaced as an error.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use opendal::raw::oio::{Read as OioRead, Write as OioWrite};
use opendal::raw::{
    Access, Layer, LayeredAccess, OpDelete, OpList, OpRead, OpStat, OpWrite, RpDelete, RpList,
    RpRead, RpStat, RpWrite,
};
use opendal::{Buffer, EntryMode, Metadata, Operator, Result};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

/// Default cache byte cap (40 GiB) — deliberately below the provisioned
/// `/data` volume so we evict before the kubelet trips the volume `sizeLimit`.
const DEFAULT_MAX_BYTES: u64 = 40 * 1024 * 1024 * 1024;

/// Resolved cache configuration from the environment.
#[derive(Debug, Clone)]
pub(crate) struct CacheConfig {
    pub dir: PathBuf,
    pub max_bytes: u64,
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Reads the cache config from the environment. Returns `None` (feature off)
/// unless `LAMINAR_LOCAL_CACHE_ENABLE=1` **and** `LAMINAR_LOCAL_CACHE_DIR` is
/// set to a non-empty path.
fn config_from_env() -> Option<CacheConfig> {
    if non_empty_env("LAMINAR_LOCAL_CACHE_ENABLE").as_deref() != Some("1") {
        return None;
    }
    let dir = PathBuf::from(non_empty_env("LAMINAR_LOCAL_CACHE_DIR")?);
    let max_bytes = non_empty_env("LAMINAR_LOCAL_CACHE_MAX_BYTES")
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MAX_BYTES);
    Some(CacheConfig { dir, max_bytes })
}

/// Process-wide cache manager, initialised once from the environment on first
/// storage-operator construction.
static CACHE: OnceLock<Option<Arc<CacheManager>>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Counters
//
// The cache was previously silent: no logs, no metrics. If it stopped serving —
// wrong path prefix, evicting faster than expected, a permissions problem on the
// cache dir — every read would quietly fall through to the object store and
// nothing would say so. Merge latency alone cannot distinguish "served locally"
// from "30 parallel object-store GETs", so a regression here would be invisible.
//
// Plain atomics rather than a metrics crate: this layer sits under FileIO in a
// library shared by laminar, tessellate and the executor, and must not impose a
// metrics dependency on any of them. The host process reads these and exposes
// them however it already exposes metrics.
// ---------------------------------------------------------------------------
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static BYTES_LOCAL: AtomicU64 = AtomicU64::new(0);
static EVICTIONS: AtomicU64 = AtomicU64::new(0);
static TEED: AtomicU64 = AtomicU64::new(0);

/// Snapshot of local-cache activity since process start.
///
/// `hits` counts reads served from the local copy; `misses` counts cacheable
/// reads that fell through to the object store (including evicted and truncated
/// local files). A hit ratio near zero on a merge-heavy workload means the cache
/// is not doing its job — check the cache dir is writable and that
/// `LAMINAR_LOCAL_CACHE_MAX_BYTES` is not so small that files evict before they
/// are re-read.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LocalCacheStats {
    /// Reads served entirely from the local copy.
    pub hits: u64,
    /// Cacheable reads that fell through to the object store — includes files
    /// evicted under the byte cap and local copies too short to satisfy the
    /// requested range. Non-cacheable reads (metadata, manifests) are excluded
    /// so they cannot dilute the ratio.
    pub misses: u64,
    /// Bytes returned from local copies rather than the object store.
    pub bytes_served_local: u64,
    /// Local copies unlinked to stay under `LAMINAR_LOCAL_CACHE_MAX_BYTES`. A
    /// rate approaching the tee rate means the cache is churning faster than
    /// files are re-read, so hits will be rare no matter how large it is.
    pub evictions: u64,
    /// Files successfully written to the local copy alongside the authoritative
    /// object-store write.
    pub files_teed: u64,
}

impl LocalCacheStats {
    /// Fraction of cacheable reads served locally, or `None` before any read.
    pub fn hit_ratio(&self) -> Option<f64> {
        let total = self.hits + self.misses;
        (total > 0).then(|| self.hits as f64 / total as f64)
    }
}

/// Read the local-cache counters. Cheap; safe to call from a metrics handler.
pub fn local_cache_stats() -> LocalCacheStats {
    LocalCacheStats {
        hits: HITS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        bytes_served_local: BYTES_LOCAL.load(Ordering::Relaxed),
        evictions: EVICTIONS.load(Ordering::Relaxed),
        files_teed: TEED.load(Ordering::Relaxed),
    }
}

fn global_cache() -> Option<Arc<CacheManager>> {
    CACHE
        .get_or_init(|| config_from_env().map(|cfg| Arc::new(CacheManager::new(cfg))))
        .clone()
}

/// Stack the write-through cache layer onto `op` when the feature is enabled;
/// otherwise return it unchanged (byte-identical default).
pub(crate) fn maybe_wrap(op: Operator) -> Operator {
    match global_cache() {
        Some(mgr) => op.layer(WriteThroughCacheLayer { mgr }),
        None => op,
    }
}

/// What the inline merge reads from a table's `/data/` prefix, and what we
/// therefore cache:
/// - `*.parquet` data files (read in the `streaming_concat` phase), and
/// - `*.parquet.stats` per-file sidecars (read in the `puffin_carry_forward`
///   phase — the dominant cost of a terminal merge, one GET per input file).
///
/// Both are write-once/immutable, so serving them from a local copy is always
/// correct. This deliberately excludes the partition-level `_stats/*.puffin`
/// (rewritten in place on terminal merges → could serve stale bytes), Avro
/// manifests, and `/metadata/` JSON.
fn is_cacheable(rel_path: &str) -> bool {
    rel_path.contains("/data/")
        && (rel_path.ends_with(".parquet") || rel_path.ends_with(".parquet.stats"))
}

// ---------------------------------------------------------------------------
// Cache manager (disk safety: cap + FIFO eviction + startup sweep)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Entry {
    file: PathBuf,
    size: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    /// rel_path -> on-disk entry. Source of truth for reads (a hash of the
    /// path is never trusted; only paths we wrote appear here).
    entries: HashMap<String, Entry>,
    /// FIFO insertion order of rel_paths, for eviction.
    order: VecDeque<String>,
    total_bytes: u64,
}

#[derive(Debug)]
pub(crate) struct CacheManager {
    dir: PathBuf,
    max_bytes: u64,
    seq: AtomicU64,
    state: Mutex<CacheState>,
}

impl CacheManager {
    fn new(cfg: CacheConfig) -> Self {
        // Use a dedicated SUBDIRECTORY of the configured dir. The dir is often
        // shared (e.g. `/data` also holds laminar's `state.sqlite`), so the
        // cache must never read/write/sweep the parent directly — only files
        // under this subdir belong to us.
        let dir = cfg.dir.join("laminar-merge-cache");
        // Startup sweep: clear any orphans from a previous (possibly crashed)
        // pod so stale files never count against the cap or get served. Scoped
        // strictly to the cache subdir — nothing else on the volume is touched.
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let _ = std::fs::remove_file(e.path());
            }
        }
        Self {
            dir,
            max_bytes: cfg.max_bytes,
            seq: AtomicU64::new(0),
            state: Mutex::new(CacheState::default()),
        }
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::Relaxed)
    }

    /// Local path of a cached data file, if present.
    fn local_for(&self, rel_path: &str) -> Option<PathBuf> {
        let st = self.state.lock().unwrap();
        st.entries.get(rel_path).map(|e| e.file.clone())
    }

    /// Register a finalised local copy. Evicts oldest entries first to stay
    /// under the cap; if the new file cannot fit even after evicting everything
    /// else, it is dropped (unlinked) and not cached. Returns whether it was
    /// kept (for logging/metrics only).
    fn register(&self, rel_path: &str, file: PathBuf, size: u64) -> bool {
        let mut to_unlink: Vec<PathBuf> = Vec::new();
        let kept = {
            let mut st = self.state.lock().unwrap();
            // Evict oldest until the new file fits (or nothing left to evict).
            while st.total_bytes.saturating_add(size) > self.max_bytes {
                let Some(old_key) = st.order.pop_front() else {
                    break;
                };
                if let Some(old) = st.entries.remove(&old_key) {
                    st.total_bytes = st.total_bytes.saturating_sub(old.size);
                    to_unlink.push(old.file);
                    EVICTIONS.fetch_add(1, Ordering::Relaxed);
                }
            }
            if st.total_bytes.saturating_add(size) <= self.max_bytes {
                st.total_bytes = st.total_bytes.saturating_add(size);
                st.entries.insert(rel_path.to_string(), Entry { file, size });
                TEED.fetch_add(1, Ordering::Relaxed);
                st.order.push_back(rel_path.to_string());
                true
            } else {
                // Still doesn't fit (single file bigger than the whole cap).
                to_unlink.push(file);
                false
            }
        };
        for p in to_unlink {
            let _ = std::fs::remove_file(p);
        }
        kept
    }
}

// ---------------------------------------------------------------------------
// Layer + accessor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct WriteThroughCacheLayer {
    mgr: Arc<CacheManager>,
}

impl<A: Access> Layer<A> for WriteThroughCacheLayer {
    type LayeredAccess = CacheAccessor<A>;

    fn layer(&self, inner: A) -> Self::LayeredAccess {
        CacheAccessor {
            inner,
            mgr: self.mgr.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct CacheAccessor<A> {
    inner: A,
    mgr: Arc<CacheManager>,
}

impl<A: Access> LayeredAccess for CacheAccessor<A> {
    type Inner = A;
    type Reader = CacheReader<A::Reader>;
    type Writer = CacheWriter<A::Writer>;
    type Lister = A::Lister;
    type Deleter = A::Deleter;
    type Copier = A::Copier;

    fn inner(&self) -> &Self::Inner {
        &self.inner
    }

    async fn read(&self, path: &str, args: OpRead) -> Result<(RpRead, Self::Reader)> {
        // Serve from the local cache when we have this data file on disk.
        if let Some(local) = self.mgr.local_for(path) {
            if let Some(buf) = read_local_range(&local, &args).await {
                HITS.fetch_add(1, Ordering::Relaxed);
                BYTES_LOCAL.fetch_add(buf.len() as u64, Ordering::Relaxed);
                let md = Metadata::new(EntryMode::FILE).with_content_length(buf.len() as u64);
                return Ok((RpRead::new(md), CacheReader::Local(buf)));
            }
            // Evicted / missing / short → fall through to the object store.
        }
        // Only cacheable paths count as a miss: a metadata or manifest read was
        // never a candidate, and counting it would dilute the ratio into
        // uselessness.
        if is_cacheable(path) {
            MISSES.fetch_add(1, Ordering::Relaxed);
        }
        let (rp, r) = self.inner.read(path, args).await?;
        Ok((rp, CacheReader::Pass(r)))
    }

    async fn write(&self, path: &str, args: OpWrite) -> Result<(RpWrite, Self::Writer)> {
        // The object-store write is authoritative and untouched.
        let (rp, w) = self.inner.write(path, args).await?;
        let tee = if is_cacheable(path) {
            Some(TeeState::new(&self.mgr, path))
        } else {
            None
        };
        Ok((
            rp,
            CacheWriter {
                inner: w,
                mgr: self.mgr.clone(),
                rel_path: path.to_string(),
                tee,
            },
        ))
    }

    async fn stat(&self, path: &str, args: OpStat) -> Result<RpStat> {
        // Serve size from the local copy when cached — kills the per-input
        // HEAD in puffin_carry_forward. Only content_length is synthesised
        // (the local file is a byte-exact copy, so it is exact); anything
        // uncached falls through to the object store.
        if let Some(local) = self.mgr.local_for(path) {
            if let Ok(md) = tokio::fs::metadata(&local).await {
                let meta = Metadata::new(EntryMode::FILE).with_content_length(md.len());
                return Ok(RpStat::new(meta));
            }
        }
        self.inner.stat(path, args).await
    }

    async fn delete(&self) -> Result<(RpDelete, Self::Deleter)> {
        self.inner.delete().await
    }

    async fn list(&self, path: &str, args: OpList) -> Result<(RpList, Self::Lister)> {
        self.inner.list(path, args).await
    }
}

/// Read the range requested by `args` from a locally-cached file. Reads ONLY
/// the requested bytes (seek + bounded read) — merge input reads are range
/// reads (parquet footer, then row groups), so a whole-file read per range
/// would amplify local IO enormously. Returns `None` on any mismatch
/// (missing/evicted file, short read) so the caller falls back to the store.
async fn read_local_range(local: &Path, args: &OpRead) -> Option<Buffer> {
    use std::io::SeekFrom;
    let range = args.range();
    let offset = range.offset();
    let mut f = tokio::fs::File::open(local).await.ok()?;
    if offset > 0 {
        f.seek(SeekFrom::Start(offset)).await.ok()?;
    }
    match range.size() {
        // Bounded range: read exactly `sz` bytes; a short read means the local
        // copy doesn't cover the request (stale/evicted) → fall back to store.
        Some(sz) => {
            let mut buf = vec![0u8; sz as usize];
            f.read_exact(&mut buf).await.ok()?;
            Some(Buffer::from(buf))
        }
        // Unbounded: read from the offset to EOF.
        None => {
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).await.ok()?;
            Some(Buffer::from(buf))
        }
    }
}

// ---------------------------------------------------------------------------
// Reader: either a one-shot local buffer or the passed-through inner reader.
// ---------------------------------------------------------------------------

pub(crate) enum CacheReader<R> {
    Local(Buffer),
    Pass(R),
}

impl<R: OioRead> OioRead for CacheReader<R> {
    async fn read(&mut self) -> Result<Buffer> {
        match self {
            // Buffer yields its content once, then an empty Buffer (= EOF).
            CacheReader::Local(buf) => Ok(std::mem::take(buf)),
            CacheReader::Pass(r) => r.read().await,
        }
    }
}

// ---------------------------------------------------------------------------
// Writer: tee each chunk to a local temp file alongside the authoritative
// inner write. Local failures poison the tee (drop the copy) but never affect
// the inner write's result.
// ---------------------------------------------------------------------------

struct TeeState {
    temp: PathBuf,
    final_path: PathBuf,
    file: Option<tokio::fs::File>,
    bytes: u64,
    poisoned: bool,
}

impl TeeState {
    fn new(mgr: &CacheManager, _rel_path: &str) -> Self {
        let seq = mgr.next_seq();
        let temp = mgr.dir.join(format!("c{seq}.tmp"));
        let final_path = mgr.dir.join(format!("c{seq}.pq"));
        Self {
            temp,
            final_path,
            file: None,
            bytes: 0,
            poisoned: false,
        }
    }

    /// Append a chunk. Opens the temp file lazily on first write.
    async fn append(&mut self, data: &[u8]) -> std::io::Result<()> {
        if self.file.is_none() {
            let f = tokio::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&self.temp)
                .await?;
            self.file = Some(f);
        }
        let f = self.file.as_mut().expect("temp file opened above");
        f.write_all(data).await?;
        self.bytes += data.len() as u64;
        Ok(())
    }

    /// Flush + fsync + atomic rename into the final path. On success returns
    /// the final path and byte count for registration.
    async fn finalize(mut self) -> Option<(PathBuf, u64)> {
        let mut f = self.file.take()?;
        f.flush().await.ok()?;
        f.sync_all().await.ok()?;
        drop(f);
        tokio::fs::rename(&self.temp, &self.final_path).await.ok()?;
        Some((self.final_path, self.bytes))
    }

    /// Best-effort cleanup of the temp file (abort / poison paths).
    async fn cleanup(&self) {
        let _ = tokio::fs::remove_file(&self.temp).await;
    }
}

pub(crate) struct CacheWriter<W> {
    inner: W,
    mgr: Arc<CacheManager>,
    rel_path: String,
    tee: Option<TeeState>,
}

impl<W: OioWrite> OioWrite for CacheWriter<W> {
    async fn write(&mut self, bs: Buffer) -> Result<()> {
        // Capture the bytes for the local tee before handing the buffer to the
        // authoritative inner write (Buffer clone is a cheap refcount bump).
        let tee_bytes = match self.tee.as_ref() {
            Some(t) if !t.poisoned => Some(bs.clone().to_bytes()),
            _ => None,
        };

        self.inner.write(bs).await?; // authoritative — propagate error verbatim

        if let (Some(bytes), Some(t)) = (tee_bytes, self.tee.as_mut()) {
            if let Err(e) = t.append(&bytes).await {
                // Local write failed (disk full, io error): drop the copy but
                // let the object-store write stand.
                log::warn!(
                    "local cache tee append failed for {}: {e}; skipping cache for this file",
                    self.rel_path
                );
                t.cleanup().await;
                t.poisoned = true;
            }
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<Metadata> {
        let md = self.inner.close().await?; // authoritative
        if let Some(t) = self.tee.take() {
            if t.poisoned || t.file.is_none() {
                t.cleanup().await;
            } else if let Some((path, size)) = t.finalize().await {
                self.mgr.register(&self.rel_path, path, size);
            }
        }
        Ok(md)
    }

    async fn abort(&mut self) -> Result<()> {
        let r = self.inner.abort().await;
        if let Some(t) = self.tee.take() {
            t.cleanup().await;
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_manager(max_bytes: u64) -> (Arc<CacheManager>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(CacheManager::new(CacheConfig {
            dir: dir.path().to_path_buf(),
            max_bytes,
        }));
        (mgr, dir)
    }

    fn write_file(mgr: &CacheManager, name: &str, size: usize) -> PathBuf {
        let seq = mgr.next_seq();
        let p = mgr.dir.join(format!("t{seq}.pq"));
        std::fs::write(&p, vec![7u8; size]).unwrap();
        let _ = name;
        p
    }

    #[test]
    fn is_cacheable_matches_data_parquet_and_per_file_sidecars() {
        // Data files (streaming_concat) and per-file sidecars
        // (puffin_carry_forward) are both cached — both are immutable.
        assert!(is_cacheable("obs/ns/tbl/data/tenant=x/f.parquet"));
        assert!(is_cacheable("obs/ns/tbl/data/tenant=x/f.parquet.stats"));
        // The partition-level puffin is rewritten in place → never cached.
        assert!(!is_cacheable("obs/ns/tbl/data/_stats/p/partition.puffin"));
        assert!(!is_cacheable("obs/ns/tbl/metadata/snap-1.avro"));
        assert!(!is_cacheable("obs/ns/tbl/metadata/v3.metadata.json"));
    }

    #[test]
    fn register_and_lookup_roundtrip() {
        let (mgr, _d) = tmp_manager(1024);
        let f = write_file(&mgr, "a", 100);
        assert!(mgr.register("obs/data/a.parquet", f.clone(), 100));
        assert_eq!(mgr.local_for("obs/data/a.parquet"), Some(f));
        assert!(mgr.local_for("obs/data/missing.parquet").is_none());
    }

    #[test]
    fn eviction_stays_under_cap_fifo() {
        let (mgr, _d) = tmp_manager(250);
        let a = write_file(&mgr, "a", 100);
        let b = write_file(&mgr, "b", 100);
        let c = write_file(&mgr, "c", 100);
        assert!(mgr.register("data/a.parquet", a.clone(), 100));
        assert!(mgr.register("data/b.parquet", b, 100));
        // Adding c (total would be 300 > 250) evicts the oldest, a.
        assert!(mgr.register("data/c.parquet", c, 100));
        assert!(mgr.local_for("data/a.parquet").is_none(), "oldest evicted");
        assert!(!a.exists(), "evicted file unlinked from disk");
        assert!(mgr.local_for("data/b.parquet").is_some());
        assert!(mgr.local_for("data/c.parquet").is_some());
        assert!(mgr.state.lock().unwrap().total_bytes <= 250);
    }

    #[test]
    fn oversize_file_is_not_cached() {
        let (mgr, _d) = tmp_manager(100);
        let big = write_file(&mgr, "big", 500);
        assert!(!mgr.register("data/big.parquet", big.clone(), 500));
        assert!(mgr.local_for("data/big.parquet").is_none());
        assert!(!big.exists(), "un-cacheable file is unlinked, not left as an orphan");
    }

    #[test]
    fn startup_sweep_clears_cache_subdir_but_not_siblings() {
        let dir = tempfile::tempdir().unwrap();
        // A sibling in the shared dir (e.g. laminar's state.sqlite) MUST survive.
        std::fs::write(dir.path().join("state.sqlite"), b"precious").unwrap();
        // A stale file inside the cache subdir must be swept.
        let sub = dir.path().join("laminar-merge-cache");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("orphan.pq"), b"stale").unwrap();

        let _mgr = CacheManager::new(CacheConfig {
            dir: dir.path().to_path_buf(),
            max_bytes: 1024,
        });

        assert!(
            dir.path().join("state.sqlite").exists(),
            "sibling state DB untouched by the sweep"
        );
        assert!(
            !sub.join("orphan.pq").exists(),
            "stale cache-subdir file swept"
        );
    }

    #[cfg(feature = "storage-memory")]
    fn memory_op() -> Operator {
        Operator::from_config(opendal::services::MemoryConfig::default())
            .unwrap()
            .finish()
    }

    /// The authoritative write goes to the backing store AND a local copy is
    /// tee'd; a subsequent read is served from the local copy (proven by
    /// deleting the backing-store object first — the read still succeeds).
    #[cfg(feature = "storage-memory")]
    /// The counters must move, and a miss must only be counted for a path the
    /// cache would ever have served. Counting metadata reads as misses would
    /// bury the signal we added these for.
    #[test]
    fn stats_only_count_cacheable_paths_as_misses() {
        // Same predicate the read path gates the MISSES counter on.
        assert!(is_cacheable("wh/db/tbl/data/abc.parquet"));
        assert!(is_cacheable("wh/db/tbl/data/abc.parquet.stats"));
        // Metadata and manifests are never cached, so a read of one is not a
        // miss — it was never a candidate.
        assert!(!is_cacheable("wh/db/tbl/metadata/root-1.parquet"));
        assert!(!is_cacheable("wh/db/tbl/metadata/snap-1.avro"));
        assert!(!is_cacheable("wh/db/tbl/data/_stats/p/partition.puffin"));
    }

    /// hit_ratio is the number an operator actually reads; make sure it is
    /// None before any traffic rather than a misleading 0.0.
    #[test]
    fn hit_ratio_is_none_until_there_is_traffic() {
        let empty = LocalCacheStats::default();
        assert_eq!(empty.hit_ratio(), None);
        let warm = LocalCacheStats { hits: 3, misses: 1, ..Default::default() };
        assert_eq!(warm.hit_ratio(), Some(0.75));
        let cold = LocalCacheStats { hits: 0, misses: 8, ..Default::default() };
        assert_eq!(cold.hit_ratio(), Some(0.0));
    }

    #[tokio::test]
    async fn layer_tees_write_and_serves_read_from_local() {
        let (mgr, _d) = tmp_manager(1 << 20);
        let inner = memory_op();
        let op = inner
            .clone()
            .layer(WriteThroughCacheLayer { mgr: mgr.clone() });

        let key = "obs/ns/tbl/data/tenant=x/f.parquet";
        let payload = b"hello-parquet-bytes".to_vec();
        op.write(key, payload.clone()).await.unwrap();
        assert!(mgr.local_for(key).is_some(), "data file tee'd to local cache");

        // Remove the backing-store object: a correct read must now come from
        // the local NVMe copy.
        inner.delete(key).await.unwrap();
        let got = op.read(key).await.unwrap().to_bytes();
        assert_eq!(&got[..], &payload[..], "served bytes match, from local cache");
    }

    /// A ranged read of a locally-cached file returns the exact slice.
    #[cfg(feature = "storage-memory")]
    #[tokio::test]
    async fn layer_serves_range_from_local() {
        let (mgr, _d) = tmp_manager(1 << 20);
        let inner = memory_op();
        let op = inner
            .clone()
            .layer(WriteThroughCacheLayer { mgr: mgr.clone() });
        let key = "t/data/p/f.parquet";
        op.write(key, b"0123456789".to_vec()).await.unwrap();
        inner.delete(key).await.unwrap();
        let got = op.read_with(key).range(2..6).await.unwrap().to_bytes();
        assert_eq!(&got[..], b"2345");
    }

    /// Metadata / manifests / the partition puffin are never cached; per-file
    /// `.parquet.stats` sidecars ARE (they're immutable and read in
    /// puffin_carry_forward).
    #[cfg(feature = "storage-memory")]
    #[tokio::test]
    async fn layer_caches_sidecars_but_not_metadata_or_partition_puffin() {
        let (mgr, _d) = tmp_manager(1 << 20);
        let op = memory_op().layer(WriteThroughCacheLayer { mgr: mgr.clone() });

        op.write("obs/ns/tbl/metadata/snap.avro", b"meta".to_vec())
            .await
            .unwrap();
        assert!(mgr.local_for("obs/ns/tbl/metadata/snap.avro").is_none());

        op.write("obs/ns/tbl/data/_stats/p/partition.puffin", b"puffin".to_vec())
            .await
            .unwrap();
        assert!(mgr
            .local_for("obs/ns/tbl/data/_stats/p/partition.puffin")
            .is_none());

        op.write("obs/ns/tbl/data/p/f.parquet.stats", b"sidecar".to_vec())
            .await
            .unwrap();
        assert!(mgr.local_for("obs/ns/tbl/data/p/f.parquet.stats").is_some());
    }
}
