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

//! Walk the current snapshot's manifest chain and count LIVE data-file
//! entries per partition tuple.
//!
//! Answers "what does the executor actually see per (tenant × hour)" —
//! the S3 raw listing includes tombstones (post-merge originals awaiting
//! GC) that the reader never touches. Only the manifest tells the truth.
//!
//! Uses `Snapshot::load_manifest_list`, which already handles V4 tiered
//! reconstruction (root → prev_root chain / tree; inline entries; cold
//! bucket-index leaves). We then walk each ManifestFile → its entries and
//! bucket by DataFile.partition().
//!
//! Run:
//!   cargo run --example count-files-per-partition --release -- \
//!     --uri http://localhost:18181/catalog \
//!     --warehouse 65f34891-d13f-465b-ae48-3820cab8525d \
//!     --namespace observability \
//!     --table logs
//!
//! (Warehouse takes the UUID, not the human name; get it from
//! `/catalog/v1/config?warehouse=<name>` → `defaults.prefix`.)
//!
//! Requires AWS credentials in the environment so the FileIO the catalog
//! hands back can read manifest parquets from S3. On EKS pod-identity
//! setups outside the cluster, either export the STS trio or set
//! AWS_PROFILE.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::process::ExitCode;

use futures::stream::{self, StreamExt};
use iceberg::spec::root_manifest::ManifestDeleteVector;
use iceberg::spec::{Literal, ManifestStatus, PrimitiveLiteral};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableIdent};
use iceberg_catalog_rest::{REST_CATALOG_PROP_URI, REST_CATALOG_PROP_WAREHOUSE, RestCatalogBuilder};

/// Parallel manifest fetches. Manifests are small (~15-30 KB each) but
/// there can be hundreds — serial fetch on a chatty S3 gets slow. 16 keeps
/// us well under any sensible IOPS cap while cutting wall-clock ~10-15×.
const MANIFEST_FETCH_CONCURRENCY: usize = 16;

#[derive(Debug, Clone)]
struct Args {
    uri: String,
    warehouse: String,
    namespace: String,
    table: String,
    /// Opt-in: verify puffin coverage. Costs one HEAD per live data file plus
    /// one per partition, so it is off by default to keep the common path fast.
    check_stats: bool,
    /// Opt-in: dump the live data-file and manifest path sets to stdout so they
    /// can be diffed against an S3 listing. Counting objects on S3 conflates
    /// live files with superseded manifests and removed-but-not-yet-GC'd
    /// parquets (see the `removed_paths` note in main) — this emits the set the
    /// reader actually sees, so the difference is the orphan population.
    /// Costs no extra I/O; the walk already visits every live entry.
    dump_live_paths: bool,
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1).collect())
}

/// Split from [`parse_args`] so flag handling is testable without touching the
/// process environment.
fn parse_args_from(raw: Vec<String>) -> Result<Args, String> {
    let mut uri = None;
    let mut warehouse = None;
    let mut namespace = None;
    let mut table = None;
    let mut check_stats = false;
    let mut dump_live_paths = false;
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].as_str();
        // Boolean flags: consume no value, so handle before the value lookup.
        if key == "--check-stats" {
            check_stats = true;
            i += 1;
            continue;
        }
        if key == "--dump-live-paths" {
            dump_live_paths = true;
            i += 1;
            continue;
        }
        let val = raw
            .get(i + 1)
            .ok_or_else(|| format!("flag {key} missing value"))?
            .clone();
        match key {
            "--uri" => uri = Some(val),
            "--warehouse" => warehouse = Some(val),
            "--namespace" => namespace = Some(val),
            "--table" => table = Some(val),
            "-h" | "--help" => {
                return Err(
                    "usage: --uri URL --warehouse UUID --namespace NS --table T \
                     [--check-stats] [--dump-live-paths]"
                        .into(),
                );
            }
            other => return Err(format!("unknown flag: {other}")),
        }
        i += 2;
    }
    Ok(Args {
        uri: uri.ok_or("--uri required")?,
        warehouse: warehouse.ok_or("--warehouse required")?,
        namespace: namespace.ok_or("--namespace required")?,
        table: table.ok_or("--table required")?,
        check_stats,
        dump_live_paths,
    })
}

/// Render a single partition literal for display. Bytes stay hex so a
/// binary-encoded key doesn't corrupt the output.
fn render_literal(lit: &Literal) -> String {
    match lit {
        Literal::Primitive(p) => match p {
            PrimitiveLiteral::Boolean(b) => b.to_string(),
            PrimitiveLiteral::Int(v) => v.to_string(),
            PrimitiveLiteral::Long(v) => v.to_string(),
            PrimitiveLiteral::Float(v) => v.to_string(),
            PrimitiveLiteral::Double(v) => v.to_string(),
            PrimitiveLiteral::String(s) => s.clone(),
            PrimitiveLiteral::Binary(b) => format!("0x{}", hex(b)),
            other => format!("{other:?}"),
        },
        other => format!("{other:?}"),
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Turn a partition Struct into a stable "field1=val1,field2=val2,..." key.
fn partition_key(part: &iceberg::spec::Struct, field_names: &[String]) -> String {
    let fields = part.fields();
    let mut parts = Vec::with_capacity(field_names.len());
    for (i, name) in field_names.iter().enumerate() {
        let val = match fields.get(i) {
            Some(Some(lit)) => render_literal(lit),
            Some(None) => "NULL".to_string(),
            None => "?".to_string(),
        };
        parts.push(format!("{name}={val}"));
    }
    parts.join(",")
}


/// Path of a partition's consolidated puffin.
///
/// Laminar writes one per (partition tuple) under `<table>/data/_stats/`, e.g.
/// `…/data/_stats/signallake_tenant=nishant/tenant=samsung/timestamp_hour=496131/partition.puffin`.
/// Note the hour segment here is the RAW bucket integer, unlike the data-file
/// path which uses a formatted `2026-08-07-03` — so build it from the partition
/// struct rather than by rewriting a data path.
fn partition_puffin_path(
    table_location: &str,
    part: &iceberg::spec::Struct,
    field_names: &[String],
) -> String {
    let fields = part.fields();
    let mut segs = Vec::with_capacity(field_names.len());
    for (i, name) in field_names.iter().enumerate() {
        let val = match fields.get(i) {
            Some(Some(lit)) => render_literal(lit),
            Some(None) => "null".to_string(),
            None => "unknown".to_string(),
        };
        segs.push(format!("{name}={val}"));
    }
    format!(
        "{}/data/_stats/{}/partition.puffin",
        table_location.trim_end_matches('/'),
        segs.join("/")
    )
}

#[derive(Default, Debug, Clone)]
struct PartitionStats {
    live_files: u64,
    live_bytes: u64,
    live_records: u64,
    // Sorted by size for quick min/max
    file_sizes: Vec<u64>,
}

impl PartitionStats {
    fn observe(&mut self, size: u64, records: u64) {
        self.live_files += 1;
        self.live_bytes = self.live_bytes.saturating_add(size);
        self.live_records = self.live_records.saturating_add(records);
        self.file_sizes.push(size);
    }
    fn avg_bytes(&self) -> u64 {
        if self.live_files == 0 { 0 } else { self.live_bytes / self.live_files }
    }
    fn min_bytes(&self) -> u64 {
        self.file_sizes.iter().copied().min().unwrap_or(0)
    }
    fn max_bytes(&self) -> u64 {
        self.file_sizes.iter().copied().max().unwrap_or(0)
    }
}

fn human_bytes(n: u64) -> String {
    let f = n as f64;
    if f < 1024.0 {
        format!("{n} B")
    } else if f < 1024.0 * 1024.0 {
        format!("{:.1} KB", f / 1024.0)
    } else if f < 1024.0 * 1024.0 * 1024.0 {
        format!("{:.1} MB", f / (1024.0 * 1024.0))
    } else {
        format!("{:.2} GB", f / (1024.0 * 1024.0 * 1024.0))
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    // Build REST catalog. Warehouse is the UUID string (RestCatalog forwards
    // it as the `warehouse` config; lakekeeper injects the UUID as `prefix`
    // in every subsequent request).
    let catalog = match RestCatalogBuilder::default()
        .load(
            "rest",
            HashMap::from([
                (REST_CATALOG_PROP_URI.to_string(), args.uri.clone()),
                (REST_CATALOG_PROP_WAREHOUSE.to_string(), args.warehouse.clone()),
            ]),
        )
        .await
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("catalog build failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let ns = match NamespaceIdent::from_vec(vec![args.namespace.clone()]) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("bad namespace: {e}");
            return ExitCode::FAILURE;
        }
    };
    let ident = TableIdent::new(ns, args.table.clone());
    let table = match catalog.load_table(&ident).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("load_table {ident} failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    let metadata = table.metadata();
    let snapshot = match metadata.current_snapshot() {
        Some(s) => s,
        None => {
            eprintln!("table has no current snapshot");
            return ExitCode::FAILURE;
        }
    };

    // Resolve partition-field NAMES for the current spec so the key we
    // emit is human-readable ("tenant=cops-beta,timestamp_hour=495978")
    // rather than positional ("0=..,1=..,2=..").
    let spec = metadata.default_partition_spec();
    let field_names: Vec<String> = spec.fields().iter().map(|f| f.name.clone()).collect();

    println!(
        "== {}.{} ==",
        args.namespace, args.table
    );
    println!("snapshot_id      : {}", snapshot.snapshot_id());
    println!("format_version   : {:?} (effective {:?})",
        metadata.format_version(), metadata.effective_format_version());
    println!("partition_fields : {}", field_names.join(", "));
    println!("manifest_list    : {}", snapshot.manifest_list());

    let file_io = table.file_io();
    // load_manifest_list handles all V4 tiered reconstruction (delta chain,
    // tree nodes, inline entries + cold bucket-index leaves) — a single
    // call gives us every manifest that contributes to the live view.
    let manifest_list = match snapshot
        .load_manifest_list(file_io, &metadata.clone())
        .await
    {
        Ok(ml) => ml,
        Err(e) => {
            eprintln!("load_manifest_list failed: {e}");
            return ExitCode::FAILURE;
        }
    };
    // Ref-resident path tombstones. On an incremental root a REMOVED file keeps
    // manifest status Added/Existing — the removal lives in the root's
    // removed_paths, not in the entry. Counting on status alone therefore reports
    // deleted files as live: on sri-olly 2026-08-06 that inflated a hot-root
    // reading from ~1k to ~15.6k, because ~11,000 merge inputs had been removed
    // but not yet GC'd. Take a copy before `entries()` borrows the list.
    let removed_paths: std::collections::HashSet<String> =
        manifest_list.removed_paths().iter().cloned().collect();
    // Manifest delete vectors. A removal is recorded EITHER as a path tombstone
    // (above) or as an MDV bit on the manifest ref — the collapse-time sweep
    // converts the former into the latter, and the scan honours both. A walker
    // checking only `removed_paths` therefore reports MDV-suppressed files as
    // live: right after the first sweep on sri-olly that read 171,687 live files
    // against a true ~9k. Take the bitmaps before `entries()` borrows the list.
    let mdv_bitmaps: std::collections::HashMap<String, Vec<u8>> = manifest_list
        .entries()
        .iter()
        .filter_map(|mf| {
            manifest_list
                .mdv_for(&mf.manifest_path)
                .map(|b| (mf.manifest_path.clone(), b.to_vec()))
        })
        .collect();
    let manifests = manifest_list.entries();
    println!("manifest_count   : {}", manifests.len());
    println!("removed_paths    : {} (ref-resident tombstones, excluded below)", removed_paths.len());
    println!("mdv_manifests    : {} (manifests carrying a delete vector)", mdv_bitmaps.len());

    // Bucket manifests by their partition-summary hour bound (upper).
    // Multi-hour manifests (rare but legal — no partition_scoped rebalance)
    // get counted under EACH hour they touch. Missing bound → "no_hour".
    let hour_field_idx = spec
        .fields()
        .iter()
        .position(|f| matches!(f.transform, iceberg::spec::Transform::Hour));
    let mut manifests_per_hour: BTreeMap<i64, usize> = BTreeMap::new();
    let mut manifests_no_hour: usize = 0;
    // Reachable manifest paths, for the --dump-live-paths orphan diff. These are
    // the manifests the reader loads; every other *.parquet under metadata/ is
    // either a superseded root or an unreferenced leaf.
    let mut live_manifest_paths: Vec<String> = Vec::new();
    for mf in manifests {
        if args.dump_live_paths {
            live_manifest_paths.push(mf.manifest_path.clone());
        }
        let mut recorded = false;
        if let Some(idx) = hour_field_idx {
            if let Some(fs) = mf.partitions.as_ref().and_then(|p| p.get(idx)) {
                if let (Some(lo), Some(hi)) = (fs.lower_bound.as_ref(), fs.upper_bound.as_ref()) {
                    if lo.len() >= 4 && hi.len() >= 4 {
                        let lo_h = i32::from_le_bytes([lo[0], lo[1], lo[2], lo[3]]) as i64;
                        let hi_h = i32::from_le_bytes([hi[0], hi[1], hi[2], hi[3]]) as i64;
                        for h in lo_h..=hi_h {
                            *manifests_per_hour.entry(h).or_insert(0) += 1;
                        }
                        recorded = true;
                    }
                }
            }
        }
        if !recorded {
            manifests_no_hour += 1;
        }
    }

    // Fan-out the per-manifest loads. Each load reads a small parquet
    // (~15-30 KB) but there can be hundreds; serial is painfully slow.
    let loaded = stream::iter(manifests.iter().cloned())
        .map(|mf| {
            let io = file_io.clone();
            async move {
                match mf.load_manifest(&io).await {
                    Ok(m) => Some((mf.manifest_path.clone(), m)),
                    Err(e) => {
                        eprintln!("manifest load failed for {}: {e}", mf.manifest_path);
                        None
                    }
                }
            }
        })
        .buffer_unordered(MANIFEST_FETCH_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    let mut by_partition: BTreeMap<String, PartitionStats> = BTreeMap::new();
    // Only populated under --check-stats or --dump-live-paths; keeps the
    // default path allocation-free.
    let mut live_file_paths: Vec<String> = Vec::new();
    let mut live_partition_puffins: std::collections::BTreeSet<String> =
        std::collections::BTreeSet::new();
    let mut total_live_files: u64 = 0;
    let mut total_live_bytes: u64 = 0;
    let mut total_live_records: u64 = 0;
    let mut total_dead: u64 = 0;
    // Manifests that still hold at least one LIVE file, keyed by hour.
    //
    // `manifests_per_hour` below is derived from partition-summary bounds
    // WITHOUT loading the manifest, so it counts a manifest whose files have all
    // been merged away, and counts a multi-hour manifest once per hour it spans.
    // Neither is planner work for that hour. This map only counts manifests that
    // would actually contribute a row.
    let mut live_manifests_per_hour: BTreeMap<i64, std::collections::HashSet<String>> =
        BTreeMap::new();
    for (manifest_path, manifest) in loaded.into_iter().flatten() {
        // This manifest's delete vector, if the root ref carries one. Decoded
        // once per manifest, then consulted per row by position — the same way
        // `scan::context` applies it.
        let mdv = match mdv_bitmaps.get(&manifest_path) {
            Some(bytes) => match ManifestDeleteVector::deserialize(bytes) {
                Ok(d) => Some(d),
                Err(e) => {
                    eprintln!("mdv decode failed for {manifest_path}: {e}");
                    None
                }
            },
            None => None,
        };
        for (idx, entry) in manifest.entries().iter().enumerate() {
            // ManifestStatus::Deleted = entry was invalidated in this
            // snapshot; the reader skips it. Added/Existing are LIVE.
            match entry.status() {
                ManifestStatus::Added | ManifestStatus::Existing => {
                    let df = entry.data_file();
                    // Status says live, but a root tombstone overrides it...
                    if removed_paths.contains(df.file_path()) {
                        total_dead += 1;
                        continue;
                    }
                    // ...and so does a delete bit for this row.
                    if mdv.as_ref().is_some_and(|d| d.is_deleted(idx as u32)) {
                        total_dead += 1;
                        continue;
                    }
                    if args.check_stats || args.dump_live_paths {
                        live_file_paths.push(df.file_path().to_string());
                    }
                    if args.check_stats {
                        live_partition_puffins.insert(partition_puffin_path(
                            metadata.location(),
                            df.partition(),
                            &field_names,
                        ));
                    }
                    if let Some(idx) = hour_field_idx {
                        if let Some(Some(lit)) = df.partition().fields().get(idx) {
                            if let Ok(h) = render_literal(lit).parse::<i64>() {
                                live_manifests_per_hour
                                    .entry(h)
                                    .or_default()
                                    .insert(manifest_path.clone());
                            }
                        }
                    }
                    let key = partition_key(df.partition(), &field_names);
                    let stats = by_partition.entry(key).or_default();
                    stats.observe(df.file_size_in_bytes(), df.record_count());
                    total_live_files += 1;
                    total_live_bytes = total_live_bytes.saturating_add(df.file_size_in_bytes());
                    total_live_records =
                        total_live_records.saturating_add(df.record_count());
                }
                ManifestStatus::Deleted => {
                    total_dead += 1;
                }
            }
        }
    }

    // ---- puffin / stats coverage (opt-in) ----------------------------------
    if args.check_stats {
        println!();
        println!("=== puffin coverage ===");

        // 1. per-file sidecars: laminar writes `<data>.parquet.stats` beside every
        //    data file, and the merge carries them forward into the output's
        //    sidecar. A live file without one means per-file index lookups fall
        //    back to a full scan for its rows.
        let mut missing_sidecars: Vec<&String> = Vec::new();
        for path in &live_file_paths {
            let sidecar = format!("{path}.stats");
            match file_io.exists(&sidecar).await {
                Ok(true) => {}
                Ok(false) => missing_sidecars.push(path),
                Err(e) => eprintln!("  sidecar check failed for {sidecar}: {e}"),
            }
        }
        println!(
            "per_file_sidecars : {}/{} present{}",
            live_file_paths.len() - missing_sidecars.len(),
            live_file_paths.len(),
            if missing_sidecars.is_empty() { "" } else { "  <-- GAPS" }
        );
        for p in missing_sidecars.iter().take(10) {
            println!("    MISSING {}", p.rsplit('/').next().unwrap_or(p));
        }

        // 2. consolidated per-partition puffin under data/_stats/. This is what
        //    lets the planner answer label/zone-map questions without fanning in
        //    every per-file sidecar; a missing one is why a query would go slow
        //    rather than wrong.
        let mut missing_parts: Vec<&String> = Vec::new();
        for path in &live_partition_puffins {
            match file_io.exists(path).await {
                Ok(true) => {}
                Ok(false) => missing_parts.push(path),
                Err(e) => eprintln!("  partition puffin check failed for {path}: {e}"),
            }
        }
        println!(
            "partition_puffins : {}/{} present{}",
            live_partition_puffins.len() - missing_parts.len(),
            live_partition_puffins.len(),
            if missing_parts.is_empty() { "" } else { "  <-- GAPS" }
        );
        for p in missing_parts.iter().take(10) {
            let tail: Vec<&str> = p.rsplit('/').take(4).collect();
            println!("    MISSING .../{}", tail.into_iter().rev().collect::<Vec<_>>().join("/"));
        }

        // 3. catalog registration. Entries accumulate per (snapshot_id, path);
        //    they are carried into the table metadata document, so a large total
        //    inflates every load_table/update_table response.
        let stats_entries = metadata.statistics_iter().count();
        let cur_snap = metadata.current_snapshot().map(|s| s.snapshot_id());
        let for_current = match cur_snap {
            Some(id) => metadata.statistics_iter().filter(|e| e.snapshot_id == id).count(),
            None => 0,
        };
        println!(
            "registered_stats  : {stats_entries} total in table metadata, {for_current} for the current snapshot"
        );
        if stats_entries > 200 {
            println!(
                "    NOTE {stats_entries} entries ride in every metadata response — check snapshot expiry is pruning them"
            );
        }
    }

    println!();
    println!("live_files       : {total_live_files}");
    println!("live_bytes       : {} ({} B)", human_bytes(total_live_bytes), total_live_bytes);
    println!("live_records     : {total_live_records}");
    println!("deleted_entries  : {total_dead} (tombstoned in this snapshot; invisible to reader)");
    println!();

    // Manifests-per-hour summary (before per-partition file table).
    if !manifests_per_hour.is_empty() || manifests_no_hour > 0 {
        // Show every hour, not a top-N. A truncated list is worse than a long
        // one here: a recent hour that has already consolidated drops off the
        // bottom, and its absence reads as "zero manifests" rather than "few".
        if !live_manifests_per_hour.is_empty() {
            println!("live_manifests_per_hour (still holding >=1 live file — the real planner cost):");
            let mut lrows: Vec<(i64, usize)> = live_manifests_per_hour
                .iter()
                .map(|(h, set)| (*h, set.len()))
                .collect();
            lrows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            for (h, n) in lrows.into_iter().take(12) {
                println!("  hour={h:<10} live_manifests={n}");
            }
            println!();
        }
        println!("manifests_per_hour (RAW from summary bounds — counts dead + multi-hour):");
        let mut rows: Vec<(i64, usize)> = manifests_per_hour.into_iter().collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (h, n) in rows.into_iter() {
            println!("  hour={h:<10} manifests={n}");
        }
        if manifests_no_hour > 0 {
            println!("  no_hour_bound manifests={manifests_no_hour}");
        }
        println!();
    }

    // Files-per-hour aggregate (across all tenants in that hour).
    let mut files_per_hour: BTreeMap<i64, (u64, u64)> = BTreeMap::new(); // hour → (files, bytes)
    for (key, s) in &by_partition {
        // partition_key emits "field=value,..." — parse timestamp_hour=X.
        if let Some(pos) = key.find("timestamp_hour=") {
            let after = &key[pos + "timestamp_hour=".len()..];
            let end = after.find(',').unwrap_or(after.len());
            if let Ok(h) = after[..end].parse::<i64>() {
                let e = files_per_hour.entry(h).or_insert((0, 0));
                e.0 = e.0.saturating_add(s.live_files);
                e.1 = e.1.saturating_add(s.live_bytes);
            }
        }
    }
    if !files_per_hour.is_empty() {
        println!("files_per_hour (all tenants aggregate, newest 10 hours):");
        let mut rows: Vec<(i64, (u64, u64))> = files_per_hour.into_iter().collect();
        rows.sort_by(|a, b| b.0.cmp(&a.0)); // newest first
        for (h, (files, bytes)) in rows.into_iter().take(10) {
            println!("  hour={h:<10} files={files:<6} bytes={}", human_bytes(bytes));
        }
        println!();
    }

    // Emitted before the by_partition early-return so an empty table still
    // produces a (possibly empty) dump rather than silently nothing.
    if args.dump_live_paths {
        println!("=== live path dump ===");
        println!(
            "# LIVE_ROOT is the snapshot's manifest_list. Its delta-chain ancestors are\n\
             # walked internally by load_manifest_list and are NOT enumerated here, so\n\
             # treat unmatched root-*.parquet as 'not the current root', not as proven\n\
             # garbage. LIVE_MANIFEST and LIVE_DATA are exact."
        );
        println!("LIVE_ROOT {}", snapshot.manifest_list());
        live_manifest_paths.sort();
        live_manifest_paths.dedup();
        for p in &live_manifest_paths {
            println!("LIVE_MANIFEST {p}");
        }
        live_file_paths.sort();
        live_file_paths.dedup();
        for p in &live_file_paths {
            println!("LIVE_DATA {p}");
        }
        println!(
            "# dump totals: manifests={} data_files={} (data must equal live_files={})",
            live_manifest_paths.len(),
            live_file_paths.len(),
            total_live_files
        );
        println!();
    }

    if by_partition.is_empty() {
        println!("(no live entries)");
        return ExitCode::SUCCESS;
    }

    // Sort partitions by live_files desc so the noisiest is on top.
    let mut rows: Vec<(&String, &PartitionStats)> = by_partition.iter().collect();
    rows.sort_by(|a, b| b.1.live_files.cmp(&a.1.live_files));

    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}  partition",
        "files", "total_sz", "avg_sz", "min_sz", "max_sz", "records"
    );
    println!("{}", "-".repeat(96));
    for (key, s) in rows {
        println!(
            "{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}  {}",
            s.live_files,
            human_bytes(s.live_bytes),
            human_bytes(s.avg_bytes()),
            human_bytes(s.min_bytes()),
            human_bytes(s.max_bytes()),
            s.live_records,
            key
        );
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::parse_args_from;

    fn base() -> Vec<String> {
        ["--uri", "http://localhost:18181/catalog",
         "--warehouse", "laminar",
         "--namespace", "observability",
         "--table", "logs"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn boolean_flags_default_off() {
        let args = parse_args_from(base()).expect("parse");
        assert!(!args.check_stats);
        assert!(!args.dump_live_paths);
        assert_eq!(args.table, "logs");
    }

    #[test]
    fn dump_live_paths_parses_and_consumes_no_value() {
        // The flag takes no value, so the --table that follows must still bind.
        let mut raw = vec!["--dump-live-paths".to_string()];
        raw.extend(base());
        let args = parse_args_from(raw).expect("parse");
        assert!(args.dump_live_paths);
        assert!(!args.check_stats);
        assert_eq!(args.table, "logs");
        assert_eq!(args.namespace, "observability");
    }

    #[test]
    fn dump_and_check_stats_are_independent() {
        let mut raw = base();
        raw.push("--dump-live-paths".to_string());
        raw.push("--check-stats".to_string());
        let args = parse_args_from(raw).expect("parse");
        assert!(args.dump_live_paths);
        assert!(args.check_stats);
    }

    #[test]
    fn help_mentions_dump_flag() {
        let err = parse_args_from(vec!["--help".to_string(), "x".to_string()])
            .expect_err("help returns Err");
        assert!(err.contains("--dump-live-paths"), "usage text: {err}");
    }

    #[test]
    fn unknown_flag_rejected() {
        let mut raw = base();
        raw.push("--dump-live-path".to_string()); // singular typo
        raw.push("v".to_string());
        assert!(parse_args_from(raw).is_err());
    }
}
