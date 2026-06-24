//! Iceberg OCC Concurrency Test
//!
//! Tests concurrent FastAppend and ReplaceDataFiles commits against
//! MemoryCatalog to verify OCC rejects stale commits.
//!
//! Run: cargo test -p iceberg --test occ_concurrency_test -- --nocapture

use std::collections::HashMap;
use std::sync::Arc;

use futures::TryStreamExt;

use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};
use iceberg::spec::{
    DataContentType, DataFile, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
    Struct, Type,
};
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::memory::{MemoryCatalogBuilder, MEMORY_CATALOG_WAREHOUSE};
use tokio::sync::Barrier;

fn temp_warehouse() -> String {
    let dir = tempfile::tempdir().unwrap();
    // Leak the tempdir so it doesn't get cleaned up during the test
    let path = dir.path().to_str().unwrap().to_string();
    std::mem::forget(dir);
    path
}

fn test_schema() -> Schema {
    Schema::builder()
        .with_fields(vec![
            NestedField::required(1, "ts", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(2, "writer_id", Type::Primitive(PrimitiveType::Long)).into(),
            NestedField::required(3, "value", Type::Primitive(PrimitiveType::String)).into(),
        ])
        .build()
        .unwrap()
}

fn test_data_file(path: &str, rows: u64) -> DataFile {
    DataFileBuilder::default()
        .content(DataContentType::Data)
        .file_format(DataFileFormat::Parquet)
        .file_path(path.to_string())
        .file_size_in_bytes(rows * 100)
        .record_count(rows)
        .partition_spec_id(0)
        .partition(Struct::empty())
        .build()
        .unwrap()
}

async fn setup_catalog() -> impl Catalog {
    let warehouse = temp_warehouse();
    MemoryCatalogBuilder::default()
        .load(
            "test",
            HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
        )
        .await
        .unwrap()
}

async fn create_test_table(catalog: &impl Catalog, ns: &str, name: &str) -> iceberg::table::Table {
    let ns_ident = NamespaceIdent::new(ns.to_string());
    let _ = catalog
        .create_namespace(&ns_ident, HashMap::new())
        .await;

    catalog
        .create_table(
            &ns_ident,
            TableCreation::builder()
                .name(name.to_string())
                .schema(test_schema())
                .build(),
        )
        .await
        .unwrap()
}

/// Seed a table with N files via sequential FastAppend commits.
async fn seed_table(catalog: &impl Catalog, ns: &str, name: &str, num_files: usize) -> iceberg::table::Table {
    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new(ns.to_string()),
        name.to_string(),
    );

    for i in 0..num_files {
        let table = catalog.load_table(&ident).await.unwrap();
        let tx = Transaction::new(&table);
        let action = tx
            .fast_append()
            .add_data_files(vec![test_data_file(
                &format!("s3://test/data/seed_{}.parquet", i),
                100,
            )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog).await.unwrap();
    }

    catalog.load_table(&ident).await.unwrap()
}

// ─────────────────────────────────────────────────────────
// Test 1: Concurrent FastAppend
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_fast_append() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_ns", "fast_append").await;

    let n_writers = 4;
    let barrier = Arc::new(Barrier::new(n_writers));
    let mut handles = Vec::new();

    for writer_id in 0..n_writers {
        let catalog = catalog.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            let ident = iceberg::TableIdent::new(
                NamespaceIdent::new("test_ns".to_string()),
                "fast_append".to_string(),
            );

            barrier.wait().await;

            let table = catalog.load_table(&ident).await.unwrap();
            let tx = Transaction::new(&table);
            let action = tx.fast_append().add_data_files(vec![test_data_file(
                &format!("s3://test/data/writer_{}.parquet", writer_id),
                100,
            )]);
            let tx = action.apply(tx).unwrap();
            let result = tx.commit(catalog.as_ref()).await;

            (writer_id, result.is_ok())
        }));
    }

    let mut successes = 0;
    let mut failures = 0;
    for handle in handles {
        let (writer_id, ok) = handle.await.unwrap();
        if ok {
            successes += 1;
            println!("  Writer {}: SUCCESS", writer_id);
        } else {
            failures += 1;
            println!("  Writer {}: REJECTED (OCC)", writer_id);
        }
    }

    println!("  FastAppend: {} success, {} rejected", successes, failures);

    // At least one must succeed. Others rejected by OCC.
    // With MemoryCatalog (mutex-protected), likely only 1 succeeds.
    assert!(successes >= 1, "at least one writer must succeed");

    // Load final table and count files
    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_ns".to_string()),
        "fast_append".to_string(),
    );
    let table = catalog.load_table(&ident).await.unwrap();
    let snapshot = table.metadata().current_snapshot().unwrap();
    println!(
        "  Final snapshot: {} (summary: {:?})",
        snapshot.snapshot_id(),
        snapshot.summary()
    );
}

// ─────────────────────────────────────────────────────────
// Test 2: Concurrent ReplaceDataFiles
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_concurrent_replace_data_files() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_ns2", "replace_race").await;

    // Seed with 5 files
    let table = seed_table(catalog.as_ref(), "test_ns2", "replace_race", 5).await;
    let initial_snapshot_id = table
        .metadata()
        .current_snapshot()
        .unwrap()
        .snapshot_id();

    println!("  Seeded: snapshot_id={}", initial_snapshot_id);

    // Collect the current data files for deletion
    let current_snapshot = table.metadata().current_snapshot().unwrap();
    println!(
        "  Seed snapshot summary: {:?}",
        current_snapshot.summary()
    );

    // Now N writers all try to replace the files concurrently
    let n_writers = 4;
    let barrier = Arc::new(Barrier::new(n_writers));
    let mut handles = Vec::new();

    for writer_id in 0..n_writers {
        let catalog = catalog.clone();
        let barrier = barrier.clone();

        // Each writer will read the seed files and try to replace them
        // We capture the file list from the seeded table
        let seed_files: Vec<DataFile> = (0..5)
            .map(|i| test_data_file(&format!("s3://test/data/seed_{}.parquet", i), 100))
            .collect();

        handles.push(tokio::spawn(async move {
            let ident = iceberg::TableIdent::new(
                NamespaceIdent::new("test_ns2".to_string()),
                "replace_race".to_string(),
            );

            // Load table — all writers see the same snapshot
            let table = catalog.load_table(&ident).await.unwrap();
            let snap_id = table
                .metadata()
                .current_snapshot()
                .map(|s| s.snapshot_id())
                .unwrap_or(0);

            barrier.wait().await;

            // Build replace_data_files: delete seed files, add one merged file
            let tx = Transaction::new(&table);
            let action = tx
                .replace_data_files()
                .delete_files(seed_files)
                .add_files(vec![test_data_file(
                    &format!("s3://test/data/merged_by_writer_{}.parquet", writer_id),
                    500, // 5 * 100 rows merged
                )]);

            match action.apply(tx) {
                Ok(tx) => match tx.commit(catalog.as_ref()).await {
                    Ok(_) => {
                        println!(
                            "  Writer {}: COMMITTED replace_data_files (base={})",
                            writer_id, snap_id
                        );
                        (writer_id, "SUCCESS".to_string())
                    }
                    Err(e) => {
                        println!(
                            "  Writer {}: REJECTED by catalog (base={}): {}",
                            writer_id,
                            snap_id,
                            e.to_string().chars().take(80).collect::<String>()
                        );
                        (writer_id, format!("COMMIT_REJECTED: {}", e))
                    }
                },
                Err(e) => {
                    println!("  Writer {}: APPLY FAILED: {}", writer_id, e);
                    (writer_id, format!("APPLY_FAILED: {}", e))
                }
            }
        }));
    }

    let mut successes = 0;
    let mut rejections = 0;
    for handle in handles {
        let (_writer_id, result) = handle.await.unwrap();
        if result == "SUCCESS" {
            successes += 1;
        } else {
            rejections += 1;
        }
    }

    println!(
        "\n  ReplaceDataFiles race: {} success, {} rejected",
        successes, rejections
    );
    println!("  Expected: 1 success, {} rejected", n_writers - 1);

    assert_eq!(
        successes, 1,
        "exactly one replace_data_files should succeed; got {}",
        successes
    );
    assert_eq!(
        rejections,
        n_writers - 1,
        "all other writers should be rejected"
    );
}

// ─────────────────────────────────────────────────────────
// Test 3: Replace with retry (simulates Laminar retry)
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_replace_with_retry_no_duplicates() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_ns3", "replace_retry").await;

    // Seed with 5 files (500 rows total)
    let _table = seed_table(catalog.as_ref(), "test_ns3", "replace_retry", 5).await;

    let n_writers = 3;
    let max_retries = 3;
    let barrier = Arc::new(Barrier::new(n_writers));
    let mut handles = Vec::new();

    for writer_id in 0..n_writers {
        let catalog = catalog.clone();
        let barrier = barrier.clone();

        handles.push(tokio::spawn(async move {
            let ident = iceberg::TableIdent::new(
                NamespaceIdent::new("test_ns3".to_string()),
                "replace_retry".to_string(),
            );

            barrier.wait().await;

            for attempt in 0..max_retries {
                // Re-load table on each retry (fresh snapshot)
                let table = catalog.load_table(&ident).await.unwrap();
                let snap = table
                    .metadata()
                    .current_snapshot()
                    .map(|s| s.snapshot_id())
                    .unwrap_or(0);

                // Read current files from the latest snapshot
                // For simplicity, we create synthetic delete list based on what we know
                // In real code, this would scan the manifest
                let seed_files: Vec<DataFile> = (0..5)
                    .map(|i| test_data_file(&format!("s3://test/data/seed_{}.parquet", i), 100))
                    .collect();

                let tx = Transaction::new(&table);
                let action = tx
                    .replace_data_files()
                    .delete_files(seed_files)
                    .add_files(vec![test_data_file(
                        &format!(
                            "s3://test/data/merged_w{}_attempt{}.parquet",
                            writer_id, attempt
                        ),
                        500,
                    )]);

                match action.apply(tx) {
                    Ok(tx) => match tx.commit(catalog.as_ref()).await {
                        Ok(_) => {
                            println!(
                                "  Writer {} attempt {}: SUCCESS (base={})",
                                writer_id, attempt, snap
                            );
                            return (writer_id, "SUCCESS".to_string(), attempt + 1);
                        }
                        Err(_) => {
                            println!(
                                "  Writer {} attempt {}: REJECTED, will retry",
                                writer_id, attempt
                            );
                            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                        }
                    },
                    Err(e) => {
                        println!(
                            "  Writer {} attempt {}: APPLY FAILED ({}), will retry",
                            writer_id,
                            attempt,
                            e.to_string().chars().take(60).collect::<String>()
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                    }
                }
            }

            (
                writer_id,
                "ALL_RETRIES_FAILED".to_string(),
                max_retries,
            )
        }));
    }

    for handle in handles {
        let (writer_id, result, attempts) = handle.await.unwrap();
        println!(
            "  Writer {}: {} (attempts={})",
            writer_id, result, attempts
        );
    }

    // Verify: load final table, check total record count from metadata
    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_ns3".to_string()),
        "replace_retry".to_string(),
    );
    let table = catalog.load_table(&ident).await.unwrap();
    let snapshot = table.metadata().current_snapshot().unwrap();
    let summary = snapshot.summary();

    println!("\n  Final snapshot summary: {:?}", summary);

    // The total-records in the summary should still be 500 (no duplicates)
    // With disable_retry for ReplaceDataFiles, only 1 writer succeeds
    // and others fail all retries. The summary is unreliable (shows 0 due
    // to the overwrite), but the key guarantee is: no duplicates.
    // Verify via scan that the data is correct.
    let scan = table.scan().build().unwrap();
    let file_tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let actual_records: u64 = file_tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();
    println!("  Actual records from scan: {}", actual_records);
    // Should be 500 (the merged file from the one successful writer)
    // OR 500 (original 5 files, if no merge succeeded at all because
    // the first writer happened to also fail for some reason)
    assert!(
        actual_records == 500,
        "expected 500 records, got {} — data integrity violation",
        actual_records
    );
    println!("  PASS: {} records, no duplicates", actual_records);
}

// ─────────────────────────────────────────────────────────
// Test 4: Two-replica ingest + merge simulation
//
// Simulates the production Laminar scenario:
//   - 2 replicas, each doing FastAppend (always succeeds)
//   - Each replica tries merge-on-write (ReplaceDataFiles)
//   - Merge is NON-retryable: on OCC rejection, drop the merge
//   - Verify: ALL ingested data is present (no loss)
//   - Verify: no duplicate rows
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_two_replica_ingest_and_merge_no_data_loss() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_ns4", "replica_sim").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_ns4".to_string()),
        "replica_sim".to_string(),
    );

    let n_replicas = 2;
    let n_checkpoints = 5;
    let rows_per_file = 100;
    let merge_threshold = 3; // merge after 3 files accumulate

    println!("\n  === Two-replica simulation ===");
    println!("  Replicas: {}, Checkpoints: {}, Merge threshold: {}",
             n_replicas, n_checkpoints, merge_threshold);

    // Track what each replica appends
    let mut total_appended_rows: u64 = 0;
    let mut total_merge_attempts: u32 = 0;
    let mut total_merge_success: u32 = 0;
    let mut total_merge_rejected: u32 = 0;

    // Simulate checkpoint cycles
    for checkpoint in 0..n_checkpoints {
        println!("\n  --- Checkpoint {} ---", checkpoint);

        // Phase 1: Both replicas FastAppend their data
        for replica in 0..n_replicas {
            let table = catalog.load_table(&ident).await.unwrap();
            let snap_before = table.metadata().current_snapshot().map(|s| s.snapshot_id());
            let file_path = format!(
                "s3://test/data/replica_{}_cp_{}.parquet",
                replica, checkpoint
            );
            let tx = Transaction::new(&table);
            let action = tx
                .fast_append()
                .add_data_files(vec![test_data_file(&file_path, rows_per_file as u64)]);
            let tx = action.apply(tx).unwrap();
            let updated = tx.commit(catalog.as_ref()).await.unwrap();
            let snap_after = updated.metadata().current_snapshot().map(|s| s.snapshot_id());
            let summary = updated.metadata().current_snapshot().unwrap().summary();
            total_appended_rows += rows_per_file as u64;
            println!("    Replica {} FastAppend: {} snap={:?}->{:?} total-records={} total-files={}",
                     replica, file_path, snap_before, snap_after,
                     summary.additional_properties.get("total-records").unwrap_or(&"?".to_string()),
                     summary.additional_properties.get("total-data-files").unwrap_or(&"?".to_string()));
        }

        // Phase 2: Both replicas check if merge is needed, both try
        // This simulates the race condition
        let table = catalog.load_table(&ident).await.unwrap();
        let snap = table.metadata().current_snapshot().unwrap();
        let total_files: u64 = snap
            .summary()
            .additional_properties
            .get("total-data-files")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        println!("    Table state: {} total files", total_files);

        if total_files >= merge_threshold as u64 {
            println!("    Merge threshold reached — both replicas attempt merge");

            // Collect current files for deletion
            // Both replicas see the same snapshot and same files
            let files_to_merge: Vec<DataFile> = (0..checkpoint + 1)
                .flat_map(|cp| {
                    (0..n_replicas).map(move |r| {
                        test_data_file(
                            &format!("s3://test/data/replica_{}_cp_{}.parquet", r, cp),
                            rows_per_file as u64,
                        )
                    })
                })
                .collect();

            let merged_rows = files_to_merge.len() as u64 * rows_per_file as u64;
            let barrier = Arc::new(Barrier::new(n_replicas));
            let mut handles = Vec::new();

            for replica in 0..n_replicas {
                let catalog = catalog.clone();
                let ident = ident.clone();
                let barrier = barrier.clone();
                let files = files_to_merge.clone();

                handles.push(tokio::spawn(async move {
                    let table = catalog.load_table(&ident).await.unwrap();

                    barrier.wait().await;

                    // Build merge commit — NON-retryable
                    let tx = Transaction::new(&table);
                    let action = tx
                        .replace_data_files()
                        .delete_files(files)
                        .add_files(vec![test_data_file(
                            &format!(
                                "s3://test/data/merged_replica_{}_cp_{}.parquet",
                                replica, checkpoint
                            ),
                            merged_rows,
                        )]);

                    match action.apply(tx) {
                        Ok(tx) => {
                            // Use catalog.update_table directly to avoid
                            // Transaction::commit's built-in retry
                            // For now, just try once via the normal path
                            // and check if it would have conflicted
                            match tx.commit(catalog.as_ref()).await {
                                Ok(_) => {
                                    println!("    Replica {} merge: COMMITTED", replica);
                                    (replica, "SUCCESS")
                                }
                                Err(e) => {
                                    println!("    Replica {} merge: REJECTED ({})",
                                             replica,
                                             e.to_string().chars().take(60).collect::<String>());
                                    (replica, "REJECTED")
                                }
                            }
                        }
                        Err(e) => {
                            println!("    Replica {} merge: APPLY FAILED ({})",
                                     replica,
                                     e.to_string().chars().take(60).collect::<String>());
                            (replica, "APPLY_FAILED")
                        }
                    }
                }));
            }

            for handle in handles {
                let (_replica, result) = handle.await.unwrap();
                total_merge_attempts += 1;
                match result {
                    "SUCCESS" => total_merge_success += 1,
                    _ => total_merge_rejected += 1,
                }
            }
        }
    }

    // Final verification
    let table = catalog.load_table(&ident).await.unwrap();
    let snap = table.metadata().current_snapshot().unwrap();
    let summary = snap.summary();

    let final_records: u64 = summary
        .additional_properties
        .get("total-records")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let final_files: u64 = summary
        .additional_properties
        .get("total-data-files")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    println!("\n  === RESULTS ===");
    println!("  Total rows appended:    {}", total_appended_rows);
    println!("  Total records in table: {}", final_records);
    println!("  Total files in table:   {}", final_files);
    println!("  Merge attempts:         {}", total_merge_attempts);
    println!("  Merge success:          {}", total_merge_success);
    println!("  Merge rejected:         {}", total_merge_rejected);
    println!("  Summary: {:?}", summary.additional_properties);

    // Verify via manifest scan — count actual files, not summary
    let snapshot = table.metadata().current_snapshot().unwrap();
    let manifest_list_path = snapshot.manifest_list();
    println!("  Manifest list: {}", manifest_list_path);

    // Count snapshots in history
    let snapshots: Vec<_> = table.metadata().snapshots().collect();
    println!("  Total snapshots in history: {}", snapshots.len());

    // The summary may be inaccurate for cumulative totals.
    // What matters is: can we read all the data back?
    // Use the table scan to count actual files.
    let scan = table.scan().build().unwrap();
    let file_tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();

    let actual_files = file_tasks.len();
    let actual_records: u64 = file_tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();

    println!("  Actual files from scan: {}", actual_files);
    println!("  Actual records from scan: {}", actual_records);

    // THE KEY ASSERTIONS:
    assert_eq!(
        actual_records, total_appended_rows,
        "DATA INTEGRITY: appended {} rows, scan found {} rows",
        total_appended_rows, actual_records
    );

    // File count may be less than n_replicas * n_checkpoints if merges succeeded.
    // What matters: no data loss (records match) and no duplicates.
    println!("  Files: {} (some may be merged)", actual_files);
    println!("  PASS: all {} rows present in {} files — no data loss", actual_records, actual_files);
}

