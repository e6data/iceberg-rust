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
// Test 4b: Disjoint-partition replace with OCC retry
//
// Two writers each replace files in DIFFERENT partitions (disjoint).
// With correct cache invalidation on retry, the loser should retry
// and succeed — its delete list targets files the winner didn't touch.
//
// This test verifies the fix for the stale-cache bug:
// - W1 compacts partition A files → merged_A
// - W2 compacts partition B files → merged_B
// - One wins CAS, other retries
// - After fix: retry re-reads manifest list from new snapshot,
//   finds its target files still present, succeeds
// - Result: both merged files present, no data loss
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_disjoint_partition_replace_retry_no_data_loss() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_ns5", "disjoint_replace").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_ns5".to_string()),
        "disjoint_replace".to_string(),
    );

    // Seed: 3 files for "partition A" and 3 files for "partition B"
    // (using file path to distinguish; unpartitioned table for simplicity)
    for i in 0..3 {
        let table = catalog.load_table(&ident).await.unwrap();
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![
            test_data_file(&format!("s3://test/data/partA_file_{}.parquet", i), 100),
        ]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog.as_ref()).await.unwrap();
    }
    for i in 0..3 {
        let table = catalog.load_table(&ident).await.unwrap();
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![
            test_data_file(&format!("s3://test/data/partB_file_{}.parquet", i), 100),
        ]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog.as_ref()).await.unwrap();
    }

    // Verify seed: 6 files, 600 rows
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let seed_tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let seed_records: u64 = seed_tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();
    assert_eq!(seed_records, 600, "seed should have 600 rows");
    assert_eq!(seed_tasks.len(), 6, "seed should have 6 files");
    println!("  Seed OK: 6 files, 600 rows");

    // Writer A: replace partA_file_0..2 → merged_A (300 rows)
    // Writer B: replace partB_file_0..2 → merged_B (300 rows)
    // These are DISJOINT — no overlap in files_to_delete.
    let barrier = Arc::new(Barrier::new(2));

    let part_a_files: Vec<DataFile> = (0..3)
        .map(|i| test_data_file(&format!("s3://test/data/partA_file_{}.parquet", i), 100))
        .collect();
    let part_b_files: Vec<DataFile> = (0..3)
        .map(|i| test_data_file(&format!("s3://test/data/partB_file_{}.parquet", i), 100))
        .collect();

    let catalog_a = catalog.clone();
    let catalog_b = catalog.clone();
    let ident_a = ident.clone();
    let ident_b = ident.clone();
    let barrier_a = barrier.clone();
    let barrier_b = barrier.clone();
    let files_a = part_a_files.clone();
    let files_b = part_b_files.clone();

    let handle_a = tokio::spawn(async move {
        let table = catalog_a.load_table(&ident_a).await.unwrap();
        barrier_a.wait().await;

        let tx = Transaction::new(&table);
        let action = tx
            .replace_data_files()
            .delete_files(files_a)
            .add_files(vec![test_data_file(
                "s3://test/data/merged_A.parquet",
                300,
            )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog_a.as_ref()).await
    });

    let handle_b = tokio::spawn(async move {
        let table = catalog_b.load_table(&ident_b).await.unwrap();
        barrier_b.wait().await;

        let tx = Transaction::new(&table);
        let action = tx
            .replace_data_files()
            .delete_files(files_b)
            .add_files(vec![test_data_file(
                "s3://test/data/merged_B.parquet",
                300,
            )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog_b.as_ref()).await
    });

    let result_a = handle_a.await.unwrap();
    let result_b = handle_b.await.unwrap();

    println!(
        "  Writer A: {}",
        if result_a.is_ok() { "SUCCESS" } else { "FAILED" }
    );
    println!(
        "  Writer B: {}",
        if result_b.is_ok() { "SUCCESS" } else { "FAILED" }
    );

    // BOTH should succeed — disjoint partitions, retry should handle OCC
    assert!(
        result_a.is_ok(),
        "Writer A should succeed: {:?}",
        result_a.err()
    );
    assert!(
        result_b.is_ok(),
        "Writer B should succeed: {:?}",
        result_b.err()
    );

    // Verify final state: 2 files (merged_A + merged_B), 600 rows total
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let final_tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let final_records: u64 = final_tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();
    let final_paths: Vec<String> = final_tasks.iter().map(|t| t.data_file_path().to_string()).collect();

    println!("  Final files: {:?}", final_paths);
    println!("  Final records: {}", final_records);

    assert_eq!(
        final_records, 600,
        "expected 600 rows (no data loss), got {}",
        final_records
    );
    assert_eq!(
        final_tasks.len(),
        2,
        "expected 2 merged files, got {}",
        final_tasks.len()
    );
    assert!(
        final_paths.contains(&"s3://test/data/merged_A.parquet".to_string()),
        "merged_A missing from final files"
    );
    assert!(
        final_paths.contains(&"s3://test/data/merged_B.parquet".to_string()),
        "merged_B missing from final files"
    );

    println!("  PASS: disjoint replace with OCC retry — 2 files, 600 rows, no data loss");
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

// ─────────────────────────────────────────────────────────
// Test 5: Overlapping replace — correct conflict detection
//
// Two writers replace THE SAME files. One wins, the other's retry
// correctly fails because the files are already gone.
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_overlapping_replace_correct_conflict() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_overlap", "same_files").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_overlap".to_string()),
        "same_files".to_string(),
    );

    // Seed with 4 files (400 rows)
    let _table = seed_table(catalog.as_ref(), "test_overlap", "same_files", 4).await;

    let seed_files: Vec<DataFile> = (0..4)
        .map(|i| test_data_file(&format!("s3://test/data/seed_{}.parquet", i), 100))
        .collect();

    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();

    for writer_id in 0..2u32 {
        let catalog = catalog.clone();
        let ident = ident.clone();
        let barrier = barrier.clone();
        let files = seed_files.clone();

        handles.push(tokio::spawn(async move {
            let table = catalog.load_table(&ident).await.unwrap();
            barrier.wait().await;

            let tx = Transaction::new(&table);
            let action = tx
                .replace_data_files()
                .delete_files(files)
                .add_files(vec![test_data_file(
                    &format!("s3://test/data/merged_w{}.parquet", writer_id),
                    400,
                )]);
            let tx = action.apply(tx).unwrap();
            tx.commit(catalog.as_ref()).await
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    let successes = results.iter().filter(|r| r.is_ok()).count();
    let failures = results.iter().filter(|r| r.is_err()).count();

    println!("  Overlapping replace: {} success, {} failed", successes, failures);

    // Exactly one should succeed. The other retries, refreshes the table,
    // rebuilds manifests from the new snapshot, can't find files to delete.
    assert_eq!(successes, 1, "exactly one writer should succeed");
    assert_eq!(failures, 1, "the other should fail (files already replaced)");

    // Verify: 1 merged file, 400 rows, no duplicates
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let total_records: u64 = tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();

    println!("  Final: {} files, {} records", tasks.len(), total_records);
    assert_eq!(total_records, 400, "no duplicates: expected 400, got {}", total_records);
    assert_eq!(tasks.len(), 1, "expected 1 merged file");

    println!("  PASS: overlapping replace — correct conflict, no duplicates");
}

// ─────────────────────────────────────────────────────────
// Test 6: Replace + concurrent FastAppend
//
// Compaction races with ingest. Both should succeed.
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_replace_plus_concurrent_fast_append() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_mixed", "replace_append").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_mixed".to_string()),
        "replace_append".to_string(),
    );

    // Seed 3 files (300 rows)
    let _table = seed_table(catalog.as_ref(), "test_mixed", "replace_append", 3).await;

    let seed_files: Vec<DataFile> = (0..3)
        .map(|i| test_data_file(&format!("s3://test/data/seed_{}.parquet", i), 100))
        .collect();

    let barrier = Arc::new(Barrier::new(2));

    // Writer A: Replace (compact seed files)
    let catalog_a = catalog.clone();
    let ident_a = ident.clone();
    let barrier_a = barrier.clone();
    let files_a = seed_files.clone();

    let handle_replace = tokio::spawn(async move {
        let table = catalog_a.load_table(&ident_a).await.unwrap();
        barrier_a.wait().await;

        let tx = Transaction::new(&table);
        let action = tx
            .replace_data_files()
            .delete_files(files_a)
            .add_files(vec![test_data_file("s3://test/data/merged.parquet", 300)]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog_a.as_ref()).await
    });

    // Writer B: FastAppend (new data)
    let catalog_b = catalog.clone();
    let ident_b = ident.clone();
    let barrier_b = barrier.clone();

    let handle_append = tokio::spawn(async move {
        let table = catalog_b.load_table(&ident_b).await.unwrap();
        barrier_b.wait().await;

        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![test_data_file(
            "s3://test/data/new_ingest.parquet",
            200,
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog_b.as_ref()).await
    });

    let result_replace = handle_replace.await.unwrap();
    let result_append = handle_append.await.unwrap();

    println!(
        "  Replace: {}, Append: {}",
        if result_replace.is_ok() { "SUCCESS" } else { "FAILED" },
        if result_append.is_ok() { "SUCCESS" } else { "FAILED" },
    );

    // Both should succeed
    assert!(result_replace.is_ok(), "Replace should succeed: {:?}", result_replace.err());
    assert!(result_append.is_ok(), "Append should succeed: {:?}", result_append.err());

    // Verify: merged.parquet (300) + new_ingest.parquet (200) = 500 rows, 2 files
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let total_records: u64 = tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();
    let paths: Vec<String> = tasks.iter().map(|t| t.data_file_path().to_string()).collect();

    println!("  Final: {:?}, {} records", paths, total_records);
    assert_eq!(total_records, 500, "expected 500 rows, got {}", total_records);
    assert_eq!(tasks.len(), 2, "expected 2 files");

    println!("  PASS: replace + append both succeeded, no data loss");
}

// ─────────────────────────────────────────────────────────
// Test 7: 4-way disjoint replace via OCC retry cascade
//
// Four writers each compact their own disjoint file set.
// All should succeed via retry cascade (up to 3 retries).
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_four_way_disjoint_replace() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_4way", "disjoint4").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_4way".to_string()),
        "disjoint4".to_string(),
    );

    // Seed: 4 groups of 2 files each (8 files, 800 rows)
    for group in 0..4u32 {
        for file_idx in 0..2u32 {
            let table = catalog.load_table(&ident).await.unwrap();
            let tx = Transaction::new(&table);
            let action = tx.fast_append().add_data_files(vec![test_data_file(
                &format!("s3://test/data/g{}_f{}.parquet", group, file_idx),
                100,
            )]);
            let tx = action.apply(tx).unwrap();
            tx.commit(catalog.as_ref()).await.unwrap();
        }
    }

    // Verify seed
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let seed_tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(seed_tasks.len(), 8, "seed should have 8 files");
    println!("  Seed OK: 8 files, 800 rows");

    // 4 concurrent writers, each compacting their own group
    let barrier = Arc::new(Barrier::new(4));
    let mut handles = Vec::new();

    for group in 0..4u32 {
        let catalog = catalog.clone();
        let ident = ident.clone();
        let barrier = barrier.clone();

        let group_files: Vec<DataFile> = (0..2)
            .map(|f| test_data_file(&format!("s3://test/data/g{}_f{}.parquet", group, f), 100))
            .collect();

        handles.push(tokio::spawn(async move {
            let table = catalog.load_table(&ident).await.unwrap();
            barrier.wait().await;

            let tx = Transaction::new(&table);
            let action = tx
                .replace_data_files()
                .delete_files(group_files)
                .add_files(vec![test_data_file(
                    &format!("s3://test/data/merged_g{}.parquet", group),
                    200,
                )]);
            let tx = action.apply(tx).unwrap();
            let result = tx.commit(catalog.as_ref()).await;
            (group, result)
        }));
    }

    let mut successes = 0u32;
    for handle in handles {
        let (group, result) = handle.await.unwrap();
        let status = if result.is_ok() { "SUCCESS" } else { "FAILED" };
        println!("  Group {}: {}", group, status);
        if result.is_ok() {
            successes += 1;
        } else {
            panic!("Group {} failed: {:?}", group, result.err());
        }
    }

    assert_eq!(successes, 4, "all 4 disjoint writers should succeed");

    // Verify: 4 merged files, 800 rows
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let total_records: u64 = tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();
    let paths: Vec<String> = tasks.iter().map(|t| t.data_file_path().to_string()).collect();

    println!("  Final: {} files, {} records", tasks.len(), total_records);
    assert_eq!(total_records, 800, "expected 800 rows, got {}", total_records);
    assert_eq!(tasks.len(), 4, "expected 4 merged files");

    for g in 0..4 {
        let expected = format!("s3://test/data/merged_g{}.parquet", g);
        assert!(paths.contains(&expected), "missing {}", expected);
    }

    println!("  PASS: 4-way disjoint replace — all succeeded, 800 rows, no data loss");
}

// ─────────────────────────────────────────────────────────
// Test 8: Many-manifest cache invalidation stress
//
// 20 individual commits (20 manifests), then two disjoint
// writers. Stresses cache rebuild with a large manifest list.
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_disjoint_replace_many_manifests() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_many", "manifests").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_many".to_string()),
        "manifests".to_string(),
    );

    // Seed 20 files, each in its own commit (= 20 manifests)
    for i in 0..20u32 {
        let table = catalog.load_table(&ident).await.unwrap();
        let tx = Transaction::new(&table);
        let action = tx.fast_append().add_data_files(vec![test_data_file(
            &format!("s3://test/data/file_{:02}.parquet", i),
            50,
        )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog.as_ref()).await.unwrap();
    }

    // Verify seed
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let seed_tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert_eq!(seed_tasks.len(), 20, "seed should have 20 files");
    println!("  Seed OK: 20 files in 20 manifests");

    // Writer A: compact files 0-9 → merged_first_half
    // Writer B: compact files 10-19 → merged_second_half
    let files_a: Vec<DataFile> = (0..10)
        .map(|i| test_data_file(&format!("s3://test/data/file_{:02}.parquet", i), 50))
        .collect();
    let files_b: Vec<DataFile> = (10..20)
        .map(|i| test_data_file(&format!("s3://test/data/file_{:02}.parquet", i), 50))
        .collect();

    let barrier = Arc::new(Barrier::new(2));

    let catalog_a = catalog.clone();
    let ident_a = ident.clone();
    let barrier_a = barrier.clone();
    let handle_a = tokio::spawn(async move {
        let table = catalog_a.load_table(&ident_a).await.unwrap();
        barrier_a.wait().await;

        let tx = Transaction::new(&table);
        let action = tx
            .replace_data_files()
            .delete_files(files_a)
            .add_files(vec![test_data_file(
                "s3://test/data/merged_first_half.parquet",
                500,
            )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog_a.as_ref()).await
    });

    let catalog_b = catalog.clone();
    let ident_b = ident.clone();
    let barrier_b = barrier.clone();
    let handle_b = tokio::spawn(async move {
        let table = catalog_b.load_table(&ident_b).await.unwrap();
        barrier_b.wait().await;

        let tx = Transaction::new(&table);
        let action = tx
            .replace_data_files()
            .delete_files(files_b)
            .add_files(vec![test_data_file(
                "s3://test/data/merged_second_half.parquet",
                500,
            )]);
        let tx = action.apply(tx).unwrap();
        tx.commit(catalog_b.as_ref()).await
    });

    let result_a = handle_a.await.unwrap();
    let result_b = handle_b.await.unwrap();

    assert!(result_a.is_ok(), "Writer A failed: {:?}", result_a.err());
    assert!(result_b.is_ok(), "Writer B failed: {:?}", result_b.err());

    // Verify: 2 merged files, 1000 rows
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let total_records: u64 = tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();

    println!("  Final: {} files, {} records", tasks.len(), total_records);
    assert_eq!(total_records, 1000, "expected 1000 rows, got {}", total_records);
    assert_eq!(tasks.len(), 2, "expected 2 merged files");

    println!("  PASS: many-manifest disjoint replace — cache invalidation correct");
}

// ─────────────────────────────────────────────────────────
// Test 9: Sequential replace — no contention baseline
//
// Single writer, no contention. Regression guard for cache-tagging.
// ─────────────────────────────────────────────────────────

#[tokio::test]
async fn test_sequential_replace_no_contention() {
    let catalog = Arc::new(setup_catalog().await);
    let _ = create_test_table(catalog.as_ref(), "test_seq", "no_contention").await;

    let ident = iceberg::TableIdent::new(
        NamespaceIdent::new("test_seq".to_string()),
        "no_contention".to_string(),
    );

    // Seed 5 files
    let _table = seed_table(catalog.as_ref(), "test_seq", "no_contention", 5).await;

    let seed_files: Vec<DataFile> = (0..5)
        .map(|i| test_data_file(&format!("s3://test/data/seed_{}.parquet", i), 100))
        .collect();

    // Single writer: replace all 5 → 1 merged
    let table = catalog.load_table(&ident).await.unwrap();
    let tx = Transaction::new(&table);
    let action = tx
        .replace_data_files()
        .delete_files(seed_files)
        .add_files(vec![test_data_file("s3://test/data/merged.parquet", 500)]);
    let tx = action.apply(tx).unwrap();
    let result = tx.commit(catalog.as_ref()).await;

    assert!(result.is_ok(), "single writer should succeed: {:?}", result.err());

    // Verify
    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let total_records: u64 = tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();

    assert_eq!(total_records, 500, "expected 500, got {}", total_records);
    assert_eq!(tasks.len(), 1, "expected 1 merged file");

    // Second replace on the merged file — cache doesn't interfere
    let table = catalog.load_table(&ident).await.unwrap();
    let tx = Transaction::new(&table);
    let action = tx
        .replace_data_files()
        .delete_files(vec![test_data_file("s3://test/data/merged.parquet", 500)])
        .add_files(vec![test_data_file("s3://test/data/remerged.parquet", 500)]);
    let tx = action.apply(tx).unwrap();
    let result = tx.commit(catalog.as_ref()).await;

    assert!(result.is_ok(), "second replace should succeed: {:?}", result.err());

    let table = catalog.load_table(&ident).await.unwrap();
    let scan = table.scan().build().unwrap();
    let tasks: Vec<_> = scan
        .plan_files()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let total_records: u64 = tasks.iter().map(|t| t.record_count.unwrap_or(0)).sum();

    assert_eq!(total_records, 500);
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].data_file_path(), "s3://test/data/remerged.parquet");

    println!("  PASS: sequential replace — no contention, cache doesn't interfere");
}

