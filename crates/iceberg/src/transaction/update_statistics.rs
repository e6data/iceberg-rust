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

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::StatisticsFile;
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Result, TableUpdate};

/// A transactional action for updating statistics files in a table.
///
/// The Iceberg spec models a table's `statistics` field as a *list* of
/// [`StatisticsFile`] entries with `(snapshot_id, statistics_path)` as
/// the unique key — multiple entries with the same `snapshot_id` are
/// allowed when their paths differ. Tessellate's compaction commit path
/// relies on this: each chunk replaces ~25 source files with ~25
/// compacted outputs, each carrying its own Puffin sidecar (one
/// `StatisticsFile` per output, sharing the new snapshot id but with
/// distinct paths).
///
/// The previous implementation kept `statistics_to_set` as
/// `HashMap<i64, Option<StatisticsFile>>` keyed on `snapshot_id`, which
/// silently collapsed multiple calls with the same snapshot id to the
/// last-write-wins. That was too narrow for the spec and caused
/// tessellate to lose blobs at commit time: log lines showed 25
/// `set_statistics` calls per chunk while the catalog persisted 1.
///
/// Storage is now a `Vec` for `set_statistics` (preserves order +
/// duplicates per spec) plus a `HashSet` for `remove_statistics`
/// (which still applies to *all* entries for a snapshot — removal is
/// snapshot-keyed in the existing semantics; callers that need
/// per-path removal can issue a finer-grained TableUpdate themselves).
pub struct UpdateStatisticsAction {
    statistics_to_set: Vec<StatisticsFile>,
    statistics_to_remove: HashSet<i64>,
}

impl UpdateStatisticsAction {
    pub fn new() -> Self {
        Self {
            statistics_to_set: Vec::new(),
            statistics_to_remove: HashSet::new(),
        }
    }

    /// Append a statistics file to the table's `statistics` list.
    ///
    /// Per the Iceberg spec, multiple `StatisticsFile` entries may
    /// share the same `snapshot_id` as long as their `statistics_path`
    /// differs. Callers chain this method once per file they want to
    /// register; previous calls are NOT overwritten by later ones with
    /// the same snapshot id.
    ///
    /// # Arguments
    ///
    /// * `statistics_file` - The [`StatisticsFile`] to register.
    ///
    /// # Returns
    ///
    /// An updated [`UpdateStatisticsAction`] with the new statistics
    /// file appended.
    pub fn set_statistics(mut self, statistics_file: StatisticsFile) -> Self {
        self.statistics_to_set.push(statistics_file);
        self
    }

    /// Remove all statistics file entries for the given snapshot.
    ///
    /// Removal is keyed by snapshot id — every `StatisticsFile` whose
    /// `snapshot_id` matches is dropped, regardless of path. Mirrors
    /// the previous behaviour. Calling this *after* `set_statistics`
    /// for the same snapshot id will, at commit time, both add the
    /// new entry AND emit a remove for the snapshot; the catalog
    /// semantics decide ordering.
    ///
    /// # Arguments
    ///
    /// * `snapshot_id` - The ID of the snapshot whose statistics
    ///   entries should be removed.
    ///
    /// # Returns
    ///
    /// An updated [`UpdateStatisticsAction`] with the removal recorded.
    pub fn remove_statistics(mut self, snapshot_id: i64) -> Self {
        self.statistics_to_remove.insert(snapshot_id);
        self
    }
}

impl Default for UpdateStatisticsAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for UpdateStatisticsAction {
    async fn commit(self: Arc<Self>, _table: &Table) -> Result<ActionCommit> {
        let mut updates: Vec<TableUpdate> =
            Vec::with_capacity(self.statistics_to_set.len() + self.statistics_to_remove.len());

        for statistics in &self.statistics_to_set {
            updates.push(TableUpdate::SetStatistics {
                statistics: statistics.clone(),
            });
        }
        for snapshot_id in &self.statistics_to_remove {
            updates.push(TableUpdate::RemoveStatistics {
                snapshot_id: *snapshot_id,
            });
        }

        Ok(ActionCommit::new(updates, vec![]))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use as_any::Downcast;

    use crate::spec::{BlobMetadata, StatisticsFile};
    use crate::transaction::tests::make_v2_table;
    use crate::transaction::update_statistics::UpdateStatisticsAction;
    use crate::transaction::{ApplyTransactionAction, Transaction};

    #[test]
    fn test_update_statistics() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);

        let statistics_file_1 = StatisticsFile {
            snapshot_id: 3055729675574597004i64,
            statistics_path: "s3://a/b/stats.puffin".to_string(),
            file_size_in_bytes: 413,
            file_footer_size_in_bytes: 42,
            key_metadata: None,
            blob_metadata: vec![BlobMetadata {
                r#type: "ndv".to_string(),
                snapshot_id: 3055729675574597004i64,
                sequence_number: 1,
                fields: vec![1],
                properties: HashMap::new(),
            }],
        };

        let statistics_file_2 = StatisticsFile {
            snapshot_id: 3366729675595277004i64,
            statistics_path: "s3://a/b/stats.puffin".to_string(),
            file_size_in_bytes: 413,
            file_footer_size_in_bytes: 42,
            key_metadata: None,
            blob_metadata: vec![BlobMetadata {
                r#type: "ndv".to_string(),
                snapshot_id: 3366729675595277004i64,
                sequence_number: 1,
                fields: vec![1],
                properties: HashMap::new(),
            }],
        };

        // set stats1, set stats2, remove stats1
        let tx = tx
            .update_statistics()
            .set_statistics(statistics_file_1.clone())
            .set_statistics(statistics_file_2.clone())
            .remove_statistics(3055729675574597004i64) // remove stats1
            .apply(tx)
            .unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdateStatisticsAction>()
            .unwrap();
        // Both `set_statistics` calls are recorded — stats1 and stats2
        // both land in `statistics_to_set` (the catalog applies the
        // remove on top per the TableUpdate ordering committed below).
        assert_eq!(action.statistics_to_set.len(), 2);
        assert!(
            action
                .statistics_to_set
                .iter()
                .any(|sf| *sf == statistics_file_1)
        );
        assert!(
            action
                .statistics_to_set
                .iter()
                .any(|sf| *sf == statistics_file_2)
        );
        assert!(
            action
                .statistics_to_remove
                .contains(&3055729675574597004i64)
        );
    }

    #[test]
    fn test_set_single_statistics() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);

        let statistics_file = StatisticsFile {
            snapshot_id: 1234567890i64,
            statistics_path: "s3://a/b/stats1.puffin".to_string(),
            file_size_in_bytes: 500,
            file_footer_size_in_bytes: 50,
            key_metadata: None,
            blob_metadata: vec![],
        };

        // Set statistics
        let tx = tx
            .update_statistics()
            .set_statistics(statistics_file.clone())
            .apply(tx)
            .unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdateStatisticsAction>()
            .unwrap();

        // Verify that the statistics file is set correctly
        assert_eq!(action.statistics_to_set.len(), 1);
        assert_eq!(action.statistics_to_set[0], statistics_file);
    }

    #[test]
    fn test_set_multiple_statistics_same_snapshot() {
        // Regression guard. Per spec, multiple StatisticsFile entries
        // may share a snapshot_id as long as paths differ -- e.g.
        // tessellate's chunk commit registers one Puffin per compacted
        // file, all sharing the new snapshot id. The previous
        // HashMap<snapshot_id, _> storage silently collapsed these
        // to one (last-write-wins), losing 24-of-25 entries per chunk.
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let snapshot_id = 9876543210i64;
        let mk = |path: &str| StatisticsFile {
            snapshot_id,
            statistics_path: path.to_string(),
            file_size_in_bytes: 100,
            file_footer_size_in_bytes: 10,
            key_metadata: None,
            blob_metadata: vec![],
        };

        let tx = tx
            .update_statistics()
            .set_statistics(mk("s3://a/b/p1.puffin"))
            .set_statistics(mk("s3://a/b/p2.puffin"))
            .set_statistics(mk("s3://a/b/p3.puffin"))
            .apply(tx)
            .unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdateStatisticsAction>()
            .unwrap();
        assert_eq!(action.statistics_to_set.len(), 3);
        let paths: Vec<&str> = action
            .statistics_to_set
            .iter()
            .map(|s| s.statistics_path.as_str())
            .collect();
        assert!(paths.contains(&"s3://a/b/p1.puffin"));
        assert!(paths.contains(&"s3://a/b/p2.puffin"));
        assert!(paths.contains(&"s3://a/b/p3.puffin"));
    }

    #[test]
    fn test_no_statistics_set() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);

        // No statistics are set or removed
        let tx = tx.update_statistics().apply(tx).unwrap();

        let action = (*tx.actions[0])
            .downcast_ref::<UpdateStatisticsAction>()
            .unwrap();

        // Verify that no statistics are set
        assert!(action.statistics_to_set.is_empty());
        assert!(action.statistics_to_remove.is_empty());
    }
}
