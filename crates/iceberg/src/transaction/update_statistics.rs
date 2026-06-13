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
use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::StatisticsFile;
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Result, TableUpdate};

/// A transactional action for updating statistics files in a table.
///
/// Per the Iceberg spec, multiple `StatisticsFile` entries may share
/// the same `snapshot_id` as long as their `statistics_path` differs.
/// Storage is keyed on `(snapshot_id, statistics_path)` for
/// `set_statistics`; for `remove_statistics` the caller still removes
/// every entry for a snapshot id (matching the catalog protocol's
/// `RemoveStatistics` semantics).
pub struct UpdateStatisticsAction {
    statistics_to_set: HashMap<(i64, String), StatisticsFile>,
    statistics_to_remove: Vec<i64>,
}

impl UpdateStatisticsAction {
    pub fn new() -> Self {
        Self {
            statistics_to_set: HashMap::default(),
            statistics_to_remove: Vec::new(),
        }
    }

    /// Add a statistics file entry for its snapshot.
    ///
    /// Keyed on `(snapshot_id, statistics_path)`. Calling N times with
    /// the same snapshot_id but distinct paths adds N entries (this is
    /// the multi-stats-per-snapshot path); a second call with the same
    /// `(snapshot_id, path)` pair overwrites the prior entry, which is
    /// a true in-place update.
    pub fn set_statistics(mut self, statistics_file: StatisticsFile) -> Self {
        let key = (
            statistics_file.snapshot_id,
            statistics_file.statistics_path.clone(),
        );
        self.statistics_to_set.insert(key, statistics_file);
        self
    }

    /// Remove every statistics file entry for the given snapshot id.
    ///
    /// Matches the catalog protocol — `RemoveStatistics` is keyed by
    /// snapshot id, not by `(snapshot_id, path)`, so callers can't
    /// selectively drop just one of several entries.
    pub fn remove_statistics(mut self, snapshot_id: i64) -> Self {
        self.statistics_to_remove.push(snapshot_id);
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
        let mut updates: Vec<TableUpdate> = vec![];

        // Emit removes first so a set on the same snapshot id in the
        // same action lands AFTER the remove and survives.
        for snapshot_id in &self.statistics_to_remove {
            updates.push(TableUpdate::RemoveStatistics {
                snapshot_id: *snapshot_id,
            });
        }
        for statistics in self.statistics_to_set.values() {
            updates.push(TableUpdate::SetStatistics {
                statistics: statistics.clone(),
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

        // set stats1
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
        // stats1 should still be present in the set map (composite key
        // doesn't collide with remove), and the remove enqueues a
        // snapshot-id-scoped RemoveStatistics that hits both.
        assert!(action.statistics_to_remove.contains(&3055729675574597004i64));
        assert_eq!(
            action
                .statistics_to_set
                .get(&(
                    statistics_file_2.snapshot_id,
                    statistics_file_2.statistics_path.clone()
                ))
                .unwrap()
                .clone(),
            statistics_file_2
        );
    }

    #[test]
    fn test_multi_stats_per_snapshot() {
        // Two StatisticsFile entries sharing the same snapshot_id but
        // distinct statistics_path values must both survive — this is
        // the B2 multi-stats-per-snapshot path.
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let snap_id = 9999i64;
        let s1 = StatisticsFile {
            snapshot_id: snap_id,
            statistics_path: "s3://b/p1.puffin".to_string(),
            file_size_in_bytes: 100,
            file_footer_size_in_bytes: 10,
            key_metadata: None,
            blob_metadata: vec![],
        };
        let s2 = StatisticsFile {
            snapshot_id: snap_id,
            statistics_path: "s3://b/p2.puffin".to_string(),
            file_size_in_bytes: 200,
            file_footer_size_in_bytes: 20,
            key_metadata: None,
            blob_metadata: vec![],
        };
        let tx = tx
            .update_statistics()
            .set_statistics(s1.clone())
            .set_statistics(s2.clone())
            .apply(tx)
            .unwrap();
        let action = (*tx.actions[0])
            .downcast_ref::<UpdateStatisticsAction>()
            .unwrap();
        assert_eq!(action.statistics_to_set.len(), 2);
        assert!(
            action
                .statistics_to_set
                .contains_key(&(snap_id, s1.statistics_path))
        );
        assert!(
            action
                .statistics_to_set
                .contains_key(&(snap_id, s2.statistics_path))
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
        assert_eq!(
            action
                .statistics_to_set
                .get(&(
                    statistics_file.snapshot_id,
                    statistics_file.statistics_path.clone()
                ))
                .unwrap()
                .clone(),
            statistics_file
        );
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
    }
}
