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

use std::mem::take;
use std::sync::Arc;

use as_any::AsAny;
use async_trait::async_trait;

use crate::table::Table;
use crate::transaction::Transaction;
use crate::{Result, TableRequirement, TableUpdate};

/// A boxed, thread-safe reference to a `TransactionAction`.
pub(crate) type BoxedTransactionAction = Arc<dyn TransactionAction>;

/// A trait representing an atomic action that can be part of a transaction.
///
/// Implementors of this trait define how a specific action is committed to a table.
/// Each action is responsible for generating the updates and requirements needed
/// to modify the table metadata.
#[async_trait]
pub(crate) trait TransactionAction: AsAny + Sync + Send {
    /// Commits this action against the provided table and returns the resulting updates.
    /// NOTE: This function is intended for internal use only and should not be called directly by users.
    ///
    /// # Arguments
    ///
    /// * `table` - The current state of the table this action should apply to.
    ///
    /// # Returns
    ///
    /// An `ActionCommit` containing table updates and table requirements,
    /// or an error if the commit fails.
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit>;

    /// Short stable name for logs and metrics.
    ///
    /// Exists because `actions_ms` is emitted per transaction with only a
    /// positional `per_action_ms` array, which makes an expensive action
    /// impossible to identify without guessing. That cost three
    /// mis-targeted instrumentation rounds: `replace_data_files`'
    /// commit_v4 path measures 61-96ms while single-action commits in the
    /// same tick reach 36s, so the expensive one was a DIFFERENT action all
    /// along. Naming them removes the guesswork.
    ///
    /// Defaulted so no implementor is forced to change; override where the
    /// action can appear in a hot commit path.
    fn action_name(&self) -> &'static str {
        "unknown"
    }

    /// Whether this action must disable the commit retry loop. An action returns
    /// `true` when a retry after a commit conflict could reuse now-stale inputs and
    /// corrupt data. `ReplaceDataFiles` overrides this: a retry with a stale
    /// delete-file list can duplicate or resurrect data, so the whole transaction
    /// must fail fast and let the caller re-plan against the fresh table.
    fn disables_retry(&self) -> bool {
        false
    }
}

/// A helper trait for applying a `TransactionAction` to a `Transaction`.
///
/// This is implemented for all `TransactionAction` types
/// to allow easy chaining of actions into a transaction context.
pub trait ApplyTransactionAction {
    /// Adds this action to the given transaction.
    ///
    /// # Arguments
    ///
    /// * `tx` - The transaction to apply the action to.
    ///
    /// # Returns
    ///
    /// The modified transaction containing this action, or an error if the operation fails.
    fn apply(self, tx: Transaction) -> Result<Transaction>;
}

impl<T: TransactionAction + 'static> ApplyTransactionAction for T {
    fn apply(self, mut tx: Transaction) -> Result<Transaction>
    where Self: Sized {
        if self.disables_retry() {
            tx.disable_retry = true;
        }
        tx.actions.push(Arc::new(self));
        Ok(tx)
    }
}

/// The result of committing a `TransactionAction`.
///
/// This struct contains the updates to apply to the table's metadata
/// and any preconditions that must be satisfied before the update can be committed.
pub struct ActionCommit {
    updates: Vec<TableUpdate>,
    requirements: Vec<TableRequirement>,
    created_manifest_paths: Vec<String>,
    /// Cached root manifest entries with the snapshot_id they were built for.
    root_manifest_entries: Option<(Option<i64>, Vec<crate::spec::root_manifest::RootManifestEntry>)>,
}

impl ActionCommit {
    /// Creates a new `ActionCommit` from the given updates and requirements.
    pub fn new(updates: Vec<TableUpdate>, requirements: Vec<TableRequirement>) -> Self {
        Self {
            updates,
            requirements,
            created_manifest_paths: Vec::new(),
            root_manifest_entries: None,
        }
    }

    /// Consumes and returns the list of table updates.
    pub fn take_updates(&mut self) -> Vec<TableUpdate> {
        take(&mut self.updates)
    }

    /// Consumes and returns the list of table requirements.
    pub fn take_requirements(&mut self) -> Vec<TableRequirement> {
        take(&mut self.requirements)
    }

    /// Sets the manifest paths created during this action.
    pub fn with_manifest_paths(mut self, paths: Vec<String>) -> Self {
        self.created_manifest_paths = paths;
        self
    }

    /// Consumes and returns the list of created manifest paths.
    pub fn take_manifest_paths(&mut self) -> Vec<String> {
        take(&mut self.created_manifest_paths)
    }

    /// Sets the cached root manifest entries produced during this action,
    /// alongside the snapshot_id they were built for (used for cache validation).
    pub fn with_root_manifest_entries(mut self, snapshot_id: Option<i64>, entries: Vec<crate::spec::root_manifest::RootManifestEntry>) -> Self {
        self.root_manifest_entries = Some((snapshot_id, entries));
        self
    }

    /// Consumes and returns the cached root manifest entries with their snapshot_id.
    pub fn take_root_manifest_entries(&mut self) -> Option<(Option<i64>, Vec<crate::spec::root_manifest::RootManifestEntry>)> {
        self.root_manifest_entries.take()
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;
    use std::sync::Arc;

    use as_any::Downcast;
    use async_trait::async_trait;
    use uuid::Uuid;

    use crate::table::Table;
    use crate::transaction::Transaction;
    use crate::transaction::action::{ActionCommit, ApplyTransactionAction, TransactionAction};
    use crate::transaction::tests::make_v2_table;
    use crate::{Result, TableRequirement, TableUpdate};

    struct TestAction;

    #[async_trait]
    impl TransactionAction for TestAction {
        async fn commit(self: Arc<Self>, _table: &Table) -> Result<ActionCommit> {
            Ok(ActionCommit::new(
                vec![TableUpdate::SetLocation {
                    location: String::from("s3://bucket/prefix/table/"),
                }],
                vec![TableRequirement::UuidMatch {
                    uuid: Uuid::from_str("9c12d441-03fe-4693-9a96-a0705ddf69c1")?,
                }],
            ))
        }
    }

    #[tokio::test]
    async fn test_commit_transaction_action() {
        let table = make_v2_table();
        let action = TestAction;

        let mut action_commit = Arc::new(action).commit(&table).await.unwrap();

        let updates = action_commit.take_updates();
        let requirements = action_commit.take_requirements();

        assert_eq!(updates[0], TableUpdate::SetLocation {
            location: String::from("s3://bucket/prefix/table/")
        });
        assert_eq!(requirements[0], TableRequirement::UuidMatch {
            uuid: Uuid::from_str("9c12d441-03fe-4693-9a96-a0705ddf69c1").unwrap()
        });
    }

    #[test]
    fn test_apply_transaction_action() {
        let table = make_v2_table();
        let action = TestAction;
        let tx = Transaction::new(&table);

        let updated_tx = action.apply(tx).unwrap();
        // There should be one action in the transaction now
        assert_eq!(updated_tx.actions.len(), 1);

        (*updated_tx.actions[0])
            .downcast_ref::<TestAction>()
            .expect("TestAction was not applied to Transaction!");
    }

    #[test]
    fn test_action_commit() {
        // Create dummy updates and requirements
        let location = String::from("s3://bucket/prefix/table/");
        let uuid = Uuid::new_v4();
        let updates = vec![TableUpdate::SetLocation { location }];
        let requirements = vec![TableRequirement::UuidMatch { uuid }];

        let mut action_commit = ActionCommit::new(updates.clone(), requirements.clone());

        let taken_updates = action_commit.take_updates();
        let taken_requirements = action_commit.take_requirements();

        // Check values are returned correctly
        assert_eq!(taken_updates, updates);
        assert_eq!(taken_requirements, requirements);

        assert!(action_commit.take_updates().is_empty());
        assert!(action_commit.take_requirements().is_empty());
    }
}
