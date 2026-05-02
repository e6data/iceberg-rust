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

//! Partition-spec evolution via `add_field`.
//!
//! Each `add_field` call declares one partition field of the *target*
//! default partition spec, in the order paths should be generated. The
//! action does NOT merge with the table's current default spec —
//! callers pass the complete intended field list and the action emits
//! exactly that spec. Removals and renames vs. the prior spec are
//! handled implicitly: any field in the prior spec that isn't declared
//! here is dropped from the *new* default; existing data files keep
//! their old `spec_id` and remain readable.
//!
//! Why a full-spec replacement instead of additive append: a streaming
//! sink's partition writer builds its own spec from config in config
//! order, and stamps DataFiles with the table's current `spec_id`. If
//! the two specs disagree on field order, the partition Struct on each
//! DataFile is in the partitioner's order but interpreted under the
//! table spec's order, leaving partition-pruning metadata mis-aligned.
//! Replacing the spec keeps the two ordered identically.
//!
//! Field IDs are reused across specs when the same `(source_id,
//! transform)` partition field already exists in *any* prior spec
//! (Iceberg requires globally-unique partition field IDs in V2;
//! reusing them is the canonical "this is the same partition" signal).
//! Genuinely new fields get fresh IDs from the metadata builder.

use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::{Transform, UnboundPartitionField, UnboundPartitionSpec};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableUpdate};

/// A transaction action that sets the table's default partition spec
/// to the field list declared via `add_field` calls.
pub struct UpdateSpecAction {
    declared_fields: Vec<(String, String, Transform)>,
}

impl UpdateSpecAction {
    pub(crate) fn new() -> Self {
        Self {
            declared_fields: Vec::new(),
        }
    }

    /// Declare one partition field of the target spec.
    ///
    /// `source_name` is the schema column the partition value is derived
    /// from. `target_name` is the partition field's display name (used in
    /// path segments like `{target_name}=value/`). `transform` describes
    /// how to derive the partition value from the source column.
    ///
    /// Order matters: fields are placed in the resulting spec in the
    /// order of `add_field` calls, which determines the partition path
    /// segment order on disk.
    ///
    /// Errors are deferred to commit time so the call site can chain
    /// `add_field` for many fields without per-call error handling.
    pub fn add_field(
        mut self,
        source_name: impl Into<String>,
        target_name: impl Into<String>,
        transform: Transform,
    ) -> Self {
        self.declared_fields
            .push((source_name.into(), target_name.into(), transform));
        self
    }
}

impl Default for UpdateSpecAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for UpdateSpecAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if self.declared_fields.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        let metadata = table.metadata();
        let schema = metadata.current_schema();
        let current_spec = metadata.default_partition_spec();

        // Resolve each declared field to (source_id, target_name, transform,
        // optional reused field_id). Field IDs are reused from any prior
        // spec where the same (source_id, transform) already exists; this
        // both (a) honors Iceberg's "field IDs are stable for a given
        // partition concept" convention and (b) makes the action idempotent
        // so reruns produce a spec compatible with the existing default.
        let mut target_fields: Vec<UnboundPartitionField> =
            Vec::with_capacity(self.declared_fields.len());
        for (source_name, target_name, transform) in &self.declared_fields {
            let source_id = schema.field_id_by_name(source_name).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot add partition field '{target_name}': source column \
                         '{source_name}' not found in current schema"
                    ),
                )
            })?;

            let reused_field_id = metadata
                .partition_specs_iter()
                .flat_map(|spec| spec.fields().iter())
                .find(|f| f.source_id == source_id && &f.transform == transform)
                .map(|f| f.field_id);

            target_fields.push(UnboundPartitionField {
                source_id,
                field_id: reused_field_id,
                name: target_name.clone(),
                transform: transform.clone(),
            });
        }

        // Idempotence: if the declared spec is structurally identical to
        // the current default (same length, same source_id / name /
        // transform tuples in order), don't emit any updates.
        if target_fields.len() == current_spec.fields().len()
            && target_fields
                .iter()
                .zip(current_spec.fields().iter())
                .all(|(declared, current)| {
                    declared.source_id == current.source_id
                        && declared.name == current.name
                        && declared.transform == current.transform
                })
        {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        let new_spec = UnboundPartitionSpec {
            spec_id: None,
            fields: target_fields,
        };

        let updates = vec![
            TableUpdate::AddSpec { spec: new_spec },
            // -1 = use last-added spec as default.
            TableUpdate::SetDefaultSpec { spec_id: -1 },
        ];

        Ok(ActionCommit::new(updates, vec![]))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use as_any::Downcast;

    use crate::spec::Transform;
    use crate::transaction::Transaction;
    use crate::transaction::action::{ApplyTransactionAction, TransactionAction};
    use crate::transaction::tests::make_v2_table;
    use crate::transaction::update_spec::UpdateSpecAction;
    use crate::TableUpdate;

    #[test]
    fn test_add_field_queues_action() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let tx = tx
            .update_spec()
            .add_field("x", "x", Transform::Identity)
            .add_field("y", "y_bucket", Transform::Bucket(8))
            .apply(tx)
            .unwrap();

        assert_eq!(tx.actions.len(), 1);
        let action = (*tx.actions[0])
            .downcast_ref::<UpdateSpecAction>()
            .unwrap();
        assert_eq!(action.declared_fields.len(), 2);
        assert_eq!(action.declared_fields[0].0, "x");
        assert_eq!(action.declared_fields[0].1, "x");
        assert_eq!(action.declared_fields[1].1, "y_bucket");
    }

    #[tokio::test]
    async fn test_commit_emits_spec_in_declared_order() {
        // V2 fixture: default spec_0 = [{x identity, source_id=1}], schema has
        // x/y/z (long). Declaring [y identity, x identity] should produce a
        // spec in THAT order — y first, then x — and reuse x's existing
        // field_id from spec_0 since (source_id=1, identity) already exists.
        let table = make_v2_table();
        let action = Arc::new(
            UpdateSpecAction::new()
                .add_field("y", "y", Transform::Identity)
                .add_field("x", "x", Transform::Identity),
        );

        let mut commit = action.commit(&table).await.unwrap();
        let updates = commit.take_updates();
        assert_eq!(updates.len(), 2, "expected AddSpec + SetDefaultSpec");

        match &updates[0] {
            TableUpdate::AddSpec { spec } => {
                assert_eq!(spec.fields().len(), 2);
                assert_eq!(spec.fields()[0].name, "y", "y declared first");
                assert_eq!(spec.fields()[0].source_id, 2);
                assert_eq!(spec.fields()[0].transform, Transform::Identity);
                assert_eq!(spec.fields()[1].name, "x", "x declared second");
                assert_eq!(spec.fields()[1].source_id, 1);
                assert!(
                    spec.fields()[1].field_id.is_some(),
                    "x should have its existing field_id reused from spec_0"
                );
            }
            other => panic!("expected AddSpec, got {other:?}"),
        }
        match &updates[1] {
            TableUpdate::SetDefaultSpec { spec_id } => assert_eq!(*spec_id, -1),
            other => panic!("expected SetDefaultSpec, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_commit_idempotent_on_identical_spec() {
        // V2 fixture default spec_0 = [{x identity}]. Declaring exactly [x
        // identity] should be a no-op — same length, same fields in same
        // order.
        let table = make_v2_table();
        let action = Arc::new(
            UpdateSpecAction::new()
                .add_field("x", "x", Transform::Identity),
        );

        let mut commit = action.commit(&table).await.unwrap();
        assert!(
            commit.take_updates().is_empty(),
            "declaring the existing default spec verbatim should be a no-op"
        );
    }

    #[tokio::test]
    async fn test_commit_emits_when_only_order_differs() {
        // Declaring [y identity, x identity] vs current [x identity] is a
        // real change (different fields). Emit updates.
        let table = make_v2_table();
        let action = Arc::new(
            UpdateSpecAction::new()
                .add_field("y", "y", Transform::Identity)
                .add_field("x", "x", Transform::Identity),
        );

        let mut commit = action.commit(&table).await.unwrap();
        assert_eq!(commit.take_updates().len(), 2);
    }

    #[tokio::test]
    async fn test_commit_errors_on_unknown_source_column() {
        let table = make_v2_table();
        let action = Arc::new(
            UpdateSpecAction::new()
                .add_field("does_not_exist", "dne", Transform::Identity),
        );

        let err = match action.commit(&table).await {
            Ok(_) => panic!("expected error for unknown source column, got Ok"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("not found in current schema"),
            "expected schema lookup error, got: {err}"
        );
    }
}
