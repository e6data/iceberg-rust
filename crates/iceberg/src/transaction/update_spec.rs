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
//! This action covers the additive case (new partition fields appended to
//! the table's default partition spec). Removals, renames, and field-id
//! reassignment are not supported — those require Iceberg's full
//! UpdatePartitionSpec API. The intended caller is a streaming sink that
//! wants to start partitioning by an additional column without rewriting
//! existing data; Iceberg supports two specs coexisting and existing
//! files keep their original spec id.

use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::{Transform, UnboundPartitionField, UnboundPartitionSpec};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableUpdate};

/// A transaction action that adds new partition fields to a table's
/// default partition spec.
///
/// The new spec is built as `current default spec ∪ requested fields`
/// and committed via `TableUpdate::AddSpec` followed by
/// `TableUpdate::SetDefaultSpec { spec_id: -1 }` (last-added). Field IDs
/// are assigned by the metadata builder; spec-id deduplication is also
/// handled there, so re-applying the same evolution twice is a no-op.
pub struct UpdateSpecAction {
    new_fields: Vec<(String, String, Transform)>,
}

impl UpdateSpecAction {
    pub(crate) fn new() -> Self {
        Self {
            new_fields: Vec::new(),
        }
    }

    /// Append a partition field to the default spec.
    ///
    /// `source_name` is the schema column the partition value is derived
    /// from. `target_name` is the partition field's display name (used in
    /// path segments like `{target_name}=value/`). `transform` describes
    /// how to derive the partition value from the source column.
    ///
    /// Errors are deferred to commit time so the call site can chain
    /// `add_field` for many fields without per-call error handling.
    pub fn add_field(
        mut self,
        source_name: impl Into<String>,
        target_name: impl Into<String>,
        transform: Transform,
    ) -> Self {
        self.new_fields
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
        if self.new_fields.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        let metadata = table.metadata();
        let schema = metadata.current_schema();
        let current_spec = metadata.default_partition_spec();

        // Skip fields whose target_name already exists in the current
        // spec. This makes the action idempotent so concurrent or
        // restarting writers all calling `add_field("workspace", ...)`
        // race safely — the second commit becomes a no-op rather than
        // failing with "already exists". Mirrors UpdateSchemaAction's
        // skip-on-duplicate-name behavior; conflict detection on
        // (source_id, transform) mismatch is left to the metadata
        // builder's `add_partition_spec` validator.
        let mut additions: Vec<UnboundPartitionField> = Vec::new();
        for (source_name, target_name, transform) in &self.new_fields {
            if current_spec
                .fields()
                .iter()
                .any(|f| &f.name == target_name)
            {
                continue;
            }
            let source_id = schema.field_id_by_name(source_name).ok_or_else(|| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot add partition field '{target_name}': source column \
                         '{source_name}' not found in current schema"
                    ),
                )
            })?;
            additions.push(UnboundPartitionField {
                source_id,
                field_id: None,
                name: target_name.clone(),
                transform: transform.clone(),
            });
        }

        if additions.is_empty() {
            // All requested fields already exist in the default spec.
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // Build the target spec: existing fields, then new ones.
        // `From<PartitionSpec>` preserves source_id / field_id / name /
        // transform for existing fields, so the result is a strict
        // superset that the metadata builder will accept.
        let mut combined: Vec<UnboundPartitionField> = current_spec
            .fields()
            .iter()
            .cloned()
            .map(UnboundPartitionField::from)
            .collect();
        combined.extend(additions);

        let new_spec = UnboundPartitionSpec {
            spec_id: None,
            fields: combined,
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
        assert_eq!(action.new_fields.len(), 2);
        assert_eq!(action.new_fields[0].0, "x");
        assert_eq!(action.new_fields[0].1, "x");
        assert_eq!(action.new_fields[1].1, "y_bucket");
    }

    #[tokio::test]
    async fn test_commit_emits_add_spec_and_set_default() {
        // V2 fixture: default spec-id=0 = [{x identity}], schema has x/y/z (long).
        // Adding "y" (identity) should produce AddSpec[x identity, y identity]
        // and SetDefaultSpec{-1}.
        let table = make_v2_table();
        let action = Arc::new(
            UpdateSpecAction::new()
                .add_field("y", "y", Transform::Identity),
        );

        let mut commit = action.commit(&table).await.unwrap();
        let updates = commit.take_updates();
        assert_eq!(updates.len(), 2, "expected AddSpec + SetDefaultSpec");

        match &updates[0] {
            TableUpdate::AddSpec { spec } => {
                assert_eq!(spec.fields().len(), 2);
                assert_eq!(spec.fields()[0].name, "x");
                assert_eq!(spec.fields()[0].source_id, 1);
                assert_eq!(spec.fields()[0].transform, Transform::Identity);
                assert_eq!(spec.fields()[1].name, "y");
                assert_eq!(spec.fields()[1].source_id, 2);
                assert_eq!(spec.fields()[1].transform, Transform::Identity);
            }
            other => panic!("expected AddSpec, got {other:?}"),
        }
        match &updates[1] {
            TableUpdate::SetDefaultSpec { spec_id } => assert_eq!(*spec_id, -1),
            other => panic!("expected SetDefaultSpec, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_commit_idempotent_when_field_present() {
        // "x" is already in spec 0; adding it again is a no-op.
        let table = make_v2_table();
        let action = Arc::new(
            UpdateSpecAction::new()
                .add_field("x", "x", Transform::Identity),
        );

        let mut commit = action.commit(&table).await.unwrap();
        assert!(
            commit.take_updates().is_empty(),
            "re-adding existing partition field should produce zero updates"
        );
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
