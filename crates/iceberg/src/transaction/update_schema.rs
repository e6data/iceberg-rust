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

//! Schema evolution via `add_column`.
//!
//! This action covers the additive case (new optional columns appended
//! to the table schema). Renames, drops, and type promotions are not
//! supported — those require Iceberg's full UpdateSchema API which has
//! ordering and compatibility rules beyond what's needed here. The
//! intended caller is a streaming sink that just learned about a new
//! field in an inbound batch.

use std::sync::Arc;

use async_trait::async_trait;

use crate::spec::{NestedField, Schema, Type};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableUpdate};

/// A transaction action that adds new optional columns to a table's
/// current schema.
///
/// New columns are appended with field IDs starting at
/// `current_schema.highest_field_id() + 1`. The resulting schema is
/// committed via `TableUpdate::AddSchema` followed by
/// `TableUpdate::SetCurrentSchema { schema_id: -1 }` (last-added).
pub struct UpdateSchemaAction {
    new_columns: Vec<(String, Type)>,
}

impl UpdateSchemaAction {
    pub(crate) fn new() -> Self {
        Self {
            new_columns: Vec::new(),
        }
    }

    /// Append an optional column to the schema.
    ///
    /// Errors are deferred to commit time so the call site can chain
    /// `add_column` for many fields without per-call error handling.
    pub fn add_column(mut self, name: impl Into<String>, field_type: Type) -> Self {
        self.new_columns.push((name.into(), field_type));
        self
    }
}

impl Default for UpdateSchemaAction {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl TransactionAction for UpdateSchemaAction {
    fn action_name(&self) -> &'static str {
        "update_schema"
    }

    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        if self.new_columns.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        let current_schema = table.metadata().current_schema();
        let mut next_field_id = current_schema.highest_field_id() + 1;

        // Skip columns that already exist. This makes the action idempotent
        // so concurrent writers all calling `add_column("foo", String)` race
        // safely — the second commit becomes a no-op rather than failing
        // with "already exists". Type-conflict checking is intentionally
        // not done here (additive evolution only).
        let mut new_fields = Vec::with_capacity(self.new_columns.len());
        for (name, field_type) in &self.new_columns {
            if current_schema.field_id_by_name(name).is_some() {
                continue;
            }
            new_fields.push(Arc::new(NestedField::optional(
                next_field_id,
                name.clone(),
                field_type.clone(),
            )));
            next_field_id += 1;
        }

        if new_fields.is_empty() {
            // All requested columns already exist — nothing to commit.
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // Rebuild the schema with original fields + new fields appended.
        // schema_id is left as 0; `add_schema` in the metadata builder
        // assigns the real ID via `reuse_or_create_new_schema_id`.
        let combined: Vec<_> = current_schema
            .as_struct()
            .fields()
            .iter()
            .cloned()
            .chain(new_fields.into_iter())
            .collect();

        let identifier_ids: Vec<i32> = current_schema.identifier_field_ids().collect();
        let new_schema = Schema::builder()
            .with_schema_id(0)
            .with_identifier_field_ids(identifier_ids)
            .with_fields(combined)
            .build()
            .map_err(|e| {
                Error::new(
                    ErrorKind::DataInvalid,
                    format!("Failed to build evolved schema: {e}"),
                )
            })?;

        let updates = vec![
            TableUpdate::AddSchema { schema: new_schema },
            // -1 = use last-added schema as current.
            TableUpdate::SetCurrentSchema { schema_id: -1 },
        ];

        Ok(ActionCommit::new(updates, vec![]))
    }
}

#[cfg(test)]
mod tests {
    use as_any::Downcast;

    use crate::spec::{PrimitiveType, Type};
    use crate::transaction::Transaction;
    use crate::transaction::action::ApplyTransactionAction;
    use crate::transaction::tests::make_v2_table;
    use crate::transaction::update_schema::UpdateSchemaAction;

    #[test]
    fn test_add_column_queues_action() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let tx = tx
            .update_schema()
            .add_column("app", Type::Primitive(PrimitiveType::String))
            .add_column("workspace", Type::Primitive(PrimitiveType::String))
            .apply(tx)
            .unwrap();

        assert_eq!(tx.actions.len(), 1);
        let action = (*tx.actions[0])
            .downcast_ref::<UpdateSchemaAction>()
            .unwrap();
        assert_eq!(action.new_columns.len(), 2);
        assert_eq!(action.new_columns[0].0, "app");
    }
}
