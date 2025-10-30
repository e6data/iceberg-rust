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

//! Schema evolution transaction action.
//!
//! This module provides the `UpdateSchemaAction` for evolving table schemas,
//! including adding, deleting, renaming, and modifying columns.
//!
//! # Overview
//!
//! Schema evolution allows tables to change their schema over time while maintaining
//! backward compatibility with existing data files. This is achieved through:
//!
//! - **Type Promotion**: Widening primitive types (int→long, float→double, decimal precision)
//! - **Adding Columns**: New optional or required fields (with validation)
//! - **Deleting Columns**: Removing fields from the schema
//! - **Renaming Columns**: Changing field names while preserving field IDs
//! - **Column Moves**: Reordering fields within structs
//! - **Nullability Changes**: Making fields optional or required
//! - **Identifier Fields**: Setting fields used for row uniqueness
//!
//! # Example
//!
//! ```ignore
//! use iceberg::transaction::Transaction;
//! use iceberg::spec::PrimitiveType;
//!
//! let tx = Transaction::new(&table);
//! let action = tx
//!     .update_schema()
//!     .add_column(None, "email", Type::Primitive(PrimitiveType::String))?
//!     .rename_column("name", "full_name")?
//!     .update_column_type("age", PrimitiveType::Long)?;
//!
//! let tx = action.apply(tx)?;
//! let updated_table = tx.commit(&catalog).await?;
//! ```
//!
//! # Implementation Details
//!
//! The implementation follows the Iceberg Java reference implementation and uses
//! a visitor pattern to apply changes to the schema tree. Changes are tracked
//! internally and applied atomically when `commit()` is called.
//!
//! See also: [`is_promotion_allowed`](crate::spec::is_promotion_allowed) for type promotion rules.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::spec::{
    ListType, MapType, NestedField, NestedFieldRef, PrimitiveType, Schema, StructType, Type,
    is_promotion_allowed,
};
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Represents a column move operation within a struct.
#[derive(Debug, Clone)]
enum Move {
    /// Move column to the first position
    First { field_id: i32 },
    /// Move column before another column
    Before {
        field_id: i32,
        reference_field_id: i32,
    },
    /// Move column after another column
    After {
        field_id: i32,
        reference_field_id: i32,
    },
}

impl Move {
    fn field_id(&self) -> i32 {
        match self {
            Move::First { field_id } => *field_id,
            Move::Before { field_id, .. } => *field_id,
            Move::After { field_id, .. } => *field_id,
        }
    }

    fn reference_field_id(&self) -> Option<i32> {
        match self {
            Move::First { .. } => None,
            Move::Before {
                reference_field_id, ..
            } => Some(*reference_field_id),
            Move::After {
                reference_field_id, ..
            } => Some(*reference_field_id),
        }
    }
}

/// UpdateSchemaAction is a transaction action for evolving the schema of a table.
///
/// This action supports:
/// - Adding new columns (optional or required)
/// - Deleting existing columns
/// - Renaming columns
/// - Updating column types (widening only)
/// - Updating column documentation
/// - Making columns optional or required
/// - Reordering columns
/// - Setting identifier fields
#[derive(Debug)]
pub struct UpdateSchemaAction {
    /// The base schema to evolve from
    base_schema: Schema,
    /// The last column ID in the base schema
    last_column_id: i32,
    /// Whether to allow incompatible changes
    allow_incompatible_changes: bool,
    /// Whether name matching is case-sensitive
    case_sensitive: bool,

    // Tracking pending changes
    /// Field IDs to delete
    deletes: Vec<i32>,
    /// Field updates (renames, type changes, doc changes)
    updates: HashMap<i32, NestedField>,
    /// Parent ID -> list of field IDs to add as children
    parent_to_added_ids: HashMap<i32, Vec<i32>>,
    /// Added field name -> field ID mapping
    added_name_to_id: HashMap<String, i32>,
    /// Parent ID -> list of moves
    moves: HashMap<i32, Vec<Move>>,
    /// New identifier field names
    identifier_field_names: Option<HashSet<String>>,
}

/// Constant representing the table root (top-level fields)
const TABLE_ROOT_ID: i32 = -1;

impl UpdateSchemaAction {
    /// Creates a new UpdateSchemaAction.
    pub(crate) fn new(base_schema: Schema, last_column_id: i32) -> Self {
        Self {
            base_schema,
            last_column_id,
            allow_incompatible_changes: false,
            case_sensitive: true,
            deletes: vec![],
            updates: HashMap::new(),
            parent_to_added_ids: HashMap::new(),
            added_name_to_id: HashMap::new(),
            moves: HashMap::new(),
            identifier_field_names: None,
        }
    }

    /// Allow incompatible changes to the schema.
    ///
    /// Incompatible changes can cause failures when reading older data files.
    /// For example, adding a required column without a default value and attempting
    /// to read data files without that column will cause a failure.
    ///
    /// This option allows incompatible changes to be made to a schema. This should
    /// be used when the caller has validated that the change will not break existing data.
    pub fn allow_incompatible_changes(mut self) -> Self {
        self.allow_incompatible_changes = true;
        self
    }

    /// Set whether column name matching is case-sensitive.
    ///
    /// By default, matching is case-sensitive.
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// Add a new optional column to the schema.
    ///
    /// The parent name is used to find the parent struct. If parent is None,
    /// the column is added at the top level. If parent identifies a struct,
    /// the column is added to that struct. If it identifies a list, the column
    /// is added to the list element struct. If it identifies a map, the column
    /// is added to the map value struct.
    ///
    /// # Arguments
    /// * `parent` - Optional parent field name (use None for top-level)
    /// * `name` - Name of the new column
    /// * `field_type` - Type of the new column
    pub fn add_column(
        mut self,
        parent: Option<&str>,
        name: impl Into<String>,
        field_type: Type,
    ) -> Result<Self> {
        self.internal_add_column(parent, name.into(), field_type, None, true)?;
        Ok(self)
    }

    /// Add a new optional column with documentation.
    pub fn add_column_with_doc(
        mut self,
        parent: Option<&str>,
        name: impl Into<String>,
        field_type: Type,
        doc: impl Into<String>,
    ) -> Result<Self> {
        self.internal_add_column(parent, name.into(), field_type, Some(doc.into()), true)?;
        Ok(self)
    }

    /// Add a new required column to the schema.
    ///
    /// Adding a required column is an incompatible change that can break reading
    /// older data files. To suppress errors, call `allow_incompatible_changes()`.
    pub fn add_required_column(
        mut self,
        parent: Option<&str>,
        name: impl Into<String>,
        field_type: Type,
    ) -> Result<Self> {
        self.internal_add_column(parent, name.into(), field_type, None, false)?;
        Ok(self)
    }

    /// Add a new required column with documentation.
    pub fn add_required_column_with_doc(
        mut self,
        parent: Option<&str>,
        name: impl Into<String>,
        field_type: Type,
        doc: impl Into<String>,
    ) -> Result<Self> {
        self.internal_add_column(parent, name.into(), field_type, Some(doc.into()), false)?;
        Ok(self)
    }

    /// Internal method to add a column.
    fn internal_add_column(
        &mut self,
        parent: Option<&str>,
        name: String,
        field_type: Type,
        doc: Option<String>,
        is_optional: bool,
    ) -> Result<()> {
        let (parent_id, full_name) = if let Some(parent_name) = parent {
            let parent_field = self.find_field(parent_name)?;

            // Determine the actual parent for adding fields
            let actual_parent = match parent_field.field_type.as_ref() {
                Type::Map(map_type) => {
                    // Add to map value
                    &map_type.value_field
                }
                Type::List(list_type) => {
                    // Add to list element
                    &list_type.element_field
                }
                Type::Struct(_) => parent_field,
                _ => {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Cannot add column to non-struct type: {}", parent_name),
                    ));
                }
            };

            if !actual_parent.field_type.is_struct() {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot add to non-struct column: {}: {:?}",
                        parent_name, actual_parent.field_type
                    ),
                ));
            }

            if self.deletes.contains(&actual_parent.id) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    format!(
                        "Cannot add to a column that will be deleted: {}",
                        parent_name
                    ),
                ));
            }

            let parent_name_str = self
                .base_schema
                .name_by_field_id(actual_parent.id)
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot find parent column name for id: {}",
                            actual_parent.id
                        ),
                    )
                })?;
            let full_name = format!("{}.{}", parent_name_str, name);

            // Check if field already exists
            let existing_field_name = format!("{}.{}", parent_name, name);
            if let Ok(existing) = self.find_field(&existing_field_name) {
                if !self.deletes.contains(&existing.id) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot add column, name already exists: {}.{}",
                            parent_name, name
                        ),
                    ));
                }
            }

            (actual_parent.id, full_name)
        } else {
            // Top-level column
            if let Ok(existing) = self.find_field(&name) {
                if !self.deletes.contains(&existing.id) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        format!("Cannot add column, name already exists: {}", name),
                    ));
                }
            }
            (TABLE_ROOT_ID, name.clone())
        };

        // Validate incompatible change
        if !is_optional && !self.allow_incompatible_changes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Incompatible change: cannot add required column without allowing incompatible changes: {}",
                    full_name
                ),
            ));
        }

        // Assign new field ID
        self.last_column_id += 1;
        let new_id = self.last_column_id;

        // Create new field
        let mut new_field = if is_optional {
            NestedField::optional(new_id, &name, field_type)
        } else {
            NestedField::required(new_id, &name, field_type)
        };

        if let Some(doc_str) = doc {
            new_field = new_field.with_doc(doc_str);
        }

        // Track the addition
        self.added_name_to_id.insert(full_name, new_id);
        self.updates.insert(new_id, new_field);
        self.parent_to_added_ids
            .entry(parent_id)
            .or_insert_with(Vec::new)
            .push(new_id);

        Ok(())
    }

    /// Delete a column from the schema.
    ///
    /// The name is used to find the column to delete.
    pub fn delete_column(mut self, name: impl AsRef<str>) -> Result<Self> {
        let field = self.find_field(name.as_ref())?;

        // Check for conflicts
        if self.parent_to_added_ids.contains_key(&field.id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot delete a column that has additions: {}",
                    name.as_ref()
                ),
            ));
        }

        if self.updates.contains_key(&field.id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot delete a column that has updates: {}", name.as_ref()),
            ));
        }

        self.deletes.push(field.id);
        Ok(self)
    }

    /// Rename a column in the schema.
    ///
    /// The old name is used to find the column, and it will be renamed to new_name.
    /// Columns may be updated and renamed in the same schema update.
    pub fn rename_column(
        mut self,
        old_name: impl AsRef<str>,
        new_name: impl Into<String>,
    ) -> Result<Self> {
        let field = self.find_field(old_name.as_ref())?;
        let new_name = new_name.into();

        if self.deletes.contains(&field.id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot rename a column that will be deleted: {}",
                    old_name.as_ref()
                ),
            ));
        }

        // Merge with existing update or create new one
        let base_field = if let Some(existing_update) = self.updates.get(&field.id) {
            existing_update
        } else {
            field.as_ref()
        };

        let updated_field = NestedField {
            id: base_field.id,
            name: new_name,
            required: base_field.required,
            field_type: base_field.field_type.clone(),
            doc: base_field.doc.clone(),
            initial_default: base_field.initial_default.clone(),
            write_default: base_field.write_default.clone(),
        };

        self.updates.insert(field.id, updated_field);
        Ok(self)
    }

    /// Update a column type to a new primitive type.
    ///
    /// Only widening type changes are allowed (e.g., int -> long, float -> double).
    pub fn update_column_type(
        mut self,
        name: impl AsRef<str>,
        new_type: PrimitiveType,
    ) -> Result<Self> {
        let field = self.find_for_update(name.as_ref())?;

        if self.deletes.contains(&field.id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot update a column that will be deleted: {}",
                    name.as_ref()
                ),
            ));
        }

        let new_type_wrapped = Type::Primitive(new_type);

        // Check if type change is needed
        if field.field_type.as_ref() == &new_type_wrapped {
            return Ok(self);
        }

        // Validate type promotion
        if !is_promotion_allowed(field.field_type.as_ref(), &new_type_wrapped) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot change column type: {}: {:?} -> {:?}",
                    name.as_ref(),
                    field.field_type,
                    new_type_wrapped
                ),
            ));
        }

        // Create updated field
        let updated_field = NestedField {
            id: field.id,
            name: field.name.clone(),
            required: field.required,
            field_type: Box::new(new_type_wrapped),
            doc: field.doc.clone(),
            initial_default: field.initial_default.clone(),
            write_default: field.write_default.clone(),
        };
        self.updates.insert(field.id, updated_field);

        Ok(self)
    }

    /// Update the documentation for a column.
    pub fn update_column_doc(
        mut self,
        name: impl AsRef<str>,
        doc: impl Into<String>,
    ) -> Result<Self> {
        let field = self.find_for_update(name.as_ref())?;
        let doc = doc.into();

        if self.deletes.contains(&field.id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot update a column that will be deleted: {}",
                    name.as_ref()
                ),
            ));
        }

        // Check if doc change is needed
        if field.doc.as_deref() == Some(doc.as_str()) {
            return Ok(self);
        }

        let updated_field = NestedField {
            id: field.id,
            name: field.name.clone(),
            required: field.required,
            field_type: field.field_type.clone(),
            doc: Some(doc),
            initial_default: field.initial_default.clone(),
            write_default: field.write_default.clone(),
        };
        self.updates.insert(field.id, updated_field);

        Ok(self)
    }

    /// Make a column optional.
    pub fn make_column_optional(mut self, name: impl AsRef<str>) -> Result<Self> {
        self.internal_update_column_requirement(name.as_ref(), true)?;
        Ok(self)
    }

    /// Make a column required.
    ///
    /// This is an incompatible change. Call `allow_incompatible_changes()` to suppress errors.
    pub fn require_column(mut self, name: impl AsRef<str>) -> Result<Self> {
        self.internal_update_column_requirement(name.as_ref(), false)?;
        Ok(self)
    }

    /// Internal method to update column requirement.
    fn internal_update_column_requirement(&mut self, name: &str, is_optional: bool) -> Result<()> {
        let field = self.find_for_update(name)?;

        // Check if change is needed
        if (is_optional && !field.required) || (!is_optional && field.required) {
            return Ok(());
        }

        // Validate incompatible change
        if !is_optional && !self.allow_incompatible_changes {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!(
                    "Cannot change column nullability: {}: optional -> required",
                    name
                ),
            ));
        }

        if self.deletes.contains(&field.id) {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot update a column that will be deleted: {}", name),
            ));
        }

        let updated_field = NestedField {
            id: field.id,
            name: field.name.clone(),
            required: !is_optional,
            field_type: field.field_type.clone(),
            doc: field.doc.clone(),
            initial_default: field.initial_default.clone(),
            write_default: field.write_default.clone(),
        };

        self.updates.insert(field.id, updated_field);
        Ok(())
    }

    /// Move a column to the first position in its parent struct.
    pub fn move_first(mut self, name: impl AsRef<str>) -> Result<Self> {
        let field_id = self.find_for_move(name.as_ref())?;
        let parent_id = self.find_parent_id(field_id)?;

        self.moves
            .entry(parent_id)
            .or_insert_with(Vec::new)
            .push(Move::First { field_id });

        Ok(self)
    }

    /// Move a column to directly before another column.
    pub fn move_before(
        mut self,
        name: impl AsRef<str>,
        before_name: impl AsRef<str>,
    ) -> Result<Self> {
        let field_id = self.find_for_move(name.as_ref())?;
        let reference_field_id = self.find_for_move(before_name.as_ref())?;

        if field_id == reference_field_id {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot move {} before itself", name.as_ref()),
            ));
        }

        // Validate both fields are in the same parent
        let parent_id = self.find_parent_id(field_id)?;
        let ref_parent_id = self.find_parent_id(reference_field_id)?;

        if parent_id != ref_parent_id {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot move field {} to a different struct", name.as_ref()),
            ));
        }

        self.moves
            .entry(parent_id)
            .or_insert_with(Vec::new)
            .push(Move::Before {
                field_id,
                reference_field_id,
            });

        Ok(self)
    }

    /// Move a column to directly after another column.
    pub fn move_after(
        mut self,
        name: impl AsRef<str>,
        after_name: impl AsRef<str>,
    ) -> Result<Self> {
        let field_id = self.find_for_move(name.as_ref())?;
        let reference_field_id = self.find_for_move(after_name.as_ref())?;

        if field_id == reference_field_id {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot move {} after itself", name.as_ref()),
            ));
        }

        // Validate both fields are in the same parent
        let parent_id = self.find_parent_id(field_id)?;
        let ref_parent_id = self.find_parent_id(reference_field_id)?;

        if parent_id != ref_parent_id {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot move field {} to a different struct", name.as_ref()),
            ));
        }

        self.moves
            .entry(parent_id)
            .or_insert_with(Vec::new)
            .push(Move::After {
                field_id,
                reference_field_id,
            });

        Ok(self)
    }

    /// Set the identifier fields for the table.
    ///
    /// Identifier fields are used to uniquely identify rows and must meet certain requirements.
    pub fn set_identifier_fields(
        mut self,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.identifier_field_names = Some(names.into_iter().map(|n| n.into()).collect());
        self
    }

    /// Find a field by name, considering pending updates and case sensitivity.
    fn find_field(&self, name: &str) -> Result<&NestedFieldRef> {
        let field = if self.case_sensitive {
            self.base_schema.field_by_name(name)
        } else {
            self.base_schema.field_by_name_case_insensitive(name)
        };

        field.ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot find column: {}", name),
            )
        })
    }

    /// Find a field for update, considering pending changes.
    fn find_for_update(&self, name: &str) -> Result<NestedField> {
        // First check existing fields
        if let Ok(field) = self.find_field(name) {
            // Check if there's a pending update
            if let Some(updated_field) = self.updates.get(&field.id) {
                return Ok(updated_field.clone());
            }
            return Ok(field.as_ref().clone());
        }

        // Check added fields
        if let Some(&field_id) = self.added_name_to_id.get(name) {
            if let Some(field) = self.updates.get(&field_id) {
                return Ok(field.clone());
            }
        }

        Err(Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot find column: {}", name),
        ))
    }

    /// Find field ID for move operation.
    fn find_for_move(&self, name: &str) -> Result<i32> {
        // Check added fields first
        if let Some(&field_id) = self.added_name_to_id.get(name) {
            return Ok(field_id);
        }

        // Check existing fields
        let field = self.find_field(name)?;
        Ok(field.id)
    }

    /// Find the parent ID for a field.
    fn find_parent_id(&self, _field_id: i32) -> Result<i32> {
        // For now, we only support top-level moves
        // TODO: Build parent index from base_schema to support nested moves
        Ok(TABLE_ROOT_ID)
    }
}

impl UpdateSchemaAction {
    /// Apply the pending changes to create a new schema.
    ///
    /// This method applies all pending changes (deletes, updates, adds, moves) to the
    /// base schema and returns the resulting schema.
    fn apply(&self) -> Result<Schema> {
        // Validate identifier fields before applying changes
        self.validate_identifier_fields()?;

        // Apply changes using visitor pattern
        let visitor = ApplyChanges {
            deletes: &self.deletes,
            updates: &self.updates,
            parent_to_added_ids: &self.parent_to_added_ids,
            moves: &self.moves,
        };

        let new_struct = visitor.visit_schema(&self.base_schema)?;

        // Build new schema with identifier fields
        let identifier_field_ids = self.compute_identifier_field_ids(&new_struct)?;

        Schema::builder()
            .with_schema_id(self.base_schema.schema_id() + 1)
            .with_identifier_field_ids(identifier_field_ids)
            .with_fields(new_struct.fields().to_vec())
            .build()
    }

    /// Validate that identifier fields are not deleted and meet requirements.
    fn validate_identifier_fields(&self) -> Result<()> {
        if let Some(ref identifier_names) = self.identifier_field_names {
            for name in identifier_names {
                let field = if self.case_sensitive {
                    self.base_schema.field_by_name(name)
                } else {
                    self.base_schema.field_by_name_case_insensitive(name)
                };

                if let Some(field) = field {
                    if self.deletes.contains(&field.id) {
                        return Err(Error::new(
                            ErrorKind::DataInvalid,
                            format!(
                                "Cannot delete identifier field: {}. To force deletion, \
                                also call set_identifier_fields to update identifier fields.",
                                name
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Compute identifier field IDs from names.
    fn compute_identifier_field_ids(&self, new_struct: &StructType) -> Result<Vec<i32>> {
        let identifier_names: HashSet<String> = if let Some(ref names) = self.identifier_field_names
        {
            names.clone()
        } else {
            // Get identifier field IDs from base schema and convert to names
            let mut names = HashSet::new();
            for field_id in self.base_schema.identifier_field_ids() {
                if let Some(field) = self.base_schema.field_by_id(field_id) {
                    names.insert(field.name.clone());
                }
            }
            names
        };

        let mut identifier_field_ids = Vec::new();

        for name in identifier_names.iter() {
            // Find field in new schema
            let field = new_struct
                .fields()
                .iter()
                .find(|f| {
                    if self.case_sensitive {
                        &f.name == name
                    } else {
                        f.name.eq_ignore_ascii_case(name)
                    }
                })
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!(
                            "Cannot add field {} as an identifier field: \
                            not found in current schema or added columns",
                            name
                        ),
                    )
                })?;

            identifier_field_ids.push(field.id);
        }

        Ok(identifier_field_ids)
    }
}

/// Visitor to apply schema changes.
struct ApplyChanges<'a> {
    deletes: &'a [i32],
    updates: &'a HashMap<i32, NestedField>,
    parent_to_added_ids: &'a HashMap<i32, Vec<i32>>,
    moves: &'a HashMap<i32, Vec<Move>>,
}

impl<'a> ApplyChanges<'a> {
    /// Visit a schema and apply all changes.
    fn visit_schema(&self, schema: &Schema) -> Result<StructType> {
        // Process the root struct
        let root_struct = schema.as_struct();
        let new_struct = self.visit_struct(root_struct, TABLE_ROOT_ID)?;

        Ok(new_struct)
    }

    /// Visit a struct type and apply changes.
    fn visit_struct(&self, struct_type: &StructType, parent_id: i32) -> Result<StructType> {
        let mut new_fields = Vec::new();

        // Process existing fields
        for field in struct_type.fields() {
            if let Some(new_field) = self.visit_field(field)? {
                new_fields.push(new_field);
            }
        }

        // Add new fields
        if let Some(added_ids) = self.parent_to_added_ids.get(&parent_id) {
            for &field_id in added_ids {
                if let Some(field) = self.updates.get(&field_id) {
                    new_fields.push(Arc::new(field.clone()));
                }
            }
        }

        // Apply moves
        if let Some(moves) = self.moves.get(&parent_id) {
            new_fields = self.apply_moves(new_fields, moves)?;
        }

        Ok(StructType::new(new_fields))
    }

    /// Visit a field and apply changes.
    fn visit_field(&self, field: &NestedFieldRef) -> Result<Option<NestedFieldRef>> {
        // Check if field is deleted
        if self.deletes.contains(&field.id) {
            return Ok(None);
        }

        // Get pending update for this field
        let base_field = if let Some(update) = self.updates.get(&field.id) {
            update
        } else {
            field.as_ref()
        };

        // Visit field type
        let new_type = self.visit_type(&field.field_type, field.id)?;

        // Check if there were any changes
        let type_changed = *field.field_type != new_type;
        let field_updated = base_field as *const _ != field.as_ref() as *const _;

        // Create updated field if needed
        let new_field = if type_changed || field_updated {
            Arc::new(NestedField {
                id: base_field.id,
                name: base_field.name.clone(),
                required: base_field.required,
                field_type: Box::new(new_type),
                doc: base_field.doc.clone(),
                initial_default: base_field.initial_default.clone(),
                write_default: base_field.write_default.clone(),
            })
        } else {
            // No changes
            field.clone()
        };

        Ok(Some(new_field))
    }

    /// Visit a type and apply changes.
    fn visit_type(&self, type_box: &Box<Type>, field_id: i32) -> Result<Type> {
        match type_box.as_ref() {
            Type::Struct(struct_type) => {
                let new_struct = self.visit_struct(struct_type, field_id)?;
                if new_struct == *struct_type {
                    Ok((**type_box).clone())
                } else {
                    Ok(Type::Struct(new_struct))
                }
            }
            Type::List(list_type) => {
                let element_field = &list_type.element_field;

                // Visit element field
                let new_element = self.visit_field(element_field)?.ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        "Cannot delete element type from list",
                    )
                })?;

                if Arc::ptr_eq(element_field, &new_element) {
                    Ok((**type_box).clone())
                } else {
                    Ok(Type::List(ListType {
                        element_field: new_element,
                    }))
                }
            }
            Type::Map(map_type) => {
                // Validate no changes to key
                let key_field = &map_type.key_field;
                if self.deletes.contains(&key_field.id) {
                    return Err(Error::new(ErrorKind::DataInvalid, "Cannot delete map keys"));
                }
                if self.updates.contains_key(&key_field.id) {
                    return Err(Error::new(ErrorKind::DataInvalid, "Cannot update map keys"));
                }
                if self.parent_to_added_ids.contains_key(&key_field.id) {
                    return Err(Error::new(
                        ErrorKind::DataInvalid,
                        "Cannot add fields to map keys",
                    ));
                }

                // Visit value field
                let value_field = &map_type.value_field;
                let new_value = self.visit_field(value_field)?.ok_or_else(|| {
                    Error::new(ErrorKind::DataInvalid, "Cannot delete value type from map")
                })?;

                if Arc::ptr_eq(value_field, &new_value) {
                    Ok((**type_box).clone())
                } else {
                    Ok(Type::Map(MapType {
                        key_field: key_field.clone(),
                        value_field: new_value,
                    }))
                }
            }
            Type::Primitive(_) => Ok((**type_box).clone()),
        }
    }

    /// Apply move operations to reorder fields.
    fn apply_moves(
        &self,
        mut fields: Vec<NestedFieldRef>,
        moves: &[Move],
    ) -> Result<Vec<NestedFieldRef>> {
        for mv in moves {
            // Find the field to move
            let field_pos = fields
                .iter()
                .position(|f| f.id == mv.field_id())
                .ok_or_else(|| {
                    Error::new(
                        ErrorKind::DataInvalid,
                        format!("Cannot find field to move: {}", mv.field_id()),
                    )
                })?;

            let field = fields.remove(field_pos);

            match mv {
                Move::First { .. } => {
                    fields.insert(0, field);
                }
                Move::Before {
                    reference_field_id, ..
                } => {
                    let ref_pos = fields
                        .iter()
                        .position(|f| f.id == *reference_field_id)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("Cannot find reference field: {}", reference_field_id),
                            )
                        })?;
                    fields.insert(ref_pos, field);
                }
                Move::After {
                    reference_field_id, ..
                } => {
                    let ref_pos = fields
                        .iter()
                        .position(|f| f.id == *reference_field_id)
                        .ok_or_else(|| {
                            Error::new(
                                ErrorKind::DataInvalid,
                                format!("Cannot find reference field: {}", reference_field_id),
                            )
                        })?;
                    fields.insert(ref_pos + 1, field);
                }
            }
        }

        Ok(fields)
    }
}

#[async_trait]
impl TransactionAction for UpdateSchemaAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        // Apply changes to create new schema
        let new_schema = self.apply()?;

        // Create table updates
        let updates = vec![
            TableUpdate::AddSchema {
                schema: new_schema.clone(),
            },
            TableUpdate::SetCurrentSchema { schema_id: -1 },
        ];

        // Create requirements - schema must match current
        let requirements = vec![TableRequirement::CurrentSchemaIdMatch {
            current_schema_id: table.metadata().current_schema().schema_id(),
        }];

        Ok(ActionCommit::new(updates, requirements))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{NestedField, PrimitiveType, Schema, Type};

    fn create_test_schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                NestedField::optional(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                NestedField::optional(3, "age", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    #[test]
    fn test_add_optional_column() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        let result = action.add_column(None, "email", Type::Primitive(PrimitiveType::String));

        assert!(result.is_ok());
        let action = result.unwrap();
        assert_eq!(action.last_column_id, 4);
        assert!(action.updates.contains_key(&4));
        assert!(action.parent_to_added_ids.contains_key(&TABLE_ROOT_ID));
    }

    #[test]
    fn test_add_required_column_without_allowing_incompatible() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        let result =
            action.add_required_column(None, "required_field", Type::Primitive(PrimitiveType::Int));

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .message()
                .contains("Incompatible change")
        );
    }

    #[test]
    fn test_add_required_column_with_allowing_incompatible() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3).allow_incompatible_changes();

        let result =
            action.add_required_column(None, "required_field", Type::Primitive(PrimitiveType::Int));

        assert!(result.is_ok());
    }

    #[test]
    fn test_delete_column() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        let result = action.delete_column("age");

        assert!(result.is_ok());
        let action = result.unwrap();
        assert!(action.deletes.contains(&3));
    }

    #[test]
    fn test_delete_nonexistent_column() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        let result = action.delete_column("nonexistent");

        assert!(result.is_err());
    }

    #[test]
    fn test_rename_column() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        let result = action.rename_column("name", "full_name");

        assert!(result.is_ok());
        let action = result.unwrap();
        assert!(action.updates.contains_key(&2));
        assert_eq!(action.updates.get(&2).unwrap().name, "full_name");
    }

    #[test]
    fn test_update_column_type_allowed() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        // int -> long is allowed
        let result = action.update_column_type("age", PrimitiveType::Long);

        assert!(result.is_ok());
        let action = result.unwrap();
        assert!(action.updates.contains_key(&3));
    }

    #[test]
    fn test_update_column_type_not_allowed() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        // long -> int is not allowed (narrowing)
        let result = action.update_column_type("id", PrimitiveType::Int);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .message()
                .contains("Cannot change column type")
        );
    }

    #[test]
    fn test_make_column_optional() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        // Make "id" (required) optional
        let result = action.make_column_optional("id");

        assert!(result.is_ok());
        let action = result.unwrap();
        assert!(action.updates.contains_key(&1));
        assert!(!action.updates.get(&1).unwrap().required);
    }

    #[test]
    fn test_require_column_without_allowing_incompatible() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3);

        // Make "name" (optional) required without allowing incompatible changes
        let result = action.require_column("name");

        assert!(result.is_err());
    }

    #[test]
    fn test_require_column_with_allowing_incompatible() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema, 3).allow_incompatible_changes();

        // Make "name" (optional) required with allowing incompatible changes
        let result = action.require_column("name");

        assert!(result.is_ok());
        let action = result.unwrap();
        assert!(action.updates.contains_key(&2));
        assert!(action.updates.get(&2).unwrap().required);
    }

    #[test]
    fn test_chain_operations() {
        let schema = create_test_schema();
        let result = UpdateSchemaAction::new(schema, 3)
            .add_column(None, "email", Type::Primitive(PrimitiveType::String))
            .and_then(|a| a.rename_column("name", "full_name"))
            .and_then(|a| a.update_column_type("age", PrimitiveType::Long))
            .and_then(|a| a.update_column_doc("id", "Unique identifier"));

        assert!(result.is_ok());
        let action = result.unwrap();
        assert_eq!(action.last_column_id, 4);
        assert_eq!(action.updates.len(), 4); // email, renamed name, age type, id doc
    }

    #[test]
    fn test_apply_creates_new_schema() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema.clone(), 3)
            .add_column(None, "email", Type::Primitive(PrimitiveType::String))
            .unwrap()
            .rename_column("name", "full_name")
            .unwrap();

        let result = action.apply();
        assert!(result.is_ok());

        let new_schema = result.unwrap();

        // Check that schema ID was incremented
        assert_eq!(new_schema.schema_id(), schema.schema_id() + 1);

        // Check field count (3 original + 1 added = 4)
        assert_eq!(new_schema.as_struct().fields().len(), 4);

        // Check the renamed field exists
        assert!(new_schema.field_by_name("full_name").is_some());
        assert!(new_schema.field_by_name("name").is_none());

        // Check the added field exists
        let email_field = new_schema.field_by_name("email");
        assert!(email_field.is_some());
        assert_eq!(email_field.unwrap().id, 4);
    }

    #[test]
    fn test_delete_column_from_schema() {
        let schema = create_test_schema();
        let action = UpdateSchemaAction::new(schema.clone(), 3)
            .delete_column("age")
            .unwrap();

        let result = action.apply();
        assert!(result.is_ok());

        let new_schema = result.unwrap();

        // Check field count (3 original - 1 deleted = 2)
        assert_eq!(new_schema.as_struct().fields().len(), 2);

        // Check the deleted field is gone
        assert!(new_schema.field_by_name("age").is_none());

        // Check other fields still exist
        assert!(new_schema.field_by_name("id").is_some());
        assert!(new_schema.field_by_name("name").is_some());
    }
}
