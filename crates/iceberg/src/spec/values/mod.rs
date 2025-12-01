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

//! This module contains Iceberg value types

pub(crate) mod datum;
mod literal;
mod map;
mod primitive;
pub(crate) mod serde;
mod struct_value;
mod temporal;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use datum::Datum;
pub use literal::Literal;
pub use map::Map;
pub use primitive::PrimitiveLiteral;
pub(crate) use serde::_serde::RawLiteral;
pub use struct_value::Struct;

// =============================================================================
// Position Delete File Metadata Column IDs
// =============================================================================
// These are reserved field IDs from the Iceberg spec for position delete files.
// The spec reserves Integer.MAX_VALUE - (101-200) for delete file metadata columns.
// Reference: https://iceberg.apache.org/spec/#position-delete-files

/// Field ID for the `file_path` column in position delete files.
/// This column contains the path of the data file where the row to be deleted exists.
pub const DELETE_FILE_PATH_FIELD_ID: i32 = i32::MAX - 101; // 2147483546

/// Field ID for the `pos` column in position delete files.
/// This column contains the 0-indexed row position within the data file.
pub const DELETE_FILE_POS_FIELD_ID: i32 = i32::MAX - 102; // 2147483545
