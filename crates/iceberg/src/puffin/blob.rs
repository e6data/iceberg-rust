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

use typed_builder::TypedBuilder;

/// A serialized form of a "compact" Theta sketch produced by the Apache DataSketches library.
pub const APACHE_DATASKETCHES_THETA_V1: &str = "apache-datasketches-theta-v1";
/// A serialized form of a deletion vector.
pub const DELETION_VECTOR_V1: &str = "deletion-vector-v1";

// -----------------------------------------------------------------------------
// e6-fork extensions. Not part of the upstream Iceberg puffin spec; identified
// by their string types so any upstream reader that doesn't know them can skip.
// -----------------------------------------------------------------------------

/// Per-partition label-value inverted index (Lever A). Payload format lives
/// outside this crate — writer in `laminar`'s puffin_index module, reader in
/// `e6-native-executor`'s e6-puffin-indexes submodule. The fork only registers
/// this string so both sides agree on the blob type name. See
/// `LEVER_A_DESIGN.md` in laminar for the payload spec (roaring-bitmap-per-
/// (label_key, label_value)) and rollout gates.
pub const LABEL_VALUE_INDEX_V1: &str = "label-value-index-v1";

/// The blob
#[derive(Debug, PartialEq, Clone, TypedBuilder)]
pub struct Blob {
    pub(crate) r#type: String,
    pub(crate) fields: Vec<i32>,
    pub(crate) snapshot_id: i64,
    pub(crate) sequence_number: i64,
    pub(crate) data: Vec<u8>,
    pub(crate) properties: HashMap<String, String>,
}

impl Blob {
    #[inline]
    /// See blob types: https://iceberg.apache.org/puffin-spec/#blob-types
    pub fn blob_type(&self) -> &str {
        &self.r#type
    }

    #[inline]
    /// List of field IDs the blob was computed for; the order of items is used to compute sketches stored in the blob.
    pub fn fields(&self) -> &[i32] {
        &self.fields
    }

    #[inline]
    /// ID of the Iceberg table's snapshot the blob was computed from
    pub fn snapshot_id(&self) -> i64 {
        self.snapshot_id
    }

    #[inline]
    /// Sequence number of the Iceberg table's snapshot the blob was computed from
    pub fn sequence_number(&self) -> i64 {
        self.sequence_number
    }

    #[inline]
    /// The uncompressed blob data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    /// Arbitrary meta-information about the blob
    pub fn properties(&self) -> &HashMap<String, String> {
        &self.properties
    }
}
