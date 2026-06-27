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

//! Tiered V4 metadata: the **bucket-index** (super-manifest).
//!
//! The bucket-index is the middle tier of a three-level metadata structure that
//! keeps streaming-ingest commits cheap while still allowing tight partition
//! pruning over long history:
//!
//! ```text
//!   root (live, rewritten every commit)
//!     └─ bucket_index_ptr ──► bucket-index (cold, rewritten only on bucket-close)
//!                               └─ refs ──► leaf manifests (immutable per bucket)
//!                                             └─ data files
//! ```
//!
//! Each bucket-index entry references one **closed** leaf manifest and carries
//! that leaf's partition summary + sequence/row counts (a [`ManifestFile`]), so
//! the planner can skip whole leaves by partition predicate before reading them.
//!
//! The hot commit path never reads or rewrites the bucket-index — it only
//! carries the pointer forward — so per-commit cost stays bounded by the live
//! window, independent of how much history has accumulated. The bucket-index is
//! appended only when a bucket closes (the cold/compaction path).
//!
//! On disk a bucket-index is, deliberately, **a root manifest that contains only
//! manifest references** (no inline entries): it reuses the root-manifest Parquet
//! encoding verbatim, so there is no second on-wire format to maintain. The
//! "is this a leaf or a bucket-index" distinction is semantic — carried by *how*
//! the file is referenced from the root, not by the bytes.

use bytes::Bytes;

use super::root_manifest::{
    read_root_manifest, write_root_manifest, RootManifestEntry, RootManifestMetadata,
};
use crate::error::Result;
use crate::io::FileIO;
use crate::spec::{ManifestFile, StructType};

/// A bucket-index: the cold tier listing references to closed leaf manifests.
#[derive(Debug, Clone)]
pub struct BucketIndex {
    metadata: RootManifestMetadata,
    leaves: Vec<ManifestFile>,
}

impl BucketIndex {
    /// Create a bucket-index from its metadata and the closed leaves it indexes.
    pub fn new(metadata: RootManifestMetadata, leaves: Vec<ManifestFile>) -> Self {
        Self { metadata, leaves }
    }

    /// Metadata (schema, partition spec, snapshot/seq) the index was written under.
    pub fn metadata(&self) -> &RootManifestMetadata {
        &self.metadata
    }

    /// The indexed closed leaf manifests.
    pub fn leaves(&self) -> &[ManifestFile] {
        &self.leaves
    }

    /// Number of indexed leaves.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Whether the index has no leaves.
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Append a newly-closed leaf manifest. Called by the cold/compaction path
    /// when a bucket closes — never on the hot commit path.
    pub fn push(&mut self, leaf: ManifestFile) {
        self.leaves.push(leaf);
    }

    /// Read-path pruning: return the leaves for which `keep` holds. The caller
    /// supplies a predicate over each leaf's partition summary / bounds (e.g.
    /// "this leaf's partition range overlaps the query"), so the planner can
    /// drop whole leaves before opening them.
    pub fn prune<F: Fn(&ManifestFile) -> bool>(&self, keep: F) -> Vec<&ManifestFile> {
        self.leaves.iter().filter(|l| keep(l)).collect()
    }
}

/// Serialize a bucket-index to Parquet bytes, reusing the root-manifest ref
/// encoding (every leaf becomes a `ManifestRef`; there are no inline entries).
pub fn write_bucket_index(
    leaves: &[ManifestFile],
    metadata: &RootManifestMetadata,
    partition_type: &StructType,
) -> Result<Vec<u8>> {
    let entries: Vec<RootManifestEntry> = leaves
        .iter()
        .cloned()
        .map(|manifest_file| RootManifestEntry::ManifestRef {
            manifest_file,
            mdv: None,
        })
        .collect();
    write_root_manifest(&entries, metadata, partition_type)
}

/// Read a bucket-index from Parquet bytes. Inline entries (which a well-formed
/// bucket-index never has) are ignored defensively.
pub fn read_bucket_index(bytes: Bytes) -> Result<BucketIndex> {
    let (metadata, entries) = read_root_manifest(bytes)?;
    let leaves = entries
        .into_iter()
        .filter_map(|e| match e {
            RootManifestEntry::ManifestRef { manifest_file, .. } => Some(manifest_file),
            RootManifestEntry::Inline(_) => None,
        })
        .collect();
    Ok(BucketIndex::new(metadata, leaves))
}

/// Resolve and read the cold bucket-index referenced by a root manifest, if any.
///
/// This is the **root → bucket-index** recursion step of the read path: given a
/// root manifest's metadata, follow its `bucket_index_path` and read the cold
/// tier. Returns `None` for flat-layout tables (no pointer), so callers can
/// treat tiered and non-tiered tables uniformly.
pub async fn load_bucket_index_for_root(
    file_io: &FileIO,
    root_meta: &RootManifestMetadata,
) -> Result<Option<BucketIndex>> {
    let Some(path) = &root_meta.bucket_index_path else {
        return Ok(None);
    };
    let bytes = file_io.new_input(path)?.read().await?;
    Ok(Some(read_bucket_index(bytes)?))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_bytes::ByteBuf;

    use super::*;
    use crate::spec::{
        FieldSummary, FormatVersion, ManifestContentType, NestedField, PartitionSpec,
        PrimitiveType, Schema, SchemaRef, Transform, Type, UnboundPartitionField,
    };

    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_fields(vec![
                    Arc::new(NestedField::required(
                        1,
                        "id",
                        Type::Primitive(PrimitiveType::Long),
                    )),
                    Arc::new(NestedField::optional(
                        2,
                        "ts",
                        Type::Primitive(PrimitiveType::Timestamptz),
                    )),
                ])
                .build()
                .unwrap(),
        )
    }

    fn test_partition_spec(schema: &SchemaRef) -> PartitionSpec {
        PartitionSpec::builder(schema.clone())
            .with_spec_id(0)
            .add_unbound_field(UnboundPartitionField {
                source_id: 1,
                field_id: None,
                name: "id".to_string(),
                transform: Transform::Identity,
            })
            .unwrap()
            .build()
            .unwrap()
    }

    fn test_metadata(schema: &SchemaRef, spec: &PartitionSpec) -> RootManifestMetadata {
        RootManifestMetadata {
            schema: schema.clone(),
            schema_id: 0,
            partition_spec: Arc::new(spec.clone()),
            format_version: FormatVersion::V2,
            snapshot_id: 100,
            sequence_number: 5,
            parent_snapshot_id: Some(99),
            bucket_index_path: None,
            prev_root_path: None,
            chain_depth: 0,
        }
    }

    fn leaf(path: &str, partitions: Option<Vec<FieldSummary>>) -> ManifestFile {
        ManifestFile {
            manifest_path: path.to_string(),
            manifest_length: 4096,
            partition_spec_id: 0,
            content: ManifestContentType::Data,
            sequence_number: 5,
            min_sequence_number: 1,
            added_snapshot_id: 100,
            added_files_count: Some(10),
            existing_files_count: Some(0),
            deleted_files_count: Some(0),
            added_rows_count: Some(1000),
            existing_rows_count: Some(0),
            deleted_rows_count: Some(0),
            partitions,
            key_metadata: None,
            first_row_id: None,
        }
    }

    fn summary(lo: &[u8], hi: &[u8]) -> FieldSummary {
        FieldSummary {
            contains_null: false,
            contains_nan: Some(false),
            lower_bound: Some(ByteBuf::from(lo.to_vec())),
            upper_bound: Some(ByteBuf::from(hi.to_vec())),
        }
    }

    #[test]
    fn round_trip_preserves_leaves() {
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let partition_type = spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &spec);

        let leaves = vec![
            leaf("s3://b/blocks/leaf-0.parquet", Some(vec![summary(&[0], &[0])])),
            leaf("s3://b/blocks/leaf-1.parquet", Some(vec![summary(&[1], &[1])])),
        ];

        let bytes = write_bucket_index(&leaves, &metadata, &partition_type).unwrap();
        let bi = read_bucket_index(Bytes::from(bytes)).unwrap();

        assert_eq!(bi.metadata().snapshot_id, 100);
        assert_eq!(bi.len(), 2);
        assert!(!bi.is_empty());
        assert_eq!(bi.leaves()[0].manifest_path, "s3://b/blocks/leaf-0.parquet");
        assert_eq!(bi.leaves()[0].added_files_count, Some(10));
        assert_eq!(bi.leaves()[1].manifest_path, "s3://b/blocks/leaf-1.parquet");
    }

    #[test]
    fn push_appends() {
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let metadata = test_metadata(&schema, &spec);

        let mut bi = BucketIndex::new(metadata, vec![]);
        assert!(bi.is_empty());
        bi.push(leaf("s3://b/blocks/leaf-0.parquet", None));
        bi.push(leaf("s3://b/blocks/leaf-1.parquet", None));
        assert_eq!(bi.len(), 2);
    }

    #[test]
    fn prune_filters_by_partition_summary() {
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let metadata = test_metadata(&schema, &spec);

        // Three leaves at partition values 1, 5, 9 (single-value summaries).
        let bi = BucketIndex::new(metadata, vec![
            leaf("leaf-1", Some(vec![summary(&[1], &[1])])),
            leaf("leaf-5", Some(vec![summary(&[5], &[5])])),
            leaf("leaf-9", Some(vec![summary(&[9], &[9])])),
        ]);

        // Keep only leaves whose lower_bound byte == 5.
        let kept = bi.prune(|mf| {
            mf.partitions
                .as_ref()
                .and_then(|p| p.first())
                .and_then(|f| f.lower_bound.as_ref())
                .map(|b| b.as_ref() == [5u8])
                .unwrap_or(false)
        });
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].manifest_path, "leaf-5");
    }

    #[test]
    fn empty_round_trip() {
        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let partition_type = spec.partition_type(&schema).unwrap();
        let metadata = test_metadata(&schema, &spec);

        let bytes = write_bucket_index(&[], &metadata, &partition_type).unwrap();
        let bi = read_bucket_index(Bytes::from(bytes)).unwrap();
        assert_eq!(bi.len(), 0);
        assert_eq!(bi.metadata().snapshot_id, 100);
    }

    #[test]
    fn bucket_index_path_survives_root_round_trip() {
        use super::super::root_manifest::{read_root_manifest, write_root_manifest};

        let schema = test_schema();
        let spec = test_partition_spec(&schema);
        let partition_type = spec.partition_type(&schema).unwrap();

        // With a pointer set.
        let mut meta = test_metadata(&schema, &spec);
        meta.bucket_index_path = Some("s3://b/metadata/bucket-index-7.parquet".to_string());
        let bytes = write_root_manifest(&[], &meta, &partition_type).unwrap();
        let (decoded, _) = read_root_manifest(Bytes::from(bytes)).unwrap();
        assert_eq!(
            decoded.bucket_index_path.as_deref(),
            Some("s3://b/metadata/bucket-index-7.parquet")
        );

        // Absent pointer (flat layout) decodes to None — backward compatible.
        let flat = test_metadata(&schema, &spec);
        let bytes = write_root_manifest(&[], &flat, &partition_type).unwrap();
        let (decoded, _) = read_root_manifest(Bytes::from(bytes)).unwrap();
        assert_eq!(decoded.bucket_index_path, None);
    }
}
