# Parquet Manifests in iceberg-rust (e6 fork)

## Goal

Replace Avro manifest files with Parquet to enable columnar projection during query planning. Same v2 semantics, different serialization format.

## Why

OTel tables have 200+ columns. Each manifest entry stores min/max statistics for every column. Avro manifests require full deserialization — the executor reads all 200 columns of stats when it only needs `service_name` bounds. Parquet manifests allow projecting just the columns needed, giving 50-100x speedup on query planning for wide tables.

## Approach

**v2-compatible Parquet manifests.** Same ManifestEntry/DataFile structs, same field IDs, same semantics. Just Parquet instead of Avro. ManifestList stays Avro (it's small and not the bottleneck).

### Parquet Schema

Top-level: `manifest_entry` (one row per entry)

```
status:                  INT32 (required)    -- field 0
snapshot_id:             INT64 (optional)    -- field 1
sequence_number:         INT64 (optional)    -- field 3
file_sequence_number:    INT64 (optional)    -- field 4
-- data_file fields (flattened to top level for projection):
content:                 INT32 (required)    -- field 134
file_path:               STRING (required)   -- field 100
file_format:             STRING (required)   -- field 101
partition:               GROUP (required)    -- field 102, schema varies by partition spec
record_count:            INT64 (required)    -- field 103
file_size_in_bytes:      INT64 (required)    -- field 104
column_sizes:            MAP<INT32, INT64> (optional)     -- field 108
value_counts:            MAP<INT32, INT64> (optional)     -- field 109
null_value_counts:       MAP<INT32, INT64> (optional)     -- field 110
nan_value_counts:        MAP<INT32, INT64> (optional)     -- field 137
lower_bounds:            MAP<INT32, BINARY> (optional)    -- field 125
upper_bounds:            MAP<INT32, BINARY> (optional)    -- field 128
key_metadata:            BINARY (optional)   -- field 131
split_offsets:           LIST<INT64> (optional) -- field 132
equality_ids:            LIST<INT32> (optional) -- field 135
sort_order_id:           INT32 (optional)    -- field 140
```

**Key design choice:** DataFile fields are flattened to the top level (not nested inside a `data_file` struct). This maximizes projection efficiency — the executor can read `file_path` + `lower_bounds` without touching any other column.

### Metadata

Manifest metadata (schema, partition_spec, format_version, content) stored as Parquet file key-value metadata, same keys as Avro user_metadata:
- `schema`: JSON schema
- `schema-id`: schema ID
- `partition-spec`: JSON partition fields
- `partition-spec-id`: spec ID
- `format-version`: "2"
- `content`: "data" or "deletes"
- `manifest-format`: "parquet" (new key to distinguish from Avro)

### Reader (Phase 1 — highest ROI)

```rust
impl Manifest {
    pub(crate) fn try_from_parquet_bytes(bs: &[u8]) -> Result<(ManifestMetadata, Vec<ManifestEntry>)> {
        // 1. Read Parquet file metadata (key-value pairs) → ManifestMetadata
        // 2. Build projection: only columns needed for current operation
        //    - Scan: status, file_path, file_format, partition, record_count,
        //            file_size_in_bytes, lower_bounds, upper_bounds, content
        //    - Full: all columns
        // 3. Read with projection → RecordBatch
        // 4. Convert RecordBatch rows → ManifestEntry structs
    }

    pub(crate) fn try_from_parquet_projected(
        bs: &[u8],
        columns: &[&str],
    ) -> Result<(ManifestMetadata, Vec<ManifestEntry>)> {
        // Projected read — only requested columns
    }
}
```

### Writer (Phase 2)

```rust
impl ManifestWriter {
    pub async fn write_manifest_file_parquet(mut self) -> Result<ManifestFile> {
        // 1. Build Arrow schema from manifest entry schema
        // 2. Convert entries to RecordBatch
        // 3. Write Parquet with file key-value metadata
        // 4. Return ManifestFile
    }
}
```

### Detection

```rust
impl ManifestFile {
    pub async fn load_manifest(&self, file_io: &FileIO) -> Result<Manifest> {
        let bytes = file_io.new_input(&self.manifest_path)?.read().await?;
        if self.manifest_path.ends_with(".parquet") {
            Manifest::parse_parquet(&bytes)
        } else {
            Manifest::parse_avro(&bytes)
        }
    }
}
```

### Implementation Order

1. **Parquet schema builder** — function to create Arrow schema from partition type
2. **Parquet reader** — `try_from_parquet_bytes()` with full and projected variants
3. **Parquet writer** — `write_manifest_file_parquet()` in ManifestWriter
4. **Format detection** — in ManifestFile::load_manifest()
5. **Wire writer** — SnapshotProducer uses Parquet manifests when format is Parquet
6. **Tests** — round-trip: write Parquet manifest, read back, compare with Avro

### Dependencies

- `parquet` crate (already in Cargo.toml via arrow-deps)
- `arrow` crate (already available)

### Backward Compatibility

- Old Avro manifests continue to work (detection by file extension)
- New manifests written as Parquet
- ManifestList still references manifests by path — no format awareness needed
- Lakekeeper doesn't inspect manifest contents — transparent change
