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

//! OpenDAL layer that emits Prometheus metrics for Iceberg storage operations.

use std::fmt::{Debug, Formatter};
use std::sync::LazyLock;
use std::time::Instant;

use opendal::raw::{
    Access, Layer, LayeredAccess, OpCreateDir, OpList, OpRead, OpStat, OpWrite, RpCreateDir,
    RpDelete, RpList, RpRead, RpStat, RpWrite, oio,
};
use prometheus::{HistogramVec, IntCounterVec, register_histogram_vec, register_int_counter_vec};

const LABELS: &[&str] = &["op", "scheme"];
const BYTES_LABELS: &[&str] = &["direction", "scheme"];

static OPS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "iceberg_storage_ops_total",
        "Total Iceberg storage operations",
        LABELS
    )
    .expect("failed to register iceberg_storage_ops_total")
});

static OPS_ERRORS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "iceberg_storage_ops_errors_total",
        "Total Iceberg storage operation errors",
        LABELS
    )
    .expect("failed to register iceberg_storage_ops_errors_total")
});

static OPS_DURATION: LazyLock<HistogramVec> = LazyLock::new(|| {
    register_histogram_vec!(
        "iceberg_storage_ops_duration_seconds",
        "Iceberg storage operation latency",
        LABELS,
        vec![
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0
        ]
    )
    .expect("failed to register iceberg_storage_ops_duration_seconds")
});

static BYTES_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register_int_counter_vec!(
        "iceberg_storage_bytes_total",
        "Total Iceberg storage bytes transferred",
        BYTES_LABELS
    )
    .expect("failed to register iceberg_storage_bytes_total")
});

fn record_op(op: &str, scheme: &str, start: Instant, is_err: bool) {
    OPS_TOTAL.with_label_values(&[op, scheme]).inc();
    OPS_DURATION
        .with_label_values(&[op, scheme])
        .observe(start.elapsed().as_secs_f64());
    if is_err {
        OPS_ERRORS.with_label_values(&[op, scheme]).inc();
    }
}

fn record_bytes_read(scheme: &str, n: u64) {
    if n > 0 {
        BYTES_TOTAL.with_label_values(&["read", scheme]).inc_by(n);
    }
}

fn record_bytes_written(scheme: &str, n: u64) {
    if n > 0 {
        BYTES_TOTAL.with_label_values(&["write", scheme]).inc_by(n);
    }
}

/// OpenDAL [`Layer`] that records Prometheus metrics for storage operations.
#[derive(Clone)]
pub struct IcebergMetricsLayer;

impl<A: Access> Layer<A> for IcebergMetricsLayer {
    type LayeredAccess = IcebergMetricsAccess<A>;

    fn layer(&self, inner: A) -> Self::LayeredAccess {
        let scheme = inner.info().scheme().to_string();
        IcebergMetricsAccess { inner, scheme }
    }
}

pub struct IcebergMetricsAccess<A: Access> {
    inner: A,
    scheme: String,
}

impl<A: Access> Debug for IcebergMetricsAccess<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergMetricsAccess")
            .field("scheme", &self.scheme)
            .finish_non_exhaustive()
    }
}

impl<A: Access> LayeredAccess for IcebergMetricsAccess<A> {
    type Inner = A;
    type Reader = MetricsReader<A::Reader>;
    type Writer = MetricsWriter<A::Writer>;
    type Lister = A::Lister;
    type Deleter = A::Deleter;

    fn inner(&self) -> &Self::Inner {
        &self.inner
    }

    async fn create_dir(&self, path: &str, args: OpCreateDir) -> opendal::Result<RpCreateDir> {
        let start = Instant::now();
        let result = self.inner.create_dir(path, args).await;
        record_op("create_dir", &self.scheme, start, result.is_err());
        result
    }

    async fn read(&self, path: &str, args: OpRead) -> opendal::Result<(RpRead, Self::Reader)> {
        let start = Instant::now();
        let result = self.inner.read(path, args).await;
        record_op("read", &self.scheme, start, result.is_err());
        result.map(|(rp, reader)| (rp, MetricsReader::new(reader, self.scheme.clone())))
    }

    async fn write(&self, path: &str, args: OpWrite) -> opendal::Result<(RpWrite, Self::Writer)> {
        let start = Instant::now();
        let result = self.inner.write(path, args).await;
        record_op("write", &self.scheme, start, result.is_err());
        result.map(|(rp, writer)| (rp, MetricsWriter::new(writer, self.scheme.clone())))
    }

    async fn stat(&self, path: &str, args: OpStat) -> opendal::Result<RpStat> {
        let start = Instant::now();
        let result = self.inner.stat(path, args).await;
        record_op("stat", &self.scheme, start, result.is_err());
        result
    }

    async fn delete(&self) -> opendal::Result<(RpDelete, Self::Deleter)> {
        let start = Instant::now();
        let result = self.inner.delete().await;
        record_op("delete", &self.scheme, start, result.is_err());
        result
    }

    async fn list(&self, path: &str, args: OpList) -> opendal::Result<(RpList, Self::Lister)> {
        let start = Instant::now();
        let result = self.inner.list(path, args).await;
        record_op("list", &self.scheme, start, result.is_err());
        result
    }
}

pub struct MetricsReader<R> {
    inner: R,
    scheme: String,
}

impl<R> MetricsReader<R> {
    fn new(inner: R, scheme: String) -> Self {
        Self { inner, scheme }
    }
}

impl<R: oio::Read> oio::Read for MetricsReader<R> {
    async fn read(&mut self) -> opendal::Result<opendal::Buffer> {
        let result = self.inner.read().await;
        if let Ok(buf) = &result {
            record_bytes_read(&self.scheme, buf.len() as u64);
        }
        result
    }
}

pub struct MetricsWriter<W> {
    inner: W,
    scheme: String,
}

impl<W> MetricsWriter<W> {
    fn new(inner: W, scheme: String) -> Self {
        Self { inner, scheme }
    }
}

impl<W: oio::Write> oio::Write for MetricsWriter<W> {
    async fn write(&mut self, bs: opendal::Buffer) -> opendal::Result<()> {
        let n = bs.len() as u64;
        let result = self.inner.write(bs).await;
        if result.is_ok() {
            record_bytes_written(&self.scheme, n);
        }
        result
    }

    async fn close(&mut self) -> opendal::Result<opendal::Metadata> {
        self.inner.close().await
    }

    async fn abort(&mut self) -> opendal::Result<()> {
        self.inner.abort().await
    }
}
