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

//! A deterministic byte-level fault-injection opendal [`Layer`] for DST.
//!
//! It sits BELOW iceberg's `FileIO` — on the object-store `Operator` — so it can
//! fail the raw read/write/delete operations that the commit and scan paths perform:
//! a manifest write that fails mid-commit, a manifest read that fails during scan
//! planning, an orphan-GC delete that fails. This is strictly deeper than the
//! catalog-level `FaultyCatalog` (which can only fail the register/`update_table`
//! step) and catches storage faults that never reach the catalog at all.
//!
//! All knobs are exhaustible counters, so a test says "fail the next N writes" and
//! then observes recovery; combined with a seeded schedule this is replayable.

use std::sync::{Arc, Mutex};

use opendal::raw::{
    Access, Layer, LayeredAccess, OpList, OpRead, OpWrite, RpDelete, RpList, RpRead, RpWrite,
};
use opendal::services::MemoryConfig;
use opendal::{Error as OpError, ErrorKind as OpErrorKind, Operator};

/// A deterministic fault schedule shared between a test and the layer.
#[derive(Debug, Default)]
pub(crate) struct FaultController {
    state: Mutex<FaultState>,
}

#[derive(Debug, Default)]
struct FaultState {
    fail_writes: usize,
    fail_reads: usize,
    fail_deletes: usize,
    fail_write_substr: Option<(String, usize)>,
    fail_read_substr: Option<(String, usize)>,
    writes: usize,
    reads: usize,
    deletes: usize,
}

impl FaultController {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Fail the next `n` writes (opening a writer) with a non-retryable error.
    pub(crate) fn fail_next_writes(&self, n: usize) {
        self.state.lock().unwrap().fail_writes = n;
    }
    /// Fail the next `n` reads with a non-retryable error.
    pub(crate) fn fail_next_reads(&self, n: usize) {
        self.state.lock().unwrap().fail_reads = n;
    }
    /// Fail the next `n` deletes with a non-retryable error.
    #[allow(dead_code)]
    pub(crate) fn fail_next_deletes(&self, n: usize) {
        self.state.lock().unwrap().fail_deletes = n;
    }
    /// Fail the next `n` reads whose path contains `substr` (e.g. `.avro` manifests).
    #[allow(dead_code)]
    pub(crate) fn fail_reads_containing(&self, substr: &str, n: usize) {
        self.state.lock().unwrap().fail_read_substr = Some((substr.to_string(), n));
    }
    /// Fail the next `n` writes whose path contains `substr`.
    #[allow(dead_code)]
    pub(crate) fn fail_writes_containing(&self, substr: &str, n: usize) {
        self.state.lock().unwrap().fail_write_substr = Some((substr.to_string(), n));
    }
    /// Total writes observed (for assertions about retry).
    #[allow(dead_code)]
    pub(crate) fn writes(&self) -> usize {
        self.state.lock().unwrap().writes
    }
    /// Total reads observed.
    pub(crate) fn reads(&self) -> usize {
        self.state.lock().unwrap().reads
    }

    fn should_fail_write(&self, path: &str) -> bool {
        let mut s = self.state.lock().unwrap();
        s.writes += 1;
        let mut fail = false;
        if s.fail_writes > 0 {
            s.fail_writes -= 1;
            fail = true;
        }
        if let Some((sub, n)) = s.fail_write_substr.as_mut() {
            let matches = *n > 0 && path.contains(sub.as_str());
            if matches {
                *n -= 1;
                fail = true;
            }
        }
        fail
    }

    fn should_fail_read(&self, path: &str) -> bool {
        let mut s = self.state.lock().unwrap();
        s.reads += 1;
        let mut fail = false;
        if s.fail_reads > 0 {
            s.fail_reads -= 1;
            fail = true;
        }
        if let Some((sub, n)) = s.fail_read_substr.as_mut() {
            let matches = *n > 0 && path.contains(sub.as_str());
            if matches {
                *n -= 1;
                fail = true;
            }
        }
        fail
    }

    fn should_fail_delete(&self) -> bool {
        let mut s = self.state.lock().unwrap();
        s.deletes += 1;
        if s.fail_deletes > 0 {
            s.fail_deletes -= 1;
            return true;
        }
        false
    }
}

fn injected(op: &str, path: &str) -> OpError {
    OpError::new(
        OpErrorKind::Unexpected,
        format!("injected byte-level fault on {op} {path}"),
    )
}

/// The opendal layer that installs a [`FaultAccessor`] over any backend.
#[derive(Debug, Clone)]
pub(crate) struct FaultLayer {
    ctrl: Arc<FaultController>,
}

impl FaultLayer {
    pub(crate) fn new(ctrl: Arc<FaultController>) -> Self {
        Self { ctrl }
    }
}

impl<A: Access> Layer<A> for FaultLayer {
    type LayeredAccess = FaultAccessor<A>;

    fn layer(&self, inner: A) -> Self::LayeredAccess {
        FaultAccessor {
            inner,
            ctrl: self.ctrl.clone(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct FaultAccessor<A: Access> {
    inner: A,
    ctrl: Arc<FaultController>,
}

impl<A: Access> LayeredAccess for FaultAccessor<A> {
    type Inner = A;
    type Reader = A::Reader;
    type Writer = A::Writer;
    type Lister = A::Lister;
    type Deleter = A::Deleter;
    type Copier = A::Copier;

    fn inner(&self) -> &Self::Inner {
        &self.inner
    }

    async fn read(&self, path: &str, args: OpRead) -> opendal::Result<(RpRead, Self::Reader)> {
        if self.ctrl.should_fail_read(path) {
            return Err(injected("read", path));
        }
        self.inner.read(path, args).await
    }

    async fn write(&self, path: &str, args: OpWrite) -> opendal::Result<(RpWrite, Self::Writer)> {
        if self.ctrl.should_fail_write(path) {
            return Err(injected("write", path));
        }
        self.inner.write(path, args).await
    }

    async fn delete(&self) -> opendal::Result<(RpDelete, Self::Deleter)> {
        if self.ctrl.should_fail_delete() {
            return Err(injected("delete", ""));
        }
        self.inner.delete().await
    }

    async fn list(&self, path: &str, args: OpList) -> opendal::Result<(RpList, Self::Lister)> {
        self.inner.list(path, args).await
    }
}

/// Build an in-memory opendal [`Operator`] wrapped with the fault layer.
pub(crate) fn memory_operator_with_faults(ctrl: Arc<FaultController>) -> Operator {
    let base = Operator::from_config(MemoryConfig::default())
        .expect("memory operator config")
        .finish();
    base.layer(FaultLayer::new(ctrl))
}
