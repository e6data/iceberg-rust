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

//! Error conversion utilities for BigLake gRPC errors.

use google_cloud_biglake_v1::Error as BiglakeError;
use google_cloud_gax::error::rpc::Code;
use iceberg::{Error, ErrorKind};

/// Convert a BigLake SDK error to an Iceberg Error.
pub fn from_biglake_error(err: BiglakeError) -> Error {
    // Check for RPC status errors first
    if let Some(status) = err.status() {
        let (kind, retryable) = match status.code {
            Code::NotFound => (ErrorKind::DataInvalid, false),
            Code::AlreadyExists => (ErrorKind::DataInvalid, false),
            Code::PermissionDenied => (ErrorKind::Unexpected, false),
            Code::Unauthenticated => (ErrorKind::Unexpected, false),
            Code::InvalidArgument => (ErrorKind::DataInvalid, false),
            Code::FailedPrecondition => (ErrorKind::DataInvalid, false),
            Code::Aborted => (ErrorKind::CatalogCommitConflicts, true),
            Code::Unavailable => (ErrorKind::Unexpected, true),
            Code::DeadlineExceeded => (ErrorKind::Unexpected, true),
            Code::ResourceExhausted => (ErrorKind::Unexpected, true),
            Code::Internal => (ErrorKind::Unexpected, false),
            _ => (ErrorKind::Unexpected, false),
        };

        let mut error =
            Error::new(kind, format!("BigLake error: {}", status.message)).with_source(err);

        if retryable {
            error = error.with_retryable(true);
        }

        return error;
    }

    // Check for other error types
    if err.is_timeout() {
        return Error::new(ErrorKind::Unexpected, "BigLake request timed out")
            .with_source(err)
            .with_retryable(true);
    }

    // Generic error
    Error::new(ErrorKind::Unexpected, "BigLake error").with_source(err)
}

/// Check if a BigLake error represents a "not found" error.
pub fn is_not_found(err: &BiglakeError) -> bool {
    if let Some(status) = err.status() {
        return status.code == Code::NotFound;
    }
    false
}

/// Check if a BigLake error represents an "already exists" error.
#[allow(dead_code)]
pub fn is_already_exists(err: &BiglakeError) -> bool {
    if let Some(status) = err.status() {
        return status.code == Code::AlreadyExists;
    }
    false
}

/// Check if a BigLake error represents a commit conflict error.
#[allow(dead_code)]
pub fn is_commit_conflict(err: &BiglakeError) -> bool {
    if let Some(status) = err.status() {
        return status.code == Code::Aborted || status.code == Code::FailedPrecondition;
    }
    false
}
