// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The runtime's errors.

use std::fmt;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::DataFusionError;

/// An error of the runtime's own. It reaches callers inside a
/// `DataFusionError::External`, and [`Error::find`] gets it back, so that a
/// caller can match on it instead of parsing a message.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The session was built without the runtime.
    NotAttached,
    /// The session already has a table by this name.
    NameTaken {
        /// The name, as SQL writes it.
        name: String,
    },
    /// A batch for a table has a schema the table's doesn't contain.
    SchemaMismatch {
        /// The table's name, as SQL writes it.
        table: String,
        /// The table's schema.
        expected: SchemaRef,
        /// The batch's schema.
        found: SchemaRef,
    },
    /// A plan ran with the id of a run that has ended, or that belongs to
    /// another session or process (D20).
    RunEnded {
        /// The run's id, from the `accelerated.run` setting.
        run: u64,
    },
}

impl Error {
    /// The runtime's error inside `error`, if it holds one.
    #[must_use]
    pub fn find(error: &DataFusionError) -> Option<&Error> {
        match error.find_root() {
            DataFusionError::External(inner) => inner.downcast_ref::<Error>(),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NotAttached => write!(
                f,
                "this session has no DataFusion Accelerated runtime: build it with \
                 with_accelerator(), and on a SessionStateBuilder call with_config() \
                 before with_accelerator(), since with_config() replaces the \
                 configuration"
            ),
            Error::NameTaken { name } => write!(f, "the session already has a table named {name}"),
            Error::SchemaMismatch {
                table,
                expected,
                found,
            } => write!(
                f,
                "a batch for the table {table} has the schema {found}, not the table's {expected}"
            ),
            Error::RunEnded { run } => write!(
                f,
                "run {run} has ended, or belongs to another session or process"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        DataFusionError::External(Box::new(error))
    }
}
