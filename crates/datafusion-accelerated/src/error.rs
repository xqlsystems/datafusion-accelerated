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
    /// A runtime setting has a value the runtime can't read.
    UnreadableSetting {
        /// The setting's key, such as `accelerated.run`.
        key: String,
        /// Its value, as text.
        value: String,
    },
    /// A run read a table created after the run began, which isn't in the
    /// run's snapshot (D20).
    CreatedAfterRun {
        /// The table's name, as SQL writes it.
        table: String,
        /// The run's id.
        run: u64,
    },
    /// Two of a program's names, or one of them and a session table's, are
    /// the same.
    NameClash {
        /// The name, as SQL writes it.
        name: String,
        /// What the first one names, such as "step" or "session table".
        first: &'static str,
        /// What the second one names.
        second: &'static str,
    },
    /// A program's steps read each other in a cycle.
    Cycle {
        /// The steps in the cycle, or that depend on it.
        steps: Vec<String>,
    },
    /// A name that a program doesn't declare, such as an input or an
    /// output.
    UnknownName {
        /// The name.
        name: String,
        /// What it should have named: "input", "parameter", "output" or
        /// "state table".
        kind: &'static str,
    },
    /// A run of a program wasn't given one of its inputs.
    MissingInput {
        /// The input's name.
        name: String,
    },
    /// A state table was replaced by another caller during a run, so the
    /// run published nothing (design §5.4).
    Conflict {
        /// The state table's name.
        table: String,
        /// The version the run read.
        expected: u64,
        /// The version it found when it published.
        found: u64,
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
            Error::UnreadableSetting { key, value } => {
                write!(
                    f,
                    "the setting {key} has the value {value:?}, which isn't valid"
                )
            }
            Error::CreatedAfterRun { table, run } => write!(
                f,
                "the table {table} was created after run {run} began, so it isn't in the \
                 run's snapshot"
            ),
            Error::NameClash {
                name,
                first,
                second,
            } => write!(
                f,
                "the program's {first} and {second} are both named {name}"
            ),
            Error::Cycle { steps } => write!(
                f,
                "the program's steps read each other in a cycle: {}",
                steps.join(", ")
            ),
            Error::UnknownName { name, kind } => {
                write!(f, "the program has no {kind} named {name}")
            }
            Error::MissingInput { name } => write!(f, "the run wasn't given the input {name}"),
            Error::Conflict {
                table,
                expected,
                found,
            } => write!(
                f,
                "the state table {table} was replaced during the run (version {expected} was \
                 read, {found} was found), so nothing was published; \
                 RunArgs::on_conflict(OnConflict::Overwrite) publishes anyway"
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
