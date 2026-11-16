// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The runtime's errors.

use std::fmt;

use datafusion::arrow::datatypes::{DataType, SchemaRef};
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
        /// What the first one names.
        first: NameKind,
        /// What the second one names.
        second: NameKind,
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
        /// What it should have named.
        kind: NameKind,
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
    /// A table a compiled program reads, or one of its state tables, is no
    /// longer the table the session gives that name, because it was dropped
    /// or registered again after `lower`. Lower the program again.
    Stale {
        /// The table's name, as SQL writes it.
        table: String,
    },
    /// A table handle was replaced after its session stopped having the
    /// table under its name.
    Detached {
        /// The table's name, as SQL writes it.
        table: String,
    },
    /// A parameter is in a place where its value is needed to plan the
    /// step, so it can't change between runs (design §5.4).
    ParameterPosition {
        /// The step's name.
        step: String,
        /// Where the parameter is, such as `LIMIT`.
        position: &'static str,
    },
    /// A run was given one input or parameter twice.
    GivenTwice {
        /// The name.
        name: String,
        /// What it names.
        kind: NameKind,
    },
    /// A run gave a parameter a value that its type can't hold without
    /// loss, or a value of another kind, such as text for a number.
    ParameterType {
        /// The parameter's name.
        name: String,
        /// The parameter's type.
        expected: DataType,
        /// The value's type.
        found: DataType,
    },
    /// The step that computes a state table's next value gives a schema
    /// other than the table's. A field that the step's result may leave
    /// null where the table's can't isn't a difference: each run checks
    /// that it has no nulls.
    StateSchema {
        /// The state table's name, and its step's.
        table: String,
        /// The table's schema.
        expected: SchemaRef,
        /// The step's.
        found: SchemaRef,
    },
    /// A step given as a plan scans one of the program's tables, by name,
    /// with fields other than the table's.
    ScanSchema {
        /// The step's name.
        step: String,
        /// The table's name, as SQL writes it.
        table: String,
        /// The table's schema.
        expected: SchemaRef,
        /// The schema the plan's scan has.
        found: SchemaRef,
    },
    /// A run's next value for a state table has a null in a field the
    /// table's schema says isn't nullable.
    NullInState {
        /// The state table's name.
        table: String,
        /// The field's name.
        field: String,
    },
    /// SQL tried to change a table the runtime owns, which only
    /// `TableHandle::replace` and programs change.
    SqlWrite {
        /// The table's name, as SQL writes it.
        table: String,
        /// The statement, such as `INSERT`.
        statement: &'static str,
    },
    /// A plan ran with the id of a run that has ended, or that belongs to
    /// another session or process (D20).
    RunEnded {
        /// The run's id, from the `accelerated.run` setting.
        run: u64,
    },
}

/// What one of a program's names names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NameKind {
    /// A state table.
    StateTable,
    /// An input.
    Input,
    /// A parameter.
    Parameter,
    /// A step.
    Step,
    /// An output: a step or a state table.
    Output,
    /// A table of the session, outside the program.
    SessionTable,
    /// A table of the session that the runtime owns.
    RuntimeTable,
}

impl fmt::Display for NameKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            NameKind::StateTable => "state table",
            NameKind::Input => "input",
            NameKind::Parameter => "parameter",
            NameKind::Step => "step",
            NameKind::Output => "output",
            NameKind::SessionTable => "session table",
            NameKind::RuntimeTable => "runtime table",
        })
    }
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
            Error::Stale { table } => write!(
                f,
                "the table {table} was dropped or registered again after the program was \
                 lowered, so the program would read or publish a table the session no longer \
                 has; lower the program again"
            ),
            Error::Detached { table } => write!(
                f,
                "the session no longer has this table under the name {table}: it was dropped, \
                 another table took its name, or the session is gone, so replacing it would \
                 change rows nobody reads by that name"
            ),
            Error::ParameterPosition { step, position } => write!(
                f,
                "the step {step} has a parameter in {position}, where its value is needed to \
                 plan the step, so it couldn't change between runs; write a literal there"
            ),
            Error::GivenTwice { name, kind } => {
                write!(f, "the run was given the {kind} {name} twice")
            }
            Error::ParameterType {
                name,
                expected,
                found,
            } => write!(
                f,
                "the parameter {name} is {expected}, and a {found} value can't be one without \
                 loss"
            ),
            Error::StateSchema {
                table,
                expected,
                found,
            } => {
                write!(
                    f,
                    "the step {table} computes the state table {table}'s next value"
                )?;
                match first_difference(expected, found) {
                    Some((want, got)) => write!(f, " with {got}, where the table has {want}"),
                    None => write!(f, " with the schema {found}, not the table's {expected}"),
                }
            }
            Error::ScanSchema {
                step,
                table,
                expected,
                found,
            } => {
                write!(f, "the step {step} scans the table {table}")?;
                match first_difference(expected, found) {
                    Some((want, got)) => write!(f, " with {got}, where the table has {want}"),
                    None => write!(f, " with the schema {found}, not the table's {expected}"),
                }
            }
            Error::NullInState { table, field } => write!(
                f,
                "the next value of the state table {table} has a null in {field}, which the \
                 table's schema says isn't nullable"
            ),
            Error::SqlWrite { table, statement } => write!(
                f,
                "{statement} can't change {table}, a table the runtime owns: replace its rows \
                 with TableHandle::replace, or compute them in a program"
            ),
            Error::RunEnded { run } => write!(
                f,
                "run {run} has ended, or belongs to another session or process"
            ),
        }
    }
}

/// The first field in which two schemas differ, each as text, or "no
/// field" where one has fewer.
fn first_difference(expected: &SchemaRef, found: &SchemaRef) -> Option<(String, String)> {
    let describe = |field: Option<&datafusion::arrow::datatypes::FieldRef>| match field {
        Some(field) => format!(
            "the field {} of type {}{}",
            field.name(),
            field.data_type(),
            if field.is_nullable() {
                ", nullable"
            } else {
                ""
            }
        ),
        None => String::from("no field"),
    };
    let count = expected.fields().len().max(found.fields().len());
    (0..count).find_map(|i| {
        let (want, got) = (expected.fields().get(i), found.fields().get(i));
        let same = match (want, got) {
            (Some(want), Some(got)) => {
                want.name() == got.name() && want.data_type() == got.data_type()
            }
            _ => false,
        };
        (!same).then(|| (describe(want), describe(got)))
    })
}

impl std::error::Error for Error {}

impl From<Error> for DataFusionError {
    fn from(error: Error) -> Self {
        DataFusionError::External(Box::new(error))
    }
}
