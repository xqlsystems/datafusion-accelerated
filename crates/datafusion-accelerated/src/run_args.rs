// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What one run of a compiled program takes.

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;

/// What a run does when another caller replaced one of the program's state
/// tables after the run began (design §5.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum OnConflict {
    /// Publish nothing, and fail with `Error::Conflict`.
    #[default]
    Fail,
    /// Publish anyway, and record each table overwritten in the run's
    /// outputs. An explicit override, so binding (R10).
    Overwrite,
}

/// What one run of a compiled program takes: its inputs' rows, new values
/// of its parameters, and what to do on a conflict.
#[derive(Debug, Clone, Default)]
pub struct RunArgs {
    pub(crate) inputs: Vec<(String, Vec<RecordBatch>)>,
    pub(crate) params: Vec<(String, ScalarValue)>,
    pub(crate) on_conflict: OnConflict,
}

impl RunArgs {
    /// No inputs, the parameters' current values, and `OnConflict::Fail`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The rows of the input `name` for this run. Giving one input twice
    /// is an error.
    #[must_use]
    pub fn input(mut self, name: impl Into<String>, batches: Vec<RecordBatch>) -> Self {
        self.inputs.push((name.into(), batches));
        self
    }

    /// A value of the parameter `name`, for this run only: a run not given
    /// one uses the parameter's initial value, not the last run's. Each
    /// run's arguments are explicit, as a function's are (design §5.4).
    ///
    /// The value must be of the parameter's type, or a number that casts to
    /// it and back unchanged: 2 for a `Float64`, but not 2.9 for an `Int64`,
    /// and not text for either. Giving one parameter twice is an error.
    #[must_use]
    pub fn param(mut self, name: impl Into<String>, value: ScalarValue) -> Self {
        self.params.push((name.into(), value));
        self
    }

    /// What to do if a state table was replaced during the run.
    #[must_use]
    pub fn on_conflict(mut self, on_conflict: OnConflict) -> Self {
        self.on_conflict = on_conflict;
        self
    }
}
