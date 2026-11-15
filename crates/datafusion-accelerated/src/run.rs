// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A run, which reads one snapshot of the runtime's tables (D20).

use std::fmt;
use std::sync::Arc;

use datafusion::execution::TaskContext;

use crate::table_store::TableStore;

/// A run: the executions of plans that read one snapshot of the runtime's
/// tables, whatever is replaced meanwhile (D20).
///
/// [`Accelerator::begin_run`](crate::Accelerator::begin_run) begins one and
/// pins the snapshot. Its [`task_ctx`](Run::task_ctx) carries its id in the
/// `accelerated.run` setting, by which scans find the snapshot. Dropping the
/// run ends it and drops the pin, after which a plan given its context fails
/// instead of reading other rows.
pub struct Run {
    id: u64,
    task: Arc<TaskContext>,
    store: Arc<TableStore>,
}

impl Run {
    pub(crate) fn new(id: u64, task: Arc<TaskContext>, store: Arc<TableStore>) -> Self {
        Self { id, task, store }
    }

    /// The run's id, as the `accelerated.run` setting holds it.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The context to execute the run's plans with.
    #[must_use]
    pub fn task_ctx(&self) -> Arc<TaskContext> {
        Arc::clone(&self.task)
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.store.end_run(self.id);
    }
}

impl fmt::Debug for Run {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Run").field("id", &self.id).finish()
    }
}
