// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A run, which reads one snapshot of the runtime's tables (D20).

use std::fmt;
use std::sync::Arc;

use datafusion::common::Result;
use datafusion::execution::session_state::SessionState;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;

use crate::table_store::TableStore;
use crate::{AcceleratedConfig, Error, Snapshot};

/// A run: the executions of plans that read one snapshot of the runtime's
/// tables, whatever is replaced meanwhile (D20).
///
/// [`Accelerator::begin_run`](crate::Accelerator::begin_run) begins one and
/// pins the snapshot. Its [`task_ctx`](Run::task_ctx) carries its id in the
/// `accelerated.run` setting, by which scans find the snapshot. Dropping the
/// run ends it and drops the pin, after which a plan given its context fails
/// instead of reading other rows. A stream from
/// [`execute_stream`](Run::execute_stream) keeps the run going until the
/// stream is dropped.
pub struct Run {
    pin: Arc<Pin>,
    task: Arc<TaskContext>,
}

/// The run's pin on its snapshot, dropped with the run and its streams.
struct Pin {
    id: u64,
    store: Arc<TableStore>,
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.store.end_run(self.id);
    }
}

impl Run {
    /// Begins a run of plans planned with `state`, over the tables of
    /// `store`.
    pub(crate) fn begin(mut state: SessionState, store: Arc<TableStore>) -> Result<Self> {
        let id = store.begin_run();
        // The pin first, so that a failure below still ends the run.
        let pin = Arc::new(Pin { id, store });
        state
            .config_mut()
            .options_mut()
            .extensions
            .get_mut::<AcceleratedConfig>()
            .ok_or(Error::NotAttached)?
            .run = id;
        let task = Arc::new(TaskContext::from(&state));
        Ok(Self { pin, task })
    }

    /// The versions of the session's tables that the run reads.
    ///
    /// # Panics
    ///
    /// Never: the run holds its pin until it is dropped.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        self.pin
            .store
            .snapshot(self.pin.id)
            .expect("a live run has a snapshot")
    }

    pub(crate) fn store(&self) -> &Arc<TableStore> {
        &self.pin.store
    }

    /// The run's id, as the `accelerated.run` setting holds it.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.pin.id
    }

    /// The context to execute the run's plans with.
    #[must_use]
    pub fn task_ctx(&self) -> Arc<TaskContext> {
        Arc::clone(&self.task)
    }

    /// Executes `plan` in the run, as one stream of all its partitions. The
    /// stream holds the run, so it can be returned from where the run was
    /// begun.
    pub fn execute_stream(
        &self,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<SendableRecordBatchStream> {
        let stream = datafusion::physical_plan::execute_stream(plan, self.task_ctx())?;
        let schema = stream.schema();
        let pin = Arc::clone(&self.pin);
        let stream = stream.map(move |batch| {
            // The closure owns the pin, so the run lasts as long as the
            // stream.
            let _ = &pin;
            batch
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

impl fmt::Debug for Run {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Run").field("id", &self.pin.id).finish()
    }
}
