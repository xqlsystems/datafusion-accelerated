// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The rows of every table a session's runtime owns, by version.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::TaskContext;

/// A table handle's identity within its session's runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HandleId(u64);

/// The current rows of every table, as one value: replacing a table makes a
/// new map, so a run that holds the old one keeps its rows.
type Versions = HashMap<HandleId, Arc<[RecordBatch]>>;

/// The tables of one session's runtime.
///
/// Each run reads one snapshot of them. A run is identified by the
/// `TaskContext` its plan executes with, since DataFusion passes one context
/// to every node of an execution. The first scan of a handle in a run pins
/// the snapshot current at that moment, and every later scan in the run
/// reads the same one, whatever is replaced meanwhile.
#[derive(Debug, Default)]
pub(crate) struct TableStore {
    state: Mutex<StoreState>,
}

#[derive(Debug, Default)]
struct StoreState {
    next_id: u64,
    current: Arc<Versions>,
    runs: Vec<RunPin>,
}

/// The snapshot a run pinned, held until the run's `TaskContext` is dropped.
#[derive(Debug)]
struct RunPin {
    task: Weak<TaskContext>,
    versions: Arc<Versions>,
}

impl TableStore {
    /// Adds a table, and returns its id.
    pub(crate) fn insert(&self, batches: Vec<RecordBatch>) -> HandleId {
        let mut state = self.lock();
        let id = HandleId(state.next_id);
        state.next_id += 1;
        Arc::make_mut(&mut state.current).insert(id, batches.into());
        id
    }

    /// Replaces a table's rows. Returns false if the table was released.
    pub(crate) fn replace(&self, id: HandleId, batches: Vec<RecordBatch>) -> bool {
        let mut state = self.lock();
        let versions = Arc::make_mut(&mut state.current);
        match versions.get_mut(&id) {
            Some(rows) => {
                *rows = batches.into();
                true
            }
            None => false,
        }
    }

    /// Releases a table. Runs that pinned it keep its rows until they end.
    pub(crate) fn remove(&self, id: HandleId) {
        let mut state = self.lock();
        Arc::make_mut(&mut state.current).remove(&id);
    }

    /// A table's current rows, outside any run.
    pub(crate) fn current(&self, id: HandleId) -> Option<Arc<[RecordBatch]>> {
        self.lock().current.get(&id).cloned()
    }

    /// A table's rows as the run of `task` sees them.
    pub(crate) fn for_run(
        &self,
        task: &Arc<TaskContext>,
        id: HandleId,
    ) -> Option<Arc<[RecordBatch]>> {
        let mut state = self.lock();
        // A pin whose context is gone belongs to a finished run. Dropping
        // those first also means a live pin at the same address as `task`
        // is `task`'s own, since a live allocation's address is unique.
        state.runs.retain(|pin| pin.task.strong_count() > 0);
        let position = state
            .runs
            .iter()
            .position(|pin| std::ptr::eq(pin.task.as_ptr(), Arc::as_ptr(task)));
        let index = match position {
            Some(index) => index,
            None => {
                let versions = Arc::clone(&state.current);
                state.runs.push(RunPin {
                    task: Arc::downgrade(task),
                    versions,
                });
                state.runs.len() - 1
            }
        };
        if let Some(rows) = state.runs[index].versions.get(&id) {
            return Some(Arc::clone(rows));
        }
        // A table created after the run pinned its snapshot: the run reads
        // its rows as they are now, and keeps them for the rest of the run.
        let rows = Arc::clone(state.current.get(&id)?);
        Arc::make_mut(&mut state.runs[index].versions).insert(id, Arc::clone(&rows));
        Some(rows)
    }

    fn lock(&self) -> MutexGuard<'_, StoreState> {
        self.state
            .lock()
            .expect("a thread panicked while it held the runtime's table store")
    }
}
