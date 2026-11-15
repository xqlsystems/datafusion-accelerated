// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The rows of every table a session's runtime owns, by version.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use datafusion::arrow::record_batch::RecordBatch;

use crate::Error;

/// A table handle's identity, unique in the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HandleId(u64);

/// The current rows of every table, as one value: replacing a table makes a
/// new map, so a run that holds the old one keeps its rows.
type Versions = HashMap<HandleId, Arc<[RecordBatch]>>;

/// Handle and run ids come from one counter for the process, so an id from
/// another session is never mistaken for one of this session's.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// The tables of one session's runtime, and the snapshots its runs read.
///
/// A run pins the snapshot current when it begins, and every scan in the
/// run reads it, whatever is replaced meanwhile (D20). The pin is dropped
/// when the run ends.
#[derive(Default)]
pub(crate) struct TableStore {
    state: Mutex<StoreState>,
}

#[derive(Default)]
struct StoreState {
    current: Arc<Versions>,
    runs: HashMap<u64, Arc<Versions>>,
}

impl TableStore {
    /// Adds a table, and returns its id.
    pub(crate) fn insert(&self, batches: Vec<RecordBatch>) -> HandleId {
        let id = HandleId(next_id());
        Arc::make_mut(&mut self.lock().current).insert(id, batches.into());
        id
    }

    /// Replaces a table's rows.
    pub(crate) fn replace(&self, id: HandleId, batches: Vec<RecordBatch>) {
        Arc::make_mut(&mut self.lock().current).insert(id, batches.into());
    }

    /// Drops a table, when its last handle is dropped. Runs that pinned it
    /// keep its rows until they end.
    pub(crate) fn remove(&self, id: HandleId) {
        Arc::make_mut(&mut self.lock().current).remove(&id);
    }

    /// A table's current rows.
    pub(crate) fn current(&self, id: HandleId) -> Arc<[RecordBatch]> {
        current(&self.lock(), id)
    }

    /// Begins a run: pins the current snapshot, and returns the run's id.
    pub(crate) fn begin_run(&self) -> u64 {
        let run = next_id();
        let mut state = self.lock();
        let versions = Arc::clone(&state.current);
        state.runs.insert(run, versions);
        run
    }

    /// Ends a run, and drops its pin.
    pub(crate) fn end_run(&self, run: u64) {
        self.lock().runs.remove(&run);
    }

    /// A table's rows as run `run` sees them.
    pub(crate) fn for_run(&self, run: u64, id: HandleId) -> Result<Arc<[RecordBatch]>, Error> {
        let mut state = self.lock();
        let Some(versions) = state.runs.get(&run) else {
            return Err(Error::RunEnded { run });
        };
        if let Some(rows) = versions.get(&id) {
            return Ok(Arc::clone(rows));
        }
        // A table created after the run began: the run reads its rows as
        // they are now, and keeps them for the rest of the run.
        let rows = current(&state, id);
        if let Some(versions) = state.runs.get_mut(&run) {
            Arc::make_mut(versions).insert(id, Arc::clone(&rows));
        }
        Ok(rows)
    }

    fn lock(&self) -> MutexGuard<'_, StoreState> {
        // Every change to the state is one assignment, so a thread that
        // panicked while it held the lock left the state whole.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn current(state: &StoreState, id: HandleId) -> Arc<[RecordBatch]> {
    // A table's rows are removed only when its last handle is dropped, and
    // a caller here holds a handle.
    Arc::clone(
        state
            .current
            .get(&id)
            .expect("a live handle's rows are in its store"),
    )
}

// By hand, so that printing a plan doesn't print every table's rows.
impl fmt::Debug for TableStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.lock();
        f.debug_struct("TableStore")
            .field("tables", &state.current.len())
            .field("runs", &state.runs.len())
            .finish()
    }
}
