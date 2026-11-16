// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The rows of every table a session's runtime owns, by version.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use datafusion::arrow::record_batch::RecordBatch;
use uuid::Uuid;

use crate::{Error, OnConflict, Snapshot, TableVersion};

/// A table handle's identity, unique in the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct HandleId(u64);

/// What the store knows of a table besides its rows.
#[derive(Debug)]
pub(crate) struct TableInfo {
    /// The table's name, as SQL writes it.
    pub(crate) name: String,
    /// The table's identity across processes and sessions, so that two
    /// tables of one name never share a version.
    pub(crate) table: Uuid,
    /// Whether the table is a program's own (an input, a step's result or
    /// its parameters), which a snapshot doesn't list.
    pub(crate) private: bool,
}

/// One version of a table.
#[derive(Debug, Clone)]
struct Stored {
    info: Arc<TableInfo>,
    rows: Arc<[RecordBatch]>,
    number: u64,
}

impl Stored {
    fn version(&self) -> TableVersion {
        TableVersion::new(self.info.table, self.number)
    }
}

/// The current version of every table, as one value: replacing a table
/// makes a new map, so a run that holds the old one keeps its rows.
type Versions = HashMap<HandleId, Stored>;

/// Handle and run ids come from one counter for the process, so an id from
/// another session is never mistaken for one of this session's.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// The tables of one session's runtime, and the snapshots its runs read.
///
/// A run pins the snapshot current when it begins, and every scan in the
/// run reads it, whatever is replaced meanwhile (D20). A table created
/// after the run began isn't in it. A run may also hold rows of its own,
/// such as a program's inputs and its steps' results, which only it reads.
/// The pin is dropped when the run ends.
#[derive(Default)]
pub(crate) struct TableStore {
    state: Mutex<StoreState>,
}

#[derive(Default)]
struct StoreState {
    current: Arc<Versions>,
    runs: HashMap<u64, RunState>,
}

struct RunState {
    versions: Arc<Versions>,
    local: HashMap<HandleId, Arc<[RecordBatch]>>,
}

/// What a run published.
#[derive(Debug)]
pub(crate) struct Publication {
    pub(crate) published: Snapshot,
    pub(crate) overwritten: Vec<String>,
}

impl TableStore {
    /// Adds a table, at version 1, and returns its id.
    pub(crate) fn insert(&self, info: TableInfo, batches: Vec<RecordBatch>) -> HandleId {
        let id = HandleId(next_id());
        let stored = Stored {
            info: Arc::new(info),
            rows: batches.into(),
            number: 1,
        };
        Arc::make_mut(&mut self.lock().current).insert(id, stored);
        id
    }

    /// Replaces a table's rows, as its next version.
    pub(crate) fn replace(&self, id: HandleId, batches: Vec<RecordBatch>) {
        let mut state = self.lock();
        let stored = Arc::make_mut(&mut state.current)
            .get_mut(&id)
            .expect("a live handle's rows are in its store");
        stored.rows = batches.into();
        stored.number += 1;
    }

    /// Drops a table, when its last handle is dropped. Runs that pinned it
    /// keep its rows until they end.
    pub(crate) fn remove(&self, id: HandleId) {
        Arc::make_mut(&mut self.lock().current).remove(&id);
    }

    /// A table's current rows.
    pub(crate) fn current(&self, id: HandleId) -> Arc<[RecordBatch]> {
        Arc::clone(&self.stored(id).rows)
    }

    /// A table's current version.
    pub(crate) fn version(&self, id: HandleId) -> TableVersion {
        self.stored(id).version()
    }

    /// Begins a run: pins the current snapshot, and returns the run's id.
    pub(crate) fn begin_run(&self) -> u64 {
        let run = next_id();
        let mut state = self.lock();
        let versions = Arc::clone(&state.current);
        state.runs.insert(
            run,
            RunState {
                versions,
                local: HashMap::new(),
            },
        );
        run
    }

    /// Ends a run, and drops its pin and its own rows.
    pub(crate) fn end_run(&self, run: u64) {
        self.lock().runs.remove(&run);
    }

    /// A table's rows as run `run` sees them: its own rows for the table if
    /// it wrote some, else the snapshot's. `None` if the table was created
    /// after the run began, since reading its current rows would mix two
    /// instants in one run.
    pub(crate) fn for_run(
        &self,
        run: u64,
        id: HandleId,
    ) -> Result<Option<Arc<[RecordBatch]>>, Error> {
        let state = self.lock();
        let run_state = state.runs.get(&run).ok_or(Error::RunEnded { run })?;
        if let Some(rows) = run_state.local.get(&id) {
            return Ok(Some(Arc::clone(rows)));
        }
        Ok(run_state
            .versions
            .get(&id)
            .map(|stored| Arc::clone(&stored.rows)))
    }

    /// Gives run `run` rows of its own for a table, which only it reads.
    pub(crate) fn write_local(
        &self,
        run: u64,
        id: HandleId,
        batches: Vec<RecordBatch>,
    ) -> Result<(), Error> {
        let mut state = self.lock();
        let run_state = state.runs.get_mut(&run).ok_or(Error::RunEnded { run })?;
        run_state.local.insert(id, batches.into());
        Ok(())
    }

    /// The versions of the session's tables that run `run` reads.
    pub(crate) fn snapshot(&self, run: u64) -> Result<Snapshot, Error> {
        let state = self.lock();
        let run_state = state.runs.get(&run).ok_or(Error::RunEnded { run })?;
        Ok(Snapshot::new(
            run_state
                .versions
                .values()
                .filter(|stored| !stored.info.private)
                .map(|stored| (stored.info.name.clone(), stored.version())),
        ))
    }

    /// The versions run `run` reads of the tables `ids`. A table created
    /// after the run began isn't listed.
    pub(crate) fn snapshot_of(&self, run: u64, ids: &[HandleId]) -> Result<Snapshot, Error> {
        let state = self.lock();
        let run_state = state.runs.get(&run).ok_or(Error::RunEnded { run })?;
        Ok(Snapshot::new(ids.iter().filter_map(|id| {
            run_state
                .versions
                .get(id)
                .map(|stored| (stored.info.name.clone(), stored.version()))
        })))
    }

    /// Publishes `writes` as the next version of each table, all at once
    /// (R5). A table that another caller replaced after the run began is a
    /// conflict: with `OnConflict::Fail`, nothing is published.
    pub(crate) fn publish(
        &self,
        run: u64,
        writes: Vec<(HandleId, Vec<RecordBatch>)>,
        on_conflict: OnConflict,
    ) -> Result<Publication, Error> {
        let mut state = self.lock();
        let run_state = state.runs.get(&run).ok_or(Error::RunEnded { run })?;
        let mut overwritten = Vec::new();
        for (id, _) in &writes {
            let read = run_state
                .versions
                .get(id)
                .expect("a program's state tables exist before its runs");
            let now = state
                .current
                .get(id)
                .expect("a program holds its state tables");
            if now.number != read.number {
                match on_conflict {
                    OnConflict::Fail => {
                        return Err(Error::Conflict {
                            table: now.info.name.clone(),
                            expected: read.number,
                            found: now.number,
                        });
                    }
                    OnConflict::Overwrite => overwritten.push(now.info.name.clone()),
                }
            }
        }
        let current = Arc::make_mut(&mut state.current);
        let mut published = Vec::with_capacity(writes.len());
        for (id, batches) in writes {
            let stored = current
                .get_mut(&id)
                .expect("a program holds its state tables");
            stored.rows = batches.into();
            stored.number += 1;
            published.push((stored.info.name.clone(), stored.version()));
        }
        Ok(Publication {
            published: Snapshot::new(published),
            overwritten,
        })
    }

    fn stored(&self, id: HandleId) -> Stored {
        // A table's rows are removed only when its last handle is dropped,
        // and a caller here holds a handle.
        self.lock()
            .current
            .get(&id)
            .expect("a live handle's rows are in its store")
            .clone()
    }

    fn lock(&self) -> MutexGuard<'_, StoreState> {
        // Every change to the state is made whole under the lock before
        // anything can panic, so a thread that panicked while it held the
        // lock left the state whole.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
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
