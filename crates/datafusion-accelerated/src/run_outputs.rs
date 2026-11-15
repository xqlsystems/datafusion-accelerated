// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What one run of a compiled program gives back.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;

use crate::{Error, Snapshot};

/// What one run of a compiled program gives back: its outputs' rows, the
/// snapshot it read, and the versions it published.
#[derive(Debug)]
pub struct RunOutputs {
    pub(crate) tables: HashMap<String, Arc<[RecordBatch]>>,
    pub(crate) snapshot: Snapshot,
    pub(crate) published: Snapshot,
    pub(crate) overwritten: Vec<String>,
}

impl RunOutputs {
    /// The rows of the output `name`, a step or a state table. A state
    /// table's rows are its next value.
    pub fn table(&self, name: &str) -> Result<Arc<[RecordBatch]>> {
        self.tables.get(name).cloned().ok_or_else(|| {
            Error::UnknownName {
                name: name.to_string(),
                kind: "output",
            }
            .into()
        })
    }

    /// The versions of the session's tables that the run read.
    #[must_use]
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// The versions the run published, one for each state table it
    /// computed.
    #[must_use]
    pub fn published(&self) -> &Snapshot {
        &self.published
    }

    /// The state tables the run overwrote under `OnConflict::Overwrite`,
    /// which another caller had replaced during the run.
    #[must_use]
    pub fn overwritten(&self) -> &[String] {
        &self.overwritten
    }
}
