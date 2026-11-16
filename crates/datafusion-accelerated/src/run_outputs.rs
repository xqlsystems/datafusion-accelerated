// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What one run of a compiled program gives back.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{Result, ScalarValue, plan_err};

use crate::{Error, NameKind, Snapshot};

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
                kind: NameKind::Output,
            }
            .into()
        })
    }

    /// The one value of the output `name`, such as a loss, which must have
    /// one row and one column. `f64::try_from` and the like read it.
    pub fn scalar(&self, name: &str) -> Result<ScalarValue> {
        let rows = self.table(name)?;
        let count: usize = rows.iter().map(RecordBatch::num_rows).sum();
        let columns = rows.first().map_or(0, RecordBatch::num_columns);
        let Some(batch) = rows.iter().find(|batch| batch.num_rows() > 0) else {
            return plan_err!("the output {name} has no rows, so no one value");
        };
        if count != 1 || columns != 1 {
            return plan_err!(
                "the output {name} has {count} rows and {columns} columns, not one value"
            );
        }
        ScalarValue::try_from_array(batch.column(0), 0)
    }

    /// The versions the run read of the program's state tables and of the
    /// runtime's tables its steps' plans scan, through a view too, sorted by
    /// name. A table the program doesn't read isn't listed, so replacing it
    /// doesn't change the snapshot.
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
