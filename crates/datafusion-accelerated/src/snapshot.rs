// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Versions of tables, and the snapshot a run reads.

use std::fmt;

use uuid::Uuid;

/// One version of one table: the table's identity and a counter.
///
/// The counter starts at 1 and goes up by one with each replace or publish.
/// The identity is a UUID given when the table is created, so a table that
/// is dropped and created again, or one of the same name in another
/// session or process, never has the same version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TableVersion {
    table: Uuid,
    number: u64,
}

impl TableVersion {
    pub(crate) fn new(table: Uuid, number: u64) -> Self {
        Self { table, number }
    }

    /// The table's identity.
    #[must_use]
    pub fn table(&self) -> Uuid {
        self.table
    }

    /// The version's counter.
    #[must_use]
    pub fn number(&self) -> u64 {
        self.number
    }
}

/// The versions of a set of tables, such as those a run reads (D20).
///
/// It names a state of the runtime's tables, so it can be logged, shown by
/// `explain()`, or used as a key for checkpoints and caches. It's shown as
/// `w@4, x@1`, by name and counter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Snapshot {
    // Sorted by name, then identity, so that two snapshots of one state are
    // equal and print alike.
    tables: Vec<(String, TableVersion)>,
}

impl Snapshot {
    pub(crate) fn new(tables: impl IntoIterator<Item = (String, TableVersion)>) -> Self {
        let mut tables: Vec<_> = tables.into_iter().collect();
        tables.sort_by(|a, b| (&a.0, a.1.table).cmp(&(&b.0, b.1.table)));
        Self { tables }
    }

    /// The version of the table named `name`, if the snapshot has one
    /// table of that name.
    ///
    /// A run's snapshot can have two: one dropped from the session, which a
    /// plan still reads, and the table created in its place. Then this is
    /// `None`, and [`tables`](Snapshot::tables) lists both. A program's
    /// snapshot never has two, since a program fails on a dropped table
    /// (`Error::Stale`).
    #[must_use]
    pub fn version(&self, name: &str) -> Option<TableVersion> {
        let mut named = self.tables.iter().filter(|(table, _)| table == name);
        match (named.next(), named.next()) {
            (Some((_, version)), None) => Some(*version),
            _ => None,
        }
    }

    /// The snapshot's tables and their versions, by name, then identity.
    pub fn tables(&self) -> impl Iterator<Item = (&str, TableVersion)> {
        self.tables
            .iter()
            .map(|(name, version)| (name.as_str(), *version))
    }
}

impl fmt::Display for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (name, version)) in self.tables.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{name}@{}", version.number)?;
        }
        Ok(())
    }
}
