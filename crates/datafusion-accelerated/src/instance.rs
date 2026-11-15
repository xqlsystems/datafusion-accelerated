// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Where a table's rows are, and how they are held (design §5.2).

/// A memory tier, named by Arrow's device type where one exists (design
/// §5.2). M0 has host memory only. Device tiers will be variants that carry
/// a device id, as the Arrow C Device Data Interface's `ArrowDeviceArray`
/// does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tier {
    /// Host memory, Arrow's `ARROW_DEVICE_CPU`. In M0 the runtime's tables
    /// aren't yet counted in the session's `MemoryPool`; M1 counts them.
    Cpu,
}

/// How an instance holds its rows. M0 has Arrow rows only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Layout {
    /// Arrow record batches, one row per cell.
    Rows,
}

/// Who owns an instance's memory. In M0, the runtime owns every instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Owner {
    /// The runtime, through a table handle.
    Runtime,
}

/// One copy of a table: its tier, its layout and its owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Instance {
    tier: Tier,
    layout: Layout,
    owner: Owner,
}

impl Instance {
    pub(crate) fn new(tier: Tier, layout: Layout, owner: Owner) -> Self {
        Self {
            tier,
            layout,
            owner,
        }
    }

    /// The tier the instance is in.
    #[must_use]
    pub fn tier(&self) -> Tier {
        self.tier
    }

    /// How the instance holds its rows.
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Who owns the instance's memory.
    #[must_use]
    pub fn owner(&self) -> Owner {
        self.owner
    }
}
