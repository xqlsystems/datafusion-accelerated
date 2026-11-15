// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Where a table's rows are, and how they are held (design §5.2).

/// A memory tier (design §5.2).
///
/// Tiers follow the Arrow C Device Data Interface, whose `ArrowDeviceArray`
/// names memory by an open integer device type and a device id. So a
/// device integration, in a crate of its own (§5.8), names its devices
/// without a change to the core. M0 keeps its tables in host memory only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Tier {
    /// Host memory, Arrow's `ARROW_DEVICE_CPU`. In M0 the runtime's tables
    /// aren't yet counted in the session's `MemoryPool`; M1 counts them.
    Cpu,
    /// A device's memory, or host memory pinned for a device.
    Device {
        /// The device's family.
        device_type: DeviceType,
        /// Which device of that family, as Arrow numbers them.
        device_id: i64,
    },
    /// Files: DataFusion's `DiskManager`, Zarr and Arrow IPC files.
    Storage,
}

/// A device family, as an Arrow C Device Data Interface `ArrowDeviceType`
/// code.
///
/// A family Arrow doesn't list, such as a TPU, takes a negative code that
/// its integration crate defines. Arrow's codes are positive, so the two
/// never collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceType(i32);

impl DeviceType {
    /// `ARROW_DEVICE_CUDA`.
    pub const CUDA: Self = Self(2);
    /// `ARROW_DEVICE_CUDA_HOST`, host memory pinned for CUDA.
    pub const CUDA_HOST: Self = Self(3);
    /// `ARROW_DEVICE_CUDA_MANAGED`.
    pub const CUDA_MANAGED: Self = Self(13);
    /// `ARROW_DEVICE_METAL`.
    pub const METAL: Self = Self(8);
    /// `ARROW_DEVICE_ROCM`.
    pub const ROCM: Self = Self(10);
    /// `ARROW_DEVICE_ROCM_HOST`, host memory pinned for ROCm.
    pub const ROCM_HOST: Self = Self(11);
    /// `ARROW_DEVICE_ONEAPI`.
    pub const ONEAPI: Self = Self(14);

    /// The family with Arrow's code `code`, or an integration's negative
    /// one.
    #[must_use]
    pub const fn new(code: i32) -> Self {
        Self(code)
    }

    /// The family's code.
    #[must_use]
    pub const fn code(self) -> i32 {
        self.0
    }
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
