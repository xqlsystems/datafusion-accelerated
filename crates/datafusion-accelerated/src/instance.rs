// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Where a table's rows are, and how they are held (design §5.2).

/// A memory tier (design §5.2).
///
/// Tiers follow the Arrow C Device Data Interface, whose `ArrowDeviceArray`
/// names memory by a device type and a device id. A device type is open,
/// so a device integration, in a crate of its own (§5.8), names its devices
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

/// A device family: an Arrow C Device Data Interface `ArrowDeviceType`
/// code where Arrow has one, and a name the device's integration defines
/// where it doesn't, as for a TPU (design §5.2).
///
/// A name is namespaced by the integration's crate, such as `"pjrt.tpu"`,
/// so that two integrations can't choose the same one by accident. Arrow's
/// codes follow DLPack's, and Arrow reserves only `ARROW_DEVICE_EXT_DEV`
/// (12) for devices it doesn't list, with meanings that differ by
/// implementation. So a named family can't be exported in an
/// `ArrowDeviceArray` as itself: an export must map it to `EXT_DEV` or
/// refuse, explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DeviceType {
    /// An `ArrowDeviceType` code.
    Arrow(i32),
    /// A family Arrow doesn't list, named by its integration.
    Named(&'static str),
}

impl DeviceType {
    /// `ARROW_DEVICE_CUDA`.
    pub const CUDA: Self = Self::Arrow(2);
    /// `ARROW_DEVICE_CUDA_HOST`, host memory pinned for CUDA.
    pub const CUDA_HOST: Self = Self::Arrow(3);
    /// `ARROW_DEVICE_METAL`.
    pub const METAL: Self = Self::Arrow(8);
    /// `ARROW_DEVICE_ROCM`.
    pub const ROCM: Self = Self::Arrow(10);
    /// `ARROW_DEVICE_ROCM_HOST`, host memory pinned for ROCm.
    pub const ROCM_HOST: Self = Self::Arrow(11);
    /// `ARROW_DEVICE_CUDA_MANAGED`.
    pub const CUDA_MANAGED: Self = Self::Arrow(13);
    /// `ARROW_DEVICE_ONEAPI`.
    pub const ONEAPI: Self = Self::Arrow(14);

    /// The family's `ArrowDeviceType` code, if Arrow has one.
    #[must_use]
    pub const fn arrow_code(self) -> Option<i32> {
        match self {
            Self::Arrow(code) => Some(code),
            Self::Named(_) => None,
        }
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
