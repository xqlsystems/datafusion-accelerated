// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The `accelerated` settings namespace (design §5.1).

use datafusion::common::config::ConfigExtension;
use datafusion::common::extensions_options;

extensions_options! {
    /// The runtime's settings, in the `accelerated` namespace of a session's
    /// configuration, so that SQL can set them:
    /// `SET accelerated.strict = true`.
    ///
    /// Each setting arrives with the feature that reads it.
    pub struct AcceleratedConfig {
        /// Whether a decline is an error instead of a fallback to
        /// DataFusion's own operators (principle R8).
        pub strict: bool, default = false

        /// Internal: the run whose snapshot of the runtime's tables a plan
        /// reads (D20). [`Run::task_ctx`](crate::Run::task_ctx) sets it,
        /// and it's a setting only so that it crosses datafusion-ffi.
        /// Without it, each scan of a runtime table reads the table's
        /// current rows. A run's id names its snapshot only in the process
        /// that began it.
        ///
        /// `0`, the default, is no run, since no run has that id. Set by
        /// hand with `SET`, it makes the session's queries read that run's
        /// snapshot while the run lasts, and fail with `RunEnded` after it,
        /// until `SET accelerated.run = 0`. (DataFusion 55's `RESET`
        /// resets only `datafusion.*` settings.)
        pub run: u64, default = 0
    }
}

impl ConfigExtension for AcceleratedConfig {
    const PREFIX: &'static str = "accelerated";
}
