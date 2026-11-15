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

        /// The run whose snapshot of the runtime's tables a plan reads
        /// (D20). [`Run::task_ctx`](crate::Run::task_ctx) sets it. As a
        /// setting, it crosses datafusion-ffi and reaches
        /// datafusion-distributed's workers. Without it, each scan of a
        /// runtime table reads the table's current rows.
        pub run: Option<u64>, default = None
    }
}

impl ConfigExtension for AcceleratedConfig {
    const PREFIX: &'static str = "accelerated";
}
