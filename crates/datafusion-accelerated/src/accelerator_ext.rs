// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Adds the runtime to a session's configuration, builder or state.

use std::sync::Arc;

use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::prelude::SessionConfig;

use crate::{Accelerator, AcceleratorOptions};

/// Adds the runtime to a session (design §5.1).
///
/// It's named after the crate, as datafusion-distributed's `DistributedExt`
/// is, so that it doesn't collide with other extensions' traits, and it's
/// implemented for each of the types a session is assembled from.
pub trait AcceleratorExt {
    /// Adds the runtime: its state, and the `accelerated` settings with
    /// `options`' initial values. Calling it again replaces the earlier
    /// call's runtime and options.
    ///
    /// On a `SessionStateBuilder` it adds them to the builder's
    /// configuration, so a later `with_config()` drops them. Adding the
    /// runtime to the `SessionConfig` itself avoids that.
    #[must_use]
    fn with_accelerator(self, options: AcceleratorOptions) -> Self;
}

impl AcceleratorExt for SessionConfig {
    fn with_accelerator(mut self, options: AcceleratorOptions) -> Self {
        attach(&mut self, &options);
        self
    }
}

impl AcceleratorExt for SessionStateBuilder {
    fn with_accelerator(mut self, options: AcceleratorOptions) -> Self {
        attach(
            self.config().get_or_insert_with(SessionConfig::new),
            &options,
        );
        self
    }
}

impl AcceleratorExt for SessionState {
    fn with_accelerator(mut self, options: AcceleratorOptions) -> Self {
        attach(self.config_mut(), &options);
        self
    }
}

fn attach(config: &mut SessionConfig, options: &AcceleratorOptions) {
    config.options_mut().extensions.insert(options.settings());
    config.set_extension(Arc::new(Accelerator::new()));
}
