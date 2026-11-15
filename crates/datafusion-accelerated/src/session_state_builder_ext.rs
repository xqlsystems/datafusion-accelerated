// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Adds the runtime to DataFusion's session builder.

use std::sync::Arc;

use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionConfig;

use crate::{Accelerator, AcceleratorOptions};

/// Adds the runtime to DataFusion's session builder (design §5.1).
pub trait SessionStateBuilderExt {
    /// Adds the runtime to the session: its state, and the `accelerated`
    /// settings with `options`' initial values.
    ///
    /// It adds them to the builder's configuration, so a later
    /// `with_config()` drops them. Calling it again replaces the earlier
    /// call's options.
    #[must_use]
    fn with_accelerator(self, options: AcceleratorOptions) -> Self;
}

impl SessionStateBuilderExt for SessionStateBuilder {
    fn with_accelerator(mut self, options: AcceleratorOptions) -> Self {
        let config: &mut SessionConfig = self.config().get_or_insert_with(SessionConfig::new);
        config.options_mut().extensions.insert(options.settings());
        config.set_extension(Arc::new(Accelerator::new(options)));
        self
    }
}
