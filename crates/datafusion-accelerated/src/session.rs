// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Attaching the runtime to a DataFusion session (design §5.1).

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionConfig;

use crate::config::AcceleratedConfig;

/// What a session's runtime starts with. Settings in it are only initial
/// values: SQL can change them with `SET`.
#[derive(Debug, Clone, Default)]
pub struct AcceleratorOptions {
    strict: bool,
}

impl AcceleratorOptions {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// The initial value of `accelerated.strict`.
    pub fn strict(mut self, on: bool) -> Self {
        self.strict = on;
        self
    }

    fn settings(&self) -> AcceleratedConfig {
        AcceleratedConfig {
            strict: self.strict,
        }
    }
}

/// The runtime's state for one session.
///
/// It is kept in the session's configuration, as a typed extension
/// (`SessionConfig::with_extension`), so the runtime's nodes find it
/// through the `TaskContext` that DataFusion gives each of them.
#[derive(Debug)]
pub struct Accelerator {
    options: AcceleratorOptions,
}

impl Accelerator {
    fn new(options: AcceleratorOptions) -> Self {
        Self { options }
    }

    /// The options the session was built with.
    pub fn options(&self) -> &AcceleratorOptions {
        &self.options
    }

    /// The runtime of a session.
    ///
    /// Fails if the session was built without
    /// [`with_accelerator`](SessionStateBuilderExt::with_accelerator).
    pub fn of(session: &dyn Session) -> Result<Arc<Accelerator>> {
        Self::of_config(session.config())
    }

    /// The runtime of the session a running node belongs to.
    pub fn of_task(task: &TaskContext) -> Result<Arc<Accelerator>> {
        Self::of_config(task.session_config())
    }

    /// The session's current `accelerated` settings, including any `SET`.
    pub fn settings(session: &dyn Session) -> Result<AcceleratedConfig> {
        Self::of(session)?;
        session
            .config()
            .options()
            .extensions
            .get::<AcceleratedConfig>()
            .cloned()
            .ok_or_else(not_attached)
    }

    fn of_config(config: &SessionConfig) -> Result<Arc<Accelerator>> {
        config
            .get_extension::<Accelerator>()
            .ok_or_else(not_attached)
    }
}

fn not_attached() -> DataFusionError {
    DataFusionError::Configuration(
        "this session has no DataFusion Accelerated runtime: build it with \
         SessionStateBuilder::with_accelerator(), and call with_config() \
         before with_accelerator(), since with_config() replaces the \
         configuration"
            .to_string(),
    )
}

/// Adds the runtime to DataFusion's session builder.
pub trait SessionStateBuilderExt {
    /// Adds the runtime to the session: its state, and the `accelerated`
    /// settings with `options`' initial values.
    ///
    /// It adds them to the builder's configuration, so a later
    /// `with_config()` drops them. Calling it again replaces the earlier
    /// call's options.
    fn with_accelerator(self, options: AcceleratorOptions) -> Self;
}

impl SessionStateBuilderExt for SessionStateBuilder {
    fn with_accelerator(mut self, options: AcceleratorOptions) -> Self {
        let config = self.config().get_or_insert_with(SessionConfig::new);
        config.options_mut().extensions.insert(options.settings());
        config.set_extension(Arc::new(Accelerator::new(options)));
        self
    }
}
