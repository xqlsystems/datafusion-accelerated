// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The runtime's state for one session.

use std::sync::Arc;

use datafusion::catalog::Session;
use datafusion::common::{DataFusionError, Result};
use datafusion::execution::TaskContext;
use datafusion::prelude::SessionConfig;

use crate::{AcceleratedConfig, AcceleratorOptions};

/// The runtime's state for one session.
///
/// It is kept in the session's configuration, as a typed extension
/// (`SessionConfig::with_extension`), so the runtime's nodes find it
/// through the `TaskContext` that DataFusion gives each of them. That's why
/// it's shared through an `Arc`: DataFusion's extension map holds it so.
#[derive(Debug)]
pub struct Accelerator {
    options: AcceleratorOptions,
}

impl Accelerator {
    pub(crate) fn new(options: AcceleratorOptions) -> Self {
        Self { options }
    }

    /// The options the session was built with.
    #[must_use]
    pub fn options(&self) -> &AcceleratorOptions {
        &self.options
    }

    /// The runtime of a session.
    ///
    /// Fails if the session was built without
    /// [`with_accelerator`](crate::SessionStateBuilderExt::with_accelerator).
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
