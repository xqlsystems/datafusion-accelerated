// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What a session's runtime starts with.

use crate::AcceleratedConfig;

/// What a session's runtime starts with. Settings in it are only initial
/// values: SQL can change them with `SET`.
#[derive(Debug, Clone, Default)]
pub struct AcceleratorOptions {
    strict: bool,
}

impl AcceleratorOptions {
    /// The defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial value of `accelerated.strict`: whether a decline is
    /// an error instead of a fallback (principle R8).
    #[must_use]
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// The initial settings.
    pub(crate) fn settings(&self) -> AcceleratedConfig {
        AcceleratedConfig {
            strict: self.strict,
            run: None,
        }
    }
}
