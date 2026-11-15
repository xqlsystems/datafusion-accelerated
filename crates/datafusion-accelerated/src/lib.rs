// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! DataFusion Accelerated, the core.
//!
//! [Apache DataFusion](https://datafusion.apache.org/) is a query engine in
//! Rust. It plans SQL into a tree of physical operators, an `ExecutionPlan`,
//! and runs it. This crate extends DataFusion so that a session can keep
//! tables in device memory, run kernels in place of DataFusion's operators,
//! and compile a program of steps once and run it many times. It attaches
//! only through DataFusion's own extension points: the session builder,
//! settings, the `RuntimeEnv`, physical optimizer rules, `ExecutionPlan`
//! nodes, metrics and codecs.
//!
//! The core depends on no hardware or vendor library. Its tests run on its
//! CPU reference implementations, and each device integration is a crate of
//! its own that implements the core's interfaces.
//!
//! A session is accelerated by building it with
//! [`with_accelerator`](SessionStateBuilderExt::with_accelerator):
//!
//! ```
//! use datafusion::execution::session_state::SessionStateBuilder;
//! use datafusion::prelude::SessionContext;
//! use datafusion_accelerated::{AcceleratorOptions, SessionStateBuilderExt};
//!
//! let state = SessionStateBuilder::new()
//!     .with_default_features()
//!     .with_accelerator(AcceleratorOptions::new().strict(true))
//!     .build();
//! let ctx = SessionContext::new_with_state(state);
//! ```
//!
//! The design is in `docs/design.md`, in this crate's repository.

mod accelerated_config;
mod accelerator;
mod accelerator_options;
mod handle_scan_exec;
mod handle_table;
mod instance;
mod session_state_builder_ext;
mod table_handle;
mod table_store;

pub use accelerated_config::AcceleratedConfig;
pub use accelerator::Accelerator;
pub use accelerator_options::AcceleratorOptions;
pub use instance::{Instance, Layout, Owner, Tier};
pub use session_state_builder_ext::SessionStateBuilderExt;
pub use table_handle::TableHandle;
