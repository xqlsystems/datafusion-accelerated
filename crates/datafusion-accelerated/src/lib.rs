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
//! The design is in `docs/design.md`, in this crate's repository. This
//! release is scaffolding: the runtime's interfaces arrive with milestone
//! M0 (design §12.2).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
