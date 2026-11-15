// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Bugs found by review and by `simulation.rs`, each reduced to a small case.
//!
//! Each test asserts what the crate's docs or the design promise. A test
//! here fails until its bug is fixed, and then stays as a regression test.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{Accelerator, AcceleratorOptions, SessionStateBuilderExt};

fn session() -> (SessionContext, Arc<Accelerator>) {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    let accelerator = Accelerator::of(&state).expect("the session was built with the runtime");
    (SessionContext::new_with_state(state), accelerator)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]))
}

fn column(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}

fn batch(values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(schema(), vec![column(values)]).expect("the column matches")
}

async fn plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

async fn sum(plan: &Arc<dyn ExecutionPlan>, task: Arc<TaskContext>) -> Result<i64> {
    let batches = collect(reset_plan_states(Arc::clone(plan))?, task).await?;
    let column = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("an Int64 column");
    Ok(column.value(0))
}

/// A caller that keeps one `TaskContext` and runs a plan with it on every
/// training step reads the first step's rows forever. Nothing fails: the
/// replace succeeds and the answer is silently stale. Spike 4a's Python
/// path stores one task context and runs every compiled run with it.
#[tokio::test]
async fn a_reused_task_context_still_reads_each_replace() -> Result<()> {
    let (ctx, accelerator) = session();
    let w = accelerator.create_table(&ctx, "w", schema(), vec![batch(&[1])])?;
    let total = plan(&ctx, "SELECT SUM(v) FROM w").await?;
    let task = ctx.task_ctx();
    assert_eq!(sum(&total, Arc::clone(&task)).await?, 1);
    w.replace(vec![batch(&[2])])?;
    assert_eq!(
        sum(&total, Arc::clone(&task)).await?,
        2,
        "a run is one collect(); the second one should see the replace"
    );
    Ok(())
}

/// datafusion-ffi rebuilds the `TaskContext` on every `execute` call that
/// crosses the boundary (`From<FFI_TaskContext> for Arc<TaskContext>`,
/// datafusion-ffi 55.2.0 src/execution/task_ctx.rs:184, which ends in
/// `Arc::new(TaskContext::new(..))`), and drops it when `execute` returns.
/// This reproduces that: each partition of one logical run executes with
/// its own context, as a plan over a handle table does when datafusion-python
/// runs it. Each scan then pins its own snapshot, so D20 doesn't hold.
#[tokio::test]
async fn one_run_across_the_ffi_reads_one_snapshot() -> Result<()> {
    let (ctx, accelerator) = session();
    let t = accelerator.create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    let both = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM t").await?;
    let execute = |partition: usize| {
        // What the foreign side of the FFI builds for each call.
        let task = Arc::new(TaskContext::from(&ctx.state()));
        both.execute(partition, task)
    };
    let first = datafusion::physical_plan::common::collect(execute(0)?).await?;
    t.replace(vec![batch(&[2])])?;
    let second = datafusion::physical_plan::common::collect(execute(1)?).await?;
    assert_eq!(
        first, second,
        "both scans of t belong to one run, through the FFI as in Rust"
    );
    Ok(())
}

/// Session names are normalized (unquoted identifiers are lowercased), but
/// the runtime keys its tables by the raw string. A table created as "W"
/// is `w` to SQL and to `table_exist`, and `W` to the runtime.
#[tokio::test]
async fn names_are_normalized_as_the_session_normalizes_them() -> Result<()> {
    let (ctx, accelerator) = session();
    accelerator.create_table(&ctx, "W", schema(), vec![batch(&[1])])?;
    assert!(
        ctx.table_exist("w")?,
        "SQL reads it as w, so the session has w"
    );
    assert!(
        accelerator.table("w").is_some(),
        "the runtime should find the table SQL reads as w"
    );
    accelerator.release(&ctx, "w")?;
    Ok(())
}

/// The runtime and the session each keep a map from names to tables, and
/// nothing keeps them in step. `DROP TABLE` (or `deregister_table`)
/// removes the provider, the runtime keeps the rows and the name, and the
/// name can't be created again although the session has no such table.
#[tokio::test]
async fn a_dropped_handle_table_frees_its_name() -> Result<()> {
    let (ctx, accelerator) = session();
    accelerator.create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    ctx.sql("DROP TABLE t").await?.collect().await?;
    assert!(!ctx.table_exist("t")?, "DROP TABLE removed it");
    accelerator.create_table(&ctx, "t", schema(), vec![batch(&[2])])?;
    Ok(())
}

/// `release` says it "drops its rows". A finished run's pin is pruned only
/// by the next scan of any table, so the last run's snapshot, which holds
/// every table of the session, outlives the run and the release.
#[tokio::test]
async fn release_after_the_last_run_drops_the_rows() -> Result<()> {
    let (ctx, accelerator) = session();
    let values = column(&[1, 2, 3]);
    let rows = RecordBatch::try_new(schema(), vec![Arc::clone(&values)])?;
    accelerator.create_table(&ctx, "t", schema(), vec![rows])?;
    let total = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    sum(&total, ctx.task_ctx()).await?;
    drop(total);
    accelerator.release(&ctx, "t")?;
    assert_eq!(
        Arc::strong_count(&values),
        1,
        "no run is live and the table is released, so nothing should hold its rows"
    );
    Ok(())
}
