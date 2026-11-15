// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Table handles, and plans that read a handle's rows when they run.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorOptions, Layout, Owner, SessionStateBuilderExt, Tier,
};

fn session() -> (SessionContext, Arc<Accelerator>) {
    let config = SessionConfig::new().with_target_partitions(4);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_accelerator(AcceleratorOptions::new())
        .build();
    let accelerator = Accelerator::of(&state).expect("the session was built with the runtime");
    (SessionContext::new_with_state(state), accelerator)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// A batch of rows `(k, k * scale)` for `k` in `keys`.
fn batch(keys: std::ops::Range<i64>, scale: i64) -> RecordBatch {
    let k: Int64Array = keys.clone().collect();
    let v: Int64Array = keys.map(|key| key * scale).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(v)]).expect("the columns match")
}

/// The single Int64 value of a one-row result.
fn value(batches: &[RecordBatch]) -> i64 {
    assert_eq!(
        batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
        1,
        "one row"
    );
    let batch = batches
        .iter()
        .find(|batch| batch.num_rows() == 1)
        .expect("a row");
    let column = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("an Int64 column");
    column.value(0)
}

async fn plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

/// Runs a plan again, as a compiled program does: reset, then execute.
async fn run(ctx: &SessionContext, plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<RecordBatch>> {
    collect(reset_plan_states(Arc::clone(plan))?, ctx.task_ctx()).await
}

#[tokio::test]
async fn a_plan_reads_the_rows_of_each_run() -> Result<()> {
    let (ctx, accelerator) = session();
    let t = accelerator.create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let sum = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    assert_eq!(value(&run(&ctx, &sum).await?), 6, "0 + 1 + 2 + 3");

    t.replace(vec![batch(0..4, 10)])?;
    assert_eq!(
        value(&run(&ctx, &sum).await?),
        60,
        "the same plan should read the new rows"
    );

    t.replace(vec![batch(0..2, 1), batch(2..5, 1), batch(5..6, 1)])?;
    assert_eq!(
        value(&run(&ctx, &sum).await?),
        15,
        "a different number of batches"
    );

    t.replace(vec![])?;
    let empty = run(&ctx, &sum).await?;
    let column = empty[0].column(0);
    assert!(
        column.is_null(0),
        "SUM over no rows is NULL, got {column:?}"
    );
    Ok(())
}

#[tokio::test]
async fn count_stays_right_after_a_replace() -> Result<()> {
    let (ctx, accelerator) = session();
    let t = accelerator.create_table(&ctx, "t", schema(), vec![batch(0..3, 1)])?;
    let count = plan(&ctx, "SELECT COUNT(*) FROM t").await?;
    t.replace(vec![batch(0..5, 1)])?;
    assert_eq!(
        value(&run(&ctx, &count).await?),
        5,
        "an exact row count would let DataFusion answer from the plan's statistics (spike 4a)"
    );
    Ok(())
}

#[tokio::test]
async fn joins_of_small_handles_collect_their_left_side() -> Result<()> {
    let (ctx, accelerator) = session();
    accelerator.create_table(&ctx, "a", schema(), vec![batch(0..1000, 1)])?;
    accelerator.create_table(&ctx, "b", schema(), vec![batch(0..10, 1)])?;
    let join = plan(&ctx, "SELECT SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k").await?;
    let text = displayable(join.as_ref()).indent(false).to_string();
    assert!(
        text.contains("mode=CollectLeft"),
        "with row counts, DataFusion should collect the small side:\n{text}"
    );
    Ok(())
}

#[tokio::test]
async fn a_run_reads_one_snapshot() -> Result<()> {
    use datafusion::physical_plan::ExecutionPlanProperties;

    let (ctx, accelerator) = session();
    let t = accelerator.create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let u = accelerator.create_table(&ctx, "u", schema(), vec![batch(0..4, 1)])?;
    let both = plan(
        &ctx,
        "SELECT v FROM t UNION ALL SELECT v FROM t UNION ALL SELECT v FROM u",
    )
    .await?;
    let partitions = both.output_partitioning().partition_count();
    assert_eq!(
        partitions, 3,
        "one partition per scan, so each can be run on its own"
    );

    // One run: the first scan pins the snapshot, and both tables change before the others run.
    let task: Arc<TaskContext> = ctx.task_ctx();
    let first = collect_partition(&both, 0, &task).await?;
    t.replace(vec![batch(0..4, 100)])?;
    u.replace(vec![batch(0..4, 100)])?;
    let second = collect_partition(&both, 1, &task).await?;
    let third = collect_partition(&both, 2, &task).await?;
    assert_eq!(
        first,
        vec![0, 1, 2, 3],
        "the first scan reads the table as it was"
    );
    assert_eq!(
        second, first,
        "a second scan of the same table in the run reads the same rows"
    );
    assert_eq!(
        third,
        vec![0, 1, 2, 3],
        "another table in the run reads the same snapshot"
    );

    // A new run reads the new rows.
    let task: Arc<TaskContext> = ctx.task_ctx();
    let fresh = collect_partition(&both, 1, &task).await?;
    assert_eq!(
        fresh,
        vec![0, 100, 200, 300],
        "a new run should read the replaced rows"
    );
    Ok(())
}

async fn collect_partition(
    plan: &Arc<dyn ExecutionPlan>,
    partition: usize,
    task: &Arc<TaskContext>,
) -> Result<Vec<i64>> {
    let stream = plan.execute(partition, Arc::clone(task))?;
    let batches = datafusion::physical_plan::common::collect(stream).await?;
    let mut values = Vec::new();
    for batch in &batches {
        let column = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64");
        values.extend(column.values().iter().copied());
    }
    Ok(values)
}

#[tokio::test]
async fn a_replace_with_the_wrong_schema_fails() -> Result<()> {
    let (ctx, accelerator) = session();
    let t = accelerator.create_table(&ctx, "t", schema(), vec![batch(0..2, 1)])?;
    let other = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let wrong = RecordBatch::try_new(other, vec![Arc::new(Int64Array::from(vec![1]))])?;
    let err = t.replace(vec![wrong]).unwrap_err();
    assert!(
        err.to_string().contains("table t"),
        "the error should name the table: {err}"
    );
    assert_eq!(
        t.read()?.len(),
        1,
        "a failed replace should leave the rows as they were"
    );
    Ok(())
}

#[tokio::test]
async fn names_are_unique_and_release_frees_them() -> Result<()> {
    let (ctx, accelerator) = session();
    let t = accelerator.create_table(&ctx, "t", schema(), vec![batch(0..2, 1)])?;
    assert!(
        accelerator
            .create_table(&ctx, "t", schema(), vec![])
            .is_err(),
        "a second table named t should be refused, not replace the first"
    );
    ctx.register_batch("plain", batch(0..1, 1))?;
    assert!(
        accelerator
            .create_table(&ctx, "plain", schema(), vec![])
            .is_err(),
        "a session's own table names are taken too"
    );
    assert_eq!(
        accelerator.table("t").map(|h| h.name().to_string()),
        Some("t".to_string())
    );

    accelerator.release(&ctx, "t")?;
    assert!(
        ctx.table("t").await.is_err(),
        "release should deregister the table"
    );
    assert!(t.read().is_err(), "a released handle has no rows");
    assert!(
        t.replace(vec![]).is_err(),
        "a released handle can't be replaced"
    );
    assert!(
        accelerator.table("t").is_none(),
        "the runtime should forget the name"
    );
    assert!(
        accelerator.release(&ctx, "t").is_err(),
        "releasing twice is an error"
    );
    accelerator.create_table(&ctx, "t", schema(), vec![])?;
    Ok(())
}

#[tokio::test]
async fn a_run_keeps_the_rows_of_a_table_released_during_it() -> Result<()> {
    let (ctx, accelerator) = session();
    accelerator.create_table(&ctx, "t", schema(), vec![batch(0..3, 1)])?;
    let both = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM t").await?;
    let task: Arc<TaskContext> = ctx.task_ctx();
    collect_partition(&both, 0, &task).await?;
    accelerator.release(&ctx, "t")?;
    let second = collect_partition(&both, 1, &task).await?;
    assert_eq!(
        second,
        vec![0, 1, 2],
        "the run pinned the table before it was released"
    );
    let task: Arc<TaskContext> = ctx.task_ctx();
    assert!(
        collect_partition(&both, 0, &task).await.is_err(),
        "a new run of a plan over a released table fails"
    );
    Ok(())
}

#[test]
fn a_handle_has_one_instance_in_host_memory() {
    let (ctx, accelerator) = session();
    let t = accelerator
        .create_table(&ctx, "t", schema(), vec![])
        .expect("a new name");
    let instances = t.instances();
    assert_eq!(instances.len(), 1, "M0 keeps one copy of each table");
    let instance = instances[0];
    assert_eq!(
        (instance.tier(), instance.layout(), instance.owner()),
        (Tier::Host, Layout::Rows, Owner::Runtime)
    );
}
