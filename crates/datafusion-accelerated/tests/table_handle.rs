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
    Accelerator, AcceleratorExt, AcceleratorOptions, DeviceType, Error, Layout, Owner, Tier,
};

fn session() -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(4);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
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
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
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
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..3, 1)])?;
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
    let ctx = session();
    Accelerator::create_table(&ctx, "a", schema(), vec![batch(0..1000, 1)])?;
    Accelerator::create_table(&ctx, "b", schema(), vec![batch(0..10, 1)])?;
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

    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let u = Accelerator::create_table(&ctx, "u", schema(), vec![batch(0..4, 1)])?;
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

    // One run, in which both tables change after the first scan.
    let run = Accelerator::begin_run(&ctx)?;
    let first = collect_partition(&both, 0, &run.task_ctx()).await?;
    t.replace(vec![batch(0..4, 100)])?;
    u.replace(vec![batch(0..4, 100)])?;
    let second = collect_partition(&both, 1, &run.task_ctx()).await?;
    let third = collect_partition(&both, 2, &run.task_ctx()).await?;
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
    drop(run);

    // A new run reads the new rows.
    let run = Accelerator::begin_run(&ctx)?;
    let fresh = collect_partition(&both, 1, &run.task_ctx()).await?;
    assert_eq!(
        fresh,
        vec![0, 100, 200, 300],
        "a new run should read the replaced rows"
    );
    Ok(())
}

#[tokio::test]
async fn a_plan_outside_a_run_reads_the_current_rows() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let sum = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    let task = ctx.task_ctx();
    let first = collect(reset_plan_states(Arc::clone(&sum))?, Arc::clone(&task)).await?;
    t.replace(vec![batch(0..4, 10)])?;
    let second = collect(reset_plan_states(Arc::clone(&sum))?, task).await?;
    assert_eq!(
        (value(&first), value(&second)),
        (6, 60),
        "a context reused across executions isn't a run, so each reads the current rows"
    );
    Ok(())
}

#[tokio::test]
async fn a_plan_given_an_ended_run_fails() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let sum = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    let run = Accelerator::begin_run(&ctx)?;
    let task = run.task_ctx();
    let id = run.id();
    drop(run);
    let err = collect(reset_plan_states(sum)?, task).await.unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::RunEnded { run }) if *run == id),
        "a plan given an ended run's context should fail, not read other rows: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn a_run_cant_read_a_table_created_during_it() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..2, 1)])?;
    let run = Accelerator::begin_run(&ctx)?;
    // u is made from the new t, so reading the old t with u would read a
    // state that never existed.
    t.replace(vec![batch(0..2, 10)])?;
    Accelerator::create_table(&ctx, "u", schema(), vec![batch(0..2, 10)])?;
    let both = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM u").await?;
    assert_eq!(
        collect_partition(&both, 0, &run.task_ctx()).await?,
        vec![0, 1],
        "t is read as the run began"
    );
    let err = collect_partition(&both, 1, &run.task_ctx())
        .await
        .unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::CreatedAfterRun { table, .. }) if table == "u"),
        "u isn't in the run's snapshot: {err}"
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
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..2, 1)])?;
    let other = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
    let wrong = RecordBatch::try_new(other, vec![Arc::new(Int64Array::from(vec![1]))])?;
    let err = t.replace(vec![wrong]).unwrap_err();
    assert!(
        err.to_string().contains("table t"),
        "the error should name the table: {err}"
    );
    assert!(
        matches!(Error::find(&err), Some(Error::SchemaMismatch { .. })),
        "the error should be a schema mismatch: {err}"
    );
    assert_eq!(
        t.read().len(),
        1,
        "a failed replace should leave the rows as they were"
    );
    Ok(())
}

#[tokio::test]
async fn names_are_unique_and_drop_table_frees_them() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..2, 1)])?;
    let err = Accelerator::create_table(&ctx, "t", schema(), vec![]).unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::NameTaken { name }) if name == "t"),
        "a second table named t should be refused, not replace the first: {err}"
    );
    ctx.register_batch("plain", batch(0..1, 1))?;
    assert!(
        Accelerator::create_table(&ctx, "plain", schema(), vec![]).is_err(),
        "a session's own table names are taken too"
    );
    assert_eq!(
        Accelerator::table(&ctx, "t")
            .await?
            .map(|h| h.name().to_string()),
        Some("t".to_string()),
        "the catalog finds the runtime's table"
    );
    assert!(
        Accelerator::table(&ctx, "plain").await?.is_none(),
        "a session's own table isn't the runtime's"
    );

    ctx.sql("DROP TABLE t").await?.collect().await?;
    assert!(
        Accelerator::table(&ctx, "t").await?.is_none(),
        "DROP TABLE removes the runtime's table"
    );
    assert_eq!(
        t.read().len(),
        1,
        "a handle the caller holds keeps its rows"
    );
    Accelerator::create_table(&ctx, "t", schema(), vec![])?;
    Ok(())
}

#[tokio::test]
async fn a_run_keeps_the_rows_of_a_table_dropped_during_it() -> Result<()> {
    let ctx = session();
    let values: Int64Array = (0..3).collect();
    let values: Arc<dyn Array> = Arc::new(values);
    let rows = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)])),
        vec![Arc::clone(&values)],
    )?;
    Accelerator::create_table(&ctx, "t", rows.schema(), vec![rows])?;
    let both = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM t").await?;
    let run = Accelerator::begin_run(&ctx)?;
    collect_partition(&both, 0, &run.task_ctx()).await?;
    ctx.deregister_table("t")?;
    drop(both);
    assert_eq!(
        Arc::strong_count(&values),
        2,
        "the run pinned the table before it was dropped"
    );
    drop(run);
    assert_eq!(
        Arc::strong_count(&values),
        1,
        "once the run ends and nothing holds the table, its rows are freed"
    );
    Ok(())
}

#[tokio::test]
async fn reads_give_the_tables_schema() -> Result<()> {
    let ctx = session();
    let nullable = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("v", DataType::Int64, true),
    ]));
    let t = Accelerator::create_table(&ctx, "t", Arc::clone(&nullable), vec![])?;
    // A batch with non-null fields is contained in the nullable schema.
    t.replace(vec![batch(0..2, 1)])?;
    assert_eq!(
        t.read()[0].schema(),
        nullable,
        "a batch read back has the table's schema, not the caller's"
    );
    Ok(())
}

#[tokio::test]
async fn explain_analyze_counts_the_rows_a_scan_read() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..12, 1)])?;
    let text = ctx
        .sql("EXPLAIN ANALYZE SELECT SUM(v) FROM t")
        .await?
        .collect()
        .await?;
    let text = datafusion::arrow::util::pretty::pretty_format_batches(&text)?.to_string();
    let scan = text
        .lines()
        .find(|line| line.contains("HandleScanExec"))
        .expect("the plan scans t");
    assert!(
        scan.contains("output_rows=12"),
        "the scan should report the rows it read: {scan}"
    );
    Ok(())
}

#[tokio::test]
async fn debug_doesnt_print_the_rows() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..1000, 7)])?;
    let sum = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    let text = format!("{t:?} {sum:?}");
    assert!(
        !text.contains("6993"),
        "Debug shouldn't print a table's values: {} bytes",
        text.len()
    );
    Ok(())
}

#[tokio::test]
async fn a_runs_stream_keeps_the_run() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let sum = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    let stream = {
        let run = Accelerator::begin_run(&ctx)?;
        run.execute_stream(sum)?
    };
    let batches = datafusion::physical_plan::common::collect(stream).await?;
    assert_eq!(
        value(&batches),
        6,
        "the stream outlives the Run value, and its run with it"
    );
    Ok(())
}

#[tokio::test]
async fn a_stale_run_setting_fails_until_cleared() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(0..4, 1)])?;
    let id = Accelerator::begin_run(&ctx)?.id();
    ctx.sql(&format!("SET accelerated.run = {id}"))
        .await?
        .collect()
        .await?;
    let err = ctx
        .sql("SELECT SUM(v) FROM t")
        .await?
        .collect()
        .await
        .unwrap_err();
    assert!(
        matches!(Error::find(&err), Some(Error::RunEnded { .. })),
        "the run has ended: {err}"
    );
    ctx.sql("SET accelerated.run = 0").await?.collect().await?;
    let batches = ctx.sql("SELECT SUM(v) FROM t").await?.collect().await?;
    assert_eq!(value(&batches), 6, "0 returns to the current rows");
    Ok(())
}

#[test]
fn a_device_tier_carries_arrows_device_type_and_id() {
    let tier = Tier::Device {
        device_type: DeviceType::CUDA,
        device_id: 1,
    };
    assert_ne!(
        tier,
        Tier::Device {
            device_type: DeviceType::CUDA,
            device_id: 0,
        },
        "two GPUs of one family are two tiers"
    );
    assert_eq!(DeviceType::CUDA.code(), 2, "Arrow's ARROW_DEVICE_CUDA");
    assert_eq!(
        DeviceType::new(-1).code(),
        -1,
        "an integration names a family Arrow doesn't list"
    );
}

#[test]
fn a_handle_has_one_instance_in_host_memory() {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![]).expect("a new name");
    let instances = t.instances();
    assert_eq!(instances.len(), 1, "M0 keeps one copy of each table");
    let instance = instances[0];
    assert_eq!(
        (instance.tier(), instance.layout(), instance.owner()),
        (Tier::Cpu, Layout::Rows, Owner::Runtime)
    );
}
