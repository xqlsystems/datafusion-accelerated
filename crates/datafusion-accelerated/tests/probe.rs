// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Probes from the second adversarial review of #40. Each prints what happened.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::sync::Arc;

use datafusion::arrow::array::{Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{Accelerator, AcceleratorExt, AcceleratorOptions};
use datafusion_ffi::config::FFI_ConfigOptions;
use datafusion_ffi::config::extension_options::FFI_ExtensionOptions;

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, true)]))
}

fn batch(values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
}

async fn plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>> {
    ctx.sql(sql).await?.create_physical_plan().await
}

fn values(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let c = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..c.len()).map(|i| c.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

/// The context a foreign library builds: the settings cross as
/// `FFI_ConfigOptions` (strings), and come back as `ConfigOptions` whose
/// extensions are one `FFI_ExtensionOptions`, exactly as
/// `SessionConfig::try_from(&FFI_SessionConfig)` does when the marker
/// differs (datafusion-ffi 55.2.0 src/session/config.rs:132).
fn foreign_task(task: &TaskContext) -> Arc<TaskContext> {
    // datafusion-python's session knows the runtime's settings only as an
    // FFI_ExtensionOptions, the way spike 4a's __datafusion_extension_options__
    // registers them. A Python caller sets the run id on it.
    let run = task
        .session_config()
        .options()
        .extensions
        .get::<datafusion_accelerated::AcceleratedConfig>()
        .unwrap()
        .run;
    let mut python = ConfigOptions::default();
    let mut ffi_ext = FFI_ExtensionOptions::default();
    let mut native = datafusion_accelerated::AcceleratedConfig::default();
    native.run = run;
    ffi_ext.add_config(&native).unwrap();
    python.extensions.insert(ffi_ext);
    // Python executes our plan: its options cross as FFI_ConfigOptions and
    // come back on our side as ConfigOptions (datafusion-ffi 55.2.0
    // src/session/config.rs:132, src/config/mod.rs:71).
    let ffi = FFI_ConfigOptions::from(&python);
    let options = ConfigOptions::try_from(ffi).unwrap();
    let config = SessionConfig::from(options);
    Arc::new(TaskContext::new(
        None,
        "foreign".into(),
        config,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        task.runtime_env(),
    ))
}

#[tokio::test]
async fn probe_ffi_run_is_silently_dropped() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    let both = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM t").await?;
    let run = Accelerator::begin_run(&ctx)?;
    let foreign = foreign_task(&run.task_ctx());
    let keys: Vec<_> = foreign
        .session_config()
        .options()
        .entries()
        .into_iter()
        .filter(|e| e.key.contains("accelerated"))
        .map(|e| format!("{}={:?}", e.key, e.value))
        .collect();
    println!("foreign entries: {keys:?}");
    let first =
        datafusion::physical_plan::common::collect(both.execute(0, foreign_task(&run.task_ctx()))?)
            .await?;
    t.replace(vec![batch(&[2])])?;
    let second =
        datafusion::physical_plan::common::collect(both.execute(1, foreign_task(&run.task_ctx()))?)
            .await?;
    println!(
        "ffi: first={:?} second={:?}",
        values(&first),
        values(&second)
    );
    drop(run);
    // And after the run has ended: does the foreign path fail, as the doc promises?
    let after = datafusion::physical_plan::common::collect(both.execute(0, foreign)?).await;
    println!("ffi after run ended: {:?}", after.map(|b| values(&b)));
    Ok(())
}

#[tokio::test]
async fn probe_stream_outlives_run() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[1, 2, 3])])?;
    let total = plan(&ctx, "SELECT SUM(v) FROM t").await?;
    let run = Accelerator::begin_run(&ctx)?;
    let stream = total.execute(0, run.task_ctx())?;
    drop(run);
    let got = datafusion::physical_plan::common::collect(stream).await;
    println!(
        "stream polled after drop(run): {:?}",
        got.map(|b| values(&b))
    );
    Ok(())
}

#[tokio::test]
async fn probe_sql_can_set_the_run() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    let run = Accelerator::begin_run(&ctx)?;
    t.replace(vec![batch(&[2])])?;
    ctx.sql(&format!("SET accelerated.run = {}", run.id()))
        .await?
        .collect()
        .await?;
    let joined = ctx.sql("SELECT v FROM t").await?.collect().await?;
    println!("plain SQL after SET accelerated.run: {:?}", values(&joined));
    drop(run);
    let after = ctx.sql("SELECT v FROM t").await?.collect().await;
    println!(
        "plain SQL after the run ended: {:?}",
        after.map(|b| values(&b)).map_err(|e| e.to_string())
    );
    let shown = ctx
        .sql("SELECT name, value FROM information_schema.df_settings WHERE name LIKE 'accelerated.%'")
        .await;
    if let Ok(df) = shown {
        let _ = df.show().await;
    }
    Ok(())
}

#[tokio::test]
async fn probe_snapshot_mixes_instants() -> Result<()> {
    let ctx = session();
    let t = Accelerator::create_table(&ctx, "t", schema(), vec![batch(&[1])])?;
    let run = Accelerator::begin_run(&ctx)?;
    t.replace(vec![batch(&[2])])?;
    // u is derived from t's new rows, after the run began.
    Accelerator::create_table(&ctx, "u", schema(), vec![batch(&[2])])?;
    let p = plan(&ctx, "SELECT v FROM t UNION ALL SELECT v FROM u").await?;
    let got = collect(p, run.task_ctx()).await?;
    println!("run sees t and u: {:?}", values(&got));
    Ok(())
}

#[tokio::test]
async fn probe_insert_overwrite_in_datafusion() -> Result<()> {
    let ctx = SessionContext::new();
    ctx.sql("CREATE TABLE w (i BIGINT, v DOUBLE) AS VALUES (1, 1.0), (2, 2.0)")
        .await?
        .collect()
        .await?;
    let r = ctx.sql("INSERT OVERWRITE w SELECT i, v - 0.5 FROM w").await;
    match r {
        Ok(df) => {
            let x = df.collect().await;
            println!(
                "memtable INSERT OVERWRITE: {:?}",
                x.map(|_| ()).map_err(|e| e.to_string())
            );
            ctx.sql("SELECT * FROM w ORDER BY i").await?.show().await?;
        }
        Err(e) => println!("memtable INSERT OVERWRITE plan error: {e}"),
    }
    Ok(())
}

#[test]
fn probe_native_extension_crosses_ffi() {
    let mut options = ConfigOptions::default();
    options
        .extensions
        .insert(datafusion_accelerated::AcceleratedConfig::default());
    let keys: Vec<_> = options
        .entries()
        .into_iter()
        .map(|e| e.key)
        .filter(|k| !k.starts_with("datafusion."))
        .collect();
    println!("native entries outside datafusion.*: {keys:?}");
    let back = ConfigOptions::try_from(FFI_ConfigOptions::from(&options));
    println!("native extension across the FFI: {:?}", back.map(|_| ()));
}
