// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A user's first day with M0, written from the docs: design §5.4 and §10.1,
//! the crate docs, and the JAX and PyTorch guide. Each scenario prints what
//! happened, so the transcript reads as a user's notes.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::ScalarValue;
use datafusion::dataframe::DataFrameWriteOptions;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorExt, AcceleratorOptions, CompiledProgram, Program, RunArgs,
};

fn session() -> SessionContext {
    // As the crate docs say.
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new().strict(true))
        .build();
    SessionContext::new_with_state(state)
}

async fn sql(ctx: &SessionContext, q: &str) -> Vec<RecordBatch> {
    ctx.sql(q).await.unwrap().collect().await.unwrap()
}

fn show(b: &[RecordBatch]) -> String {
    pretty_format_batches(b).unwrap().to_string()
}

/// The first f64 of a one-row result, the way I'd read a loss.
fn first_f64(b: &[RecordBatch]) -> Option<f64> {
    let batch = b.iter().find(|b| b.num_rows() > 0)?;
    let c = batch.column(0).as_any().downcast_ref::<Float64Array>()?;
    (!c.is_null(0)).then(|| c.value(0))
}

/// Features: y = 3 x + 1, with x in [0, 1), 64 rows, in long format.
async fn features(ctx: &SessionContext) {
    sql(
        ctx,
        "CREATE TABLE features AS \
         SELECT value AS row, CAST(value AS DOUBLE) / 64.0 AS x, \
                3.0 * CAST(value AS DOUBLE) / 64.0 + 1.0 AS y \
         FROM generate_series(0, 63)",
    )
    .await;
}

const PRED: &str = "SELECT b.row, b.x * w.v + bias.v AS p, b.y FROM batch b, w, bias";
const LOSS: &str = "SELECT AVG((p - y) * (p - y)) AS loss FROM pred";
const GRAD_W: &str =
    "SELECT AVG(2 * (pred.p - pred.y) * batch.x) AS g FROM pred JOIN batch USING (row)";
const GRAD_B: &str = "SELECT AVG(2 * (p - y)) AS g FROM pred";
const W: &str = "SELECT w.k, w.v - $lr * g.g AS v FROM w, grad_w g";
const BIAS: &str = "SELECT bias.k, bias.v - $lr * g.g AS v FROM bias, grad_b g";

/// 1. The design's single-statement example, with my own gradient instead
/// of ddx's `grad`, and `.output("loss")` naming a CTE, as §10.1 does.
#[tokio::test]
async fn s1_the_design_example() {
    let ctx = session();
    let w0 = sql(&ctx, "SELECT 0 AS k, 0.0 AS v").await;
    let step = "WITH loss AS (SELECT AVG(v) AS l FROM w) SELECT k, v - $lr * 1.0 AS v FROM w";
    let lowered = Program::new(&ctx, step)
        .state("w", w0)
        .param("lr", ScalarValue::from(0.01))
        .output("loss")
        .lower()
        .await;
    println!(
        "S1 Program::new(STEP).output(\"loss\"), loss a CTE: {:?}",
        lowered.map(|_| "lowers").map_err(|e| e.to_string())
    );
}

/// 2. Initial state from SQL, the way I'd write it.
#[tokio::test]
async fn s2_state_from_sql() {
    let ctx = session();
    features(&ctx).await;
    for (what, q) in [
        ("SELECT 0 AS k, 0.0 AS v", "SELECT 0 AS k, 0.0 AS v"),
        (
            "VALUES",
            "SELECT column1 AS k, column2 AS v FROM (VALUES (0, 0.0))",
        ),
        ("an empty query", "SELECT 0 AS k, 0.0 AS v WHERE false"),
    ] {
        let w0 = sql(&ctx, q).await;
        let schema = w0.first().map(|b| b.schema().to_string());
        let lowered = Program::from_steps(&ctx, [("w", "SELECT k, v - $lr AS v FROM w")])
            .state("w", w0)
            .param("lr", ScalarValue::from(0.01))
            .lower()
            .await;
        println!(
            "S2 state from {what} (schema {schema:?}): {:?}",
            lowered.map(|_| "lowers").map_err(|e| e.to_string())
        );
        let _ = ctx.deregister_table("w");
    }
}

async fn training(ctx: &SessionContext) -> CompiledProgram {
    // By hand: a state updated with a parameter must have a nullable v.
    let w0 = vec![
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Float64, true),
            ])),
            vec![
                Arc::new(datafusion::arrow::array::Int64Array::from(vec![0])),
                Arc::new(Float64Array::from(vec![Some(0.0)])),
            ],
        )
        .unwrap(),
    ];
    let batch_schema = Arc::new(
        sql(ctx, "SELECT * FROM features LIMIT 0")
            .await
            .first()
            .map(|b| b.schema().as_ref().clone())
            .unwrap_or_else(|| {
                Schema::new(vec![
                    Field::new("row", DataType::Int64, false),
                    Field::new("x", DataType::Float64, false),
                    Field::new("y", DataType::Float64, false),
                ])
            }),
    );
    Program::from_steps(
        ctx,
        [
            ("pred", PRED),
            ("loss", LOSS),
            ("grad_w", GRAD_W),
            ("grad_b", GRAD_B),
            ("w", W),
            ("bias", BIAS),
        ],
    )
    .state("w", w0.clone())
    .state("bias", w0)
    .input("batch", batch_schema)
    .param("lr", ScalarValue::from(0.5))
    .output("loss")
    .lower()
    .await
    .unwrap()
    .compile()
    .unwrap()
}

/// 3. A training loop over mini-batches, with a learning-rate change.
#[tokio::test]
async fn s3_train() {
    let ctx = session();
    features(&ctx).await;
    let lowered_ok = Program::from_steps(&ctx, [("w", "SELECT k, v - 0.01 AS v FROM w")])
        .state("w", sql(&ctx, "SELECT 0 AS k, 0.0 AS v").await)
        .lower()
        .await;
    println!(
        "S3 the same NOT NULL state, updated with a literal instead of $lr: {:?}",
        lowered_ok.map(|_| "lowers").map_err(|e| e.to_string())
    );
    let _ = ctx.deregister_table("w");

    let compiled = training(&ctx).await;
    let batches: Vec<Vec<RecordBatch>> = futures_batches(&ctx).await;
    let start = Instant::now();
    let mut losses = Vec::new();
    for step in 0..400 {
        let mut args = RunArgs::new().input("batch", batches[step % 4].clone());
        if step == 200 {
            // Decay the learning rate from here on, as I'd do with a scheduler.
            args = args.param("lr", ScalarValue::from(0.05));
        }
        let out = compiled.run(args).await.unwrap();
        losses.push(first_f64(&out.table("loss").unwrap()).unwrap());
    }
    let elapsed = start.elapsed();
    println!(
        "S3 400 runs in {elapsed:?} ({:?} per run); loss at 0, 199, 200, 201, 399: {:.5} {:.5} {:.5} {:.5} {:.5}",
        elapsed / 400,
        losses[0],
        losses[199],
        losses[200],
        losses[201],
        losses[399]
    );
    println!(
        "S3 the model, read with SQL:\n{}",
        show(&sql(&ctx, "SELECT w.v AS w, bias.v AS bias FROM w, bias").await)
    );
    println!(
        "S3 the runtime after 400 runs: {:?}",
        Accelerator::of(&ctx.state()).unwrap()
    );
    println!(
        "S3 explain:\n{}",
        compiled
            .explain()
            .lines()
            .filter(|l| l.starts_with("step")
                || l.starts_with("state")
                || l.starts_with("inputs")
                || l.starts_with("param")
                || l.starts_with("publish")
                || l.starts_with("outputs"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

async fn futures_batches(ctx: &SessionContext) -> Vec<Vec<RecordBatch>> {
    let mut out = Vec::new();
    for k in 0..4 {
        out.push(sql(ctx, &format!("SELECT * FROM features WHERE row % 4 = {k}")).await);
    }
    out
}

/// 4. Notebook-style: I edit a step and lower again, keeping what I trained.
#[tokio::test]
async fn s4_edit_and_lower_again() {
    let ctx = session();
    features(&ctx).await;
    let first = training(&ctx).await;
    let batches = futures_batches(&ctx).await;
    first
        .run(RunArgs::new().input("batch", batches[0].clone()))
        .await
        .unwrap();
    let again = Program::from_steps(&ctx, [("w", W)])
        .state("w", vec![])
        .lower()
        .await;
    println!(
        "S4 lower a second program with state w: {:?}",
        again.map(|_| "lowers").map_err(|e| e.to_string())
    );
    let trained = Accelerator::table(&ctx, "w").await.unwrap().unwrap().read();
    drop(first);
    let after_drop = sql(&ctx, "SELECT COUNT(*) AS n FROM w").await;
    println!(
        "S4 after dropping the compiled program, w is still in the session: {}",
        show(&after_drop).replace('\n', " ")
    );
    sql(&ctx, "DROP TABLE w").await;
    let _ = ctx.deregister_table("bias");
    let relowered = Program::from_steps(&ctx, [("w", "SELECT k, v * 1.0 AS v FROM w")])
        .state("w", trained.to_vec())
        .lower()
        .await;
    println!(
        "S4 after DROP TABLE w, lower with the trained rows: {:?}",
        relowered.map(|_| "lowers").map_err(|e| e.to_string())
    );
}

/// 5. Checkpoint to Parquet, and resume in a new session.
#[tokio::test]
async fn s5_checkpoint_and_resume() {
    let ctx = session();
    features(&ctx).await;
    let compiled = training(&ctx).await;
    let batches = futures_batches(&ctx).await;
    for step in 0..20 {
        compiled
            .run(RunArgs::new().input("batch", batches[step % 4].clone()))
            .await
            .unwrap();
    }
    let dir = std::env::temp_dir().join(format!("dfa-user-{}", std::process::id()));
    let w = Accelerator::table(&ctx, "w").await.unwrap().unwrap();
    let written = ctx
        .read_batches(w.read().to_vec())
        .unwrap()
        .write_parquet(
            dir.join("w").to_str().unwrap(),
            DataFrameWriteOptions::new(),
            None,
        )
        .await;
    println!(
        "S5 write w with DataFusion's own writer: {:?}",
        written.map(|_| "written").map_err(|e| e.to_string())
    );
    // And the SQL way.
    let copied = ctx
        .sql(&format!(
            "COPY w TO '{}' STORED AS PARQUET",
            dir.join("w2.parquet").display()
        ))
        .await;
    let copied = match copied {
        Ok(df) => df
            .collect()
            .await
            .map(|_| "copied")
            .map_err(|e| e.to_string()),
        Err(e) => Err(e.to_string()),
    };
    println!("S5 COPY w TO ... : {copied:?}");

    let resumed = session();
    features(&resumed).await;
    let w0 = resumed
        .read_parquet(
            dir.join("w").to_str().unwrap(),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    println!(
        "S5 the checkpoint's schema: {:?}",
        w0.first().map(|b| b.schema().to_string())
    );
    let lowered = Program::from_steps(&resumed, [("w", W), ("grad_w", "SELECT 0.0 AS g")])
        .state("w", w0)
        .param("lr", ScalarValue::from(0.1))
        .lower()
        .await;
    println!(
        "S5 resume from the checkpoint: {:?}",
        lowered.map(|_| "lowers").map_err(|e| e.to_string())
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// 6. A lookup table the runtime owns, edited with SQL.
#[tokio::test]
async fn s6_handle_with_sql() {
    let ctx = session();
    let rows = sql(&ctx, "SELECT 1 AS id, 'a' AS name").await;
    let schema = rows[0].schema();
    let lookup = Accelerator::create_table(&ctx, "lookup", schema, rows).unwrap();
    for q in [
        "INSERT INTO lookup VALUES (2, 'b')",
        "DELETE FROM lookup WHERE id = 1",
        "UPDATE lookup SET name = 'z'",
        "TRUNCATE TABLE lookup",
        "CREATE OR REPLACE TABLE lookup AS SELECT 3 AS id, 'c' AS name",
    ] {
        let r = match ctx.sql(q).await {
            Ok(df) => df
                .collect()
                .await
                .map(|_| "ok".to_string())
                .map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        println!("S6 {q}: {r:?}");
    }
    let now = sql(&ctx, "SELECT * FROM lookup").await;
    println!(
        "S6 SQL now reads:\n{}\nand the handle I hold reads:\n{}",
        show(&now),
        show(&lookup.read())
    );
}

/// 7. Mistakes I'd make, and what the errors say.
#[tokio::test]
async fn s7_mistakes() {
    let ctx = session();
    features(&ctx).await;
    let compiled = training(&ctx).await;
    let batches = futures_batches(&ctx).await;
    let cases: Vec<(&str, RunArgs)> = vec![
        ("no input", RunArgs::new()),
        (
            "misspelled input",
            RunArgs::new().input("batches", batches[0].clone()),
        ),
        (
            "misspelled parameter",
            RunArgs::new()
                .input("batch", batches[0].clone())
                .param("learning_rate", ScalarValue::from(0.1)),
        ),
        (
            "parameter as text",
            RunArgs::new()
                .input("batch", batches[0].clone())
                .param("lr", ScalarValue::from("0.1")),
        ),
    ];
    for (what, args) in cases {
        println!(
            "S7 run with {what}: {:?}",
            compiled
                .run(args)
                .await
                .map(|_| "ok")
                .map_err(|e| e.to_string())
        );
    }
    let typo = Program::from_steps(
        &ctx,
        [
            ("a", "SELECT 1 AS x"),
            ("b", "SELECT x FROM aa"),
            ("c", "SELECT nope FROM a"),
        ],
    )
    .state("s", sql(&ctx, "SELECT 1 AS x").await)
    .lower()
    .await;
    println!(
        "S7 a step reading a misspelled step: {:?}",
        typo.map(|_| "lowers").map_err(|e| e.to_string())
    );
    let col = Program::from_steps(&ctx, [("a", "SELECT 1 AS x"), ("c", "SELECT nope FROM a")])
        .state("s2", sql(&ctx, "SELECT 1 AS x").await)
        .lower()
        .await;
    println!(
        "S7 a step reading a misspelled column: {:?}",
        col.map(|_| "lowers").map_err(|e| e.to_string())
    );
    let wrong_output = Program::from_steps(&ctx, [("a", "SELECT 1 AS x")])
        .state("s3", sql(&ctx, "SELECT 1 AS x").await)
        .lower()
        .await
        .unwrap()
        .compile()
        .unwrap()
        .run(RunArgs::new())
        .await
        .unwrap();
    println!(
        "S7 reading a step I didn't list as an output: {:?}",
        wrong_output
            .table("a")
            .map(|_| "ok")
            .map_err(|e| e.to_string())
    );
}

/// 8. An inference program: no state, one input.
#[tokio::test]
async fn s8_inference_without_state() {
    let ctx = session();
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let p = Program::from_steps(&ctx, [("y", "SELECT x * $a AS y FROM xs")])
        .input("xs", schema)
        .param("a", ScalarValue::from(2.0))
        .output("y")
        .lower()
        .await;
    match p {
        Ok(lowered) => {
            let compiled = lowered.compile().unwrap();
            let xs = sql(&ctx, "SELECT CAST(1.5 AS DOUBLE) AS x").await;
            let out = compiled.run(RunArgs::new().input("xs", xs)).await;
            println!(
                "S8 inference run: {:?}",
                out.map(|o| show(&o.table("y").unwrap()).replace('\n', " "))
                    .map_err(|e| e.to_string())
            );
        }
        Err(e) => println!("S8 inference program: lower failed: {e}"),
    }
}

/// 9. Serving: several tasks share one compiled program.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s9_shared_across_tasks() {
    let ctx = session();
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let compiled = Arc::new(
        Program::from_steps(&ctx, [("y", "SELECT x * 2 AS y FROM xs")])
            .input("xs", schema)
            .output("y")
            .lower()
            .await
            .unwrap()
            .compile()
            .unwrap(),
    );
    let xs = sql(&ctx, "SELECT CAST(1.5 AS DOUBLE) AS x").await;
    let start = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..8 {
        let compiled = Arc::clone(&compiled);
        let xs = xs.clone();
        tasks.push(tokio::spawn(async move {
            for _ in 0..50 {
                compiled
                    .run(RunArgs::new().input("xs", xs.clone()))
                    .await
                    .unwrap();
            }
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    println!(
        "S9 8 tasks x 50 runs of a one-step program: {:?}",
        start.elapsed()
    );
}
