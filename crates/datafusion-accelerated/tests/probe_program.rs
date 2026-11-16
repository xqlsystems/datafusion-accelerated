// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Probes from the adversarial review of `Program`. Each prints what happened.

#![allow(clippy::unwrap_used, clippy::expect_used, missing_docs)]

use std::sync::Arc;

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{Accelerator, AcceleratorExt, AcceleratorOptions, Program, RunArgs};

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(2))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn batch(rows: &[(i64, i64)]) -> RecordBatch {
    let i: Int64Array = rows.iter().map(|r| r.0).collect();
    let v: Int64Array = rows.iter().map(|r| Some(r.1)).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(i), Arc::new(v)]).unwrap()
}

fn show(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches)
        .unwrap()
        .to_string()
        .replace('\n', " ")
}

/// Where a literal works, does a parameter?
#[tokio::test]
async fn probe_parameter_positions() {
    let cases = [
        ("LIMIT", "SELECT i, v FROM w ORDER BY i LIMIT $k"),
        ("OFFSET", "SELECT i, v FROM w ORDER BY i OFFSET $k"),
        ("IN list", "SELECT i, v FROM w WHERE v IN ($k, 7)"),
        (
            "JOIN ON",
            "SELECT w.i, w.v FROM w JOIN w AS u ON w.i = u.i AND u.v > $k",
        ),
        (
            "window frame",
            "SELECT i, SUM(v) OVER (ORDER BY i ROWS BETWEEN $k PRECEDING AND CURRENT ROW) AS v FROM w",
        ),
        (
            "table function",
            "SELECT value AS i, value AS v FROM generate_series(0, $k)",
        ),
        (
            "VALUES",
            "SELECT column1 AS i, column2 AS v FROM (VALUES (0, $k))",
        ),
        (
            "CASE",
            "SELECT i, CASE WHEN v > $k THEN 1 ELSE 0 END AS v FROM w",
        ),
        (
            "aggregate FILTER",
            "SELECT i, SUM(v) FILTER (WHERE v > $k) AS v FROM w GROUP BY i",
        ),
        (
            "GROUP BY expr",
            "SELECT i, v FROM w GROUP BY i, v HAVING SUM(v) > $k",
        ),
    ];
    for (what, sql) in cases {
        let literal = sql.replace("$k", "CAST(1 AS BIGINT)");
        let plain = SessionContext::new();
        plain
            .register_table(
                "w",
                Arc::new(
                    MemTable::try_new(schema(), vec![vec![batch(&[(0, 1), (1, 2), (2, 3)])]])
                        .unwrap(),
                ),
            )
            .unwrap();
        let want = match plain.sql(&literal).await {
            Ok(df) => df
                .collect()
                .await
                .map(|b| show(&b))
                .map_err(|e| e.to_string()),
            Err(e) => Err(e.to_string()),
        };
        let ctx = session();
        let got = async {
            let program = Program::from_steps(&ctx, [("out", sql)])
                .state("w", vec![batch(&[(0, 1), (1, 2), (2, 3)])])
                .param("k", ScalarValue::from(1_i64))
                .output("out")
                .lower()
                .await?
                .compile()?;
            let outputs = program.run(RunArgs::new()).await?;
            Ok::<_, datafusion::error::DataFusionError>(show(&outputs.table("out")?))
        }
        .await
        .map_err(|e| e.to_string());
        let verdict = match (&want, &got) {
            (Ok(a), Ok(b)) if a == b => "same",
            (Ok(_), Ok(_)) => "DIFFERENT",
            (Ok(_), Err(_)) => "PROGRAM FAILS, LITERAL WORKS",
            (Err(_), _) => "literal fails too",
        };
        println!("[{what}] {verdict}\n   literal: {want:?}\n   program: {got:?}");
    }
}

/// A compiled program keeps reading a session table that was replaced in
/// the catalog.
#[tokio::test]
async fn probe_reregistered_session_table() {
    let ctx = session();
    let mem = |v| Arc::new(MemTable::try_new(schema(), vec![vec![batch(&[(0, v)])]]).unwrap());
    ctx.register_table("lookup", mem(1)).unwrap();
    let program = Program::from_steps(&ctx, [("out", "SELECT i, v FROM lookup")])
        .state("w", vec![batch(&[(0, 0)])])
        .output("out")
        .lower()
        .await
        .unwrap()
        .compile()
        .unwrap();
    ctx.deregister_table("lookup").unwrap();
    ctx.register_table("lookup", mem(2)).unwrap();
    let sql = ctx
        .sql("SELECT v FROM lookup")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let run = program
        .run(RunArgs::new())
        .await
        .map(|o| show(&o.table("out").unwrap()));
    println!("SQL reads: {}\nprogram reads: {run:?}", show(&sql));
    ctx.deregister_table("lookup").unwrap();
    let run = program
        .run(RunArgs::new())
        .await
        .map(|o| show(&o.table("out").unwrap()));
    println!("after DROP, program reads: {run:?}");
}

/// What a run's snapshot lists.
#[tokio::test]
async fn probe_snapshot_contents() {
    let ctx = session();
    let unrelated =
        Accelerator::create_table(&ctx, "unrelated", schema(), vec![batch(&[(0, 0)])]).unwrap();
    let program = Program::new(&ctx, "SELECT i, v + 1 AS v FROM w")
        .state("w", vec![batch(&[(0, 0)])])
        .lower()
        .await
        .unwrap()
        .compile()
        .unwrap();
    let a = program.run(RunArgs::new()).await.unwrap();
    unrelated.replace(vec![batch(&[(0, 1)])]).unwrap();
    let b = program.run(RunArgs::new()).await.unwrap();
    println!("run 1 read {}\nrun 2 read {}", a.snapshot(), b.snapshot());

    // A plan keeps an old table alive after DROP TABLE, and a new one takes
    // its name: two entries with one name.
    let old = ctx
        .sql("SELECT * FROM unrelated")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    ctx.sql("DROP TABLE unrelated")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    drop(unrelated);
    Accelerator::create_table(&ctx, "unrelated", schema(), vec![batch(&[(0, 9)])]).unwrap();
    let c = program.run(RunArgs::new()).await.unwrap();
    let d = program.run(RunArgs::new()).await.unwrap();
    println!(
        "with an old and a new `unrelated`: run 3 read {}, run 4 read {}; version(unrelated) = {:?} then {:?}",
        c.snapshot(),
        d.snapshot(),
        c.snapshot().version("unrelated").map(|v| v.table()),
        d.snapshot().version("unrelated").map(|v| v.table())
    );
    drop(old);
}

/// Arguments given twice, and a value of another type.
#[tokio::test]
async fn probe_run_arguments() {
    let ctx = session();
    let program = Program::from_steps(&ctx, [("out", "SELECT i, v + $k AS v FROM d")])
        .state("w", vec![batch(&[(0, 0)])])
        .input("d", schema())
        .param("k", ScalarValue::from(10_i64))
        .output("out")
        .lower()
        .await
        .unwrap()
        .compile()
        .unwrap();
    let twice = program
        .run(
            RunArgs::new()
                .input("d", vec![batch(&[(0, 1)])])
                .input("d", vec![batch(&[(0, 2)])])
                .param("k", ScalarValue::from(100_i64))
                .param("k", ScalarValue::from(1000_i64)),
        )
        .await
        .map(|o| show(&o.table("out").unwrap()));
    println!("input and param given twice: {twice:?}");
    let float = program
        .run(
            RunArgs::new()
                .input("d", vec![batch(&[(0, 0)])])
                .param("k", ScalarValue::from(2.9_f64)),
        )
        .await
        .map(|o| show(&o.table("out").unwrap()));
    println!("k = 2.9 for an Int64 parameter: {float:?}");
    let null = program
        .run(
            RunArgs::new()
                .input("d", vec![batch(&[(0, 0)])])
                .param("k", ScalarValue::Int64(None)),
        )
        .await
        .map(|o| show(&o.table("out").unwrap()));
    println!("k = NULL: {null:?}");
}

/// A CTE named like another step, and a step read by a qualified name.
#[tokio::test]
async fn probe_step_name_resolution() {
    let ctx = session();
    let cte = Program::from_steps(
        &ctx,
        [
            ("a", "WITH b AS (SELECT i, v FROM w) SELECT i, v FROM b"),
            ("b", "SELECT i, v FROM a"),
        ],
    )
    .state("w", vec![batch(&[(0, 0)])])
    .output("b")
    .lower()
    .await
    .map(|_| "lowers");
    println!(
        "a CTE named like a step: {:?}",
        cte.map_err(|e| e.to_string())
    );

    for order in [["s0", "s1"], ["s1", "s0"]] {
        let ctx = session();
        let steps: Vec<(&str, &str)> = order
            .iter()
            .map(|name| match *name {
                "s0" => ("s0", "SELECT i, v FROM w"),
                _ => ("s1", "SELECT i, v FROM public.s0"),
            })
            .collect();
        let lowered = Program::from_steps(&ctx, steps)
            .state("w", vec![batch(&[(0, 0)])])
            .output("s1")
            .lower()
            .await
            .map(|_| "lowers");
        println!(
            "public.s0, listed {order:?}: {:?}",
            lowered.map_err(|e| e.to_string())
        );
    }
}

/// How joins on an input are planned, when the input's statistics are
/// those of no rows.
#[tokio::test]
async fn probe_input_join_plan() {
    let ctx = session();
    let program = Program::from_steps(
        &ctx,
        [(
            "out",
            "SELECT d.i, d.v + w.v AS v FROM d JOIN w ON d.i = w.i",
        )],
    )
    .state(
        "w",
        vec![batch(&(0..100_000).map(|i| (i, i)).collect::<Vec<_>>())],
    )
    .input("d", schema())
    .output("out")
    .lower()
    .await
    .unwrap();
    for line in lowered_lines(&lowered_explain(&program)) {
        println!("{line}");
    }
}

fn lowered_explain(p: &datafusion_accelerated::LoweredProgram) -> String {
    p.explain()
}

fn lowered_lines(s: &str) -> Vec<String> {
    s.lines()
        .filter(|l| {
            l.contains("Join")
                || l.contains("Scan")
                || l.contains("Repartition")
                || l.contains("Coalesce")
        })
        .map(str::to_string)
        .collect()
}
