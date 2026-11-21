// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Programs at their edges: stale tables, snapshots, parameters, run
//! arguments, state schemas, and what a first user meets. Each test is a
//! finding of the review of `Program` on #40.

use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, ScalarValue};
use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorExt, AcceleratorOptions, Error, NameKind, Program, RunArgs,
};

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn kv_schema(nullable: bool) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, nullable),
    ]))
}

fn kv(rows: &[(i64, i64)]) -> RecordBatch {
    let k: Int64Array = rows.iter().map(|(k, _)| *k).collect();
    let v: Int64Array = rows.iter().map(|(_, v)| *v).collect();
    RecordBatch::try_new(kv_schema(true), vec![Arc::new(k), Arc::new(v)])
        .expect("the columns match")
}

/// The one Int64 value of a one-row output.
fn int(batches: &[RecordBatch]) -> i64 {
    batches
        .iter()
        .find(|batch| batch.num_rows() > 0)
        .expect("a row")
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64")
        .value(0)
}

fn found(err: &DataFusionError) -> Option<&Error> {
    Error::find(err)
}

#[tokio::test]
async fn a_dropped_or_registered_again_table_is_stale() -> Result<()> {
    let ctx = session();
    ctx.register_batch("lookup", kv(&[(0, 1)]))?;
    let compiled = Program::from_steps(&ctx, [("out", "SELECT v FROM lookup")])
        .output("out")
        .lower()
        .await?
        .compile()?;
    let outputs = compiled.run(RunArgs::new()).await?;
    assert_eq!(int(&outputs.table("out")?), 1);

    // SQL now reads 2, so a program that read 1 would be a wrong answer.
    ctx.deregister_table("lookup")?;
    ctx.register_batch("lookup", kv(&[(0, 2)]))?;
    let err = compiled.run(RunArgs::new()).await.unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::Stale { table }) if table == "lookup"),
        "{err}"
    );
    ctx.sql("DROP TABLE lookup").await?.collect().await?;
    let err = compiled.run(RunArgs::new()).await.unwrap_err();
    assert!(matches!(found(&err), Some(Error::Stale { .. })), "{err}");

    // A state table dropped and created again: publishing into the old one
    // would change rows nobody reads.
    let compiled = Program::new(&ctx, "SELECT k, v + 1 AS v FROM w")
        .state("w", vec![kv(&[(0, 0)])])
        .lower()
        .await?
        .compile()?;
    compiled.run(RunArgs::new()).await?;
    ctx.sql("DROP TABLE w").await?.collect().await?;
    Accelerator::create_table(&ctx, "w", kv_schema(true), vec![kv(&[(0, 9)])])?;
    let err = compiled.run(RunArgs::new()).await.unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::Stale { table }) if table == "w"),
        "{err}"
    );
    Ok(())
}

#[tokio::test]
async fn a_snapshot_lists_what_the_program_reads_in_one_order() -> Result<()> {
    let ctx = session();
    let unrelated =
        Accelerator::create_table(&ctx, "unrelated", kv_schema(true), vec![kv(&[(0, 0)])])?;
    Accelerator::create_table(&ctx, "r", kv_schema(true), vec![kv(&[(0, 5)])])?;
    let compiled = Program::new(
        &ctx,
        "SELECT w.k, w.v + r.v AS v FROM w JOIN public.r ON w.k = r.k",
    )
    .state("w", vec![kv(&[(0, 0)])])
    .lower()
    .await?
    .compile()?;
    let first = compiled.run(RunArgs::new()).await?;
    unrelated.replace(vec![kv(&[(0, 1)])])?;
    let second = compiled.run(RunArgs::new()).await?;
    let names = |outputs: &datafusion_accelerated::RunOutputs| {
        outputs
            .snapshot()
            .tables()
            .map(|(name, version)| (name.to_string(), version.number()))
            .collect::<Vec<_>>()
    };
    assert_eq!(names(&first), [("r".into(), 1), ("w".into(), 1)]);
    assert_eq!(
        names(&second),
        [("r".into(), 1), ("w".into(), 2)],
        "replacing a table the program doesn't read leaves its snapshot alone"
    );

    // A dropped table a plan still reads, and a new one of its name.
    let plan = ctx
        .sql("SELECT * FROM unrelated")
        .await?
        .create_physical_plan()
        .await?;
    ctx.sql("DROP TABLE unrelated").await?.collect().await?;
    Accelerator::create_table(&ctx, "unrelated", kv_schema(true), vec![kv(&[(0, 2)])])?;
    let a = Accelerator::begin_run(&ctx)?.snapshot();
    let b = Accelerator::begin_run(&ctx)?.snapshot();
    assert_eq!(a, b, "two snapshots of one state are equal");
    assert_eq!(
        a.tables().filter(|(name, _)| *name == "unrelated").count(),
        2
    );
    assert_eq!(a.version("unrelated"), None, "the name is ambiguous");
    drop(plan);
    Ok(())
}

#[tokio::test]
async fn a_parameter_where_planning_needs_it_fails_at_lower() -> Result<()> {
    let ctx = session();
    ctx.register_batch("t", kv(&[(0, 1), (1, 2)]))?;
    let refused = [
        ("SELECT * FROM (VALUES (0, $k)) AS x(a, b)", "VALUES"),
        ("SELECT v FROM t LIMIT $k", "LIMIT"),
        ("SELECT v FROM t OFFSET $k", "OFFSET"),
        (
            "SELECT * FROM generate_series(0, $k)",
            "a table function's arguments",
        ),
        ("SELECT v FROM t UNION ALL VALUES ($k)", "VALUES"),
    ];
    for (sql, position) in refused {
        let err = Program::from_steps(&ctx, [("s", sql)])
            .param("k", ScalarValue::Int64(Some(1)))
            .lower()
            .await
            .unwrap_err();
        assert!(
            matches!(
                found(&err),
                Some(Error::ParameterPosition { step, position: p }) if step == "s" && *p == position
            ),
            "{sql}: {err}"
        );
    }
    let allowed = [
        "SELECT v FROM t WHERE v IN ($k, 7)",
        "SELECT CASE WHEN v > $k THEN 1 ELSE 0 END AS c FROM t",
        "SELECT k, SUM(v) AS s FROM t GROUP BY k HAVING SUM(v) > $k",
        "SELECT v FROM (SELECT v + $k AS v FROM t) LIMIT 1",
    ];
    for sql in allowed {
        Program::from_steps(&ctx, [("s", sql)])
            .param("k", ScalarValue::Int64(Some(1)))
            .lower()
            .await
            .map_err(|err| err.context(sql.to_string()))?;
    }
    Ok(())
}

#[tokio::test]
async fn run_arguments_given_twice_or_cast_with_loss_fail() -> Result<()> {
    let ctx = session();
    let compiled = Program::from_steps(&ctx, [("out", "SELECT SUM(v) + $k AS s FROM d")])
        .input("d", kv_schema(true))
        .param("k", ScalarValue::Int64(Some(0)))
        .output("out")
        .lower()
        .await?
        .compile()?;
    let args = || RunArgs::new().input("d", vec![kv(&[(0, 1)])]);

    let err = compiled
        .run(args().input("d", vec![kv(&[(0, 2)])]))
        .await
        .unwrap_err();
    assert!(
        matches!(
            found(&err),
            Some(Error::GivenTwice {
                kind: NameKind::Input,
                ..
            })
        ),
        "{err}"
    );
    let twice = args()
        .param("k", ScalarValue::Int64(Some(1)))
        .param("k", ScalarValue::Int64(Some(2)));
    let err = compiled.run(twice).await.unwrap_err();
    assert!(
        matches!(
            found(&err),
            Some(Error::GivenTwice {
                kind: NameKind::Parameter,
                ..
            })
        ),
        "{err}"
    );
    for value in [ScalarValue::from(2.9), ScalarValue::from("2")] {
        let err = compiled
            .run(args().param("k", value.clone()))
            .await
            .unwrap_err();
        assert!(
            matches!(
                found(&err),
                Some(Error::ParameterType {
                    expected: DataType::Int64,
                    ..
                })
            ),
            "{value:?}: {err}"
        );
    }
    // Numbers that fit without loss.
    for value in [ScalarValue::Int32(Some(2)), ScalarValue::from(2.0)] {
        let outputs = compiled.run(args().param("k", value)).await?;
        assert_eq!(int(&outputs.table("out")?), 3);
    }
    Ok(())
}

#[tokio::test]
async fn a_qualified_read_of_a_step_is_a_read() -> Result<()> {
    let ctx = session();
    let compiled = Program::from_steps(
        &ctx,
        [
            ("s1", "SELECT v * 2 AS v FROM public.s0"),
            ("s0", "SELECT 3 AS v"),
        ],
    )
    .output("s1")
    .lower()
    .await?
    .compile()?;
    let outputs = compiled.run(RunArgs::new()).await?;
    assert_eq!(int(&outputs.table("s1")?), 6);
    Ok(())
}

#[tokio::test]
async fn an_input_is_not_planned_as_empty() -> Result<()> {
    let ctx = session();
    let rows: Vec<(i64, i64)> = (0..10_000).map(|k| (k, k)).collect();
    Program::new(
        &ctx,
        "SELECT w.k, w.v + d.v AS v FROM d JOIN w ON d.k = w.k",
    )
    .state("w", vec![kv(&rows)])
    .input("d", kv_schema(true))
    .lower()
    .await
    .map(|lowered| {
        // A join's build side is its first child, so the first scan
        // after it.
        let text = lowered.explain();
        let join = text.find("HashJoinExec").expect("a hash join");
        let build = text[join..].find("HandleScanExec: table=").expect("a scan") + join;
        assert!(
            text[build..].starts_with("HandleScanExec: table=w"),
            "an input planned as empty would be the build side of every run:\n{text}"
        );
    })?;
    Ok(())
}

#[tokio::test]
async fn a_state_step_may_be_nullable_where_the_state_isnt() -> Result<()> {
    let ctx = session();
    // Literals are NOT NULL, and a parameter makes the update nullable.
    let w0 = ctx.sql("SELECT 0 AS k, 0.0 AS v").await?.collect().await?;
    let compiled = Program::new(&ctx, "SELECT k, v - $lr * 2.0 AS v FROM w")
        .state("w", w0)
        .param("lr", ScalarValue::from(0.5))
        .output("w")
        .lower()
        .await?
        .compile()?;
    let outputs = compiled.run(RunArgs::new()).await?;
    let w = outputs.table("w")?;
    let v = w[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    assert_eq!(v.value(0), -1.0);
    assert!(!w[0].schema().field(1).is_nullable(), "the state's schema");

    // A null where the state says none may be fails the run.
    let null = ScalarValue::Float64(None);
    let err = compiled
        .run(RunArgs::new().param("lr", null))
        .await
        .unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::NullInState { field, .. }) if field == "v"),
        "{err}"
    );

    // A different type is still refused, naming the field.
    let err = Program::new(&ctx, "SELECT k, CAST(v AS DOUBLE) AS v FROM w3")
        .state("w3", vec![kv(&[(0, 0)])])
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::StateSchema { .. })),
        "{err}"
    );
    assert!(
        err.to_string().contains("the field v of type Float64"),
        "{err}"
    );
    Ok(())
}

#[tokio::test]
async fn a_state_can_start_empty() -> Result<()> {
    let ctx = session();
    let compiled = Program::new(
        &ctx,
        "SELECT k, v FROM log UNION ALL SELECT $n AS k, 0 AS v",
    )
    .state_with_schema("log", kv_schema(true), vec![])
    .param("n", ScalarValue::Int64(Some(0)))
    .output("log")
    .lower()
    .await?
    .compile()?;
    for n in 0..3 {
        compiled
            .run(RunArgs::new().param("n", ScalarValue::Int64(Some(n))))
            .await?;
    }
    let count = ctx.sql("SELECT COUNT(*) FROM log").await?.collect().await?;
    assert_eq!(int(&count), 3);
    Ok(())
}

#[tokio::test]
async fn a_planning_error_names_its_step() -> Result<()> {
    let ctx = session();
    let err = Program::from_steps(&ctx, [("a", "SELECT 1 AS x"), ("b", "SELECT nope FROM a")])
        .lower()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("in step b"), "{err}");
    let err = Program::from_steps(&ctx, [("a", "SELECT $missing AS x")])
        .lower()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("in step a"), "{err}");
    assert!(
        matches!(
            found(&err),
            Some(Error::UnknownName {
                kind: NameKind::Parameter,
                ..
            })
        ),
        "the runtime's error is still found inside the context: {err}"
    );
    Ok(())
}

#[tokio::test]
async fn a_program_continues_from_a_runtime_table() -> Result<()> {
    let ctx = session();
    const STEP: &str = "SELECT k, v + 1 AS v FROM w";
    let first = Program::new(&ctx, STEP)
        .state("w", vec![kv(&[(0, 0)])])
        .lower()
        .await?
        .compile()?;
    first.run(RunArgs::new()).await?;
    first.run(RunArgs::new()).await?;

    // Edit the step and lower again, from what was trained.
    let second = Program::new(&ctx, "SELECT k, v + 10 AS v FROM w")
        .state_from("w")
        .output("w")
        .lower()
        .await?
        .compile()?;
    second.run(RunArgs::new()).await?;
    let v = ctx.sql("SELECT v FROM w").await?.collect().await?;
    assert_eq!(int(&v), 12);

    // Both write one table, under compare-and-set.
    first.run(RunArgs::new()).await?;
    let v = ctx.sql("SELECT v FROM w").await?.collect().await?;
    assert_eq!(int(&v), 13);

    let err = Program::new(&ctx, STEP)
        .state_from("nothing")
        .lower()
        .await
        .unwrap_err();
    assert!(
        matches!(
            found(&err),
            Some(Error::UnknownName {
                kind: NameKind::RuntimeTable,
                ..
            })
        ),
        "{err}"
    );
    Ok(())
}

#[tokio::test]
async fn sql_that_writes_a_runtime_table_says_what_to_use() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "lookup", kv_schema(true), vec![kv(&[(0, 1)])])?;
    for (sql, statement) in [
        ("INSERT INTO lookup VALUES (1, 2)", "INSERT"),
        ("DELETE FROM lookup WHERE k = 0", "DELETE"),
        ("UPDATE lookup SET v = 3", "UPDATE"),
    ] {
        let result = async {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            collect(plan, ctx.task_ctx()).await
        }
        .await;
        let err = result.unwrap_err();
        assert!(
            matches!(
                found(&err),
                Some(Error::SqlWrite { statement: s, .. }) if *s == statement
            ),
            "{sql}: {err}"
        );
    }
    Ok(())
}

#[tokio::test]
async fn a_loss_reads_as_one_value() -> Result<()> {
    let ctx = session();
    let compiled = Program::from_steps(&ctx, [("loss", "SELECT AVG(CAST(v AS DOUBLE)) FROM d")])
        .input("d", kv_schema(true))
        .output("loss")
        .lower()
        .await?
        .compile()?;
    let outputs = compiled
        .run(RunArgs::new().input("d", vec![kv(&[(0, 1), (1, 2)])]))
        .await?;
    assert_eq!(f64::try_from(outputs.scalar("loss")?)?, 1.5);
    Ok(())
}

#[tokio::test]
async fn a_handle_the_session_no_longer_has_cant_be_replaced() -> Result<()> {
    let ctx = session();
    let lookup = Accelerator::create_table(&ctx, "lookup", kv_schema(true), vec![kv(&[(0, 1)])])?;
    lookup.replace(vec![kv(&[(0, 2)])])?;
    ctx.sql("CREATE OR REPLACE TABLE lookup AS SELECT 0 AS k, 3 AS v")
        .await?
        .collect()
        .await?;
    let err = lookup.replace(vec![kv(&[(0, 4)])]).unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::Detached { table }) if table == "lookup"),
        "{err}"
    );

    let other = Accelerator::create_table(&ctx, "other", kv_schema(true), vec![kv(&[(0, 1)])])?;
    ctx.sql("DROP TABLE other").await?.collect().await?;
    let err = other.replace(vec![kv(&[(0, 2)])]).unwrap_err();
    assert!(matches!(found(&err), Some(Error::Detached { .. })), "{err}");

    let kept = Accelerator::create_table(&ctx, "kept", kv_schema(true), vec![kv(&[(0, 1)])])?;
    drop(ctx);
    let err = kept.replace(vec![kv(&[(0, 2)])]).unwrap_err();
    assert!(
        matches!(found(&err), Some(Error::Detached { .. })),
        "the session is gone: {err}"
    );
    Ok(())
}
