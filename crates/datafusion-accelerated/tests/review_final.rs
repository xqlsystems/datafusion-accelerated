// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The adversarial review's last pass over M0 before it merges: tables
//! shared by programs through `state_from`, names, parameters in the
//! positions the earlier reviews didn't try, and views.
//!
//! Each test's **Assumes** line states what it takes the docs or the design
//! to promise. A test of a finding not yet fixed is ignored, with a link to
//! the finding.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorExt, AcceleratorOptions, CompiledProgram, Error, OnConflict, Program,
    RunArgs,
};

fn session() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(2))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

/// `(i, v)`, both Int64, `v` nullable.
fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn rows(values: &[(i64, i64)]) -> Vec<RecordBatch> {
    let i: Int64Array = values.iter().map(|r| r.0).collect();
    let v: Int64Array = values.iter().map(|r| Some(r.1)).collect();
    vec![RecordBatch::try_new(schema(), vec![Arc::new(i), Arc::new(v)]).expect("the columns match")]
}

/// The `v` column of a result, sorted.
fn values(batches: &[RecordBatch]) -> Vec<i64> {
    let mut out: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let v = b
                .column_by_name("v")
                .expect("a v column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .clone();
            (0..v.len()).map(move |k| v.value(k))
        })
        .collect();
    out.sort_unstable();
    out
}

async fn sql_values(ctx: &SessionContext, sql: &str) -> Result<Vec<i64>> {
    Ok(values(&ctx.sql(sql).await?.collect().await?))
}

// ---------------------------------------------------------------------------
// A seeded property: programs sharing one state table.
// ---------------------------------------------------------------------------

// SplitMix64, as in ddx's test_utils and `simulation_program.rs`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

const ADD: &str = "SELECT i, v + $c AS v FROM w";

/// One published run: the version of `w` it read, the version it
/// published, what it added, and whether it overwrote.
#[derive(Debug, Clone, Copy)]
struct Published {
    read: u64,
    wrote: u64,
    added: i64,
    overwrote: bool,
}

/// **Assumes:** design §5.4 and `Program::state_from`: programs sharing a
/// state table publish by compare-and-set, so every published version
/// equals the version it read plus what its run added, a run that doesn't
/// overwrite read the version just before its own, and conflicts are the
/// only errors.
async fn shared_state_case(seed: u64) -> Result<()> {
    let mut rng = Rng::new(seed);
    let ctx = session();
    let mut programs: Vec<Arc<CompiledProgram>> = Vec::new();
    let first = Program::from_steps(&ctx, [("w", ADD)])
        .state("w", rows(&[(0, 0)]))
        .param("c", ScalarValue::from(0_i64))
        .lower()
        .await?
        .compile()?;
    programs.push(Arc::new(first));
    for _ in 0..1 + rng.below(3) {
        let next = Program::from_steps(&ctx, [("w", ADD)])
            .state_from("w")
            .param("c", ScalarValue::from(0_i64))
            .lower()
            .await?
            .compile()?;
        programs.push(Arc::new(next));
    }
    let w = Accelerator::table(&ctx, "w")
        .await?
        .expect("w is the runtime's");
    let start = w.version().number();

    let mut tasks = Vec::new();
    for program in &programs {
        let plan: Vec<(i64, bool)> = (0..1 + rng.below(6))
            .map(|_| (1 + rng.below(9) as i64, rng.below(100) < 30))
            .collect();
        let program = Arc::clone(program);
        tasks.push(tokio::spawn(async move {
            let mut published = Vec::new();
            for (added, overwrite) in plan {
                let on_conflict = if overwrite {
                    OnConflict::Overwrite
                } else {
                    OnConflict::Fail
                };
                let args = RunArgs::new()
                    .param("c", ScalarValue::from(added))
                    .on_conflict(on_conflict);
                match program.run(args).await {
                    Ok(outputs) => published.push(Published {
                        read: outputs
                            .snapshot()
                            .version("w")
                            .expect("w was read")
                            .number(),
                        wrote: outputs
                            .published()
                            .version("w")
                            .expect("w was written")
                            .number(),
                        added,
                        overwrote: !outputs.overwritten().is_empty(),
                    }),
                    Err(e) if matches!(Error::find(&e), Some(Error::Conflict { .. })) => {}
                    Err(e) => return Err(e),
                }
            }
            Ok(published)
        }));
    }
    let mut all = Vec::new();
    for task in tasks {
        all.extend(task.await.expect("the task ran")?);
    }
    all.sort_by_key(|p| p.wrote);

    let mut value: HashMap<u64, i64> = HashMap::from([(start, 0)]);
    for (k, p) in all.iter().enumerate() {
        assert_eq!(
            p.wrote,
            start + 1 + k as u64,
            "seed {seed}: versions {all:?}"
        );
        if !p.overwrote {
            assert_eq!(p.read + 1, p.wrote, "seed {seed}: a lost update in {all:?}");
        }
        let read = value.get(&p.read).copied();
        let read = read.unwrap_or_else(|| panic!("seed {seed}: read an unknown version {p:?}"));
        value.insert(p.wrote, read + p.added);
    }
    let last = all.last().map_or(start, |p| p.wrote);
    assert_eq!(
        sql_values(&ctx, "SELECT v FROM w").await?,
        vec![value[&last]],
        "seed {seed}: {all:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn programs_sharing_a_state_table_publish_by_compare_and_set() -> Result<()> {
    let seeds: u64 = std::env::var("REVIEW_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    for seed in 0..seeds {
        shared_state_case(seed).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Names.
// ---------------------------------------------------------------------------

/// **Assumes:** that the API has one rule for names, and that a state
/// table's name is the one SQL reads. `Accelerator::create_table` says
/// DataFusion normalizes a name as SQL does, so `Weights` is `weights`.
#[tokio::test]
async fn a_state_table_named_with_capitals_is_read_by_that_name() -> Result<()> {
    let ctx = session();
    let program = Program::from_steps(&ctx, [("Weights", "SELECT i, v + 1 AS v FROM Weights")])
        .state("Weights", rows(&[(0, 1)]))
        .lower()
        .await?
        .compile()?;
    program.run(RunArgs::new()).await?;
    assert_eq!(sql_values(&ctx, "SELECT v FROM Weights").await?, vec![2]);
    Ok(())
}

/// **Assumes:** that `state_from` takes a table by the name
/// `create_table` took, and the step computing it by that name too.
#[tokio::test]
async fn state_from_takes_the_name_create_table_took() -> Result<()> {
    let ctx = session();
    let weights = Accelerator::create_table(&ctx, "Weights", schema(), rows(&[(0, 1)]))?;
    let program = Program::from_steps(&ctx, [("Weights", "SELECT i, v + 1 AS v FROM Weights")])
        .state_from("Weights")
        .lower()
        .await?
        .compile()?;
    program.run(RunArgs::new()).await?;
    assert_eq!(values(&weights.read()), vec![2]);
    Ok(())
}

/// **Assumes:** `Program::state_from`'s doc: "A table the runtime already
/// owns, such as one an earlier program trained". A session table that
/// isn't the runtime's is refused, naming it.
#[tokio::test]
async fn state_from_refuses_a_table_the_runtime_doesnt_own() -> Result<()> {
    let ctx = session();
    ctx.sql("CREATE TABLE w AS SELECT 0 AS i, 0 AS v")
        .await?
        .collect()
        .await?;
    let Err(err) = Program::from_steps(&ctx, [("w", "SELECT i, v FROM w")])
        .state_from("w")
        .lower()
        .await
    else {
        panic!("w isn't the runtime's");
    };
    assert!(
        matches!(Error::find(&err), Some(Error::UnknownName { name, .. }) if name.contains('w')),
        "{err}"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Parameters.
// ---------------------------------------------------------------------------

/// The step `sql`, with the parameter `k` set to 1, as a program's one
/// output, against `w = (0, 1), (1, 2), (2, 3)`: `Ok(None)` if `lower`
/// refused it with `Error::ParameterPosition`, or the run's `v`s.
async fn with_k(sql: &str) -> Result<Option<Vec<i64>>> {
    let ctx = session();
    let lowered = Program::from_steps(&ctx, [("out", sql)])
        .state("w", rows(&[(0, 1), (1, 2), (2, 3)]))
        .param("k", ScalarValue::from(1_i64))
        .output("out")
        .lower()
        .await;
    let program = match lowered {
        Ok(lowered) => lowered.compile()?,
        Err(e) if matches!(Error::find(&e), Some(Error::ParameterPosition { .. })) => {
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    Ok(Some(values(
        &program.run(RunArgs::new()).await?.table("out")?,
    )))
}

/// The same `sql` with `$k` as the literal 1, in plain SQL.
async fn literal(sql: &str) -> Result<Vec<i64>> {
    let ctx = session();
    Accelerator::create_table(&ctx, "w", schema(), rows(&[(0, 1), (1, 2), (2, 3)]))?;
    sql_values(&ctx, &sql.replace("$k", "1")).await
}

/// **Assumes:** `Program`'s doc, that a parameter goes where a value is
/// read, and that where planning needs its value, `lower` fails with
/// `Error::ParameterPosition`. So a step either gives what the literal
/// gives, or is refused with that error, never another.
#[tokio::test]
#[ignore = "finding: https://github.com/xqlsystems/datafusion-accelerated/pull/40#discussion_r4581636765"]
async fn a_parameter_anywhere_is_its_literal_or_a_parameter_position_error() -> Result<()> {
    let mut wrong = Vec::new();
    for sql in [
        "SELECT i, SUM(v) OVER (ORDER BY i ROWS BETWEEN $k PRECEDING AND CURRENT ROW) AS v \
         FROM w",
        "SELECT CAST(0 AS BIGINT) AS i, u AS v FROM unnest(make_array($k, 2)) AS t(u)",
        "SELECT i, v FROM w WHERE v IN (SELECT v FROM w LIMIT $k)",
        "SELECT i, v FROM w ORDER BY i FETCH FIRST $k ROWS ONLY",
        "SELECT i, nth_value(v, $k) OVER (ORDER BY i) AS v FROM w",
    ] {
        // SQL DataFusion can't run with the literal may fail any way.
        let Ok(expected) = literal(sql).await else {
            continue;
        };
        match with_k(sql).await {
            Ok(None) => {}
            Ok(Some(found)) if found == expected => {}
            Ok(Some(found)) => wrong.push(format!("{sql}: {found:?}, not {expected:?}")),
            Err(e) => wrong.push(format!("{sql}: {e}")),
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    Ok(())
}

// ---------------------------------------------------------------------------
// Views.
// ---------------------------------------------------------------------------

/// **Assumes:** `RunOutputs::snapshot`'s doc, that it lists "the
/// runtime's tables its steps read", including through a view.
#[tokio::test]
#[ignore = "finding: https://github.com/xqlsystems/datafusion-accelerated/pull/40#discussion_r4581636759"]
async fn a_snapshot_lists_a_runtime_table_read_through_a_view() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), rows(&[(0, 1)]))?;
    ctx.sql("CREATE VIEW tv AS SELECT i, v FROM t")
        .await?
        .collect()
        .await?;
    let program = Program::from_steps(&ctx, [("out", "SELECT i, v FROM tv")])
        .output("out")
        .lower()
        .await?
        .compile()?;
    let outputs = program.run(RunArgs::new()).await?;
    assert!(
        outputs.snapshot().version("t").is_some(),
        "the snapshot is {:?}",
        outputs.snapshot()
    );
    Ok(())
}

/// **Assumes:** R1, that a run reads what the same SQL reads in the
/// session, after the table under a view is dropped and created again.
#[tokio::test]
async fn a_program_reads_a_view_as_the_session_does_after_its_table_changes() -> Result<()> {
    let ctx = session();
    Accelerator::create_table(&ctx, "t", schema(), rows(&[(0, 1)]))?;
    ctx.sql("CREATE VIEW tv AS SELECT i, v FROM t")
        .await?
        .collect()
        .await?;
    let program = Program::from_steps(&ctx, [("out", "SELECT i, v FROM tv")])
        .output("out")
        .lower()
        .await?
        .compile()?;
    ctx.sql("DROP TABLE t").await?.collect().await?;
    Accelerator::create_table(&ctx, "t", schema(), rows(&[(0, 2)]))?;
    let session = match ctx.sql("SELECT i, v FROM tv").await {
        Ok(df) => df.collect().await.map(|b| values(&b)),
        Err(e) => Err(e),
    };
    let run = program.run(RunArgs::new()).await;
    match (session, run) {
        (Ok(session), Ok(run)) => assert_eq!(values(&run.table("out")?), session),
        (Err(_), Err(_)) => {}
        (session, run) => panic!(
            "the session gives {:?}, the run {:?}",
            session.map_err(|e| e.to_string()),
            run.map(|r| values(&r.table("out").expect("out")))
                .map_err(|e| e.to_string())
        ),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// A state table dropped during a run.
// ---------------------------------------------------------------------------

/// **Assumes:** `TableHandle::replace`'s `Error::Detached` and
/// `CompiledProgram`'s `Error::Stale`, that the runtime never writes rows
/// into a table the session no longer has, so a run whose state table is
/// dropped while it runs publishes nothing and fails.
#[tokio::test]
#[ignore = "finding: https://github.com/xqlsystems/datafusion-accelerated/pull/40#discussion_r4581636752"]
async fn a_state_table_dropped_during_a_run_isnt_published_into() -> Result<()> {
    let ctx = session();
    // `drop_w(v)` is `v`, and deregisters `w` once armed.
    let armed: Arc<Mutex<Option<SessionContext>>> = Arc::default();
    {
        let armed = Arc::clone(&armed);
        ctx.register_udf(create_udf(
            "drop_w",
            vec![DataType::Int64],
            DataType::Int64,
            Volatility::Volatile,
            Arc::new(move |args: &[ColumnarValue]| {
                if let Some(ctx) = armed.lock().expect("not poisoned").take() {
                    ctx.deregister_table("w")?;
                }
                Ok(args[0].clone())
            }),
        ));
    }
    let program = Program::from_steps(&ctx, [("w", "SELECT i, drop_w(v) + 1 AS v FROM w")])
        .state("w", rows(&[(0, 1)]))
        .lower()
        .await?
        .compile()?;
    *armed.lock().expect("not poisoned") = Some(ctx.clone());
    let run = program.run(RunArgs::new()).await;
    assert!(
        run.is_err(),
        "published {:?} into a table the session dropped",
        run.map(|r| r.published().clone())
    );
    Ok(())
}
