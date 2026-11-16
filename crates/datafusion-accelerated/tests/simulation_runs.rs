// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Seed-reproducible property tests for runs (D20), over every way a plan
//! reaches a run.
//!
//! `simulation.rs` checks that one run reads one snapshot when every scan
//! gets the run's own context. This file checks the same invariant, and the
//! errors that guard it, over each path a scan's context can take:
//!
//! - the run's own context (`Run::task_ctx`);
//! - a new context for each `execute`, built from the session's settings as
//!   strings;
//! - the form datafusion-python's settings take across datafusion-ffi, an
//!   `FFI_ExtensionOptions`;
//! - plain SQL after `SET accelerated.run`;
//! - `Run::execute_stream`, with the run dropped before the stream is read.
//!
//! The property: within a run, every scan reads each table as it was when
//! the run began, whatever is replaced meanwhile. After the run ends, a plan
//! given its id fails with `RunEnded`, and one given another session's run
//! id fails the same way. A table created during a run fails with
//! `CreatedAfterRun` in that run. Outside a run, a scan reads the current
//! rows. No path reads other rows, and no path fails differently.
//!
//! Every failure prints its seed. `DFA_SEED=<seed>` with the ignored
//! `replay_one_seed` test reruns that case and prints what it ran. The
//! ignored soak explores past the bounded seed range:
//!
//! ```sh
//! DFA_SOAK_SECS=120 DFA_SOAK_BASE=0 systemd-run --user --scope -q -p MemoryMax=8G \
//!   cargo test -p datafusion-accelerated --test simulation_runs --release \
//!   -- --ignored --nocapture soak
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::config::ConfigOptions;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    AcceleratedConfig, Accelerator, AcceleratorExt, AcceleratorOptions, Error, Run, TableHandle,
};
use datafusion_ffi::config::FFI_ConfigOptions;
use datafusion_ffi::config::extension_options::FFI_ExtensionOptions;

// ---------------------------------------------------------------------------
// A deterministic PRNG, SplitMix64, as in ddx's test_utils and
// `simulation.rs`.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64, stream: u64) -> Self {
        Rng(seed ^ stream.rotate_left(32) ^ 0x9e37_79b9_7f4a_7c15)
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

    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

type Outcome = Result<(), String>;

// ---------------------------------------------------------------------------
// Tables whose rows say which table and version they are.
// ---------------------------------------------------------------------------

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

/// Two rows, `(table, version)`, so a result shows what it read.
fn tagged(table: usize, version: i64) -> Vec<RecordBatch> {
    let k = Int64Array::from(vec![table as i64; 2]);
    let v = Int64Array::from(vec![version, version]);
    vec![RecordBatch::try_new(schema(), vec![Arc::new(k), Arc::new(v)]).expect("the columns match")]
}

/// The rows of a result as sorted strings: a multiset.
fn render(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            let cells: Vec<String> = batch
                .columns()
                .iter()
                .map(|column| array_value_to_string(column, row).expect("a printable value"))
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    out
}

/// What reading `tables` at `versions` gives, as a multiset.
fn expected(tables: &[usize], versions: &[i64]) -> Vec<String> {
    let batches: Vec<RecordBatch> = tables
        .iter()
        .flat_map(|&table| tagged(table, versions[table]))
        .collect();
    render(&batches)
}

fn accelerated(partitions: usize) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(partitions))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

async fn plan(ctx: &SessionContext, sql: &str) -> Result<Arc<dyn ExecutionPlan>, String> {
    ctx.sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .create_physical_plan()
        .await
        .map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// The paths a scan's context can take.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Path {
    /// `Run::task_ctx`, for every partition.
    Native,
    /// A new context for each `execute`, from the session's settings with
    /// `accelerated.run` set as a string.
    PerExecute,
    /// The settings as datafusion-python holds them, across datafusion-ffi.
    Python,
}

const PATHS: [Path; 3] = [Path::Native, Path::PerExecute, Path::Python];

/// A context for one `execute` of a plan in run `run` (0 for no run), along
/// `path`. `run_ctx` is the run's own context, when the path uses it.
fn context(
    ctx: &SessionContext,
    path: Path,
    run: u64,
    run_ctx: Option<&Arc<TaskContext>>,
) -> Result<Arc<TaskContext>, String> {
    match path {
        Path::Native => Ok(match run_ctx {
            Some(task) => Arc::clone(task),
            None => ctx.task_ctx(),
        }),
        Path::PerExecute => {
            let mut state = ctx.state();
            state
                .config_mut()
                .options_mut()
                .set("accelerated.run", &run.to_string())
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(TaskContext::from(&state)))
        }
        Path::Python => {
            let mut settings = AcceleratedConfig::default();
            settings.run = run;
            let mut foreign = FFI_ExtensionOptions::default();
            foreign.add_config(&settings).map_err(|e| e.to_string())?;
            let mut python = ConfigOptions::default();
            python.extensions.insert(foreign);
            let options = ConfigOptions::try_from(FFI_ConfigOptions::from(&python))
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(TaskContext::new(
                None,
                "python".into(),
                SessionConfig::from(options),
                Default::default(),
                Default::default(),
                Default::default(),
                Default::default(),
                ctx.task_ctx().runtime_env(),
            )))
        }
    }
}

async fn execute(
    plan: &Arc<dyn ExecutionPlan>,
    partition: usize,
    task: Arc<TaskContext>,
) -> Result<Vec<String>, DataFusionError> {
    let stream = plan.execute(partition, task)?;
    let batches = datafusion::physical_plan::common::collect(stream).await?;
    Ok(render(&batches))
}

/// What a read should give: rows, or one of the runtime's errors.
#[derive(Debug, PartialEq, Eq)]
enum Expect {
    Rows(Vec<String>),
    RunEnded,
    CreatedAfterRun,
}

/// Checks a read's result against what the model expects.
fn check(
    got: Result<Vec<String>, DataFusionError>,
    want: &Expect,
    what: &str,
) -> Result<(), String> {
    let matches = match (&got, want) {
        (Ok(rows), Expect::Rows(want)) => rows == want,
        (Err(e), Expect::RunEnded) => matches!(Error::find(e), Some(Error::RunEnded { .. })),
        (Err(e), Expect::CreatedAfterRun) => {
            matches!(Error::find(e), Some(Error::CreatedAfterRun { .. }))
        }
        _ => false,
    };
    if matches {
        Ok(())
    } else {
        let got = got.map_err(|e| e.to_string());
        Err(format!("{what}\n  expected {want:?}\n  got {got:?}"))
    }
}

// ---------------------------------------------------------------------------
// The property.
// ---------------------------------------------------------------------------

/// The ways one run in a case is read.
#[derive(Debug, Clone, Copy)]
enum Mode {
    /// Each partition executed on its own, in a random order, with
    /// replaces between them.
    Partitions(Path),
    /// `Run::execute_stream` over the whole plan, with replaces before the
    /// stream is read and, sometimes, the run dropped first.
    Stream { drop_run_first: bool },
    /// Plain SQL after `SET accelerated.run`, then `SET accelerated.run = 0`.
    Sql,
}

struct World {
    ctx: SessionContext,
    handles: Vec<TableHandle>,
    versions: Vec<i64>,
    log: Vec<String>,
}

impl World {
    fn replace_some(&mut self, rng: &mut Rng) -> Result<(), String> {
        for table in 0..self.handles.len() {
            if rng.chance(40) {
                self.versions[table] += 1;
                self.handles[table]
                    .replace(tagged(table, self.versions[table]))
                    .map_err(|e| e.to_string())?;
                self.log
                    .push(format!("replace t{table} -> {}", self.versions[table]));
            }
        }
        Ok(())
    }
}

async fn runs_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 7);
    let ctx = accelerated(1 + rng.below(4) as usize);
    let tables = 2 + rng.below(2) as usize;
    let mut handles = Vec::new();
    for table in 0..tables {
        handles.push(
            Accelerator::create_table(&ctx, &format!("t{table}"), schema(), tagged(table, 0))
                .map_err(|e| e.to_string())?,
        );
    }
    let mut world = World {
        ctx,
        handles,
        versions: vec![0; tables],
        log: Vec::new(),
    };
    // Another session, whose runs' ids this session must never accept.
    let other = accelerated(1);
    Accelerator::create_table(&other, "t0", schema(), tagged(9, 0)).map_err(|e| e.to_string())?;

    // One plan, planned before any run: one scan per UNION ALL arm.
    let arms: Vec<usize> = (0..1 + rng.below(4))
        .map(|_| rng.below(tables as u64) as usize)
        .collect();
    let sql = arms
        .iter()
        .map(|t| format!("SELECT k, v FROM t{t}"))
        .collect::<Vec<_>>()
        .join(" UNION ALL ");
    let plan = plan(&world.ctx, &sql).await?;
    let partitions = plan.properties().partitioning.partition_count();
    let per_partition = partitions == arms.len();
    let fail = |world: &World, message: String| {
        format!(
            "runs seed={seed}\n  sql: {sql}\n  {message}\n  log: {:?}",
            world.log
        )
    };

    let mut late = 0usize;
    for run_index in 0..2 + rng.below(3) {
        let mode = match rng.below(if per_partition { 5 } else { 2 }) {
            0 => Mode::Stream {
                drop_run_first: rng.chance(50),
            },
            1 => Mode::Sql,
            n => Mode::Partitions(PATHS[(n - 2) as usize]),
        };
        world.log.push(format!("run {run_index}: {mode:?}"));
        let run = Accelerator::begin_run(&world.ctx).map_err(|e| e.to_string())?;
        let id = run.id();
        let pinned = world.versions.clone();
        let path = match mode {
            Mode::Partitions(path) => path,
            _ => Path::Native,
        };

        match mode {
            Mode::Partitions(path) => {
                let run_ctx = run.task_ctx();
                let mut order: Vec<usize> = (0..partitions).collect();
                rng.shuffle(&mut order);
                for partition in order {
                    world.replace_some(&mut rng)?;
                    let task = context(&world.ctx, path, id, Some(&run_ctx))?;
                    let got = execute(&plan, partition, task).await;
                    let want = Expect::Rows(expected(&[arms[partition]], &pinned));
                    check(got, &want, &format!("partition {partition}"))
                        .map_err(|m| fail(&world, m))?;
                }
            }
            Mode::Stream { drop_run_first } => {
                let stream = run
                    .execute_stream(Arc::clone(&plan))
                    .map_err(|e| fail(&world, e.to_string()))?;
                world.replace_some(&mut rng)?;
                let run = if drop_run_first {
                    drop(run);
                    None
                } else {
                    Some(run)
                };
                let got = datafusion::physical_plan::common::collect(stream)
                    .await
                    .map(|b| render(&b));
                world.replace_some(&mut rng)?;
                check(got, &Expect::Rows(expected(&arms, &pinned)), "stream")
                    .map_err(|m| fail(&world, m))?;
                drop(run);
                // The stream is gone, so the run has ended, whichever was
                // dropped first.
                let task = context(&world.ctx, Path::PerExecute, id, None)?;
                let got = execute(&plan, 0, task).await;
                check(got, &Expect::RunEnded, "after the stream").map_err(|m| fail(&world, m))?;
                continue;
            }
            Mode::Sql => {
                world
                    .ctx
                    .sql(&format!("SET accelerated.run = {id}"))
                    .await
                    .map_err(|e| e.to_string())?
                    .collect()
                    .await
                    .map_err(|e| e.to_string())?;
                world.replace_some(&mut rng)?;
                let got = async {
                    let batches = world.ctx.sql(&sql).await?.collect().await?;
                    Ok(render(&batches))
                }
                .await;
                check(got, &Expect::Rows(expected(&arms, &pinned)), "SQL in a run")
                    .map_err(|m| fail(&world, m))?;
            }
        }

        // A table created during the run is outside its snapshot.
        if rng.chance(30) {
            let name = format!("late{late}");
            late += 1;
            let created = Accelerator::create_table(&world.ctx, &name, schema(), tagged(7, 0))
                .map_err(|e| e.to_string())?;
            world.log.push(format!("create {name}"));
            let late_plan = self::plan(&world.ctx, &format!("SELECT k, v FROM {name}")).await?;
            let task = context(&world.ctx, path, id, Some(&run.task_ctx()))?;
            let got = execute(&late_plan, 0, task).await;
            check(got, &Expect::CreatedAfterRun, &format!("{name} in the run"))
                .map_err(|m| fail(&world, m))?;
            // Not `Native`: after `SET accelerated.run`, the session's own
            // context is in the run.
            let outside = if path == Path::Native {
                Path::PerExecute
            } else {
                path
            };
            let task = context(&world.ctx, outside, 0, None)?;
            let got = execute(&late_plan, 0, task).await;
            check(
                got,
                &Expect::Rows(render(&tagged(7, 0))),
                &format!("{name} outside"),
            )
            .map_err(|m| fail(&world, m))?;
            drop(created);
        }

        // Another session's live run is not this session's.
        if rng.chance(20) {
            let foreign: Run = Accelerator::begin_run(&other).map_err(|e| e.to_string())?;
            let task = context(&world.ctx, Path::PerExecute, foreign.id(), None)?;
            let got = execute(&plan, 0, task).await;
            check(got, &Expect::RunEnded, "another session's run").map_err(|m| fail(&world, m))?;
        }

        // A caller may keep the run's context after the run ends.
        let run_task = run.task_ctx();
        drop(run);
        world.log.push(format!("end run {run_index}"));

        // The ended run's id fails along the same path, never reads rows.
        match mode {
            Mode::Sql => {
                let got = async {
                    let batches = world.ctx.sql(&sql).await?.collect().await?;
                    Ok(render(&batches))
                }
                .await;
                check(got, &Expect::RunEnded, "stale SET accelerated.run")
                    .map_err(|m| fail(&world, m))?;
                world
                    .ctx
                    .sql("SET accelerated.run = 0")
                    .await
                    .map_err(|e| e.to_string())?
                    .collect()
                    .await
                    .map_err(|e| e.to_string())?;
            }
            _ => {
                let task = context(&world.ctx, path, id, Some(&run_task))?;
                let partition = rng.below(partitions as u64) as usize;
                let got = execute(&plan, partition, task).await;
                check(got, &Expect::RunEnded, "after the run").map_err(|m| fail(&world, m))?;
            }
        }

        // Outside a run, every path reads the current rows.
        let outside = PATHS[rng.below(3) as usize];
        let task = context(&world.ctx, outside, 0, None)?;
        if per_partition {
            let partition = rng.below(partitions as u64) as usize;
            let got = execute(&plan, partition, task).await;
            let want = Expect::Rows(expected(&[arms[partition]], &world.versions));
            check(got, &want, &format!("outside a run, {outside:?}"))
                .map_err(|m| fail(&world, m))?;
        }
    }
    if verbose {
        eprintln!("sql: {sql}\nlog: {:?}", world.log);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Drivers, as in `simulation.rs`.
// ---------------------------------------------------------------------------

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::test]
async fn every_path_to_a_run_reads_its_snapshot() {
    let mut failures = Vec::new();
    for seed in 0..200 {
        if let Err(message) = runs_case(seed, false).await {
            failures.push(message);
        }
    }
    assert!(
        failures.is_empty(),
        "{} failing seed(s); the first three:\n{}",
        failures.len(),
        failures
            .iter()
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[tokio::test]
#[ignore = "soak: runs for DFA_SOAK_SECS seconds"]
async fn soak() {
    let budget = Duration::from_secs(env_u64("DFA_SOAK_SECS", 15));
    let base = env_u64("DFA_SOAK_BASE", 1 << 20);
    let start = Instant::now();
    let mut failures = BTreeSet::new();
    let mut seed = base;
    while start.elapsed() < budget {
        if let Err(message) = runs_case(seed, false).await {
            eprintln!("\nFAILURE (seed={seed}):\n{message}");
            failures.insert(seed);
        }
        seed = seed.wrapping_add(1);
    }
    eprintln!(
        "soak: {} cases in {:?}",
        seed.wrapping_sub(base),
        start.elapsed()
    );
    assert!(
        failures.is_empty(),
        "{} failing case(s), each with its seed above",
        failures.len()
    );
}

#[tokio::test]
#[ignore = "replay: set DFA_SEED"]
async fn replay_one_seed() {
    let seed = env_u64("DFA_SEED", 0);
    if let Err(message) = runs_case(seed, true).await {
        panic!("{message}");
    }
}
