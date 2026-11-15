// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Seed-reproducible property tests for table handles and run snapshots.
//!
//! Each property states an invariant from the design and searches for
//! inputs that break it, as ddx's `simulation.rs` and einfold's equivalence
//! harness do:
//!
//! - **Plans outlive their rows (D16, R1).** A plan planned once over
//!   runtime tables, then run after any sequence of replaces, returns what
//!   DataFusion returns for the same SQL over plain in-memory tables of the
//!   same rows. The generator varies row counts (including none), batch
//!   splits (including empty batches and no batches), NULLs, key ties and
//!   the session's partition count, since planning reads statistics from
//!   rows that a replace then changes.
//! - **One run reads one snapshot (D20).** Scans of one run, executed in a
//!   random order with replaces between them, each read the tables as they
//!   were when the run began, and the next run reads the newest rows.
//! - **The runtime and the session agree on names.** After any sequence of
//!   creates, deregistrations, replaces, `DROP TABLE`s and plain registrations,
//!   the runtime owns a name exactly when the session's table of that name
//!   is the runtime's, and each operation succeeds exactly when a model of
//!   one shared namespace says it should.
//!
//! Every failure prints its seed. `DFA_SEED=<seed>` with the ignored
//! `replay_one_seed` test reruns only that case and prints what it ran.
//! The ignored soak explores past the bounded seed ranges:
//!
//! ```sh
//! DFA_SOAK_SECS=120 DFA_SOAK_BASE=0 systemd-run --user --scope -q -p MemoryMax=8G \
//!   cargo test -p datafusion-accelerated --test simulation --release \
//!   -- --ignored --nocapture soak
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{ArrayRef, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::datasource::MemTable;
use datafusion::execution::TaskContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::execution_plan::reset_plan_states;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{Accelerator, AcceleratorExt, AcceleratorOptions, TableHandle};

// ---------------------------------------------------------------------------
// A deterministic PRNG, SplitMix64, as in ddx's test_utils: no dependencies,
// and a seed reported in a failure reproduces it on any machine.
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

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

type Outcome = Result<(), String>;

// ---------------------------------------------------------------------------
// Tables: t(k, v) and u(k, w), Int64, with NULLs in the value columns and
// few distinct keys, so joins and groups have ties.
// ---------------------------------------------------------------------------

fn schema(value: &str) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new(value, DataType::Int64, true),
    ]))
}

/// Random rows for a table, split into random batches.
fn rows(rng: &mut Rng, schema: &SchemaRef) -> Vec<RecordBatch> {
    let n = match rng.below(10) {
        0 => 0,
        1 => 1,
        2 => 300 + rng.below(700) as usize,
        _ => rng.below(40) as usize,
    };
    let keys = 1 + rng.below(8) as i64;
    let k: Vec<i64> = (0..n).map(|_| rng.below(keys as u64) as i64).collect();
    let v: Vec<Option<i64>> = (0..n)
        .map(|_| (!rng.chance(15)).then(|| rng.below(101) as i64 - 50))
        .collect();
    let batch = |start: usize, end: usize| {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(k[start..end].to_vec())),
            Arc::new(Int64Array::from(v[start..end].to_vec())),
        ];
        RecordBatch::try_new(Arc::clone(schema), columns).expect("the columns match")
    };
    // Single rows, long runs, and empty batches anywhere, including a table
    // of empty batches only.
    let mut batches = Vec::new();
    let mut start = 0;
    loop {
        if rng.chance(10) {
            batches.push(batch(start, start));
        }
        if start == n {
            break;
        }
        let len = if rng.chance(30) {
            1
        } else {
            1 + rng.below((n - start) as u64) as usize
        };
        batches.push(batch(start, start + len));
        start += len;
    }
    batches
}

fn count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// The rows of a result as sorted strings: a multiset, since SQL's
/// results are unordered unless the query orders them.
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

fn accelerated(partitions: usize) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(partitions))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

// ---------------------------------------------------------------------------
// Property 1: plans outlive their rows.
// ---------------------------------------------------------------------------

/// Queries whose results are a deterministic multiset, so they compare
/// across engines and plans. `{c}` is a random constant.
const QUERIES: &[&str] = &[
    "SELECT COUNT(*) FROM t",
    "SELECT COUNT(v), SUM(v), MIN(v), MAX(v) FROM t",
    "SELECT k, COUNT(*), SUM(v) FROM t GROUP BY k",
    "SELECT DISTINCT k FROM t",
    "SELECT k, v FROM t WHERE v > {c}",
    "SELECT COUNT(*) FROM (SELECT * FROM t LIMIT {c})",
    "SELECT k, v FROM t ORDER BY k, v NULLS FIRST LIMIT {c}",
    "SELECT t.k, t.v, u.w FROM t JOIN u ON t.k = u.k",
    "SELECT t.k, t.v, u.w FROM t LEFT JOIN u ON t.k = u.k",
    "SELECT u.k, u.w, t.v FROM t RIGHT JOIN u ON t.k = u.k",
    "SELECT COUNT(*) FROM t JOIN u ON t.k = u.k",
    "SELECT k FROM t WHERE k IN (SELECT k FROM u)",
    "SELECT k FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k)",
    "SELECT (SELECT COUNT(*) FROM u) AS n, COUNT(*) AS m FROM t",
    "SELECT k FROM t UNION ALL SELECT k FROM u",
    "SELECT k, SUM(v) OVER (PARTITION BY k) FROM t",
    "SELECT a.k, COUNT(*) FROM t a JOIN t b ON a.k = b.k GROUP BY a.k",
    "SELECT t.k, SUM(u.w) FROM t JOIN u ON t.k = u.k WHERE t.v > {c} GROUP BY t.k",
];

struct Case {
    sql: String,
    partitions: usize,
}

fn query(rng: &mut Rng) -> Case {
    let template = rng.pick(QUERIES);
    let constant = rng.below(60) as i64 - 20;
    Case {
        sql: template.replace("{c}", &constant.max(0).to_string()),
        partitions: *rng.pick(&[1, 2, 4, 8]),
    }
}

/// What DataFusion returns for `sql` over plain in-memory tables.
async fn reference(sql: &str, t: &[RecordBatch], u: &[RecordBatch]) -> Result<Vec<String>, String> {
    let ctx = SessionContext::new();
    for (name, value, batches) in [("t", "v", t), ("u", "w", u)] {
        let table =
            MemTable::try_new(schema(value), vec![batches.to_vec()]).map_err(|e| e.to_string())?;
        ctx.register_table(name, Arc::new(table))
            .map_err(|e| e.to_string())?;
    }
    let batches = ctx
        .sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .collect()
        .await
        .map_err(|e| e.to_string())?;
    Ok(render(&batches))
}

async fn run(ctx: &SessionContext, plan: &Arc<dyn ExecutionPlan>) -> Result<Vec<String>, String> {
    let plan = reset_plan_states(Arc::clone(plan)).map_err(|e| e.to_string())?;
    let batches = collect(plan, ctx.task_ctx())
        .await
        .map_err(|e| e.to_string())?;
    Ok(render(&batches))
}

async fn plans_outlive_their_rows_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 1);
    let case = query(&mut rng);
    let ctx = accelerated(case.partitions);
    let mut t_rows = rows(&mut rng, &schema("v"));
    let mut u_rows = rows(&mut rng, &schema("w"));
    let t = Accelerator::create_table(&ctx, "t", schema("v"), t_rows.clone())
        .map_err(|e| e.to_string())?;
    let u = Accelerator::create_table(&ctx, "u", schema("w"), u_rows.clone())
        .map_err(|e| e.to_string())?;
    let plan = ctx
        .sql(&case.sql)
        .await
        .map_err(|e| e.to_string())?
        .create_physical_plan()
        .await
        .map_err(|e| e.to_string())?;
    if verbose {
        eprintln!(
            "sql: {}\npartitions: {}\nplan:\n{}",
            case.sql,
            case.partitions,
            datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
        );
    }
    let runs = 2 + rng.below(4);
    for step in 0..runs {
        // The first run reads the rows the plan was planned on.
        if step > 0 {
            if rng.chance(60) {
                t_rows = rows(&mut rng, &schema("v"));
                t.replace(t_rows.clone()).map_err(|e| e.to_string())?;
            }
            if rng.chance(60) {
                u_rows = rows(&mut rng, &schema("w"));
                u.replace(u_rows.clone()).map_err(|e| e.to_string())?;
            }
        }
        let expected = reference(&case.sql, &t_rows, &u_rows).await?;
        let got = run(&ctx, &plan).await?;
        if got != expected {
            return Err(format!(
                "plans_outlive_their_rows seed={seed} run={step}\n  sql: {}\n  partitions: {}\n  \
                 rows: t={} in {} batches, u={} in {} batches\n  expected {} rows: {:?}\n  got {} rows: {:?}",
                case.sql,
                case.partitions,
                count(&t_rows),
                t_rows.len(),
                count(&u_rows),
                u_rows.len(),
                expected.len(),
                expected.iter().take(8).collect::<Vec<_>>(),
                got.len(),
                got.iter().take(8).collect::<Vec<_>>(),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 2: one run reads one snapshot.
// ---------------------------------------------------------------------------

/// Rows that say which version of which table they are.
fn tagged(schema: &SchemaRef, table: usize, version: i64) -> Vec<RecordBatch> {
    let k = Int64Array::from(vec![table as i64; 2]);
    let v = Int64Array::from(vec![version, version]);
    vec![
        RecordBatch::try_new(Arc::clone(schema), vec![Arc::new(k), Arc::new(v)])
            .expect("the columns match"),
    ]
}

async fn read_partition(
    plan: &Arc<dyn ExecutionPlan>,
    partition: usize,
    task: &Arc<TaskContext>,
) -> Result<Vec<String>, String> {
    let stream = plan
        .execute(partition, Arc::clone(task))
        .map_err(|e| e.to_string())?;
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .map_err(|e| e.to_string())?;
    Ok(render(&batches))
}

async fn one_run_reads_one_snapshot_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 2);
    let ctx = accelerated(1 + rng.below(4) as usize);
    let schemas = [schema("v"), schema("v")];
    let names = ["t", "u"];
    let mut versions = [0i64, 0];
    let handles: Vec<TableHandle> = (0..2)
        .map(|i| {
            Accelerator::create_table(
                &ctx,
                names[i],
                Arc::clone(&schemas[i]),
                tagged(&schemas[i], i, 0),
            )
        })
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;

    // One scan per UNION ALL arm, each its own partition of the plan.
    let scans: Vec<usize> = (0..2 + rng.below(4))
        .map(|_| rng.below(2) as usize)
        .collect();
    let sql = scans
        .iter()
        .map(|&i| format!("SELECT k, v FROM {}", names[i]))
        .collect::<Vec<_>>()
        .join(" UNION ALL ");
    let plan = ctx
        .sql(&sql)
        .await
        .map_err(|e| e.to_string())?
        .create_physical_plan()
        .await
        .map_err(|e| e.to_string())?;
    let partitions = plan.properties().partitioning.partition_count();
    if partitions != scans.len() {
        // The planner merged arms; this case can't run scans one at a time.
        return Ok(());
    }
    if verbose {
        eprintln!("sql: {sql}");
    }

    for run_index in 0..3 {
        let run = Accelerator::begin_run(&ctx).map_err(|e| e.to_string())?;
        let task = run.task_ctx();
        let pinned = versions;
        let mut order: Vec<usize> = (0..scans.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i as u64 + 1) as usize);
        }
        for &partition in &order {
            // Replace either table, or both, between scans.
            for i in 0..2 {
                if rng.chance(50) {
                    versions[i] += 1;
                    handles[i]
                        .replace(tagged(&schemas[i], i, versions[i]))
                        .map_err(|e| e.to_string())?;
                }
            }
            let table = scans[partition];
            let got = read_partition(&plan, partition, &task).await?;
            let expected = render(&tagged(&schemas[table], table, pinned[table]));
            if got != expected {
                return Err(format!(
                    "one_run_reads_one_snapshot seed={seed} run={run_index} partition={partition}\n  \
                     sql: {sql}\n  order: {order:?}\n  expected {} at version {}: {expected:?}\n  got {got:?}",
                    names[table], pinned[table]
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 3: the runtime and the session agree on names.
// ---------------------------------------------------------------------------

/// A name as a caller writes it, and the name SQL resolves it to: an
/// unquoted identifier is lowercased, a quoted one is kept.
const NAMES: &[(&str, &str)] = &[("t", "t"), ("T", "t"), ("\"T\"", "T"), ("u", "u")];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    Runtime,
    Session,
}

#[derive(Debug)]
enum Op {
    Create,
    Deregister,
    Replace,
    DropSql,
    RegisterPlain,
}

async fn names_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 3);
    let ctx = accelerated(2);
    let schema = schema("v");
    // The model: one namespace, each name owned by the runtime or by the
    // session, with its row count.
    let mut model: BTreeMap<&str, (Owner, usize)> = BTreeMap::new();
    let mut log = Vec::new();
    for step in 0..12 {
        let op = rng.pick(&[
            Op::Create,
            Op::Deregister,
            Op::Replace,
            Op::DropSql,
            Op::RegisterPlain,
        ]);
        let &(raw, resolved) = rng.pick(NAMES);
        let batches = rows(&mut rng, &schema);
        let n = count(&batches);
        log.push(format!("{op:?}({raw})"));
        let fail = |what: String, log: &[String]| {
            Err(format!(
                "names seed={seed} step={step}\n  ops: {}\n  {what}",
                log.join(", ")
            ))
        };
        match op {
            Op::Create => {
                let got = Accelerator::create_table(&ctx, raw, Arc::clone(&schema), batches);
                let expected = !model.contains_key(resolved);
                if got.is_ok() != expected {
                    return fail(
                        format!(
                            "create({raw}) gave {:?}, model expected ok={expected}",
                            got.err()
                        ),
                        &log,
                    );
                }
                if expected {
                    model.insert(resolved, (Owner::Runtime, n));
                }
            }
            Op::Deregister => {
                // Deregistering a name frees it, whoever owned it, and
                // deregistering a free name does nothing.
                if let Err(e) = ctx.deregister_table(raw) {
                    return fail(format!("deregister_table({raw}) gave {e}"), &log);
                }
                model.remove(resolved);
            }
            Op::Replace => {
                let owned = model
                    .get(resolved)
                    .is_some_and(|(owner, _)| *owner == Owner::Runtime);
                let found = Accelerator::table(&ctx, raw)
                    .await
                    .map_err(|e| e.to_string())?;
                match (found, owned) {
                    (Some(handle), true) => {
                        handle.replace(batches).map_err(|e| e.to_string())?;
                        model.insert(resolved, (Owner::Runtime, n));
                    }
                    (None, false) => {}
                    (got, _) => {
                        return fail(
                            format!(
                                "table({raw}) is {}, but the model says owned={owned}",
                                got.is_some()
                            ),
                            &log,
                        );
                    }
                }
            }
            Op::DropSql => {
                let got = match ctx.sql(&format!("DROP TABLE {raw}")).await {
                    Ok(frame) => frame.collect().await.map(|_| ()),
                    Err(e) => Err(e),
                };
                let exists = model.contains_key(resolved);
                match (&got, exists) {
                    // Dropping a name frees it, whoever owned it.
                    (Ok(()), true) => {
                        model.remove(resolved);
                    }
                    // Refusing to drop a runtime table is also consistent.
                    (Err(_), true) if model[resolved].0 == Owner::Runtime => {}
                    (Err(_), false) => {}
                    _ => {
                        return fail(
                            format!("DROP TABLE {raw} gave {got:?} with exists={exists}"),
                            &log,
                        );
                    }
                }
            }
            Op::RegisterPlain => {
                let table = MemTable::try_new(Arc::clone(&schema), vec![batches])
                    .map_err(|e| e.to_string())?;
                let got = ctx.register_table(raw, Arc::new(table));
                let expected = !model.contains_key(resolved);
                if got.is_ok() != expected {
                    return fail(
                        format!(
                            "register_table({raw}) gave {:?}, expected ok={expected}",
                            got.err()
                        ),
                        &log,
                    );
                }
                if expected {
                    model.insert(resolved, (Owner::Session, n));
                }
            }
        }

        // The invariants, after every operation.
        for &(raw, resolved) in NAMES {
            let owned = model
                .get(resolved)
                .is_some_and(|(owner, _)| *owner == Owner::Runtime);
            let found = Accelerator::table(&ctx, raw)
                .await
                .map_err(|e| e.to_string())?;
            if found.is_some() != owned {
                return fail(
                    format!(
                        "table({raw}) is {}, but the session's {resolved} owned-by-runtime={owned}",
                        !owned
                    ),
                    &log,
                );
            }
            let exists = ctx.table_exist(raw).map_err(|e| e.to_string())?;
            if exists != model.contains_key(resolved) {
                return fail(
                    format!("table_exist({raw}) = {exists}, model says {}", !exists),
                    &log,
                );
            }
            if let Some(&(_, rows)) = model.get(resolved) {
                let got = ctx
                    .sql(&format!("SELECT COUNT(*) FROM {raw}"))
                    .await
                    .map_err(|e| e.to_string())?
                    .collect()
                    .await
                    .map_err(|e| e.to_string())?;
                let got = render(&got);
                if got != vec![rows.to_string()] {
                    return fail(
                        format!("COUNT(*) FROM {raw} = {got:?}, model says {rows}"),
                        &log,
                    );
                }
            }
        }
    }
    if verbose {
        eprintln!("ops: {}", log.join(", "));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The harness.
// ---------------------------------------------------------------------------

async fn case(property: usize, seed: u64, verbose: bool) -> Outcome {
    match property {
        0 => plans_outlive_their_rows_case(seed, verbose).await,
        1 => one_run_reads_one_snapshot_case(seed, verbose).await,
        _ => names_case(seed, verbose).await,
    }
}

/// Runs seeds `seeds` of one property and fails with every failing seed.
async fn bounded(property: usize, seeds: std::ops::Range<u64>) {
    let mut failures = Vec::new();
    for seed in seeds {
        if let Err(message) = case(property, seed, false).await {
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

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::test]
async fn plans_outlive_their_rows() {
    bounded(0, 0..64).await;
}

#[tokio::test]
async fn one_run_reads_one_snapshot() {
    bounded(1, 0..200).await;
}

#[tokio::test]
async fn the_runtime_and_the_session_agree_on_names() {
    bounded(2, 0..200).await;
}

#[tokio::test]
#[ignore = "soak: runs for DFA_SOAK_SECS seconds"]
async fn soak() {
    let budget = Duration::from_secs(env_u64("DFA_SOAK_SECS", 15));
    let base = env_u64("DFA_SOAK_BASE", 1 << 20);
    let start = Instant::now();
    let mut failures = BTreeSet::new();
    let mut iterations = 0u64;
    while start.elapsed() < budget {
        let seed = base.wrapping_add(iterations / 3);
        let property = (iterations % 3) as usize;
        if let Err(message) = case(property, seed, false).await {
            eprintln!("\nFAILURE (property={property}, seed={seed}):\n{message}");
            failures.insert((property, seed));
        }
        iterations += 1;
    }
    eprintln!("soak: {iterations} cases in {:?}", start.elapsed());
    assert!(
        failures.is_empty(),
        "{} failing case(s), each with its seed above",
        failures.len()
    );
}

#[tokio::test]
#[ignore = "replay: set DFA_SEED (and DFA_PROPERTY=0, 1 or 2)"]
async fn replay_one_seed() {
    let seed = env_u64("DFA_SEED", 0);
    let property = env_u64("DFA_PROPERTY", 0) as usize;
    if let Err(message) = case(property, seed, true).await {
        panic!("{message}");
    }
}
