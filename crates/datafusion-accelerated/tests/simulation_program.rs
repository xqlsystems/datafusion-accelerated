// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Seed-reproducible property tests for programs (design §5.4, D20, D21).
//!
//! - **A program's runs equal its `SELECT`s run plainly (R1).** A random
//!   program of steps, each a named `SELECT` over state tables, an input,
//!   parameters and earlier steps, listed in a random order, runs several
//!   times with new inputs and parameters. Each run's outputs and published
//!   state equal what DataFusion gives for the same SQL on plain in-memory
//!   tables, run in dataflow order with each parameter written as a
//!   literal, where a state table always means its value when the run
//!   began. A run whose reference fails fails too, and publishes nothing.
//! - **Versions and conflicts follow a sequential model (R5, R10).** Runs
//!   that succeed, runs whose step fails, replaces between runs and during
//!   runs, and `OnConflict::Fail` or `Overwrite`, in random interleavings.
//!   Each table's version, rows, identity, the run's snapshot, what it
//!   published and what it overwrote match a model of the store.
//! - **Publication is atomic.** While a program's runs publish two state
//!   tables, which they keep equal, a concurrent reader's runs never see
//!   one table's new value with the other's old one.
//!
//! Every failure prints its seed. `DFA_SEED=<seed>` and `DFA_PROPERTY` with
//! the ignored `replay_one_seed` test rerun one case and print it. The
//! ignored soak explores past the bounded seed ranges:
//!
//! ```sh
//! DFA_SOAK_SECS=120 DFA_SOAK_BASE=0 systemd-run --user --scope -q -p MemoryMax=8G \
//!   cargo test -p datafusion-accelerated --test simulation_program --release \
//!   -- --ignored --nocapture soak
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use datafusion::arrow::array::Int64Array;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{
    Accelerator, AcceleratorExt, AcceleratorOptions, Error, OnConflict, Program, RunArgs,
    TableHandle,
};

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

    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }

    fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}

type Outcome = Result<(), String>;

// ---------------------------------------------------------------------------
// Tables: every table is (i, v), Int64, so templates compose. Integers, not
// floats, so results compare exactly however a plan orders a sum.
// ---------------------------------------------------------------------------

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}

fn batch(rows: &[(i64, Option<i64>)]) -> RecordBatch {
    let i: Int64Array = rows.iter().map(|(i, _)| *i).collect();
    let v: Int64Array = rows.iter().map(|(_, v)| *v).collect();
    RecordBatch::try_new(schema(), vec![Arc::new(i), Arc::new(v)]).expect("the columns match")
}

/// Random rows, split into one to three batches, some empty.
fn rows(rng: &mut Rng, keys: i64, max: u64) -> Vec<RecordBatch> {
    let n = rng.below(max + 1) as usize;
    let all: Vec<(i64, Option<i64>)> = (0..n)
        .map(|_| {
            let i = rng.int(0, keys - 1);
            let v = (!rng.chance(10)).then(|| rng.int(-9, 9));
            (i, v)
        })
        .collect();
    let cut = rng.below(n as u64 + 1) as usize;
    let mut out = vec![batch(&all[..cut]), batch(&all[cut..])];
    if rng.chance(30) {
        out.push(batch(&[]));
    }
    out
}

/// A state table: one row per key, so a join on `i` keeps its shape.
fn state_rows(rng: &mut Rng, keys: i64) -> Vec<RecordBatch> {
    let all: Vec<(i64, Option<i64>)> = (0..keys).map(|i| (i, Some(rng.int(-9, 9)))).collect();
    vec![batch(&all)]
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

fn accelerated(partitions: usize) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(partitions))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    SessionContext::new_with_state(state)
}

fn mem_table(batches: &[RecordBatch], schema: SchemaRef) -> Result<Arc<MemTable>, String> {
    let batches: Vec<RecordBatch> = batches
        .iter()
        .map(|b| b.clone().with_schema(Arc::clone(&schema)))
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(
        MemTable::try_new(schema, vec![batches]).map_err(|e| e.to_string())?,
    ))
}

// ---------------------------------------------------------------------------
// Property 0: a program's runs equal its SELECTs run plainly.
// ---------------------------------------------------------------------------

/// Steps over one source `X` or two, `X` and `Y`. Each gives `(i, v)`.
/// `$a` and `$k` are the program's parameters, in the positions a literal
/// may take.
const UNARY: &[&str] = &[
    "SELECT i, v + $a AS v FROM {x}",
    "SELECT i, SUM(v) AS v FROM {x} GROUP BY i",
    "SELECT i, v FROM {x} WHERE v > $k",
    "SELECT i, CASE WHEN v % 2 = 0 THEN v ELSE $a END AS v FROM {x}",
    "SELECT i, MAX(v) AS v FROM {x} GROUP BY i HAVING COUNT(*) > $k",
    "SELECT i, SUM(v) OVER (PARTITION BY i) AS v FROM {x}",
    "SELECT i, COUNT(v) * $a AS v FROM {x} GROUP BY i",
    "SELECT DISTINCT i, v FROM {x}",
    "SELECT i, v FROM {x} ORDER BY i, v NULLS FIRST LIMIT 3",
    "SELECT i, (SELECT COUNT(*) FROM {x}) + $k AS v FROM {x}",
];

const BINARY: &[&str] = &[
    "SELECT {x}.i, ({x}.v * {y}.v) % 101 AS v FROM {x} JOIN {y} ON {x}.i = {y}.i",
    "SELECT {x}.i, COALESCE({y}.v, $k) AS v FROM {x} LEFT JOIN {y} ON {x}.i = {y}.i",
    "SELECT i, v FROM {x} UNION ALL SELECT i, v FROM {y}",
    "SELECT i, v FROM {x} WHERE i IN (SELECT i FROM {y})",
    "SELECT i, v FROM {x} WHERE NOT EXISTS (SELECT 1 FROM {y} WHERE {y}.i = {x}.i AND {y}.v > $k)",
    "SELECT {x}.i, {x}.v - {y}.v AS v FROM {x} JOIN {y} ON {x}.i = {y}.i AND {x}.v > $k",
];

/// The next value of a state table `{s}` from a step `{x}`: one row per key
/// of `{s}`, bounded, so values never overflow however many runs there are.
const NEXT: &str = "SELECT {s}.i, (COALESCE({s}.v, 0) + COALESCE(x_.v, 0) * $a) % 1009 AS v \
                    FROM {s} LEFT JOIN (SELECT i, SUM(v) AS v FROM {x} GROUP BY i) x_ \
                    ON {s}.i = x_.i";

fn fill(template: &str, x: &str, y: &str) -> String {
    template.replace("{x}", x).replace("{y}", y)
}

/// Writes each parameter as a literal, as the reference runs it.
fn literal(sql: &str, params: &BTreeMap<&str, i64>) -> String {
    let mut out = sql.to_string();
    for (name, value) in params {
        out = out.replace(&format!("${name}"), &format!("CAST({value} AS BIGINT)"));
    }
    out
}

struct Generated {
    /// In dataflow order: each step only reads names defined before it.
    steps: Vec<(String, String)>,
    states: Vec<&'static str>,
}

fn generate(rng: &mut Rng) -> Generated {
    let mut states = vec!["w"];
    if rng.chance(50) {
        states.push("b");
    }
    let mut sources: Vec<String> = states.iter().map(|s| s.to_string()).collect();
    sources.push("d".to_string());
    let mut steps = Vec::new();
    for j in 0..1 + rng.below(4) {
        let name = format!("s{j}");
        let x = rng.pick(&sources).clone();
        let others: Vec<String> = sources.iter().filter(|s| **s != x).cloned().collect();
        let sql = if !others.is_empty() && rng.chance(45) {
            let y = rng.pick(&others).clone();
            fill(rng.pick(BINARY), &x, &y)
        } else {
            fill(rng.pick(UNARY), &x, &x)
        };
        steps.push((name.clone(), sql));
        sources.push(name);
    }
    // Each state's next value, from a step or the input; `b` sometimes has
    // none and keeps its value.
    for state in &states {
        if *state == "b" && rng.chance(30) {
            continue;
        }
        let from = rng.pick(&sources[states.len()..]).clone();
        let sql = NEXT.replace("{s}", state).replace("{x}", &from);
        steps.push((state.to_string(), sql));
    }
    Generated { steps, states }
}

/// What the steps give on plain tables, run in dataflow order, with each
/// parameter as a literal. A state table always means `state`, its value
/// when the run began.
async fn reference(
    generated: &Generated,
    state: &BTreeMap<String, Vec<RecordBatch>>,
    input: &[RecordBatch],
    params: &BTreeMap<&str, i64>,
    partitions: usize,
) -> Result<BTreeMap<String, Vec<RecordBatch>>, String> {
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(partitions));
    for (name, rows) in state {
        ctx.register_table(name.as_str(), mem_table(rows, schema())?)
            .map_err(|e| e.to_string())?;
    }
    ctx.register_table("d", mem_table(input, schema())?)
        .map_err(|e| e.to_string())?;
    let mut results = BTreeMap::new();
    for (name, sql) in &generated.steps {
        let df = ctx
            .sql(&literal(sql, params))
            .await
            .map_err(|e| format!("reference step {name}: {e}"))?;
        let step_schema = Arc::clone(df.schema().inner());
        let batches = df
            .collect()
            .await
            .map_err(|e| format!("reference step {name}: {e}"))?;
        if !generated.states.contains(&name.as_str()) {
            ctx.register_table(name.as_str(), mem_table(&batches, step_schema)?)
                .map_err(|e| e.to_string())?;
        }
        results.insert(name.clone(), batches);
    }
    Ok(results)
}

async fn equivalence_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 11);
    let partitions = 1 + rng.below(4) as usize;
    let keys = 1 + rng.int(0, 3);
    let generated = generate(&mut rng);
    let mut listed = generated.steps.clone();
    rng.shuffle(&mut listed);

    let mut params: BTreeMap<&str, i64> = BTreeMap::new();
    params.insert("a", rng.int(-3, 3));
    params.insert("k", rng.int(-2, 3));
    let initial = params.clone();
    let mut state: BTreeMap<String, Vec<RecordBatch>> = generated
        .states
        .iter()
        .map(|s| (s.to_string(), state_rows(&mut rng, keys)))
        .collect();

    let describe = |extra: &str| {
        let steps: Vec<String> = listed
            .iter()
            .map(|(name, sql)| format!("{name}: {sql}"))
            .collect();
        format!(
            "equivalence seed={seed} partitions={partitions}\n  listed steps:\n    {}\n  {extra}",
            steps.join("\n    ")
        )
    };
    if verbose {
        eprintln!("{}", describe(""));
    }

    let ctx = accelerated(partitions);
    let mut program = Program::from_steps(&ctx, listed.clone()).input("d", schema());
    for (name, rows) in &state {
        program = program.state(name.as_str(), rows.clone());
    }
    for (name, value) in &params {
        program = program.param(*name, ScalarValue::from(*value));
    }
    for (name, _) in &generated.steps {
        program = program.output(name.as_str());
    }

    // If the reference can plan and run the steps, the program must lower.
    let first_input = rows(&mut rng, keys, 10);
    let lowered = match program.lower().await {
        Ok(lowered) => lowered,
        Err(e) => {
            let reference = reference(&generated, &state, &first_input, &params, partitions).await;
            return match reference {
                Ok(_) => Err(describe(&format!(
                    "lower failed, but the same SQL with literals runs: {e}"
                ))),
                Err(_) => Ok(()),
            };
        }
    };
    let compiled = lowered.compile().map_err(|e| describe(&e.to_string()))?;
    let handles: BTreeMap<String, TableHandle> = {
        let mut out = BTreeMap::new();
        for name in &generated.states {
            let handle = Accelerator::table(&ctx, name)
                .await
                .map_err(|e| e.to_string())?
                .ok_or_else(|| describe(&format!("{name} isn't in the session after lower")))?;
            out.insert(name.to_string(), handle);
        }
        out
    };

    let mut input = first_input;
    for run in 0..2 + rng.below(4) {
        // Sometimes replace a state table between runs: the next run reads
        // the new rows, and nothing conflicts.
        if run > 0 && rng.chance(25) {
            let name = rng.pick(&generated.states).to_string();
            let rows = state_rows(&mut rng, keys);
            handles[&name]
                .replace(rows.clone())
                .map_err(|e| e.to_string())?;
            state.insert(name, rows);
        }
        let mut args = RunArgs::new().input("d", input.clone());
        // A run not given a parameter uses its initial value.
        params = initial.clone();
        for name in ["a", "k"] {
            if rng.chance(50) {
                let value = rng.int(-3, 3);
                params.insert(name, value);
                args = args.param(name, ScalarValue::from(value));
            }
        }
        let versions: BTreeMap<String, u64> = handles
            .iter()
            .map(|(name, h)| (name.clone(), h.version().number()))
            .collect();
        let want = reference(&generated, &state, &input, &params, partitions).await;
        let got = compiled.run(args).await;
        let context = || format!("run {run}, params {params:?}");
        match (want, got) {
            (Ok(want), Ok(got)) => {
                for (name, rows) in &want {
                    let got_rows = got
                        .table(name)
                        .map_err(|e| describe(&format!("{}: {e}", context())))?;
                    if render(&got_rows) != render(rows) {
                        return Err(describe(&format!(
                            "{}: output {name}\n  expected {:?}\n  got {:?}",
                            context(),
                            render(rows),
                            render(&got_rows)
                        )));
                    }
                }
                for (name, handle) in &handles {
                    let next = want
                        .get(name)
                        .cloned()
                        .unwrap_or_else(|| state[name].clone());
                    if render(&handle.read()) != render(&next) {
                        return Err(describe(&format!(
                            "{}: published {name}\n  expected {:?}\n  got {:?}",
                            context(),
                            render(&next),
                            render(&handle.read())
                        )));
                    }
                    let added = u64::from(want.contains_key(name));
                    let now = handle.version().number();
                    if now != versions[name] + added {
                        return Err(describe(&format!(
                            "{}: {name} went from version {} to {now}, not by {added}",
                            context(),
                            versions[name]
                        )));
                    }
                    state.insert(name.clone(), next);
                }
            }
            (Err(_), Err(_)) => {
                for (name, handle) in &handles {
                    if handle.version().number() != versions[name]
                        || render(&handle.read()) != render(&state[name])
                    {
                        return Err(describe(&format!(
                            "{}: a failed run changed {name}",
                            context()
                        )));
                    }
                }
            }
            (Ok(_), Err(e)) => {
                return Err(describe(&format!(
                    "{}: the run failed, and the same SQL with literals runs: {e}",
                    context()
                )));
            }
            (Err(e), Ok(_)) => {
                return Err(describe(&format!(
                    "{}: the run succeeded, and the reference failed: {e}",
                    context()
                )));
            }
        }
        input = rows(&mut rng, keys, 10);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 1: versions and conflicts follow a sequential model.
// ---------------------------------------------------------------------------

/// What a run's step does to a state table, written so the model knows the
/// result: `v + 1`, and fails when `$t` isn't a number.
const BUMP: &str = "SELECT i, poke(v) + CAST($t AS BIGINT) AS v FROM {s}";

#[derive(Debug, Clone)]
struct ModelTable {
    version: u64,
    rows: Vec<String>,
    // The value of v, the same on every row.
    value: i64,
}

fn constant(keys: i64, value: i64) -> Vec<RecordBatch> {
    let all: Vec<(i64, Option<i64>)> = (0..keys).map(|i| (i, Some(value))).collect();
    vec![batch(&all)]
}

/// A replace to make during a run: a state table and its new rows.
type Poke = (TableHandle, Vec<RecordBatch>);

async fn versions_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 13);
    let keys = 1 + rng.int(0, 2);
    let ctx = accelerated(1 + rng.below(3) as usize);
    // Which tables to replace during the run, from inside the step; the
    // `poke` function does it once per run, on its first call.
    let pokes: Arc<Mutex<Vec<Poke>>> = Arc::default();
    {
        let pokes = Arc::clone(&pokes);
        ctx.register_udf(create_udf(
            "poke",
            vec![DataType::Int64],
            DataType::Int64,
            Volatility::Volatile,
            Arc::new(move |args: &[ColumnarValue]| {
                let due: Vec<_> = pokes.lock().expect("not poisoned").drain(..).collect();
                for (handle, rows) in due {
                    handle.replace(rows)?;
                }
                let ColumnarValue::Array(v) = &args[0] else {
                    return Ok(args[0].clone());
                };
                let v = v.as_any().downcast_ref::<Int64Array>().expect("Int64");
                let out: Int64Array = v.iter().map(|x| x.map(|x| x + 1)).collect();
                Ok(ColumnarValue::Array(Arc::new(out)))
            }),
        ));
    }
    let names: Vec<&str> = if rng.chance(50) {
        vec!["w", "b"]
    } else {
        vec!["w"]
    };
    // A session table no program writes, which every snapshot lists.
    let other = Accelerator::create_table(&ctx, "other", schema(), constant(1, 0))
        .map_err(|e| e.to_string())?;
    let mut model: BTreeMap<String, ModelTable> = BTreeMap::new();
    let mut program = Program::from_steps(
        &ctx,
        names
            .iter()
            .map(|s| (s.to_string(), BUMP.replace("{s}", s)))
            .collect::<Vec<_>>(),
    )
    .param("t", ScalarValue::from("0"));
    for name in &names {
        let value = rng.int(0, 5);
        let rows = constant(keys, value);
        model.insert(
            name.to_string(),
            ModelTable {
                version: 1,
                rows: render(&rows),
                value,
            },
        );
        program = program.state(*name, rows);
    }
    let compiled = program
        .lower()
        .await
        .and_then(|l| l.compile())
        .map_err(|e| format!("versions seed={seed}: lower: {e}"))?;
    let mut handles = BTreeMap::new();
    for name in &names {
        let handle = Accelerator::table(&ctx, name)
            .await
            .map_err(|e| e.to_string())?
            .expect("lower registers the state tables");
        handles.insert(name.to_string(), handle);
    }
    let identities: BTreeSet<_> = handles.values().map(|h| h.version().table()).collect();
    if identities.len() != handles.len() || identities.contains(&other.version().table()) {
        return Err(format!(
            "versions seed={seed}: two tables share an identity"
        ));
    }

    let mut log = Vec::new();
    for op in 0..4 + rng.below(8) {
        let fail = |log: &Vec<String>, m: String| {
            format!("versions seed={seed} op={op}\n  {m}\n  log: {log:?}")
        };
        match rng.below(10) {
            // A replace between runs.
            0 | 1 => {
                let name = rng.pick(&names).to_string();
                let value = rng.int(0, 5);
                let rows = constant(keys, value);
                handles[&name]
                    .replace(rows.clone())
                    .map_err(|e| e.to_string())?;
                let entry = model.get_mut(&name).expect("modelled");
                entry.version += 1;
                entry.rows = render(&rows);
                entry.value = value;
                log.push(format!("replace {name} = {value}"));
            }
            // A run, maybe with replaces during it, maybe with a failing step.
            _ => {
                let failing = rng.chance(20);
                let on_conflict = if rng.chance(50) {
                    OnConflict::Fail
                } else {
                    OnConflict::Overwrite
                };
                let mut poked = BTreeMap::new();
                for name in &names {
                    if rng.chance(30) {
                        let value = rng.int(0, 5);
                        poked.insert(name.to_string(), value);
                        pokes
                            .lock()
                            .expect("not poisoned")
                            .push((handles[*name].clone(), constant(keys, value)));
                    }
                }
                log.push(format!(
                    "run failing={failing} {on_conflict:?} poked={poked:?}"
                ));
                let begin: BTreeMap<String, ModelTable> = model.clone();
                let mut args = RunArgs::new().on_conflict(on_conflict);
                if failing {
                    args = args.param("t", ScalarValue::from("x"));
                }
                let got = compiled.run(args).await;
                pokes.lock().expect("not poisoned").clear();

                // The replaces during the run stand, whatever the run does.
                // A failing step may fail before the first poke, so a poke
                // counts only if it happened.
                let mut happened = BTreeMap::new();
                for (name, value) in &poked {
                    let handle = &handles[name];
                    if handle.version().number() > begin[name].version {
                        happened.insert(name.clone(), *value);
                        let entry = model.get_mut(name).expect("modelled");
                        entry.version += 1;
                        entry.value = *value;
                        entry.rows = render(&constant(keys, *value));
                    }
                }
                match (&got, failing) {
                    (Err(e), true) => {
                        if matches!(Error::find(e), Some(Error::Conflict { .. })) {
                            return Err(fail(
                                &log,
                                format!("a failing step became a conflict: {e}"),
                            ));
                        }
                    }
                    (Ok(_), true) => {
                        return Err(fail(&log, "a run whose step fails succeeded".into()));
                    }
                    (Err(e), false) => {
                        let conflict = !happened.is_empty() && on_conflict == OnConflict::Fail;
                        let expected = happened.keys().next().cloned();
                        match Error::find(e) {
                            Some(Error::Conflict {
                                table,
                                expected: read,
                                found,
                            }) if conflict => {
                                let name = table.as_str();
                                if !happened.contains_key(name)
                                    || *read != begin[name].version
                                    || *found != model[name].version
                                {
                                    return Err(fail(
                                        &log,
                                        format!(
                                            "Conflict {{ {table}, {read}, {found} }}, but the \
                                             model has {name} read at {} and now {} \
                                             (first poked: {expected:?})",
                                            begin.get(name).map_or(0, |t| t.version),
                                            model.get(name).map_or(0, |t| t.version)
                                        ),
                                    ));
                                }
                            }
                            _ => return Err(fail(&log, format!("the run failed: {e}"))),
                        }
                    }
                    (Ok(outputs), false) => {
                        if !happened.is_empty() && on_conflict == OnConflict::Fail {
                            return Err(fail(&log, "a conflict published".into()));
                        }
                        let overwritten: Vec<String> = happened.keys().cloned().collect();
                        let mut got_over = outputs.overwritten().to_vec();
                        got_over.sort();
                        if got_over != overwritten {
                            return Err(fail(
                                &log,
                                format!("overwritten {got_over:?}, expected {overwritten:?}"),
                            ));
                        }
                        // The snapshot is the versions at the start, of the
                        // session's tables, and nothing private.
                        let mut want_snapshot: Vec<(String, u64)> =
                            begin.iter().map(|(n, t)| (n.clone(), t.version)).collect();
                        want_snapshot.push(("other".into(), other.version().number()));
                        want_snapshot.sort();
                        let got_snapshot: Vec<(String, u64)> = outputs
                            .snapshot()
                            .tables()
                            .map(|(n, v)| (n.to_string(), v.number()))
                            .collect();
                        if got_snapshot != want_snapshot {
                            return Err(fail(
                                &log,
                                format!("snapshot {got_snapshot:?}, expected {want_snapshot:?}"),
                            ));
                        }
                        // Each state table's next value is its value when the
                        // run began, plus one, one version after the current.
                        for name in &names {
                            let entry = model.get_mut(*name).expect("modelled");
                            entry.version += 1;
                            entry.value = begin[*name].value + 1;
                            entry.rows = render(&constant(keys, entry.value));
                            let published = outputs.published().version(name);
                            if published.map(|v| v.number()) != Some(entry.version)
                                || published.map(|v| v.table())
                                    != Some(handles[*name].version().table())
                            {
                                return Err(fail(
                                    &log,
                                    format!(
                                        "published {name} at {published:?}, expected version {}",
                                        entry.version
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }
        for (name, entry) in &model {
            let handle = &handles[name];
            if handle.version().number() != entry.version || render(&handle.read()) != entry.rows {
                return Err(fail(
                    &log,
                    format!(
                        "{name} is at {} with {:?}; the model has {} with {:?}",
                        handle.version().number(),
                        render(&handle.read()),
                        entry.version,
                        entry.rows
                    ),
                ));
            }
        }
    }
    if verbose {
        eprintln!("versions seed={seed} log: {log:?}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Property 2: publication is atomic.
// ---------------------------------------------------------------------------

async fn atomic_case(seed: u64, verbose: bool) -> Outcome {
    let mut rng = Rng::new(seed, 17);
    let keys = 1 + rng.int(0, 3);
    let ctx = accelerated(1 + rng.below(4) as usize);
    let start = rng.int(0, 5);
    let compiled = Program::from_steps(
        &ctx,
        [
            ("w", "SELECT i, v + 1 AS v FROM w"),
            ("b", "SELECT i, v + 1 AS v FROM b"),
        ],
    )
    .state("w", constant(keys, start))
    .state("b", constant(keys, start))
    .lower()
    .await
    .and_then(|l| l.compile())
    .map_err(|e| e.to_string())?;
    let check = ctx
        .sql("SELECT COUNT(*) FROM w JOIN b ON w.i = b.i WHERE w.v <> b.v")
        .await
        .map_err(|e| e.to_string())?
        .create_physical_plan()
        .await
        .map_err(|e| e.to_string())?;

    let done = Arc::new(AtomicBool::new(false));
    let reader = {
        let ctx = ctx.clone();
        let done = Arc::clone(&done);
        tokio::spawn(async move {
            let mut reads = 0u64;
            while !done.load(Ordering::Relaxed) {
                let run = Accelerator::begin_run(&ctx).map_err(|e| e.to_string())?;
                let plan = datafusion::physical_plan::execution_plan::reset_plan_states(
                    Arc::clone(&check),
                )
                .map_err(|e| e.to_string())?;
                let batches = collect(plan, run.task_ctx())
                    .await
                    .map_err(|e| e.to_string())?;
                let mismatched = render(&batches);
                if mismatched != ["0"] {
                    return Err(format!(
                        "a run saw w and b from two publications: {mismatched:?} rows differ, \
                         snapshot {}",
                        run.snapshot()
                    ));
                }
                reads += 1;
                tokio::task::yield_now().await;
            }
            Ok::<u64, String>(reads)
        })
    };
    let runs = 20 + rng.below(30);
    for _ in 0..runs {
        compiled
            .run(RunArgs::new())
            .await
            .map_err(|e| format!("atomic seed={seed}: {e}"))?;
        tokio::task::yield_now().await;
    }
    done.store(true, Ordering::Relaxed);
    let reads = reader
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| format!("atomic seed={seed}: {e}"))?;
    if verbose {
        eprintln!("atomic seed={seed}: {runs} runs, {reads} concurrent reads");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Drivers, as in `simulation.rs`.
// ---------------------------------------------------------------------------

async fn case(property: usize, seed: u64, verbose: bool) -> Outcome {
    match property {
        0 => equivalence_case(seed, verbose).await,
        1 => versions_case(seed, verbose).await,
        _ => atomic_case(seed, verbose).await,
    }
}

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
async fn a_programs_runs_equal_its_selects_run_plainly() {
    bounded(0, 0..120).await;
}

#[tokio::test]
async fn versions_and_conflicts_follow_a_sequential_model() {
    bounded(1, 0..200).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn publication_is_atomic() {
    bounded(2, 0..20).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "replay: set DFA_SEED (and DFA_PROPERTY=0, 1 or 2)"]
async fn replay_one_seed() {
    let seed = env_u64("DFA_SEED", 0);
    let property = env_u64("DFA_PROPERTY", 0) as usize;
    if let Err(message) = case(property, seed, true).await {
        panic!("{message}");
    }
}
