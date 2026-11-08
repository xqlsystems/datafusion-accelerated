// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike 7: can handle batches carry device data between DataFusion nodes?
//!
//! A *handle batch* is a record batch whose one column holds handles to
//! buffers in a device's memory. Here the "device" is a mock: a pool of
//! record batches in host memory, each tagged with the partition (standing
//! for a device) that wrote it. Three nodes stand for the runtime's:
//!
//! - `ToDeviceExec` copies host batches into the pool and emits handles;
//! - `KernelExec` reads each handle's buffer, computes, writes a new buffer;
//! - `ToHostExec` reads each handle's buffer back as an ordinary batch.
//!
//! The spike lets DataFusion's planner and its default physical optimizer
//! rules build real queries around these nodes, then checks three things:
//! 1. which of DataFusion's operators end up between device nodes;
//! 2. whether any operator that isn't a device node receives a handle batch;
//! 3. whether a kernel reads a buffer another "device" wrote.
//!
//! and that every query's result matches the same query without the nodes.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Result, config::ConfigOptions};
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::datasource::MemTable;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, displayable,
};
use datafusion::prelude::*;
use futures::StreamExt;

// ---------------------------------------------------------------- the mock device

struct Buffer {
    batch: RecordBatch,
    device: usize,
}

static POOL: LazyLock<Mutex<HashMap<u64, Buffer>>> = LazyLock::new(Default::default);
static NEXT: AtomicU64 = AtomicU64::new(1);
/// A kernel read a buffer that another partition, standing for another device, wrote.
static CROSS_DEVICE: AtomicUsize = AtomicUsize::new(0);

fn put(batch: RecordBatch, device: usize) -> u64 {
    let h = NEXT.fetch_add(1, Ordering::Relaxed);
    POOL.lock().unwrap().insert(h, Buffer { batch, device });
    h
}

fn take(h: u64, device: usize) -> RecordBatch {
    let buf = POOL.lock().unwrap().remove(&h).expect("a live handle");
    if buf.device != device {
        CROSS_DEVICE.fetch_add(1, Ordering::Relaxed);
    }
    buf.batch
}

fn live_buffers() -> usize {
    POOL.lock().unwrap().len()
}

const HANDLE: &str = "__device_handle";

fn handle_schema() -> SchemaRef {
    let meta = HashMap::from([("accelerated.handle".to_string(), "true".to_string())]);
    Arc::new(Schema::new(vec![
        Field::new(HANDLE, DataType::UInt64, false).with_metadata(meta),
    ]))
}

fn is_handle_schema(schema: &SchemaRef) -> bool {
    schema.fields().iter().any(|f| f.name() == HANDLE)
}

fn handle_batch(handles: Vec<u64>) -> Result<RecordBatch> {
    let col: ArrayRef = Arc::new(UInt64Array::from(handles));
    Ok(RecordBatch::try_new(handle_schema(), vec![col])?)
}

// ---------------------------------------------------------------- the knobs

/// What a device node reports to DataFusion's optimizer. The experiments vary
/// these, to find which settings keep DataFusion's operators out of device
/// segments.
static BENEFITS_FROM_PARTITIONING: AtomicBool = AtomicBool::new(true);
static MAINTAINS_ORDER: AtomicBool = AtomicBool::new(false);

fn benefits() -> Vec<bool> {
    vec![BENEFITS_FROM_PARTITIONING.load(Ordering::Relaxed)]
}
fn maintains() -> Vec<bool> {
    vec![MAINTAINS_ORDER.load(Ordering::Relaxed)]
}

fn props(schema: SchemaRef, partitions: usize) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(schema),
        Partitioning::UnknownPartitioning(partitions),
        EmissionType::Incremental,
        Boundedness::Bounded,
    ))
}

// ---------------------------------------------------------------- the nodes

#[derive(Debug)]
struct ToDeviceExec {
    input: Arc<dyn ExecutionPlan>,
    cache: Arc<PlanProperties>,
}

impl ToDeviceExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let n = input.output_partitioning().partition_count();
        Self { input, cache: props(handle_schema(), n) }
    }
}

#[derive(Debug, Clone, Copy)]
enum Op {
    /// v * 2 + 1, applied by two kernels in a row (v * 2, then + 1)
    Times2,
    Plus1,
    /// leaves the batch as it is
    Identity,
}

#[derive(Debug)]
struct KernelExec {
    input: Arc<dyn ExecutionPlan>,
    op: Op,
    cache: Arc<PlanProperties>,
}

impl KernelExec {
    fn new(input: Arc<dyn ExecutionPlan>, op: Op) -> Self {
        let n = input.output_partitioning().partition_count();
        Self { input, op, cache: props(handle_schema(), n) }
    }
}

#[derive(Debug)]
struct ToHostExec {
    input: Arc<dyn ExecutionPlan>,
    logical: SchemaRef,
    projection: Option<Vec<usize>>,
    cache: Arc<PlanProperties>,
}

impl ToHostExec {
    fn new(input: Arc<dyn ExecutionPlan>, logical: SchemaRef, projection: Option<Vec<usize>>) -> Self {
        let out = match &projection {
            Some(p) => Arc::new(logical.project(p).unwrap()),
            None => logical.clone(),
        };
        let n = input.output_partitioning().partition_count();
        Self { input, logical, projection, cache: props(out, n) }
    }
}

macro_rules! display {
    ($t:ty, $name:expr) => {
        impl DisplayAs for $t {
            fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{}", $name(self))
            }
        }
    };
}
display!(ToDeviceExec, |_s: &ToDeviceExec| "ToDeviceExec".to_string());
display!(KernelExec, |s: &KernelExec| format!("KernelExec: {:?}", s.op));
display!(ToHostExec, |_s: &ToHostExec| "ToHostExec".to_string());

fn no_exprs(
    _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    Ok(TreeNodeRecursion::Continue)
}

impl ExecutionPlan for ToDeviceExec {
    fn name(&self) -> &str {
        "ToDeviceExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        benefits()
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        maintains()
    }
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        no_exprs(f)
    }
    fn with_new_children(self: Arc<Self>, mut c: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(c.swap_remove(0))))
    }
    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, ctx)?;
        let s = input.map(move |b| {
            let b = b?;
            handle_batch(vec![put(b, partition)])
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(handle_schema(), s)))
    }
}

fn apply(op: Op, b: RecordBatch) -> Result<RecordBatch> {
    if matches!(op, Op::Identity) {
        return Ok(b);
    }
    let idx = b.schema().index_of("v")?;
    let v = b.column(idx).as_any().downcast_ref::<Float64Array>().unwrap();
    let out: Float64Array = match op {
        Op::Times2 => v.iter().map(|x| x.map(|x| x * 2.0)).collect(),
        Op::Plus1 => v.iter().map(|x| x.map(|x| x + 1.0)).collect(),
        Op::Identity => v.clone(),
    };
    let mut cols = b.columns().to_vec();
    cols[idx] = Arc::new(out);
    Ok(RecordBatch::try_new(b.schema(), cols)?)
}

impl ExecutionPlan for KernelExec {
    fn name(&self) -> &str {
        "KernelExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        benefits()
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        maintains()
    }
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        no_exprs(f)
    }
    fn with_new_children(self: Arc<Self>, mut c: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(c.swap_remove(0), self.op)))
    }
    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, ctx)?;
        let op = self.op;
        let s = input.map(move |b| {
            let b = b?;
            let hs = b.column(0).as_any().downcast_ref::<UInt64Array>().expect("a handle batch");
            let mut out = Vec::with_capacity(hs.len());
            for h in hs.values() {
                let data = take(*h, partition);
                out.push(put(apply(op, data)?, partition));
            }
            handle_batch(out)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(handle_schema(), s)))
    }
}

impl ExecutionPlan for ToHostExec {
    fn name(&self) -> &str {
        "ToHostExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        benefits()
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        maintains()
    }
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        no_exprs(f)
    }
    fn with_new_children(self: Arc<Self>, mut c: Vec<Arc<dyn ExecutionPlan>>) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(c.swap_remove(0), self.logical.clone(), self.projection.clone())))
    }
    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, ctx)?;
        let projection = self.projection.clone();
        let schema = self.schema();
        let s = input.flat_map(move |b| {
            let rows: Vec<Result<RecordBatch>> = match b {
                Err(e) => vec![Err(e)],
                Ok(b) => {
                    let hs = b.column(0).as_any().downcast_ref::<UInt64Array>().expect("a handle batch");
                    hs.values()
                        .iter()
                        .map(|h| {
                            let data = take(*h, partition);
                            Ok(match &projection {
                                Some(p) => data.project(p)?,
                                None => data,
                            })
                        })
                        .collect()
                }
            };
            futures::stream::iter(rows)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, s)))
    }
}

// ---------------------------------------------------------------- a table on the device

/// A table whose scan is a device segment: its rows go to the device, two
/// kernels compute v * 2 + 1 there, and the result comes back to the host.
#[derive(Debug)]
struct DeviceTable {
    schema: SchemaRef,
    partitions: Vec<Vec<RecordBatch>>,
}

#[async_trait]
impl TableProvider for DeviceTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let scan = MemorySourceConfig::try_new_exec(&self.partitions, self.schema.clone(), None)?;
        let dev = Arc::new(ToDeviceExec::new(scan));
        let k1 = Arc::new(KernelExec::new(dev, Op::Times2));
        let k2 = Arc::new(KernelExec::new(k1, Op::Plus1));
        Ok(Arc::new(ToHostExec::new(k2, self.schema.clone(), projection.cloned())))
    }
}

fn data(partitions: usize) -> (SchemaRef, Vec<Vec<RecordBatch>>, Vec<Vec<RecordBatch>>) {
    use datafusion::arrow::array::Int64Array;
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Float64, false),
    ]));
    let mut raw = vec![];
    let mut reference = vec![];
    for p in 0..partitions {
        let mut part_raw = vec![];
        let mut part_ref = vec![];
        for b in 0..3 {
            let ks: Vec<i64> = (0..50).map(|i| ((p * 7 + b * 3 + i) % 11) as i64).collect();
            let vs: Vec<f64> = (0..50).map(|i| (p * 1000 + b * 100 + i) as f64 * 0.5).collect();
            let rs: Vec<f64> = vs.iter().map(|v| v * 2.0 + 1.0).collect();
            part_raw.push(RecordBatch::try_new(schema.clone(), vec![
                Arc::new(Int64Array::from(ks.clone())), Arc::new(Float64Array::from(vs))]).unwrap());
            part_ref.push(RecordBatch::try_new(schema.clone(), vec![
                Arc::new(Int64Array::from(ks)), Arc::new(Float64Array::from(rs))]).unwrap());
        }
        raw.push(part_raw);
        reference.push(part_ref);
    }
    (schema, raw, reference)
}

// ---------------------------------------------------------------- what the checks look for

#[derive(Default, Debug)]
struct Findings {
    /// DataFusion operators that pass handle batches along: their output is
    /// a handle batch, but they aren't device nodes.
    between: Vec<String>,
    /// DataFusion operators that would read a handle batch as data: their
    /// input is a handle batch, and their output isn't.
    exposed: Vec<String>,
}

fn is_device(plan: &Arc<dyn ExecutionPlan>) -> bool {
    matches!(plan.name(), "ToDeviceExec" | "KernelExec" | "ToHostExec")
}

fn inspect(plan: &Arc<dyn ExecutionPlan>, f: &mut Findings) {
    if !is_device(plan) {
        let one = displayable(plan.as_ref()).one_line().to_string().trim().to_string();
        if is_handle_schema(&plan.schema()) {
            if !f.between.contains(&one) {
                f.between.push(one);
            }
        } else if plan.children().iter().any(|c| is_handle_schema(&c.schema())) {
            f.exposed.push(plan.name().to_string());
        }
    }
    for child in plan.children() {
        inspect(child, f);
    }
}

// ---------------------------------------------------------------- a rule that places a segment mid-plan

/// Wraps the input of the first partial aggregate in a device segment that
/// leaves values unchanged, as the runtime's rule would place device nodes
/// in the middle of a host plan.
#[derive(Debug)]
struct PlaceSegment;

impl PhysicalOptimizerRule for PlaceSegment {
    fn optimize(&self, plan: Arc<dyn ExecutionPlan>, _c: &ConfigOptions) -> Result<Arc<dyn ExecutionPlan>> {
        let mut done = false;
        plan.transform_down(|node| {
            if done {
                return Ok(Transformed::no(node));
            }
            if let Some(agg) = node.downcast_ref::<AggregateExec>() {
                if *agg.mode() == AggregateMode::Partial {
                    let input = agg.input().clone();
                    let logical = input.schema();
                    let seg: Arc<dyn ExecutionPlan> = Arc::new(ToHostExec::new(
                        Arc::new(KernelExec::new(Arc::new(ToDeviceExec::new(input)), Op::Identity)),
                        logical,
                        None,
                    ));
                    done = true;
                    return Ok(Transformed::yes(node.with_new_children(vec![seg])?));
                }
            }
            Ok(Transformed::no(node))
        })
        .map(|t| t.data)
    }
    fn name(&self) -> &str {
        "PlaceSegment"
    }
    fn schema_check(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------- the experiments

const QUERIES: &[(&str, &str)] = &[
    ("group by", "SELECT k, SUM(v) AS s FROM {t} GROUP BY k ORDER BY k"),
    ("filter, sort, limit", "SELECT k, v FROM {t} WHERE v > 300 ORDER BY v DESC LIMIT 7"),
    ("projection", "SELECT v FROM {t} ORDER BY v LIMIT 5"),
    ("count", "SELECT COUNT(*) AS n FROM {t}"),
    ("self join", "SELECT a.k, SUM(a.v + b.v) AS s FROM {t} a JOIN {t} b ON a.k = b.k GROUP BY a.k ORDER BY a.k"),
    ("window", "SELECT k, v, SUM(v) OVER (PARTITION BY k ORDER BY v) AS w FROM {t} ORDER BY k, v LIMIT 9"),
    ("union", "SELECT k, v FROM {t} UNION ALL SELECT k, v FROM {t} ORDER BY k, v LIMIT 6"),
];

async fn session(target_partitions: usize, rules: Option<Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>>) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(target_partitions);
    let mut b = SessionStateBuilder::new().with_config(config).with_default_features();
    if let Some(r) = rules {
        b = b.with_physical_optimizer_rules(r);
    }
    SessionContext::new_with_state(b.build())
}

async fn run(ctx: &SessionContext, sql: &str) -> Result<(String, Arc<dyn ExecutionPlan>)> {
    let df = ctx.sql(sql).await?;
    let plan = df.clone().create_physical_plan().await?;
    let batches = datafusion::physical_plan::collect(plan.clone(), ctx.task_ctx()).await?;
    Ok((pretty_format_batches(&batches)?.to_string(), plan))
}

async fn leaf_experiment(verbose: bool) -> Result<()> {
    println!("## E1. A device segment at the leaf, under DataFusion's planner and default rules\n");
    println!("| target partitions | source partitions | benefits from partitioning | maintains order | query | result | between device nodes | handle batch reaches | cross-device reads | leaked buffers |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for tp in [1, 4] {
        for sp in [1, 3] {
            for benefit in [true, false] {
                for order in [false, true] {
                    BENEFITS_FROM_PARTITIONING.store(benefit, Ordering::Relaxed);
                    MAINTAINS_ORDER.store(order, Ordering::Relaxed);
                    let (schema, raw, reference) = data(sp);
                    let ctx = session(tp, None).await;
                    ctx.register_table("d", Arc::new(DeviceTable { schema: schema.clone(), partitions: raw }))?;
                    ctx.register_table("r", Arc::new(MemTable::try_new(schema, reference)?))?;
                    for (name, q) in QUERIES {
                        CROSS_DEVICE.store(0, Ordering::Relaxed);
                        let before = live_buffers();
                        let got = run(&ctx, &q.replace("{t}", "d")).await;
                        let want = run(&ctx, &q.replace("{t}", "r")).await?.0;
                        let (result, between, exposed) = match &got {
                            Ok((out, plan)) => {
                                let mut f = Findings::default();
                                inspect(plan, &mut f);
                                let r = if *out == want { "matches" } else { "**differs**" };
                                (r.to_string(), f.between.join(", "), f.exposed.join(", "))
                            }
                            Err(e) => (format!("**error**: {}", e.to_string().lines().next().unwrap_or("")), String::new(), String::new()),
                        };
                        let leaked = live_buffers() - before;
                        println!("| {tp} | {sp} | {benefit} | {order} | {name} | {result} | {between} | {exposed} | {} | {leaked} |",
                            CROSS_DEVICE.load(Ordering::Relaxed));
                        if verbose && tp == 4 && sp == 3 && benefit && !order && *name == "group by" {
                            if let Ok((_, plan)) = &got {
                                eprintln!("plan, default knobs:\n{}", displayable(plan.as_ref()).indent(true));
                            }
                        }
                        if verbose && tp == 4 && sp == 3 && !benefit && !order && *name == "group by" {
                            if let Ok((_, plan)) = &got {
                                eprintln!("plan, benefits_from_input_partitioning = false:\n{}", displayable(plan.as_ref()).indent(true));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

async fn mid_plan_experiment(verbose: bool) -> Result<()> {
    println!("\n## E2. A device segment placed mid-plan by a rule, first or last among DataFusion's rules\n");
    println!("| rule position | target partitions | benefits from partitioning | query | result | between device nodes | handle batch reaches | cross-device reads |");
    println!("|---|---|---|---|---|---|---|---|");
    let defaults = PhysicalOptimizer::new().rules;
    for position in ["first", "last"] {
        for tp in [1, 4] {
            for benefit in [true, false] {
                BENEFITS_FROM_PARTITIONING.store(benefit, Ordering::Relaxed);
                MAINTAINS_ORDER.store(false, Ordering::Relaxed);
                let mut rules = defaults.clone();
                let rule: Arc<dyn PhysicalOptimizerRule + Send + Sync> = Arc::new(PlaceSegment);
                if position == "first" {
                    rules.insert(0, rule);
                } else {
                    rules.push(rule);
                }
                let (schema, raw, _) = data(3);
                let plain = session(tp, None).await;
                plain.register_table("t", Arc::new(MemTable::try_new(schema.clone(), raw.clone())?))?;
                let placed = session(tp, Some(rules)).await;
                placed.register_table("t", Arc::new(MemTable::try_new(schema, raw)?))?;
                for (name, q) in QUERIES.iter().filter(|(n, _)| ["group by", "count", "self join"].contains(n)) {
                    CROSS_DEVICE.store(0, Ordering::Relaxed);
                    let sql = q.replace("{t}", "t");
                    let want = run(&plain, &sql).await?.0;
                    let got = run(&placed, &sql).await;
                    let (result, between, exposed) = match &got {
                        Ok((out, plan)) => {
                            let mut f = Findings::default();
                            inspect(plan, &mut f);
                            let r = if *out == want { "matches" } else { "**differs**" };
                            (r.to_string(), f.between.join(", "), f.exposed.join(", "))
                        }
                        Err(e) => (format!("**error**: {}", e.to_string().lines().next().unwrap_or("")), String::new(), String::new()),
                    };
                    println!("| {position} | {tp} | {benefit} | {name} | {result} | {between} | {exposed} | {} |",
                        CROSS_DEVICE.load(Ordering::Relaxed));
                    if verbose && position == "first" && tp == 4 && benefit && *name == "group by" {
                        if let Ok((_, plan)) = &got {
                            eprintln!("plan, rule first:\n{}", displayable(plan.as_ref()).indent(true));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let verbose = std::env::args().any(|a| a == "--plans");
    leaf_experiment(verbose).await?;
    mid_plan_experiment(verbose).await?;
    Ok(())
}
