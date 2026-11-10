// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike 4a: can the runtime's Rust core plan a step once against a
//! datafusion-python session, then run it again on new rows, with no
//! replanning and no fork of datafusion-python?
//!
//! The pieces, each a miniature of what M0 builds:
//! - [`Accelerator`], an extension object for `SessionContext.with_extensions`.
//!   Its planner wraps the session's own, and keeps a copy of every physical
//!   plan the session makes, here in Rust.
//! - [`HandleTable`], a table whose scan reads the handle's current batches
//!   when it executes, not when it is planned (design D16).
//! - [`AcceleratedCodec`], the runtime's codec, which carries a handle scan
//!   across datafusion-ffi by the handle's id.
//! - [`AcceleratedOptions`], the `accelerated` settings namespace.

use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use std::time::Instant;

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatchReader;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use datafusion::arrow::record_batch::{RecordBatch, RecordBatchIterator};
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::config::{ConfigExtension, ConfigOptions};
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{
    DataFusionError, Result, Statistics, TableReference, exec_err, extensions_options,
};
use datafusion::datasource::TableType;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, LogicalPlan};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, reset_plan_states};
use datafusion::physical_plan::memory::MemoryStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties, collect, displayable,
};
use datafusion_ffi::config::extension_options::FFI_ExtensionOptions;
use datafusion_ffi::physical_optimizer::FFI_PhysicalOptimizerRule;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use datafusion_ffi::query_planner::FFI_QueryPlanner;
use datafusion_ffi::table_provider::FFI_TableProvider;
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use datafusion_proto::physical_plan::{PhysicalExtensionCodec, PhysicalProtoConverterExtension};
use datafusion_python_util as util;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

fn py_err(e: impl fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Settings: the `accelerated` namespace.
// ---------------------------------------------------------------------------

extensions_options! {
    /// The runtime's settings, set in SQL with `SET accelerated.<key> = ...`.
    pub struct AcceleratedOptions {
        /// Declines are errors.
        pub strict: bool, default = false
        /// A free value, so the spike can see a change arrive.
        pub label: String, default = "unset".to_string()
    }
}

impl ConfigExtension for AcceleratedOptions {
    const PREFIX: &'static str = "accelerated";
}

/// The settings as a datafusion-python `SessionConfig` extension.
#[pyclass(name = "AcceleratedConfig", module = "s04a", skip_from_py_object)]
#[derive(Clone, Default)]
struct PyAcceleratedConfig;

#[pymethods]
impl PyAcceleratedConfig {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __datafusion_extension_options__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let mut options = FFI_ExtensionOptions::default();
        options
            .add_config(&AcceleratedOptions::default())
            .map_err(py_err)?;
        PyCapsule::new_with_value(py, options, c"datafusion_extension_options")
    }
}

/// What the runtime can read of its settings from a session, as text.
fn settings_seen(session: &dyn Session) -> String {
    settings_in(session.config().options())
}

fn settings_in(options: &ConfigOptions) -> String {
    let extensions = &options.extensions;
    if let Some(native) = extensions.get::<AcceleratedOptions>() {
        return format!("native: strict={} label={}", native.strict, native.label);
    }
    match extensions.get::<FFI_ExtensionOptions>() {
        None => "none".to_string(),
        Some(ffi) => match ffi.to_extension::<AcceleratedOptions>() {
            Ok(o) => format!("ffi: strict={} label={}", o.strict, o.label),
            Err(e) => format!("ffi, unreadable: {e}"),
        },
    }
}

// ---------------------------------------------------------------------------
// Handles: tables the runtime owns, read at execution.
// ---------------------------------------------------------------------------

/// A table the runtime owns. Its batches can be replaced; a plan that scans
/// it reads whatever it holds when the plan executes.
#[derive(Debug)]
struct Handle {
    id: u64,
    schema: SchemaRef,
    batches: RwLock<Vec<RecordBatch>>,
}

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);
static HANDLES: LazyLock<Mutex<HashMap<u64, Arc<Handle>>>> = LazyLock::new(Default::default);

fn handle(id: u64) -> Result<Arc<Handle>> {
    HANDLES
        .lock()
        .unwrap()
        .get(&id)
        .cloned()
        .ok_or_else(|| DataFusionError::Execution(format!("no handle {id}")))
}

#[derive(Debug)]
struct HandleProvider(Arc<Handle>);

/// A handle's statistics, as of now. Always inexact: a compiled plan outlives
/// the rows it was planned on, and DataFusion answers some queries, such as
/// `COUNT(*)`, from exact statistics without reading a row.
fn handle_statistics(handle: &Handle, schema: &SchemaRef) -> Statistics {
    let batches = handle.batches.read().unwrap();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    let mut stats = Statistics::new_unknown(schema);
    if EXACT.load(Ordering::SeqCst) {
        stats.num_rows = Precision::Exact(rows);
        stats.total_byte_size = Precision::Exact(bytes);
    } else {
        stats.num_rows = Precision::Inexact(rows);
        stats.total_byte_size = Precision::Inexact(bytes);
    }
    stats
}

#[async_trait]
impl TableProvider for HandleProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.0.schema)
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    fn statistics(&self) -> Option<Statistics> {
        STATISTICS
            .load(Ordering::SeqCst)
            .then(|| handle_statistics(&self.0, &self.0.schema))
    }
    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HandleScanExec::new(
            Arc::clone(&self.0),
            projection.cloned(),
        )?))
    }
}

/// The scan of a handle. Planning records the handle, not its rows.
#[derive(Debug)]
struct HandleScanExec {
    handle: Arc<Handle>,
    projection: Option<Vec<usize>>,
    properties: Arc<PlanProperties>,
}

/// Whether handles report statistics; the spike measures both.
static STATISTICS: AtomicBool = AtomicBool::new(true);
/// The control: exact statistics, which the spike shows to be wrong here.
static EXACT: AtomicBool = AtomicBool::new(false);
/// Whether the codec carries tables and nodes it can't serialize by
/// reference, within the process.
static BY_REFERENCE: AtomicBool = AtomicBool::new(false);

/// How many times a handle scan has executed, across every plan.
static SCANS: AtomicUsize = AtomicUsize::new(0);

impl HandleScanExec {
    fn new(handle: Arc<Handle>, projection: Option<Vec<usize>>) -> Result<Self> {
        let schema = match &projection {
            Some(p) => Arc::new(handle.schema.project(p)?),
            None => Arc::clone(&handle.schema),
        };
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            handle,
            projection,
            properties,
        })
    }
}

impl DisplayAs for HandleScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "HandleScanExec: handle={}", self.handle.id)
    }
}

impl ExecutionPlan for HandleScanExec {
    fn name(&self) -> &str {
        "HandleScanExec"
    }
    fn apply_expressions(
        &self,
        _: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn partition_statistics(&self, _partition: Option<usize>) -> Result<Arc<Statistics>> {
        let schema = self.schema();
        Ok(Arc::new(if STATISTICS.load(Ordering::SeqCst) {
            handle_statistics(&self.handle, &schema)
        } else {
            Statistics::new_unknown(&schema)
        }))
    }
    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }
    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return exec_err!("HandleScanExec has one partition, not {partition}");
        }
        SCANS.fetch_add(1, Ordering::SeqCst);
        let batches = self.handle.batches.read().unwrap().clone();
        let batches = match &self.projection {
            Some(p) => batches
                .iter()
                .map(|b| b.project(p))
                .collect::<Result<_, _>>()?,
            None => batches,
        };
        Ok(Box::pin(MemoryStream::try_new(
            batches,
            self.schema(),
            None,
        )?))
    }
}

fn batches_from_stream(obj: &Bound<PyAny>) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    let capsule = obj.call_method0("__arrow_c_stream__")?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let ptr: NonNull<FFI_ArrowArrayStream> =
        capsule.pointer_checked(Some(c"arrow_array_stream"))?.cast();
    // Moves the stream out of the capsule, leaving a released one behind.
    let reader = unsafe { ArrowArrayStreamReader::from_raw(ptr.as_ptr()) }.map_err(py_err)?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(py_err)?;
    Ok((schema, batches))
}

/// A table handle, from Python.
#[pyclass(name = "HandleTable", module = "s04a")]
struct PyHandleTable(Arc<Handle>);

#[pymethods]
impl PyHandleTable {
    /// A handle holding the rows of `data`, any object with `__arrow_c_stream__`.
    #[new]
    fn new(data: &Bound<PyAny>) -> PyResult<Self> {
        let (schema, batches) = batches_from_stream(data)?;
        let id = NEXT_HANDLE.fetch_add(1, Ordering::SeqCst);
        let handle = Arc::new(Handle {
            id,
            schema,
            batches: RwLock::new(batches),
        });
        HANDLES.lock().unwrap().insert(id, Arc::clone(&handle));
        Ok(Self(handle))
    }

    /// Replace the handle's rows. The schema must stay the same.
    fn replace(&self, data: &Bound<PyAny>) -> PyResult<()> {
        let (schema, batches) = batches_from_stream(data)?;
        if schema != self.0.schema {
            return Err(py_err("replace: the schema changed"));
        }
        *self.0.batches.write().unwrap() = batches;
        Ok(())
    }

    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = util::ffi_logical_codec_from_pycapsule(session, None)?;
        let provider = FFI_TableProvider::new_with_ffi_codec(
            Arc::new(HandleProvider(Arc::clone(&self.0))),
            false,
            Some(util::get_tokio_runtime().handle().clone()),
            codec,
        );
        PyCapsule::new_with_value(py, provider, c"datafusion_table_provider")
    }
}

// ---------------------------------------------------------------------------
// The runtime's codec: a handle crosses by its id.
// ---------------------------------------------------------------------------

/// What the codec saw, so the spike can say which paths it took.
static CODEC_EVENTS: LazyLock<Mutex<HashMap<&'static str, usize>>> =
    LazyLock::new(Default::default);

/// Tables and nodes the codec carries by reference: an id into these
/// registries, valid only within this process.
static TABLE_REFS: LazyLock<Mutex<Vec<Arc<dyn TableProvider>>>> = LazyLock::new(Default::default);
static NODE_REFS: LazyLock<Mutex<Vec<Arc<dyn ExecutionPlan>>>> = LazyLock::new(Default::default);

fn by_reference<T: ?Sized>(
    registry: &Mutex<Vec<Arc<T>>>,
    item: Arc<T>,
    buf: &mut Vec<u8>,
    tag: &[u8],
) {
    let mut items = registry.lock().unwrap();
    buf.extend_from_slice(tag);
    buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
    items.push(item);
}

fn from_reference<T: ?Sized>(registry: &Mutex<Vec<Arc<T>>>, rest: &[u8]) -> Result<Arc<T>> {
    let id = u64::from_le_bytes(
        rest.try_into()
            .map_err(|_| DataFusionError::Execution("bad reference".to_string()))?,
    );
    registry
        .lock()
        .unwrap()
        .get(id as usize)
        .cloned()
        .ok_or_else(|| DataFusionError::Execution(format!("no reference {id}")))
}

fn event(name: &'static str) {
    *CODEC_EVENTS.lock().unwrap().entry(name).or_default() += 1;
}

/// The runtime's codec. It knows handle scans and handle tables, and passes
/// everything else to `inner`, the session's own codec, when there is one.
#[derive(Debug)]
struct AcceleratedCodec {
    logical_inner: Option<Arc<dyn LogicalExtensionCodec>>,
    physical_inner: Option<Arc<dyn PhysicalExtensionCodec>>,
}

fn encode_scan(scan: &HandleScanExec, buf: &mut Vec<u8>) {
    buf.extend_from_slice(&scan.handle.id.to_le_bytes());
    if let Some(p) = &scan.projection {
        for i in p {
            buf.extend_from_slice(&(*i as u64).to_le_bytes());
        }
    }
}

fn decode_scan(buf: &[u8]) -> Result<HandleScanExec> {
    let words: Vec<u64> = buf
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let Some((&id, projection)) = words.split_first() else {
        return exec_err!("empty handle scan");
    };
    let projection =
        (!projection.is_empty()).then(|| projection.iter().map(|&i| i as usize).collect());
    HandleScanExec::new(handle(id)?, projection)
}

impl PhysicalExtensionCodec for AcceleratedCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(rest) = buf.strip_prefix(b"dfacc:scan:") {
            event("physical decode: handle scan");
            return Ok(Arc::new(decode_scan(rest)?));
        }
        if let Some(rest) = buf.strip_prefix(b"dfacc:noderef:") {
            event("physical decode: node by reference");
            return from_reference(&NODE_REFS, rest);
        }
        match &self.physical_inner {
            Some(inner) => inner.try_decode(buf, inputs, ctx, proto_converter),
            None => exec_err!("not the runtime's node"),
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        if let Some(scan) = node.downcast_ref::<HandleScanExec>() {
            event("physical encode: handle scan");
            buf.extend_from_slice(b"dfacc:scan:");
            encode_scan(scan, buf);
            return Ok(());
        }
        event("physical encode: other node");
        match &self.physical_inner {
            Some(inner) => inner.try_encode(node, buf, proto_converter),
            None if BY_REFERENCE.load(Ordering::SeqCst) => {
                event("physical encode: node by reference");
                by_reference(&NODE_REFS, node, buf, b"dfacc:noderef:");
                Ok(())
            }
            None => exec_err!("not the runtime's node: {}", node.name()),
        }
    }
}

impl LogicalExtensionCodec for AcceleratedCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[LogicalPlan],
        ctx: &TaskContext,
    ) -> Result<datafusion::logical_expr::Extension> {
        match &self.logical_inner {
            Some(inner) => inner.try_decode(buf, inputs, ctx),
            None => exec_err!("the runtime has no logical nodes"),
        }
    }

    fn try_encode(
        &self,
        node: &datafusion::logical_expr::Extension,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        match &self.logical_inner {
            Some(inner) => inner.try_encode(node, buf),
            None => exec_err!("the runtime has no logical nodes"),
        }
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        table_ref: &TableReference,
        schema: SchemaRef,
        ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        if let Some(rest) = buf.strip_prefix(b"dfacc:table:") {
            event("logical decode: handle table");
            let id = u64::from_le_bytes(
                rest.try_into()
                    .map_err(|_| DataFusionError::Execution("bad handle table".to_string()))?,
            );
            return Ok(Arc::new(HandleProvider(handle(id)?)));
        }
        if let Some(rest) = buf.strip_prefix(b"dfacc:tableref:") {
            event("logical decode: table by reference");
            return from_reference(&TABLE_REFS, rest);
        }
        event("logical decode: other table");
        match &self.logical_inner {
            Some(inner) => inner.try_decode_table_provider(buf, table_ref, schema, ctx),
            None => exec_err!("not the runtime's table: {table_ref}"),
        }
    }

    fn try_encode_table_provider(
        &self,
        table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        if let Some(p) = node.downcast_ref::<HandleProvider>() {
            event("logical encode: handle table");
            buf.extend_from_slice(b"dfacc:table:");
            buf.extend_from_slice(&p.0.id.to_le_bytes());
            return Ok(());
        }
        event("logical encode: other table");
        match &self.logical_inner {
            Some(inner) => inner.try_encode_table_provider(table_ref, node, buf),
            None if BY_REFERENCE.load(Ordering::SeqCst) => {
                event("logical encode: table by reference");
                by_reference(&TABLE_REFS, node, buf, b"dfacc:tableref:");
                Ok(())
            }
            None => exec_err!("not the runtime's table: {table_ref}"),
        }
    }
}

/// A codec, as the object `SessionExtensionComponents` wants.
#[pyclass(name = "Codec", module = "s04a")]
struct PyCodec {
    logical: Option<FFI_LogicalExtensionCodec>,
    physical: Option<FFI_PhysicalExtensionCodec>,
}

#[pymethods]
impl PyCodec {
    #[classattr]
    fn __datafusion_codec_id__() -> &'static str {
        "datafusion_accelerated.spike4a"
    }

    #[pyo3(signature = (session=None))]
    fn __datafusion_logical_extension_codec__<'py>(
        &self,
        py: Python<'py>,
        session: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = session;
        let codec = self
            .logical
            .clone()
            .ok_or_else(|| py_err("not a logical codec"))?;
        PyCapsule::new_with_value(py, codec, c"datafusion_logical_extension_codec")
    }

    #[pyo3(signature = (session=None))]
    fn __datafusion_physical_extension_codec__<'py>(
        &self,
        py: Python<'py>,
        session: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = session;
        let codec = self
            .physical
            .clone()
            .ok_or_else(|| py_err("not a physical codec"))?;
        PyCapsule::new_with_value(py, codec, c"datafusion_physical_extension_codec")
    }
}

// ---------------------------------------------------------------------------
// The planner: wraps the session's, and keeps a copy of each plan.
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct State {
    planner_calls: AtomicUsize,
    last_plan: Mutex<Option<Arc<dyn ExecutionPlan>>>,
    settings: Mutex<Vec<String>>,
    compiled: Mutex<Vec<Arc<dyn ExecutionPlan>>>,
}

#[derive(Debug)]
struct RuntimePlanner {
    fallback: Arc<dyn QueryPlanner + Send + Sync>,
    state: Arc<State>,
}

#[async_trait]
impl QueryPlanner for RuntimePlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.state.planner_calls.fetch_add(1, Ordering::SeqCst);
        self.state
            .settings
            .lock()
            .unwrap()
            .push(settings_seen(session));
        let plan = self
            .fallback
            .create_physical_plan(logical_plan, session)
            .await?;
        *self.state.last_plan.lock().unwrap() = Some(Arc::clone(&plan));
        Ok(plan)
    }
}

/// The other attachment point: a physical optimizer rule, added after
/// DataFusion's own. It sees the finished plan, keeps it, and changes nothing.
#[derive(Debug)]
struct RecordingRule {
    state: Arc<State>,
}

impl PhysicalOptimizerRule for RecordingRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.state.planner_calls.fetch_add(1, Ordering::SeqCst);
        self.state
            .settings
            .lock()
            .unwrap()
            .push(settings_in(config));
        *self.state.last_plan.lock().unwrap() = Some(Arc::clone(&plan));
        Ok(plan)
    }

    fn name(&self) -> &str {
        "datafusion_accelerated_spike4a"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// The runtime's extension object, for `SessionContext.with_extensions`.
#[pyclass(name = "Accelerator", module = "s04a")]
#[derive(Default)]
struct Accelerator {
    state: Arc<State>,
}

#[pymethods]
impl Accelerator {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Phase one: the runtime's codecs, so the session can carry its nodes.
    fn __datafusion_session_components__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = util::ffi_task_context_provider_from_pycapsule(&ctx)?;
        let runtime = Some(util::get_tokio_runtime().handle().clone());
        let codec = Arc::new(AcceleratedCodec {
            logical_inner: None,
            physical_inner: None,
        });
        let logical = PyCodec {
            logical: Some(FFI_LogicalExtensionCodec::new(
                codec.clone(),
                runtime.clone(),
                provider.clone(),
            )),
            physical: None,
        };
        let physical = PyCodec {
            logical: None,
            physical: Some(FFI_PhysicalExtensionCodec::new(codec, runtime, provider)),
        };
        let kwargs = PyDict::new(py);
        kwargs.set_item("logical_extension_codecs", (Py::new(py, logical)?,))?;
        kwargs.set_item("physical_extension_codecs", (Py::new(py, physical)?,))?;
        py.import("datafusion")?
            .getattr("SessionExtensionComponents")?
            .call((), Some(&kwargs))
    }

    /// Phase two: the runtime's planner, wrapping `fallback`.
    fn __datafusion_session_planner__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
        fallback: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let fallback = util::ffi_query_planner_from_pycapsule(&fallback, Some(&ctx))?;
        let fallback: Arc<dyn QueryPlanner + Send + Sync> = (&fallback).into();
        let provider = util::ffi_task_context_provider_from_pycapsule(&ctx)?;
        let session_logical: Arc<dyn LogicalExtensionCodec> =
            (&util::ffi_logical_codec_from_pycapsule(ctx.clone(), None)?).into();
        let session_physical: Arc<dyn PhysicalExtensionCodec> =
            (&util::ffi_physical_codec_from_pycapsule(ctx.clone(), None)?).into();
        let codec = Arc::new(AcceleratedCodec {
            logical_inner: Some(session_logical),
            physical_inner: Some(session_physical),
        });
        let planner = Arc::new(RuntimePlanner {
            fallback,
            state: Arc::clone(&self.state),
        });
        let planner = FFI_QueryPlanner::new(
            planner,
            Some(util::get_tokio_runtime().handle().clone()),
            provider,
            codec.clone(),
            codec,
        );
        util::create_query_planner_capsule(py, &planner)
    }

    /// The rule, for `SessionContext.add_physical_optimizer_rule`.
    fn __datafusion_physical_optimizer_rule__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let rule = FFI_PhysicalOptimizerRule::new(
            Arc::new(RecordingRule {
                state: Arc::clone(&self.state),
            }),
            Some(util::get_tokio_runtime().handle().clone()),
        );
        PyCapsule::new_with_value(py, rule, c"datafusion_physical_optimizer_rule")
    }

    /// Plan `sql` once, through the session, and keep the runtime's copy.
    /// Returns the compiled plan's index.
    fn compile(&self, ctx: &Bound<PyAny>, sql: &str) -> PyResult<usize> {
        self.state.last_plan.lock().unwrap().take();
        ctx.call_method1("sql", (sql,))?
            .call_method0("execution_plan")?;
        let plan = self
            .state
            .last_plan
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| py_err("the session didn't plan through the runtime"))?;
        let mut compiled = self.state.compiled.lock().unwrap();
        compiled.push(plan);
        Ok(compiled.len() - 1)
    }

    /// The runtime's copy of a compiled plan, as text.
    fn explain(&self, index: usize) -> PyResult<String> {
        let plan = self.plan(index)?;
        Ok(displayable(plan.as_ref()).indent(true).to_string())
    }

    /// Run a compiled plan again, with a task context taken from the session
    /// now, and return its rows and the time spent in Rust, in seconds.
    fn run<'py>(
        &self,
        py: Python<'py>,
        ctx: &Bound<'py, PyAny>,
        index: usize,
    ) -> PyResult<(ArrowStream, f64)> {
        let plan = self.plan(index)?;
        let provider = util::ffi_task_context_provider_from_pycapsule(ctx)?;
        let started = Instant::now();
        let task_ctx: Arc<TaskContext> = (&provider).try_into().map_err(py_err)?;
        let plan = reset_plan_states(plan).map_err(py_err)?;
        let batches = py
            .detach(|| util::get_tokio_runtime().block_on(collect(plan.clone(), task_ctx)))
            .map_err(py_err)?;
        let elapsed = started.elapsed().as_secs_f64();
        Ok((
            ArrowStream {
                schema: plan.schema(),
                batches: Some(batches),
            },
            elapsed,
        ))
    }

    /// Time each part of a run, in milliseconds: the task context, the reset,
    /// and execution.
    fn run_parts(
        &self,
        py: Python<'_>,
        ctx: &Bound<'_, PyAny>,
        index: usize,
    ) -> PyResult<(f64, f64, f64)> {
        let plan = self.plan(index)?;
        let provider = util::ffi_task_context_provider_from_pycapsule(ctx)?;
        let t0 = Instant::now();
        let task_ctx: Arc<TaskContext> = (&provider).try_into().map_err(py_err)?;
        let t1 = Instant::now();
        let plan = reset_plan_states(plan).map_err(py_err)?;
        let t2 = Instant::now();
        py.detach(|| util::get_tokio_runtime().block_on(collect(plan, task_ctx)))
            .map_err(py_err)?;
        let t3 = Instant::now();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        Ok((ms(t0, t1), ms(t1, t2), ms(t2, t3)))
    }

    /// The functions in a compiled plan's expressions, by whether they are
    /// this library's own or proxies across datafusion-ffi.
    fn functions(&self, index: usize) -> PyResult<HashMap<String, usize>> {
        use datafusion::common::tree_node::TreeNode;
        use datafusion::physical_expr::ScalarFunctionExpr;
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut stack = vec![self.plan(index)?];
        while let Some(node) = stack.pop() {
            node.apply_expressions(&mut |e| {
                e.apply(|e| {
                    if let Some(f) = e.downcast_ref::<ScalarFunctionExpr>() {
                        let kind = if format!("{:?}", f.fun().inner()).contains("Foreign") {
                            "foreign"
                        } else {
                            "local"
                        };
                        *seen.entry(format!("{kind}: {}", f.name())).or_default() += 1;
                    }
                    Ok(TreeNodeRecursion::Continue)
                })
            })
            .map_err(py_err)?;
            stack.extend(node.children().into_iter().cloned());
        }
        Ok(seen)
    }

    /// The control: the same SQL, compiled once in a session of this
    /// library's own, with no FFI, over the same handles. Returns the median
    /// milliseconds of planning, and of a compiled run (reset and execute).
    #[staticmethod]
    fn local_baseline(
        py: Python<'_>,
        sql: &str,
        tables: HashMap<String, PyRef<PyHandleTable>>,
        runs: usize,
    ) -> PyResult<(f64, f64)> {
        use datafusion::prelude::SessionContext;
        let ctx = SessionContext::new();
        for (name, t) in &tables {
            ctx.register_table(name.as_str(), Arc::new(HandleProvider(Arc::clone(&t.0))))
                .map_err(py_err)?;
        }
        let sql = sql.to_string();
        py.detach(move || {
            util::get_tokio_runtime().block_on(async move {
                let median = |mut v: Vec<f64>| {
                    v.sort_by(f64::total_cmp);
                    v[v.len() / 2]
                };
                let mut planning = Vec::new();
                let mut plan = None;
                for _ in 0..runs {
                    let t = Instant::now();
                    let p = ctx.sql(&sql).await?.create_physical_plan().await?;
                    planning.push(t.elapsed().as_secs_f64() * 1e3);
                    plan = Some(p);
                }
                let plan = plan.unwrap();
                let mut running = Vec::new();
                for _ in 0..runs {
                    let t = Instant::now();
                    let p = reset_plan_states(Arc::clone(&plan))?;
                    collect(p, ctx.task_ctx()).await?;
                    running.push(t.elapsed().as_secs_f64() * 1e3);
                }
                Ok::<_, DataFusionError>((median(planning), median(running)))
            })
        })
        .map_err(py_err)
    }

    fn planner_calls(&self) -> usize {
        self.state.planner_calls.load(Ordering::SeqCst)
    }

    fn settings_seen(&self) -> Vec<String> {
        self.state.settings.lock().unwrap().clone()
    }

    #[staticmethod]
    #[pyo3(signature = (on, exact=false))]
    fn set_statistics(on: bool, exact: bool) {
        STATISTICS.store(on, Ordering::SeqCst);
        EXACT.store(exact, Ordering::SeqCst);
    }

    #[staticmethod]
    fn set_by_reference(on: bool) {
        BY_REFERENCE.store(on, Ordering::SeqCst);
    }

    #[staticmethod]
    fn scans() -> usize {
        SCANS.load(Ordering::SeqCst)
    }

    #[staticmethod]
    fn codec_events() -> HashMap<String, usize> {
        CODEC_EVENTS
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect()
    }
}

impl Accelerator {
    fn plan(&self, index: usize) -> PyResult<Arc<dyn ExecutionPlan>> {
        self.state
            .compiled
            .lock()
            .unwrap()
            .get(index)
            .cloned()
            .ok_or_else(|| py_err(format!("no compiled plan {index}")))
    }
}

/// Rows back to Python, through the Arrow PyCapsule interface.
#[pyclass(name = "ArrowStream", module = "s04a")]
struct ArrowStream {
    schema: SchemaRef,
    batches: Option<Vec<RecordBatch>>,
}

#[pymethods]
impl ArrowStream {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &mut self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        let batches = self.batches.take().ok_or_else(|| py_err("already read"))?;
        let reader =
            RecordBatchIterator::new(batches.into_iter().map(Ok), Arc::clone(&self.schema));
        let stream = FFI_ArrowArrayStream::new(Box::new(reader));
        PyCapsule::new_with_value(py, stream, c"arrow_array_stream")
    }
}

#[pymodule]
fn s04a(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Accelerator>()?;
    m.add_class::<PyHandleTable>()?;
    m.add_class::<PyAcceleratedConfig>()?;
    m.add_class::<PyCodec>()?;
    m.add_class::<ArrowStream>()?;
    Ok(())
}
