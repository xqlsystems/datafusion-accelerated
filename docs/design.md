# The XQL runtime: design

_author_: Alex Merose
_co-authors_: 🔥 Claude (torch-sql) and 🌀 Claude (xarray-sql), via Claude Code
_reviewers_: 🪭 Claude (einfold) and ∂ Claude (ddx). Their reviews of v1.1 are in [reviews/](reviews/).
_status_: Draft v1.5, for approval. Nothing is built. Collapsed sections (▸) hold background and detail.
_last updated_: 2026-11-07
_supplement_: [supplement.md](supplement.md), with the evidence, the alternatives considered and the reasoning behind each rule. Citations such as "(S1.1)" point there. S6 compares the design with JAX, PyTorch and DataFusion.
_origin_: [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)

---

## 1. What we're building

**XQL** is a group of open-source projects, under the
[xqlsystems](https://github.com/xqlsystems) organization, for computing on
scientific arrays with databases. An N-dimensional array becomes a table in
*long format*: one row per element, with one column per dimension holding
the element's position and one column holding its value. Array programs
then become SQL queries, and a database's query optimizer plans them.

XQL is organized as layers, and each layer is a separate, optional project:

```
frontends:   SQL text · xarray (xarray-sql) · PyTorch (torch-sql)
transforms:  ddx (differentiation) · einfold (plan rewrites)
runtime:     execution, device residency, kernels, memory, scheduling, distribution
engines:     DataFusion · DuckDB · GPU engines (Sirius, GQE) · any ADBC database
```

§2 introduces each project. This doc specifies the missing layer,
**xql-runtime**, which runs XQL's programs on GPUs and other devices.

**Why.** How data is held and moved between queries, more than planning
or arithmetic, decides how fast XQL runs a training step (S1). On a
123.6M-parameter GPT, sending matrix products to a GPU changed nothing,
while holding intermediates as dense arrays made the step 67 times
faster. Keeping weights and optimizer state on the GPU then took a
smaller model's step from 2.02 s to 101 ms, against PyTorch's 27 ms
(S1.1). Today each frontend builds this for itself: xarray-sql's
experiments built three executors and a training session inside
xarray-sql, and torch-sql's optimizer step runs a query per tensor (S1.3).

**Its shape: a DataFusion extension.** The runtime extends
[DataFusion](https://datafusion.apache.org/) the way
[datafusion-distributed](https://github.com/datafusion-contrib/datafusion-distributed)
does. datafusion-distributed adds distributed execution to an ordinary
DataFusion session with one call on its builder, and a distributed plan is
an ordinary DataFusion physical plan with a few extra nodes that move
Arrow data between machines. The runtime adds devices the same way: one
call on the session's builder, and a device plan is an ordinary physical
plan with a few extra nodes that move data between memory tiers and run
work on a device (§5). Nothing a DataFusion user knows changes, and the
two extensions compose: a plan can have network nodes and device nodes at
once (§10.5).

The runtime adds one thing DataFusion has no layer for: a *program*,
several queries run as one training step, with state that persists from
run to run (§5.4).

**Its values.**
- **One job.** The runtime is the bridge between relational plans and
  array hardware. It builds no compiler, allocator or communication
  library of its own, and meets the systems that do through DataFusion's
  extension points and standard formats. It is one component of a
  composable data system, a *deconstructed tensor runtime* in the sense of
  Wes McKinney's deconstructed database (R11, S5.4).
- **Every control within reach.** The SQL says what a program computes.
  Every physical choice the runtime makes, such as where a table lives and
  which kernel computes it, is visible and can be overridden, so the
  runtime is never a compiler its users have to outsmart (R10, S5.1).
- **Rust and Python.** Everything the runtime offers works from both, as
  DataFusion does, so that accelerated ML code is as easy to write in Rust
  as in Python (R12).

---

## 2. Background

Every term the design uses is introduced here. Each group is collapsed;
open the ones you need.

<details>
<summary><b>2.1 Formats and terms</b>: Arrow, ADBC, Substrait, DLPack, Zarr, CTEs, compact arrays</summary>

- **[Apache Arrow](https://arrow.apache.org/)** is a columnar memory
  format, which every XQL engine reads and writes. Its *PyCapsule
  interface* hands Arrow data between Python libraries without copying,
  and its [*C Device Data Interface*](https://arrow.apache.org/docs/format/CDeviceDataInterface.html)
  does the same for GPU memory, naming each kind of memory with a device
  type (`CPU`, `CUDA`, `CUDA_HOST` for pinned host memory, `CUDA_MANAGED`
  and others). Its *fixed-shape tensor* type is a column whose every value
  is a dense array of one shape. **ADBC** is Arrow's standard API for
  databases.
- **[Substrait](https://substrait.io/)** is a portable format for
  relational plans. A plan can share a subplan with a `ReferenceRel`, but
  neither DataFusion's nor DuckDB's producer emits one, so a subplan read
  twice is written out twice.
- **DLPack** shares a dense array between array libraries without copying.
- **Zarr** is a format for chunked N-dimensional arrays, which Xarray reads
  and writes.
- A **CTE** is a named subquery, `WITH x AS (...)`, that a query can read
  several times. A **view** is a named query stored in a session.
- A **compact array** is how the runtime holds a table densely: each
  dimension's coordinates stored once, a dense array of values, and masks
  for which rows exist and which values are NULL.

</details>

<details>
<summary><b>2.2 DataFusion</b>: the layers the runtime extends</summary>

**[DataFusion](https://datafusion.apache.org/)** is a SQL query engine in
Rust, designed to be extended. Its
[library guide](https://datafusion.apache.org/library-user-guide/index.html)
covers each layer the runtime uses:
- the session: [`SessionContext`](https://docs.rs/datafusion/latest/datafusion/execution/context/struct.SessionContext.html),
  assembled by a [`SessionStateBuilder`](https://docs.rs/datafusion/latest/datafusion/execution/session_state/struct.SessionStateBuilder.html),
  with [settings](https://datafusion.apache.org/user-guide/configs.html)
  that SQL can change with `SET`;
- [`RuntimeEnv`](https://docs.rs/datafusion/latest/datafusion/execution/runtime_env/struct.RuntimeEnv.html),
  with its [`MemoryPool`](https://docs.rs/datafusion/latest/datafusion/execution/memory_pool/trait.MemoryPool.html)
  and `DiskManager`, from which operators reserve memory and to which they
  spill;
- the physical plan: [`ExecutionPlan`](https://docs.rs/datafusion/latest/datafusion/physical_plan/trait.ExecutionPlan.html)
  nodes, each running as parallel *partitions*, rewritten by
  [`PhysicalOptimizerRule`](https://docs.rs/datafusion/latest/datafusion/physical_optimizer/trait.PhysicalOptimizerRule.html)s,
  and reporting metrics through
  [`EXPLAIN ANALYZE`](https://datafusion.apache.org/user-guide/sql/explain.html);
- [datafusion-proto](https://docs.rs/datafusion-proto/latest/datafusion_proto/),
  which serializes plans, with a codec for each custom node;
- [datafusion-python](https://datafusion.apache.org/python/), whose
  `SessionContext.with_extensions(...)` installs Rust extensions on a
  Python session through *datafusion-ffi*. einfold attaches this way.

DataFusion's Arrow library, arrow-rs, keeps buffers in host memory, so a
record batch can't hold GPU memory (S6.5).

**[datafusion-distributed](https://github.com/datafusion-contrib/datafusion-distributed)**
adds distributed execution to DataFusion: `.with_distributed_planner()` on
the builder, and workers that each run their own session. A custom node is
distributed by registering its codec on every session.

</details>

<details>
<summary><b>2.3 Other engines</b>: DuckDB, Sirius, GQE</summary>

- **[DuckDB](https://duckdb.org/)** is an embedded analytical database. ddx
  runs on it through DuckDB's community Substrait extension.
- **[Sirius](https://github.com/sirius-db/sirius)** and NVIDIA's **GQE** are
  SQL engines that run on GPUs.

</details>

<details>
<summary><b>2.4 XQL's frontends and transforms</b>: xarray-sql, torch-sql, ddx, einfold</summary>

- **[xarray-sql](https://github.com/xqlsystems/xarray-sql)** registers
  [Xarray](https://xarray.dev/) Datasets as tables, and turns results back
  into Datasets. Its demos are each checked against PyTorch or JAX: demo 1
  is attention, demo 2 a GPT trained in SQL, demo 3 nanoGPT's Shakespeare
  recipe, and demo 4 Samudra's U-Net.
- **[torch-sql](https://github.com/xqlsystems/torch-sql)** runs
  [PyTorch](https://pytorch.org/) programs as SQL: a tensor on
  `device="sql"` is a table. It trains
  [nanoGPT](https://github.com/karpathy/nanoGPT). Its design doc's spikes
  are cited here as "torch-sql's spike S4" and so on.
- **[ddx](https://github.com/xqlsystems/ddx)** differentiates SQL. From a
  query that computes a loss, it writes a *program* of *steps*, each a plan
  that saves its result as a table, together with *checks* that must return
  no rows. In SQL, its `grad`, `vjp` and `jvp` functions ask for gradients
  inside a query. `ddx-ad` is its Rust crate, `ddxdb` its Python package,
  and `Backend` the interface an engine implements to run its programs.
- **[einfold](https://github.com/xqlsystems/einfold)** is a query optimizer
  for DataFusion that finds array computations in a plan, such as a matrix
  product written as a join and a `GROUP BY`, and runs array algorithms for
  them in its CPU operator, `EinFoldExec`. Its design calls whatever runs
  its plans the *host*, and says einfold never touches a device.

</details>

<details>
<summary><b>2.5 Kernels, compilers and GPU libraries</b>: Triton, Gluon, Pallas, cuTile Rust, CUTLASS, XLA, RMM, NCCL</summary>

The runtime calls these rather than replacing them (R11). S6.2 has how
each fits.
- **[Triton](https://triton-lang.org/)** is a Python language for GPU
  kernels, run as a grid of programs that each load, compute and store a
  tile, with masks at the array's edges. **Gluon** is Triton's lower-level
  language, launched the same way, in which the author chooses layouts,
  shared memory, barriers and warp roles.
- **[Pallas](https://docs.jax.dev/en/latest/pallas/index.html)** is JAX's
  kernel language: `pallas_call` runs a kernel over a grid, and a
  `BlockSpec` says which block of each input a grid point reads. On NVIDIA
  GPUs it compiles through Triton or *Mosaic GPU*.
- **[cuTile Rust](https://github.com/NVIDIA/cuda-rust/tree/main/cutile-rs)**
  is NVIDIA's tile-based kernel language for Rust. Kernels are Rust
  functions, JIT-compiled through CUDA Tile IR, whose launches are Rust
  futures, and whose types keep outputs exclusive and inputs shared. It
  lives in NVIDIA's `cuda-rust` repository, beside NVIDIA's own CUDA
  driver bindings. **cudarc** is a widely used alternative binding of the
  CUDA driver, and **CubeCL** another Rust kernel language, used by the
  **Burn** framework.
- **CUTLASS** is NVIDIA's library of templates for fast matrix kernels, and
  its *CuTe* layer describes data layouts across a GPU's threads.
- **XLA** is JAX's compiler. It compiles **StableHLO**, a portable set of
  array operations, built on **MLIR**, a framework for layered compiler
  representations. **PJRT** is XLA's plugin interface to hardware.
- **`torch.compile`** is PyTorch's compiler, which generates Triton
  kernels among others.
- **RMM** is a GPU memory allocator that several libraries can share.
  CUDA's own *stream-ordered allocator* (`cudaMallocAsync`) keeps memory
  pools that libraries can also share, and RMM can draw from it.
- **NCCL** is NVIDIA's library for moving data between GPUs.

</details>

<details>
<summary><b>2.6 Models and hardware</b>: Samudra, MLX, the Beta cluster</summary>

- **[Samudra](https://github.com/m2lines/Samudra)** is an ocean emulator and
  XQL's first production model: a *U-Net*, a convolutional network that
  shrinks a grid level by level and grows it back, built from *ConvNeXt*
  blocks of dilated convolutions, batch norms and GELUs, on a half-degree
  grid of 360 × 720 cells.
- **MLX** is Apple's array library for its GPUs. PyTorch runs on them
  through its *MPS* backend.
- **The Beta cluster** of
  [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
  is where Samudra trains: 288 NVIDIA B200 GPUs (compute capability
  `sm_100`) in GB200 NVL72 racks. Each Grace CPU, an ARM CPU, is joined to
  two B200s by NVLink-C2C, which lets a GPU read the CPU's memory directly,
  and NVLink joins the 72 GPUs of a rack.

</details>

---

## 3. Goals and non-goals

### 3.1 Goals

1. **Run programs from any XQL frontend or transform** through DataFusion:
   ddx's gradient programs, frontends' training steps, and plain queries.
2. **Own tables and their memory.** Every table the runtime holds has an
   owner, a lifetime and a placement.
3. **Keep data where the computation is**, between operators and between
   steps, so a training loop moves only its batches and the values it
   reads.
4. **Choose an implementation for each piece of work**: DataFusion's own
   operators by default, and kernels for work einfold has marked or a
   developer has bound.
5. **Hold values across the memory hierarchy**: by default where the
   runtime judges best, and wherever the developer says. Recompute values
   only under a policy the user turns on.
6. **Schedule**: run independent steps concurrently, and prepare the next
   batch while a step runs.
7. **Distribute**, by composing with datafusion-distributed.
8. **Give developers every control**: plans they can read and edit,
   kernels of their own bound to any part of a program, and a report of
   what each run cost.
9. **Work the same from Rust and Python.**

### 3.2 Non-goals

- **An SQL engine of its own.** The runtime never interprets relational
  operators: no joins, no `GROUP BY`, no NULL rules. DataFusion runs those
  (S2.1).
- **Differentiation, plan rewriting, or drawing a program's step
  boundaries.** Those are ddx's and einfold's (§6.4).
- **A kernel compiler, a memory allocator or a communication library.**
  Kernels come from Triton, Gluon, Pallas, CUTLASS, XLA, `torch.compile`,
  MLX, CubeCL or a developer's own code. GPU memory comes from a shared
  allocator, and data moves between GPUs through NCCL.
- **A distributed planner.** datafusion-distributed is one (§10.5).
- **Its own way of assembling a session.** The runtime attaches through
  DataFusion's own mechanisms, and adds no installer of its own (§5.1).
- **A training framework.** Training loops, optimizers and learning-rate
  schedules belong to frontends. The runtime provides state that
  persists, atomic updates, and checkpoints of that state.
- **Bitwise agreement across implementations.** Results agree with
  DataFusion's to the bound of §6.3.

---

## 4. Principles

- **R1. The SQL is the meaning.** A program's result is what its SQL
  computes on the engine it runs on (S4.4). Every other implementation is
  checked against that, declines what it can't do exactly, and never
  changes which rows exist.
- **R2. Public interfaces only.** The runtime depends on its neighbours'
  released, public APIs, and they depend on its. No private imports across
  repositories, and no forks.
- **R3. Programs, not queries.** The unit of work is a program, usually a
  whole training step, compiled once and run many times.
- **R4. Every table has an owner.** Frontends hold handles, never bare
  names, and the runtime knows when each table may be dropped.
- **R5. State changes atomically.** Tables that persist across runs take
  their new values only when a run's checks pass. A failed run leaves them
  as they were.
- **R6. Each layer stays optional.** A frontend can run without the
  runtime, and the runtime runs on a CPU-only machine with no kernels. Each
  capability is opt-in and measured on its own.
- **R7. Where a value is held is never part of the program's meaning.**
  Keeping a value in DataFusion's memory, in host memory or on a device
  changes neither results nor gradients. That is what makes it safe to
  hand every such choice to the developer (R10). Recomputing a value
  instead of saving it happens only under a user's policy, and only for
  plans that give the same rows each time (§6.5).
- **R8. Declines and costs are visible.** The runtime counts every
  decline, with its reason and the label of the code it came from (§5.5),
  and reports what each run cost (§6.6). *Strict* mode turns declines into
  errors. *Checking* mode runs DataFusion's plan beside a substitute and
  compares them.
- **R9. Reproducibility is stated.** Each implementation says whether the
  same inputs give the same bits. A session may require reproducibility,
  and then gets only the implementations that give it (S3.7).
- **R10. Defaults, not hidden decisions.** Every physical choice is in a
  plan or a schedule a developer can read (§5.6). The runtime fills in
  defaults, and a developer may override any of them. An override is
  binding: if the runtime can't obey it, compiling fails and says why.
  Nothing is ignored silently (S5.1).
- **R11. Vanilla DataFusion, plus nodes.** The runtime depends on
  DataFusion heavily and on purpose, and extends it only through its own
  extension points: builder methods, settings, `RuntimeEnv`, physical
  rules, `ExecutionPlan` nodes, metrics and codecs. A device plan is a
  DataFusion physical plan with a few more nodes, so DataFusion's tools
  (`EXPLAIN`, metrics, datafusion-proto, datafusion-distributed) work on it
  unchanged. Beyond DataFusion, tables cross the runtime's boundaries as
  Arrow, arrays as DLPack and plans as Substrait (S5.4, S6.3).
- **R12. Rust and Python, one design.** Every capability is in Rust, and
  every one is reachable from Python. Kernels may be written in either,
  and a Rust program needs no Python (S6.6).

---

## 5. The model

### 5.1 The session

A runtime session is a DataFusion session with the runtime's extension
added. Each XQL project attaches the same way, and the runtime adds no
installer of its own (S6.7).

In Rust, each project adds a method to `SessionStateBuilder`, as
datafusion-distributed's `.with_distributed_planner()` does:

```rust
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_ddx()                              // grad, vjp and jvp in SQL (§10.2)
    .with_einfold()                          // einfold's detection (§10.3)
    .with_xql_runtime(RuntimeOptions::new().device("cuda:0"))
    .build();
let ctx = SessionContext::new_with_state(state);
```

In Python, each project provides an extension object for
`SessionContext.with_extensions`, as einfold's `EinFold` already is:

```python
ctx = SessionContext().with_extensions(Ddx(), EinFold(), XqlRuntime(devices=["cuda:0"]))
```

Rules:
- **Settings live in the `xql_runtime` namespace** of `SessionConfig`, so
  SQL can set them: `SET xql_runtime.strict = true`,
  `SET xql_runtime.checking.sample = 0.01`.
- **The runtime's state lives in DataFusion's.** Its device memory and
  device handles are in the session's `RuntimeEnv` (§5.2), and its nodes
  find them through the `TaskContext` they run with.
- **Frontends assemble the session.** xarray-sql's context adds whichever
  XQL extensions are installed, so that, with ddxdb installed, its SQL
  understands `grad` with no other step.
- **Workers are sessions too.** Under datafusion-distributed, each worker
  builds its session with the same extensions (§10.5).

### 5.2 Memory tiers

The runtime extends `RuntimeEnv`'s memory hierarchy. DataFusion already has
two tiers: host memory, through its `MemoryPool`, and disk, through its
`DiskManager`. The runtime adds the device tiers. Tiers are named by Arrow's
device types (§2.1), with `storage` for files.

| Tier | Arrow device type | Accounting |
|---|---|---|
| device memory | `CUDA` (or `METAL`, ...) | a device memory pool per device, added by the runtime |
| pinned host memory | `CUDA_HOST` | the device pool of the device it is pinned for |
| host memory | `CPU` | DataFusion's `MemoryPool` |
| storage | none | DataFusion's `DiskManager`, plus Zarr and Arrow IPC files |

Rules:
- **Device memory is accounted as DataFusion accounts host memory.** A
  node reserves bytes from the device pool before it allocates, and a
  refused reservation means spilling to a lower tier, as a refused host
  reservation means spilling to disk.
- **The bytes come from a shared allocator**, never from an allocator of
  the runtime's own (§7).
- **A table handle** refers to a table the runtime owns, as a buffer handle
  refers to an allocation. It carries the table's schema, its lifetime (one
  run, or until the caller releases it), and one or more *instances*. Each
  instance has a tier; a layout, either Arrow rows or a compact array; and
  an owner, either DataFusion (as one of its tables) or the runtime.
- **Runtime tables are DataFusion table providers**, read as Arrow batches
  by DataFusion's operators, and as compact arrays by device nodes that
  scan them directly (§5.3).
- **Developers may move tables themselves**: `prefetch` to a tier, `evict`
  from one, and `pin` an instance so that it stays. Each tier reports its
  capacity and what it holds.
- **Tables are values, as in SQL.** A buffer is written in place only when
  state is published (R5, §7), so aliasing never reaches a program's
  meaning (S5.6).
- **Reading a handle's rows is the only way data leaves the runtime**, and
  it is always explicit.
- **Names are stable.** Each handle has a stable name, by which plans read
  it, and the runtime passes ddx a namespace (`Options::namespace`) so that
  the same plan gives the same program byte for byte (S3.3).
- **A table that holds state reads the handle's current buffer** each time
  its plan runs, so that publishing new state never forces a plan to be
  planned again (S3.3).

### 5.3 Device plans

A **device plan** is a DataFusion physical plan in which some nodes are the
runtime's. The runtime's physical optimizer rule places them, after
einfold's rules have marked the array work (§6.2):
- **`KernelExec`** runs a bound node (§5.7) on a device. It keeps the
  subplan it replaced as its fallback, as `EinFoldExec` does.
- **`CopyExec`** copies its input from one tier or layout to another: host
  rows to a compact array on a device, say, or back. Every copy is one of
  these nodes, so every copy shows in `EXPLAIN`.

Rules:
- **Between device nodes, data stays on the device.** arrow-rs batches
  can't hold device memory, so device nodes pass *handle batches*: record
  batches whose one column holds handles to device buffers in the
  runtime's pool. Only device nodes read them. The physical rule places a
  `CopyExec` wherever a handle batch would reach one of DataFusion's
  operators (S6.5).
- **Partitions map to the hardware.** On one GPU, a node's partitions run
  on separate CUDA streams. Across GPUs in one machine, each partition
  runs on one device.
- **`EXPLAIN` shows the device plan**, with each runtime node's device,
  layout, kernel, and who chose it: a default, einfold or the developer.
  **`EXPLAIN ANALYZE`** adds each node's metrics: time in the kernel, bytes
  copied, peak device memory and declines.
- **The runtime's nodes have a codec**, so that datafusion-proto can
  serialize a device plan and datafusion-distributed can ship one (§10.5).

### 5.4 Programs

A **program** is a directed acyclic graph of **steps**. Each step is a
query, planned by DataFusion as a device plan. A program names:

- its **inputs**: tables supplied on each run, such as a batch;
- its **state**: tables that persist across runs, each with the step that
  computes its next value;
- its **parameters**: scalars that change between runs without changing
  the plans, such as a learning rate;
- its **outputs**: the tables the caller reads after a run;
- its **checks**: conditions a run must meet for its result to be valid.

A program is *lowered*, then *compiled*, as a JAX function is (S6.1):
- **`lower`** plans every step. The result can be read with `explain()`
  and edited (§5.6).
- **`compile`** fixes implementations and placements, runs the checks that
  need running only once, and returns a compiled program.
- **`run`** runs it with new inputs and parameters. A run is asynchronous:
  `run` returns at once with a future for its outputs. Runs of one
  compiled program execute in order.

A compiled program is reused only while the backend's session key is
unchanged, as ddx's `prepare` does.

Rules:
- **A training step can be one statement.** With ddx's functions in the
  session, `SELECT k, v - $lr * grad(loss, v) AS v FROM w` is a whole step
  of gradient descent. ddx writes the program from it.
- **Programs compose.** A step may also be ddx's program followed by
  update steps that read its gradient tables. The runtime takes both as
  one program.
- **ddx's checks run when they must.** Each states a promise: that some key
  columns of one table are unique. The runtime runs a check once per
  compile for a table whose keys don't change, on each run otherwise, and
  never when a frontend's fact gives the same promise.
- **einfold's checks run on every run**, as data checks on the values of
  that run (§6.3, S3.1).
- **Parameters are written as DataFusion's placeholders**, such as `$lr`.
  Where a placeholder can't reach ddx through Substrait, the runtime turns
  it into a one-row table that the plans cross join (S3.2).

### 5.5 Labels

One marker names a part of the forward program for every layer:
`ddx_name(expr, 'label')`, proposed on ddx#251, since CTE and view names
don't survive the engines' Substrait producers (torch-sql's spike S3). In
ddx's SQL path, CTE names serve too. Labels carry:
- ddx#251's regions, and the save points and recompute policies of §6.4
  and §6.5;
- provenance: which forward relation each backward step comes from
  ([ddx#259](https://github.com/xqlsystems/ddx/issues/259));
- einfold's hints (§6.2);
- bindings (§5.7);
- the runtime's counters and metrics (R8).

Frontends choose the labels. torch-sql uses module paths, such as
`unet.layers.3`, so a decline is reported against the module that caused
it. A binding or override that names a label the program doesn't have is
an error.

### 5.6 The schedule

The SQL is the program's *algorithm*: what it computes. The **schedule** is
how the runtime computes it, and it has two levels:
- **Per step, the schedule is the device plan** (§5.3): which nodes run on
  which device, with which kernel, at what precision, and where data is
  copied.
- **Per program, it adds** what no single plan holds: where each table's
  instances live between steps, save points and recompute policies passed
  to ddx (§6.4, §6.5), and donation of state buffers (§7).

Rules:
- **Every entry records who chose it**: a default, einfold or the
  developer.
- **The schedule is readable.** `explain()` prints it, and it serializes to
  JSON in a documented, versioned format. Device plans in it serialize
  through datafusion-proto, with the runtime's codec.
- **The schedule is editable.** `compile(schedule=...)` takes a whole or
  partial schedule. The runtime fills in what is missing and obeys what is
  given (R10).
- **Changing a schedule never changes a result** beyond the bound of §6.3.
  Checking mode verifies that.

### 5.7 Bindings and kernels

A **binding** says which kernel computes a labelled node. einfold's
detection makes bindings automatically, to the runtime's array backends
(§6.2). A developer, or an agent, binds a kernel by hand. Either way, the
node's SQL is the kernel's specification, and runs if the kernel declines.

**What a kernel receives.** The runtime calls a kernel with:
- each input as a device array, through DLPack, with its axes named and
  their extents given;
- each input's presence and NULL masks, unless the kernel assumes there
  are none;
- the output arrays, allocated by the runtime from the node's schema;
- the stream to launch on.

The kernel chooses its own grid, blocks and on-chip memory. Shared memory,
tensor memory and registers are below the runtime's tiers, and belong to
the kernel (S6.2). Tuning parameters a kernel chooses, such as Triton's
block sizes and warp counts, are recorded in the schedule, so that a
compiled program keeps them and a developer can set them.

**From Python**, kernels are registered on the runtime's extension object
before the session is built, so the session has them from the start.

<details>
<summary>A Triton kernel and a Pallas kernel, bound by label</summary>

```python
runtime = XqlRuntime(devices=["cuda:0"])

@runtime.kernel("predict", assumes=["complete", "no_nulls"])
def predict(args):                                     # Triton or Gluon
    x, w, out = args["x"], args["w"], args.out
    gemv[(triton.cdiv(x.extent("cell"), 128),)](x, w, out, x.extent("k"), BLOCK=128)

@runtime.kernel("attn.0")
def attention(args):                                   # Pallas, through JAX
    return flash_attention(jax.dlpack.from_dlpack(args["q"]), ...)
```

</details>

**From Rust**, a kernel implements the `Kernel` trait, and is registered
on the session builder. cuTile Rust is the first-class way to write one.

<details>
<summary>A cuTile Rust kernel, bound by label</summary>

```rust
#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry()]
    fn scale<const B: i32>(out: &mut Tensor<f32, { [B] }>, x: &Tensor<f32, { [-1] }>, a: f32) {
        let tx = x.load_like(out);
        out.store(tx * a);
    }
}

struct Scale;
#[async_trait]
impl Kernel for Scale {
    async fn launch(&self, args: &mut KernelArgs) -> Result<()> {
        let x = args.cutile_input::<f32>("x")?;                // the runtime's buffer, not a copy
        let out = args.cutile_output::<f32>()?.partition([128]);
        kernels::scale(out, x, 2.0).await?;                    // a future, on the runtime's stream
        Ok(())
    }
}

builder.with_xql_runtime(RuntimeOptions::new().device("cuda:0").kernel("scale", Scale))
```

</details>

cuTile needs no adapter of its own beyond `KernelArgs` (S6.2):
- **Memory and streams are the runtime's.** Inputs are lent as cuTile
  tensors over the runtime's buffers (`Tensor::from_foreign`), outputs as
  borrowed partitions, and the runtime lends its CUDA context and the
  partition's stream. The runtime implements cuTile's `SchedulingPolicy`.
- **Launches are awaited**, inside a `KernelExec`, on the Tokio runtime
  DataFusion already executes on.
- **JIT compilation happens at `compile`**, through cuTile's `.compile()`
  on placeholder tensors, never during a run.
- **Types are dispatched at run time.** cuTile's element types are static
  and Arrow's are not, so a binding matches on the column's type.
- **cuTile is pinned exactly, behind a feature.** It is a research release
  that requires CUDA 13.2 or later on `sm_80` or newer.

A kernel compiled elsewhere, from CUDA C, CUTLASS or Triton ahead of time,
is loaded as PTX or a cubin and launched through the same CUDA bindings.

Rules:
- **`assumes` lists the facts a kernel relies on** (§6.2). The runtime
  checks each on every run, unless a frontend's fact states it.
- **A kernel that returns a new array**, as a Pallas kernel through JAX
  does, has its buffer adopted through DLPack when it came from the shared
  allocator, and copied otherwise.
- **Hand-written kernels are tested, not proved.** einfold's nodes are
  proved exact from the plan's structure. A developer's kernel is only
  tested against its SQL, which is weaker (S5.3). So checking mode samples
  hand-written kernels more often, their runs are counted separately, and
  a session may refuse them in strict mode.

---

## 6. Execution

### 6.1 Three ways to run a step

The schedule says, per step, which way it runs. By default the runtime
chooses:

1. **On DataFusion**, as a plain query. This is the reference (R1), and
   always available.
2. **With bound nodes.** Each node einfold marked, or a developer bound,
   runs as a `KernelExec` on compact arrays kept resident, or declines, and
   then its subplan runs on DataFusion. With einfold's nodes, this is stage
   B-0 (§6.2).
3. **As a region**: several nodes, and the elementwise expressions between
   them, run as one array program, with their intermediates kept compact.
   These are stages B-1 and B-2.

A GPU SQL engine such as Sirius would be a fourth way, outside DataFusion.
Spike 2 tests whether it meets the same wall as GPU kernels on long-format
rows (S1.1).

### 6.2 Dense execution

**einfold marks the SQL, and the runtime runs what it marks.** The program
stays SQL: it is what every engine runs, what ddx differentiates, and what
R1 calls the meaning. There is no separate array language (D2, S2.2).
einfold annotates a step's plan with nodes of five kinds:
- *folds*: sums of products over a join, with their domains, NULL rules
  and fallbacks;
- *pipelines*: a fold that reads another fold's result through an
  expression, which einfold fuses as FlashAttention does (computing
  attention one tile at a time, without holding every score);
- *shifted folds*: folds with shifted outputs, which is how a convolution
  appears ([einfold#155](https://github.com/xqlsystems/einfold/pull/155));
- gathers on value indexes, such as an embedding's lookup of token ids;
- the elementwise expressions between them.

**The node contract.** Each node:
- states its named axes, and the three states each cell can be in: absent,
  NULL, or a value;
- carries the subplan it stands for, which runs if the node is declined;
- carries its run-time checks, as data checks;
- is one of einfold's published, versioned node types, with a conformance
  suite.

Each node einfold detects becomes a binding (§5.7) to one of the runtime's
array backends. How a backend lowers a node (to NumPy, MLX, Triton,
`torch.compile`, or StableHLO through JAX or PJRT) is private to the
backend and crosses no interface. Gathers on value indexes,
floating-point guards and index-table joins need particular care in
torch-sql's programs (S2.4).

**Facts and hints.** For a node to skip a check, einfold needs a fact
about its inputs from einfold's *fact table*, which is designed (einfold's
design §8.1) but not built.

<details>
<summary>The facts einfold's nodes need</summary>

- each dimension's extent, with integer keys `0 .. n - 1`;
- completeness, or a table of the domain;
- unique tuples;
- no NULL values;
- for gathers, that each index lies in range;
- for sums einfold reorders, that values are finite and bounded.

</details>

torch-sql states the structural facts for every table it writes (S1.3), as
[einfold#39](https://github.com/xqlsystems/einfold/issues/39) does for
xarray-sql, and the runtime states them for the tables it writes. Facts
about values hold for one run only, so they stay data checks (S3.1). A
frontend may also pass a *hint*, the array meaning it already knows, by
label (§5.5). einfold uses hints to guide detection, never in place of its
proof, and a hint it fails to confirm is a counted decline.

**Near misses.** Where einfold almost detects a node, it says why it
didn't: for example, "this would be a fold, but key `k` is nullable". The
runtime reports near misses with the run (§6.6), so that a developer or an
agent can change the SQL, state the missing fact, or bind a kernel by
hand, instead of guessing how to spell the query. The SQL's spelling
should decide correctness, never speed, and near misses are how the gap
between the two gets closed.

**Stages.** Build B in three stages, each measured and useful alone (S2.6):
- **B-0, one node at a time.** The runtime's physical rule replaces the
  node einfold detected with a `KernelExec`, which keeps einfold's node
  and its checks as its fallback. This replaces
  [einfold#156](https://github.com/xqlsystems/einfold/pull/156)'s per-call
  interface. Values stay compact between device nodes, and become rows
  wherever a `CopyExec` hands them to DataFusion's operators (S2.5).
- **B-1, regions within one step**: einfold nodes connected through
  elementwise projections run as one array program, in one `KernelExec`.
- **B-2, regions across steps**, through einfold's program mode, with
  ddx's step graph (ddx#157) and regions (ddx#251).

B pulls einfold's fact table and program mode ahead of its DuckDB and
sparse-kernel milestones. That is Alex's decision as einfold's maintainer,
after a sketch in einfold's design doc (§13, question 4).

Design C, in which DataFusion carries dense values itself as fixed-shape
tensor columns, stays open through spike 1 (S2.3).

### 6.3 Checks, declines and fallbacks

- **Checks on a device.** Each of einfold's run-time checks compiles to an
  on-device predicate. The predicates are ANDed into one validation flag,
  which the host reads once per run (S3.1).
- **When the flag is read.** A compiled step reads its flag after its last
  check and before its first state update.
- **Declines at compile time.** The node's subplan runs on DataFusion, and
  the rest of the program keeps its implementations.
- **Declines at run time.** A failed flag leaves the state untouched (R5),
  and the failed nodes' subplans run on DataFusion from their inputs.
  Those inputs stay readable until the flag has been read.
- **Counting.** Each `KernelExec` reports its runs and declines, by reason
  and label, as metrics (R8).
- **Strict and checking modes.** Benchmarks run strict. Checking mode runs
  the subplan beside a kernel on sampled steps, compares them to the bound
  below, and verifies the facts each node relied on.
- **The bound.** A float sum of `n` terms, in any order, must be within
  `(n − 1)·u·Σ|terms|` of the exact sum, where `u` is the unit roundoff of
  the precision the kernel computes in (S3.8). A session chooses `u`:
  computing in float32 on a GPU is an explicit choice.
- **Reproducibility.** Each kernel declares whether it gives the same bits
  for the same inputs (R9).

### 6.4 Step boundaries and saving

- **ddx draws step boundaries.** The runtime takes ddx's program as ddx
  writes it (S3.4).
- **The runtime changes them only through ddx's inputs:**
  - save points and recompute policies, by label (§5.5), passed to ddx as
    options, on one surface with ddx#251's regions;
  - engine limits, such as the depth a consumer can decode and a planning
    budget, which the runtime's `Backend` reports and ddx reads in place of
    its constants.
- **ddx never asks the `Backend` where to save**, so that a cached program
  stays a function of its plan and its options (S3.4).
- **Shared subplans come from SQL or from the frontend's program**, never
  from a Substrait plan. For plain queries, the runtime calls ddx's public
  form of ddx#168's split, and has no implementation of its own.
- **Save points are schedule entries** (§5.6), so a developer can add,
  remove or move one, and the runtime passes the result to ddx.
- **The runtime chooses default save points with einfold.** einfold
  reports whether a candidate save point would split a node it can fuse,
  and the runtime prefers save points that don't. A split taken anyway is
  a counted decline (S3.5).
- **Frontends don't run their own chain rules.** torch-sql retires its cut
  points by handing ddx SQL with a CTE at each shared node (S3.6), and
  stops gradients with `ddx_stop_gradient`, never by saving a table
  ([ddx#260](https://github.com/xqlsystems/ddx/issues/260)).

### 6.5 Recomputing

- **Recomputing is a policy the user opts into** (D1), by label. The
  vocabulary follows JAX's: `save_only_these_names(...)`, and an offload
  policy that moves a value to `CUDA_HOST` rather than recomputing it
  (S6.1). The runtime applies a memory-budget policy only when the session
  turns one on, and may always save more than a policy asks.
- **Only plans that give the same rows twice are recomputed** (S3.7). ddx
  reads volatility from the engine through its `Backend`.
- **The runtime's random function is declared deterministic**, so that a
  recomputed dropout draws the same mask.
- **`torch.utils.checkpoint`** becomes a recompute policy on a labelled
  region. Its random-state preservation comes free from the counter-based
  random function, or from masks uploaded as inputs (S3.6).

### 6.6 Reports

- **Per step, reports are DataFusion's.** `EXPLAIN ANALYZE` prints each
  node's metrics, the runtime's included (§5.3).
- **Per program, `report()` adds** what crosses steps: copies between
  tiers between steps, the peak memory of each tier, declines and near
  misses by label, and the schedule the run used, with who chose each
  entry.
- **Both are available as JSON**, in a documented format, so that an agent
  can read a run's cost, change the schedule and run again.

---

## 7. Memory

- **The runtime's defaults decide where each value is held**, from
  DataFusion's limits, each tier's capacity and the session's policies.
  The schedule records the choice, and a developer can override it (§5.6).
- **The runtime allocates no GPU memory of its own.** It draws from a
  shared allocator, so that it shares a GPU with PyTorch, JAX and cuTile
  instead of competing with them for it (S4.2). This draft leans toward
  CUDA's stream-ordered pools as the shared one: cuTile's driver layer
  allocates stream-ordered, and RMM and PyTorch can each be configured to
  draw from the same pools (§13, question 6).
- **State has room for two copies** while a step runs, unless the program
  donates: once the validation flag has been read, each state update writes
  into the old table's memory, as JAX's buffer donation does (S3.9).
- **A program that donates restarts from its last checkpoint** after a
  failure that isn't a check, such as an engine error.
- **Grace memory is a host tier** on the Beta cluster, for values a GPU
  reads directly. Spike 3 measures whether it is fast enough to spill to.

---

## 8. Scheduling

- **Within a step**, DataFusion schedules partitions, and the runtime maps
  them to streams and devices (§5.3).
- **Within a run**, each step runs once the steps it reads are done. That
  needs ddx's step graph on `Program` (ddx#157) and a `Backend` whose
  `materialize` may run concurrently. Measure the gain again on compiled
  runs before building it (S1.4).
- **Across runs**, the runtime loads the next batch while a step runs. For
  Samudra, batches come from xarray-sql's data path over a Zarr archive.

---

## 9. State, checkpoints and randomness

`TrainingSession` (S1.1) splits along the line between execution and
training:

| Part of `TrainingSession` | Goes to |
|---|---|
| State tables kept as arrays between runs, with their device buffers | runtime (§5.2) |
| Publishing new state only after validation | runtime (R5) |
| Compiling the whole step once, guarded by the session key | runtime (§5.4) |
| Reusing the loss ddx already computed ([ddx#257](https://github.com/xqlsystems/ddx/pull/257)) | runtime, through ddx's public API |
| Checkpoints of state, with the step counter and random state | runtime |
| Dropout with explicit seed and step keys | the runtime's random function; the frontend decides where dropout goes |
| AdamW, gradient clipping, the learning-rate schedule and evaluation | frontend |

- **Checkpoints are Zarr stores**, written from published state only, so
  that a checkpoint is a Dataset SQL can query. State that isn't
  array-shaped is written as Arrow IPC, Arrow's file format.
- **The random function is counter-based**: a hash of a seed, the step
  number and the element's coordinates, written in integer arithmetic that
  never overflows, so that every kernel and every engine computes the same
  bits (S3.10). It is registered as a DataFusion function.
- **Masks may also be inputs.** A frontend checked against another
  framework's random numbers, as torch-sql is against PyTorch's, uploads
  its masks as input tables (S3.10).

---

## 10. Interfaces

### 10.1 To developers and frontends

A training step from Python. The runtime's names are placeholders:

```python
from datafusion import SessionContext
from ddxdb.datafusion import Ddx
from einfold.datafusion import EinFold
from xql_runtime import XqlRuntime, Program

runtime = XqlRuntime(devices=["cuda:0"])
ctx = SessionContext().with_extensions(Ddx(), EinFold(), runtime)
ctx.sql("SET xql_runtime.strict = true")            # declines are errors (R8)
ctx.register_table("features", features)
ctx.register_table("target", target)

STEP = """
WITH pred AS (SELECT x.cell, ddx_name(SUM(x.v * w.v), 'predict') AS v
              FROM features x JOIN w ON x.k = w.k GROUP BY x.cell),
     loss AS (SELECT AVG((p.v - y.v) * (p.v - y.v)) AS l
              FROM pred p JOIN target y USING (cell))
SELECT k, v - $lr * grad(loss, v) AS v FROM w
"""
lowered = Program(ctx, STEP, state={"w": w0}, outputs=["loss"]).lower()
print(lowered.explain())                            # device plans, copies, who chose what

schedule = lowered.schedule()                       # read, edit (§5.6)
schedule.table("features").pin("CUDA")
compiled = lowered.compile(schedule=schedule)

for i in range(1_000):
    future = compiled.run(lr=1e-2 * 0.999**i)
print(future.result()["loss"])                      # the only read
print(compiled.report())                            # cost, declines and near misses (§6.6)
```

<details>
<summary>The same from Rust</summary>

```rust
let ctx = SessionContext::new_with_state(
    SessionStateBuilder::new()
        .with_default_features()
        .with_ddx()
        .with_einfold()
        .with_xql_runtime(RuntimeOptions::new().device("cuda:0").strict(true))
        .build(),
);
let lowered = Program::new(&ctx, STEP).state("w", w0).output("loss").lower().await?;
println!("{}", lowered.explain());
let compiled = lowered.compile().await?;
for i in 0..1_000 {
    compiled.run(params! { "lr" => 1e-2 * 0.999f64.powi(i) }).await?;
}
```

</details>

- **torch-sql** gives ddx the runtime as its `Backend` at M0, with no other
  change. From M1, a synced `device="sql"` tensor holds a table handle, and
  a whole training step is one program.
- **xarray-sql** adds the installed XQL extensions to its context (§5.1),
  so that its SQL understands `grad` when ddxdb is installed, and its
  Datasets can be state.

### 10.2 To ddx

The runtime implements `ddx_ad::Backend`, and `ddxdb.Backend` through its
Python binding. From the runtime's first release, `Backend` is a contract:
ddx lists its changes in its changelog, and the runtime pins ddx exactly.
The runtime needs, each as an ordinary ddx PR:

1. **ddx as a DataFusion extension**: `.with_ddx()` on the builder in Rust
   and a `Ddx` object for `with_extensions` in Python, so that `grad`,
   `vjp` and `jvp` work in any session's SQL. This replaces `ddxdb.Context`,
   a subclass of `SessionContext`, since two projects that each subclass
   the context can't both be used (S6.7);
2. `ddx_stop_gradient` on every engine and type (ddx#260);
3. volatility read from the engine through the `Backend` (§6.5);
4. what each check promises (§5.4), and `to_sql` exported from ddxdb,
   since ddxdb runs DuckDB's steps as the SQL `to_sql` writes;
5. the step graph on `Program` (ddx#157);
6. labels and policies: ddx#251's fourth part, with save points beyond
   "read more than once" (§6.4);
7. engine limits read from the `Backend` (§6.4);
8. provenance (ddx#259);
9. the primal outputs of a prepared run (ddx#257) and the derivative of
   `erf` ([ddx#256](https://github.com/xqlsystems/ddx/pull/256)), both in
   review;
10. ddx#265, a wrong value with no error for one-row tables on DuckDB,
    first among ddx#261 to #265.

### 10.3 To einfold

einfold already attaches as a DataFusion extension, and the runtime's rule
runs after its rules. The runtime needs from einfold:
1. a builder method, `.with_einfold()`, beside its `install` function, so
   that every XQL extension attaches the same way;
2. the B-0 interface: each detected node's IR, checks and subplan, which
   the runtime's rule wraps in a `KernelExec` (§6.2);
3. published, versioned node types, with a conformance suite;
4. the fact table (einfold's M5);
5. a report on whether a candidate save point splits a node einfold can
   fuse (§6.4);
6. hints and near misses, by label (§5.5, §6.2).

### 10.4 To engines and across the PyTorch boundary

- DataFusion first, with xarray-sql, ddx and einfold sharing one DataFusion
  version ([xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)).
  DuckDB runs ddx's programs without device plans, as a reference. GPU
  engines come later.
- Tables cross into and out of the runtime as Arrow: the PyCapsule
  interface on the host, the C Device Data Interface on a GPU, and DLPack
  toward PyTorch and JAX. The device path needs a device-stream export
  upstream and a converter into PyTorch, since neither exists (S1.3).

### 10.5 Distribution, through datafusion-distributed

The runtime distributes by composing with datafusion-distributed, not by
building a distributed planner:
- **A distributed device plan** has datafusion-distributed's network nodes
  and the runtime's device nodes in one physical plan.
- **Workers are built with the runtime's extension**, with the same
  devices and kernels registered, as datafusion-distributed requires of
  every custom node's codec.
- **The runtime's codec** serializes a `KernelExec` as its kernel's name
  and tuning parameters, and the worker finds the kernel in its own
  session.
- **GPU-to-GPU transfer** would use NCCL inside datafusion-distributed's
  network nodes, as a contribution there rather than a fork of them.
  Within one NVL72 rack, cuTile's peer-memory operations, which read and
  write another GPU's memory over NVLink, are a lower-level alternative
  for exchanging the cells along a tile's edges.

Partitioning a training step by batch index is data parallelism, by a
weight's index weight sharding, and by grid tile *domain decomposition*,
where each GPU owns a tile of the grid and exchanges only its edges with
its neighbours. torch-sql's hypothesis H1 is that a planner choosing the
partitioning per join distributes Samudra better than PyTorch's FSDP
(Fully Sharded Data Parallel). One NVL72 rack holds a whole Samudra run,
so the first distribution stays within a rack.

### 10.6 To kernel compilers and GPU libraries

- **Kernels arrive through bindings** (§5.7), from any compiler. The
  runtime has no kernel language, and no compiler of its own.
- **Array backends** wrap an array library, and lower einfold's nodes
  through it: NumPy and MLX first, then a CUDA backend (§13, question 1).
- **StableHLO and MLIR** appear only inside a backend, as the route to XLA
  through JAX or PJRT, never as XQL's representation of a program (D2).
- **Memory** comes from RMM or the array library's allocator (§7).

---

## 11. Where today's code goes

Each piece of the code that exists today has a home.

<details>
<summary>The table, piece by piece</summary>

| Code | Today | Goes to |
|---|---|---|
| `prepare()` for gradients, `Context(threads=)`, faster registration (#301) | xarray-sql | stays in xarray-sql |
| `SharedExecutor`'s stage caching | xarray-sql (#302) | ddx's SQL path (ddx#168) and the runtime's save points (§6.4) |
| `SharedExecutor`'s SQL parsing of CTEs (`src/shared.rs`) | xarray-sql (#302) | nowhere: ddx#168 does this |
| Arrow tables that declare their sort order (`_SortedArrowTable`) | xarray-sql (#302) | xarray-sql's registration |
| MLX kernels (`xarray_sql/mlx.py`) | xarray-sql (#302) | runtime, as an array backend for einfold's nodes |
| `DenseProgram` | einfold (#157) | a corpus for spike 6, and an oracle for einfold's new detection, which goes in Rust beside its existing detection |
| `DenseExecutor` and its MLX operators | xarray-sql (#303) | runtime, as array backends for B-0 |
| `TrainingSession` | xarray-sql (#305) | split as in §9 |
| `erf` as a DataFusion function (`xarray_sql/math.py`) | xarray-sql (#305) | xarray-sql for now; DataFusion upstream eventually |
| The Shakespeare recipe (`03_shakespeare.py`) | xarray-sql (#305) | xarray-sql, as demo 3, on public APIs |
| `ddxdb.Context`, a `SessionContext` subclass | ddx | ddx's DataFusion extension (§10.2) |
| Cut points and their chain rule (`torch_sql.program`) | torch-sql | nowhere: ddx's SQL path replaces them (§6.4) |

</details>

---

## 12. Building it

### 12.1 Packages

The project is `xql-runtime` (D3, S4.3). It straddles Rust and Python as
DataFusion, ddx and einfold do (D6):

| Channel | Package | Contents |
|---|---|---|
| crates.io | `xql-runtime` | programs, table handles, tiers, the schedule, labels, the `Kernel` trait and the counters, with no engine and no GPU library |
| crates.io | `xql-runtime-datafusion` | the DataFusion extension: the builder method, settings, device memory pools in `RuntimeEnv`, the physical rule, `KernelExec`, `CopyExec`, handle batches, table providers, the codec, and `ddx_ad::Backend`. It pins the DataFusion version xarray-sql, ddx and einfold share. |
| crates.io | `xql-runtime-cuda` | the CUDA array backend and kernel launching, on NVIDIA's `cuda-rust` crates, with cuTile kernels behind a `cutile` feature |
| PyPI | `xql-runtime` | one package built with maturin: the Python binding of all of the above, the `XqlRuntime` extension object for `with_extensions`, Python kernels, and extras `[cuda]` and `[mlx]` (Apple silicon only) |

Rules:
- **The base install has no GPU library** (R6).
- **DataFusion, ddx and einfold are pinned exactly.**
- **Python kernels and array backends are Python code**, behind a Python
  protocol, so that adding one never needs a Rust build.
- **Dependencies point toward the runtime.** Frontends depend on it only
  through an extra, such as `xarray-sql[runtime]`.
- **Linux on ARM wheels from the first release**, for the Beta cluster's
  Grace CPUs.

### 12.2 Milestones

Each milestone is measured on demo 2, demo 4 and torch-sql's nanoGPT, and
reported on [xarray-sql#285](https://github.com/xqlsystems/xarray-sql/issues/285)
with the hardware and versions, whichever way the numbers go. NVIDIA GPUs,
on the cloud or on the Beta cluster, are available half a day a week.

**M0. A CPU runtime, in Rust and Python.**
- *Builds:* `.with_xql_runtime()` and the `XqlRuntime` extension object,
  the `xql_runtime` settings, table handles and providers, `Program` with
  `lower`, `compile` and `run`, `ddx_ad::Backend` with compiled plans and
  tables kept in place, the counters of R8 as metrics, and the schedule
  and `explain()` in their first form, even while every step runs on
  DataFusion. They are cheap to build first and expensive to add later.
- *Done when:* torch-sql's test suite passes on both engines with the
  runtime as its `Backend`, demos 2 and 4 give the same results through
  it, from Python and from Rust, and an override the runtime can't obey
  fails compilation with a reason.

**M1. Resident state.**
- *Builds:* persistent tables, atomic publication, tables that read a
  handle's current buffer (§5.2), device memory pools in `RuntimeEnv`,
  `CopyExec`, the moves `prefetch`, `evict` and `pin`, and asynchronous
  runs.
- *Done when:* torch-sql's nanoGPT training step is one compiled program,
  with no query per tensor, and a failed check in a test leaves every state
  table unchanged.

**M2. A GPU, with B-0** (D4).
- *Builds:* `KernelExec`, handle batches, the CUDA array backend for
  einfold's nodes with NumPy on the CPU as the reference, memory from a
  shared allocator, and kernels bound by label from Python and from Rust.
  MLX follows, or comes first if the contributor's Apple work gets there
  sooner.
- *Needs:* einfold's B-0 interface and fact table, and spikes 6 and 7.
- *Done when:* demo 2 trains on an NVIDIA GPU in strict mode, its state
  never leaving the device between steps, within the bound of §6.3 against
  NumPy, with the share of time in device nodes shown by `EXPLAIN ANALYZE`.
  One node runs a hand-written Triton kernel and another a Rust kernel,
  and checking mode verifies both.

**M3. Save points and memory policies.**
- *Builds:* save points and recompute policies passed to ddx by label, and
  einfold's report on splits (§6.4, §6.5).
- *Done when:* demo 4 trains under a memory budget the session sets, with
  every recompute counted by label.

**M4. Concurrent steps.**
- *Needs:* ddx#157, and the gain measured again on compiled runs.
- *Done when:* concurrent runs give the same results as sequential ones,
  within R9's statement, and the time saved is reported.

**M5. B-1, regions within a step**, built only if spike 6 and M2 show B-0
stops short.

**M6. B-2, regions across steps**, with ddx#157 and ddx#251.

**M7. Distribution, through datafusion-distributed**, within one NVL72
rack.
- *Done when:* demo 4 trains with its grid split by tile across the GPUs
  of one rack, in one physical plan with network nodes and device nodes,
  and gives the same results as on one GPU.

M2 is the critical path. It waits on spikes 6 and 7, and on Alex's
decision on einfold's roadmap. Spikes 2 and 3, which need NVIDIA hardware,
are the first work for the GPU half days, since they wait on no decision.

### 12.3 Spikes

| # | Question | How | Decides |
|---|---|---|---|
| 1 | How far does a fixed-shape tensor column pass through DataFusion's joins and aggregates without being copied or expanded to rows? | Run demo 2's steps with such columns, on host and device buffers | Whether design C is worth pursuing (S2.3) |
| 2 | Do rows between operators cost a GPU SQL engine what they cost GPU kernels? | One of demo 2's steps on Sirius | Whether a GPU engine is an alternative to B |
| 3 | Can a B200 read Arrow buffers in place from Grace memory fast enough? | Read against copy, on the Beta cluster | Whether Grace memory is a tier worth spilling to (§7) |
| 4 | What does crossing from Python cost? | datafusion-ffi's overhead for the runtime's nodes (S4.1); launching a Triton kernel from a `KernelExec` through Python, and whether the call holds Python's lock | Which Python kernels are worth binding, and whether some must be compiled ahead of time |
| 5 | Where does the last 3.76× against PyTorch go? | Trace one warm resident step on the M5 Ultra against PyTorch MPS, by kind of operation, after reading the contributor's own comparison | What M2 and M5 should fix first |
| 6 | How much of each step does einfold already cover? | `EXPLAIN ANALYZE` with einfold on, for demo 2, demo 4 and torch-sql's nanoGPT: the share of time inside einfold's nodes, and the share of steps that are one node from scans to result. einfold's agent runs einfold's side | How far B-0 gets, and whether M5 is needed |
| 7 | Can handle batches carry device data between DataFusion nodes? | Two `KernelExec`s in one plan passing handle batches, through repartitioning and the planner's other rewrites, with a `CopyExec` where rows are needed | Whether §5.3's device plans work as written, or device work must stay inside single nodes |

---

## 13. Open questions

1. **Which libraries run einfold's nodes on NVIDIA GPUs.** From Rust,
   cuTile and `cutile-kernels`; from Python, JAX, PyTorch, CuPy, or PJRT
   directly (S4.2). More than one backend can exist.
2. **Substrait as the plan format between layers.** Engines without a
   Substrait producer need SQL text in their dialect
   ([ddx#159](https://github.com/xqlsystems/ddx/issues/159)), and einfold
   works on DataFusion's logical plans, so a step may need both forms.
3. **Low precision.** Which nodes may compute in bfloat16, a 16-bit float
   format GB200 hardware is built for and SQL engines lack, and who
   chooses.
4. **einfold's roadmap.** M2 needs einfold's B-0 interface and fact table
   ahead of einfold's DuckDB and sparse-kernel milestones. That is Alex's
   decision as einfold's maintainer.
5. **How often checking mode samples** in production, for einfold's nodes
   and for hand-written kernels.
6. **The shared allocator.** This draft leans toward CUDA's
   stream-ordered pools (§7). What is open is how each library in a
   process is configured to draw from them, and what happens when one
   can't be.
7. **The schedule's JSON format**, and how it is versioned as the runtime
   grows (§5.6).
8. **einfold's node types as an MLIR dialect.** Publishing them that way
   would let MLIR-based compilers consume einfold's nodes directly. That is
   a seam to the compiler ecosystem, not a change of XQL's representation
   (D2), and it is einfold's to decide.
9. **NVIDIA's CUDA bindings or cudarc**, for the runtime's own CUDA
   layer. NVIDIA's `cuda-rust` crates are what cuTile is built on, so the
   runtime would share contexts, streams and pools with cuTile directly.
   cudarc is older and more widely used, by Candle among others, and cuTile
   can borrow its handles. This draft leans toward NVIDIA's.
10. **CUDA graphs.** A compiled step could be captured as a CUDA graph, as
    cuTile can, replaying all its launches at once. A graph bakes in its
    buffers, which suits state written in place (§7) and inputs copied into
    buffers allocated once. Worth measuring after M2.
11. **Python kernels on distributed workers.** datafusion-distributed's
   workers are Rust servers. A kernel written in Python needs a worker that
   runs Python, or must be compiled ahead of time, as a Triton kernel can
   be, and launched from Rust.
12. **A Rust frontend.** A backend for Burn whose arrays are XQL tables
    would do for Rust what torch-sql does for PyTorch (S6.6).

---

## Decision log

### Decided

- **D1** (Alex, 2026-11-01, for ddx#251). Recomputing is an explicit
  policy people opt into, never the engine's own choice (§6.5).
- **D2** (Alex, 2026-11-07). StableHLO is not XQL's representation of a
  program. SQL is (§6.2, S2.2).
- **D3** (Alex, 2026-11-07). The project is named `xql-runtime`.
- **D4** (Alex, 2026-11-07). A GPU milestone, with dense intermediates and
  resident state, comes third (M2).
- **D5** (Alex, 2026-11-07). The runtime is a DataFusion extension, and
  depends on DataFusion heavily (R11, §5).
- **D6** (Alex, 2026-11-07). The runtime straddles Rust and Python, as
  DataFusion does, and accelerated ML code should be as easy to write in
  Rust as in Python (R12, §12.1).
- **D7** (Alex, 2026-11-07). Distribution composes with
  datafusion-distributed (§10.5).
- **D8** (Alex, 2026-11-07). No installer of the runtime's own: it attaches
  through DataFusion's builder and `with_extensions` (§5.1).

### Proposed

<details>
<summary>Proposals P-A to P-R, for approval</summary>

- **P-A.** The runtime is a separate project. einfold stays free of device
  execution, and keeps its CPU operator (xarray-sql#307).
- **P-B.** The unit of work is a program with explicit state, lowered and
  compiled as a JAX function is (R3, R5, §5.4).
- **P-C.** Design B, staged from B-0 (§6.2). It needs an einfold design
  change approved by its maintainer. Design C stays open through spike 1.
- **P-D.** The runtime implements ddx's and einfold's public interfaces,
  and depends on their published types, rather than either depending on
  the runtime.
- **P-E.** Where a value is held is the runtime's default, recorded in the
  schedule and open to the developer's override (R7, R10). ddx draws step
  boundaries, and the runtime changes them only through ddx's options and
  the limits its `Backend` reports (§6.4).
- **P-I.** The runtime chooses save points with einfold's report, and
  passes them to ddx as options. No frontend runs a chain rule of its own
  (§6.4).
- **P-J.** A program's reference is the engine it runs on, and
  disagreements between engines are found in tests and filed upstream
  (S4.4).
- **P-K.** One label marker serves regions, policies, provenance, hints,
  bindings and metrics (§5.5).
- **P-L.** Reproducibility is stated per implementation, and a session may
  require it (R9).
- **P-M.** The schedule is the device plans plus what crosses steps, every
  choice in it is visible, overrides are binding, and impossible ones fail
  compilation (R10, §5.6).
- **P-N.** Developers bind their own kernels to labelled nodes, from
  Python or Rust, with the node's SQL as the specification. einfold's
  nodes are bindings made automatically (§5.7).
- **P-O.** Memory tiers extend `RuntimeEnv`, named by Arrow's device types,
  with device memory accounted as DataFusion accounts host memory (§5.2).
- **P-P.** The runtime builds no compiler, allocator, communication library
  or distributed planner (§3.2, R11).
- **P-Q.** A device plan is a DataFusion physical plan with `KernelExec`
  and `CopyExec` nodes, passing handle batches between device nodes
  (§5.3, spike 7).
- **P-R.** Every XQL project attaches to DataFusion the same way: a builder
  method in Rust, and an extension object for `with_extensions` in Python
  (§5.1, §10.2, §10.3).

</details>
