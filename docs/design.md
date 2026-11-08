<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# DataFusion Accelerated: design

_author_: Alex Merose
_co-authors_: ⚡ Claude (datafusion-accelerated), formerly 🔥 Claude (torch-sql), and 🌀 Claude (xarray-sql), via Claude Code
_reviewers_: 🪭 Claude (einfold) and ∂ Claude (ddx). Their reviews of v1.1 are in [reviews/](reviews/).
_status_: Draft v1.9, for approval. Nothing is built. Collapsed sections (▸) hold background and detail.
_last updated_: 2026-11-07
_supplement_: [supplement.md](supplement.md), with the evidence, the alternatives considered and the reasoning behind each rule. Citations such as "(S1.1)" point there. S6 compares the design with JAX, PyTorch and DataFusion, and S7 covers distributed training.
_origin_: [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)
_references_: R1 to R13 are the principles (§4), D1 to D13 decisions and P-A to P-Y proposals (the decision log), M0 to M8 milestones (§12.2), and spikes 1 to 13 are experiments run to settle a design question, each with a report (§12.3). A bare "#n" is an issue in this repository; issues elsewhere carry their repository's name, such as ddx#251.

---

## 1. What we're building

**DataFusion Accelerated** (`datafusion-accelerated`) runs
[Apache DataFusion](https://datafusion.apache.org/)'s physical plans on GPUs
and other accelerators. DataFusion is a SQL query engine in Rust, designed
to be extended (§2.2), and DataFusion Accelerated is one of its
extensions. Any node of a physical plan, whether one of DataFusion's
relational operators or the array work in a query, can have a device
implementation, checked against DataFusion's own. This doc calls it *the
runtime* for short.

It is an interstitial layer. Hardware, and the libraries built on it, come
in through *integrations*, each a separate crate or package mixed in as
an extension of its own (§5.8). They cover GPU platforms such as NVIDIA's
CUDA, AMD's ROCm and Apple's Metal, and Google's TPUs; libraries that move
data between devices, such as NVIDIA's NCCL; libraries of GPU operators,
such as libcudf; and whatever replaces them. The runtime integrates none
of them itself, so its job outlives every one.

DataFusion Accelerated is not part of the Apache DataFusion project.
"Apache DataFusion" is a trademark of the Apache Software Foundation.

### 1.1 Its shape

The runtime extends DataFusion the way
[datafusion-distributed](https://github.com/datafusion-contrib/datafusion-distributed)
does. datafusion-distributed adds distributed execution to an ordinary
DataFusion session with one call on its builder, and a distributed plan is
an ordinary physical plan with a few extra nodes that move data between
machines, in Apache Arrow, the columnar format DataFusion uses (§2.1).
The runtime adds devices the same way: one call on the builder, and a
device plan is an ordinary physical plan with a few extra nodes that move
data between memory tiers, such as host memory and a GPU's, and run work
on a device (§5).
Nothing a DataFusion user knows changes, and the two extensions compose: a
plan can have network nodes and device nodes at once (§10.5).

It does for DataFusion what two existing projects do for other engines.
Spark RAPIDS is NVIDIA's plugin that runs Apache Spark's operators on GPUs,
and Sirius runs DuckDB's queries on GPUs (§2.3). Each swaps an engine's
operators for device ones, with explicit transitions between them, and
keeps the engine's own operators as the fallback and the reference
(S4.3).

It adds one thing DataFusion has no layer for: a *program*, several
queries run as one unit, many times, with state that persists from run to
run, as a training step or an iterative algorithm needs (§5.4).

### 1.2 Why

DataFusion can't run a plan on a GPU today. It has no device memory to
keep tables in between operators or between runs, and no way to hand a
node to a device library and check the answer. The measurements of its
first user (§1.4) show what matters once it can: how data is held and moved
between queries, more than planning or arithmetic (S1). On a
123.6M-parameter GPT written in SQL, sending matrix products to a GPU
changed nothing, while holding intermediates as dense arrays made the step
67 times faster. Keeping weights and optimizer state on the GPU then took a
smaller model's step from 2.02 s to 101 ms, against PyTorch's 27 ms
(S1.1). Without a shared layer, each project builds this for itself, as
XQL's projects already have (S1.1, S1.3).

### 1.3 Its values

- **One job, and delegation.** The runtime is the layer between
  DataFusion's plans and accelerator primitives. It builds no compiler,
  allocator, communication library or kernel library of its own, and
  makes the ones that exist trivially easy to integrate. It is one
  component of a composable data system, a *deconstructed tensor runtime*.
  The phrase follows Wes McKinney's *deconstructed database*, a database
  assembled from separate components that share open standards, such as
  Arrow (R11, R13, S5.4).
- **Every control within reach.** The SQL says what a program computes.
  Every physical choice the runtime makes, such as where a table lives and
  which kernel computes it, is visible and can be overridden, so the
  runtime is never a compiler its users have to outsmart (R10, S5.1).
- **Rust and Python.** Everything the runtime offers works from both, as
  DataFusion does, so that accelerated code is as easy to write in Rust as
  in Python (R12).

### 1.4 Its first user: XQL

**XQL** is a group of open-source projects, under the
[xqlsystems](https://github.com/xqlsystems) organization, for computing on
scientific arrays with databases. An N-dimensional array becomes a table in
*long format*: one row per element, with one column per dimension holding
the element's position and one column holding its value. Array programs
then become SQL queries, and a database's query optimizer plans them.
XQL is organized as layers, each a separate, optional project, and
DataFusion Accelerated is its runtime layer. *Frontends* turn programs
into SQL: xarray-sql from Xarray datasets, and torch-sql from PyTorch
programs. *Transforms* rewrite the SQL: ddx differentiates it, and einfold
finds array computations in it. *Engines* run it: DataFusion, DuckDB, GPU
SQL engines such as Sirius and NVIDIA's GQE, and any database reached
through ADBC, Arrow's database API. §2 introduces each.

```
frontends:   SQL text · xarray (xarray-sql) · PyTorch (torch-sql)
transforms:  ddx (differentiation) · einfold (plan rewrites)
runtime:     DataFusion Accelerated: execution, device residency, kernels, memory, scheduling, distribution
engines:     DataFusion · DuckDB · GPU engines (Sirius, GQE) · any ADBC database
```

Training XQL's models, the ocean emulator Samudra first (§2.6), shaped the
design, and its milestones are measured on XQL's programs (§12.2). But XQL
is a user, not a dependency: ddx and einfold come in through integrations,
as hardware does (§5.8), and a DataFusion user with neither gets the rest.

---

## 2. Background

The design's terms are introduced here, in collapsed groups. Open the
ones you need.

<details>
<summary><b>2.1 Formats and terms</b>: Arrow, ADBC, Substrait, DLPack, Zarr, CTEs, compact arrays</summary>

- **[Apache Arrow](https://arrow.apache.org/)** is a columnar memory
  format, which DataFusion and every XQL engine read and write. Its *PyCapsule
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
  dimension's coordinates stored once, dense arrays of values, and masks
  for which rows exist and which values are NULL. The runtime stores one
  as dense *blocks*, and leaves out the blocks with no rows (§5.2).

</details>

<details>
<summary><b>2.2 DataFusion</b>: the layers the runtime extends, datafusion-distributed, Ballista</summary>

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
distributed by registering its codec on every session. **Ballista** is an
older distributed scheduler for DataFusion, within the Apache project.

</details>

<details>
<summary><b>2.3 Other engines</b>: DuckDB, Sirius, GQE</summary>

- **[DuckDB](https://duckdb.org/)** is an embedded analytical database. ddx
  runs on it through DuckDB's community Substrait extension.
- **[Sirius](https://github.com/sirius-db/sirius)** and NVIDIA's **GQE** are
  SQL engines that run on GPUs.

</details>

<details>
<summary><b>2.4 XQL's frontends and transforms</b>: xarray-sql, torch-sql, ddx, einfold, the performance work</summary>

- **[xarray-sql](https://github.com/xqlsystems/xarray-sql)** registers
  [Xarray](https://xarray.dev/) Datasets as tables, and turns results back
  into Datasets. Its demos are each checked against PyTorch or JAX: demo 1
  is attention, demo 2 a GPT trained in SQL, demo 3 nanoGPT's Shakespeare
  recipe, and demo 4 Samudra's U-Net (both models are in §2.6).
- **[torch-sql](https://github.com/xqlsystems/torch-sql)** runs
  [PyTorch](https://pytorch.org/) programs as SQL: a tensor on
  `device="sql"` is a table. It trains nanoGPT (§2.6). Its design doc's
  spikes are cited here as "torch-sql's spike S4" and so on, and its
  hypotheses as "torch-sql's H1".
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
- **The performance work** is one contributor's experiments in xarray-sql
  and in a fork of einfold, on an Apple GPU, which this doc calls *the
  contributor's* (S1.1). They built `SharedExecutor`, which computes each
  CTE once per run (xarray-sql#302); `DenseExecutor`, which keeps
  intermediates as compact arrays between operators (xarray-sql#303); and
  `TrainingSession`, which keeps a training step's state on the GPU and
  compiles the whole step (xarray-sql#305). `DenseExecutor` runs on einfold's
  `DenseProgram`, an interpreter of DataFusion's plans on compact arrays
  (einfold#157). Together they are *the contributor's MLX path*.

</details>

<details>
<summary><b>2.5 Kernels, compilers and GPU libraries</b>: Triton, Gluon, Pallas, cuTile Rust, CUTLASS, XLA, RMM, NCCL, CuPy</summary>

The runtime builds none of these, and depends on no kernel library.
Developers bring their kernels, and the kernel contract (§5.7) is designed
so that each library's kernels bind without help. S6.2 checks the contract
against each.
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
- **NCCL** is NVIDIA's library for moving data between GPUs, and **NVML**
  its library that reports a machine's GPUs and their links.
- **CuPy** is NumPy's API on NVIDIA GPUs, and **Candle** Hugging Face's
  machine-learning framework in Rust.

</details>

<details>
<summary><b>2.6 Models and hardware</b>: Samudra, nanoGPT, MLX, the Beta cluster, AMD and TPU clusters</summary>

- **[Samudra](https://github.com/m2lines/Samudra)** is an ocean emulator and
  XQL's first production model: a *U-Net*, a convolutional network that
  shrinks a grid level by level and grows it back, built from *ConvNeXt*
  blocks of dilated convolutions, batch norms and GELUs, on a half-degree
  grid of 360 × 720 cells.
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** is Andrej Karpathy's
  small GPT training code, whose Shakespeare recipe trains a 10.7M-parameter
  model. **Modded-NanoGPT** is a community speedrun built on it: a
  124M-parameter GPT trained on web text at context 1,024, with Muon, an
  optimizer that orthogonalizes each update.
- **MLX** is Apple's array library for its GPUs. PyTorch runs on them
  through its *MPS* backend. The contributor's measurements ran on an
  Apple M5 Ultra.
- **The Beta cluster** of
  [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
  is where Samudra trains: 288 NVIDIA B200 GPUs (compute capability
  `sm_100`) in GB200 NVL72 racks. Each Grace CPU, an ARM CPU, is joined to
  two B200s by NVLink-C2C, which lets a GPU read the CPU's memory directly,
  and NVLink joins the 72 GPUs of a rack.
- **Other clusters** test the design's portability (S7.8). **AMD**'s
  MI300-class nodes join 8 GPUs in a full mesh of Infinity Fabric links,
  programmed through *ROCm*, whose *HIP* mirrors CUDA and whose *RCCL*
  implements NCCL's API. **Google's TPUs** join chips in a 2D or 3D torus,
  with wraparound, over links called *ICI*, and are programmed only
  through XLA, by way of PJRT.

</details>

---

## 3. Goals and non-goals

### 3.1 Goals

1. **Run DataFusion's plans on devices**, for any DataFusion user: plain
   queries, and programs, such as ddx's gradient programs and XQL's
   training steps.
2. **Own tables and their memory.** Every table the runtime holds has an
   owner, a lifetime and a placement.
3. **Keep data where the computation is**, between operators and between
   steps, so a training loop moves only its batches and the values it
   reads.
4. **Choose an implementation for each piece of work**: DataFusion's own
   operators by default, and kernels for work einfold has marked, for
   DataFusion's relational operators, or for anything a developer has
   bound. Array work comes first, since XQL's models need it.
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
10. **Make hardware trivially easy to integrate**, through integrations that
    implement the core's interfaces (§5.8), and integrate none in the
    core.

### 3.2 Non-goals

- **An SQL engine of its own.** The runtime never interprets relational
  operators itself: no joins, no `GROUP BY`, no NULL rules of its own.
  DataFusion runs those, and where they run on a device, a library such as
  libcudf, the C++ core of NVIDIA's GPU dataframe library cuDF, provides
  the kernels, bound like any other and checked against DataFusion's
  (§5.7, S2.1).
- **Differentiation, plan rewriting, or drawing a program's step
  boundaries.** Those are ddx's and einfold's (§6.4).
- **Any particular hardware, in the core.** Integrations with CUDA, ROCm,
  Metal, PJRT, NCCL or libcudf are separate crates and packages (§5.8).
- **A kernel compiler, a memory allocator or a communication library.**
  Kernels come from Triton, Gluon, Pallas, cuTile Rust, CUTLASS, XLA,
  `torch.compile`, MLX, CubeCL or a developer's own code. GPU memory comes
  from a shared allocator, and data moves between GPUs through a transport
  integration such as NCCL.
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
- **R10. Defaults, not hidden decisions.** Every physical choice is in the
  schedule, which a developer can read (§5.6). The runtime fills in
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
- **R13. The core delegates to hardware.** The core defines interfaces and
  the DataFusion integration, and depends on no hardware or vendor
  library. Every integration knows the core; the core knows no
  integration. Replacing one, as libcudf might be by a successor, changes
  only its own crate (§5.8).

---

## 5. The model

### 5.1 The session

An accelerated session is a DataFusion session with the runtime's
extension added. Every extension attaches the same way, the runtime, its
integrations and XQL's ddx and einfold alike, and none has an installer of
its own (S6.7).

In Rust, each adds a method to `SessionStateBuilder`, as
datafusion-distributed's `.with_distributed_planner()` does:

```rust
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_ddx()                              // grad, vjp and jvp in SQL (§10.2)
    .with_einfold()                          // einfold's detection (§10.3)
    .with_accelerator(AcceleratorOptions::new())
    .with_accelerated_cuda()                 // an integration, mixed in (§5.8)
    .build();
let ctx = SessionContext::new_with_state(state);
```

In Python, each provides an extension object for
`SessionContext.with_extensions`, as einfold's `EinFold` already is:

```python
ctx = SessionContext().with_extensions(Ddx(), EinFold(), Accelerator(), Cuda(devices="local"))
```

Rules:
- **Settings live in the `accelerated` namespace** of `SessionConfig`, so
  SQL can set them: `SET accelerated.strict = true`,
  `SET accelerated.checking.sample = 0.01`.
- **The runtime's state lives in DataFusion's.** Its device memory and
  device handles are in the session's `RuntimeEnv` (§5.2), and its nodes
  find them through the `TaskContext` DataFusion gives each running node.
- **Frontends may assemble the session.** xarray-sql's context adds
  whichever XQL extensions are installed, so that, with ddxdb installed,
  its SQL understands `grad` with no other step.
- **Workers are sessions too.** Under datafusion-distributed, each worker
  builds its session with the same extensions (§10.5).

### 5.2 Memory tiers

The runtime extends `RuntimeEnv`'s memory hierarchy. DataFusion already has
two tiers: host memory, through its `MemoryPool`, and disk, through its
`DiskManager`. The runtime adds the device tiers. Tiers are named by device family:
by Arrow's device type where one exists (§2.1), by a name the device
integration defines where none does, and `storage` for files
([#20](https://github.com/xqlsystems/datafusion-accelerated/issues/20)).

| Tier | Device type | Accounting |
|---|---|---|
| device memory | `CUDA`, `ROCM`, `METAL`; the TPU integration's name for a TPU | a device memory pool per device, added by the runtime |
| pinned host memory | `CUDA_HOST`, `ROCM_HOST` | the device pool of the device it is pinned for |
| host memory | `CPU` | DataFusion's `MemoryPool` |
| storage | none | DataFusion's `DiskManager`, plus Zarr and Arrow IPC files |

Rules:
- **Device memory is accounted as DataFusion accounts host memory.** A
  node reserves bytes from the device pool before it allocates, and a
  refused reservation means spilling to a lower tier, as a refused host
  reservation means spilling to disk.
- **The bytes come from a shared allocator**, never from an allocator of
  the runtime's own (§7). On a TPU, PJRT owns device memory, and the
  runtime accounts for what it holds through PJRT's buffers.
- **A TPU's tables cross into DataFusion only through the host.** Neither
  Arrow nor DLPack has a TPU device type, so device data on a TPU stays
  behind PJRT's handles.
- **A table handle** refers to a table the runtime owns, as a buffer handle
  refers to an allocation. It carries the table's schema, its lifetime (one
  run, or until the caller releases it), and one or more *instances*. Each
  instance has a tier; a layout, either Arrow rows or a compact array; and
  an owner, either DataFusion (as one of its tables) or the runtime.
- **Runtime tables are DataFusion table providers**, read as Arrow batches
  by DataFusion's operators, and as compact arrays by device nodes that
  scan them directly (§5.3).
- **A table's layout is chosen by what it would store** (S2.7). A
  compact array stores every cell of its block, so an absent cell costs
  its storage and its arithmetic, as a masked value. Only rows make
  absence free, and rows are the slow path. So:
  - **blocks are the compact layout**, and a dense array is the case of
    one block. Blocks with no rows aren't stored;
  - **a table one of einfold's nodes reads or writes is held as blocks**
    when einfold's fact table (§6.2) gives its domain. Other tables stay
    as rows;
  - **the block size is chosen by stored cells per present cell**, which
    measures how clustered absence is, as density alone doesn't. It is a
    measured fact about the domain, computed from its mask when the table
    is registered;
  - **under a memory budget** the runtime prefers smaller blocks, then
    rows, before it spills;
  - **each instance's layout and block size are schedule entries** (§5.6),
    shown by `EXPLAIN` with the ratio, and a developer can override them.
- **Developers may move tables themselves**: `prefetch` to a tier, `evict`
  from one, and `pin` an instance so that it stays. Each tier reports its
  capacity and what it holds.
- **Tables are values, as in SQL.** A buffer is written in place only when
  state is published (R5, §7), so aliasing never reaches a program's
  meaning (S5.6).
- **Reading a handle's rows is the only way data leaves the runtime**, and
  it is always explicit.
- **Names are stable.** Each handle has a stable name, by which plans read
  it. With ddx, the ddx integration passes it a namespace
  (`Options::namespace`), so that the same plan gives the same program byte
  for byte (S3.3).
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
- **Device segments are opaque to DataFusion's optimizer.** A run of
  device nodes is a *segment*. A handle batch's schema hides the logical
  columns, so DataFusion can't reason about the data inside a segment,
  and three rules keep its operators out (spike 7, S6.5):
  - device nodes report that they don't benefit from more input
    partitions (`benefits_from_input_partitioning` is `false`), so
    DataFusion's optimizer puts no repartition between them;
  - the runtime's placement rule runs after DataFusion's rules, as a rule
    added to the session builder does, so nothing rewrites a segment once
    it is placed. The rule then checks the plan's invariants itself, since
    DataFusion's `SanityCheckPlan` has already run;
  - inside a segment, every exchange is the runtime's own node, never
    DataFusion's `RepartitionExec`, which would hash the handles. A
    segment's last node restates the logical partitioning and ordering for
    the host plan above it.
- **Partitions map to the hardware.** On one GPU, a node's partitions run
  on separate queues, such as CUDA or HIP streams. Across GPUs in one machine, each partition
  runs on one device.
- **`EXPLAIN` shows the device plan**, with each runtime node's device,
  layout, kernel, and who chose it: a default, einfold or the developer.
  **`EXPLAIN ANALYZE`** adds each node's metrics: time in the kernel, bytes
  copied, peak device memory and declines.
- **The runtime's nodes have a codec**, so that datafusion-proto can
  serialize a device plan and datafusion-distributed can ship one (§10.5).

### 5.4 Programs

A **program** is a directed acyclic graph of **steps**. Each step is a
query, planned by DataFusion as a device plan. One *run* of a program is
usually one training step, so this doc says "training step" for a run and
"step" for one of its queries. A program names:

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

A compiled program is reused only while the session's *session key* is
unchanged. ddx's `Backend` reports that key, which changes whenever the
session's tables or settings change, and ddx's `prepare`, which plans a
derivative once for many runs, follows the same rule.

Rules:
- **A training step can be one statement.** With the ddx integration in
  the session (§5.8), `SELECT k, v - $lr * grad(loss, v) AS v FROM w` is a whole step
  of gradient descent. ddx writes the program from it.
- **A program may have several statements, by name.** `Program(ctx, STEP,
  ...)` takes one statement, whose result is the next value of the
  program's one state table. `Program(ctx, {"grads": GRADS_SQL, "w":
  UPDATE_SQL}, ...)` takes several. Each statement's result is a table
  that later statements read by its name, the order comes from those
  reads, and a statement named after a state table computes that table's
  next value. Outputs name a statement or a label (§5.5). In Rust,
  `Program::from_steps(&ctx, [("grads", GRADS_SQL), ("w", UPDATE_SQL)])`.
- **Programs compose.** A program may also be ddx's program followed by
  update steps that read its gradient tables, written as statements by
  name. The runtime takes both as one program.
- **ddx's checks, where there are any, run when they must.** Each states a promise: that some key
  columns of one table are unique. The runtime runs a check once per
  compile for a table whose keys don't change, on each run otherwise, and
  never when a frontend's fact gives the same promise.
- **einfold's checks run on every run**, as data checks on the values of
  that run (§6.3, S3.1).
- **Parameters are written as DataFusion's placeholders**, such as `$lr`.
  Where a placeholder can't reach ddx through Substrait, the runtime turns
  it into a one-row table that the plans cross join (S3.2).

### 5.5 Labels

One marker names a part of a program for every layer: a function call,
`ddx_name(expr, 'label')`, as proposed on ddx#251, since CTE and view names
don't survive the engines' Substrait producers (torch-sql's spike S3).
ddx#251 also proposes *regions*, labelled parts of a program that a policy
applies to as a whole. The
core registers the marker itself when ddx isn't in the session, so labels
work without ddx, and ddx reads the same marker when it is. In ddx's SQL
path, CTE names serve too. Labels carry:
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
- **Per program, it adds** what no single plan holds: which device holds
  each instance, and each partition, of each table between steps (the
  partitioning itself is declared in SQL, §10.5), save points and
  recompute policies passed to ddx (§6.4, §6.5), and *donation*, writing a
  state table's new value into its old memory (§7).

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

A **binding** says which kernel computes a node. Bindings come from three
places:
- einfold's detection, which binds array work to the runtime's array
  backends automatically (§6.2);
- **relational operators**: a binding by node type, such as DataFusion's
  hash join or hash aggregate, to a library of relational kernels such as
  libcudf, as Spark RAPIDS and Sirius bind theirs
  ([#24](https://github.com/xqlsystems/datafusion-accelerated/issues/24));
- a developer, or an agent, by hand, by label.

Whichever it is, the node's SQL, or for a relational operator
DataFusion's own operator, is the kernel's specification, and runs if the
kernel declines. A relational kernel's NULLs are Arrow's validity masks,
so its three cell states are the plan's own.

**The kernel contract.** The runtime depends on no kernel library.
Instead it gives every kernel what any library needs to run on memory and
queues it doesn't own, in terms each device family can meet
([#20](https://github.com/xqlsystems/datafusion-accelerated/issues/20)). Each item below is what a kernel library asks of
a foreign host, checked against cuTile Rust, Triton and Pallas (S6.2):
- **Arrays the kernel can borrow.** Each input, and each output the
  runtime has allocated from the node's schema, is a device array with a
  pointer, shape, strides, element type, named axes and an *owner*: a
  reference-counted handle that keeps the runtime's buffer alive for as
  long as the kernel's library holds it. Each exports DLPack where its
  device has a DLPack type, and in Python a `data_ptr()`. On a TPU, arrays
  are PJRT buffers.
- **Masks.** Each input's presence and NULL masks, unless the kernel
  assumes there are none.
- **The execution context and queue, borrowable.** The device's raw
  context and the partition's queue, each with an owner, so a library
  launches on the runtime's queue rather than its own: a CUDA or HIP
  context and stream, or on a TPU, a PJRT client and device.
- **Async launches.** A launch is awaited on the Tokio runtime DataFusion
  executes on, so a library whose launches are futures awaits them.
- **A compile hook**, run at the program's compile with the shapes and
  types every run will have, so JIT compilation and autotuning never
  happen during a run.
- **The partition's placement**: which device and stream each partition
  runs on, so a library with its own scheduler can follow DataFusion's
  partitions instead of choosing for itself.

A binding is one of two kinds, by device family:
- **a callable**, launched on the queue: CUDA and ROCm;
- **a function the array backend compiles in**, to the program of the
  region that contains the node (§6.2): a Pallas kernel on a TPU, which
  runs only inside an XLA program.

Binding a library's kernel then takes a few lines of the user's own code,
and none in the runtime.

The kernel chooses its own grid, blocks and on-chip memory. Shared memory,
tensor memory and registers are below the runtime's tiers, and belong to
the kernel (S6.2). Tuning parameters a kernel chooses, such as Triton's
block sizes and warp counts, are recorded in the schedule, so that a
compiled program keeps them and a developer can set them.

**From Python**, kernels are registered on the runtime's extension object
before the session is built, so the session has them from the start. A
Python kernel's compile hook is passed as `prepare=`, beside the kernel.

<details>
<summary>A Triton kernel and a Pallas kernel, bound by label</summary>

```python
runtime = Accelerator()

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
on the session builder. `KernelArgs` carries the contract above:

```rust
#[async_trait]
pub trait Kernel: Send + Sync {
    /// Once per shape, when the program compiles: JIT-compile, autotune, plan scratch space.
    async fn prepare(&self, signature: &Signature) -> Result<()> { Ok(()) }
    /// On every run, on the partition's stream, into outputs the runtime allocated.
    async fn launch(&self, args: KernelArgs<'_>) -> Result<()>;
}
```

<details>
<summary>Binding a cuTile Rust kernel: the user's code, not the runtime's</summary>

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
    async fn prepare(&self, sig: &Signature) -> Result<()> {
        // cuTile's .compile() on placeholder tensors of sig's shapes
    }
    async fn launch(&self, args: KernelArgs<'_>) -> Result<()> {
        // The glue: borrow the runtime's stream and buffers. cuTile keeps each owner alive.
        let stream = unsafe { Stream::borrow_with_owner(args.stream().raw(), &args.context().borrow(), args.stream().owner()) };
        let x = unsafe { Tensor::<f32>::from_foreign(Borrowed::new(args.input("x")?), ..) };
        let mut out = unsafe { Tensor::<f32>::from_foreign(Borrowed::new(args.output()?), ..) };
        kernels::scale((&mut out).partition([128]), &x, 2.0).sync_on(&stream)?;
        Ok(())
    }
}

builder.with_accelerator(AcceleratorOptions::new().kernel("scale", Scale)).with_accelerated_cuda()
```

`Borrowed` is a few lines that implement cuTile's trait for foreign
allocations over the runtime's device array. The one thing the runtime
can't remove is that kind of glue: cuTile, cudarc and others each define
their own trait for "memory someone else owns", and no Rust standard
spans them. DLPack is the standard in Python, and the runtime exports it
in Rust too for libraries that read it.

</details>

A kernel compiled elsewhere, from CUDA C, CUTLASS or Triton ahead of time,
is loaded as PTX or a cubin and launched on the borrowed stream, through
whichever CUDA bindings the user prefers (§13, question 9).

Rules:
- **`assumes` lists the facts a kernel relies on** (§6.2). The runtime
  checks each on every run, unless a frontend's fact states it.
- **A kernel that returns a new array**, as a Pallas kernel through JAX
  does, has its buffer adopted through DLPack when it came from the shared
  allocator, and copied otherwise.
- **Hand-written kernels are tested, not proved.** einfold's nodes are
  proved exact from the plan's structure (§6.2). A developer's kernel is only
  tested against its SQL, which is weaker (S5.3). So checking mode samples
  hand-written kernels more often, their runs are counted separately, and
  a session may refuse them in strict mode.

### 5.8 Integrations

The core defines what each kind of hardware or library must provide, and
implements none of them for any hardware (R13). Every integration is a
separate crate in Rust, or package in Python, that registers itself with
the accelerator as a DataFusion extension of its own, mixed in beside it.

| Interface | What it provides | Integrations, for example |
|---|---|---|
| `Device` | memory tiers and their shared allocator, queues and events, and the execution context of §5.7 | CUDA, ROCm, Metal, PJRT for TPUs |
| `ArrayBackend` | runs einfold's nodes (§6.2) on arrays | NumPy (the reference), MLX, CuPy, JAX, PyTorch |
| `KernelLibrary` | kernels bound by node type, such as DataFusion's relational operators | libcudf, and whatever succeeds it |
| `Kernel` | one kernel bound by label (§5.7) | Triton, Gluon, Pallas, cuTile Rust, a developer's own |
| `Transport` | the collective capabilities it implements, eager or compiled (§10.5) | NCCL, RCCL, XLA's collectives, and the gRPC network calls datafusion-distributed uses |
| `Topology` | devices and links, as discovered (§10.5) | NVML and NCCL, AMD's tools, PJRT |

Other DataFusion extensions integrate the same way, and XQL's are the
first:
- **ddx**, through `datafusion-accelerated-ddx`, which implements ddx's
  `Backend`, so ddx's programs run as the runtime's programs, and passes
  ddx the runtime's save points, policies and engine limits (§6.4, §10.2);
- **einfold**, through `datafusion-accelerated-einfold`, which turns the
  nodes einfold detects into bindings to array backends (§6.2, §10.3).

In Rust, each integration adds a builder method, and in Python an
extension object:

```rust
let state = SessionStateBuilder::new()
    .with_default_features()
    .with_accelerator(AcceleratorOptions::new())
    .with_accelerated_cuda()       // datafusion-accelerated-cuda: devices, memory, streams
    .with_accelerated_nccl()       // datafusion-accelerated-nccl: a transport
    .with_accelerated_cudf()       // datafusion-accelerated-cudf: relational kernels
    .build();
```

```python
ctx = SessionContext().with_extensions(Accelerator(), Cuda(devices="local"), Nccl(), Cudf())
```

Rules:
- **The core runs with no accelerator.** It ships a CPU reference
  implementation of every interface, and its whole test suite runs on it,
  in CI with no GPU.
- **Integrations declare their capabilities**: the tiers, collectives,
  node types and precisions they support. The planner uses only what is
  declared, and a declared capability that fails at run time is a decline
  (R8).
- **Integrations are pinned by the user's project**, not by the core. The
  core depends on DataFusion alone, and pins it exactly.
- **An integration in Python alone needs no Rust build**, as an MLX array
  backend or a set of Triton kernels doesn't.
- **This repository may host some integrations**, as separate crates, for
  convenience and continuous integration. That is a choice of where code
  lives, not a dependency of the core.

---

## 6. Execution

### 6.1 Three ways to run a step

The schedule says, per step, which way it runs. By default the runtime
chooses:

1. **On DataFusion**, as a plain query. This is the reference (R1), and
   always available.
2. **With bound nodes.** Each node einfold marked, a kernel library bound
   by its type, or a developer bound by label, runs as a `KernelExec` on
   data kept on the device, or declines, and then its subplan runs on
   DataFusion. With einfold's nodes, this is the stage of dense execution
   that §6.2 calls B-0.
3. **As a region**: several nodes, and the elementwise expressions between
   them, run as one array program, with their intermediates kept compact.
   These are stages B-1 and B-2 (§6.2).

A GPU SQL engine such as Sirius would be a fourth way, outside DataFusion.
Spike 2 tests whether it meets the same wall as GPU kernels on long-format
rows (S1.1).

### 6.2 Dense execution

*Dense execution* runs the array work in a step's SQL on compact arrays.
The supplement weighs three designs for it (S2):
- **A**: the runtime interprets relational plans on compact arrays, a
  second implementation of SQL (S2.1);
- **B**: einfold marks the array work in the plan, and the runtime runs
  what einfold marks;
- **C**: DataFusion carries dense tiles itself, as tensor columns (S2.3).

**The design is B: einfold marks the SQL, and the runtime runs what it
marks.** The program stays SQL: it is what every engine runs, what ddx
differentiates, and what R1 calls the meaning. There is no separate array
language (D2, S2.2). einfold annotates a step's plan with nodes of five
kinds:
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
  A region may contain exchanges, which a compiled transport lowers into
  the region's program (§10.5). A TPU's array backend needs regions to be fast,
  since its collectives exist only inside compiled programs.
- **B-2, regions across steps**, through einfold's *program mode*, in
  which einfold sees a whole program rather than one query at a time, with
  ddx's step graph, which says which steps read which (ddx#157), and its
  regions (ddx#251).

B-0 comes first as the first interface, not as enough on its own. The
evidence expects it to stop short (S2.6), so B-1 and B-2 are planned (M5,
M6), and spike 6 measures by how much.

B pulls einfold's fact table and program mode ahead of its DuckDB and
sparse-kernel milestones. That is Alex's decision as einfold's maintainer,
after a sketch in einfold's design doc (§13, question 4).

Design C is closed for the milestones: spike 1 found that DataFusion
copies tensor columns wherever it selects or reorders rows, and can't
compute on them (S2.3).

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

The rules in this section and the next apply to programs that ddx writes,
through the ddx integration (§5.8). A program written as plain steps draws
its own boundaries.

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
  form of ddx#168, which splits a query into steps at each CTE it reads
  more than once, and has no implementation of its own.
- **Save points are schedule entries** (§5.6), so a developer can add,
  remove or move one, and the runtime passes the result to ddx.
- **The runtime chooses default save points with einfold.** einfold
  reports whether a candidate save point would split a node it can fuse,
  and the runtime prefers save points that don't. A split taken anyway is
  a counted decline (S3.5).
- **Frontends don't run their own chain rules.** torch-sql today splits a
  program at *cut points* and chains the pieces' derivatives itself. It
  retires them by handing ddx SQL with a CTE at each shared node (S3.6), and
  stops gradients with `ddx_stop_gradient`, never by saving a table
  ([ddx#260](https://github.com/xqlsystems/ddx/issues/260)).

### 6.5 Recomputing

- **Recomputing is a policy the user opts into** (D1), by label. The
  vocabulary follows JAX's: `save_only_these_names(...)`, and an offload
  policy that moves a value to `CUDA_HOST` rather than recomputing it
  (S6.1). The runtime applies a memory-budget policy only when the session
  turns one on, and may always save more than a policy asks.
- **A value several steps of one run share is held, not recomputed by
  choice.** When a run's steps compute the same value, such as a forward
  product its backward steps read again, it is computed once and kept
  until the run ends. einfold proves two values identical, in its program
  mode (B-2), and the runtime holds the value as a table instance in the
  schedule (§5.6). Under memory pressure the runtime spills it to a lower
  tier, as DataFusion spills, and recomputes it only where a recompute
  policy names its label (P-X). The contributor's cache of these values
  recomputes after evicting them, which D1 rules out (S1.1).
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
  shared allocator, so that it shares a GPU with PyTorch, JAX and kernel
  libraries instead of competing with them for it (S4.2). Which allocator
  is shared is each device integration's choice (§5.8). For CUDA, this
  draft leans toward CUDA's stream-ordered pools, since RMM, PyTorch and
  kernel libraries such as cuTile Rust can each draw from them (§13,
  question 6).
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
  `materialize`, the call that runs one step and saves its table, may run
  concurrently. Measure the gain again on compiled
  runs before building it (S1.4).
- **Across runs**, the runtime loads the next batch while a step runs. For
  Samudra, batches come from xarray-sql's data path over a Zarr archive.

---

## 9. State, checkpoints and randomness

State that persists across runs (§5.4) needs checkpoints, and training
needs random numbers every implementation agrees on.

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

A query, from Python, with relational operators on a GPU. The runtime's
names are placeholders:

```python
from datafusion import SessionContext
from datafusion_accelerated import Accelerator
from datafusion_accelerated_cuda import Cuda
from datafusion_accelerated_cudf import Cudf

ctx = SessionContext().with_extensions(Accelerator(), Cuda(devices="local"), Cudf())
ctx.register_parquet("trips", "trips/")
ctx.sql("EXPLAIN ANALYZE SELECT zone, AVG(fare) FROM trips GROUP BY zone").show()
# the scan's batches are copied to the GPU once; libcudf's hash aggregate runs there,
# DataFusion's is its fallback and reference, and the plan shows both
```

A training step, with XQL's ddx and einfold mixed in:

```python
from datafusion import SessionContext
from ddxdb.datafusion import Ddx
from einfold.datafusion import EinFold
from datafusion_accelerated import Accelerator, Program
from datafusion_accelerated_cuda import Cuda

runtime = Accelerator()
ctx = SessionContext().with_extensions(Ddx(), EinFold(), runtime, Cuda(devices="local"))
ctx.sql("SET accelerated.strict = true")            # declines are errors (R8)
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
        .with_accelerator(AcceleratorOptions::new().strict(true))
        .with_accelerated_cuda()
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

ddx comes in through an integration, `datafusion-accelerated-ddx` (§5.8),
which implements `ddx_ad::Backend`, and `ddxdb.Backend` through its Python
binding. From its first release, `Backend` is a contract: ddx lists its
changes in its changelog, and the integration pins ddx exactly. The
runtime needs, each as an ordinary ddx PR:

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
    first among ddx#261 to ddx#265.

### 10.3 To einfold

einfold comes in through an integration, `datafusion-accelerated-einfold`
(§5.8). einfold already attaches as a DataFusion extension, and the
runtime's rule runs after its rules. The runtime needs from einfold:
1. a builder method, `.with_einfold()`, beside its `install` function, so
   that every extension attaches the same way;
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

### 10.5 Distribution

The runtime distributes a program by composing with datafusion-distributed
(D7), not by building a distributed planner. The aspiration, as the code a
user would run, is [#5](https://github.com/xqlsystems/datafusion-accelerated/issues/5).
S7 has the reasoning, and S8 the database theory it follows.

**Parallelism is partitioning, declared in SQL** (D9). Each table's
distribution is declared where the table is created, and the planner
derives the distribution of everything in between, as massively parallel
(MPP) databases such as Greenplum and Amazon Redshift do (S8.7):

```sql
CREATE EXTERNAL TABLE ocean STORED AS ZARR LOCATION 'om4.zarr'
  OPTIONS ('accelerated.distribution' 'tiles(lat 6, lon 6)');   -- a grid tile per GPU
CREATE EXTERNAL TABLE w STORED AS ACCELERATED
  OPTIONS ('accelerated.distribution' 'blocks(k)');             -- weights and moments sharded
CREATE EXTERNAL TABLE stats STORED AS ACCELERATED
  OPTIONS ('accelerated.distribution' 'replicated');            -- a copy on every GPU
```

| Strategy | Declared distribution |
|---|---|
| Data parallelism | the batch `hash(sample)`, the weights `replicated` |
| Sharded weights and optimizer state, as PyTorch's FSDP (Fully Sharded Data Parallel) and DeepSpeed's ZeRO do | the state tables `blocks(k)`: contiguous ranges of the key, which a reduce-scatter needs |
| Tensor parallelism, as NVIDIA's Megatron does, splitting each weight matrix across GPUs | a weight `hash` by its output axis, or `tiles` over both axes for 2D variants |
| Domain decomposition | the grid `tiles(lat, lon)` |
| Pipeline parallelism | labelled parts of the program on groups of devices, with `SET accelerated.place.<label>` |

The training step's SQL is the same under every strategy. The option
names here are proposals
([#15](https://github.com/xqlsystems/datafusion-accelerated/issues/15)).

Rules:
- **Declarations go through DataFusion's own mechanisms.** `STORED AS ...
  OPTIONS (...)` reaches the runtime through DataFusion's table provider
  factories, so it needs no new SQL syntax. A `DISTRIBUTED BY` clause, as
  Greenplum has, could be proposed to DataFusion's parser later.
- **The planner derives intermediates**, through DataFusion's physical
  properties: each node states the distribution it requires of its inputs
  and the partitioning it delivers, and DataFusion's `EnforceDistribution`
  rule inserts the exchanges where they differ, as the Volcano and
  Cascades optimizer frameworks do (S8.1). Where several partitionings would do, a cost model of the
  communication each needs chooses
  ([#16](https://github.com/xqlsystems/datafusion-accelerated/issues/16)).
  The schedule overrides the choice for one intermediate (§5.6).
  DataFusion parses SQL's `DISTRIBUTE BY`, but its physical planner
  refuses it in DataFusion 55.2 (spike 1), so a SQL form of the override
  waits on that upstream.
- **Partitioning never changes a result** (R7). A model whose meaning
  depends on its partitioning, such as batch norm with statistics per
  device, writes that as logic: a `GROUP BY` on a tile column computed
  from the coordinates. Then the dependence is in the SQL, where ddx
  differentiates it and checking mode sees it (S8.8).
- **Tiles span several axes.** DataFusion's `Range` partitioning splits
  one ordered key, which cuts a grid into bands. Square tiles need a
  partitioning over several axes, upstream in DataFusion
  ([#17](https://github.com/xqlsystems/datafusion-accelerated/issues/17)), or a tile
  key computed from the coordinates.
- **Exchanges move blocks, not rows.** A shuffle moves each partition's
  data as compact-array blocks, keyed by tile or shard, never as long-format
  rows hashed one at a time (S8.6).
- **Halos are derived.** For a convolution's join, the planner replicates
  each tile's boundary rows to the width of the join's band (the kernel's
  radius times its dilation), and each output cell is computed by the
  partition that owns it, so none is computed twice (S8.4).

**Collectives are relational operators** (S7.2). The planner writes them,
and `explain()` shows them:
- an all-reduce of gradients is a partial `SUM` per GPU, a shuffle of the
  partials by weight key, and a final `SUM` where each key lives: a
  reduce-scatter, after which each shard updates where it lives and the
  next run gathers what it reads. That is correct because `SUM` is
  distributive (S8.2);
- a halo exchange is the rows of a convolution's join that cross a tile's
  edge.

**Exchanges are lowered to collective capabilities** (S7.7). The
relational layer decides *what* moves. The core recognizes each
exchange's pattern as a *capability*, and the session's transport
integrations (§5.8) decide *how*, each declaring the capabilities it
implements. Under datafusion-distributed's gRPC as the *control plane*
(plans, stages, task assignment, metrics, and the setup of each
transport), the *data plane* is the device's own
([#6](https://github.com/xqlsystems/datafusion-accelerated/issues/6)):

| Exchange in the plan | Capability | Exact when | Implemented by, for example |
|---|---|---|---|
| partial `SUM` → shuffle by key → final `SUM` | reduce-scatter | the key's owners are contiguous ranges (`blocks(k)`), and the partials are dense over the key | NCCL, RCCL, XLA |
| the same, into a replicated table | all-reduce | as above | NCCL, RCCL, XLA |
| a replicated table read everywhere | all-gather, broadcast | always | NCCL, RCCL, XLA |
| a general shuffle | all-to-all, as grouped sends and receives | always | NCCL, RCCL, XLA |
| a halo exchange | neighbour exchange | always | NCCL's grouped sends; within an NVL72 rack, a kernel writing into fabric-mapped memory |
| host rows | host transfer | always | datafusion-distributed's gRPC, built in |

Transports come in two kinds, and the device plan is the same under either
([#21](https://github.com/xqlsystems/datafusion-accelerated/issues/21)):
- **eager**, one call per exchange, on the partition's queue: NCCL, RCCL
  and gRPC;
- **compiled**, lowered into the program of the region that holds the
  exchange: XLA's collectives on a TPU, through StableHLO with sharding
  annotations, inside the array backend (D2); and on CUDA, perhaps NCCL
  captured in a CUDA graph (§13, question 10).

Rules for the lowering:
- **A collective reduces a count beside each sum**, so a group no
  partition reached stays absent, as SQL's `SUM` over no rows gives no row.
- **Collective libraries are set up once per compiled program**: their
  communicators for exactly the devices the placement uses, bootstrapped
  over the control plane, and their buffers registered once, since state
  is written in place and exchange buffers are allocated once
  ([#19](https://github.com/xqlsystems/datafusion-accelerated/issues/19)).
- **Reductions are bucketed and overlapped.** The reductions of one step
  are fused into grouped calls over flat buffers, and a layer's reduction
  starts while earlier layers' backward steps still compute
  ([#18](https://github.com/xqlsystems/datafusion-accelerated/issues/18)).
- **A host application's communicator can be borrowed.** As a co-processor
  beside PyTorch, a transport derives its communicator from one the host
  exposes (for NCCL, `ncclCommSplit`), rather than starting a second
  world.
- **A session that requires reproducibility** (R9) pins the collective
  library's algorithm, since the order of a reduction depends on it.

**What the schedule holds.** Not the partitioning, which is in SQL. The
schedule records how partitions map to devices, and which link each
exchange uses, and a developer may override either (§5.6).

More rules:
- **Workers are sessions too**, built with the same extensions, devices
  and kernels as the coordinating session (§5.1). The runtime's codec
  ships a `KernelExec` as its kernel's name and tuning parameters, and the
  worker finds the kernel in its own session.
- **A distributed device plan** holds datafusion-distributed's network
  nodes and the runtime's device nodes in one physical plan.
- **Distributed programs persist.** A compiled program's stages are
  deployed to their workers once, and each run only triggers them. Each
  partition of a state table is resident on one worker's device across
  runs, as a table handle the coordinating session refers to remotely,
  and the tasks that read or update a partition run where it lives, as
  HaLoop, an iterative MapReduce system, schedules iterations (S8.5,
  [#7](https://github.com/xqlsystems/datafusion-accelerated/issues/7)).
- **The topology is a graph**, of devices and links with their speeds and
  tiers, taken from each platform's own discovery (NCCL and NVML, RCCL and
  AMD's tools, PJRT's device coordinates), not declared. It covers a
  switched NVLink domain, an AMD node's full mesh, a TPU slice's torus and
  the networks between domains. Placement embeds the partition grid into
  it, and `explain()` shows the link each exchange uses
  ([#8](https://github.com/xqlsystems/datafusion-accelerated/issues/8)).
- **Device data crosses machines without the host** once exchanges are
  lowered. Until then, distribution is correct but slow.
- **A distributed run is synchronous**: every device finishes a training
  step before any starts the next, as in the bulk-synchronous model. One
  slow GPU stalls a step, and `report()` names it by worker and device.
  Recovery from a failed worker restarts from the last checkpoint, whose
  shards each worker writes to one Zarr store
  ([#11](https://github.com/xqlsystems/datafusion-accelerated/issues/11)).
- **The first distribution stays within one fast domain where it can.** One
  NVL72 rack holds a whole Samudra run on NVLink. On an AMD cluster, 36
  GPUs span five nodes, so placement keeps most halos within a node. On a
  TPU slice, the grid's tiles map onto the torus's axes, and Samudra's
  periodic longitude onto its wraparound.

Whether a planner that chooses the partitioning per join distributes
Samudra better than FSDP is torch-sql's H1. Spike 8 tests it (S7.1, S8.3).

<details>
<summary>The aspiration, in Python: Samudra across a rack (#5)</summary>

```python
ctx = SessionContext().with_extensions(
    Ddx(), EinFold(),
    Accelerator(), Cuda(devices="local"),       # on each worker: that node's GPUs
    Distributed(workers=SlurmWorkers()),       # proposed Python binding (#10)
)
ctx.sql("""CREATE EXTERNAL TABLE ocean STORED AS ZARR LOCATION 'om4.zarr'
           OPTIONS ('accelerated.distribution' 'tiles(lat 6, lon 6)')""")    # 36 GPUs
ctx.sql("""CREATE EXTERNAL TABLE w STORED AS ACCELERATED
           OPTIONS ('accelerated.distribution' 'blocks(k)')""")              # sharded state

compiled = Program(ctx, STEP, state={"w": w0}, outputs=["loss"]).lower().compile()
print(compiled.explain())                      # the exchanges the planner added, and their links
for i in range(10_000):
    compiled.run(lr=1e-3 * 0.999**i)
```

```bash
srun --nodes=9 --ntasks-per-node=1 datafusion-accelerated worker --devices all &   # proposed CLI
python train.py
```

</details>

### 10.6 To kernel compilers and GPU libraries

- **Kernels arrive through bindings** (§5.7), from any compiler. The
  runtime has no kernel language, and no compiler of its own.
- **Array backends** are integrations that wrap an array library and lower
  einfold's nodes through it: NumPy in the core, as the reference, then one
  for CUDA at M2, and MLX beside it (§5.8, §13, question 1).
- **StableHLO and MLIR** appear only inside an array backend, as the route to XLA
  through JAX or PJRT, never as XQL's representation of a program (D2).
- **Memory** comes from the shared allocator (§7).

---

## 11. Where today's code goes

Each piece of today's code goes to the project whose job it does: the
runtime runs things on devices, einfold recognizes and proves array
structure, ddx differentiates, and frontends hold training loops and
demos. The [landing guide](internal/landing-the-performance-work.md) gives
each in-flight PR and branch its destination, with code before and after.
In short:

| Code | Goes to |
|---|---|
| `SharedExecutor`'s caching of CTEs (xarray-sql#302) | ddx's SQL path (ddx#168) for gradients, and DataFusion's planned-once CTEs for forward queries; the executor retires |
| `DenseExecutor` (xarray-sql#303) | the runtime's device plans over compact arrays (§5.3); the executor retires |
| MLX kernels and `DenseExecutor`'s MLX operators | the Metal integration's MLX array backend (§5.8), never the core |
| `TrainingSession` (xarray-sql#305) | the runtime's programs (§5.4), with state kept on the device and published after validation (R5); the session retires |
| AdamW, gradient clipping, learning-rate schedules, evaluation | frontends |
| `DenseProgram` and its proofs (einfold#157 and the contributor's fork) | einfold, rewritten on its Rust IR, with `DenseProgram` as a corpus and an oracle |
| `ddxdb.Context`, a `SessionContext` subclass | ddx's DataFusion extension (§10.2) |
| torch-sql's cut points and their chain rule | nowhere: ddx's SQL path replaces them (§6.4) |
| The Shakespeare recipe, and `erf` as a DataFusion function | xarray-sql, as demo 3 on public APIs; `erf` upstream eventually |

---

## 12. Building it

### 12.1 Packages

The project is `datafusion-accelerated` (D3, S4.3). It straddles Rust and
Python as DataFusion, ddx and einfold do (D6). The core is one crate and
one Python package. Every integration is a crate or package of its own
(R13, §5.8):

| Package | What it is | Depends on |
|---|---|---|
| `datafusion-accelerated` | the core: the DataFusion extension (builder method, settings, device memory pools in `RuntimeEnv`, physical rules, `KernelExec`, `CopyExec`, handle batches, table providers, the codec), programs, table handles, tiers, the schedule, labels, the interfaces of §5.8 with their CPU reference implementations, and the counters | DataFusion alone |
| `datafusion-accelerated-ddx` | ddx's `Backend`, and programs from ddx (§10.2) | ddx |
| `datafusion-accelerated-einfold` | einfold's nodes as bindings (§10.3) | einfold |
| `datafusion-accelerated-cuda` | a `Device` for CUDA | CUDA bindings |
| `datafusion-accelerated-nccl` | a `Transport` over NCCL | NCCL |
| `datafusion-accelerated-cudf` | a `KernelLibrary` of relational operators over libcudf | libcudf |
| `datafusion-accelerated-rocm`, `-rccl` | the same for AMD; later | HIP, RCCL |
| `datafusion-accelerated-metal` | a `Device` for Apple's GPUs, with MLX as an array backend; later | Metal, MLX |
| `datafusion-accelerated-pjrt` | a `Device`, and a compiled `Transport`, for TPUs; later | PJRT |

Each is published to crates.io, and to PyPI where it has a Python side,
built with maturin, the tool that packages Rust extensions for Python, as `datafusion_accelerated` and
`datafusion_accelerated_cuda` and so on. The core package offers extras
that only install integrations, such as `datafusion-accelerated[cuda]`.

Rules:
- **The core installs no GPU library** (R6, R13).
- **The core pins DataFusion exactly**, and every integration pins its own
  library. The ddx and einfold integrations pin the DataFusion version that
  xarray-sql, ddx and einfold share.
- **Python integrations, kernels and array backends are Python code**,
  behind Python protocols, so that adding one never needs a Rust build.
- **Dependencies point toward the core.** Integrations depend on it, and
  frontends depend on it only through an extra, such as
  `xarray-sql[accelerated]`.
- **Linux on ARM wheels from the first release**, for the Beta cluster's
  Grace CPUs.

### 12.2 Milestones

Each milestone is measured on XQL's programs (demo 2, demo 4 and
torch-sql's nanoGPT), and from M8 on the queries of TPC-H, the standard
decision-support benchmark, too. Results are reported on
[xarray-sql#285](https://github.com/xqlsystems/xarray-sql/issues/285) with
the hardware and versions, whichever way the numbers go. NVIDIA GPUs,
on the cloud or on the Beta cluster, are available half a day a week.

**M0. A CPU runtime, in Rust and Python.**
- *Builds:* `.with_accelerator()` and the `Accelerator` extension object,
  the `accelerated` settings, table handles and providers, `Program` with
  `lower`, `compile` and `run`, the ddx integration, implementing
  `ddx_ad::Backend` with compiled plans and tables kept in place, the
  counters of R8 as metrics, and the schedule and `explain()` in their
  first form, even while every step runs on DataFusion. They are cheap to
  build first and expensive to add later.
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
- *Builds:* `KernelExec`, handle batches, and the interfaces of §5.8, with
  the first integrations as separate crates: a CUDA `Device` and an array
  backend for einfold's nodes, with NumPy on the CPU as the reference;
  memory from a shared allocator; and kernels bound by label from Python
  and from Rust.
  MLX follows, or comes first if the contributor's Apple work gets there
  sooner.
- *Needs:* einfold's B-0 interface and fact table, and spike 6. Spike 7
  is done.
- *Done when:* demo 2 trains on an NVIDIA GPU in strict mode, its state
  never leaving the device between steps, within the bound of §6.3 against
  NumPy, with the share of time in device nodes shown by `EXPLAIN ANALYZE`.
  Hand-written kernels from three libraries, Triton and Pallas from Python
  and cuTile Rust from Rust, each run one node, each bound with no code in
  the runtime specific to its library, and checking mode verifies them.
  Speed and memory are reported against what exists: demo 2's and
  torch-sql's nanoGPT warm training step against PyTorch on the same GPU,
  eager and with `torch.compile`, with peak device memory beside
  PyTorch's, since §7 budgets two copies of state. Where the MLX
  integration runs, the same steps also run against the contributor's MLX
  path (§2.4), and B-0 is no slower there than `DenseExecutor`, the
  existing code closest to it (§13, question 14).

**M3. Save points and memory policies.**
- *Builds:* save points and recompute policies passed to ddx by label, and
  einfold's report on splits (§6.4, §6.5).
- *Done when:* demo 4 trains under a memory budget the session sets, with
  every recompute counted by label.

**M4. Concurrent steps.**
- *Needs:* ddx#157, and the gain measured again on compiled runs.
- *Done when:* concurrent runs give the same results as sequential ones,
  within R9's statement, and the time saved is reported.

**M5. B-1, regions within a step.** Planned, since the evidence expects
B-0 to stop short (§6.2). Spike 6 and M2 measure by how much, and so what
M5 must recover.
- *Done when:* torch-sql's nanoGPT step runs each of einfold's regions as
  one compiled program, and its warm step and peak memory are reported
  against M2's and PyTorch's on the same GPU.

**M6. B-2, regions across steps**, with ddx#157 and ddx#251, and the
values a run's steps share (§6.5).
- *Done when:* on an Apple GPU, Modded-NanoGPT's warm training step is no
  slower than the contributor's MLX path on the same machine (238.95 ms on
  the M5 Ultra, S1.1), with peak memory reported beside PyTorch's.

**M7. Distribution, through datafusion-distributed**, within one NVL72
rack (§10.5).
- *Builds:* distribution declared in SQL (#15), partitioning of
  intermediates derived through DataFusion's properties by cost (#16),
  tiles over several axes (#17), exchanges lowered to NCCL with bucketing
  and overlap (#6, #18), workers built with the runtime's extension,
  programs whose stages and state persist on workers (#7), and placement
  over the topology graph (#8).
- *Needs:* spikes 8 and 9. Spikes 10 and 11 test the same design on AMD
  and TPU before any integration for them is built.
- *Done when:* demo 4 trains with its grid split by tile across the GPUs
  of one rack, in one physical plan with network nodes and device nodes,
  gives the same results as on one GPU within §6.3's bound, and its
  gradient reduction runs within a stated factor of NCCL's.

**M8. Relational operators on devices.**
- *Builds:* `KernelLibrary` bindings by node type for DataFusion's hash
  join, hash aggregate, filter and sort, over handle batches, with the
  first library integrated as a separate crate over libcudf.
- *Needs:* spike 12.
- *Done when:* demo 4's data path and xarray-sql's geospatial queries run
  their joins and aggregates on a GPU in strict mode, matching DataFusion's
  results, with the time in device nodes reported, and with no change to
  the core when the libcudf integration is swapped for a CPU stand-in.

M2 is the critical path. It waits on spike 6, and on Alex's
decision on einfold's roadmap. Spikes 2 and 3, which need NVIDIA hardware,
are the first work for the GPU half days, since they wait on no decision.

### 12.3 Spikes

| # | Question | How | Decides |
|---|---|---|---|
| 1 | How far does a fixed-shape tensor column pass through DataFusion's joins and aggregates without being copied or expanded to rows? | Run demo 2's steps with such columns, on host and device buffers | Whether design C is worth pursuing (S2.3). **Done:** not for the milestones; DataFusion keeps the type, copies tiles at filters, sorts and joins, and can't compute on them ([report](spikes/s01-fixed-shape-tensors/README.md)) |
| 2 | Do rows between operators cost a GPU SQL engine what they cost GPU kernels? | One of demo 2's steps on Sirius | Whether a GPU engine is an alternative to B |
| 3 | Can a B200 read Arrow buffers in place from Grace memory fast enough? | Read against copy, on the Beta cluster | Whether Grace memory is a tier worth spilling to (§7) |
| 4 | What does crossing from Python cost? | datafusion-ffi's overhead for the runtime's nodes (S4.1); launching a Triton kernel from a `KernelExec` through Python, and whether the call holds Python's lock | Which Python kernels are worth binding, and whether some must be compiled ahead of time |
| 5 | Where does the remaining gap to PyTorch go, in time and in memory? | Start from the contributor's own ablations, which divide the step by kind of work (S1.1). Then trace one warm resident step on the M5 Ultra against PyTorch on MPS, for what they don't cover, peak memory included | What M2, M5 and M6 should fix first |
| 6 | How much of each step does einfold already cover? | `EXPLAIN ANALYZE` with einfold on, for demo 2, demo 4 and torch-sql's nanoGPT: the share of time inside einfold's nodes, and how often a DataFusion operator sits between two of them (S2.5). The contributor's `DenseProgram` corpus and spike 5's ablations are its inputs. einfold's agent runs einfold's side | How far short B-0 stops, and so what M5 must recover (§6.2) |
| 7 | Can handle batches carry device data between DataFusion nodes? | Two `KernelExec`s in one plan passing handle batches, through repartitioning and the planner's other rewrites, with a `CopyExec` where rows are needed | Whether §5.3's device plans work as written, or device work must stay inside single nodes. **Done:** they work, under three rules now in §5.3 ([report](spikes/s07-handle-batches/README.md)) |
| 8 | Can relational collectives match NCCL? ([#12](https://github.com/xqlsystems/datafusion-accelerated/issues/12)) | Reduce one demo 2 step's gradients on 8 GPUs, then 2 nodes: NCCL all-reduce, NCCL reduce-scatter and all-gather, and a shuffle by key with a final sum | Whether the runtime needs a collective node that calls NCCL, or plain shuffles suffice |
| 9 | Which host suits distributed training, datafusion-distributed or Ballista? ([#13](https://github.com/xqlsystems/datafusion-accelerated/issues/13)) | One distributed demo 2 step on each, CPU first: per-run overhead over 100 runs, and deploying stages once | Whether D7 holds for training, or the distributed contract stays neutral between them (S7.5) |
| 10 | Does AMD's stack slot in with no runtime changes beyond its integrations? ([#22](https://github.com/xqlsystems/datafusion-accelerated/issues/22)) | One distributed demo 2 step on an 8-GPU MI300-class node: HIP, RCCL, a Triton kernel, tiles placed per node | Whether the device-family contract holds for ROCm, and whether AMD needs placement from the start |
| 11 | Do a region's exchanges lower to one XLA program on a TPU? ([#23](https://github.com/xqlsystems/datafusion-accelerated/issues/23)) | One demo 4 block, with its halo exchange and a gradient reduction, through StableHLO and PJRT on a small slice, against the same in JAX's `shard_map`, which maps a function over a mesh of devices | Whether a TPU integration fits the runtime's contract |
| 12 | Can libcudf's relational kernels run DataFusion's operators over handle batches? ([#24](https://github.com/xqlsystems/datafusion-accelerated/issues/24)) | Bind DataFusion's hash aggregate and hash join to libcudf, exchanging columns through the C Device Data Interface, on TPC-H queries and one xarray-sql query | Whether relational acceleration fits the kernel contract, and what M8 needs |
| 13 | Where do compact arrays stop paying for their absent cells, and does a blocked layout keep them unstored? ([#34](https://github.com/xqlsystems/datafusion-accelerated/issues/34)) | A 3 by 3 convolution on each depth level of a real ocean mask, as rows, as compact arrays and as blocks, against the same density scattered at random | How §5.2 chooses a layout. **Done:** blocks are the compact layout, chosen by stored cells per present cell (S2.7, [report](spikes/s13-compact-absence/README.md)) |

---

## 13. Open questions

1. **Which libraries the first CUDA array backend uses** for einfold's
   nodes: from Python, JAX, PyTorch, CuPy, or PJRT directly (S4.2); from
   Rust, kernels launched through CUDA bindings. More than one array backend
   can exist, and developers can bind any library's kernels besides.
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
   process is configured to draw from them, what happens when one can't
   be, and whether NCCL's registered buffers, which its NVLink SHARP
   collectives (NVLS) need, can come from them ([#19](https://github.com/xqlsystems/datafusion-accelerated/issues/19)).
7. **The schedule's JSON format**, and how it is versioned as the runtime
   grows (§5.6).
8. **einfold's node types as an MLIR dialect.** Publishing them that way
   would let MLIR-based compilers consume einfold's nodes directly. That is
   a seam to the compiler ecosystem, not a change of XQL's representation
   (D2), and it is einfold's to decide.
9. **NVIDIA's CUDA bindings or cudarc**, for the CUDA integration:
   NVIDIA's `cuda-rust` crates, or cudarc, which is older and more widely
   used, by Candle among others (§2.5). Either way, `KernelArgs` hands out
   raw handles with owners (§5.7), so kernels from libraries built on the
   other still bind.
10. **CUDA graphs.** A compiled step could be captured as a CUDA graph,
    replaying all its launches at once. A graph bakes in its buffers, which
    suits state written in place (§7) and inputs copied into buffers
    allocated once. Worth measuring after M2.
11. **Python kernels on distributed workers.** datafusion-distributed's
    workers are Rust servers. A kernel written in Python needs a worker
    that runs Python, or must be compiled ahead of time, as a Triton kernel
    can be, and launched from Rust
    ([#9](https://github.com/xqlsystems/datafusion-accelerated/issues/9)).
12. **A Rust frontend.** A backend for Burn, the Rust framework, whose arrays are XQL tables
    would do for Rust what torch-sql does for PyTorch (S6.6).
13. **Atomic state across workers.** When state is spread across workers,
    whether publication is atomic across all of them (R5), or recovery
    restarts from a checkpoint
    ([#11](https://github.com/xqlsystems/datafusion-accelerated/issues/11)).
14. **Whether M2 waits for parity with the MLX path.** #32 asks that M2
    not pass while slower than code that exists today. This draft gives M2
    a floor against `DenseExecutor`, which keeps compact arrays between
    operators run one at a time, as B-0 does, and puts parity with the
    contributor's fastest MLX path at M6, since that path's speed comes
    from whole compiled steps and values shared across steps (§6.2). The
    alternative is to hold the GPU milestone until B-1 lands. Alex decides
    ([#32](https://github.com/xqlsystems/datafusion-accelerated/issues/32)).

---

## Decision log

### Decided

- **D1** (Alex, 2026-11-01, for ddx#251). Recomputing is an explicit
  policy people opt into, never the engine's own choice (§6.5).
- **D2** (Alex, 2026-11-07). StableHLO is not XQL's representation of a
  program. SQL is (§6.2, S2.2).
- **D3** (Alex, 2026-11-07, revised the same day). The project is named
  DataFusion Accelerated, `datafusion-accelerated`. It was first
  `xql-runtime`; the revisions made it an extension of DataFusion for every
  DataFusion user, not only XQL's (S4.3).
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
- **D9** (Alex, 2026-11-07). Partitioning is declared in SQL, not in the
  schedule (§10.5).
- **D10** (Alex, 2026-11-07). The goal widens: any DataFusion physical node
  may have a device implementation, relational operators included. The
  implementations come from integrations (D13), such as one over libcudf.
  Array work comes first (§3.1, §5.7, M8).
- **D11** (Alex, 2026-11-07). Executing on devices for DataFusion,
  relational operators included, lives here rather than in einfold, which
  keeps recognizing structure, and its paths to other engines' GPU hosts.
  einfold's design changes to match.
- **D12** (Alex, 2026-11-07). The project lives under xqlsystems for now.
  A home beside datafusion-distributed, under datafusion-contrib, is a
  later conversation.
- **D13** (Alex, 2026-11-07). DataFusion Accelerated is an interstitial
  layer between DataFusion and accelerator primitives. It makes hardware
  trivially easy to integrate and integrates none itself: every
  integration, with CUDA, ROCm, Metal, TPUs or a library such as libcudf,
  is a separate crate or package, mixed in as its own extension, so the
  core's goal outlives any of them (R13, §5.8).

### Proposed

<details>
<summary>Proposals P-A to P-Y, for approval</summary>

P-F, P-G and P-H of earlier drafts became decisions D3, D4 and D6.

- **P-A.** The runtime is a separate project. einfold stays free of device
  execution, and keeps its CPU operator (xarray-sql#307).
- **P-B.** The unit of work is a program with explicit state, lowered and
  compiled as a JAX function is (R3, R5, §5.4).
- **P-C.** Design B, staged from B-0 (§6.2). It needs an einfold design
  change approved by its maintainer. Spike 1 closed design C for the
  milestones.
- **P-D.** ddx and einfold come in through integrations, which implement
  their public interfaces and depend on their published types. Neither
  depends on the runtime, and the core depends on neither (§5.8).
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
- **P-O.** Memory tiers extend `RuntimeEnv`, named by device family, with
  device memory accounted as DataFusion accounts host memory (§5.2).
- **P-P.** The runtime builds no compiler, allocator, communication library
  or distributed planner (§3.2, R11).
- **P-Q.** A device plan is a DataFusion physical plan with `KernelExec`
  and `CopyExec` nodes, passing handle batches between device nodes, under
  the three rules spike 7 found (§5.3).
- **P-R.** Every extension attaches to DataFusion the same way, the runtime,
  its integrations and XQL's projects alike: a builder method in Rust, and
  an extension object for `with_extensions` in Python (§5.1, §5.8).
- **P-S.** Parallelism is partitioning. Base and state tables declare their
  distribution in SQL (D9), the planner derives intermediates through
  DataFusion's physical properties and a cost model, and collectives are
  relational operators it writes (§10.5).
- **P-T.** Distributed programs persist: stages deployed once, state
  resident on workers, and tasks placed where their partitions live
  (§10.5).
- **P-U.** Kernel contracts and memory tiers are defined per device
  family, so that CUDA, ROCm and TPU integrations each meet them (§5.2, §5.7).
- **P-V.** Exchanges are lowered to collective capabilities, which
  transport integrations implement, eager or compiled, under
  datafusion-distributed's control plane (§10.5).
- **P-W.** The topology is a graph taken from each platform's discovery,
  and placement embeds partitions into it (§10.5).
- **P-X.** A value that several steps of one run share is proved
  identical by einfold and held by the runtime until the run ends. Under
  memory pressure it is spilled, and recomputed only where a recompute
  policy names it (§6.5, D1).
- **P-Y.** Compact arrays are stored as blocks, with all-absent blocks
  left out, and a table's layout and block size are chosen by the cells
  each would store per present cell, a fact about its domain (§5.2,
  spike 13).

</details>
