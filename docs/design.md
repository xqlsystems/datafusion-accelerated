# The XQL runtime: design

_author_: Alex Merose
_co-authors_: 🔥 Claude (torch-sql) and 🌀 Claude (xarray-sql), via Claude Code
_reviewers_: 🪭 Claude (einfold) and ∂ Claude (ddx). Their reviews of v1.1 are in [reviews/](reviews/).
_status_: Draft v1.4, for approval. Nothing is built.
_last updated_: 2026-11-07
_supplement_: [supplement.md](supplement.md), with the evidence, the alternatives considered and the reasoning behind each rule. Citations such as "(S1.1)" point there.
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
**xql-runtime**, which runs programs. A frontend or a transform hands it a
*program*: a set of query plans, together with the tables they read and
write. The runtime owns the memory that holds those tables, keeps them on
a device such as a GPU between operators and between steps, chooses an
implementation for each piece of work, schedules the steps, and later
spreads them across machines.

The runtime is needed because how data is held and moved between queries,
more than planning or arithmetic, decides how fast XQL runs a training
step (S1). On a 123.6M-parameter GPT, sending matrix products to a GPU
changed nothing, while holding intermediates as dense arrays made the step
67 times faster. Keeping weights and optimizer state on the GPU then took
a smaller model's step from 2.02 s to 101 ms, against PyTorch's 27 ms
(S1.1). Today each frontend builds this for itself: xarray-sql's
experiments built three executors and a training session inside
xarray-sql, and torch-sql's optimizer step runs a query per tensor (S1.3).
The runtime is the one place where devices, memory and kernels are
managed, behind interfaces every frontend uses.

The runtime does one job. It is the bridge between relational plans and
array hardware: it owns the lifetime and placement of every table, and
runs each piece of work on another system's engine or kernel. Like every
XQL project, it is one component of a composable data system, in the
sense of Wes McKinney's *deconstructed database*: it builds no compiler,
allocator or communication library of its own, and meets the systems
that do through standard formats (§2.5, R11, S5.4).

It also gives developers, and the agents working with them, every
control. The SQL says what a program computes. Every physical choice the
runtime makes, such as where a table lives, how it is laid out and which
kernel computes it, is recorded in a *schedule* that anyone can read and
override (§5.5). The runtime makes good default choices, never hidden
ones: it should never be a compiler that its users have to outsmart.

---

## 2. Background

### 2.1 Formats and terms

- **Apache Arrow** is a columnar memory format: a table is a set of
  columns, each a contiguous buffer. Every XQL engine reads and writes
  Arrow. Its *PyCapsule interface* hands Arrow data between Python
  libraries without copying, and its *C Device Data Interface* does the
  same for buffers in GPU memory. Its canonical *fixed-shape tensor* type
  is a column whose every value is a dense array of one shape. **ADBC**
  (Arrow Database Connectivity) is a standard API for databases that
  return Arrow.
- **Substrait** is a portable, serialized format for relational query
  plans, which DataFusion and DuckDB both produce and consume. A plan can
  share a subplan between readers with a `ReferenceRel`, but neither
  engine's producer emits one, so a subplan read twice is written out
  twice.
- **DLPack** is a standard for sharing a dense array between array
  libraries, such as Arrow and PyTorch, without copying.
- **Zarr** is a format for chunked N-dimensional arrays in files or object
  storage, which Xarray reads and writes.
- A **CTE** (common table expression) is a named subquery,
  `WITH x AS (...)`, that the rest of a query can read, possibly several
  times. A **view** is a named query stored in a session.
- A **compact array** is how the runtime holds a table densely: each
  dimension's coordinates stored once, a dense array of values, and
  separate masks for which rows exist and which values are NULL.

### 2.2 The engines

- **[DataFusion](https://datafusion.apache.org/)** is a SQL query engine
  written in Rust and built to be extended. *datafusion-python* is its
  Python binding, and *datafusion-ffi* is its interface for extensions
  built separately from it.
- **[DuckDB](https://duckdb.org/)** is an embedded analytical database. ddx
  runs on it through DuckDB's community Substrait extension.
- **[Sirius](https://github.com/sirius-db/sirius)** and NVIDIA's **GQE** are
  SQL engines that run on GPUs.

### 2.3 The frontends and transforms

- **[xarray-sql](https://github.com/xqlsystems/xarray-sql)** registers
  [Xarray](https://xarray.dev/) Datasets as tables, and turns query results
  back into Datasets. Its numbered demos are each checked against PyTorch
  or JAX: demo 1 is attention, demo 2 a GPT trained in plain SQL, demo 3
  nanoGPT's Shakespeare recipe, and demo 4 Samudra's U-Net.
- **[torch-sql](https://github.com/xqlsystems/torch-sql)** runs
  [PyTorch](https://pytorch.org/) programs as SQL. A tensor on its
  `device="sql"` is a table, and each operation on it becomes part of a
  query. It trains [nanoGPT](https://github.com/karpathy/nanoGPT), a small
  GPT-2 implementation. Its design doc records spikes S1 to S9, cited here
  as "torch-sql's spike S4" and so on.
- **[ddx](https://github.com/xqlsystems/ddx)** differentiates SQL. Given a
  query that computes a loss, it writes a *program*: a sequence of *steps*,
  each a plan that saves its result as a table, which together compute the
  loss and its gradients. A program also carries *checks*, plans that must
  return no rows for the result to be valid. `ddxdb` is its Python package,
  and `ddxdb.Backend` the protocol any engine implements to run ddx's
  programs.
- **[einfold](https://github.com/xqlsystems/einfold)** is a query optimizer
  for DataFusion. It finds array computations in a plan, such as a matrix
  product written as a join and a `GROUP BY`, and runs array algorithms for
  them in its own CPU operator, `EinFoldExec`. Its design calls whatever
  runs its plans the *host*, and says einfold never touches a device.

### 2.4 Models and hardware

- **[Samudra](https://github.com/m2lines/Samudra)** is an ocean emulator and
  XQL's first production model. It is a *U-Net*, a convolutional network
  that shrinks a grid level by level and grows it back, built from
  *ConvNeXt* blocks: two dilated convolutions, each with a batch norm and a
  GELU activation. It trains on a half-degree grid of 360 × 720 cells.
- **MLX** is Apple's array library for its GPUs. PyTorch runs on the same
  GPUs through its *MPS* backend.
- **The Beta cluster** of
  [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
  is where Samudra trains: 288 NVIDIA B200 GPUs in GB200 NVL72 racks. In a
  GB200 system, each Grace CPU (an ARM CPU) is joined to two B200 GPUs by
  NVLink-C2C, a link that lets a GPU read the CPU's memory directly. An
  NVL72 rack joins 72 GPUs with NVLink.

### 2.5 Kernel compilers and GPU libraries

The runtime calls these rather than replacing them (R11):
- **Triton** is a language and compiler for GPU kernels, written in
  Python. **CUTLASS** is NVIDIA's library of templates for fast matrix
  kernels, and its *CuTe* layer describes how data is laid out across a
  GPU's threads and memory.
- **XLA** is the compiler behind JAX. It compiles **StableHLO**, a
  portable set of array operations. Both are built on **MLIR**, a
  framework for writing compilers as layers of intermediate
  representations.
- **`torch.compile`** is PyTorch's compiler, which generates Triton
  kernels among others.
- **RMM**, the RAPIDS Memory Manager, is a shared GPU memory allocator
  that several libraries can draw from at once.
- **NCCL** is NVIDIA's library for moving data between GPUs.

---

## 3. Goals and non-goals

### 3.1 Goals

1. **Run programs from any XQL frontend or transform** through one
   interface: ddx's gradient programs, frontends' training steps, and
   plain queries.
2. **Own tables and their memory.** Every table the runtime holds has an
   owner, a lifetime and a placement.
3. **Keep data where the computation is**, between operators and between
   steps, so a training loop moves only its batches and the values it
   reads.
4. **Choose an implementation for each piece of work**: the engine by
   default, and the runtime's array backends for work einfold has marked.
5. **Hold values across the memory hierarchy**: by default where the
   runtime judges best, and wherever the schedule says. Recompute values
   only under a policy the user turns on.
6. **Schedule**: run independent steps concurrently, and prepare the next
   batch while a step runs.
7. **Distribute**, later: run one program across GPUs and machines.
8. **Give developers every control**: a schedule they can read and edit,
   kernels of their own bound to any part of a program, and a report of
   what each run cost (§5.5, §5.6, §6.6).
9. **Compose**: meet engines, kernel compilers, allocators and array
   libraries through standard formats, so that each can be replaced
   without changing the others.

### 3.2 Non-goals

- **An SQL engine of its own.** The runtime never interprets relational
  operators: no joins, no `GROUP BY`, no NULL rules. It runs engine plans,
  and the array work einfold has marked in them (S2.1).
- **Differentiation, plan rewriting, or drawing a program's step
  boundaries.** Those are ddx's and einfold's (§6.4).
- **A kernel compiler, a memory allocator or a communication library.**
  Kernels come from Triton, CUTLASS, XLA, `torch.compile`, MLX or a
  developer's own code. GPU memory comes from a shared allocator, and data
  moves between GPUs through NCCL (§2.5).
- **A training framework.** Training loops, optimizers and learning-rate
  schedules belong to frontends. The runtime provides state that persists, atomic updates,
  and checkpoints of that state.
- **Bitwise agreement across implementations.** Results agree with the
  engine's to the bound of §6.3.

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
  Keeping a value in an engine table, in host memory or on a device changes
  neither results nor gradients. That is what makes it safe to hand every
  such choice to the developer (R10). Recomputing a value instead of saving
  it happens only under a user's policy, and only for plans that give the
  same rows each time (§6.5).
- **R8. Declines and costs are visible.** The runtime counts every
  decline, with its reason and the label of the code it came from (§5.4),
  and reports what each run cost (§6.6). *Strict* mode turns declines into
  errors. *Checking* mode runs the engine beside a substitute and compares
  them.
- **R9. Reproducibility is stated.** Each implementation says whether the
  same inputs give the same bits. A session may require reproducibility,
  and then gets only the implementations that give it (S3.7).
- **R10. Defaults, not hidden decisions.** Every physical choice is an
  entry in the schedule (§5.5). The runtime fills in defaults, and a
  developer may override any entry. An override is binding: if the runtime
  can't obey it, compiling fails and says why. Nothing is ignored silently
  (S5.1).
- **R11. One job, standard seams.** The runtime owns tables' lifetimes,
  placement and the schedule, and delegates everything else. Tables cross
  its boundaries as Arrow, arrays as DLPack, plans as Substrait, and
  schedules as a documented, versioned format, so that each side of every
  seam can be used without the other (S5.4).

---

## 5. The model

### 5.1 Programs

A **program** is a directed acyclic graph of **steps**. Each step is a
Substrait plan that reads tables and writes one table. A program names:

- its **inputs**: tables supplied on each run, such as a batch;
- its **state**: tables that persist across runs, each with the step that
  computes its next value;
- its **parameters**: scalars that change between runs without changing
  the plans, such as a learning rate;
- its **outputs**: the tables the caller reads after a run;
- its **checks**: conditions a run must meet for its result to be valid.

Rules:
- **Programs compose.** A training step is ddx's program followed by the
  optimizer's update steps, which read its gradient tables and write the
  next state. The runtime takes both as one program.
- **ddx's checks run when they must.** Each states a promise: that some key
  columns of one table are unique. The runtime runs a check once per
  compile for a table whose keys don't change, on each run otherwise, and
  never when a frontend's fact gives the same promise.
- **einfold's checks run on every run**, as data checks on the values of
  that run (§6.3, S3.1).
- **Parameters are a one-row table** that the plans cross join and the
  runtime writes into before each run (S3.2).

### 5.2 Tables and the memory hierarchy

A **table handle** refers to a table the runtime owns, as a buffer handle
refers to an allocation. It carries the table's schema, its lifetime (one
run, or until the caller releases it), and one or more *instances*. Each
instance has three independent properties, as a PyTorch tensor has a
device and, separately, strides:
- **a tier**, where the bytes live: *device* memory (a GPU's), *host*
  memory (pinned, meaning locked in place so a GPU can copy from it
  directly, or the Grace memory a GB200's GPUs read directly), or
  *storage* (Zarr or Arrow IPC files);
- **a layout**, how they are arranged: long-format Arrow rows, or a compact
  array;
- **an owner**, who manages the buffer: an engine, as one of its own
  tables, or the runtime.

Rules:
- A table can have several instances at once. The runtime copies one to
  another tier or layout when a step needs it there, and the schedule
  records each copy (§5.5).
- Developers may move tables themselves: `prefetch` to a tier, `evict`
  from one, and `pin` an instance so that it stays.
- Each tier reports its capacity and what it holds.
- Tables are values, as in SQL. A buffer is written in place only when
  state is published (R5, §7), so aliasing never reaches a program's
  meaning (S5.6).
- Reading a handle's rows is the only way data leaves the runtime, and it
  is always explicit.
- Each handle has a stable name, by which plans read it. The runtime passes
  ddx a namespace (`Options::namespace`) so that the same plan gives the
  same program byte for byte (S3.3).
- An engine table that holds state reads the handle's current buffer each
  time its plan runs, so that publishing new state never forces a plan to
  be planned again (S3.3).
- Runtime tables are DataFusion table providers that can be read two ways:
  as Arrow batches, for engine operators, and as compact arrays, for
  einfold nodes that scan them directly (S2.5).

### 5.3 Compiled programs

The runtime **compiles** a program once and runs it many times with new
inputs and parameters. Compiling plans each step, chooses implementations
and placements, and runs the checks that need running only once. A
compiled program is reused only while the backend's session key is
unchanged, as ddx's `prepare` does.

A run is asynchronous, as a kernel launch on a CUDA stream is. `run`
returns at once with a future for its outputs. Runs on one session execute
in order. A training loop overlaps loading with computing by prefetching
the next batch while a run executes, and steps within a run overlap under
the scheduler (§8).

### 5.4 Labels

One marker names a part of the forward program for every layer:
`ddx_name(expr, 'label')`, proposed on ddx#251, since CTE and view names
don't survive the engines' Substrait producers (torch-sql's spike S3). In
ddx's SQL path, CTE names serve too. Labels carry:
- ddx#251's regions, and the save points and recompute policies of §6.4
  and §6.5;
- provenance: which forward relation each backward step comes from
  ([ddx#259](https://github.com/xqlsystems/ddx/issues/259));
- einfold's hints (§6.2);
- the runtime's counters (R8).

Frontends choose the labels. torch-sql uses module paths, such as
`unet.layers.3`, so a decline is reported against the module that caused
it.

### 5.5 The schedule

The SQL is the program's *algorithm*: what it computes. The **schedule**
is everything else: how the runtime computes it. Compiling a program
produces its schedule, which holds, per step and per labelled node:
- the implementation: the engine, einfold's node on an array backend, or
  a developer's binding (§5.6);
- each table's instances: tier, layout and owner;
- copies between tiers, and when they happen;
- save points and recompute policies, passed to ddx (§6.4, §6.5);
- donation of state buffers (§7);
- the precision each implementation computes in (§6.3).

Rules:
- Each entry records who chose it: a default, einfold, or the developer.
- `compiled.schedule()` returns the schedule. It serializes to a
  documented, versioned format that tools and agents can read and write
  without the runtime.
- `session.compile(program, schedule=...)` takes a whole or partial
  schedule. The runtime fills in what is missing and obeys what is given
  (R10).
- Changing a schedule never changes a result beyond the bound of §6.3.
  Checking mode verifies that.

### 5.6 Bindings

A **binding** says which implementation computes a labelled node or step.
einfold's detection produces bindings automatically (§6.2). A developer,
or an agent, can write one by hand:

```python
session.bind("attn.0", my_flash_attention, assumes=["complete", "no_nulls"])
```

Rules:
- A bound implementation is a callable that takes and returns arrays as
  DLPack, with each axis named. It may come from Triton, CUTLASS, XLA,
  `torch.compile`, MLX or anywhere else (R11).
- The node's SQL is its specification. If a binding declines, or its
  assumptions fail a data check, the SQL runs.
- `assumes` lists the facts the binding relies on (§6.2). The runtime
  checks each on every run, unless a fact from a frontend states it.
- einfold's nodes are proved exact from the plan's structure. A
  developer's binding is only tested against its SQL, which is weaker
  (S5.3). So checking mode samples hand-written bindings, their runs are
  counted separately, and a session may refuse them in strict mode.

---

## 6. Execution

### 6.1 Three ways to run a step

The schedule says, per step, which way it runs (§5.5). By default the
runtime chooses:

1. **On the engine**, as a plain query. This is the reference (R1), and
   always available.
2. **As bound nodes.** einfold marks the array work it recognizes in the
   step's plan, and a developer may bind their own implementations by
   label (§5.6). The runtime runs each bound node on compact arrays it
   keeps resident, or declines it, and then the node's SQL runs on the
   engine. With einfold's nodes, this is stage B-0 (§6.2).
3. **As a region**: several nodes, and the elementwise expressions between
   them, run as one array program with their intermediates kept compact.
   These are stages B-1 and B-2.

A GPU SQL engine such as Sirius enters through way 1. Spike 2 tests
whether it meets the same wall as GPU kernels on long-format rows (S1.1).

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

Each node einfold detects becomes a binding (§5.6) to one of the runtime's
array backends. How a backend lowers a node (to NumPy, MLX, Triton,
`torch.compile`, or StableHLO through JAX) is private to the backend and
crosses no interface. Gathers on value indexes,
floating-point guards and index-table joins need particular care in
torch-sql's programs (S2.4).

**Facts and hints.** For a node to skip a check, einfold needs a fact
about its inputs from einfold's *fact table*, which is designed (einfold's
design §8.1) but not built:
- each dimension's extent, with integer keys `0 .. n - 1`;
- completeness, or a table of the domain;
- unique tuples;
- no NULL values;
- for gathers, that each index lies in range;
- for sums einfold reorders, that values are finite and bounded.

torch-sql states the structural facts for every table it writes (S1.3), as
[einfold#39](https://github.com/xqlsystems/einfold/issues/39) does for
xarray-sql, and the runtime states them for the tables it writes. Facts
about values hold for one run only, so they stay data checks (S3.1). A
frontend may also pass a *hint*, the array meaning it already knows, by
label (§5.4). einfold uses hints to guide detection, never in place of its
proof, and a hint it fails to confirm is a counted decline.

**Near misses.** Where einfold almost detects a node, it says why it
didn't: for example, "this would be a fold, but key `k` is nullable". The
runtime reports near misses with the run (§6.6), so that a developer or an
agent can change the SQL, state the missing fact, or bind an
implementation by hand, instead of guessing how to spell the query. The
SQL's spelling should decide correctness, never speed, and near misses are
how the gap between the two gets closed.

**Stages.** Build B in three stages, each measured and useful alone (S2.6):
- **B-0, one node at a time.** einfold hands the runtime the node it
  already detects, with its checks and its subplan, instead of running it
  in `EinFoldExec`. The runtime runs it on resident compact arrays. This
  replaces [einfold#156](https://github.com/xqlsystems/einfold/pull/156)'s
  per-call interface. Values stay compact between steps, and inside a step
  only where a node is the step's result or scans a runtime table (S2.5).
- **B-1, regions within one step**: einfold nodes connected through
  elementwise projections become one array program.
- **B-2, regions across steps**, through einfold's program mode, with
  ddx's step graph (ddx#157) and regions (ddx#251).

B pulls einfold's fact table and program mode ahead of its DuckDB and
sparse-kernel milestones. That is Alex's decision as einfold's maintainer,
after a sketch in einfold's design doc (§13, question 6).

Design C, in which the engine carries dense values itself as fixed-shape
tensor columns, stays open through spike 1 (S2.3).

### 6.3 Checks, declines and fallbacks

- **Checks on a device.** Each of einfold's run-time checks compiles to an
  on-device predicate. The predicates are ANDed into one validation flag,
  which the host reads once per run (S3.1).
- **When the flag is read.** A compiled step reads its flag after its last
  check and before its first state update.
- **Declines at compile time.** The node's SQL runs on the engine, and the
  rest of the program keeps its implementations.
- **Declines at run time.** A failed flag leaves the state untouched (R5),
  and the failed nodes' SQL runs on the engine from their inputs. Those
  inputs stay readable by the engine until the flag has been read.
- **Counting.** The runtime counts each implementation's runs and declines,
  by reason and label (R8).
- **Strict and checking modes.** Benchmarks run strict. Checking mode runs
  the engine beside a substitute on sampled steps, compares them to the
  bound below, and verifies the facts each node relied on.
- **The bound.** A float sum of `n` terms, in any order, must be within
  `(n − 1)·u·Σ|terms|` of the exact sum, where `u` is the unit roundoff of
  the precision the backend computes in (S3.8). A session chooses `u`:
  computing in float32 on a GPU is an explicit choice.
- **Reproducibility.** Each implementation declares whether it gives the
  same bits for the same inputs (R9).
- **Hand-written bindings** are checked as einfold's nodes are, and
  checking mode samples them more often, since nothing proves them (§5.6).

### 6.4 Step boundaries and saving

- **ddx draws step boundaries.** The runtime takes ddx's program as ddx
  writes it (S3.4).
- **The runtime changes them only through ddx's inputs:**
  - save points and recompute policies, by label (§5.4), passed to ddx as
    options, on one surface with ddx#251's regions;
  - engine limits, such as the depth a consumer can decode and a planning
    budget, which the runtime's `Backend` reports and ddx reads in place of
    its constants.
- **ddx never asks the `Backend` where to save**, so that a cached program
  stays a function of its plan and its options (S3.4).
- **Shared subplans come from SQL or from the frontend's program**, never
  from a Substrait plan. For plain queries, the runtime calls ddx's public
  form of ddx#168's split, and has no implementation of its own.
- **Save points are schedule entries** (§5.5), so a developer can add,
  remove or move one, and the runtime passes the result to ddx.
- **The runtime chooses default save points with einfold.** einfold reports whether
  a candidate save point would split a node it can fuse, and the runtime
  prefers save points that don't. A split taken anyway is a counted
  decline (S3.5).
- **Frontends don't run their own chain rules.** torch-sql retires its cut
  points by handing ddx SQL with a CTE at each shared node (S3.6), and
  stops gradients with `ddx_stop_gradient`, never by saving a table
  ([ddx#260](https://github.com/xqlsystems/ddx/issues/260)).

### 6.5 Recomputing

- **Recomputing is a policy the user opts into** (D1), by label. The
  runtime applies a memory-budget policy only when the session turns one
  on, and may always save more than a policy asks.
- **Only plans that give the same rows twice are recomputed** (S3.7). ddx
  reads volatility from the engine through its `Backend`.
- **The runtime's random function is declared deterministic**, so that a
  recomputed dropout draws the same mask.
- **`torch.utils.checkpoint`** becomes a recompute policy on a labelled
  region. Its random-state preservation comes free from the counter-based
  random function, or from masks uploaded as inputs (S3.6).

### 6.6 Reports

Every run can report, per step and per labelled node:
- time, split into planning, execution and copies;
- bytes moved between tiers, and the peak memory of each tier;
- the schedule it ran, with who chose each entry;
- declines and near misses, with their reasons.

`compiled.explain()` shows the same before a run, from the schedule alone.
Both are available as JSON, in a documented format, so that an agent can
read a run's cost, change the schedule and run again.

---

## 7. Memory

- **The runtime's defaults decide where each value is held**, from the
  engine's limits, each tier's capacity and the session's policies. The
  schedule records the choice, and a developer can override it (§5.5).
- **The runtime allocates no GPU memory of its own.** It draws from a
  shared allocator, RMM or the array library's own, so that it shares a
  GPU with PyTorch or JAX instead of competing with them for it (S4.2).
- **State has room for two copies** while a step runs, unless the program
  donates: once the validation flag has been read, each state update writes
  into the old table's memory, as JAX's buffer donation does (S3.9).
- **A program that donates restarts from its last checkpoint** after a
  failure that isn't a check, such as an engine error.
- **Grace memory is a host tier** on the Beta cluster, for values a GPU
  reads directly. Spike 3 measures whether it is fast enough to spill to.

---

## 8. Scheduling

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
| Compiling the whole step once, guarded by the session key | runtime (§5.3) |
| Reusing the loss ddx already computed ([ddx#257](https://github.com/xqlsystems/ddx/pull/257)) | runtime, through ddx's public API |
| Checkpoints of state, with the step counter and random state | runtime |
| Dropout with explicit seed and step keys | the runtime's random function; the frontend decides where dropout goes |
| AdamW, gradient clipping, the learning-rate schedule and evaluation | frontend |

- **Checkpoints are Zarr stores**, written from published state only, so
  that a checkpoint is a Dataset SQL can query. State that isn't
  array-shaped is written as Arrow IPC, Arrow's file format.
- **The random function is counter-based**: a hash of a seed, the step
  number and the element's coordinates, written in integer arithmetic that
  never overflows, so that every backend and every engine computes the
  same bits (S3.10).
- **Masks may also be inputs.** A frontend checked against another
  framework's random numbers, as torch-sql is against PyTorch's, uploads
  its masks as input tables (S3.10).

---

## 10. Interfaces

### 10.1 To frontends

A sketch, to make the model concrete. The names are placeholders:

```python
import xql_runtime as rt

session = rt.Session(engine="datafusion", devices=["cuda:0"])
w = session.put(weights, name="w", persist=True)   # an Arrow table, now owned

step = session.program(
    steps=ddx_program.steps,                       # Substrait plans
    inputs=["x"],
    state={"w": "w_next"},                         # w takes w_next's value
    parameters=["lr"],
    outputs=["loss"],
)
session.bind("attn.0", my_attention_kernel)       # a kernel of one's own (§5.6)
compiled = session.compile(step, strict=True)      # declines are errors (R8)
print(compiled.explain())                          # the schedule, and who chose what (§6.6)

schedule = compiled.schedule()                     # read, edit, recompile (§5.5)
schedule["w"].tier = "device"
compiled = session.compile(step, schedule=schedule, strict=True)

for batch in loader:
    out = compiled.run(inputs={"x": batch}, parameters={"lr": 6e-4})
    print(out.result()["loss"].to_pyarrow())       # a future; the only read
print(compiled.report())                           # cost, declines and near misses (§6.6)
```

- **torch-sql** hands ddx the runtime as its `Backend` at M0, with no other
  change. From M1, a synced `device="sql"` tensor holds a table handle, and
  a whole training step is one program.
- **xarray-sql** keeps registering Datasets and turning results back into
  Datasets, and hands the runtime a program when a user asks for one, for
  instance through `Context.prepare`.

### 10.2 To ddx

The runtime implements `ddxdb.Backend`, and later `ddx_ad::Backend` in
Rust. From the runtime's first release, `Backend` is a contract: ddx lists
its changes in its changelog, and the runtime pins ddxdb exactly. The
runtime needs, each as an ordinary ddx PR:

1. `ddx_stop_gradient` on every engine and type (ddx#260), first;
2. volatility read from the engine through the `Backend` (§6.5);
3. what each check promises (§5.1), and `to_sql` exported from ddxdb,
   since ddxdb runs DuckDB's steps as the SQL `to_sql` writes;
4. the step graph on `Program` (ddx#157);
5. labels and policies: ddx#251's fourth part, with save points beyond
   "read more than once" (§6.4);
6. engine limits read from the `Backend` (§6.4);
7. provenance (ddx#259);
8. the primal outputs of a prepared run (ddx#257) and the derivative of
   `erf` ([ddx#256](https://github.com/xqlsystems/ddx/pull/256)), both in
   review.

ddx#265, a wrong value with no error for one-row tables on DuckDB, comes
first among ddx#261 to #265, since parameters are one-row tables.

### 10.3 To einfold

The runtime is einfold's host. It needs from einfold:
1. the B-0 interface: each node's IR, checks and subplan, handed to the
   runtime as a binding to run or decline (§6.2);
2. published, versioned node types, with a conformance suite;
3. the fact table (einfold's M5);
4. a report on whether a candidate save point splits a node einfold can
   fuse (§6.4);
5. hints, by label (§5.4);
6. near misses, with their reasons (§6.2).

### 10.4 To engines and across the PyTorch boundary

- The runtime holds an engine session and runs plans on it: DataFusion
  first, then DuckDB, then GPU engines. xarray-sql, ddx and einfold must
  share one DataFusion version
  ([xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)).
- Tables cross into and out of the runtime as Arrow: the PyCapsule
  interface on the host, the C Device Data Interface on a GPU, and DLPack
  toward PyTorch. The device path needs a device-stream export upstream
  and a converter into PyTorch, since neither exists (S1.3).

### 10.5 Distribution, later

The first versions don't distribute, but the model must not prevent it.
Steps and tables carry the partitioning facts a distributed planner needs,
and einfold's nodes their named axes, so that a node can be split along
one. Partitioning a training step by batch index is data parallelism, by a
weight's index weight sharding, and by grid tile *domain decomposition*,
where each GPU owns a tile of the grid and exchanges only its edges with
its neighbours. torch-sql's hypothesis H1 is that a planner choosing the
partitioning per join distributes Samudra better than PyTorch's FSDP
(Fully Sharded Data Parallel). One NVL72 rack holds a whole Samudra run,
so the first distribution stays within a rack.

### 10.6 To kernel compilers and GPU libraries

- **Kernels arrive as bindings** (§5.6): callables over DLPack arrays,
  from any compiler. The runtime has no kernel language, and no compiler
  of its own.
- **Array backends** wrap an array library, and lower einfold's nodes
  through it: NumPy and MLX first, then a CUDA backend (§13, question 1).
- **StableHLO and MLIR** appear only inside a backend, as JAX's route to
  XLA, never as XQL's representation of a program (D2).
- **Memory** comes from RMM or the array library's allocator (§7).
- **Data moves between GPUs through NCCL**, once the runtime distributes
  (§10.5).

---

## 11. Where today's code goes

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
| Cut points and their chain rule (`torch_sql.program`) | torch-sql | nowhere: ddx's SQL path replaces them (§6.4) |

---

## 12. Building it

### 12.1 Packages

The project is `xql-runtime`, imported as `xql_runtime` (D3, S4.3).

- **The first releases are pure Python** (P-H). The package grows into the
  layout ddx and einfold use when the first Rust code lands: one Python
  package built with maturin, carrying a Rust core, and a Cargo workspace
  split by dependency. Moving the build to maturin then changes nothing for
  users.
- **The Rust side, when it comes:**

  | Channel | Package | Contents |
  |---|---|---|
  | crates.io | `xql-runtime` | the program model, table handles, placements, the scheduler and the counters, with an `Engine` trait and an `ArrayBackend` trait |
  | crates.io | `xql-runtime-datafusion` | the DataFusion engine: implements `ddx_ad::Backend`, hosts einfold's nodes, pins the shared DataFusion version |
  | crates.io | `xql-runtime-duckdb` | later |
  | PyPI | `xql-runtime` | the bindings, with extras `[datafusion]`, `[duckdb]`, `[mlx]` (Apple silicon only) and `[cuda]` |

- **The base install has no engine and no GPU library** (R6).
- **Engines and ddxdb are pinned exactly.**
- **Array backends that are Python libraries start as Python code** behind
  a Python protocol, so that adding one never needs a Rust build.
- **Dependencies point toward the runtime.** It implements `ddxdb.Backend`
  structurally, and frontends depend on it only through an extra, such as
  `xarray-sql[runtime]`.
- **Linux on ARM wheels from the first release**, for the Beta cluster's
  Grace CPUs.

### 12.2 Milestones

Each milestone is measured on demo 2, demo 4 and torch-sql's nanoGPT, and
reported on [xarray-sql#285](https://github.com/xqlsystems/xarray-sql/issues/285)
with the hardware and versions, whichever way the numbers go. NVIDIA GPUs,
on the cloud or on the Beta cluster, are available half a day a week.

**M0. A CPU runtime.**
- *Builds:* `Session`, table handles, `Program`, DataFusion and DuckDB
  engines implementing `ddxdb.Backend` with `compile` and tables kept in
  place, the counters of R8, and the schedule and reports in their first
  form: readable, serializable and binding, even while their only entries
  are engine steps (§5.5, §6.6). They are cheap to build first and
  expensive to add later.
- *Done when:* torch-sql's test suite passes on both engines with the
  runtime as its `Backend`, demos 2 and 4 give the same results through
  it as today, and an override the runtime can't obey fails compilation
  with a reason.

**M1. Resident state.**
- *Builds:* persistent tables, atomic publication, engine tables that read
  a handle's current buffer (§5.2), tiers in which a device is one more
  tier, the moves `prefetch`, `evict` and `pin`, and asynchronous runs.
- *Done when:* torch-sql's nanoGPT training step is one compiled program,
  with no query per tensor, and a failed check in a test leaves every state
  table unchanged.

**M2. A GPU, with B-0** (D4).
- *Builds:* an array backend for einfold's nodes on CUDA, with NumPy on the
  CPU as the reference, state resident on the device, memory from a shared
  allocator, and bindings by label for developers' own kernels. MLX
  follows, or comes first if the contributor's Apple work gets there
  sooner.
- *Needs:* einfold's B-0 interface and fact table, and spike 6.
- *Done when:* demo 2 trains on an NVIDIA GPU in strict mode, its state
  never leaving the device between steps, within the bound of §6.3 against
  NumPy, with the share of time in nodes reported. One node runs through a
  hand-written Triton binding, and checking mode verifies it.

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

**M7. Distribution**, within one NVL72 rack.

M2 is the critical path. It waits on spike 6 and on Alex's decision on
einfold's roadmap. Spikes 2 and 3, which need NVIDIA hardware, are the
first work for the GPU half days, since they wait on no decision.

### 12.3 Spikes

| # | Question | How | Decides |
|---|---|---|---|
| 1 | How far does a fixed-shape tensor column pass through DataFusion's joins and aggregates without being copied or expanded to rows? | Run demo 2's steps with such columns, on host and device buffers | Whether design C is worth pursuing (S2.3) |
| 2 | Do rows between operators cost a GPU SQL engine what they cost GPU kernels? | One of demo 2's steps on Sirius | Whether a GPU engine is an alternative to B |
| 3 | Can a B200 read Arrow buffers in place from Grace memory fast enough? | Read against copy, on the Beta cluster | Whether Grace memory is a placement worth spilling to (§7) |
| 4 | What does a Python core cost? | Whether DataFusion's and DuckDB's bindings release Python's lock while they plan and run; datafusion-ffi's overhead for einfold (S4.1); whether a Python table provider can read a handle's current buffer | P-H, and whether Rust starts at M1 or M4 |
| 5 | Where does the last 3.76× against PyTorch go? | Trace one warm resident step on the M5 Ultra against PyTorch MPS, by kind of operation, after reading the contributor's own comparison | What M2 and M5 should fix first |
| 6 | How much of each step does einfold already cover? | `EXPLAIN ANALYZE` (a query run that reports each operator's time) with einfold on, for demo 2, demo 4 and torch-sql's nanoGPT: the share of time inside einfold's nodes, and the share of steps that are one node from scans to result. einfold's agent runs einfold's side | How far B-0 gets, and whether M5 is needed |

---

## 13. Open questions

1. **Which array library runs einfold's nodes on NVIDIA GPUs**: JAX,
   PyTorch or CuPy (S4.2). More than one backend can exist.
2. **Rust or Python first.** This draft proposes Python first (P-H). Spike
   4 decides, and S4.1 has the arguments.
3. **Substrait as the plan format between layers.** Engines without a
   Substrait producer need SQL text in their dialect
   ([ddx#159](https://github.com/xqlsystems/ddx/issues/159)), and einfold
   works on DataFusion's logical plans, so a step may need both forms.
4. **Low precision.** Which nodes may compute in bfloat16, a 16-bit float
   format GB200 hardware is built for and SQL engines lack, and who
   chooses.
5. **How often checking mode samples** in production, for einfold's nodes
   and for hand-written bindings.
6. **einfold's roadmap.** M2 needs einfold's B-0 interface and fact table
   ahead of einfold's DuckDB and sparse-kernel milestones. That is Alex's
   decision as einfold's maintainer.
7. **The shared allocator.** RMM, or whichever allocator the array library
   in use provides, and what happens when two libraries in one process
   each bring their own (§7).
8. **The schedule's format.** Which existing format, if any, it should
   follow, and how it is versioned as the runtime grows (§5.5).
9. **einfold's node types as an MLIR dialect.** Publishing them that way
   would let MLIR-based compilers consume einfold's nodes directly. That is
   a seam to the compiler ecosystem, not a change of XQL's representation
   (D2), and it is einfold's to decide.

---

## Decision log

### Decided

- **D1** (Alex, 2026-11-01, for ddx#251). Recomputing is an explicit
  policy people opt into, never the engine's own choice (§6.5).
- **D2** (Alex, 2026-11-07). StableHLO is not XQL's representation of a
  program. SQL is (§6.2, S2.2).
- **D3** (Alex, 2026-11-07). The project is named `xql-runtime`, and is
  packaged as §12.1 describes.
- **D4** (Alex, 2026-11-07). A GPU milestone, with dense intermediates and
  resident state, comes third (M2).

### Proposed

- **P-A.** The runtime is a separate project. einfold stays free of device
  execution, and keeps its CPU operator (xarray-sql#307).
- **P-B.** The unit of work is a program with explicit state (R3, R5).
- **P-C.** Design B, staged from B-0 (§6.2). It needs an einfold design
  change approved by its maintainer. Design C stays open through spike 1.
- **P-D.** The runtime implements ddx's and einfold's public interfaces,
  and depends on their published types, rather than either depending on
  the runtime.
- **P-E.** Where a value is held is the runtime's default, recorded in the
  schedule and open to the developer's override (R7, R10). ddx draws step
  boundaries, and the runtime changes them only through ddx's options and
  the limits its `Backend` reports (§6.4).
- **P-H.** Python first, with explicit handles and a thin core, until
  spike 4 shows Rust is needed.
- **P-I.** The runtime chooses save points with einfold's report, and
  passes them to ddx as options. No frontend runs a chain rule of its own
  (§6.4).
- **P-J.** A program's reference is the engine it runs on, and
  disagreements between engines are found in tests and filed upstream
  (S4.4).
- **P-K.** One label marker serves regions, policies, provenance, hints
  and counters (§5.4).
- **P-L.** Reproducibility is stated per implementation, and a session may
  require it (R9).
- **P-M.** The schedule is a first-class object: every physical choice is
  an entry, overrides are binding, and impossible ones fail compilation
  (R10, §5.5).
- **P-N.** Developers bind their own kernels to labelled nodes, with the
  node's SQL as the specification. einfold's nodes are bindings made
  automatically (§5.6).
- **P-O.** Tables live in a memory hierarchy of tiers, with tier, layout
  and owner as independent properties, and runs are asynchronous (§5.2,
  §5.3).
- **P-P.** The runtime does one job, and builds no compiler, allocator or
  communication library. It meets the systems that do through Arrow,
  DLPack, Substrait and a versioned schedule format (R11).
