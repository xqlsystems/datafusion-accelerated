# The XQL runtime: running relational programs on devices

_author_: Alex Merose
_co-authors_: 🔥 Claude (torch-sql) and 🌀 Claude (xarray-sql), via Claude Code
_reviewers_: 🪭 Claude (einfold) and ∂ Claude (ddx). Their reviews of v1.1 are in [reviews/](reviews/).
_status_: Draft v1.2, for approval. Nothing is built.
_last updated_: 2026-11-07
_origin_: [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)

---

## 1. What the runtime is

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

§2 introduces each project. The runtime is the one layer that doesn't exist
yet. **The runtime runs programs.** A frontend or a transform hands it a
*program*: a set of query plans, together with the tables they read and
write. The runtime owns the memory that holds those tables, and keeps them
on a device such as a GPU between operators and between steps. It chooses
an implementation for each piece of work, schedules the steps, and later
spreads them across machines.

Today each frontend does this for itself, in different ways. torch-sql
runs its programs through a small engine wrapper of its own. xarray-sql's
nanoGPT experiments built three executors and a training session inside
xarray-sql (§3.1), which xarray-sql#307 found don't belong there. einfold
runs array algorithms on the CPU in its own operator, and leaves devices to
whatever runs the plan. Its design calls that the *host* (today, the
engine), and says einfold "never touches a device". The runtime joins the
engine as a host. It becomes the one place where devices, memory and
kernels are managed, behind interfaces every frontend can use.

### 1.1 Why now

Two lines of work found that how data is held and moved between queries,
more than planning or arithmetic, decides how fast XQL runs a training
step (§3):

1. In xarray-sql's nanoGPT experiments on an Apple GPU, sending the matrix
   products and attention to the GPU changed almost nothing for a
   123.6M-parameter model: 163.6 s per training step against 164.0 s on the
   CPU. What helped was holding intermediates as dense arrays instead of
   Arrow rows: 2.43 s per step on the GPU, 67 times faster. Then, on a
   smaller model (10.7M parameters, trained as nanoGPT trains its
   Shakespeare example), keeping weights and optimizer state on the GPU
   between steps brought a step from 2.02 s to 101 ms. PyTorch took 27 ms
   for the same step on the same GPU.
2. In torch-sql, a whole training step of nanoGPT matches PyTorch to 1e-11
   on two engines. But the optimizer's update alone takes 8 to 12 seconds
   at six layers, because each parameter and each piece of optimizer state
   is read and written by a query of its own.

Both are questions of how data is held and moved between queries, which is
the runtime's job.

---

## 2. Background

This section introduces the formats, projects and hardware the rest of the
doc uses.

### 2.1 Formats and terms

**Apache Arrow** is a columnar memory format: a table is a set of columns,
each a contiguous buffer. Every XQL engine reads and writes Arrow, so it is
how tables move between projects. Its *PyCapsule interface* hands Arrow
data between Python libraries without copying. Its *C Device Data
Interface* extends that to buffers in GPU memory. Its canonical
[fixed-shape tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
extension type is a column whose every value is a dense array of one shape.
**ADBC** (Arrow Database Connectivity) is a standard API for databases that
return Arrow.

**Substrait** is a portable, serialized format for relational query plans.
DataFusion and DuckDB can each produce and consume it, which lets ddx read
one engine's plan and hand back plans the same engine runs. A Substrait
plan can share a subplan between several readers with a `ReferenceRel`,
but neither engine's producer emits one, so a subplan read twice is
written out twice.

**DLPack** is a standard for sharing a dense array between array libraries,
such as Arrow and PyTorch, without copying.

**Zarr** is a format for chunked N-dimensional arrays in files or object
storage, which Xarray reads and writes.

A **CTE** (common table expression) is a named subquery, `WITH x AS (...)`,
that the rest of a query can read, possibly several times. A **view** is a
named query stored in a session, which later queries read like a table.

### 2.2 The engines

- **[DataFusion](https://datafusion.apache.org/)** is a SQL query engine
  written in Rust and built to be extended. xarray-sql and einfold build on
  it. *datafusion-python* is its Python binding, and *datafusion-ffi* is
  its interface for extensions built separately from it, which is how a
  Python package such as einfold's adds rules to a datafusion-python
  session.
- **[DuckDB](https://duckdb.org/)** is an embedded analytical database. ddx
  runs on it through DuckDB's community Substrait extension.
- **[Sirius](https://github.com/sirius-db/sirius)** and NVIDIA's **GQE** are
  SQL engines that run on GPUs.

### 2.3 The frontends and transforms

- **[xarray-sql](https://github.com/xqlsystems/xarray-sql)** registers
  [Xarray](https://xarray.dev/) Datasets (labelled N-dimensional arrays) as
  tables, and turns query results back into Datasets. Its numbered demos
  are each checked against a reference in PyTorch or JAX: demo 1 is
  attention, demo 2 a GPT language model trained in plain SQL, demo 3
  nanoGPT's Shakespeare recipe (§3.1), and demo 4 Samudra's U-Net (§2.4).
- **[torch-sql](https://github.com/xqlsystems/torch-sql)** runs
  [PyTorch](https://pytorch.org/) programs as SQL. A tensor on its
  `device="sql"` is a table, and each operation on it becomes part of a
  query. It trains [nanoGPT](https://github.com/karpathy/nanoGPT), a small
  GPT-2 implementation, today. Its design doc records spikes S1 to S9,
  which this doc cites as "torch-sql's spike S4" and so on.
- **[ddx](https://github.com/xqlsystems/ddx)** differentiates SQL. Given a
  query that computes a loss, it writes a *program*: a sequence of *steps*,
  each a plan that saves its result as a table, which together compute the
  loss and its gradients. A program also carries *checks*, plans that must
  return no rows for the result to be valid. `ddxdb` is its Python package.
- **[einfold](https://github.com/xqlsystems/einfold)** is a query optimizer
  for DataFusion. It finds array computations in a plan, such as a matrix
  product written as a join and a `GROUP BY`, and replaces them with array
  algorithms. It runs those algorithms on the CPU in its own operator,
  `EinFoldExec`. A draft *dense backend* interface
  ([einfold#156](https://github.com/xqlsystems/einfold/pull/156), from a
  fork, not yet reviewed by einfold's maintainer) would let a host take
  that work instead, or decline it.

### 2.4 Models and hardware

- **[Samudra](https://github.com/m2lines/Samudra)** is an ocean emulator, a
  neural network that predicts the ocean's next state from the current one,
  and XQL's first production model. It is a *U-Net*, a convolutional
  network that shrinks a grid level by level and grows it back. Its blocks
  are *ConvNeXt* blocks: two dilated convolutions, each with a batch norm
  and a GELU activation. It trains today on a half-degree grid of 360 × 720
  cells.
- **MLX** is Apple's array library for its GPUs. The nanoGPT performance
  work ran on an Apple M5 Ultra through it. PyTorch runs on the same GPUs
  through its *MPS* backend.
- **The Beta cluster** of
  [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
  is where Samudra trains. It has 288 NVIDIA B200 GPUs in GB200 NVL72
  racks. In a GB200 system, each Grace CPU (an ARM CPU) is joined to two
  B200 GPUs by NVLink-C2C, a cache-coherent link that lets a GPU read the
  CPU's memory directly. An NVL72 rack joins 72 GPUs with NVLink, NVIDIA's
  GPU-to-GPU interconnect.

---

## 3. What we know already

### 3.1 From xarray-sql's nanoGPT experiments

A contributor's experiments on an Apple M5 Ultra built a sequence of
executors, each on the one before. They are in draft PRs
([xarray-sql#302](https://github.com/xqlsystems/xarray-sql/pull/302),
[#303](https://github.com/xqlsystems/xarray-sql/pull/303) and
[#305](https://github.com/xqlsystems/xarray-sql/pull/305), with
[einfold#156](https://github.com/xqlsystems/einfold/pull/156) and
[einfold#157](https://github.com/xqlsystems/einfold/pull/157)), which
reviewers asked to be rehomed (xarray-sql#307). Every result was checked
against a reference outside the timer: JAX in float64, or nanoGPT's own
model and PyTorch's AdamW optimizer.

The measurements come from three different workloads, so their ratios
must not be multiplied together:

| Executor | What it does | Workload | Measured, per warm step |
|---|---|---|---|
| `SharedExecutor` (#302) | computes each CTE once per run, rather than at every read | tiny GPT, 4,416 parameters, CPU | 314 → 206 ms |
| MLX kernels (#302, einfold#156) | sends einfold's matrix products and attention to the GPU, through a backend that may decline | GPT-2 size, 123.6M parameters, context 16, batch 1, SGD | 164.0 s on the CPU, 163.6 s with GPU kernels: Arrow rows between operators dominate |
| `DenseExecutor` with einfold's `DenseProgram` (#303, einfold#157) | holds intermediates as *compact arrays*: each dimension's coordinates stored once, a dense array of values, and separate masks for which rows exist and which values are NULL | the same 123.6M model | 163.6 → 2.43 s on the GPU (67×); 164.0 → 4.04 s on the CPU (41×) |
| `TrainingSession` (#305) | keeps weights and optimizer state on the GPU, compiles the whole step once, and publishes new state only after the step's checks pass | nanoGPT's Shakespeare recipe: 10.7M parameters, batch 12, context 256, AdamW, dropout | 2.02 s (compact arrays, state through Arrow) → 101 ms; PyTorch MPS 27 ms |

Some limits on these numbers:
- Each comes from a single run on one machine.
- The PyTorch comparison ran without `torch.compile`, while the SQL step was
  compiled once with MLX's compiler.
- The 101 ms run didn't check each step against PyTorch; a separate
  five-step run that did took 146 ms per warm step.
- Nobody has measured where the remaining 3.76× goes (spike 5, §16).

The work has continued on unreviewed branches. Modded-NanoGPT is a
124M-parameter GPT trained on FineWeb text at context 1,024 with Muon, an
optimizer that orthogonalizes each update. On it, the latest SQL step takes
239 ms on the M5 Ultra, against PyTorch MPS's 182 ms: about 1.3 times
slower. The gains came from more proofs, in new modules in xarray-sql:
- `_dense_adjoint.py` proves that ddx's derivatives of attention match a
  kernel;
- `_dense_proofs.py` proves when a row exists, from coordinates alone;
- `_dense_attention.py` proves the shape of attention from einfold's
  reduction graph.

That is design B (§7.2) growing on its own, one proof at a time, but in a
frontend instead of in einfold. Each proof that lands in the wrong layer
makes the right layer harder to reach.

Five lessons follow:

1. **The representation between operators matters more than where the
   arithmetic runs.** Long-format rows between operators cost more than all
   the arithmetic on the CPU. Compact arrays won 41× on the CPU and 67× on
   the GPU, on the same model.
2. **Once intermediates are compact, moving state across the Arrow boundary
   becomes the cost**, so state must stay where the computation is. In the
   100 timed resident steps, the step exported no arrays and read back one
   validation flag and one loss value.
3. **Falling back must be cheap and fine-grained.** When `DenseExecutor`
   meets an operator it can't run, it restarts the whole call on the SQL
   executor. Benchmarks therefore ran in a strict mode where any decline is
   an error. The reported runs had no fallbacks, but only because strict
   mode ruled them out. Nobody counted how often each kind of operator
   declined. The one decline documented is a grouped aggregate with several
   dependent or nullable keys, in AdamW on compact arrays.
4. **Compact execution became a second implementation of SQL.**
   `DenseProgram` interprets DataFusion's logical plans operator by
   operator (joins, grouped aggregates, `UNION` with bag semantics, and NULL
   handling) in 1,564 lines of Python beside einfold's Rust. That is where
   most of the risk sits (§7.2).
5. **The pieces coupled through private names.** xarray-sql's executors
   import 18 private names from `einfold.dense_program`, einfold's private
   `einfold._native`, and ddx's private `ddxdb._ddxdb.to_sql`. The runtime
   has to be built on public interfaces instead (principle R2, §5).

### 3.2 From xarray-sql's Samudra demo

Demo 4 ([xarray-sql#300](https://github.com/xqlsystems/xarray-sql/pull/300))
writes Samudra's U-Net as one SQL view per layer, and checks it against
Samudra's own PyTorch. It runs the network two ways:
- **as views:** one query reads the last view, and the engine inlines the
  whole chain;
- **eagerly:** the same SQL, with each layer stored as a table before the
  next reads it, as PyTorch's eager mode keeps each layer's output.

At two levels, on an Intel i7-8700 with DataFusion 55.0.0, in seconds per
forecast:

| Grid | PyTorch (CPU) | DataFusion, views | DataFusion, eager | DuckDB, views | DuckDB, eager |
|---|---:|---:|---:|---:|---:|
| 10 × 20 | 0.002 | 1.80 | 0.122 | 364 | 0.289 |
| 32 × 64 | 0.002 | 4.69 | 0.417 | not run | not run |

Saving each layer once is 11–15× faster on DataFusion, and 1,260× faster on
DuckDB. That comes from a model that is not GPT, through a different
frontend, so the cost of reading a view twice is general.

Even when each layer is computed once, the SQL is still 60–180× slower than
PyTorch on the CPU. That is lesson 1 of §3.1 again: the remaining gap is
the long-format representation, not planning. The machine was under load
from a compile job, so these numbers are rough.

The demo keeps the views as its main path because ddx differentiates
through views, but not through tables the frontend saved. A saved table is
a constant to ddx, so the gradient would stop at it. Where a value is held
has to be invisible to differentiation (principle R7, §5).

### 3.3 From torch-sql

torch-sql runs a training step as a few large programs. Each finding below
was measured there or in its spikes:

- **DataFusion plans a CTE again at every read**
  ([xarray-sql#267](https://github.com/xqlsystems/xarray-sql/issues/267)).
  For a stack of ConvNeXt-like blocks as one query, the plan grows from 441
  operators for one block to 136,107 for three, and planning and running
  from 0.2 s to 212 s. DuckDB grows by 75 operators per block. torch-sql
  hands ddx each loss as one Substrait plan, where every shared subplan is
  already written out (§2.1), so it chooses *cut points*: values it saves
  as tables between queries. A value is a cut point when the query that
  computes it, with every value it reads written in, passes a per-engine
  budget (400 `SELECT` blocks on DataFusion, none on DuckDB). The input of
  every norm is a cut point whatever its size, because a norm reads it
  three times. Cut points brought one nanoGPT layer's backward pass on
  DataFusion from 9 s to 0.7 s, and two layers, which didn't fit in 8 GB
  before, run in 1.0 s.
- **Differentiating across cut points is a chain of ddx programs.** Each
  segment between cut points is one program. Each cut point passes a
  *cotangent* (the gradient of the loss with respect to that table) to the
  segment before it. torch-sql runs that chain rule itself (§7.4).
- **Per-table queries are the bottleneck once the math is right.** The
  optimizer's update at six layers takes 8 to 12 s, because it syncs each
  tensor separately.
- **Engines disagree, sometimes silently.** Running every op, forward and
  backward, on both engines against PyTorch found five gaps in ddx
  (ddx#261 to #265), four of them on DuckDB only. Four refuse or fail
  loudly. One, ddx#265, gave a wrong loss and gradient with no error. The
  engines themselves differ too: DuckDB reads `0.5` as a `DECIMAL`, and
  DataFusion's `MAX` with a `GROUP BY` skips NaN, so torch-sql writes its
  NaN handling out explicitly in SQL. Any kernel the runtime substitutes
  needs a reference to be checked against, and an engine's output is a
  good one only after such testing.
- **Every table torch-sql writes is complete.** A dense tensor has a row
  for every index, its index columns hold positions `0 .. n - 1`, its keys
  are unique, and its values are never NULL. Partial tensors, for a masked
  Samudra, will be the exception, and will carry their domain as a table
  of their own.
- **Lifetimes are hard to get right in the frontend.** torch-sql drops a
  table when the last Python object referring to it is collected, and a bug
  there dropped tables still in use. A table's owner and lifetime belong
  with whatever owns the memory.
- **Transport** (torch-sql's spike S4): an Arrow array with no nulls reaches
  PyTorch through DLPack without a copy. A fixed-shape tensor column keeps
  its type through DataFusion's projections, filters and sorts, and a
  projection keeps the same buffer. DuckDB turns it into a plain list of
  fixed size, which keeps the values but loses the shape.

### 3.4 From ddx

- **Planning was most of a small step, until programs were compiled.** In
  the program for one SGD step of demo 2, DataFusion's planning took
  208 ms of 422 ms, and execution 177 ms, on ddx 0.2.1
  ([ddx#157](https://github.com/xqlsystems/ddx/issues/157)). Since then,
  `prepare` builds a program once
  ([ddx#203](https://github.com/xqlsystems/ddx/pull/203)), and a backend
  can compile each step's physical plan to run again
  ([ddx#214](https://github.com/xqlsystems/ddx/pull/214)), so planning is
  paid once per program, not once per run.
  - The same step now takes 72 ms per warm run, in ddx's `prepared_step`
    benchmark (Rust, DataFusion 55, Intel i7-8700, ddx-ad 0.3.3).
  - From Python, ddxdb's `in_place` mode takes it from about 300 ms to
    about 100 ms (ddx#214). That mode needs an xqlsystems fork of
    datafusion-python, so it is experimental.
- **Steps could run concurrently.** Many of a program's steps read none of
  the others. Running each as soon as its inputs are ready would shorten
  xarray-sql's demos by 16 to 43 percent. That is an upper bound, from each
  program's critical path, measured before programs were compiled
  (ddx#157).
- **ddx draws its programs' step boundaries itself**, for four reasons:
  - each aggregate the backward pass reads is saved;
  - each CTE read more than once is planned alone
    ([ddx#168](https://github.com/xqlsystems/ddx/pull/168));
  - a step deeper than a consumer can decode is split every 32 relations;
  - a step that one later step reads once is computed inside it
    ([ddx#231](https://github.com/xqlsystems/ddx/pull/231)), unless its
    reader would pass 2,000 expression nodes.
- **ddx differentiates through shared subplans, given SQL or a program.**
  `grad(...)` written in SQL plans each CTE read more than once on its own,
  before the engine's producer runs, and joins the pieces through
  `ReferenceRel`s, which ddx lowers once (ddx#168). A view is written in as
  CTEs first ([ddx#169](https://github.com/xqlsystems/ddx/pull/169)). A
  thirty-layer chain, 2^30 copies when written out, is differentiated from
  SQL in about a second on an i7-8700. `grad` and `vjp` also take a
  program, and write each of its steps once. Only a loss handed over as one
  Substrait plan arrives with its copies written out, which is torch-sql's
  case (§3.3).
- **Regions are partly released.** ddx#251 lets a program name parts of
  itself and give each a policy. Its first three parts are in ddx-ad 0.3.3
  and ddxdb 0.4.7: attention's backward pass now writes one table with a
  row per score, down from two or three, whichever way the SQL spells
  attention. The fourth part, recomputing by policy, is in design.
- **ddx's `Backend` is public.** `ddxdb.Backend` (ddxdb 0.4.7) is a Python
  protocol with the primitives ddx needs: plan a query, give a table's
  schema (`select_all`), run a plan and save its rows as a table, say
  whether a plan returns rows, and drop a table. Optional methods compile a
  plan to run again, give a view's SQL (`view_sql`), give a key for a
  query's plan (`plan_key`), and report a *session key* that changes
  whenever the session would plan differently. The Rust trait
  `ddx_ad::Backend` has the same shape. Any object that implements it can
  run ddx's programs.

---

## 4. Goals and non-goals

### 4.1 Goals

1. **Run programs from any XQL frontend or transform**, through one
   interface: ddx's gradient programs, frontends' training steps, and
   plain queries.
2. **Own tables and their memory.** Every table the runtime holds has an
   owner, a lifetime and a placement (host memory, GPU memory, or inside an
   engine).
3. **Keep data where the computation is**, between operators and between
   steps, so a training loop moves only its batches and the values it
   reads.
4. **Choose an implementation for each piece of work.** The engine's own
   operators are the default. Work that einfold has recognized runs on the
   runtime's array backends, which may decline it.
5. **Place values in memory.** Decide where each value is held, and, when
   a session turns on a memory policy, which values to recompute rather
   than save (§7.5).
6. **Schedule**: run independent steps concurrently, and prepare a step's
   next batch while the current one runs.
7. **Distribute**, later: run one program across GPUs and machines (§12).

### 4.2 Non-goals

- **Its own SQL engine, in any form.** Engines give SQL its meaning, and
  the runtime never interprets relational operators itself: no joins, no
  `GROUP BY`, no NULL rules. It runs engine plans, and the array work
  einfold has recognized in them (§7.2). That rules out carrying
  `DenseProgram` into the runtime as it is.
- **Differentiation, plan rewriting, or drawing a program's step
  boundaries.** Those are ddx's and einfold's (§7.4).
- **A training framework.** Training loops and optimizers belong to
  frontends: PyTorch through torch-sql, or plain SQL in xarray-sql's demos.
  The runtime provides what a training loop needs from execution: state
  that persists, atomic updates, and checkpoints of that state. It knows
  nothing about learning rates or schedules.
- **Bitwise agreement across implementations.** Results agree with the
  engine's SQL result to a stated floating-point bound (§7.3).

---

## 5. Principles

**R1. The SQL is the meaning.** A program's result is what its SQL
computes on its reference engine (§17.5). A kernel, an array backend or a
GPU engine is an implementation of that meaning, checked against it. It
declines what it can't implement exactly. A substitution never changes a
result beyond the floating-point bound of §7.3, and never changes which
rows exist.

**R2. Public interfaces only.** The runtime depends on its neighbours'
released, public APIs, and its neighbours depend on its public API. No
private imports across repositories, and no forks.

**R3. Programs, not queries.** The unit of work is a program, usually a
whole training step. A program is compiled once and run many times, so
planning is paid once (§3.4), and a runtime that sees the whole step can
keep its state in place.

**R4. Every table has an owner.** The runtime knows who holds each table
and when it may be dropped. Frontends hold handles, not bare names.

**R5. State changes atomically.** Tables that persist across runs, such as
weights and optimizer state, take their new values only when a run's
checks pass. A failed run leaves them as they were.

**R6. Each layer stays optional.** A frontend can run without the runtime,
on a plain engine, as torch-sql does today. The runtime runs on a CPU-only
machine with no kernels at all. Each capability (array backends, a GPU,
scheduling) is opt-in and measured on its own.

**R7. Where a value is held is never part of the program's meaning.**
Frontends and ddx see the program as written. Keeping a value in an engine
table, in host memory or on a device is the runtime's choice, and changes
neither results nor gradients. Recomputing a value instead of saving it is
different: it does more work, so it happens only under a policy the user
turns on, and only for plans that give the same rows each time they run
(§7.5).

**R8. Declines are visible.** Every time an implementation declines work,
the runtime counts it, with the reason and the name of the code it came
from (§6.4). A *strict* mode turns declines into errors, for benchmarks. A
*checking* mode runs the reference beside the substitute and compares them
(§7.3).

**R9. Reproducibility is stated.** Each implementation says whether the
same inputs give the same bits. Most don't by default: a GPU kernel that
adds with atomic operations, or a `SUM` on DataFusion, which adds
partitions in the order they arrive. A session may require
reproducibility, and then gets only the implementations that can give it,
such as one partition per aggregate on DataFusion.

---

## 6. The model

### 6.1 Programs

A **program** is a directed acyclic graph of **steps**. Each step is a plan
that reads tables and writes one table. A program also names:

- its **inputs**: tables supplied on each run, such as a batch;
- its **state**: tables that persist across runs, each with the step that
  computes its next value (a weight and its update, say);
- its **parameters**: scalars that change between runs without changing
  the plans, such as a learning rate or a step count;
- its **outputs**: the tables the caller reads after a run;
- its **checks**: conditions a run must meet for its result to be valid.

ddx's programs fit this already: their steps, value and gradients are
steps and outputs. A plain query is a program of one step. Steps travel as
Substrait plans, which ddx already writes. Inside a compiled program, each
step may be held in its engine's own plan format.

**Programs compose.** A training step is ddx's program followed by the
optimizer's update steps, which read its gradient tables and write the next
state. In torch-sql these are separate today: ddx's program runs at
`backward()`, and the update runs later, a query per tensor (§3.3). The
runtime takes both as one program, so that the step is compiled once and
the gradients never leave it.

**Checks come from two places.**
- ddx's checks each state a promise: that some key columns of one table
  are unique. None reads a value. The runtime decides when to run each:
  once per compile for a table whose keys don't change, on each run
  otherwise, and never when a frontend's fact gives the same promise. In a
  training step only the weights are differentiated, so a compiled step
  runs none of them.
- einfold's checks are about the values of this run: dense coordinate
  ranges, unique tuples, no NULLs, finiteness (§7.2). Facts about values
  hold for one run only. [einfold#39](https://github.com/xqlsystems/einfold/issues/39)
  found that xarray-sql's values change between queries without the table
  being registered again. So a compiled program nearly always keeps some
  data checks, and §7.3 says how they run on a device.

**Parameters** need a form every engine accepts. Substrait has a dynamic
parameter expression, but neither the engines' consumers nor ddx have been
tested with one. A one-row table of parameters works everywhere: the plans
cross join it, and the runtime writes new values into it before each run.
ddx treats it as a constant, since it isn't differentiated, and a compiled
program reads the new values. torch-sql already reads its non-finite
constants from such a table. On DuckDB, ddx has two bugs in one-row tables,
ddx#264 and #265. ddx#265 gives wrong values with no error, so it comes
first among ddx#261 to #265.

### 6.2 Tables and placements

A **table handle** refers to a table the runtime owns. It carries the
table's schema, its lifetime (one run, or until the caller releases it),
and its *placement*:

- **in an engine**, as the engine's own table;
- **as Arrow**, in host memory or, with the C Device Data Interface, in GPU
  memory;
- **as a compact array**, on the host or a device (§7.2).

A table can be in several placements at once, and the runtime moves it
when a step needs it somewhere else. Reading a handle's rows is the only
way data leaves the runtime, and it is always explicit.

SQL refers to tables by name, so each handle also has a name, which is how
plans read the table. The handle is what keeps it alive. Names must be
stable for compiled programs to be found again. ddx names every table it
writes under a fresh prefix, or under a namespace its caller passes
(`Options::namespace`). With a namespace, the same plan gives the same
program byte for byte, so the runtime always passes one.

A compiled plan keeps the tables it read when it was made, which matters
for state. ddx's Rust path swaps in fresh scans on each run (`LiveExec`),
so a table registered again is read correctly. Its Python path can't, so
ddxdb's `in_place` mode is a promise from the caller: a table changes only
by writes into the memory it already holds, and is never registered again.
Publishing new state by swapping in a second copy (R5) breaks that promise.
So, from M1, an engine table that holds state must read the handle's
current buffer when its plan runs, or the runtime's engine side must be in
Rust (§17.2).

### 6.3 Compiled programs

A program depends on the structure of its plans and on the schemas of its
tables, not on their values. The runtime **compiles** a program once and
runs it many times with new inputs and parameters. Compiling plans each
step, chooses implementations and placements, and runs the checks that
need running only once. ddx's `prepare`, and the `compile` method of its
`Backend`, already work this way, and so did `TrainingSession`'s
whole-step compilation. A compiled program is reused only while the
session would plan every step the same way, which ddx tests with the
backend's session key.

### 6.4 Names

Three things need a name on a part of the forward program that survives
planning, and that reaches the steps ddx writes from it:
- ddx#251's regions, and the save points and recompute policies of §7.4
  and §7.5;
- provenance: which forward relation each backward step comes from
  ([ddx#259](https://github.com/xqlsystems/ddx/issues/259)), so that a
  kernel can be chosen for a backward step, such as a convolution's
  gradient with respect to its weights;
- einfold's hints (§7.2).

One marker serves all three. CTE and view names don't survive either
engine's Substrait producer (torch-sql's spike S3), so the marker is a
function call, `ddx_name(expr, 'label')`, proposed on ddx#251. In ddx's SQL
path, which reads the SQL itself, CTE names serve too. Frontends choose the
labels: torch-sql would use module paths such as `unet.layers.3`. The
runtime keys its counters (R8) by the same labels, so a decline is reported
against the module that caused it.

---

## 7. Execution

### 7.1 Three ways to run a step

A step can run in three ways, and the runtime chooses per step:

1. **On the engine**, as a plain query. This is the reference (R1), and
   always available.
2. **As einfold nodes on the runtime's backends.** einfold marks the array
   work it recognizes in a step's plan. The runtime runs each node on
   compact arrays it keeps resident, or declines it, and then the node's
   SQL runs on the engine. This is stage B-0 of §7.2.
3. **As a region**: several nodes, and the elementwise expressions between
   them, run as one array program, with their intermediates kept as
   compact arrays. This is stages B-1 and B-2 of §7.2.

A GPU SQL engine such as Sirius would be a fourth option. But it is an
engine, so it enters through way 1. §3.1 suggests that a GPU engine passing
long-format rows between its operators would meet the same wall the MLX
kernels did. Spike 2 tests that (§16).

### 7.2 Dense execution: where SQL becomes arrays

Compact arrays gave the largest single gain (§3.1). The question is which
layer turns a relational plan into array operations, and which runs them.
There are three designs.

**A. The runtime interprets relational plans on compact arrays.** This is
`DenseProgram` as it is today, moved into the runtime. The runtime would
walk DataFusion's logical operators and implement each on arrays: joins as
gathers, `GROUP BY` as reductions, `UNION` with bag semantics, NULLs as
masks. It works, and it is measured. But it is a second implementation of
SQL's semantics, beside the engines'. Every operator, NULL rule and edge
case would have to agree with every engine, forever. §3.3 shows how hard
that is even between two mature engines. It also breaks the first
non-goal of §4.2.

**B. einfold marks the SQL, and the runtime runs what it marks.** This is
the proposal.

What einfold does today has two parts:
- **It proves the plan's structure before it runs.** For example, that an
  aggregate is a `SUM` of products over an inner equi-join, with each
  factor reading one side.
- **It checks the data at run time.** `EinFoldExec` checks dense
  coordinate ranges, unique tuples, NULLs and finiteness on the rows as
  they arrive. Where a check fails, it runs the engine's own plan, which
  every node einfold detects keeps.

einfold reads no facts from frontends yet. Its *fact table*, through which
frontends would state such properties so that einfold can skip checking
them, is designed (einfold's design §8.1) but not built.

Under B, the program stays SQL. SQL is what every engine runs, what ddx
differentiates, and what R1 calls the meaning, so it is already the most
portable form a program can take. Alex has decided that StableHLO, the
operation set JAX and XLA compile, is not XQL's representation, and that
SQL is. einfold's IR annotates the SQL rather than translating it. A
*region* is a SQL plan in which einfold has marked nodes of five kinds:
- *folds*: sums of products over a join, with their domains, NULL rules
  and fallbacks;
- *pipelines*: a fold that reads another fold's result through an
  expression, which einfold fuses as FlashAttention does (FlashAttention
  computes attention one tile at a time, without holding every score);
- *shifted folds*: folds with shifted outputs, which is how a convolution
  appears ([einfold#155](https://github.com/xqlsystems/einfold/pull/155));
- gathers on value indexes, such as an embedding's lookup of token ids;
- the elementwise expressions between them.

Each node states its named axes and the three states each cell can be in
(absent, NULL, or a value), and carries the subplan it stands for. The node
types are versioned, and einfold publishes them with a conformance suite,
as its design already plans for its `Fold` Substrait relation (einfold's
design §7.2). A backend that can't run a node declines it, and the node's
SQL runs. How a backend lowers a node, to NumPy, MLX, `torch.compile` or
JAX, is private to the backend and crosses no interface. The runtime
depends on einfold's published IR types, which follows P-D.

Two reasons rule out a separate array language, whether einfold's own, the
Python array API standard or StableHLO:
- **Existence isn't a mask on values.** SQL's rules for the three states
  don't reduce to `where`. A group exists only if some row reached it.
  `SUM` over no rows is no row, not `0`. A left join from a domain turns
  absence into NULL, or into a default such as ddx's `0.0`. The sign of a
  zero sum follows the engine: DataFusion's `SUM` starts from `+0.0`.
  einfold's property tests check each rule, bit for bit, against
  DataFusion. An array language with masks would make every backend
  reimplement them, which is design A's second copy of SQL, one layer down.
- **Primitive operations lose the structure backends need for speed.**
  Attention lowered to `einsum`, `exp` and `einsum` holds all `N²` scores,
  which demo 1's fused kernel exists to avoid. A convolution lowered to
  gathers becomes an index table and a gather, where a shifted fold is one
  matrix product per filter position (einfold#155).

B has a cost. SQL is portable in meaning, not in speed: everything fast
depends on einfold recognizing structure, and whatever it doesn't
recognize runs at the engine's speed. That is the right way to fail, but
it makes einfold's coverage the number to watch (spike 6).

**What torch-sql's lowerings ask of B.** Three kinds of step in
torch-sql's nanoGPT need care:
- **Gathers whose index is a value.** An embedding is a join on a token
  id. If an id falls outside the table, SQL's inner join drops the row,
  where a gather would fail or read the wrong memory. So a gather needs a
  fact that every id lies in range, or a data check on every run.
- **Floating-point edge cases.** torch-sql writes PyTorch's behaviour out
  in SQL (NaN checks in comparisons and in `MAX`, guards around `ln` and
  `sqrt`), because the engines differ (§3.3). Those guards are `CASE`
  expressions that einfold's elementwise nodes must carry exactly.
- **Shape-changing steps.** torch-sql writes reshapes, slices and
  convolutions as joins with *index tables*, small tables that map each
  output position to an input position. Reshapes and slices become gathers
  on those tables, and convolutions shifted folds.

**What B's nodes need to know.** For a node to run without its checks,
einfold needs these facts about its inputs:
- each dimension's extent, with integer keys `0 .. n - 1`;
- completeness, or a table of the domain;
- unique tuples;
- no NULL values;
- for gathers, that each index lies in range;
- for sums einfold reorders, that values are finite and bounded (its float
  guard, einfold's design §8.6).

torch-sql can state the structural ones for every table it writes (§3.3),
and einfold#39 states them for xarray-sql. No fact table carries any of
them yet, so the fact table is on B's critical path. The runtime should
state facts about the tables it writes too. A frontend may also pass a
*hint*, the array meaning it already knows, such as the PyTorch op that
produced a node. einfold uses a hint to guide detection, never in place of
its proof. A hint einfold fails to confirm is a counted decline (R8), which
measures detection's coverage for the first time.

**Stages.** B comes in three stages, each measured and useful alone:
- **B-0, one node at a time.** einfold already detects the node. Instead of
  running it on Arrow rows in `EinFoldExec`, einfold hands the runtime the
  node's IR, its run-time checks, and the engine's plan for the fallback.
  The runtime runs the node on compact arrays it keeps resident. This
  replaces einfold#156's interface, which hands over one call at a time,
  and needs no new detection.
- **B-1, regions within one step.** einfold nodes connected through
  elementwise projections in one plan become one array program. This is
  most of a forward pass.
- **B-2, regions across steps.** This is einfold's *program mode*, in which
  it sees a whole program rather than one query. It needs ddx's step graph
  (ddx#157) and regions (ddx#251).

einfold's roadmap puts program mode at its M7, after DuckDB (M4), facts
(M5) and sparse kernels (M6). B pulls facts and program mode forward, which
is a decision for Alex as einfold's maintainer, and einfold's rules require
a sketch in its own design doc before code.

**Where B-0 keeps values resident.** In B-0, a node's neighbours inside a
step are engine operators, which read rows. So a node's output stays a
compact array only when it is the step's result, and a node reads a compact
array only when its input is a scan of a runtime table. Between those,
values become rows. B-0's gain therefore depends on how many steps are one
node from their scans to their result, as well as on how much time is
spent inside nodes, and spike 6 measures both. The runtime's tables should
be DataFusion table providers that can be read two ways: as Arrow batches,
for engine operators, and as compact arrays, for einfold nodes that scan
them directly.

**C. The engine carries dense values itself.** If intermediates could
travel between DataFusion's own operators as fixed-shape tensor columns, in
GPU memory through the C Device Data Interface, the engine would keep SQL's
semantics, and the runtime would provide only kernels. torch-sql's spike S4
found that DataFusion keeps such a column through projections, filters and
sorts without copying it.

But a tensor-valued column changes what a table is: a row holds a tile of
an array rather than one element. A matrix product is then a function on
tiles, not a join and a `GROUP BY`. That moves XQL from long format to
tiled tables, which changes every frontend's schema, and ddx's rules too.
It may be the right long-term representation, and it would make
distribution by tiles (§12) natural. But it is a change to the whole stack,
not to the runtime.

For torch-sql, tiles would be a redesign of its lowerings, though not of
its interfaces. The device, the op table and the layout code describe a
tensor by its shape and dtype, so a tile layout could sit behind them. But
every lowering writes arithmetic on one value per row, and would have to
become functions on arrays, each of which ddx would then need a derivative
rule for. The larger cost is absence. A tile is present or absent as a
whole, so a coastline cuts tiles in half, and land cells come back as
values in a mask: the dense box that torch-sql's design rejected StableHLO
for. Tiles along one trailing dimension only, such as an embedding's 384
features per token, would keep cell-level absence for the other
dimensions, and may be where C starts. Spike 1 tests C.

### 7.3 Checks, declines and fallbacks

**Checks on a device.** On a device, a check is a reduction followed by a
copy to the host, which a compiled step can't branch on while it runs.
`TrainingSession` therefore declined to compile any step whose checks read
device data. Under B, each of einfold's run-time checks compiles instead to
an on-device predicate. The predicates are ANDed into one validation flag,
which the host reads once per run, as §3.1's resident step did. A run whose
flag fails is a decline at run time.

**Declines.** A node that declines when the program compiles runs its SQL
on the engine, and the rest of the program keeps its implementations. A
decline at run time leaves the state untouched (R5), and runs the failed
nodes' SQL on the engine from their inputs. So those inputs must stay
readable by the engine until the flag has been read, which costs memory
(§8).

**Counting and checking.** Following R8, the runtime counts each
implementation's runs and declines, by reason and by name (§6.4).
Benchmarks run strict, so that a decline fails loudly rather than slowing a
result silently. A checking mode runs the engine beside a substitute on
sampled steps, compares the two to the bound below, and verifies the facts
each node relied on. That is how xarray-sql's demos and einfold's tests
already check every substitution, and §3.3 shows why it must stay possible
in production.

**The floating-point bound.** A fixed tolerance isn't enough. A float sum
of `n` terms, added in any order, is within `(n − 1)·u·Σ|terms|` of the
exact sum, as long as nothing overflows (einfold's design §8.6). Here `u`
is the *unit roundoff* of the precision the backend computes in: 2⁻⁵³ for
float64 and 2⁻²⁴ for float32. Computing in float32 on a GPU is then an
explicit choice of `u` for a session, as einfold#156 already makes
precision the host's choice. einfold's kernels give the same bits for any
thread count; the runtime's rule on reproducibility is R9.

### 7.4 Step boundaries and saving

**ddx draws the boundaries.** ddx already decides where its programs'
steps begin and end (§3.4). A second chooser with its own budget would undo
the first, so the runtime takes ddx's program as ddx writes it, and changes
its boundaries only through ddx's inputs:
- **save points and recompute policies**, by name (§6.4), passed to ddx as
  options. This is one surface with ddx#251's regions: a name on a
  relation, with a policy for each name.
- **engine limits**, such as the depth a consumer can decode and a planning
  budget, which the runtime's `Backend` reports as facts and ddx reads in
  place of its constants.

ddx doesn't ask its `Backend` where to save. `prepare` caches a program by
its plan and the session key, and an answer that depended on the runtime's
memory at that moment would leave a cached program stale.

**Shared subplans** can't be found in a Substrait plan, since the copies
are already written out (§2.1). They come from SQL before it is planned, as
ddx#168 finds them, or from the frontend's own program. ddx can offer
ddx#168's split as a public function for plain queries, so that the runtime
has no second implementation of it. `SharedExecutor` parsed SQL to find
CTEs, and had to turn ddx's plans back into SQL through a private function
to do it. Neither is needed. All of this should shrink once DataFusion
computes a CTE read several times only once (xarray-sql#267, on
DataFusion's roadmap for late 2026).

**Saving with einfold.** Saving a value can destroy structure einfold
fuses. If attention's scores are saved, no pipeline is left to fuse, and
attention's fused backward pass
([einfold#135](https://github.com/xqlsystems/einfold/issues/135)) needs
ddx to recompute the scores in each backward step instead (ddx#251). The
reverse also happens: inlined views make einfold's matching worse. Demo 4
as views is 3,862 operators, and the GPT's loss planned 2.7 times longer
with einfold on, through 9,882 lines of inlined plan
([einfold#81](https://github.com/xqlsystems/einfold/issues/81)), before
einfold#86 to #92 fixed most of that. So the runtime chooses save points
together with einfold. einfold reports, for a candidate save point,
whether it would split a node einfold can fuse, and the runtime prefers
save points that don't. A split taken anyway counts as a decline with that
reason.

**torch-sql's cut points retire.** torch-sql chooses cut points and runs a
chain rule across them (§3.3) only because it hands ddx each loss as one
Substrait plan. It has two ways out, and neither waits for the runtime:
hand ddx SQL with a CTE at each node the loss reads more than once, which
ddx#168 differentiates through, or hand ddx its segments as one program.
torch-sql's graph already knows which nodes are shared, so it will try the
SQL path first. Then torch-sql keeps only two things:
- **hints for norms.** A norm reads its input three times. ddx finds such
  reads in the SQL, and torch-sql can name the rest (§6.4).
- **stop-gradients.** torch-sql saves a detached value as a table today
  so that ddx treats it as a constant. By R7, holding a value somewhere
  can't change a gradient, so torch-sql must use `ddx_stop_gradient`
  instead. That works only on DataFusion and only for `DOUBLE` today
  ([ddx#260](https://github.com/xqlsystems/ddx/issues/260)), which is
  ddx's next fix.

### 7.5 Recomputing

Alex decided on 2026-11-01, for ddx#251, that recomputing is an explicit
policy people opt into, as with `jax.checkpoint` (JAX's marker for a
computation to recompute in the backward pass), and never the engine's own
choice. Where a value is held changes no work, so it stays the runtime's
choice (R7). Recomputing does more work, so the user chooses it, by name
(§6.4). The runtime applies a memory-budget policy only when the session
turns one on, and may always save more than a policy asks.

Recomputing gives the same result only for a plan that gives the same rows
each time it runs, and then only within R1's bound, since a `SUM` may add
in a different order. A volatile function, a `LIMIT` without a total order,
or a rank over ties rules it out. ddx already refuses to compute such a
plan twice ([ddx#254](https://github.com/xqlsystems/ddx/pull/254)), but it
tests for volatility with a list of function names. It should read
volatility from the engine through its `Backend` instead. The random
function of §10 must be declared deterministic, so that a recomputed
dropout draws the same mask.

`torch.utils.checkpoint`, PyTorch's way of trading memory for
recomputation, becomes a recompute policy on a named region. It also
preserves the random state, so that a recomputed dropout matches. The
runtime's counter-based random function (§10), or a mask uploaded as an
input, gives that for free.

---

## 8. Memory

Deciding where to hold each value, and what to save or recompute, is one
trade-off whatever the reason:
- planning (a value saved so that a plan stays small, on DataFusion);
- memory (gradient checkpointing);
- a slow link (a value in Grace memory against one in GPU memory).

The runtime decides from the engine's limits, the memory of each placement
on the hardware, and the policies a session names (§7.5). It holds values
where it likes, and recomputes only under a policy.

**Two copies of state.** Atomic state (R5) has a memory cost: the old state
must survive until the new state's checks pass, so state needs room for
two copies while a step runs: weights, optimizer moments and their
replacements. For a Samudra on a quarter-degree grid, with four times the
cells of today's, that is worth budgeting on one B200.

The second copy can be avoided table by table, as JAX's *buffer donation*
lets a compiled function write its result into an input's memory. That
keeps R5 only if no check can fail after the first table is overwritten.
In a training step, the checks are on the batch and the gradients, which
come before the optimizer's update. So a compiled step reads its
validation flag once, after its last check and before its first state
update, and the updates then write in place. Failures that aren't checks,
such as an engine error or running out of memory, can still strike after
that, so a program that donates restarts from its last checkpoint (§10).

On the Beta cluster, each GB200's Grace memory is a placement a GPU can
read directly, which makes spilling there cheaper than on most machines.
Whether a GPU reads Arrow buffers from Grace memory at a useful speed is
untested (spike 3).

---

## 9. Scheduling

Within a run, the runtime runs each step once the steps it reads are done.
ddx's run loop is sequential today, in Python and in Rust, so this needs
ddx to expose which earlier steps each step reads (ddx#157), and a
`Backend` whose `materialize` may run concurrently. ddx#157's estimate of
16 to 43 percent was measured before programs were compiled, so it needs
measuring again before M4.

Across runs, the runtime overlaps a step's computation with loading the
next batch. On Samudra, the batches come from xarray-sql's data path over a
Zarr archive, so the runtime is also where the data path and the training
step meet.

---

## 10. State, checkpoints and randomness

xarray-sql#307 proposed splitting `TrainingSession` along the line between
execution and training:

| Part of `TrainingSession` | Goes to |
|---|---|
| State tables kept as arrays between runs, with their device buffers | runtime (§6.2) |
| Publishing new state only after validation, keeping the old state on failure | runtime (R5) |
| Compiling the whole step once, and guarding reuse with the session key | runtime (§6.3) |
| Reusing the loss ddx already computed, instead of a separate forward query ([ddx#257](https://github.com/xqlsystems/ddx/pull/257)) | runtime, through ddx's public API |
| Checkpoints of state, with the step counter and random state | runtime (below) |
| Dropout with explicit seed and step keys | runtime provides the random function; the frontend decides where dropout goes (below) |
| AdamW, gradient clipping, the learning-rate schedule and evaluation | frontend: xarray-sql's demo, in SQL; torch-sql, from `torch.optim` |

**Checkpoints.** `TrainingSession` writes NPZ files (NumPy's archive of
arrays). The runtime should write state in a format the other layers
already read. Zarr, through xarray-sql, makes a checkpoint a Dataset that
SQL can query, which torch-sql's design plans for its weights. Arrow IPC,
Arrow's file format, is the alternative for state that isn't array-shaped.
A checkpoint is written from published state only, so it is always
consistent.

**Randomness.** Dropout needs random numbers that every implementation of
a step agrees on, the engine included, or checking mode (§7.3) can't
compare them. The runtime should provide a *counter-based* random function:
a hash of a seed, the step number and the element's coordinates. Each
backend implements it identically, and the engine implements it in
portable SQL. The engines make that hard: on integer overflow, DuckDB
raises an error and DataFusion 55 wraps around silently. So the hash must
be written in integer arithmetic that never overflows. `TrainingSession`'s
GPU dropout used MLX's own random keys, so its masks differed from the CPU
path's, and checking it against PyTorch meant exporting the masks.

A frontend that must reproduce another framework's random numbers can't use
that function. torch-sql checks nanoGPT against PyTorch step by step, so it
draws each dropout mask with PyTorch's own generator on the CPU and uploads
it as an input table. Its masks then match PyTorch's bit for bit, at the
cost of one upload per mask. The runtime supports both: its own random
function for programs that own their randomness, and masks as inputs for
frontends checked against another framework.

---

## 11. Interfaces

### 11.1 To frontends

A sketch, to make the model of §6 concrete. The names are placeholders:

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
compiled = session.compile(step, strict=True)      # declines are errors (R8)
for batch in loader:
    out = compiled.run(inputs={"x": batch}, parameters={"lr": 6e-4})
    print(out["loss"].to_pyarrow())                # the only read
print(compiled.statistics())                       # runs and declines, by reason and name
```

torch-sql would move in two stages. At M0 it needs no new API: the runtime
implements `ddxdb.Backend`, so torch-sql's engine wrapper
(`torch_sql.engines`) hands ddx the runtime instead of an engine. From M1,
a synced `device="sql"` tensor holds a table handle instead of a table name
and a Python finalizer, and a whole training step, from the loss to the
optimizer's update, becomes one program (§6.1). Its cut points
(`torch_sql.program`) retire sooner, through ddx's SQL path (§7.4).

xarray-sql would keep registering Datasets and turning results back into
Datasets, and would hand the runtime a program when a user asks for one,
for instance through `Context.prepare`.

### 11.2 To ddx

The runtime implements `ddxdb.Backend`, and later `ddx_ad::Backend` in
Rust, so every ddx program runs on it unchanged. Once it does, `Backend` is
an interface between projects: ddx lists changes to it in its changelog,
and the runtime pins ddxdb exactly, as ddxdb pins DataFusion.

From ddx, the runtime needs:
- **the step graph** on `Program`, for scheduling and lifetimes (ddx#157);
- **names and policies**: ddx#251's fourth part, with save points beyond
  "read more than once" (§6.4, §7.4);
- **engine limits read from the `Backend`**, in place of ddx's constants
  (§7.4);
- **volatility read from the engine** through the `Backend` (§7.5);
- **what each check promises** (§6.1);
- **provenance** (ddx#259, §6.4);
- **the primal outputs of a prepared run** (ddx#257, in review), so the
  loss comes from the backward program instead of a second forward query;
- **the derivative of `erf`**
  ([ddx#256](https://github.com/xqlsystems/ddx/pull/256), in review), for
  the exact GELU of nanoGPT and Samudra;
- **`ddx_stop_gradient` on every engine and type** (ddx#260, §7.4);
- **`to_sql` exported from ddxdb.** ddxdb runs each DuckDB step as the SQL
  `to_sql` writes. `to_sql` is public in Rust and private in ddxdb.

None of these waits for the runtime: each is an ordinary ddx PR. ddx#260
comes first.

### 11.3 To einfold

The runtime is einfold's host. For B-0 (§7.2), einfold hands it each node's
IR, the node's run-time checks and the engine's plan for the node, and the
runtime runs the node or declines it. That replaces einfold#156's
interface. Later stages hand it regions. Beyond that, the runtime needs
from einfold:
- **published, versioned IR types**, with a conformance suite (§7.2);
- **the fact table** (einfold's M5), on B's critical path (§7.2);
- **a report on save points**: whether a candidate splits a node einfold
  can fuse (§7.4);
- **hints**, through the names of §6.4.

### 11.4 To engines

The runtime holds an engine session and runs plans on it: DataFusion
first, since xarray-sql and einfold live there, then DuckDB, then GPU
engines. xarray-sql, ddx and einfold need one DataFusion version for this
to work, which
[xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)
tracks.

### 11.5 Transport

Tables cross into and out of the runtime as Arrow: through the PyCapsule
interface on the host, through the C Device Data Interface on a GPU, and
through DLPack toward PyTorch. Today nothing in the stack exports a device
stream, and PyTorch reads no Arrow capsule, so the device path needs both
(torch-sql's spike S4).

---

## 12. Distribution (later)

A database spreads a query across machines by partitioning its tables, and
moving rows between partitions where a join or an aggregate needs them. In
a training step, the partitioning key gives the usual kinds of parallelism:
- partitioning by batch index is data parallelism;
- by a weight's index, weight sharding;
- by grid tile, *domain decomposition*, as numerical ocean models run:
  each GPU owns a tile and exchanges only the cells along its edges with
  its neighbours.

torch-sql's hypothesis H1 is that a planner choosing the partitioning per
join, from the whole step's plan, distributes a model like Samudra better
than PyTorch's FSDP (Fully Sharded Data Parallel, which splits a model's
weights across GPUs) does.

That choice belongs to the runtime. The first versions don't distribute,
but the program model must not prevent it: steps and tables need to carry
the partitioning facts a distributed planner would use, and einfold's
nodes their named axes, so that a node can be split along one. On the
Beta cluster, one NVL72 rack holds a whole Samudra run, so the first
distribution never crosses a rack.

---

## 13. Where today's code goes

The executors of §3.1 exist as draft PRs. This is where each part would
land, so that none of it is lost and none of it stays in the wrong layer:

| Code | Today | Goes to |
|---|---|---|
| `prepare()` for gradients, `Context(threads=)`, faster registration (#301) | xarray-sql | stays in xarray-sql |
| `SharedExecutor`'s stage caching | xarray-sql (#302) | ddx's SQL path (ddx#168) and the runtime's save points (§7.4) |
| `SharedExecutor`'s SQL parsing of CTEs (`src/shared.rs`) | xarray-sql (#302) | nowhere: ddx#168 already does this |
| Arrow tables that declare their sort order (`_SortedArrowTable`) | xarray-sql (#302) | xarray-sql's registration, which knows a Dataset's dimension order |
| MLX kernels (`xarray_sql/mlx.py`) | xarray-sql (#302) | runtime, as an array backend for einfold's nodes |
| `DenseProgram` | einfold (#157) | a prototype to learn from: a corpus for spike 6, and an oracle for einfold's new detection, which belongs in Rust beside einfold's existing detection |
| `DenseExecutor` and its MLX operators | xarray-sql (#303) | runtime, as array backends for B-0 |
| `TrainingSession` | xarray-sql (#305) | split as in §10 |
| `erf` as a DataFusion function (`xarray_sql/math.py`) | xarray-sql (#305) | xarray-sql for now; DataFusion upstream eventually, since it has no `erf` |
| The Shakespeare recipe, checked against PyTorch step by step (`03_shakespeare.py`) | xarray-sql (#305) | xarray-sql, as demo 3, on public APIs |
| torch-sql's cut points and chain rule (`torch_sql.program`) | torch-sql | nowhere: ddx's SQL path replaces them (§7.4) |

---

## 14. Name and packages

### 14.1 The name

The project is **`xql-runtime`**: the package and repository
`xql-runtime`, imported as `xql_runtime`, free on both PyPI and crates.io.
It says what the project is in the vocabulary the doc already uses, the
runtime of XQL, and follows the plain naming of xarray-sql and torch-sql.
It also keeps XQL's habit of avoiding the word *tensor* for N-dimensional
arrays.

The alternatives considered:
- **`sql-host`.** einfold's design already uses *host* for whatever runs a
  plan, usually the engine, so the name would blur the runtime with the
  engines.
- **`array-runtime`.** Generic: it could be any array library's runtime,
  and nothing ties it to XQL.
- **A coined name, in the spirit of ddx and einfold**, such as `einrun` or
  `foldhost`. These read as einfold's own executor, which B rejects.

### 14.2 Packages

The first releases are one pure-Python package, `xql-runtime` on PyPI
(§17.2). It grows into the layout ddx and einfold use: one Python package,
built with maturin, carrying a Rust core, and a Cargo workspace split by
dependency. The crates' names are reserved until the first Rust code
lands, and moving the build to maturin then changes nothing for users.

| Channel | Package | Contents |
|---|---|---|
| crates.io | `xql-runtime` | the program model, table handles, placements, the scheduler and the counters. No engine or array library: an `Engine` trait and an `ArrayBackend` trait. |
| crates.io | `xql-runtime-datafusion` | the DataFusion engine. It implements `ddx_ad::Backend`, and hosts einfold's nodes. It pins the one DataFusion version that xarray-sql, ddx and einfold share (xarray-sql#310). |
| crates.io | `xql-runtime-duckdb` | later |
| PyPI | `xql-runtime` | the bindings, with extras: `[datafusion]` and `[duckdb]`, `[mlx]` (Apple silicon only) and `[cuda]` |

Some rules:
- **The base install has no engine and no GPU library** (R6).
- **Engines and ddxdb are pinned exactly**, as ddxdb pins its engines
  (§11.2).
- **Array backends that are Python libraries** (NumPy, MLX, and the CUDA
  backend's library, §17.1) **start as Python code behind a Python
  protocol**, so that adding one never needs a Rust build. A backend moves
  into Rust only when spike 4 shows it must.
- **Dependencies point toward the runtime.** On the Python side, the
  runtime implements `ddxdb.Backend` structurally, since it is a protocol.
  Frontends depend on the runtime only through an extra, such as
  `xarray-sql[runtime]`.
- **Wheels for Linux on ARM from the first release.** The Beta cluster's
  Grace CPUs are ARM, and torch-sql's spike S7 found ARM packaging is where
  installs break.

---

## 15. Milestones

Each milestone is measured on three programs, and reported whichever way
the numbers go, on [xarray-sql#285](https://github.com/xqlsystems/xarray-sql/issues/285),
with the hardware and versions:
- demo 2, the GPT;
- demo 4, Samudra's U-Net;
- torch-sql's nanoGPT.

GPUs come early. §3.1 shows a GPU pays off only together with resident
state and dense intermediates, so the GPU milestone (M2) brings both.
NVIDIA GPUs, on the cloud or on the Beta cluster, are available to the
project half a day a week. The contributor's Apple GPU work continues in
parallel on MLX.

- **M0. A CPU runtime.** Sessions, table handles and programs on DataFusion
  and DuckDB, implementing `ddxdb.Backend` with `compile` and tables kept
  in place, and the counters of R8. xarray-sql's demos and torch-sql's
  nanoGPT run on it, with the same results as today. Kept small on
  purpose.
- **M1. Resident state.** Persistent tables and atomic publication, on the
  host, with placements designed so that a device is one more placement,
  and engine tables that read a handle's current buffer (§6.2). torch-sql's
  training step becomes one program instead of a query per tensor.
- **M2. A GPU, with B-0.** einfold's nodes run on compact arrays resident
  on the device, CUDA on an NVIDIA GPU, with NumPy on the CPU as the
  reference. MLX follows, or comes first if the contributor's Apple work
  gets there sooner. Needs einfold's B-0 interface, the fact table for the
  facts torch-sql and xarray-sql can state, and spike 6.
- **M3. Save points and memory policies**, passed to ddx as options
  (§7.4, §7.5).
- **M4. Concurrent steps** (ddx#157), once its gain is measured again on
  compiled runs.
- **M5. B-1, regions within a step**, if spike 6 and M2 show B-0 stops
  short.
- **M6. B-2, regions across steps**, with ddx#157 and ddx#251.
- **M7. Distribution**, within one NVL72 rack.

The spikes that need NVIDIA hardware (2, Sirius; and 3, Grace memory) are
the first work for those half days, since they wait on no design decision.
M2 is the critical path. It waits on spike 6, which needs no special
hardware, and on Alex's decision to bring einfold's B-0 interface and fact
table forward. B-0 needs no new detection in einfold, so M2 is far less
exposed to einfold's roadmap than regions would be.

---

## 16. Spikes

1. **Fixed-shape tensors through DataFusion's operators.** How far a dense
   column passes through joins and aggregates without being copied or
   expanded to rows, on the host and on device buffers. This tests design C
   (§7.2).
2. **A GPU SQL engine on long-format rows.** One of xarray-sql's nanoGPT
   steps on Sirius, to test whether rows between operators cost a GPU
   engine what they cost the MLX kernels.
3. **Grace memory.** A B200 reading Arrow buffers in place from Grace
   memory, against copying them to the GPU.
4. **The runtime's language.** Three costs of a Python core:
   - whether DataFusion's and DuckDB's Python bindings release Python's
     global interpreter lock while they plan and execute, without which
     steps can't run concurrently from Python;
   - datafusion-ffi's cost, through which einfold reaches datafusion-python
     sessions. Steps einfold doesn't match run 1.25 times slower and plan
     2.9 times slower through it than on DataFusion alone
     ([einfold#58](https://github.com/xqlsystems/einfold/issues/58),
     einfold#81). Object stores don't cross it (einfold#149), and functions
     planned through it lose the session's configuration (einfold#137);
   - whether a Python table provider can read a handle's current buffer
     each time its plan runs (§6.2).
5. **Where the last 3.76× goes.** A trace of one warm resident nanoGPT step
   on the M5 Ultra, against PyTorch MPS's, both with and without
   compilation, by kind of operation. Gathers and scatters from index
   joins, unfused elementwise work, and attention's backward pass are the
   candidates. The contributor's branches have started on this ("Compare
   the GPU programs for PyTorch MPS and SQL training", and the
   Modded-NanoGPT step of §3.1), and their findings come first.
6. **einfold's coverage.** For demo 2, demo 4 and torch-sql's nanoGPT, run
   `EXPLAIN ANALYZE` (which runs a query and reports the time in each
   operator) with einfold on. Report the share of each step's time inside
   einfold's nodes, and the share of steps that are one node from their
   scans to their result (§7.2). `DenseProgram`'s record of what it ran for
   the 123.6M model is the corpus to compare against. einfold's agent runs
   einfold's side.

---

## 17. Open questions

1. **Which array library runs einfold's nodes on NVIDIA GPUs.** Lowering is
   private to each backend (§7.2), so this no longer waits on einfold, and
   more than one backend can exist. The candidates:
   - JAX, whose compiler XLA fuses whole programs. By default, JAX takes 75
     percent of a GPU's memory when it starts, so beside PyTorch on one GPU,
     as torch-sql is as a co-processor, the two allocators compete for the
     rest.
   - PyTorch, where torch-sql already lives. A node's results reach PyTorch
     with no transport, and `torch.compile` can fuse them.
   - CuPy, a thin NumPy-like library for CUDA.
2. **Rust or Python first.** The co-authors propose Python first (P-H).
   The interfaces are still moving and cheaper to change in Python; the
   costly work, engine planning, kernels and array libraries, is native
   already; and `ddxdb.Backend`, xarray-sql and torch-sql are Python.
   torch-sql's prototype is the evidence: it is all Python, and Python was
   never its bottleneck. Its slow parts are the number of queries and
   planning, both native. Its internals changed in every round of review,
   which Python made cheap. Its one bug of the kind Rust prevents, dropping
   a table still in use, came from tying a table's lifetime to Python's
   garbage collector, which explicit handles (R4) avoid in either language.

   Three conditions keep the way to Rust open:
   - table handles and their lifetimes are explicit, never tied to garbage
     collection;
   - the Python core stays a thin layer over the engines' and array
     libraries' bindings;
   - spike 4 decides.

   Three costs on the other side are known:
   - einfold reaches datafusion-python only through datafusion-ffi, which
     is slower and has gaps (spike 4). A Rust core would link DataFusion
     and einfold directly.
   - ddxdb's compiled path from Python (`in_place`, §3.4) needs an
     xqlsystems fork of datafusion-python today, which R2 rules out until
     the change is upstream.
   - Atomic state in Python needs an engine table that reads a handle's
     current buffer (§6.2). If spike 4 finds that Python can't provide one,
     Rust starts at M1, for the engine side, rather than at M4.
3. **Substrait as the plan format between layers.** ddx already speaks it.
   Engines without a Substrait producer would need SQL text in their
   dialect, which ddx tracks in
   [ddx#159](https://github.com/xqlsystems/ddx/issues/159). einfold works on
   DataFusion's logical plans, so a step may need both forms.
4. **Low precision.** SQL engines have no bfloat16, a 16-bit float format
   that GB200 hardware is built for. A backend could compute in it with
   the SQL types as the reference, with `u` chosen per session (§7.3). The
   nanoGPT work already ran in float32 on the GPU against float64 SQL. What
   is open is which nodes may compute in bfloat16, and who chooses.
5. **Which engine is the reference** when two engines disagree (§3.3). This
   draft proposes (P-J) that the engine a program runs on is its
   reference, and that disagreements between engines are found in tests,
   as torch-sql's suite runs every op on both engines, and filed against
   whichever engine departs from SQL's standard or from IEEE 754
   arithmetic. How often checking mode should sample in production is
   open.
6. **einfold's roadmap.** M2 needs einfold's B-0 interface and fact table
   ahead of einfold's DuckDB and sparse-kernel milestones (§7.2). That is
   Alex's decision as einfold's maintainer, after a sketch in einfold's
   design doc.

---

## Decision log

### Decided

- **D1** (Alex, 2026-11-01, for ddx#251). Recomputing is an explicit
  policy people opt into, never the engine's own choice (§7.5).
- **D2** (Alex, 2026-11-07). StableHLO is not XQL's representation of a
  program. SQL is (§7.2).
- **D3** (Alex, 2026-11-07). The project is named `xql-runtime`, and is
  packaged as §14.2 describes.
- **D4** (Alex, 2026-11-07). A GPU milestone, with dense intermediates and
  resident state, comes third (M2), using NVIDIA GPUs on the cloud or the
  Beta cluster.

### Proposed

- **P-A.** The runtime is a separate project. einfold stays free of device
  execution, and keeps its CPU reference operator (from xarray-sql#307).
- **P-B.** The unit of work is a program with explicit state, not a query
  (R3, R5).
- **P-C.** Design B (§7.2): the program stays SQL, einfold marks the array
  work in it and proves its structure, and the runtime runs the marked
  nodes or declines them, never interpreting relational operators. B comes
  in stages, B-0 first, and needs an einfold design change approved by its
  maintainer. Design C stays open through spike 1.
- **P-D.** The runtime implements ddx's and einfold's public interfaces,
  and depends on their published types, rather than either project
  depending on the runtime.
- **P-E.** Where a value is held is the runtime's choice and invisible to
  differentiation (R7). ddx draws step boundaries, and the runtime changes
  them only through ddx's options and the engine limits its `Backend`
  reports (§7.4). Declines are counted by reason and name, and benchmarks
  run strict (R8).
- **P-H.** Python first, with explicit handles and a thin core, until
  spike 4 shows Rust is needed (§17.2).
- **P-I.** The runtime chooses save points with einfold's report on what
  they split, and passes them to ddx as options. No frontend runs a chain
  rule of its own (§7.4).
- **P-J.** Each program's reference is the engine it runs on, and
  disagreements between engines are found in tests and filed upstream
  (§17.5).
- **P-K.** One naming marker serves regions, policies, provenance, hints
  and the runtime's counters (§6.4).
- **P-L.** Reproducibility is stated per implementation, and a session may
  require it (R9).
