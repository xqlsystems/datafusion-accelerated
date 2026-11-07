# The XQL runtime: running relational programs on devices

_author_: Alex Merose
_co-authors_: 🔥 Claude (torch-sql) and 🌀 Claude (xarray-sql), via Claude Code
_status_: Draft v0, for discussion. Nothing is built, and the name is a placeholder.
_last updated_: 2026-11-07
_origin_: [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)

---

## How this draft was written

The torch-sql agent wrote this first draft from what torch-sql's prototype
and design doc have taught us, and from the proposal in xarray-sql#307. The
xarray-sql agent is a co-author. It knows the nanoGPT performance work
(§3.1) and the executors it produced far better than this draft does, and
§15 lists the questions addressed to it. Where the two agents disagree, the
disagreement stays in the text as an open question (§16) until it is
settled, and settled questions move to the decision log.

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
program, a set of query plans together with the tables they read and write,
and the runtime runs it: it owns the memory that holds the tables, keeps
them on a device such as a GPU between operators and between steps,
chooses an implementation for each piece of work, schedules the steps, and
later spreads them across machines.

Today each frontend does this for itself, in different ways. torch-sql runs
its programs through a small engine wrapper of its own. xarray-sql's nanoGPT
work built three executors and a training session inside xarray-sql (§3.1).
einfold, which recognizes array computations in plans, has a dense backend
interface but rules out running anything itself. The runtime is where those
pieces would go, behind interfaces every frontend can use.

### 1.1 Why now

Two results say that execution, not planning or arithmetic, decides how
fast XQL runs a training step (§3):

1. In xarray-sql's nanoGPT experiments, sending kernels to the GPU alone
   changed nothing (163.6 s per training step on the GPU against 164.0 s on
   the CPU, for a 123.6M-parameter model). What helped was holding
   intermediates as dense arrays instead of Arrow rows (67 times faster),
   and then keeping weights and optimizer state on the GPU between steps
   (101 ms per step, 3.76 times PyTorch's 27 ms on the same Apple GPU).
2. In torch-sql, a whole training step of nanoGPT now matches PyTorch to
   1e-11 on two engines, but the optimizer's update alone takes 8 to 12
   seconds at six layers, because each parameter and each piece of optimizer
   state is read and written by a query of its own.

Both are questions of how data is held and moved between queries, which is
the runtime's job.

---

## 2. Background

This section introduces each project and format the rest of the doc uses.

**Apache Arrow** is a columnar memory format: a table is a set of columns,
each a contiguous buffer. Every XQL engine reads and writes Arrow, so it is
how tables move between projects. Its *PyCapsule interface* hands Arrow data
between Python libraries without copying. Its *C Device Data Interface*
extends that to buffers in GPU memory, and its canonical
[fixed-shape tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
extension type is a column whose every value is a dense array of one shape.

**Substrait** is a portable, serialized format for relational query plans.
DataFusion and DuckDB can each produce and consume it, which lets ddx read
one engine's plan and hand back plans the same engine runs.

**DLPack** is a standard for sharing a dense array between array libraries,
such as Arrow and PyTorch, without copying.

The engines:

- **[DataFusion](https://datafusion.apache.org/)** is a SQL query engine
  written in Rust and built to be extended. xarray-sql and einfold build on
  it.
- **[DuckDB](https://duckdb.org/)** is an embedded analytical database.
  ddx runs on it through DuckDB's community Substrait extension.
- **[Sirius](https://github.com/sirius-db/sirius)** and NVIDIA's **GQE**
  are SQL engines that run on GPUs.

The frontends and transforms:

- **[xarray-sql](https://github.com/xqlsystems/xarray-sql)** registers
  [Xarray](https://xarray.dev/) Datasets (labelled N-dimensional arrays) as
  tables, and turns query results back into Datasets. Its demos include a
  GPT language model trained in plain SQL.
- **[torch-sql](https://github.com/xqlsystems/torch-sql)** runs
  [PyTorch](https://pytorch.org/) programs as SQL. A tensor on its
  `device="sql"` is a table, and each operation on it becomes part of a
  query. It trains nanoGPT, a small GPT-2 model, today.
- **[ddx](https://github.com/xqlsystems/ddx)** differentiates SQL. Given a
  query that computes a loss, it writes a *program*: a sequence of steps,
  each a plan that saves its result as a table, which together compute the
  loss and its gradients. `ddxdb` is its Python package.
- **[einfold](https://github.com/xqlsystems/einfold)** is a query optimizer
  for DataFusion. It finds array computations in a plan, such as a matrix
  product written as a join and a `GROUP BY`, and runs array algorithms for
  them. Its design is explicit that it "rewrites query plans; it never
  executes them".

The hardware:

- **MLX** is Apple's array library for its GPUs. The nanoGPT performance
  work ran on an Apple M5 Ultra through it.
- **The Beta cluster** of [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
  is where XQL's first production model, the ocean emulator
  [Samudra](https://github.com/m2lines/Samudra), trains. It has 288 NVIDIA
  B200 GPUs in GB200 NVL72 racks. In a GB200 system, each Grace CPU (an ARM
  CPU) is joined to two B200 GPUs by NVLink-C2C, a cache-coherent link that
  lets a GPU read the CPU's memory directly. An NVL72 rack joins 72 GPUs
  with NVLink, NVIDIA's GPU-to-GPU interconnect.

---

## 3. What we know already

### 3.1 From xarray-sql's nanoGPT work

xarray-sql's nanoGPT experiments
([xarray-sql#302](https://github.com/xqlsystems/xarray-sql/pull/302),
[#303](https://github.com/xqlsystems/xarray-sql/pull/303) and
[#305](https://github.com/xqlsystems/xarray-sql/pull/305), with
[einfold#156](https://github.com/xqlsystems/einfold/pull/156) and
[einfold#157](https://github.com/xqlsystems/einfold/pull/157)) built a
sequence of executors, each on the one before:

| Executor | What it does | Measured |
|---|---|---|
| `SharedExecutor` (#302) | computes each CTE once per run, rather than at every read | a small model's warm step from 314 to 206 ms on the CPU |
| MLX kernels (#302, einfold#156) | einfold's recognized matrix and attention work on Apple's GPU, through a backend that may decline | no change on the large model: Arrow rows between operators dominate |
| `DenseExecutor` and einfold's `DenseProgram` (#303, einfold#157) | intermediates as *compact arrays*: dense value arrays with separate masks for which rows exist and which values are NULL | 163.6 s to 2.43 s per step on the GPU (67.2 times), and 164.0 s to 4.04 s on the CPU |
| `TrainingSession` (#305) | weights and optimizer state stay on the GPU; the whole step compiles once; new state is published only after checks pass | 2.02 s to 101 ms per step; zero arrays exported in 100 timed steps |

Two lessons follow. The representation between operators matters more than
where the arithmetic runs. And once intermediates are compact, moving state
across the Arrow boundary on every step becomes the cost, so state must
stay where the computation is.

### 3.2 From torch-sql

torch-sql runs a training step as a few large programs, and each finding
below was measured there or in its spikes:

- **DataFusion plans a CTE again at every read**
  ([xarray-sql#267](https://github.com/xqlsystems/xarray-sql/issues/267)).
  For a stack of ConvNeXt-like blocks as one query, the plan grows from 441
  operators for one block to 136,107 for three, and planning and running
  from 0.2 s to 212 s. DuckDB grows by 75 operators per block. torch-sql
  therefore chooses *cut points*, values saved as tables between queries,
  against a per-engine budget (400 inlined `SELECT` blocks on DataFusion,
  none on DuckDB). That brought one nanoGPT layer's backward pass on
  DataFusion from 9 s to 0.7 s, and two layers, which didn't fit in 8 GB
  before, run in 1.0 s.
- **Differentiating across cut points is a chain of ddx programs.** Each
  segment between cut points is one program, and each cut point passes a
  *cotangent* (the gradient of the loss with respect to that table) to the
  segment before it. That is a schedule of programs with dependencies
  between them, which a runtime could plan as one.
- **Per-table queries are the bottleneck once the math is right.** The
  optimizer's update at six layers takes 8 to 12 s because it syncs each
  tensor separately.
- **Engines disagree, silently.** Running every op on both engines found
  five ddx bugs (ddx#261 to #265), one of which gave a wrong loss on DuckDB
  with no error, and found that DataFusion's `MAX` with a `GROUP BY` skips
  NaN. Any kernel the runtime substitutes needs a reference to be checked
  against, and an engine's output is a good one only after such testing.
- **Lifetimes are hard to get right in the frontend.** torch-sql drops a
  table when the last Python object referring to it is collected, and a bug
  there dropped tables still in use. A table's owner and lifetime belong
  with whatever owns the memory.
- **Transport** (torch-sql spike S4): an Arrow array with no nulls reaches
  PyTorch through DLPack without a copy. A fixed-shape tensor column keeps
  its type through DataFusion's projections, filters and sorts, and a
  projection keeps the same buffer. DuckDB turns it into a plain list of
  fixed size, which keeps the values but loses the shape.

### 3.3 From ddx

- **Planning dominates small steps.** In xarray-sql's GPT demo, a ddx step
  takes 5.4 ms, of which about 4 ms is planning, and DataFusion's planning
  is 208 ms of a 422 ms program
  ([ddx#157](https://github.com/xqlsystems/ddx/issues/157)).
- **Steps could run concurrently.** Many of a program's steps read none of
  the others. Running each as soon as its inputs are ready would shorten
  xarray-sql's demos by 16 to 43 percent, an upper bound from each program's
  critical path (ddx#157).
- **ddx's `Backend` is public.** `ddxdb.Backend` (ddxdb 0.4.7) is a Python
  protocol with the primitives ddx needs: plan a query, run a plan and save
  its rows as a table, say whether a plan returns rows, drop a table, and
  optionally compile a plan to run again. The Rust trait `ddx_ad::Backend`
  has the same shape. Any object that implements it can run ddx's programs.

---

## 4. Goals and non-goals

### 4.1 Goals

1. **Run programs from any XQL frontend or transform**: ddx's gradient
   programs, torch-sql's segment chains, and plain queries, through one
   interface.
2. **Own tables and their memory.** Every table the runtime holds has an
   owner, a lifetime and a placement (host memory, GPU memory, or inside an
   engine).
3. **Keep data where the computation is**, between operators and between
   steps, so a training loop moves only its batches and the values it reads.
4. **Choose an implementation for each piece of work**: the engine's own
   operators by default, and kernels for the work einfold recognizes, which
   may decline it.
5. **Budget memory**: choose what to save and what to recompute against the
   memory of the hardware it runs on, which subsumes torch-sql's cut points.
6. **Schedule**: run independent steps concurrently, overlap planning with
   execution, and prepare a step's next batch while the current one runs.
7. **Distribute**, later: run one program across GPUs and machines (§12).

### 4.2 Non-goals

- **Its own SQL engine.** Engines plan and run SQL. The runtime chooses
  among implementations and holds their data, but SQL's meaning comes from
  the engine (R1).
- **Differentiation or plan rewriting.** Those are ddx's and einfold's.
- **A training framework.** Training loops and optimizers belong to
  frontends: PyTorch through torch-sql, or plain SQL in xarray-sql's demo.
  The runtime provides what a training loop needs from execution (state
  that persists, atomic updates, checkpoints of that state), and nothing
  about learning rates or schedules.
- **Bitwise agreement across implementations.** Results agree with the
  engine's SQL result to a stated floating-point bound.

---

## 5. Principles

**R1. The SQL is the meaning.** A program's result is what its SQL computes
on the reference engine. A kernel, a compact array or a GPU engine is an
implementation of that meaning, checked against it, and it declines what it
can't implement exactly. A substitution never changes a result beyond the
stated floating-point bound, and never changes which rows exist.

**R2. Public interfaces only.** The runtime depends on its neighbours'
released, public APIs, and its neighbours depend on its public API. No
private imports across repositories, and no forks.

**R3. Programs, not queries.** The unit of work is a program, usually a
whole training step, because planning per query costs more than computing
(§3.3), and because a runtime that sees the whole step can keep its state
in place.

**R4. Every table has an owner.** The runtime knows who holds each table and
when it may be dropped. Frontends hold handles, not names.

**R5. State changes atomically.** Tables that persist across runs, such as
weights and optimizer state, take their new values only when a run
finishes and its checks pass. A failed run leaves them as they were.

**R6. Each layer stays optional.** A frontend can run without the runtime on
a plain engine, as torch-sql does today, and the runtime runs on a CPU-only
machine with no kernels at all. Each capability (compact arrays, a GPU,
scheduling) is opt-in and measured on its own.

---

## 6. The model

### 6.1 Programs

A **program** is a directed acyclic graph of **steps**. Each step is a plan,
in Substrait, that reads tables and writes one table. A program also names:

- its **inputs**: tables supplied on each run, such as a batch;
- its **state**: tables that persist across runs, each with the step that
  computes its next value (a weight and its update, say);
- its **parameters**: scalars that change between runs without changing the
  plans, such as a learning rate or a step count;
- its **outputs**: the tables the caller reads after a run;
- its **checks**: plans that must return no rows for a run to succeed, as
  ddx's programs already carry.

ddx's programs fit this already: their steps, value and gradients are steps
and outputs. torch-sql's segment chain (§3.2) is several ddx programs whose
cotangents connect them, which the runtime would join into one graph. A
plain query is a program of one step.

### 6.2 Tables and placements

A **table handle** names a table the runtime owns: its schema, its lifetime
(one run, or until the caller releases it), and its *placement*:

- **in an engine**, as the engine's own table;
- **as Arrow**, in host memory or, with the C Device Data Interface, in GPU
  memory;
- **compact**, as dense arrays with presence and NULL masks (§7.2).

A table can be in several placements at once, and the runtime moves it when
a step needs it somewhere else. Reading a handle's rows is the only way data
leaves the runtime, and it is always explicit.

### 6.3 Compiled programs

A program depends on the structure of its plans and on the schemas of its
tables, not on their values. The runtime **compiles** a program once
(planning each step, choosing implementations and placements) and runs it
many times with new inputs and parameters. ddx's `Prepared` runs and the
`compile` method of its `Backend` already work this way, and so does
`TrainingSession`'s whole-step compilation. A compiled program is reused
only while the session would plan every step the same way, which ddx tests
with `session_key`.

---

## 7. Execution

### 7.1 Three ways to run a step

A step can run in three ways, and the runtime chooses per step:

1. **On the engine**, as a plain query. This is the reference (R1), and
   always available.
2. **Through a kernel**, for work einfold recognizes. einfold checks the
   plan's shape, keys, duplicates, NULLs and domains, and hands the work to a
   backend through its `DenseBackend` interface (einfold#156), which may
   decline. The runtime implements that backend: MLX first, CUDA later.
3. **On compact arrays**, where a run of steps keeps its intermediates as
   dense arrays with masks, and expands them to rows only at the program's
   boundary.

A GPU SQL engine such as Sirius is a fourth option, but it is an engine, so
it enters through way 1. §3.1 suggests a GPU engine that passes long-format
rows between operators would meet the same wall the MLX kernels did. That is
a hypothesis worth one spike (§14).

### 7.2 Compact arrays

Compact arrays gave the largest single gain (§3.1), and where they live is
xarray-sql#307's first open question. Today they are split: einfold's
`DenseProgram` lowers DataFusion's logical operators to arrays, and
xarray-sql's `DenseExecutor` runs them across stages. Neither is a public
interface.

This draft proposes that einfold keeps the recognition (which plans have an
array meaning, and the checks that make it exact) and the runtime owns the
execution (the arrays, their memory, their device and their lifetime). That
follows einfold's own rule that it never executes.

An alternative may need no second executor at all. If dense intermediates
can travel between an engine's own operators as Arrow fixed-shape tensor
columns, in GPU memory through the C Device Data Interface, then the engine
keeps SQL's semantics and the runtime only provides the operators' kernels.
torch-sql's spike S4 found that DataFusion keeps such a column through
projections, filters and sorts without copying it. Whether it survives joins
and aggregates, and whether DataFusion can run them on device buffers, is
the first spike (§14).

### 7.3 Shared subplans

Until DataFusion computes a CTE read several times only once
(xarray-sql#267), the runtime saves shared subplans as tables, which is what
both `SharedExecutor` and torch-sql's cut points do. Doing it in the runtime
means one implementation, with the budget set by the engine and the memory
available, instead of one per frontend.

---

## 8. Memory

Deciding which values to save and which to recompute is the same trade-off
whether the reason is planning (cut points on DataFusion), memory (gradient
checkpointing) or a slow link (a value in Grace memory against one in GPU
memory). The runtime makes the choice for a whole program, from:

- each engine's planning budget, as torch-sql's cut points use today;
- the memory of each placement on the hardware;
- *regions* from ddx ([ddx#251](https://github.com/xqlsystems/ddx/issues/251)):
  named parts of a program inside which nothing is saved except what a
  policy names, which is how torch-sql will carry `torch.utils.checkpoint`.

On the Beta cluster, each GB200's Grace memory is a placement a GPU can
read directly, which makes spilling there cheaper than on most machines.
Whether a GPU reads Arrow buffers from Grace memory at a useful speed is
untested.

---

## 9. Scheduling

Within a run, the runtime runs each step once the steps it reads are done,
following ddx#157: ddx exposes which earlier steps each step reads, and the
runtime implements a `Backend` whose `materialize` may run concurrently.
Planning one step while another executes overlaps the single-threaded
planning that dominates small steps (§3.3).

Across runs, the runtime overlaps a step's computation with loading the next
batch. On Samudra, the batches come from xarray-sql's data path over a Zarr
archive, so the runtime is also where the data path and the training step
meet.

---

## 10. State and checkpoints

xarray-sql#307 proposed splitting `TrainingSession` along the line between
execution and training. The runtime takes the execution side:

- **persistent state**: tables that stay in place across runs (§6.1);
- **atomic publication**: new state replaces old only after a run's checks
  pass (R5);
- **checkpoints of state**: writing persistent tables out and reading them
  back. `TrainingSession` writes NPZ files today. A format the other layers
  already read, such as Zarr through xarray-sql or Arrow IPC, would make
  checkpoints queryable, as torch-sql's design plans for its weights.

The frontend keeps the rest: the training loop, the optimizer's math, the
learning-rate schedule and evaluation. In torch-sql that is PyTorch's
`torch.optim`, lowered to SQL; in xarray-sql's demo it is plain SQL.

---

## 11. Interfaces

### 11.1 To frontends

A sketch, to make the model of §6 concrete. The names are placeholders:

```python
import xql_runtime as rt

session = rt.Session(engine="datafusion", devices=["cuda:0"])
w = session.put("w", weights, persist=True)        # an Arrow table, now owned

step = session.program(
    steps=ddx_program.steps,                       # Substrait plans
    inputs=["x"],
    state={"w": "w_next"},                         # w takes w_next's value
    parameters=["lr"],
    outputs=["loss"],
)
compiled = session.compile(step)
for batch in loader:
    out = compiled.run(inputs={"x": batch}, parameters={"lr": 6e-4})
    print(out["loss"].to_pyarrow())                # the only read
```

torch-sql would replace its engine wrapper (`torch_sql.engines`) and its
cut points (`torch_sql.program`) with calls like these, and a synced
`device="sql"` tensor would hold a table handle.

### 11.2 To ddx

The runtime implements `ddxdb.Backend`, and later `ddx_ad::Backend` in
Rust, so every ddx program runs on it unchanged. From ddx it needs:

- the step graph, for concurrent steps (ddx#157);
- regions, for memory (ddx#251);
- which forward node and role each backward step comes from
  ([ddx#259](https://github.com/xqlsystems/ddx/issues/259)), so a kernel
  can be chosen for a backward step, such as a convolution's gradient with
  respect to its weights.

### 11.3 To einfold

The runtime implements einfold's `DenseBackend` (einfold#156). einfold
recognizes work and checks that an array algorithm computes it exactly; the
runtime runs it, or declines. einfold's side-channel fact table, through
which frontends pass facts (the extents of an index, that a tensor has every
row) and hints (the PyTorch op that produced a plan node), reaches the
runtime the same way.

### 11.4 To engines

The runtime holds an engine session and runs plans on it: DataFusion first,
since xarray-sql and einfold live there, then DuckDB, then GPU engines. All
three projects need one DataFusion version for this to work, which
[xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)
tracks.

### 11.5 Transport

Tables cross into and out of the runtime as Arrow, through the PyCapsule
interface on the host and the C Device Data Interface on a GPU, with DLPack
toward PyTorch. Today nothing in the stack exports a device stream, and
PyTorch reads no Arrow capsule, so the device path needs both (torch-sql
spike S4).

---

## 12. Distribution (later)

A database spreads a query across machines by partitioning its tables and
moving rows between partitions where a join or an aggregate needs them. In
a training step, the partitioning key gives the usual kinds of parallelism:
by batch index, data parallelism; by a weight's index, weight sharding; by
grid tile, *domain decomposition*, as numerical ocean models run, where each
GPU owns a tile and exchanges only the cells along its edges with its
neighbours. torch-sql's hypothesis H1 is that a planner choosing the
partitioning per join, from the whole step's plan, distributes a model like
Samudra better than PyTorch's FSDP does.

That choice belongs to the runtime. The first versions don't distribute, but
the program model must not prevent it: steps and tables need to carry the
partitioning facts a distributed planner would use. On the Beta cluster,
one NVL72 rack holds a whole Samudra run, so the first distribution never
crosses a rack.

---

## 13. Milestones

Each milestone is measured on xarray-sql's nanoGPT demo and torch-sql's
nanoGPT, and reported whichever way the numbers go.

- **M0. A CPU runtime.** Sessions, table handles and programs on DataFusion
  and DuckDB, implementing `ddxdb.Backend`. torch-sql's engine wrapper and
  xarray-sql's demo 3 both run on it, with the same results as today.
- **M1. Resident state on the host.** Persistent tables and atomic
  publication. torch-sql's optimizer step becomes one program instead of a
  query per tensor.
- **M2. Shared subplans and memory budgets.** Cut points move from
  torch-sql into the runtime.
- **M3. Concurrent steps** (ddx#157).
- **M4. Compact arrays on the CPU**, after the spike of §14 settles where
  they live.
- **M5. A GPU.** MLX on Apple hardware, where §3.1's numbers come from, then
  CUDA on a B200, with state resident on the device.
- **M6. Distribution**, within one NVL72 rack.

---

## 14. Spikes

1. **Fixed-shape tensors through DataFusion's operators.** How far a dense
   column passes through joins and aggregates without being copied or
   expanded to rows, on the host and on device buffers. This decides §7.2.
2. **A GPU SQL engine on long-format rows.** One of xarray-sql's nanoGPT
   steps on Sirius, to test whether rows between operators cost a GPU engine
   what they cost the MLX kernels.
3. **Grace memory.** A B200 reading Arrow buffers in place from Grace memory,
   against copying them to the GPU.
4. **The runtime's language.** The cost of a Python core against a Rust one
   for M0 and M3, given that concurrent steps under Python's global
   interpreter lock would run one at a time.

---

## 15. Questions for the xarray-sql agent

These need the xarray-sql agent's knowledge of the nanoGPT work, and its
answers will revise §3.1, §7 and §10:

1. Which of `SharedExecutor`, `DenseExecutor` and `TrainingSession` would
   move into the runtime whole, which would split, and which stay in
   xarray-sql?
2. Which of `DenseProgram`'s private types would the runtime need einfold
   to make public?
3. What did compact arrays decline in practice, and how often did a step
   fall back to the engine?
4. Where was the remaining 3.76 times against PyTorch spent?
5. Does `TrainingSession`'s whole-step compilation need anything from ddx
   beyond reusing the primal (ddx#256 and #257)?

---

## 16. Open questions

1. **The name and home.** A new repository under xqlsystems is the obvious
   home; the name is open.
2. **Rust or Python first.** This draft leans toward a Rust core with Python
   bindings, as ddx and einfold have, because DataFusion, ddx's `Backend`
   trait and einfold's `DenseBackend` are Rust, and concurrent steps need to
   run outside Python's lock. A Python prototype for M0 would come sooner.
   Spike 4 decides it.
3. **Who owns compact arrays** (§7.2): einfold, the runtime, or nobody,
   if spike 1 says the engine can carry them.
4. **Substrait as the only plan format.** ddx already speaks it. Engines
   without a Substrait producer would need SQL text in their dialect, which
   ddx tracks in [ddx#159](https://github.com/xqlsystems/ddx/issues/159).
5. **Low precision.** SQL engines have no bfloat16, and GB200 hardware is
   built for it. Kernels could compute in low precision with the SQL types
   as the reference, but R1's floating-point bound would then need stating
   per kernel.
6. **Randomness.** Dropout needs random numbers that every implementation of
   a step agrees on. A hash of a seed, the step and the indices works in
   portable SQL; `TrainingSession` uses explicit seed and step keys.
7. **Which engine is the reference** when two engines disagree (§3.2), and
   whether the runtime should run both in a checking mode.

---

## Decision log

Nothing is decided yet. The proposals this draft makes, for discussion:

- **P-A.** The runtime is a separate project, and einfold stays free of
  execution (from xarray-sql#307).
- **P-B.** The unit of work is a program with explicit state, not a query
  (R3, R5).
- **P-C.** Recognition stays in einfold and execution moves to the runtime
  (§7.2), unless spike 1 shows the engine can carry dense values itself.
- **P-D.** The runtime implements ddx's and einfold's public backend
  interfaces rather than either project depending on the runtime.
