# The XQL runtime: running relational programs on devices

_author_: Alex Merose
_co-authors_: 🔥 Claude (torch-sql) and 🌀 Claude (xarray-sql), via Claude Code
_status_: Draft v1.1, for review by the einfold and ddx agents and by Alex. Nothing is built.
_last updated_: 2026-11-07
_origin_: [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)

---

## How this doc is being written

The torch-sql agent wrote draft v0 from torch-sql's prototype and from
xarray-sql#307. The xarray-sql agent revised it into this draft v1. It
answered v0's questions about xarray-sql's nanoGPT work in the text (§3.1,
§7, §10 and §13), checked the measurements against their sources, and
proposed a different answer to the central question of where dense
execution lives (§7.2).

The torch-sql agent then reviewed v1 as co-author, for v1.1. It agreed
to design B, and answered its questions in the text: on cut points (§7.5),
on tiles (§7.2) and on Rust against Python (§18.2), where it now agrees
with Python first. It corrected §3.3, and found a gap between cut points
and differentiation that needs ddx (§7.5). Next, the einfold and ddx
agents review as the gate, starting with einfold, since design B rests on
its answer. The contributor behind §3.1's work is asked
to review §3.1, §7.2 and §13. A fresh agent joins to implement the spikes
once the decision log settles. Alex reviews throughout.

The einfold agent (🪭 Claude (einfold)) then reviewed v1.1. It accepts
design B in principle, and doesn't yet approve the doc. Its full review
is in [reviews/einfold-on-v1.1.md](reviews/einfold-on-v1.1.md). It corrected
what the doc says einfold does today (§2, §7.2 and §11.3), answered §17.2
in the text, and stated four positions in §18 (§18.3, §18.7 and §18.8) that
it asks to be settled before it approves. Alex rejected StableHLO as XQL's
IR during this review (§18.3).

The ddx agent (∂ Claude (ddx)) then reviewed v1.1. It accepts the runtime
and design B, and doesn't yet approve the doc. Its full review is in
[reviews/ddx-on-v1.1.md](reviews/ddx-on-v1.1.md). It corrected what the doc
says ddx does today (§3.4, §7.5 and §11.2), answered §17.1 in the text, and
stated two positions in §18 (§18.9 and §18.10) that it asks to be settled
before it approves.
Questions for each reviewer are in §17. Where co-authors or reviewers
disagree, the disagreement stays in §18 with each position stated, until it
is settled and moves to the decision log. When the questions are settled,
§17 goes away and the doc should read as a design, not a conversation.

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
write. The runtime runs it. It owns the memory that holds the tables, keeps
them on a device such as a GPU between operators and between steps, and
chooses an implementation for each piece of work. It also schedules the
steps, and later spreads them across machines.

Today each frontend does this for itself, in different ways. torch-sql
runs its programs through a small engine wrapper of its own. xarray-sql's
nanoGPT experiments built three executors and a training session inside
xarray-sql (§3.1), which xarray-sql#307 found don't belong there. einfold,
which finds array computations in plans, runs array algorithms on the CPU
in its own reference operator, but leaves devices to whatever runs the plan:
its design calls whatever runs its plans the *host* (today, the engine),
and says einfold "never touches a device". The runtime joins the engine in
that role: it becomes the one place where devices, memory and kernels are
managed, behind interfaces every frontend can use.

### 1.1 Why now

Two lines of work found that *how data is held and moved between queries*,
more than planning or arithmetic, decides how fast XQL runs a training step
(§3):

1. In xarray-sql's nanoGPT experiments on an Apple GPU, sending the matrix
   products and attention to the GPU changed almost nothing for a
   123.6M-parameter model: 163.6 s per training step against 164.0 s on the
   CPU. What helped was holding intermediates as dense arrays instead of
   Arrow rows: 2.43 s per step on the GPU, 67 times faster. Then, on a
   smaller model (10.7M parameters, trained as nanoGPT trains its
   Shakespeare example), keeping weights and optimizer state on the GPU
   between steps brought a step from 2.02 s to 101 ms. PyTorch took 27 ms
   for the same step on the same GPU.
2. In torch-sql, a whole training step of nanoGPT now matches PyTorch to
   1e-11 on two engines. But the optimizer's update alone takes 8 to 12
   seconds at six layers, because each parameter and each piece of
   optimizer state is read and written by a query of its own.

Both are about how data is held and moved between queries, which is the
runtime's job.

---

## 2. Background

This section introduces each project and format the rest of the doc uses.

**Apache Arrow** is a columnar memory format: a table is a set of columns,
each a contiguous buffer. Every XQL engine reads and writes Arrow, so it is
how tables move between projects. Its *PyCapsule interface* hands Arrow
data between Python libraries without copying. Its *C Device Data
Interface* extends that to buffers in GPU memory. Its canonical
[fixed-shape tensor](https://arrow.apache.org/docs/format/CanonicalExtensions.html)
extension type is a column whose every value is a dense array of one shape.

**Substrait** is a portable, serialized format for relational query plans.
DataFusion and DuckDB can each produce and consume it, which lets ddx read
one engine's plan and hand back plans the same engine runs.

**DLPack** is a standard for sharing a dense array between array libraries,
such as Arrow and PyTorch, without copying.

**Zarr** is a format for chunked N-dimensional arrays in files or object
storage, which Xarray reads and writes.

A **CTE** (common table expression) is a named subquery, `WITH x AS (...)`,
that the rest of a query can read, possibly several times. A **view** is a
named query stored in a session, which later queries read like a table.

The engines:

- **[DataFusion](https://datafusion.apache.org/)** is a SQL query engine
  written in Rust and built to be extended. xarray-sql and einfold build on
  it.
- **[DuckDB](https://duckdb.org/)** is an embedded analytical database. ddx
  runs on it through DuckDB's community Substrait extension.
- **[Sirius](https://github.com/sirius-db/sirius)** and NVIDIA's **GQE** are
  SQL engines that run on GPUs.

The frontends and transforms:

- **[xarray-sql](https://github.com/xqlsystems/xarray-sql)** registers
  [Xarray](https://xarray.dev/) Datasets (labelled N-dimensional arrays) as
  tables, and turns query results back into Datasets. Its demos include
  attention, a GPT language model trained in plain SQL, and the U-Net of
  the ocean emulator Samudra (below), each checked against a reference in
  PyTorch or JAX.
- **[torch-sql](https://github.com/xqlsystems/torch-sql)** runs
  [PyTorch](https://pytorch.org/) programs as SQL. A tensor on its
  `device="sql"` is a table, and each operation on it becomes part of a
  query. It trains [nanoGPT](https://github.com/karpathy/nanoGPT), a small
  GPT-2 implementation, today.
- **[ddx](https://github.com/xqlsystems/ddx)** differentiates SQL. Given a
  query that computes a loss, it writes a *program*: a sequence of *steps*,
  each a plan that saves its result as a table, which together compute the
  loss and its gradients. A program also carries *checks*, plans that must
  return no rows for the result to be valid. `ddxdb` is its Python package.
- **[einfold](https://github.com/xqlsystems/einfold)** is a query optimizer
  for DataFusion. It finds array computations in a plan, such as a matrix
  product written as a join and a `GROUP BY`, and replaces them with array
  algorithms. It runs those algorithms on the CPU in its own operator,
  `EinFoldExec`. Its *dense backend* interface
  ([einfold#156](https://github.com/xqlsystems/einfold/pull/156), a draft
  from a fork, which einfold's maintainer hasn't reviewed) would let a host
  take that work instead, or decline it.

The hardware:

- **MLX** is Apple's array library for its GPUs. The nanoGPT performance
  work ran on an Apple M5 Ultra through it. PyTorch runs on the same GPUs
  through its *MPS* backend.
- **The Beta cluster** of
  [Empire AI](https://empireai.freshdesk.com/support/solutions/articles/157000363466-alpha-beta-hardware-2026-)
  is where XQL's first production model, the ocean emulator
  [Samudra](https://github.com/m2lines/Samudra), trains. It has 288 NVIDIA
  B200 GPUs in GB200 NVL72 racks. In a GB200 system, each Grace CPU (an ARM
  CPU) is joined to two B200 GPUs by NVLink-C2C, a cache-coherent link that
  lets a GPU read the CPU's memory directly. An NVL72 rack joins 72 GPUs
  with NVLink, NVIDIA's GPU-to-GPU interconnect.

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

The work has continued since, on unreviewed branches. On Modded-NanoGPT, a
124M-parameter GPT trained on FineWeb text at context 1,024 with the Muon
optimizer, the latest SQL step takes 239 ms on the M5 Ultra, against
PyTorch MPS's 182 ms: about 1.3 times slower. The gains came from more
proofs. New modules in xarray-sql:
- prove that ddx's derivatives of attention match a kernel
  (`_dense_adjoint.py`);
- prove when a row exists, from coordinates alone (`_dense_proofs.py`);
- prove the shape of attention from einfold's reduction graph
  (`_dense_attention.py`).

That is design B of §7.2 emerging on its own, one proof at a time, but in a
frontend instead of in einfold. Each step forward that lands in the wrong
layer makes the right one harder to reach, which is why this design is
urgent.

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
   meets an operator it can't run, it restarts the *whole call* on the SQL
   executor. Benchmarks therefore ran in a strict mode where any decline is
   an error. The reported runs had no fallbacks, but only because strict
   mode ruled them out. Nobody counted how often each kind of operator
   declined. The one decline documented is a grouped aggregate with several
   dependent or nullable keys, in AdamW on compact arrays.
4. **Compact execution became a second implementation of SQL.**
   `DenseProgram` interprets DataFusion's logical plans operator by
   operator: joins, grouped aggregates, `UNION` with bag semantics, and NULL
   handling, in 1,564 lines of Python beside einfold's Rust. That is where
   most of the risk sits (§7.2).
5. **The pieces coupled through private names.** xarray-sql's executors
   import 18 private names from `einfold.dense_program`, einfold's
   private `einfold._native`, and ddx's private `ddxdb._ddxdb.to_sql`. The
   runtime has to be built on interfaces instead (R2).

### 3.2 From xarray-sql's Samudra demo

xarray-sql's demo 4
([xarray-sql#300](https://github.com/xqlsystems/xarray-sql/pull/300)) writes
Samudra's U-Net as one SQL view per layer, and checks it against Samudra's
own PyTorch. Samudra's U-Net is built from ConvNeXt blocks: two dilated
convolutions, each with a batch norm and a GELU. The demo runs the network
two ways:
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
the long-format representation, not planning. (The machine was under load
from a compile job, so these are rough.)

One more finding: the demo keeps the views as its main path because
**ddx differentiates through views, not through stored tables**. A table
saved by the frontend is a constant to ddx, so the gradient would stop at
it. Where to save is an execution decision, and it has to be invisible to
differentiation (R7).

### 3.3 From torch-sql

torch-sql runs a training step as a few large programs. Each finding below
was measured there or in its spikes:

- **DataFusion plans a CTE again at every read**
  ([xarray-sql#267](https://github.com/xqlsystems/xarray-sql/issues/267)).
  For a stack of ConvNeXt-like blocks as one query, the plan grows from 441
  operators for one block to 136,107 for three. Planning and running grow
  from 0.2 s to 212 s. DuckDB grows by 75 operators per block.
  torch-sql therefore chooses *cut points*, values saved as tables between
  queries. A value is a cut point when the query that computes it, with
  every value it reads written in, passes a per-engine budget (400 `SELECT`
  blocks on DataFusion, none on DuckDB), and the input of every norm is one
  whatever its size, because a norm reads it three times. That brought one nanoGPT layer's backward
  pass on DataFusion from 9 s to 0.7 s. Two layers, which didn't fit in 8 GB
  before, now run in 1.0 s.
- **Differentiating across cut points is a chain of ddx programs.** Each
  segment between cut points is one program. Each cut point passes a
  *cotangent* (the gradient of the loss with respect to that table) to the
  segment before it. That is a schedule of programs with dependencies
  between them, which a runtime could plan as one.
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
  are unique, and its values are never NULL. So torch-sql can state, as
  exact facts, what einfold's proofs need for every table it hands over
  (§7.2). Partial tensors, for a masked Samudra, will be the exception, and
  will carry their domain as a table of their own.
- **Lifetimes are hard to get right in the frontend.** torch-sql drops a
  table when the last Python object referring to it is collected, and a bug
  there dropped tables still in use. A table's owner and lifetime belong
  with whatever owns the memory.
- **Transport** (torch-sql spike S4): an Arrow array with no nulls reaches
  PyTorch through DLPack without a copy. A fixed-shape tensor column keeps
  its type through DataFusion's projections, filters and sorts, and a
  projection keeps the same buffer. DuckDB turns it into a plain list of
  fixed size, which keeps the values but loses the shape.

### 3.4 From ddx

- **Planning dominated small steps, until steps were compiled.** In the
  program for one SGD step of xarray-sql's GPT demo, DataFusion's planning
  took 208 ms of 422 ms, and execution 177 ms, on ddx 0.2.1
  ([ddx#157](https://github.com/xqlsystems/ddx/issues/157)). Since then,
  `prepare` builds a program once
  ([ddx#203](https://github.com/xqlsystems/ddx/pull/203)), and a backend
  can compile each step's physical plan to run again
  ([ddx#214](https://github.com/xqlsystems/ddx/pull/214)).
  - The same step now takes 72 ms per warm run, in ddx's `prepared_step`
    benchmark (Rust, DataFusion 55, Intel i7-8700, ddx-ad 0.3.3).
  - From Python, ddxdb's `in_place` mode takes it from about 300 ms to
    about 100 ms (ddx#214). That mode needs an xqlsystems fork of
    datafusion-python, so it is experimental.
- **Steps could run concurrently.** Many of a program's steps read none of
  the others. Running each as soon as its inputs are ready would shorten
  xarray-sql's demos by 16 to 43 percent. That is an upper bound, from each
  program's critical path (ddx#157).
- **ddx's `Backend` is public.** `ddxdb.Backend` (ddxdb 0.4.7) is a Python
  protocol with the primitives ddx needs: plan a query, give a table's
  schema, run a plan and save its rows as a table, say whether a plan
  returns rows, and drop a table. It can optionally compile a plan to run
  again, give a view's SQL, and report a *session key* that changes
  whenever the session would plan differently. The Rust trait
  `ddx_ad::Backend` has the same shape. Any object that implements it can
  run ddx's programs.

---

## 4. Goals and non-goals

### 4.1 Goals

1. **Run programs from any XQL frontend or transform**, through one
   interface: ddx's gradient programs, torch-sql's segment chains, and
   plain queries.
2. **Own tables and their memory.** Every table the runtime holds has an
   owner, a lifetime and a placement (host memory, GPU memory, or inside an
   engine).
3. **Keep data where the computation is**, between operators and between
   steps, so a training loop moves only its batches and the values it
   reads.
4. **Choose an implementation for each piece of work.** The engine's own
   operators are the default. Work that einfold has proved is an array
   computation can run as an array program, which may decline it.
5. **Budget memory**: choose what to save and what to recompute against the
   memory of the hardware it runs on. That subsumes torch-sql's cut points.
6. **Schedule**: run independent steps concurrently, overlap planning with
   execution, and prepare a step's next batch while the current one runs.
7. **Distribute**, later: run one program across GPUs and machines (§12).

### 4.2 Non-goals

- **Its own SQL engine, in any form.** Engines give SQL its meaning, and
  the runtime never interprets relational operators itself: no joins, no
  `GROUP BY`, no NULL rules. It runs engine plans and array programs. The
  translation between the two is einfold's, which proves it exact (§7.2).
  That rules out carrying `DenseProgram` into the runtime as it is.
- **Differentiation or plan rewriting.** Those are ddx's and einfold's.
- **A training framework.** Training loops and optimizers belong to
  frontends: PyTorch through torch-sql, or plain SQL in xarray-sql's demos.
  The runtime provides what a training loop needs from execution: state
  that persists, atomic updates, and checkpoints of that state. It knows
  nothing about learning rates or schedules.
- **Bitwise agreement across implementations.** Results agree with the
  engine's SQL result to a stated floating-point bound.

---

## 5. Principles

**R1. The SQL is the meaning.** A program's result is what its SQL
computes on the reference engine. A kernel, an array program or a GPU
engine is an implementation of that meaning, checked against it. It
declines what it can't implement exactly. A substitution never changes a
result beyond the stated floating-point bound, and never changes which rows
exist.

**R2. Public interfaces only.** The runtime depends on its neighbours'
released, public APIs, and its neighbours depend on its public API. No
private imports across repositories, and no forks.

**R3. Programs, not queries.** The unit of work is a program, usually a
whole training step. Planning per query costs more than computing (§3.4),
and a runtime that sees the whole step can keep its state in place.

**R4. Every table has an owner.** The runtime knows who holds each table
and when it may be dropped. Frontends hold handles, not bare names.

**R5. State changes atomically.** Tables that persist across runs, such as
weights and optimizer state, take their new values only when a run
finishes and its checks pass. A failed run leaves them as they were.

**R6. Each layer stays optional.** A frontend can run without the runtime,
on a plain engine, as torch-sql does today. The runtime runs on a CPU-only
machine with no kernels at all. Each capability (array programs, a GPU,
scheduling) is opt-in and measured on its own.

**R7. Where a value is saved is never part of the program's meaning.**
Frontends and ddx see the program as written, views and CTEs included.
Saving a value as a table, recomputing it, or keeping it on a device is the
runtime's choice. It changes neither results nor gradients.

**R8. Declines are visible.** Every time an implementation declines a step,
the runtime counts it, with the reason. A *strict* mode turns declines into
errors, for benchmarks. A *checking* mode runs the reference beside the
substitute and compares them (§7.3).

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
- its **checks**: plans that must return no rows for a run to succeed, as
  ddx's programs already carry. A check is *structural* if it depends only
  on schemas and on the domains of the inputs' coordinates, or a *data
  check* if it reads values. A compiled program evaluates its structural
  checks once, when it compiles, and its data checks on every run (§11.2).

ddx's programs fit this already: their steps, value and gradients are
steps and outputs. torch-sql's segment chain (§3.3) is several ddx programs
whose cotangents connect them, which the runtime would join into one graph.
A plain query is a program of one step.

Programs compose. A training step is ddx's program followed by the
optimizer's update steps, which read its gradient tables and write the next
state. In torch-sql these are separate today: ddx's program runs at
`backward()`, and the update runs later, a query per tensor (§3.3). The
runtime should take both as one program, so that the step is compiled
once and the gradients never leave it.

Parameters need a portable form. Substrait has a dynamic parameter
expression, but whether DataFusion's and DuckDB's consumers accept it is
unverified. A one-row table of parameters, which the plans cross join and
the runtime rewrites before each run, works on every engine. torch-sql
already reads its non-finite constants from such a table. Differentiating
through one is where ddx#264 and #265 arose on DuckDB, though, so a
parameter ddx differentiates through needs those fixed first.

Steps travel as Substrait plans, which ddx already writes. Inside a
compiled program, each step may be held in its engine's own plan format.

### 6.2 Tables and placements

A **table handle** refers to a table the runtime owns. It carries the
table's schema, its lifetime (one run, or until the caller releases it),
and its *placement*:

- **in an engine**, as the engine's own table;
- **as Arrow**, in host memory or, with the C Device Data Interface, in GPU
  memory;
- **as a compact array**, on the host or a device (§7.2).

A table can be in several placements at once, and the runtime moves it
when a step needs it somewhere else. SQL refers to tables by name, so each
handle also has a name in the session's namespace. The name is how plans
read the table, and the handle is what keeps it alive. Reading a handle's
rows is the only way data leaves the runtime, and it is always explicit.

### 6.3 Compiled programs

A program depends on the structure of its plans and on the schemas of its
tables, not on their values. The runtime **compiles** a program once and
runs it many times with new inputs and parameters. Compiling plans each
step, chooses implementations and placements, and evaluates the structural
checks. ddx's `Prepared` runs, and the `compile` method of its `Backend`,
already work this way, and so did `TrainingSession`'s whole-step
compilation. A compiled program is reused only while the session would plan
every step the same way, which ddx tests with the backend's session key.

---

## 7. Execution

### 7.1 Three ways to run a step

A step can run in three ways, and the runtime chooses per step:

1. **On the engine**, as a plain query. This is the reference (R1), and
   always available.
2. **Through a kernel**, for one computation einfold recognizes, such as a
   matrix product or attention. einfold checks the plan's shape, keys,
   duplicates, NULLs and domains, and hands the work to a backend through
   its dense backend interface (einfold#156). The backend may decline. The
   runtime implements that backend: MLX first, CUDA later.
3. **As part of an array region**: a run of consecutive steps that einfold
   has translated into one array program. Its intermediates stay as dense
   arrays, and become rows only where the region ends (§7.2).

A GPU SQL engine such as Sirius would be a fourth option. But it is an
engine, so it enters through way 1. §3.1 suggests that a GPU engine passing
long-format rows between its operators would meet the same wall the MLX
kernels did. That is a hypothesis worth a spike (§16).

### 7.2 Dense execution: where SQL becomes arrays

Compact arrays gave the largest single gain (§3.1). The question is which
layer turns a relational plan into array operations, and which runs them.
There are three designs:

**A. The runtime interprets relational plans on compact arrays.** This is
`DenseProgram` as it is today, moved into the runtime. The runtime would
walk DataFusion's logical operators and implement each on arrays: joins as
gathers, `GROUP BY` as reductions, `UNION` with bag semantics, NULLs as
masks. It works, and it is measured. But it is a second implementation of
SQL's semantics, beside the engines'. Every operator, NULL rule and edge
case would have to agree with every engine, forever. §3.3 shows how hard
that is even between two mature engines. It also violates the non-goal of
§4.2.

**B. einfold translates, the runtime runs array programs.** einfold
already shows that a plan fragment computes the same thing as an array
algorithm, in two parts:
- **the plan's structure, before it runs.** For example, that an aggregate
  is a `SUM` of products over an inner equi-join, with each factor reading
  one side.
- **the data, at run time.** `EinFoldExec` checks dense coordinate ranges,
  unique keys, NULLs and finiteness on the rows as they arrive. Where a
  check fails, it runs the engine's own plan, which every node einfold
  detects keeps.

einfold reads no facts from frontends yet. Its fact table is designed, in
einfold's design §8.1, but not built (einfold's M5). Design B extends that
proof from one kernel (a matrix
product) to a *region*: a run of steps that einfold translates, as a whole,
into an **array program**. An array program is written in a small,
versioned set of array operations, and has no relational operators left in
it:
- elementwise expressions;
- contractions (`einsum`);
- reductions over named axes;
- gathers and scatters along an axis;
- `where` with a mask.

The runtime runs array programs. It compiles each one for its backend
(NumPy on the CPU, MLX, later CUDA through PyTorch or another array
library), and keeps their values resident. Whatever einfold can't translate
stays relational, and runs on the engine.

Three things B must handle, from torch-sql's lowerings of nanoGPT:
- **Gathers whose index is a value.** An embedding is a join on a token
  id. If an id falls outside the table, SQL's inner join drops the row,
  where a gather would fail or read the wrong memory. A gather is exact only
  given a fact that every id lies in range, and when no frontend can state
  that fact, a data check (§6.1) on each run.
- **Floating-point edge cases.** Each array operation needs its NaN,
  infinity and signed-zero behaviour stated, and einfold's proof must match
  it to the SQL's. torch-sql writes PyTorch's behaviour out in SQL (NaN
  checks in comparisons and in `MAX`, guards around `ln` and `sqrt`),
  because the engines differ (§3.3). Those guards become `CASE` expressions
  that einfold has to translate, or prove unnecessary.
- **Shape-changing steps.** torch-sql writes reshapes, slices and
  convolutions as joins with *index tables*, small tables that map each
  output position to an input position. Under B each becomes a gather, or a
  reshape when einfold can prove the map is one.

Under B:
- SQL's semantics live only in the engines, and in einfold's proofs, which
  einfold already tests against the engine.
- The runtime needs none of `DenseProgram`'s private types, only the array
  program format and einfold's decline reasons.
- einfold#156's dense backend is the special case of a region holding one
  kernel.

**C. The engine carries dense values itself.** If intermediates could
travel between DataFusion's own operators as fixed-shape tensor columns, in
GPU memory through the C Device Data Interface, the engine would keep SQL's
semantics, and the runtime would provide only kernels. Spike S4 found that
DataFusion keeps such a column through projections, filters and sorts
without copying it.

But a tensor-valued column changes what a table is: a row holds a tile of
an array rather than one element. A matrix product is then a function on
tiles, not a join and a `GROUP BY`. That moves XQL from long format to
tiled tables, which changes every frontend's schema, and ddx's rules too.
It may be the right long-term representation, and it would make
distribution by tiles (§12) natural. But it is a change to the whole stack,
not to the runtime.

For torch-sql, tiles would be a redesign of its lowerings, though not of
its interfaces. The device, the op table and layout describe a tensor by
its shape and dtype, so a tile layout could sit behind them. But every
lowering writes arithmetic on one value
per row, and would have to become functions on arrays, each of which ddx
would then need a derivative rule for. The loss is absence. A tile is
present or absent as a whole, so a coastline cuts tiles in half, and land
cells come back as values in a mask. That is the dense box that torch-sql's
design rejected StableHLO for. Tiles along one trailing dimension only,
such as an embedding's 384 features per token, would keep cell-level
absence for the other dimensions, and may be where C starts.

**This draft proposes B**, with C as a spike that could change the
long-term answer:
- B keeps the measured win of A, since it holds the same compact arrays at
  run time.
- B keeps one implementation of SQL's semantics, and follows einfold's own
  split: einfold rewrites, hosts execute.

This revises v0's proposal (P-C). v0 had einfold keep "recognition" and
the runtime own "execution" of compact arrays. In `DenseProgram`, though,
recognition and execution are the same code, an interpreter of relational
operators. B moves the line to a format both sides can test: einfold emits
array programs, and the runtime runs them.

Whether B is feasible is einfold's call (§17.2). The questions are:
- whether einfold can translate whole regions, not just single kernels;
- how large the array operation set must be for nanoGPT and Samudra;
- whether `DenseProgram` is the place to start.

### 7.3 Declines, fallbacks and checking

`DenseExecutor` fell back by restarting the whole call on SQL (§3.1). The
runtime falls back per step instead. A region that declines at compile time
is split, or runs on the engine, and the rest of the program keeps its
implementations. A decline at run time, from an exceptional value say, runs
that step on the engine and leaves the state untouched (R5). For a region,
that means running the whole region on the engine, from its inputs. So a
region's inputs must stay readable by the engine until the region finishes,
which costs memory (§8), or the region's data checks must run before any of
its work.

Following R8, the runtime keeps counters of each implementation's runs and
declines, by reason. Benchmarks run strict, so that a decline fails loudly
rather than slowing a result down silently. A checking mode runs the engine
beside a substitute on sampled steps and compares them to the stated bound.
That is how xarray-sql's demos and einfold's tests already check every
substitution, and §3.3 shows why it must stay possible in production.

### 7.4 Shared subplans

Until DataFusion computes a CTE read several times only once
(xarray-sql#267, on DataFusion's roadmap for late 2026), the runtime saves
shared subplans as tables. That is what both `SharedExecutor` and
torch-sql's cut points do. Doing it in the runtime means one
implementation, with the budget set by the engine and the memory available,
instead of one per frontend. By R7 it is invisible to differentiation: ddx
differentiates the program as written, and the runtime chooses what to save
while running it. §7.5 says what that needs from ddx.

The runtime finds shared subplans in the program's plans, not in SQL text.
`SharedExecutor` parsed SQL to find CTEs. To do that it had to turn ddx's
plans back into SQL through a private function. Working on the step graph
(§11.2) avoids both. This work should shrink when DataFusion's fix lands,
so it should stay small.

### 7.5 Cut points and differentiation

§7.4 says the runtime chooses what to save while running a program, and
ddx differentiates the program as written. For a loss of any depth on
DataFusion, that order doesn't work yet. ddx differentiates the plan of the
whole loss, which the engine produces first, and DataFusion's Substrait plan
already has each CTE written in at each read (neither engine's producer
keeps CTEs, torch-sql's spike S3 found). Without cut points, two nanoGPT
layers ran out of 8 GB (§3.3). By the time the runtime sees ddx's steps,
the growth has happened. And a value the frontend saves first is a
constant to ddx (§3.2).

That holds for a loss handed to ddx as one Substrait plan. It doesn't hold
for ddx's two other inputs (∂, corrected by ddx's review):
- **SQL.** `grad(...)` written in SQL plans each CTE the objective reads
  more than once on its own, before the engine's producer runs, and joins
  the pieces through `ReferenceRel`s, which ddx lowers once
  ([ddx#168](https://github.com/xqlsystems/ddx/pull/168)). A view is written
  in as CTEs first ([ddx#169](https://github.com/xqlsystems/ddx/pull/169)).
  A thirty-layer chain, 2^30 copies when written out, is differentiated
  from SQL in about a second on an i7-8700 (ddx#168).
- **Programs.** `grad` and `vjp` take a program, and write each of its
  steps once, so a frontend can hand ddx its segments as one program.

The gradient goes through these pieces; it doesn't stop at them. §18.10
has what remains.

torch-sql gets around this by cutting before differentiating
(`torch_sql.program`). It saves the cut points, then asks ddx for each
segment's gradient separately, and runs the chain rule across segments
itself, passing each cut point's cotangent back. That works, and it matches
PyTorch to 1e-11, but it is a second tape around ddx's, in a frontend.

The fix belongs in ddx, with the runtime choosing. ddx would take a set of
*save points* with the loss: subplans to compute as steps of their own and
differentiate through, rather than treat as constants. The runtime, which
knows the engine's budget and the memory, chooses them. Two ways to connect
them keep the dependencies pointing toward the runtime (P-D):
- the frontend asks the runtime for save points and passes them to ddx;
- ddx asks its `Backend` for them, through an optional method that the
  runtime implements and a plain engine leaves out.

The second keeps frontends out of it. Either way, save points are not
regions (ddx#251). A region says what *not* to save, for memory; a save
point says what to save, so that planning stays small. ddx would need both.

Cut points then move out of torch-sql whole: the runtime chooses them, ddx
differentiates across them, and torch-sql loses only code. It keeps two
things the runtime can't infer:
- **Where a norm reads its input three times.** The runtime's budget, which
  counts reads, finds most of these from the plan alone. torch-sql can pass
  the rest as hints.
- **Stop-gradients.** torch-sql saves a detached value as a table today,
  *so that* ddx treats it as a constant. Under R7, saving can't change a
  gradient, so torch-sql must use ddx's `ddx_stop_gradient` instead, which
  works only on DataFusion and only for `DOUBLE`
  ([ddx#260](https://github.com/xqlsystems/ddx/issues/260)). ddx#260 is a
  prerequisite for torch-sql on the runtime.

`torch.utils.checkpoint` loses nothing either. It has no numerical meaning:
it trades memory for recomputation, and keeps the random state so a
recomputed dropout draws the same mask. A region expresses the first, and
a counter-based random function (§10), or a mask uploaded as an input,
gives the second for free. The runtime may save more than a region asks
when memory allows, as PyTorch's selective activation checkpointing does.

---

## 8. Memory

Deciding which values to save and which to recompute is the same trade-off
whatever the reason:
- planning (cut points on DataFusion);
- memory (gradient checkpointing);
- a slow link (a value in Grace memory against one in GPU memory).

The runtime makes the choice for a whole program, from:

- each engine's planning budget, as torch-sql's cut points use today;
- the memory of each placement on the hardware;
- *regions* from ddx
  ([ddx#251](https://github.com/xqlsystems/ddx/issues/251)): named parts of
  a program inside which nothing is saved except what a policy names. That
  is how torch-sql will carry `torch.utils.checkpoint`, and how ddx will
  let einfold fuse attention's backward pass as FlashAttention does.

Atomic state (R5) has a memory cost. The old state must survive until the
new state's checks pass, so state needs space for two copies while a step
runs: weights, optimizer moments and their replacements. For Samudra 2 at
1/4°, on one B200, that is worth budgeting.

The second copy can be avoided table by table. Once every step that reads a
state table's old value has run, the new value can take over the old
buffer, as JAX's *buffer donation* lets a compiled function do. That keeps
R5 only if nothing can fail after the first table is overwritten. So a
compiled program runs every data check (§6.1) before it overwrites any
state, and a program whose checks are all structural can donate from the
start. Failures that aren't checks, such as an engine error or running out
of memory, still need the old state. A program may therefore choose between
two copies and a checkpoint to restart from (§10).

On the Beta cluster, each GB200's Grace memory is a placement a GPU can
read directly, which makes spilling there cheaper than on most machines.
Whether a GPU reads Arrow buffers from Grace memory at a useful speed is
untested (spike 3).

---

## 9. Scheduling

Within a run, the runtime runs each step once the steps it reads are done,
following ddx#157. ddx exposes which earlier steps each step reads. The
runtime implements a `Backend` whose `materialize` may run concurrently.
Planning one step while another executes overlaps the single-threaded
planning that dominates small steps (§3.4).

Across runs, the runtime overlaps a step's computation with loading the
next batch. On Samudra, the batches come from xarray-sql's data path over a
Zarr archive, so the runtime is also where the data path and the training
step meet.

---

## 10. State, checkpoints and randomness

xarray-sql#307 proposed splitting `TrainingSession` along the line between
execution and training. This is how its parts divide:

| Part of `TrainingSession` | Goes to |
|---|---|
| State tables kept as arrays between runs, with their device buffers | runtime (§6.2) |
| Publishing new state only after validation, keeping the old state on failure | runtime (R5) |
| Compiling the whole step once, and guarding reuse with the session key | runtime (§6.3) |
| Reusing the loss ddx already computed, instead of a separate forward query (ddx#257) | runtime, through ddx's public API |
| Checkpoints of state, with the step counter and random state | runtime (below) |
| Dropout with explicit seed and step keys | runtime provides the random function; the frontend decides where dropout goes (below) |
| AdamW, gradient clipping, the learning-rate schedule and evaluation | frontend: xarray-sql's demo, in SQL; torch-sql, from `torch.optim` |

**Checkpoints.** `TrainingSession` writes NPZ files (NumPy's archive of
arrays). The runtime should write state in a format the other layers
already read. Zarr, through xarray-sql, makes a checkpoint a Dataset that
SQL can query, which torch-sql's design plans for its weights. Arrow IPC is
the alternative for state that isn't array-shaped. A checkpoint is written
from published state only, so it is always consistent.

**Randomness.** Dropout needs random numbers that every implementation of
a step agrees on, the engine included, or checking mode (§7.3) can't
compare them. The runtime should provide a *counter-based* random function:
a hash of a seed, the step number and the element's coordinates. Each
backend implements it identically, and the engine implements it in portable
SQL. That last part is the constraint. Engines treat integer overflow
differently: DuckDB raises an error, and DataFusion 55 wraps around
silently. So the hash must be written in integer arithmetic that never
overflows. `TrainingSession`'s GPU dropout
used MLX's own random keys, so its masks differed from the CPU path's.
Checking it against PyTorch meant exporting the masks.

A frontend that must reproduce another framework's random numbers can't use
that function. torch-sql checks nanoGPT against PyTorch step by step, so it
draws each dropout mask with PyTorch's own generator on the CPU and uploads
it as an input table. Its masks then match PyTorch's bit for bit, at the
cost of one upload per mask. The runtime should support both: its own
random function for programs that own their randomness, and masks as
inputs for frontends that are checked against another framework.

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
print(compiled.statistics())                       # runs and declines, by reason
```

torch-sql would move in two stages. At M0 it needs no new API: the runtime
implements `ddxdb.Backend`, so torch-sql's engine wrapper
(`torch_sql.engines`) hands ddx the runtime instead of an engine, and
everything else stays. From M1, a synced `device="sql"` tensor holds a
table handle instead of a table name and a Python finalizer, and a whole
training step, from the loss to the optimizer's update, becomes one program
(§6.1). Its cut points (`torch_sql.program`) move out once ddx takes save
points (§7.5). xarray-sql would keep
registering Datasets and turning results back into Datasets, and would
hand the runtime a program when a user asks for one, for instance through
`Context.prepare`.

### 11.2 To ddx

The runtime implements `ddxdb.Backend`, and later `ddx_ad::Backend` in
Rust, so every ddx program runs on it unchanged. From ddx it needs:

- **the step graph**, for concurrent steps and for finding shared subplans
  (ddx#157);
- **regions**, for memory and fusion (ddx#251);
- **provenance**: which forward node and role each backward step comes from
  ([ddx#259](https://github.com/xqlsystems/ddx/issues/259)), so a kernel can
  be chosen for a backward step, such as a convolution's gradient with
  respect to its weights;
- **checks marked structural or data** (§6.1). `TrainingSession` declined to
  compile a step when a check read device data, because a compiled step
  can't branch on device values while it is being traced;
- **the primal outputs of a prepared run**
  ([ddx#257](https://github.com/xqlsystems/ddx/pull/257), in review), so the
  loss comes from the backward program instead of a second forward query;
- **the derivative of `erf`**
  ([ddx#256](https://github.com/xqlsystems/ddx/pull/256), in review), for
  exact GELU in nanoGPT and Samudra;
- **save points** (§7.5): subplans ddx computes as steps of their own and
  differentiates through, so that the runtime can bound the size of each
  plan without changing a gradient;
- **`ddx_stop_gradient` on every engine and type**
  ([ddx#260](https://github.com/xqlsystems/ddx/issues/260)), so that no
  frontend has to stop a gradient by saving a table (§7.5).

With the step graph, the runtime needs no way to turn ddx's plans back into
SQL text on DataFusion. `SharedExecutor` used ddx's private `to_sql` only
for that. On DuckDB it does: ddxdb runs each DuckDB step as the SQL `to_sql`
writes. `to_sql` is public in Rust and private in ddxdb, so ddxdb should
export it (∂).

### 11.3 To einfold

The runtime is einfold's host:

- it implements einfold's dense backend (einfold#156) for single kernels;
- under design B (§7.2), it runs the array programs einfold emits for
  regions.

einfold recognizes the work and proves that an array algorithm computes it
exactly. The runtime runs it, or declines. einfold's fact table would
carry facts from frontends, such as the extents of an index, or that a
tensor has every row. It is designed (einfold's design §8.1) but not
built. There is no channel for hints, such as the PyTorch op that produced
a plan node. Until the fact table exists, einfold checks each of these on
the data at run time (§7.2). Under design B, those checks become the
region's data checks (§18.3).

### 11.4 To engines

The runtime holds an engine session and runs plans on it: DataFusion
first, since xarray-sql and einfold live there, then DuckDB, then GPU
engines. All three projects need one DataFusion version for this to work,
which [xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)
tracks.

### 11.5 Transport

Tables cross into and out of the runtime as Arrow: through the PyCapsule
interface on the host, through the C Device Data Interface on a GPU, and
through DLPack toward PyTorch. Today nothing in the stack exports a device
stream, and PyTorch reads no Arrow capsule, so the device path needs both
(torch-sql spike S4).

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
than PyTorch's FSDP (its fully sharded data parallelism) does.

That choice belongs to the runtime. The first versions don't distribute,
but the program model must not prevent it: steps and tables need to carry
the partitioning facts a distributed planner would use. Array programs
(§7.2) need named axes for the same reason, so a region can be split along
one. On the Beta cluster, one NVL72 rack holds a whole Samudra run, so the
first distribution never crosses a rack.

---

## 13. Where today's code goes

The executors of §3.1 exist as draft PRs. This is where each part would
land, so that none of it is lost and none of it stays in the wrong layer:

| Code | Today | Goes to |
|---|---|---|
| `prepare()` for gradients, `Context(threads=)`, faster registration (#301) | xarray-sql | stays in xarray-sql |
| `SharedExecutor`'s stage caching | xarray-sql (#302) | runtime, as shared subplans (§7.4), working on plans rather than SQL text |
| `SharedExecutor`'s SQL parsing of CTEs (`src/shared.rs`) | xarray-sql (#302) | nowhere: the step graph replaces it |
| Arrow tables that declare their sort order (`_SortedArrowTable`) | xarray-sql (#302) | xarray-sql's registration, which knows a Dataset's dimension order |
| MLX kernels (`xarray_sql/mlx.py`) | xarray-sql (#302) | runtime, as a dense backend for einfold |
| `DenseProgram` | einfold (#157) | under design B, the start of einfold's translation to array programs |
| `DenseExecutor` and its MLX operators | xarray-sql (#303) | under design B, the runtime's array-program backends |
| `TrainingSession` | xarray-sql (#305) | split as in §10 |
| `erf` as a DataFusion function (`xarray_sql/math.py`) | xarray-sql (#305) | xarray-sql for now; DataFusion upstream eventually, since it has no `erf` |
| The Shakespeare recipe, checked against PyTorch step by step (`03_shakespeare.py`) | xarray-sql (#305) | xarray-sql, as a demo on public APIs |

---

## 14. Name and distribution

### 14.1 The name

This draft proposes **`xql-runtime`**: the package and repository
`xql-runtime`, imported as `xql_runtime`. It is free on both PyPI and
crates.io. It says what the project is in the vocabulary the doc already
uses: the runtime of XQL. It follows the plain naming of xarray-sql and
torch-sql, which say what they connect.

The alternatives considered:
- **`sql-host`.** einfold's design already uses *host* for whatever runs a
  plan, usually the engine, so the name would blur the runtime with the
  engines.
- **`array-runtime`.** Generic: it could be any array library's runtime,
  and nothing ties it to XQL.
- **A coined name, in the spirit of ddx and einfold**, such as `einrun` or
  `foldhost`. These read as einfold's own executor, which design B
  explicitly rejects.

The project avoids the word *tensor* for N-dimensional arrays, as the rest
of XQL does. `xql-runtime` honours that.

### 14.2 Packages

The runtime follows the conventions of ddx and einfold. One Python
package, built with maturin, carries a Rust core. Engines come as extras,
each pinned exactly. A Cargo workspace splits the Rust side by
dependency:

| Channel | Package | Contents |
|---|---|---|
| crates.io | `xql-runtime` | the program model, table handles, placements, the scheduler and the counters. No engine or array library: an `Engine` trait and an `ArrayBackend` trait. |
| crates.io | `xql-runtime-datafusion` | the DataFusion engine. It implements `ddx_ad::Backend`, and registers the runtime with einfold. It pins the one DataFusion version that xarray-sql, ddx and einfold share (xarray-sql#310). |
| crates.io | `xql-runtime-duckdb` | later |
| PyPI | `xql-runtime` | the bindings, with extras: `[datafusion]` and `[duckdb]` (each engine pinned exactly, as ddxdb pins them), `[mlx]` (Apple silicon only) and `[cuda]` |

Some rules:
- **The base install has no engine and no GPU library** (R6).
- **Array backends that are Python libraries** (NumPy, MLX, and the CUDA
  backend's library, §18) **start as Python code behind a Python protocol**,
  so that adding one never needs a Rust build. A backend moves into Rust
  only when spike 4 shows it must.
- **Dependencies point toward the runtime.** On the Python side, the
  runtime implements `ddxdb.Backend` structurally, since it is a protocol,
  and so needs no dependency on ddx's package. Frontends depend on the
  runtime only through an extra, such as `xarray-sql[runtime]`.
- **Wheels for Linux on ARM from the first release.** The Beta cluster's
  Grace CPUs are ARM, and torch-sql's spike S7 found ARM packaging is where
  installs break.
- **Pure Python until the first Rust code.** With Python first (§18.2), the
  early releases are this one package, built without maturin, and the
  crates are reserved names. Moving the build to maturin when Rust arrives
  changes nothing for users.


---

## 15. Milestones

Each milestone is measured on these programs, and reported whichever way
the numbers go, on [xarray-sql#285](https://github.com/xqlsystems/xarray-sql/issues/285)
with the hardware and versions:
- xarray-sql's GPT demo (demo 2);
- xarray-sql's Samudra demo (demo 4);
- torch-sql's nanoGPT.

GPUs come early. §3.1 shows a GPU pays off only together with resident
state and dense intermediates, so the GPU milestone (M2) brings both. NVIDIA
GPUs, on the cloud or on the Beta cluster, are available to the project half
a day a week. The contributor's Apple GPU work continues in parallel on MLX.

- **M0. A CPU runtime.** Sessions, table handles and programs on DataFusion
  and DuckDB, implementing `ddxdb.Backend`, with the counters of R8.
  xarray-sql's demos and torch-sql's nanoGPT run on it, with the same
  results as today. Kept small on purpose.
- **M1. Resident state.** Persistent tables and atomic publication, on the
  host, with placements designed so a device is just another placement.
  torch-sql's optimizer step becomes one program instead of a query per
  tensor.
- **M2. A GPU, with array regions.** Array regions (§7.2), with NumPy on
  the CPU as the reference and one GPU backend, CUDA on an NVIDIA GPU, with
  state resident on the device. MLX follows, or comes first if the
  contributor's Apple work gets there sooner. Needs einfold's answer on
  design B and spike 6.
- **M3. Shared subplans and memory budgets.** Cut points move from
  torch-sql into the runtime. If DataFusion's CTE fix has landed by then,
  this milestone shrinks to memory budgets.
- **M4. Concurrent steps** (ddx#157).
- **M5. Distribution**, within one NVL72 rack.

The spikes that need NVIDIA hardware (2, Sirius; and 3, Grace memory) are
the first work for those half days, since they don't wait on any design
decision.

M2 is the critical path, and it waits on einfold's answer on design B and
on spike 6. Spike 6 needs no special hardware, so it is the first work for
the agent that implements the runtime. If einfold declines design B, M2
loses its array regions, and §3.1 says a GPU without them gains almost
nothing; the milestones would then need replanning.

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
4. **The runtime's language.** The cost of a Python core against a Rust one
   for M0 and M4. First, whether DataFusion's and DuckDB's Python bindings
   release Python's global interpreter lock while they plan and execute. If
   they do, concurrent steps can run from Python. If not, they run one at a
   time.
5. **Where the last 3.76× goes.** A trace of one warm resident nanoGPT step
   on the M5 Ultra, against PyTorch MPS's, both with and without
   compilation, by kind of operation. Gathers and scatters from index
   joins, unfused elementwise work, and attention's backward pass (which
   ddx#251 targets) are the candidates. The contributor's branches have
   started on this ("Compare the GPU programs for PyTorch MPS and SQL
   training"), and their findings should be read before the spike runs.
6. **The array operations nanoGPT and Samudra need.** Translate demo 2's and
   demo 4's programs by hand into the operation set of design B (§7.2), to
   size it before einfold commits to it.

---

## 17. Questions for reviewers

### 17.1 For the ddx agent

The ddx agent's answers follow each question. Its review has the details.

1. Can a program's checks be marked structural or data (§6.1, §11.2), or
   emitted as separate steps that the runtime can evaluate once, when it
   compiles?

   *ddx:* They are separate plans already, run before the steps. ddx has
   three checks, and all are of one kind: some key columns of one table are
   unique. None reads a value column. ddx would state each check's promise
   (the table, its columns, unique), not a structural or data label, and
   the runtime would decide when to run it:
   - once per compile, for a table whose keys don't change;
   - on each run, otherwise;
   - never, when a fact from a frontend gives the same promise.

   In a training step only the weights are in `wrt`, so a compiled step
   runs no checks.
2. Is the step graph of ddx#157 the right form for the runtime to find
   shared subplans (§7.4), so that nothing needs to turn plans back into
   SQL?

   *ddx:* Yes for scheduling and lifetimes, and ddx will expose it on
   `Program`. No for shared subplans. Neither producer emits a
   `ReferenceRel`, so sharing has to be found before planning: in SQL, as
   ddx#168 does, or in the frontend's program. Inside ddx's programs it is
   found already (§18.10).
3. Should the runtime implement `ddx_ad::Backend` in Rust from the start, or
   is `ddxdb.Backend` in Python enough until M4?

   *ddx:* Python is enough until M4, if the runtime implements `compile` and
   keeps tables in place; otherwise every run plans every step. But
   Python's compiled path needs a fork of datafusion-python today (§3.4),
   which R2 rules out, and DuckDB needs `to_sql` exported (§11.2). ddx's run
   loop is sequential in both languages, so M4 needs ddx#157 either way.
4. Would ddx take *save points* with a loss, subplans it computes as steps
   and differentiates through (§7.5)? And should it learn them from the
   frontend, or from an optional `Backend` method that the runtime
   implements?

   *ddx:* It does already, for CTEs read more than once and for a
   program's steps (§7.5). It would also take extra save points, by name,
   as options from its caller, not from a `Backend` method. ddx's `prepare`
   caches a program by its plan and the session key, and a method that
   answers from the runtime's memory would leave a cached program stale.
   Save points and ddx#251's regions should be one surface: a name on a
   relation, with a policy per name (§18.10).
5. Where does ddx#260 stand? `ddx_stop_gradient` on DuckDB, and for types
   other than `DOUBLE`, is a prerequisite for torch-sql on the runtime
   (§7.5).

   *ddx:* Open, and not started. It is small, and it is ddx's next fix:
   - ddxdb's `Context` registers the Rust function, which takes any type,
     by default;
   - ddxdb's DuckDB backend registers it on the connection it plans on.

### 17.2 For the einfold agent

The einfold agent's answers follow each question. Alex decides einfold's
scope, so design B also needs his approval in einfold's own design doc.

1. Is design B (§7.2) one einfold wants to own? That means translating
   regions of steps into array programs, with proofs of exactness, rather
   than single kernels.

   *einfold:* Yes, on the conditions of §18.3, §18.7 and §18.8. B fits
   einfold's architecture as a third output form. The other two are
   rewritten SQL and einfold's own operator. B keeps einfold's rule that it
   rewrites plans and never executes them: the runtime is the host.
2. What should the array operation set be, and should it be einfold's own
   format or an existing one? The existing candidates are the Python array
   API standard's operations, or a subset of StableHLO, the operation set
   that JAX and XLA compile.

   *einfold:* Neither, and no new set either. The program stays SQL, and
   einfold's IR nodes annotate it (§18.3). Alex rejected StableHLO as XQL's
   IR. It, like the array API standard, may still be one backend's private
   lowering target. Neither one can say whether a row exists.
3. Is `DenseProgram` (einfold#157) the place to start, a prototype to learn
   from, or neither?

   *einfold:* A prototype to learn from, and two things more:
   - a corpus: the operators and spellings it handled to train a 123.6M
     model are the best input to spike 6;
   - an oracle for einfold's translations.

   It isn't the place to start. It interprets DataFusion's operators one at
   a time, which is design A, and the translation belongs in Rust beside
   einfold's detection and property tests, so that DuckDB can use it too.
4. Which facts does a region's proof need from frontends, and does the fact
   table carry them today? torch-sql can state completeness, positional
   indexes, unique keys and no NULLs for every table (§3.3), and ranges for
   index values where it knows them.

   *einfold:* The facts are:
   - each dimension's extent, with integer keys `0 .. n − 1`;
   - completeness, or a domain table;
   - unique tuples;
   - no NULL values;
   - for gathers, that each index lies in range;
   - for sums einfold reorders, that values are finite and bounded
     (einfold's float guard, its design §8.6).

   torch-sql's list covers the structural facts, and
   [einfold#39](https://github.com/xqlsystems/einfold/issues/39) states
   the same for xarray-sql. No fact table carries any of them yet (§11.3).
   Facts about values are true for one run only: einfold#39 found that
   xarray-sql's values change between queries. So a compiled program
   always keeps some data checks. The runtime should provide facts about
   the tables it writes, and checking mode should verify the facts a region
   relied on.
5. Would einfold rather *check* a candidate array program than *find* one?
   torch-sql knows each op's array meaning before it writes the SQL, and
   could pass a candidate as a hint. Checking that a candidate computes the
   SQL may be easier than searching for one, and a host that ignores the
   hint still gets the SQL's answer.

   *einfold:* Yes to hints, as guidance for detection, never in place of
   its proof. A hint einfold fails to confirm becomes a counted decline
   with a reason (R8). That gives the measure of detection's coverage that
   nobody has today.

---

## 18. Open questions

1. **Which array library runs array programs on NVIDIA GPUs.** There are
   three candidates:
   - JAX, whose compiler XLA runs StableHLO programs and fuses them. That
     would pair naturally with a StableHLO-based operation set (§17.2).
   - PyTorch, where torch-sql already lives.
   - CuPy, a thin NumPy-like library for CUDA.

   Two facts bear on it. When PyTorch shares a GPU, as it does for torch-sql
   as a co-processor, JAX by default takes 75 percent of the GPU's memory
   when it starts, and the two allocators then compete for the rest.
   PyTorch as the backend has no such conflict: an array program's results
   reach PyTorch without any transport, and `torch.compile` can fuse them,
   as XLA would for JAX. The choice follows einfold's answer on the
   operation set.
2. **Rust or Python first.** v0 leaned to Rust first: DataFusion, ddx's
   `Backend` trait and einfold's dense backend are Rust, and concurrent
   steps need to run outside Python's lock. v1 leaned to Python first: the
   interfaces are still moving and are cheaper to change in Python, the
   costly work (engine planning, kernels, array libraries) is native
   already, and `ddxdb.Backend`, xarray-sql and torch-sql are Python.

   The co-authors now agree on Python first. torch-sql's prototype is the
   evidence. It is all Python, and Python was never its bottleneck: its
   slow parts are the number of queries (a query per tensor in the
   optimizer, §3.3) and planning, which are native. Its internals changed
   in every round of review, which Python made cheap. Its one bug of the
   kind Rust prevents, dropping a table still in use, came from tying a
   table's lifetime to Python's garbage collector, which explicit handles
   (R4) avoid in either language.

   Three conditions keep the way to Rust open:
   - table handles and their lifetimes are explicit, never tied to garbage
     collection;
   - the Python core stays a thin layer over the engines' and array
     libraries' bindings;
   - spike 4 still decides concurrency. If the engines' bindings hold
     Python's lock while they plan or execute, concurrent steps (M4) need
     Rust, and that milestone is where it starts.

   einfold adds a cost to weigh (🪭). From Python, einfold reaches
   datafusion-python's sessions through datafusion-ffi, DataFusion's
   interface for extensions built separately from it. Through it, steps
   einfold doesn't match run 1.25× slower and plan 2.9× slower than on
   DataFusion alone
   ([einfold#58](https://github.com/xqlsystems/einfold/issues/58),
   [#81](https://github.com/xqlsystems/einfold/issues/81)). The FFI also
   has open gaps: object stores don't cross it (einfold#149), and functions
   planned through it lose the session's configuration (einfold#137).
   Spike 4 should measure the FFI beside the lock. A Rust core would link
   DataFusion and einfold directly, and pay neither cost.
3. **Who owns dense execution** (§7.2): design A, B or C. This draft
   proposes B, and einfold's answer settles it.

   einfold's position (🪭): B, with two changes to §7.2.

   **The program stays SQL.** Alex rejects StableHLO as XQL's IR. An
   array program written in SQL is already the most portable IR, because
   every engine runs it. einfold would also drop §7.2's list of array
   operations. A separate array program, in any format, would have to be
   kept equal to the SQL forever, which is design A's risk. It would also
   lose two things:
   - **existence**: a compact array has three states per cell (absent,
     NULL or a value), and SQL's rules for them aren't masks:
     - a group exists only if some row reached it;
     - `SUM` over no rows is no row;
     - a zero sum's sign follows the engine;
   - **structure**: lowered to `einsum`, `exp` and `einsum`, fused
     attention holds `N²` scores again, and a convolution lowered to
     gathers loses its one matrix product per tap (einfold#155).

   Instead, a region is a SQL plan in which einfold has marked nodes:
   - folds;
   - pipelines;
   - shifted folds (convolutions);
   - gathers on value indexes;
   - the elementwise expressions between them.

   Each node carries its subplan, as every node einfold detects already
   does. A backend runs the node on arrays, or declines it, and then the
   node's SQL runs. Lowering to NumPy, MLX, PyTorch or JAX is private to
   each backend.

   The cost is that SQL is portable in meaning, not in speed: everything
   fast still depends on einfold recognizing structure.

   **Checks move onto the device.** einfold checks data at run time
   today (§7.2). Under B, each such check compiles to an on-device
   predicate, ANDed into the one validation flag that R5 reads at the end
   of a run. A failed flag is a decline at run time (§7.3).

4. **Substrait as the plan format between layers.** ddx already speaks it.
   Engines without a Substrait producer would need SQL text in their
   dialect, which ddx tracks in
   [ddx#159](https://github.com/xqlsystems/ddx/issues/159). einfold works on
   DataFusion's logical plans, so a step may need both forms.
5. **Low precision.** SQL engines have no bfloat16, and GB200 hardware is
   built for it. Kernels could compute in low precision with the SQL types
   as the reference, but R1's floating-point bound would then need stating
   per kernel. The nanoGPT work already ran in float32 on the GPU against
   float64 SQL, with its tolerance stated per backend.

   einfold proposes a definition (🪭). A fixed tolerance isn't enough. A
   float sum of `n` terms, in any order, is within `(n − 1)·u·Σ|terms|` of
   the exact sum, as long as nothing overflows (einfold's design §8.6),
   with `u` the unit roundoff of the precision the backend computes in.
   Then float32 on a GPU is an explicit, per-session choice of `u`, as
   einfold#156 already makes precision the host's choice. The same section
   of einfold's design gives einfold's rule on determinism: einfold never
   adds nondeterminism, and its kernels give the same bits for any thread
   count. The runtime should state its own rule, because GPU atomic adds
   don't give the same bits.
6. **Which engine is the reference** when two engines disagree (§3.3), and
   how often checking mode (§7.3) should sample in production.
7. **Who chooses save points.** §7.5 has the runtime choose them, from
   planning budgets and memory. einfold's position (🪭): the runtime
   chooses them together with einfold, because saving can destroy what
   einfold fuses. Saving attention's scores leaves no pipeline to fuse, and
   [einfold#135](https://github.com/xqlsystems/einfold/issues/135), the
   fused backward pass, needs ddx to recompute the scores instead
   (ddx#251). einfold would report whether a candidate save point splits a
   node it can fuse, and a split taken anyway counts as a decline with
   that reason (R8).

   ddx's position (∂): the runtime chooses, and passes its choice to ddx as
   options. ddx doesn't ask its `Backend`, so that a cached program stays a
   function of its plan and its options (§17.1, question 4).
8. **What M2's array regions are.** M2 as written needs einfold to
   translate regions across steps. That is einfold's program mode, planned
   for einfold's M7, after DuckDB, facts and sparse kernels. einfold's
   position (🪭) is to stage it:
   - **B-0, one einfold node per region.** The runtime runs the node on
     compact arrays it keeps resident, with the node's checks as data
     checks and the node's SQL as the fallback. This is M2.
   - **B-1, regions within one step.**
   - **B-2, regions across steps,** needing ddx#157 and ddx#251.

   Before B-0, spike 6 should report how much of each step's time in demo 2,
   demo 4 and torch-sql's nanoGPT already runs inside einfold's nodes,
   from `EXPLAIN ANALYZE` with einfold on. Reordering einfold's roadmap is
   Alex's decision as einfold's maintainer.
9. **Who chooses to recompute** (R7, §4.1's goal 5, §8). The draft makes
   saving, recomputing and placement all the runtime's choice. ddx's
   position (∂) has two parts.

   **Recomputing is opt-in.** Alex decided on 2026-11-01, for ddx#251,
   that recomputing is an explicit policy people opt into, as with
   `jax.checkpoint`, and never the engine's own choice. Where a value is
   held (a saved shared subplan, a table on a device) changes no work, so
   it can stay the runtime's choice. Recomputing instead of saving does
   more work on plain engines, and the user chooses it, as a policy (§8).
   The runtime applies a memory-budget policy only when the session turns
   one on. Saving more than a policy asks stays the runtime's choice.

   **R7 needs a condition.** Recomputing gives the same result only for a
   plan that gives the same rows each time it runs, and then only within
   R1's bound, since a `SUM` adds partitions in the order they arrive. A
   volatile function, a `LIMIT` without a total order, or a rank over ties
   rules it out. ddx already refuses to compute such a plan twice
   (ddx#254). But its test for volatility is a list of function names, so
   ddx should read volatility from the engine through its `Backend`. §10's
   random function must be declared deterministic for a recomputed dropout
   to draw the same mask (§7.5).
10. **Who draws step boundaries** (§7.4, §7.5, §8). ddx's position (∂).

    ddx already draws them, for four reasons:
    - each aggregate the backward pass reads is saved;
    - each CTE read more than once is planned alone (ddx#168);
    - a step deeper than a consumer can decode is split every 32
      relations;
    - a step that one later step reads once is computed inside it
      ([ddx#231](https://github.com/xqlsystems/ddx/pull/231)), unless its
      reader would pass 2,000 expression nodes.

    A second chooser with its own budget would undo the first. So the
    runtime should take ddx's program as ddx writes it, and change its
    boundaries only through ddx's inputs:
    - save points and recompute policies, by name, as options to ddx (one
      surface with ddx#251's regions);
    - engine limits, such as the depth a consumer decodes and a planning
      budget, as facts the `Backend` reports, which ddx reads in place of
      its constants.

    The names come from a marker, `ddx_name(expr, 'label')` (proposed on
    ddx#251, since CTE and view names don't survive either producer), or
    from CTE names in ddx's SQL path, which reads the SQL itself. The same
    names would carry ddx#259's provenance and einfold's hints.

    §7.4 also changes. The runtime can't find shared subplans in plans,
    because neither producer emits a `ReferenceRel` and the copies are
    already written out (§3.3). It takes them from the frontend's program,
    or from SQL before planning. For plain queries, ddx can offer ddx#168's
    split as a public function, so that the runtime has no second
    implementation of it.

---

## Decision log

Nothing is decided yet. The proposals this draft makes, for discussion:

- **P-A.** The runtime is a separate project. einfold stays free of device
  execution, and keeps its CPU reference operator (from xarray-sql#307).
- **P-B.** The unit of work is a program with explicit state, not a query
  (R3, R5).
- **P-C** (revised in v1). einfold translates relational regions into array
  programs and proves them exact; the runtime runs array programs and
  never interprets relational operators (design B, §7.2). Design C stays
  open through spike 1.
- **P-D.** The runtime implements ddx's and einfold's public backend
  interfaces, rather than either project depending on the runtime.
- **P-E** (new in v1). Where values are saved is the runtime's decision and
  invisible to differentiation (R7). Declines are counted, and benchmarks
  run strict (R8).
- **P-F** (new in v1). The project is named `xql-runtime`, and is packaged
  as §14.2 describes.
- **P-G** (new in v1). A GPU milestone, with array regions and resident
  state, comes third (M2), using NVIDIA GPUs on the cloud or the Beta
  cluster.
- **P-H** (new in v1.1). Python first, with explicit handles and a thin
  core, until spike 4 or M4 shows Rust is needed (§18.2).
- **P-I** (new in v1.1). The runtime chooses save points and ddx
  differentiates through them, so that no frontend runs a chain rule of its
  own (§7.5).
