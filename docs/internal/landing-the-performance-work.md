<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# Landing the performance work: a migration guide for in-flight changes

_status_: Internal draft, for review. The destinations follow the
[design](../design.md) as of v1.8, and change if it does.
_audience_: the contributor whose nanoGPT and Modded-NanoGPT work this is,
the maintainers of xarray-sql, einfold and ddx, and their agents.

The nanoGPT performance work showed where the speed in an SQL training
step comes from (supplement [S1.1](../supplement.md)). Each finding comes
from its own workload, so the ratios don't multiply:
- **compact arrays between operators** took a 123.6M-parameter GPT at
  context 16 from 163.6 s to 2.43 s per warm step on an Apple GPU;
- **state that stays on the device** took nanoGPT's 10.7M-parameter
  Shakespeare recipe from 2.02 s to 101 ms per warm step;
- **a growing set of proofs** brought Modded-NanoGPT, 124M parameters at
  context 1,024, to 238.95 ms per warm step, against 185.40 ms for PyTorch
  on the same M5 Ultra: 1.29 times. Both ran at batch 1 in float32, and
  PyTorch ran eagerly, without `torch.compile` and with unfused AdamW.

That work landed in xarray-sql and in a fork of einfold, because nothing
else existed to hold it. Most of it belongs in DataFusion Accelerated, some
in einfold and ddx, and some stays where it is.

This guide says, for each piece, where it lands, what changes on the way,
and what the code looks like before and after. "Before" is the code in the
in-flight PRs and branches. "After" is the design's proposed API, which
doesn't exist yet.

**What survives, plainly.** Most of the code doesn't move. `SharedExecutor`,
`DenseExecutor`, `TrainingSession` and einfold's `DenseProgram` retire. The
proofs are rewritten on einfold's Rust IR, because they are written on
`DenseProgram`, an interpreter of relational plans that the design rejected
as design A (supplement S2.1). What survives is what made the work worth
doing:
- the **measurements**, which are the evidence the design rests on, and
  the numbers every step of the move is measured against;
- the **tests and ablations**, which become the specification the rewrite
  has to pass;
- the **ideas**, which are now the design's own: compact arrays, resident
  state, validation before publishing, and proofs that let more of a step
  run as array code.

Some pieces do move nearly as they are: ddx#256 and #257, #301's own
changes, the MLX kernels and the recipes.

---

## 1. Summary

| Change | Lands in | Disposition |
|---|---|---|
| ddx#256, the derivative of `erf` | ddx | merge after review |
| ddx#257, primal outputs of a prepared run | ddx | merge after review; the ddx integration consumes it |
| xarray-sql#301, plan reuse and faster registration | xarray-sql | merge after the changes its review asked for |
| xarray-sql#302, `SharedExecutor` and MLX kernels | split | the caching retires: ddx's SQL path covers gradients, and forward queries store shared views as tables until DataFusion plans a shared CTE once; MLX kernels become an array backend integration |
| xarray-sql#303, `DenseExecutor` | DataFusion Accelerated | retires; device plans over compact arrays do its job, and its MLX operators seed the MLX integration |
| xarray-sql#305, `TrainingSession` | split | retires; the runtime's `Program` does its job, and the Shakespeare recipe stays as demo 3 |
| xarray-sql#304, reports | xarray-sql docs and benchmarks | keep as evidence; numbers already cited in S1 |
| einfold#156, the dense backend callback | split | detection changes go to einfold; the callback becomes the B-0 interface |
| einfold#157, `DenseProgram`, and 12 commits on it in einfold's fork | einfold | retires; a corpus, an oracle and a specification, with its proofs rewritten on einfold's Rust IR |
| 33 branches in xarray-sql, one stack of 44 commits, not yet PRs | by commit | see §6 |

The rule behind every row: DataFusion Accelerated runs things on devices,
einfold recognizes and proves array structure, ddx differentiates, and
xarray-sql registers Datasets and holds the demos. A piece of code goes to
the project whose job it does.

---

## 2. The rules the move follows

- **Public interfaces only** (design R2). The executors import 18 private
  names from einfold, einfold's `_native` module, and ddxdb's `_ddxdb`. Each
  becomes a public interface in its own project, or the code that needs it
  moves into that project.
- **Released packages, not forks.** `accelerators/einfold_mlx/build.py`
  clones and builds pinned commits of personal forks. That retires once the
  pieces are released from their own repositories.
- **The SQL is the meaning** (R1). Every fast path keeps its SQL as the
  fallback and the reference, as the executors already do.
- **The core integrates no hardware** (R13, D13). MLX, like CUDA, comes in
  through an integration, a separate package, never the core.
- **Declines are counted, not restarted** (R8). `DenseExecutor` restarts a
  whole call on the SQL path when one operator declines. The runtime falls
  back one node at a time.
- **Moved code takes its new project's header.** Code that moves, or is
  rewritten from the contributor's, carries the receiving project's REUSE
  header, with Alex's copyright and the project's authors, and the
  contributor is credited as an XQL author in the commits that carry it.
  einfold's `dense_program.py` carries `SPDX-FileCopyrightText: 2026
  OpenAI`, which is a mistake, since the contributor doesn't work for
  OpenAI. It is corrected wherever that code, or a rewrite of it, lands.

---

## 3. ddx: review, then merge both

### 3.1 ddx#256, the derivative of `erf`

Small and self-contained: it lets ddx differentiate exact GELU, which
nanoGPT and Samudra both use. Neither it nor #257 has been reviewed yet.
Merge it in ddx as it is, once it has.

### 3.2 ddx#257, primal outputs of a prepared run

`Prepared.values` exposes the loss a prepared derivative already computed,
so a training step needs no separate forward query. That is a ddx
interface, and it belongs in ddx. The consumer moves: `TrainingSession`
reads it today, and the ddx integration of DataFusion Accelerated will.

```python
# Before (xarray-sql#305, through TrainingSession): the step reads ddx's primal
result = resident_step()
loss = result.loss_value()            # from Prepared.values, ddx#257
```

```python
# After (proposed): a program names its outputs, and the ddx integration
# fills them from Prepared.values, with no second forward query
compiled = Program(ctx, STEP, state=STATE, outputs=["loss"]).lower().compile()
loss = compiled.run(lr=lr).result()["loss"]
```

---

## 4. xarray-sql

### 4.1 #301, plan reuse and faster registration: merge after changes

`prepare()` for gradients, `Context(threads=)` and faster Dataset
registration are xarray-sql's own work. They stand alone, and #302 to #305
are stacked on them, so merging #301 first shrinks every later diff. Its
review asked for three changes first:
- `_safe_physical_reuse` turns off the plan reuse of #287, which is
  merged, in DataFusion 55's default configuration: a prepared join went
  from 0.30 to 0.60 ms per call on an i7-8700;
- the UDF tracking moves to #302;
- the second `ad.prepare` API goes.

### 4.2 #302, `SharedExecutor`: retire the caching, keep the kernels

`SharedExecutor` computes each CTE once per run, which DataFusion doesn't
do yet (xarray-sql#267). DataFusion is adding `WITH … AS MATERIALIZED`, a
CTE planned once, with several follow-ups still open upstream. DuckDB
already computes a CTE read more than once a single time. Until
DataFusion's lands:
- **gradients** use ddx's SQL path, which plans each CTE read more than
  once on its own (ddx#168). It is the only one of these paths merged and
  public today;
- **forward queries** store each shared view once, as demo 4's eager mode
  does, with `CREATE TABLE … AS`;
- **xarray-sql#290**, which makes `prepare` run each CTE or view once, is
  a draft waiting on DataFusion;
- **the runtime's save points**, chosen with einfold (design §6.4), come
  later.

```python
# Before (xarray-sql#302)
from xarray_sql.shared import SharedExecutor

ex = SharedExecutor(ctx, inline_ctes=["scores", "weights"])
for name, sql in MODEL_VIEWS.items():
    ex.register_view(name, sql)
run = ex.prepare(LOSS_SQL)
loss = run()
```

```python
# After, today: no executor. A view in inline_ctes stays a view, so einfold
# can fuse it; every other shared view is stored once per run, as demo 4's
# eager mode does. Gradients go through ddx's SQL path (ddx#168).
for name, sql in MODEL_VIEWS.items():
    if name in ("scores", "weights"):
        ctx.sql(f"CREATE VIEW {name} AS {sql}")
    else:
        ctx.sql(f"CREATE OR REPLACE TABLE {name} AS {sql}")   # again each run
loss = ctx.prepare(LOSS_SQL)()                                # xarray-sql#287

# Later, once DataFusion plans a shared CTE once, and with the runtime:
# plain views, and inline_ctes becomes a policy by label (design §6.4, §6.5)
schedule.recompute(save_only_these_names("attn.out"))
```

The SQL parser in `src/shared.rs` retires with it, since ddx#168 already
finds CTEs, and the use of ddxdb's private `to_sql` goes too. What moves:
- **MLX kernels** (`xarray_sql/mlx.py`) become an *array backend*
  integration for Apple GPUs, `datafusion-accelerated-metal` (§4.3);
- **Arrow tables that declare their sort order** (`_SortedArrowTable`) move
  into xarray-sql's registration, which knows a Dataset's dimension order.

### 4.3 #303, `DenseExecutor`: becomes device plans

`DenseExecutor` keeps compact arrays across ddx's stages, which is what
DataFusion Accelerated's device plans do: `KernelExec` nodes run on compact
arrays, and handle batches carry them between nodes without expanding to
rows (design §5.3). The executor retires. What carries over is its job,
its tests and measurements, and its MLX operators, which seed the MLX
integration.

```python
# Before (xarray-sql#303)
from xarray_sql.mlx import MLX
from xarray_sql.dense import DenseExecutor

ctx = xql.Context().with_extensions(MLX(precision="float32", min_gpu_ops=1 << 33))
ex = DenseExecutor(ctx, precision="float32", fallback=False, max_array_bytes=1 << 30)
run = ex.prepare(LOSS_SQL)
loss = run()
print(ex.statistics())
```

```python
# After (proposed)
from datafusion_accelerated import Accelerator
from datafusion_accelerated_metal import Metal, Mlx

ctx = xql.Context().with_extensions(EinFold(), Accelerator(), Metal(), Mlx(min_gpu_ops=1 << 33))
ctx.sql("SET accelerated.strict = true")             # fallback=False
ctx.sql("SET accelerated.precision = 'float32'")     # an explicit choice of the error bound's u
loss = ctx.prepare(LOSS_SQL)()                        # device plans; EXPLAIN shows them
```

| `DenseExecutor` | DataFusion Accelerated |
|---|---|
| `fallback=False` | `SET accelerated.strict = true` (R8) |
| `fallback=True`, restarting the whole call | a decline per node, counted, with the node's SQL as its fallback (§6.3) |
| `precision="float32"` | the session's precision, and its error bound (§6.3) |
| `max_array_bytes`, `max_cached_bytes` | reservations from the device memory pool (§5.2) |
| `statistics()` | `EXPLAIN ANALYZE` per step, `report()` per program (§6.6) |
| `_dense_mlx.py`'s operators | the MLX array backend's lowering of einfold's nodes, private to the integration, reusing the kernels where the operations match |
| `DenseProgram`'s interpretation of relational operators | einfold's nodes, with the proofs rewritten on einfold's Rust IR (§5.1 below) |

### 4.4 #305, `TrainingSession`: becomes a program

`TrainingSession` is the first resident training loop in SQL, and the
design's program model (§5.4) is built from it. Each of its parts has a
home:

```python
# Before (xarray-sql#305, benchmarks/ml/03_shakespeare.py)
from xarray_sql.training import TrainingSession

resident = TrainingSession(ctx, precision="float32", max_cached_bytes=16 << 30,
                           inline_ctes=host.executor.inline_ctes)
for name, query in host.executor._views.items():          # a private attribute
    resident.executor.register_view(name, query)
for name in host._tables:
    resident.bind(name)
resident.index_domains("batch", {"tok": ("wte", "tok"), "next": ("wte", "tok")})
resident.dropout(DROPOUT_TABLES, probability=0.2, seed=seed + 1, step=iteration)
step = resident.prepare_step(
    gradients=GRADIENT_SQL,                  # name -> SQL
    stages={"clip": CLIP_SQL},
    updates=ADAMW_SQL,                       # name -> SQL
    assignments=ASSIGNMENTS,                 # state table -> (result, columns)
    compile=True,
)
loss = step().loss_value()
```

```python
# After (proposed)
from datafusion_accelerated import Accelerator, Program
from datafusion_accelerated_metal import Metal, Mlx

einfold = EinFold()
ctx = xql.Context().with_extensions(Ddx(), einfold, Accelerator(), Metal(), Mlx())
for name, sql in MODEL_VIEWS.items():
    ctx.sql(f"CREATE VIEW {name} AS {sql}")

# index_domains becomes an exact fact in einfold's fact table (einfold's
# design §8.1), for its gathers. The fact table has no user interface yet,
# so these two calls are placeholders for one (§7).
einfold.facts.declare("batch", "tok", within=("wte", "tok"))
einfold.facts.declare("batch", "next", within=("wte", "tok"))

program = Program(
    ctx,
    {**GRADIENT_SQL, "clip": CLIP_SQL, **ADAMW_SQL},  # statements by name (design §5.4)
    state=PARAMETERS_AND_MOMENTS,    # initial values; the statement of the same name computes each next value
    inputs=["batch", *DROPOUT_TABLES],               # dropout masks are uploaded (below)
    outputs=["loss"],
)
compiled = program.lower().compile()
masks = torch_masks(DROPOUT_TABLES, probability=0.2, seed=seed + 1, step=iteration)
loss = compiled.run(lr=lr, batch=batch, **masks).result()["loss"]
```

| `TrainingSession` | DataFusion Accelerated |
|---|---|
| `bind(name)` | a state table in `Program(state=...)`, or a registered input |
| `index_domains(...)` | an exact fact in einfold's fact table, which needs a user interface (§7) |
| `dropout(..., seed, step)` | masks uploaded as input tables, since demo 3 is checked against PyTorch's random numbers (design §9, S3.10). The counter-based random function, written in the SQL with `$step` as a parameter, serves where no other framework's numbers must match |
| `prepare_step(gradients, stages, updates, assignments)` | one `Program` of statements by name, composing ddx's program with the update steps (§5.4) |
| `assignments` | statements named after the state tables they compute |
| `compile=True` | `.lower().compile()` |
| validation before publishing | R5: state changes only when a run's checks pass, with one flag read per run (§6.3) |
| `export(...)` | an explicit read of a table handle |
| NPZ checkpoints | Zarr checkpoints, which xarray-sql reads back as Datasets (§9) |
| reading ddx's primal | the program's `outputs`, through ddx#257 |

What stays in xarray-sql: the Shakespeare recipe (`03_shakespeare.py`) as
demo 3, rewritten on the public API, with its PyTorch oracle; and `erf` as a
DataFusion function, until DataFusion has one.

### 4.5 #304, reports: keep as evidence

The reports are the evidence the design rests on. Their headline numbers
are cited in supplement S1.1 with their limits. The raw records stay in
xarray-sql's docs, and new runs are reported through the benchmark tracking
in xarray-sql#285 and #306, so later milestones are measured the same way.

---

## 5. einfold

### 5.1 #157 and the fork's stack: `DenseProgram` retires

`DenseProgram` interprets DataFusion's logical operators on compact arrays,
which is design A, a second implementation of SQL (supplement S2.1). It
isn't merged, and it doesn't move. In the contributor's fork of einfold, 12
more commits stack on #157, on `codex/compact-graph-optimizer`,
`codex/typed-reduction-patterns` and ten `m7/*` branches, from
`m7/compact-reduction-metadata` to `m7/numerical-value-identities`. Their
tip, `3108f7a`, is the commit xarray-sql's builder pins. They add 2,366
lines, 660 of them to `python/einfold/dense_program.py`, which is now 2,211
lines, and they change no Rust. Most of the proofs live there.

So the proofs are rewritten on einfold's Rust IR, beside its detection, as
einfold's B-0 interface: each detected node's IR, checks and subplan,
handed to the runtime as a binding (design §6.2, §10.3). What carries over:
- **a corpus**: the operators and spellings `DenseProgram` handled to
  train a 123.6M model are the best input to spike 6, which measures
  einfold's coverage;
- **an oracle**: its results check einfold's new translations;
- **a specification**: the stack's tests (`test_compact_optimizer.py`,
  `test_group_proofs.py`, `test_presence_predicates.py`,
  `test_reduction_metadata.py` and `test_coordinate_reshape.py`) say what
  each proof must accept and reject, and the ablations in its reports say
  what each must gain;
- **the reports**: thirteen notes at the fork's root, such as
  `NUMERICAL_VALUE_IDENTITIES.md`, explain each proof.

### 5.2 #156, the dense backend: split

Two kinds of change are in #156:
- **Detection** (`detect_pipelines.rs`, `fused.rs`, `dense.rs`): einfold's
  own, to review in einfold on its merits.
- **The backend callback** (`dense_backend.rs`, `backend.rs`): a session's
  Python callback that takes einfold's matrix products and attention. The
  B-0 interface replaces it, and the callback's kernels become the MLX
  array backend's.

```python
# Before (einfold#156): a callback einfold calls from inside EinFoldExec
class MlxBackend:
    def matmul(self, a, b, ...): ...
    def attention(self, q, k, v, ...): ...
ctx = SessionContext().with_extensions(EinFold(dense_backend=MlxBackend()))
```

```python
# After (proposed): einfold marks the node; the runtime runs it on an
# array backend integration, which may decline it
ctx = SessionContext().with_extensions(EinFold(), Accelerator(), Metal(), Mlx())
```

### 5.3 Proofs in xarray-sql that belong in einfold

Five modules on xarray-sql's later branches build on `DenseProgram`:
- `_dense_adjoint.py` proves that ddx's derivatives of attention match a
  kernel;
- `_dense_proofs.py` proves when a row exists, from coordinates alone;
- `_dense_attention.py` proves the shape of attention from einfold's
  reduction graph;
- `_dense_indexed.py` (210 lines) checks when compact indexed partial sums
  can be added on the GPU in the order of SQL's `UNION`;
- `_dense_segment.py` (42 lines) sums dynamic SQL groups on the GPU in a
  fixed order. It is a kernel, not a proof.

Proving that SQL computes an array algorithm is einfold's job, so the four
proofs are rewritten there, in Rust, with the fork's. `_dense_segment.py`
goes to the MLX integration with the other kernels, as a fixed-order sum
(design R9).

They can't be rewritten before einfold's B-0 interface exists, so the
rewrite is step 3 of §8. Their specification goes first: in step 1, the
tests and ablations move into einfold, marked as expected to fail, so the
rewrite has a target from its first commit. Which proofs are rewritten
first follows from which closed the most of the gap (§9).

---

## 6. The unreviewed branches

These are classified by each branch's name and latest commit, so the
destinations are proposals to confirm with the contributor. The branches
aren't independent topics:
- **In xarray-sql**, 32 of the 33 branches form one stack of 44 commits on
  `main`, mostly one branch per commit. `indexed-gradient-evidence` sits
  beside it. The stack adds 76,901 lines: about 55,000 are JSON, logs and
  CSV files of evidence, and 14,810 are Python. A destination per branch
  is a destination per commit, and each commit depends on the ones before
  it, so they land in stack order or are cut again by topic.
- **Most of the branches listed under einfold hold no proof code.** Their
  commits change `accelerators/einfold_mlx/build.py`'s pin, plus reports
  and benchmarks, and the code they pin is the fork's stack (§5.1).

<details>
<summary>All the branches, by destination</summary>

| Destination | Branches |
|---|---|
| **einfold**: proofs and detection, rewritten on its Rust IR | in einfold's fork, `codex/compact-graph-optimizer`, `codex/typed-reduction-patterns` and the ten `m7/*` branches. In xarray-sql, the pins they set: `predicate-proof-integration`, `symbolic-group-validity`, `numerical-value-identities`, `coordinate-map-reuse` and `einfold-graph-optimizer` |
| **DataFusion Accelerated, the MLX integration**: execution on Apple GPUs | `gpu-domain-indexing`, `gpu-output-pruning`, `indexed-gradient-accumulation`, `shared-indexed-gradient`, `numerical-pattern-lowering`, `resident-attention-lowering` |
| **DataFusion Accelerated, the core**: rules every integration shares | `fixed-segment-sums` (fixed-order sums, design R9, which adds `_dense_indexed.py` and `_dense_segment.py`), `frozen-training-policies` (validation policies, design §6.3), `resident-attention-adjoints` (choosing kernels for ddx's backward steps, through ddx#259's provenance) |
| **xarray-sql, demos**: recipes on public APIs | `modded-nanogpt-recipe`, `modded-nanogpt-runner` |
| **xarray-sql, evidence**: reports and benchmarks | `combined-mask-indexed`, `attention-kernel-headroom`, `attention-performance-evidence`, `indexed-gradient-evidence`, `modded-nanogpt-evidence`, `modded-step-profile`, `reduction-profiling`, `primitive-kernel-evidence`, `jax-mlx-training-control`, `native-loss-backward`, `nanogpt-integration` |
| **spikes 5 and 6's input**: where the last gap to PyTorch goes | `mps-compiler-comparison`, and the reports on `numerical-value-identities` |
| **already open as PRs** | `training-plan-reuse` (#301), `shared-kernel-execution` (#302), `compact-sql-execution` (#303), `nanogpt-performance-reports` (#304), `resident-sql-training` (#305) |

</details>

---

## 7. Work outstanding across XQL, beyond these changes

<details>
<summary><b>einfold</b></summary>

- Update einfold's design doc for D11: device execution for DataFusion,
  relational operators included, lives in DataFusion Accelerated. Its M9,
  "GPU adoption", points there for DataFusion, and keeps other engines' GPU
  hosts. This is Alex's decision, as einfold's maintainer.
- The B-0 interface: each detected node's IR, checks and subplan.
- Published, versioned node types, with a conformance suite.
- The fact table, brought forward from einfold's M5, with a user interface
  for facts such as `index_domains`' (§4.4).
- Near misses, with their reasons, by label.
- A report on whether a candidate save point splits a node einfold fuses.
- A builder method, `.with_einfold()`, beside `install`.
- The proofs from the fork and from xarray-sql, rewritten on the Rust IR,
  their tests and ablations first (§5.1, §5.3).

</details>

<details>
<summary><b>ddx</b></summary>

From design §10.2, in order:
1. ddx as a DataFusion extension, `.with_ddx()` and `Ddx()`, replacing
   `ddxdb.Context`;
2. `ddx_stop_gradient` on every engine and type (ddx#260);
3. volatility read from the engine;
4. what each check promises, and `to_sql` exported from ddxdb;
5. the step graph on `Program` (ddx#157);
6. labels and policies, ddx#251's fourth part;
7. engine limits read from the `Backend`;
8. provenance (ddx#259);
9. #256 and #257 (§3);
10. ddx#265 first among #261 to #265, and ddx#266, found by torch-sql's
    adversarial tests.

</details>

<details>
<summary><b>xarray-sql</b></summary>

- Merge #301 after its review's changes (§4.1). #287 is merged. #290
  waits on DataFusion's `WITH … AS MATERIALIZED`; until then, forward
  queries store shared views as tables (§4.2).
- Add installed XQL extensions to `Context`, so `grad` works in its SQL
  when ddxdb is installed.
- A table provider factory for Zarr, so `STORED AS ZARR ... OPTIONS
  ('accelerated.distribution' ...)` works (datafusion-accelerated#15).
- A release on DataFusion 55, shared with ddx and einfold (#310).
- Move `_SortedArrowTable` into registration, as a fact about row order.
- Demo 3 on public APIs; close #302, #303 and #305 with pointers once their
  pieces have landed.

</details>

<details>
<summary><b>torch-sql</b></summary>

- The adversarial tester's open findings on torch-sql #4 and #5: in-place
  ops that don't reach a tensor's aliases (so an EMA of the weights never
  updates), `relu` and `maximum` losing a NaN on DataFusion, `sin` and `cos`
  of infinity raising on DuckDB, a DataFusion optimizer bug behind
  `stack(...)[0]`, `int / int`, `sync()` with no arguments, `deepcopy`, and
  `Adam(amsgrad=True, foreach=True)`.
- Retire cut points through ddx's SQL path, and stop gradients with
  `ddx_stop_gradient` (design §6.4).
- Use the runtime as its ddx `Backend` at M0.

</details>

<details>
<summary><b>Upstream, each only with Alex's go-ahead</b></summary>

- **DataFusion**: `WITH … AS MATERIALIZED` and its follow-ups are already
  in progress upstream, and #290 waits on them. The bugs found so far (`MAX` skipping NaN with a
  `GROUP BY`; NaN ordering by total order; a filter pushed below an
  ungrouped aggregate; `amax` with a `dim` on infinities), partitioning
  over several axes (datafusion-accelerated#17), and later a
  `DISTRIBUTED BY` clause.
- **datafusion-distributed**: device-aware exchanges (#6), persistent
  stages and placement by affinity (#7), Python kernels on workers (#9),
  and a Python binding (#10).

</details>

---

## 8. Order of landing

1. **Now, independently:** ddx#256 and #257, after review; xarray-sql#301,
   after its review's changes; einfold's design update; the proofs' tests
   and ablations into einfold, as its specification (§5.3).
2. **Interfaces:** einfold's B-0 interface and node types; ddx as an
   extension; DataFusion Accelerated's M0, with the ddx integration.
3. **Execution moves:** compact arrays and device plans into the core;
   `DenseExecutor`'s MLX operators and the GPU branches of §6 into the MLX
   integration (M2 on Apple's GPUs); `TrainingSession` into `Program` (M1);
   the proofs rewritten on einfold's Rust IR. xarray-sql#290 once
   DataFusion's materialized CTEs land.
4. **Retire:** `SharedExecutor`, `DenseExecutor` and `TrainingSession` from
   xarray-sql, and `DenseProgram` from einfold's fork;
   `accelerators/einfold_mlx/build.py` with them. Demo 3 runs on the public
   APIs.

Each step is measured against the numbers it replaces, on the same
machine, and reported on xarray-sql#285, so nothing that made the work fast
is lost in the move. The design's milestones say which numbers: M2 is no
slower than `DenseExecutor`, and M6 is no slower than the 238.95 ms step
(design §12.2).

---

## 9. Questions for the contributor

1. Is §6's classification right? In particular, do
   `resident-attention-adjoints` and `numerical-value-identities` belong in
   einfold or in the runtime?
2. Which proofs closed the most of the gap, so they're rewritten first?
3. What did `DenseExecutor` decline in practice, and how often? The design
   assumes per-node fallback costs little, which only those numbers can
   confirm.
4. Where does the memory go? The Nov 2 run peaked at 24.67 GB of process
   memory against PyTorch's 2.44 GB, and the Nov 7 run at 20.34 GB. Spike 5
   will look, and you may already know.
5. Would you like to own the MLX integration in DataFusion Accelerated? It
   is your code, and the first device integration to exist.
