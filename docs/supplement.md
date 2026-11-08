# The XQL runtime: supplement

_companion to_: [the design](design.md)
_last updated_: 2026-11-07

This supplement holds the evidence behind the [design](design.md), the
alternatives it rejected, and the reasoning behind its rules. The design
says what to build. This says why. Read it with the design open: it uses
the terms the design's §2 introduces, and it introduces only the terms
the design doesn't need.

---

## S1. Evidence

### S1.1 xarray-sql's nanoGPT experiments

A contributor's experiments on an Apple M5 Ultra built a sequence of
executors, each on the one before. They are in draft PRs
([xarray-sql#302](https://github.com/xqlsystems/xarray-sql/pull/302),
[#303](https://github.com/xqlsystems/xarray-sql/pull/303) and
[#305](https://github.com/xqlsystems/xarray-sql/pull/305), with
[einfold#156](https://github.com/xqlsystems/einfold/pull/156) and
[einfold#157](https://github.com/xqlsystems/einfold/pull/157)), which
reviewers asked to be rehomed
([xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307)).
Every result was checked against a reference outside the timer: JAX in
float64, or nanoGPT's own model and PyTorch's AdamW optimizer.

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
- Nobody has measured where the remaining 3.76× goes (the design's spike
  5).

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

That is the design's design B growing on its own, one proof at a time, but
in a frontend instead of in einfold. Each proof that lands in the wrong
layer makes the right layer harder to reach.

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
   handling) in 1,564 lines of Python beside einfold's Rust (S2.1).
5. **The pieces coupled through private names.** xarray-sql's executors
   import 18 private names from `einfold.dense_program`, einfold's private
   `einfold._native`, and ddx's private `ddxdb._ddxdb.to_sql`.

### S1.2 xarray-sql's Samudra demo

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
PyTorch on the CPU. That is lesson 1 of S1.1 again: the remaining gap is
the long-format representation, not planning. The machine was under load
from a compile job, so these numbers are rough.

The demo keeps the views as its main path because ddx differentiates
through views, but not through tables the frontend saved. A saved table is
a constant to ddx, so the gradient would stop at it. That is why the
design's R7 keeps where a value is held out of the program's meaning.

### S1.3 torch-sql

torch-sql runs a training step as a few large programs. Each finding below
was measured there or in its spikes:

- **DataFusion plans a CTE again at every read**
  ([xarray-sql#267](https://github.com/xqlsystems/xarray-sql/issues/267)).
  For a stack of ConvNeXt-like blocks as one query, the plan grows from 441
  operators for one block to 136,107 for three, and planning and running
  from 0.2 s to 212 s. DuckDB grows by 75 operators per block. torch-sql
  hands ddx each loss as one Substrait plan, where every shared subplan is
  already written out, so it chooses *cut points*: values it saves as
  tables between queries. A value is a cut point when the query that
  computes it, with every value it reads written in, passes a per-engine
  budget (400 `SELECT` blocks on DataFusion, none on DuckDB). The input of
  every norm is a cut point whatever its size, because a norm reads it
  three times. Cut points brought one nanoGPT layer's backward pass on
  DataFusion from 9 s to 0.7 s, and two layers, which didn't fit in 8 GB
  before, run in 1.0 s.
- **Differentiating across cut points is a chain of ddx programs.** Each
  segment between cut points is one program. Each cut point passes a
  *cotangent* (the gradient of the loss with respect to that table) to the
  segment before it. torch-sql runs that chain rule itself (S3.6).
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
  there dropped tables still in use.
- **Transport** (torch-sql's spike S4): an Arrow array with no nulls reaches
  PyTorch through DLPack without a copy. A fixed-shape tensor column keeps
  its type through DataFusion's projections, filters and sorts, and a
  projection keeps the same buffer. DuckDB turns it into a plain list of
  fixed size, which keeps the values but loses the shape.

### S1.4 ddx

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
  case (S1.3).
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
  query's plan (`plan_key`), and report a session key. The Rust trait
  `ddx_ad::Backend` has the same shape.

---

## S2. Dense execution: the alternatives

The design adopts design B: einfold marks the array work in a step's SQL,
and the runtime runs what it marks. This section records the two designs
it rejected, and why B takes the form it does.

### S2.1 Design A: the runtime interprets relational plans

This is `DenseProgram` as it is today, moved into the runtime. The runtime
would walk DataFusion's logical operators and implement each on compact
arrays: joins as gathers, `GROUP BY` as reductions, `UNION` with bag
semantics, NULLs as masks. It works, and it is measured (S1.1). But it is a
second implementation of SQL's semantics, beside the engines'. Every
operator, NULL rule and edge case would have to agree with every engine,
forever, and S1.3 shows how hard that is even between two mature engines.

### S2.2 Why B has no array language of its own

An earlier draft of B had einfold translate regions into a small array
language (elementwise expressions, `einsum`, reductions, gathers and
scatters, `where`). Alex decided that StableHLO, the operation set JAX and
XLA compile, is not XQL's representation, and einfold's review went
further: no separate array language at all, whether einfold's own, the
Python array API standard or StableHLO. A second program would have to be
kept equal to the SQL forever, which is design A's risk. It would also
lose two things:

- **Existence isn't a mask on values.** A compact array has three states
  per cell: absent (no row), NULL, or a value. SQL's rules for them don't
  reduce to `where`:
  - a group exists only if some row reached it;
  - `SUM` over no rows is no row, not `0`;
  - a left join from a domain turns absence into NULL, or into a default
    such as ddx's `0.0`;
  - the sign of a zero sum follows the engine: DataFusion's `SUM` starts
    from `+0.0`.

  einfold's kernels already carry the three states as per-row flags, and
  its property tests check each rule, bit for bit, against DataFusion. An
  array language with masks would make every backend reimplement them,
  which is design A's second copy of SQL, one layer down.
- **Primitive operations lose the structure backends need for speed.**
  Attention lowered to `einsum`, `exp` and `einsum` holds all `N²` scores,
  which demo 1's fused kernel exists to avoid: it runs 65,536 tokens in one
  tile per thread (einfold's design §10.2). A convolution lowered to
  gathers becomes an index table and a gather, where a shifted fold is one
  matrix product per filter position
  ([einfold#155](https://github.com/xqlsystems/einfold/pull/155)).

The cost of B as adopted is that SQL is portable in meaning, not in speed.
Everything fast depends on einfold recognizing structure, and whatever it
doesn't recognize runs at the engine's speed.

### S2.3 Design C: the engine carries dense values

If intermediates could travel between DataFusion's own operators as Arrow
fixed-shape tensor columns, in GPU memory through the C Device Data
Interface, the engine would keep SQL's semantics, and the runtime would
provide only kernels. torch-sql's spike S4 found that DataFusion keeps such
a column through projections, filters and sorts without copying it.

But a tensor-valued column changes what a table is: a row holds a tile of
an array rather than one element. A matrix product is then a function on
tiles, not a join and a `GROUP BY`. That moves XQL from long format to
tiled tables, which changes every frontend's schema, and ddx's rules too.
It may be the right long-term representation, and it would make
distribution by tiles natural. But it is a change to the whole stack, not
to the runtime.

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
dimensions, and may be where C starts. The design's spike 1 tests it.

### S2.4 What torch-sql's lowerings ask of B

Three kinds of step in torch-sql's nanoGPT need care:
- **Gathers whose index is a value.** An embedding is a join on a token
  id. If an id falls outside the table, SQL's inner join drops the row,
  where a gather would fail or read the wrong memory. So a gather needs a
  fact that every id lies in range, or a data check on every run.
- **Floating-point edge cases.** torch-sql writes PyTorch's behaviour out
  in SQL (NaN checks in comparisons and in `MAX`, guards around `ln` and
  `sqrt`), because the engines differ (S1.3). Those guards are `CASE`
  expressions that einfold's elementwise nodes must carry exactly.
- **Shape-changing steps.** torch-sql writes reshapes, slices and
  convolutions as joins with *index tables*, small tables that map each
  output position to an input position. Reshapes and slices become gathers
  on those tables, and convolutions shifted folds.

### S2.5 Why B-0 keeps values resident only between steps

In B-0, a node's neighbours inside a step are engine operators, which read
rows. So a node's output stays a compact array only when it is the step's
result, and a node reads a compact array only when its input is a scan of
a runtime table. Between those, values become rows. B-0's gain therefore
depends on how many steps are one node from their scans to their result,
as well as on how much time is spent inside nodes. That is why the
design's runtime tables can be read two ways, and why spike 6 measures
both numbers.

### S2.6 Why B is staged

einfold today is a DataFusion optimizer rule, called once per query. ddx
saves each step as a table, so einfold sees one step at a time. Seeing a
whole program is einfold's *program mode*, planned for its M7, after DuckDB
(its M4), facts (its M5) and sparse kernels (its M6). Regions across steps
would make the runtime's GPU milestone wait on all of those. B-0 needs no
new detection, only a new way to hand over the node einfold already finds.

---

## S3. Reasoning behind the execution rules

### S3.1 Checks

ddx has three checks, and all are of one kind: some key columns of one
table are unique. None reads a value column. That is why ddx states each
check's promise rather than a label, and the runtime decides when to run
it. In a training step only the weights are differentiated, so a compiled
step runs none of them.

einfold's checks read values, and facts about values hold for one run
only. [einfold#39](https://github.com/xqlsystems/einfold/issues/39) found
that xarray-sql's values change between queries without the table being
registered again, because its `Context.prepare` writes into the arrays. The
stencil kernel of einfold#155 needs every weight finite before it accepts
holes in the image, and a chain of folds needs einfold's float guard. So a
compiled program nearly always keeps some data checks.

On a device, a check is a reduction followed by a copy to the host, which a
compiled step can't branch on while it runs. `TrainingSession` therefore
declined to compile any step whose checks read device data. Its resident
step read back one validation flag (S1.1), which is where the design's
single flag per run comes from.

### S3.2 Parameters as a one-row table

Substrait has a dynamic parameter expression, but neither the engines'
consumers nor ddx have been tested with one: ddx's traversal admits one,
and no ddx test runs one. A one-row table works everywhere. ddx treats it
as a constant, since it isn't differentiated, and a compiled program reads
new values written in place. torch-sql already reads its non-finite
constants from such a table. ddx#264 and #265 are bugs in ddx's handling
of one-row tables on DuckDB, not limits of the approach.

### S3.3 Stable names and tables kept in place

ddx names every table it writes under a fresh prefix, or under a namespace
its caller passes (`Options::namespace`). With a namespace, the same plan
gives the same program byte for byte, so compiled programs can be found
again.

A DataFusion physical plan keeps the tables it read when it was made.
ddx's two paths handle new data differently:
- **Rust.** ddx-datafusion's compiled plans swap in fresh scans on each run
  (`LiveExec`), so a table registered again is read correctly.
- **Python.** ddxdb can't swap scans, so `in_place=True` is a promise from
  the caller: a table changes only by writes into the memory it already
  holds, and is never registered again (ddx#214).

The design publishes new state by swapping in a second copy (R5). In
Python, that breaks the promise and forces every plan that reads the state
to be planned again. Hence the design's rule that an engine table holding
state reads the handle's current buffer when its plan runs.

### S3.4 Step boundaries and shared subplans

ddx already draws its programs' step boundaries, for four reasons (S1.4).
A second chooser with its own budget would undo the first. ddx doesn't ask
its `Backend` where to save, because `prepare` caches a program by its plan
and the session key, and an answer that depended on the runtime's memory at
that moment would leave a cached program stale.

Shared subplans can't be found in a Substrait plan, because neither
engine's producer emits a `ReferenceRel` and the copies are already
written out. `SharedExecutor` parsed SQL to find CTEs, and had to turn
ddx's plans back into SQL through ddx's private `to_sql` to do it. ddx#168
already finds them in SQL, before planning. All of this should shrink once
DataFusion computes a CTE read several times only once (xarray-sql#267, on
DataFusion's roadmap for late 2026).

### S3.5 Saving with einfold

Saving a value can destroy structure einfold fuses. If attention's scores
are saved, no pipeline is left to fuse, and attention's fused backward
pass ([einfold#135](https://github.com/xqlsystems/einfold/issues/135))
needs ddx to recompute the scores in each backward step instead (ddx#251).
The reverse also happens: inlined views make einfold's matching worse.
Demo 4 as views is 3,862 operators, with each block's input recomputed, and
the GPT's loss planned 2.7 times longer with einfold on, through 9,882
lines of inlined plan ([einfold#81](https://github.com/xqlsystems/einfold/issues/81)),
before einfold#86 to #92 fixed most of that.

### S3.6 torch-sql's cut points

torch-sql chooses cut points and runs a chain rule across them only
because it hands ddx each loss as one Substrait plan (S1.3). That works,
and matches PyTorch to 1e-11, but it is a second tape around ddx's, in a
frontend. ddx's SQL path differentiates through a CTE read more than once
(S1.4), and torch-sql's graph already knows which nodes are shared, so
torch-sql can hand ddx SQL with a CTE at each of them instead. Handing ddx
its segments as one program is the other way out.

torch-sql saves a detached value as a table today so that ddx treats it as
a constant. By R7, holding a value somewhere can't change a gradient, so
torch-sql must use `ddx_stop_gradient` instead, which works only on
DataFusion and only for `DOUBLE` today
([ddx#260](https://github.com/xqlsystems/ddx/issues/260)).

`torch.utils.checkpoint` has no numerical meaning. It trades memory for
recomputation, and preserves the random state so that a recomputed dropout
draws the same mask. A recompute policy on a named region expresses the
first, and a counter-based random function, or a mask uploaded as an
input, gives the second for free.

### S3.7 Recomputing and determinism

Alex decided on 2026-11-01, for ddx#251, that recomputing is an explicit
policy people opt into, as with `jax.checkpoint` (JAX's marker for a
computation to recompute in the backward pass), and never the engine's own
choice. Where a value is held changes no work. Recomputing does, on plain
engines especially.

Recomputing gives the same result only for a plan that gives the same rows
each time it runs, and then only within the floating-point bound, since a
`SUM` may add in a different order. A volatile function, a `LIMIT` without
a total order, or a rank over ties rules it out. ddx already refuses to
compute such a plan twice
([ddx#254](https://github.com/xqlsystems/ddx/pull/254)), but it tests for
volatility with a list of function names.

On reproducibility: einfold never adds nondeterminism, and its kernels give
the same bits for any thread count (einfold's design §8.6). GPU kernels
that add with atomic operations don't. Neither do ddx's gradients on
DataFusion: the last bits of a `SUM` follow the order partitions arrive,
and ddx's `Functions::rounds` says which functions can round.

### S3.8 The floating-point bound

A fixed tolerance isn't enough. A float sum of `n` terms, added in any
order, is within `(n − 1)·u·Σ|terms|` of the exact sum, as long as nothing
overflows (einfold's design §8.6). `u` is the *unit roundoff* of the
precision the backend computes in: 2⁻⁵³ for float64, 2⁻²⁴ for float32.
Computing in float32 on a GPU is then an explicit choice of `u`, as
einfold#156 already makes precision the host's choice. The nanoGPT work
already ran in float32 on the GPU against float64 SQL, with its tolerance
stated per backend.

### S3.9 Two copies of state

The old state must survive until the new state's checks pass, so state
needs room for two copies while a step runs: weights, optimizer moments
and their replacements. For a Samudra on a quarter-degree grid, with four
times the cells of today's, that is worth budgeting on one B200.

JAX's *buffer donation* lets a compiled function write its result into an
input's memory. The same works table by table here, but keeps R5 only if
no check can fail after the first table is overwritten. In a training
step, the checks are on the batch and the gradients, which come before the
optimizer's update, which is where the design reads its flag.

### S3.10 Random numbers

Dropout needs random numbers that every implementation of a step agrees
on, the engine included, or checking mode can't compare them. The engines
make portable SQL hard: on integer overflow, DuckDB raises an error and
DataFusion 55 wraps around silently, so the hash must be written in integer
arithmetic that never overflows. `TrainingSession`'s GPU dropout used MLX's
own random keys, so its masks differed from the CPU path's, and checking
it against PyTorch meant exporting the masks.

torch-sql checks nanoGPT against PyTorch step by step, so it draws each
dropout mask with PyTorch's own generator on the CPU and uploads it as an
input table. Its masks then match PyTorch's bit for bit, at the cost of one
upload per mask. That is why the design accepts masks as inputs as well as
providing its own function.

---

## S4. Choices considered

### S4.1 Rust or Python first

*Settled (design D6): the runtime straddles both, as DataFusion does. The
arguments below shaped that answer, and the costs of crossing between the
two remain (design spike 4).*

The first draft leaned to Rust first: DataFusion, ddx's `Backend` trait
and einfold's dense backend are Rust, and concurrent steps need to run
outside Python's lock. The second leaned to Python first: the interfaces
are still moving and are cheaper to change in Python, the costly work
(engine planning, kernels, array libraries) is native already, and
`ddxdb.Backend`, xarray-sql and torch-sql are Python.

torch-sql's prototype settled the co-authors on Python first. It is all
Python, and Python was never its bottleneck: its slow parts are the number
of queries and planning, both native. Its internals changed in every round
of review, which Python made cheap. Its one bug of the kind Rust prevents,
dropping a table still in use, came from tying a table's lifetime to
Python's garbage collector, which explicit handles avoid in either
language.

Three costs on the other side are known:
- einfold reaches datafusion-python only through datafusion-ffi. Through
  it, steps einfold doesn't match run 1.25 times slower and plan 2.9 times
  slower than on DataFusion alone
  ([einfold#58](https://github.com/xqlsystems/einfold/issues/58),
  einfold#81). Object stores don't cross it (einfold#149), and functions
  planned through it lose the session's configuration (einfold#137). A Rust
  core would link DataFusion and einfold directly.
- ddxdb's compiled path from Python (`in_place`) needs an xqlsystems fork
  of datafusion-python today, which the design's R2 rules out until the
  change is upstream.
- Atomic state in Python needs an engine table that reads a handle's
  current buffer (S3.3).

ddx's answer on timing: Python is enough until concurrent steps, if the
runtime implements `compile` and keeps tables in place. ddx's run loop is
sequential in both languages, so concurrency needs ddx#157 either way.

### S4.2 The array library for NVIDIA GPUs

Lowering a node is private to each backend, so more than one backend can
exist. The candidates:
- JAX, whose compiler XLA fuses whole programs. By default JAX takes 75
  percent of a GPU's memory when it starts, so beside PyTorch on one GPU,
  as torch-sql is as a co-processor, the two allocators compete for the
  rest.
- PyTorch, where torch-sql already lives. A node's results reach PyTorch
  with no transport, and `torch.compile` can fuse them.
- CuPy, a thin NumPy-like library for CUDA.

### S4.3 The name

`xql-runtime` says what the project is in XQL's vocabulary, follows the
plain naming of xarray-sql and torch-sql, and avoids the word *tensor* for
N-dimensional arrays, as the rest of XQL does. The alternatives:
- **`sql-host`.** einfold's design already uses *host* for whatever runs a
  plan, usually the engine, so the name would blur the runtime with the
  engines.
- **`array-runtime`.** Generic: it could be any array library's runtime,
  and nothing ties it to XQL.
- **A coined name, in the spirit of ddx and einfold**, such as `einrun` or
  `foldhost`. These read as einfold's own executor, which design B
  rejects.

### S4.4 The reference engine

When two engines disagree (S1.3), something has to be the reference. The
design makes it the engine a program runs on, and finds disagreements in
tests, as torch-sql's suite does by running every op on both engines. The
alternative, a single reference engine for all programs, would make every
run on the other engine a comparison, and the reference would still have
bugs of its own, such as DataFusion's `MAX` skipping NaN.

---

## S5. Control and composition

The design has three aims beyond running programs fast:
- the model should match how systems programmers already think about
  memory;
- developers and agents should be able to reach every low-level choice,
  rather than fight a compiler that makes the choices for them;
- the runtime should be one component of a composable data system.

This section records why the design takes the form it does for each.

### S5.1 Algorithm and schedule

Halide, a language for image-processing pipelines, separates a program's
*algorithm*, what it computes, from its *schedule*, how it is computed:
the order of loops, what is stored and where, and what runs in parallel.
A developer can change the schedule freely, because no schedule changes the
result. The design takes the same split. The SQL is the algorithm, and the
schedule (design §5.5) holds every physical choice.

The split matters because both of the traditions XQL draws on hid the
schedule:
- **Databases.** PostgreSQL has no query hints in its core, only through
  an extension, and systems that do have hints often ignore an invalid one
  without saying so. A developer who knows a better plan has to coax the
  optimizer into finding it.
- **GPU compilers.** Triton chooses layouts, pipelining and other
  low-level details by heuristics, which made the last part of the
  performance hard to reach. Newer tools hand those choices back: Triton's
  own Gluon dialect exposes layouts the compiler used to choose, and
  CUTLASS's CuTe DSL lets Python code state them directly.

Agents change the trade-off further. An agent improves a program by a
loop: change something, run, read the cost, repeat. That needs three
things, which design R10, §5.5 and §6.6 provide: a schedule the agent can
edit, a report it can parse, and a check that its change is still correct.
XQL has an unusually strong version of the third, since the SQL computes
the right answer on any engine, and checking mode compares against it.

### S5.2 Why control is safe here

Codd's relational model rests on *data independence*: how data is stored
can change without changing what a query means. Design R7 is that
principle applied to the runtime. Where a value lives, how it is laid out,
which kernel computes it and what is recomputed never change a result
beyond the stated bound. So handing those choices to a developer can make
a program slower, but never wrong, and checking mode catches the
exceptions. The relational model is what makes low-level control safe to
give, not what stands in its way.

### S5.3 Proof against test, for bindings

einfold's nodes are exact by proof: einfold shows from the plan's
structure that the array algorithm computes the SQL, and checks at run
time the facts the proof needs. A developer's kernel has no such proof. It
is tested against its SQL, on the runs that checking mode samples, which
finds most errors but can miss one that appears only on rare inputs. The
design therefore treats hand-written bindings as a separate class: counted
apart, sampled more often, and refusable in strict mode (design §5.6). The
two classes share one mechanism, so a binding can move from hand-written
to detected when einfold learns its pattern.

### S5.4 A deconstructed tensor runtime

Wes McKinney's idea of the *deconstructed database* is that a database can
be assembled from independent parts, joined by standard formats: Arrow for
data in memory, Substrait for plans, and engines such as DataFusion and
Velox as libraries rather than products. The *Composable Data Management
System Manifesto* (Pedreira et al., VLDB 2023) states the case at length.
XQL is built that way, and the runtime is meant to be the same thing for
array programs: a *deconstructed tensor runtime*.

That means the runtime builds only what no other component does:

| Concern | Who provides it |
|---|---|
| Planning and running SQL | DataFusion, DuckDB, GPU engines |
| Differentiation | ddx |
| Recognizing array work in SQL | einfold |
| Kernels | Triton, CUTLASS, XLA, `torch.compile`, MLX, or a developer's code |
| GPU memory | RMM, or the array library's allocator |
| Moving data between GPUs | NCCL |
| Tables, arrays and plans across each seam | Arrow, DLPack, Substrait |
| Lifetimes, placement across tiers, the schedule, bindings, checks and reports | the runtime |

The risk of a bridge layer is that it grows into everything it connects.
Design R11 is the test against that: if a feature could be a library of
its own, used without the runtime, it should be one. Borrowing the
allocator has a direct payoff too. JAX takes most of a GPU's memory when
it starts (S4.2), and two libraries that each own their allocator compete
for one GPU. A runtime that draws from a shared allocator doesn't.

### S5.5 StableHLO and MLIR

StableHLO and MLIR are the compiler ecosystem's standard formats, and a
composable runtime should meet them. It does so at two seams, without
making either XQL's representation of a program (design D2):
- **Inside a backend.** A JAX backend lowers einfold's nodes to StableHLO
  for XLA to compile. That stays private to the backend.
- **At einfold's node types**, possibly. If einfold published its nodes as
  an MLIR dialect, MLIR-based compilers could consume them directly. That
  is einfold's decision (design §13, question 8).

Building the runtime itself on MLIR would make it a compiler, which is a
non-goal, and would tie the first, Python-only releases to a large C++
framework.

### S5.6 Values and buffers

The memory hierarchy and the relational model disagree on one point. In
SQL, a table is a value: it has no identity, nothing aliases it, and
nothing changes it in place. In systems programming, a buffer is mutable
and can be shared. The design keeps the two apart by allowing in-place
writes in one place only, the publication of state (design R5 and §7),
where buffer donation and ddxdb's `in_place` mode both live. Everywhere
else, a new value is a new table.

### S5.7 Who the memory model is for

A memory hierarchy with tiers, moves and pins is the right model for
someone tuning a training step, and the wrong one for a scientist querying
a Dataset through xarray-sql. So every control has a default, and a
program that names no tier, binding or save point runs well without them.
The controls are there to be reached for, not required.

---

## S6. JAX, PyTorch and DataFusion

The design borrows its interfaces from three communities, so that each
finds the runtime familiar: JAX and PyTorch for programs, memory and
kernels, and DataFusion for everything about sessions and plans. This
section maps the design onto each, and records how kernel languages fit.

### S6.1 Concepts across the three

The last column is what the design proposes, not what exists.

| Concept | JAX | PyTorch | DataFusion today | xql-runtime (proposed) |
|---|---|---|---|---|
| Attaching to a system | PJRT plugins | device backends (PrivateUse1) | builder methods; `with_extensions` in Python | `.with_xql_runtime(...)`; `XqlRuntime(...)` for `with_extensions` |
| Settings | `jax.config` | `torch.backends`, `torch._dynamo.config` | `SessionConfig`, `SET` | `SET xql_runtime.*` |
| Memory kinds | `device`, `pinned_host`, `unpinned_host` | `.to(device)`, `pin_memory()` | `MemoryPool`, `DiskManager` | tiers named by Arrow device types: `CUDA`, `CUDA_HOST`, `CPU`, `storage` |
| Memory accounting | XLA's allocator | caching allocator | `MemoryReservation` from a `MemoryPool` | device reservations from a device pool in `RuntimeEnv` |
| Allocator | XLA's own (takes 75% at start) | pluggable; `rmm.allocators.torch` | the system allocator | RMM, or the array library's |
| Copies between tiers | `jax.device_put` | `.to()`, non-blocking copies | none | `CopyExec` nodes; `prefetch`, `evict`, `pin` |
| Where work runs | XLA's compiler | the tensor's device | the physical plan | the device plan: `KernelExec` nodes |
| A program | `jax.jit(f)` | `torch.compile(f)` | none (one query at a time) | `Program(ctx, sql, state=..., outputs=...)` |
| Lower and compile | `.lower(...)`, `.compile()` | Dynamo, then Inductor | `create_physical_plan` | `.lower()`, `.compile()` |
| Inspecting it | `Lowered.as_text()`, `Compiled.cost_analysis()`, `.memory_analysis()` | `torch._dynamo.explain`, profiler | `EXPLAIN`, `EXPLAIN ANALYZE` | `explain()`; `EXPLAIN ANALYZE` per step; `report()` per program |
| Editing it | none | Inductor's config | optimizer rules | `compile(schedule=...)` |
| Donating buffers | `donate_argnums` | in-place ops | none | state buffers written in place after the flag (design §7) |
| Asynchronous runs | async dispatch, `block_until_ready()` | streams and events | `execute_stream` | `run()` returns a future |
| Names in a program | `checkpoint_name(x, "name")` | NVTX ranges, module paths | none | `ddx_name(expr, 'label')` |
| Recompute policies | `save_only_these_names`, `save_and_offload_only_these_names` | selective activation checkpointing | none | the same names, passed to ddx by label |
| Custom kernels | `pallas_call`, `jax.ffi.ffi_call` | `torch.library.custom_op`, `register_kernel`, `triton_op` | a custom `ExecutionPlan` | `@runtime.kernel("label")`; the `Kernel` trait in Rust, with cuTile Rust kernels |
| A kernel's gradient | `custom_vjp` | `register_autograd` | none | none needed: ddx writes the backward SQL, and its steps are bound by label and role (ddx#259) |
| Failing loudly | none | `fullgraph=True` | none (Polars has `GPUEngine(raise_on_fail=True)`) | `SET xql_runtime.strict = true` |
| Distribution | sharding, IFRT | DTensor, FSDP | datafusion-distributed, Ballista | datafusion-distributed, with device nodes |
| Serializing a plan | StableHLO | `torch.export` | datafusion-proto | datafusion-proto, with the runtime's codec |

Two rows have no counterpart elsewhere, and are why XQL is worth building
this way. A kernel needs no gradient of its own, because ddx writes the
backward pass as SQL, whose steps can be bound to kernels in turn. And a
kernel's specification is executable: its SQL runs on any engine, which is
what checking mode compares against.

### S6.2 Kernel languages

The design's kernel contract (design §5.7) comes from how kernels are
written today. All of them work on tiles of dense arrays, and none of them
knows about rows, so the runtime's job is to hand them dense arrays with
named axes and masks, and to stay out of the kernel.

**Triton.** A kernel is a function decorated with `@triton.jit`, launched
as `kernel[grid](args...)`, where each of the grid's programs loads a tile
with `tl.load(ptr + offsets, mask=...)`, computes, and stores. The mask is
how Triton handles a tile that runs past an array's edge, and it is the
same idea as the runtime's presence mask, so a kernel that accepts masks
can take absent cells the same way. Kernels take pointers, which Triton
reads from any object with a `data_ptr()`, so the runtime's device arrays
provide one beside DLPack. `@triton.autotune` chooses block sizes and warp
counts per shape. A compiled program's shapes are fixed, so tuning runs
once, at compile, and its choice is recorded in the schedule.

**Gluon.** Gluon kernels are launched exactly as Triton's are, which is
what makes them easy to bind. What changes is inside the kernel: the
author chooses each tile's layout across threads (`gl.BlockedLayout`),
allocates shared memory, synchronizes with barriers (`mbarrier`), and
splits warps into specialized roles. Triton's own tutorial says why it
exists: when Triton's compiler is beaten by hand-tuned code, "there is
little the user can do", and Gluon exposes the details. This is the level
the runtime must never reach into. The runtime's tiers stop at device
memory; shared memory, tensor memory and registers are the kernel's.
Gluon's tutorials include a convolution through im2col and gathers and
scatters through the tensor memory accelerator, which are the shapes of
einfold's shifted folds and gathers.

**Pallas.** `pl.pallas_call(kernel, grid=..., in_specs=[pl.BlockSpec(block_shape, index_map)], out_shape=...)`
runs a kernel over a grid, and each `BlockSpec`'s `index_map` says which
block of an input each grid point reads. That is the closest of the three
to einfold's nodes, which state their named axes and extents: a binding can
build its `BlockSpec`s from the node's axes. Pallas on NVIDIA GPUs compiles
through Triton or through Mosaic GPU, whose memory spaces (global, shared,
and on Blackwell tensor memory) are, again, below the runtime's tiers.
Pallas kernels are functions that return new arrays, so the runtime adopts
the result's buffer through DLPack when JAX draws from the shared
allocator, and copies it otherwise (design §5.7).

**cuTile Rust.** A kernel is a Rust function in a `#[cutile::module]`,
marked `#[cutile::entry]`, whose parameters are tiles of tensors:
`&mut Tensor` for an output, which the host splits into disjoint pieces
with `.partition([...])`, and `&Tensor` for shared, read-only inputs. The
output's partition sets the launch grid, as Pallas's grid and `BlockSpec`s
do. Kernels are JIT-compiled through CUDA Tile IR, and a launch is a
future: `.await` it, `.sync_on(&stream)`, or capture it in a CUDA graph.

It is the most natural of these to bind, because its host side was built
to run on memory and streams someone else owns:
- `unsafe Tensor::from_foreign(owner, shape, strides)` borrows device
  memory owned by another framework, keeping its owner alive by reference
  count. The runtime's table handle is that owner.
- `Context::borrow_with_owner` and `Stream::borrow_with_owner` borrow
  another framework's CUDA context and stream, so cuTile runs on the
  runtime's streams rather than its own.
- `.schedule(&policy)` takes a `SchedulingPolicy`, which the runtime can
  implement so that cuTile places work as DataFusion's partitions say.
- Launches are futures on Tokio, the async runtime DataFusion executes on,
  so a `KernelExec` awaits a kernel without blocking a thread.
- `.compile()` with `api::meta` placeholder tensors compiles a kernel for a
  shape without running it, which is what a program's `compile` step
  needs.
- Its ownership types, exclusive outputs and shared inputs, keep a kernel
  from aliasing its output with an input. That is the kernel-level form of
  the design's rule that tables are values (S5.6).
- `cutile-kernels` has attention, embeddings, norms and pointwise kernels
  to start a CUDA backend from.

The frictions are small. cuTile's element types and some shapes are
static, where Arrow's are known only at run time, so a binding dispatches
on type. Its shapes are positional, so the runtime orders a node's named
axes as the node declares them. And it is a research release, requiring
CUDA 13.2 or later and `sm_80` or newer, with API changes expected.

Its peer-memory operations, which read and write another GPU's memory
over NVLink, may also matter later: within one NVL72 rack they are a way
to exchange the edges of grid tiles without going through NCCL.

**Other Rust routes.** NVIDIA's `cuda-rust` crates, which cuTile is built
on, and cudarc both load and launch compiled kernels: PTX or cubins from
CUDA C or CUTLASS, or a Triton kernel compiled ahead of time. CubeCL
writes kernels in Rust for several kinds of GPU. Each implements the
runtime's `Kernel` trait, so a Rust program can bind kernels with no Python
at all.

**How kernels fit the session and distributed models.** A kernel is
registered on the runtime's extension before the session is built, in
Python on the `XqlRuntime` object and in Rust on the builder. That follows
`with_extensions`'s rule that extension objects are configuration only,
and it is what datafusion-distributed needs: a `KernelExec` crosses the
network as its kernel's name and tuning parameters, and each worker,
built with the same extension, finds the kernel by name. The open case is a
kernel written in Python on a worker that is a Rust server (design §13,
question 11).

**Which kernels fit which nodes.**

| einfold node | Kernels that implement it |
|---|---|
| fold | matrix products: CUTLASS, Triton's matmul, Gluon on Hopper and Blackwell tensor cores |
| pipeline | fused attention: Triton's and Pallas's flash attention, `cutile-kernels`' attention |
| shifted fold | convolutions: CUTLASS, cuDNN, Gluon's im2col convolution |
| gather on a value index | embedding lookups: Gluon's gathers and scatters, Triton, `cutile-kernels`' embeddings |
| elementwise | `torch.compile`, Triton, `cutile-kernels`' pointwise kernels |

### S6.3 Where each concept sits in DataFusion

| Runtime concept | DataFusion layer | The runtime's piece |
|---|---|---|
| Attaching | `SessionStateBuilder`; `with_extensions` | `.with_xql_runtime`; `XqlRuntime` |
| Settings | `SessionConfig` namespaces | `xql_runtime.*` |
| Memory tiers | `RuntimeEnv`: `MemoryPool`, `DiskManager` | device pools, beside DataFusion's |
| Tables | table providers | runtime tables, readable as rows or compact arrays |
| Device placement | `PhysicalOptimizerRule`s | the runtime's rule, after einfold's |
| Kernels | `ExecutionPlan` | `KernelExec` |
| Copies | `ExecutionPlan` | `CopyExec` |
| Parallelism | partitions | streams, and devices |
| Costs and declines | metrics, `EXPLAIN ANALYZE` | the runtime's metrics |
| Serialization | datafusion-proto, `PhysicalExtensionCodec` | the runtime's codec |
| Distribution | datafusion-distributed | composition, not a planner |
| Programs and state | none | `Program`, `lower`, `compile`, `run` |

The last row is the runtime's own layer, and the only one.

### S6.4 datafusion-distributed as the precedent

datafusion-distributed states its tenets plainly: be "as close as possible
to vanilla DataFusion", be "a library, not an engine", and stay
unopinionated about networking. A distributed plan "is a normal DataFusion
physical plan, with a few extra nodes that stream Arrow data between
machines". A custom node is distributed by giving it a codec and
registering the codec on both the coordinating session and the workers.

Each of those carries over. The runtime is close to vanilla DataFusion
(R11), a library rather than an engine (non-goals), unopinionated about
kernels the way datafusion-distributed is about networking (R10, §5.7), and
its device plans are physical plans with a few extra nodes (§5.3). Because
both are built the same way, they compose: one plan can hold network nodes
and device nodes, and the runtime never needs a distributed planner.

### S6.5 Device memory in a host-memory engine

arrow-rs keeps its buffers in host memory, so a DataFusion record batch
can't hold a GPU buffer. There are three ways around it:
- **Keep device work inside single nodes.** Each `KernelExec` copies in,
  computes and copies out. Simple, but every node boundary is a round trip
  to the host, which is the cost S1.1 measured.
- **Handle batches.** Device nodes pass record batches whose one column
  holds handles to device buffers in the runtime's pool, and only device
  nodes read them. The data stays on the device between nodes, at the cost
  of a rule that no DataFusion operator ever receives one. The design takes
  this, and spike 7 tests it through DataFusion's own rewrites, such as
  repartitioning, that move batches between nodes.
- **Device support in arrow-rs.** The lasting answer, and an upstream
  change.

On GB200 the line blurs, since Grace and its GPUs share coherent memory,
and a host buffer can live where a GPU reads it directly. Spike 3 measures
whether that is fast enough to matter.

### S6.6 Accelerated ML in Rust

DataFusion is a Rust project with a Python binding, and ddx and einfold are
built the same way. A runtime whose features only Python could reach would
leave Rust users with DataFusion and no devices. So every capability is in
Rust first, and the Python package binds it (design R12):
- **Programs** are Rust types: `Program::new(&ctx, sql)`, `lower`,
  `compile`, `run`.
- **Kernels** implement the `Kernel` trait, written in cuTile Rust first,
  or loaded and launched through NVIDIA's CUDA bindings or cudarc.
- **Arrays** cross to Rust array libraries through DLPack.
- **Differentiation** is ddx-ad, already Rust.

What Rust lacks is a frontend like torch-sql: a way to write a model as
array code rather than SQL. Burn, a Rust deep-learning framework built on
CubeCL, has pluggable backends, and a Burn backend whose arrays are XQL
tables would be that frontend (design §13, question 12).

### S6.7 Why no installer, and no subclasses

DataFusion already has two ways to add to a session: builder methods in
Rust, and `with_extensions` in Python. A runtime `install()` function on
top would be sugar over them, and every XQL project would grow its own.

Subclassing the session is worse. `ddxdb.Context` subclasses
`SessionContext` and overrides `.sql()` to understand `grad`. xarray-sql's
context is a session too, and so would the runtime's be, and a class can
inherit only one override of `.sql()`. Extensions compose where subclasses
collide, which is why the design asks ddx for an extension that plans
`grad` inside DataFusion (design §10.2). `with_extensions` also brings an
order and a transaction: each extension contributes codecs and planners
in two phases, and nothing is written to the session until every
extension has returned.
