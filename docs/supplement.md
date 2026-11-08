# DataFusion Accelerated: supplement

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

### S2.5 Where B-0 keeps values resident

In B-0, values stay on the device for as long as they pass from one device
node to another, as handle batches (design §5.3), and become rows wherever
one of DataFusion's operators sits between two device nodes. A node reads
a compact array without a copy when its input is a scan of a runtime
table, which is why runtime tables can be read two ways. B-0's gain
therefore depends on two numbers, which spike 6 measures: how much of each
step's time is inside einfold's nodes, and how often a DataFusion operator
sits between two of them.

If handle batches don't survive DataFusion's rewrites (design spike 7),
device data can stay on the device only inside one node, and residency
shrinks to what crosses steps as runtime tables.

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

The design publishes new state by swapping in a second copy (R5). The
runtime runs ddx's programs through Rust (design D6), whose compiled plans
take fresh scans, but a Python caller of ddxdb's `in_place` mode would see
the promise broken, and every plan that reads the state planned again.
Hence the design's rule that a table holding state reads the handle's
current buffer when its plan runs, whichever path runs it.

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

Before D6, torch-sql's prototype led the co-authors to Python first. It is all
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

The project was first called `xql-runtime`: the runtime layer of XQL. The
design's revisions changed what it is. It became an extension of
DataFusion (design D5) that attaches through DataFusion's own mechanisms,
extends `RuntimeEnv`, adds nodes to ordinary physical plans and composes
with datafusion-distributed. Most of it would serve any DataFusion user,
and the parts specific to XQL are integrations with ddx, einfold and XQL's
frontends, each a DataFusion extension of its own. So it is named for what
it is: **DataFusion Accelerated**, `datafusion-accelerated`.

The name pairs with datafusion-distributed, the project whose shape it
copies: one adds machines to a DataFusion session, the other adds devices.
And it puts the project where its precedents sit. Spark RAPIDS accelerates
Spark by swapping its operators for GPU ones, DataFusion Comet accelerates
Spark with DataFusion's operators, and Sirius accelerates DuckDB with
libcudf's. Each keeps the engine's operators as the fallback.

That precedent is also why the goal widened with the name (design D10). An
accelerator for DataFusion that sped up only array work would promise
more than it gives, since a DataFusion user expects joins and aggregates
to run faster too. The kernel contract already allows any node to be
bound, so relational operators come in as bindings to libcudf, with
DataFusion's own operators as their specification. Array work still comes
first.

Three facts checked when the name was chosen:
- `datafusion-accelerated` was free on crates.io and PyPI;
- `datafusion-gpu` on crates.io is a placeholder from 2022, whose
  repository is DataFusion's own, so DataFusion's community has wanted
  this before;
- "Apache DataFusion" is a trademark of the Apache Software Foundation, so
  the project says plainly that it isn't part of Apache DataFusion, as
  datafusion-distributed does.

The names considered before:
- **`xql-runtime`**: right while the project was XQL's runtime, and too
  narrow once it was DataFusion's accelerator.
- **`datafusion-gpu`**: taken, and too narrow for TPUs.
- **`sql-host`**: einfold's design uses *host* for whatever runs a plan,
  so the name would blur the runtime with the engines.
- **A coined name**, such as `einrun`: reads as einfold's own executor.

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
schedule (design §5.6) holds every physical choice.

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
things, which design R10, §5.6 and §6.6 provide: a schedule the agent can
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
apart, sampled more often, and refusable in strict mode (design §5.7). The
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
| Kernels | Triton, Gluon, Pallas, cuTile Rust, CUTLASS, XLA, `torch.compile`, MLX, or a developer's code |
| GPU memory | a shared allocator, such as CUDA's stream-ordered pools or RMM |
| Moving data between GPUs | NCCL |
| Distribution | datafusion-distributed |
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
non-goal, and would tie the runtime to a large C++ framework that
DataFusion doesn't use.

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

| Concept | JAX | PyTorch | DataFusion today | DataFusion Accelerated (proposed) |
|---|---|---|---|---|
| Attaching to a system | PJRT plugins | device backends (PrivateUse1) | builder methods; `with_extensions` in Python | `.with_accelerator(...)`; `Accelerator(...)` for `with_extensions` |
| Settings | `jax.config` | `torch.backends`, `torch._dynamo.config` | `SessionConfig`, `SET` | `SET accelerated.*` |
| Memory kinds | `device`, `pinned_host`, `unpinned_host` | `.to(device)`, `pin_memory()` | `MemoryPool`, `DiskManager` | tiers named by Arrow device types: `CUDA`, `CUDA_HOST`, `CPU`, `storage` |
| Memory accounting | XLA's allocator | caching allocator | `MemoryReservation` from a `MemoryPool` | device reservations from a device pool in `RuntimeEnv` |
| Allocator | XLA's own (takes 75% at start) | pluggable; `rmm.allocators.torch` | the system allocator | a shared allocator: CUDA's stream-ordered pools, or RMM |
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
| Custom kernels | `pallas_call`, `jax.ffi.ffi_call` | `torch.library.custom_op`, `register_kernel`, `triton_op` | a custom `ExecutionPlan` | `@runtime.kernel("label")`; the `Kernel` trait in Rust |
| A kernel's gradient | `custom_vjp` | `register_autograd` | none | none needed: ddx writes the backward SQL, and its steps are bound by label and role (ddx#259) |
| Failing loudly | none | `fullgraph=True` | none (Polars has `GPUEngine(raise_on_fail=True)`) | `SET accelerated.strict = true` |
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

The runtime doesn't depend on cuTile. It is a test of the design's kernel
contract (design §5.7): if a cuTile user can bind a kernel with a few lines
of their own code, the contract is right. cuTile passes easily, because
its host side was built to run on memory and streams someone else owns,
and each of the contract's requirements meets one of its interfaces:
- `unsafe Tensor::from_foreign(owner, shape, strides)` borrows device
  memory owned by another framework, keeping its owner alive by reference
  count. The runtime's table handle is that owner.
- `Context::borrow_with_owner` and `Stream::borrow_with_owner` borrow
  another framework's CUDA context and stream, so cuTile runs on the
  runtime's streams rather than its own.
- `.schedule(&policy)` takes a `SchedulingPolicy`, which a user can
  implement from the partition's placement in `KernelArgs`, so that cuTile
  places work as DataFusion's partitions say.
- Launches are futures on Tokio, the async runtime DataFusion executes on,
  so a `KernelExec` awaits a kernel without blocking a thread.
- `.compile()` with `api::meta` placeholder tensors compiles a kernel for a
  shape without running it, which is what the `Kernel` trait's `prepare`
  hook is for.
- Its ownership types, exclusive outputs and shared inputs, keep a kernel
  from aliasing its output with an input. That is the kernel-level form of
  the design's rule that tables are values (S5.6).
- `cutile-kernels` has attention, embeddings, norms and pointwise kernels
  that a user could bind to einfold's nodes.

The frictions are small, and they sit in the user's glue, not in the
runtime:
- cuTile has its own trait for memory someone else owns, so the user
  writes a few lines implementing it over the runtime's device array. No
  Rust standard spans such traits, as DLPack does in Python.
- cuTile's element types and some shapes are static, where Arrow's are
  known only at run time, so a binding dispatches on type.
- Its shapes are positional, so `KernelArgs` orders a node's named axes as
  the node declares them.

The same check applied to Triton passes through `data_ptr()` and the
stream, and to Pallas through DLPack and the shared allocator. A library
that fails it would show the contract is missing something.

Its peer-memory operations, which read and write another GPU's memory
over NVLink, may also matter later: within one NVL72 rack, a user's
kernel built on them could exchange the edges of grid tiles without going
through NCCL.

**Other Rust routes.** NVIDIA's `cuda-rust` crates, which cuTile is built
on, and cudarc both load and launch compiled kernels: PTX or cubins from
CUDA C or CUTLASS, or a Triton kernel compiled ahead of time. CubeCL
writes kernels in Rust for several kinds of GPU. Each implements the
runtime's `Kernel` trait, so a Rust program can bind kernels with no Python
at all.

**How kernels fit the session and distributed models.** A kernel is
registered on the runtime's extension before the session is built, in
Python on the `Accelerator` object and in Rust on the builder. That follows
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
| Attaching | `SessionStateBuilder`; `with_extensions` | `.with_accelerator`; `Accelerator` |
| Settings | `SessionConfig` namespaces | `accelerated.*` |
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
- **Kernels** implement the `Kernel` trait, written in any Rust kernel
  library, such as cuTile Rust or CubeCL, or loaded and launched through
  NVIDIA's CUDA bindings or cudarc.
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

---

## S7. Distributed training

The design distributes a program by partitioning its tables and composing
with datafusion-distributed (design §10.5). This section records why, and
what is still unproven. The aspiration, as code, is
[#5](https://github.com/xqlsystems/datafusion-accelerated/issues/5).

### S7.1 Parallelism as partitioning

A database spreads a query across machines by partitioning its tables, and
moving rows between partitions where a join or an aggregate needs them. A
training step is a query over the batch, the weights, the optimizer's state
and, for Samudra, the grid, so the usual kinds of parallelism are choices
of which table to partition, and by what:
- **data parallelism** partitions the batch by sample, and keeps a copy of
  the weights on every GPU, as PyTorch's DDP does;
- **sharded state** partitions the weights and the optimizer's moments by
  weight index, as FSDP and ZeRO do;
- **domain decomposition** partitions the grid by tile, as numerical ocean
  models run on HPC. For a partial tensor, such as a masked Samudra, land
  cells have no rows, so they are never stored or moved at all.

In PyTorch these are different wrappers around a model, each with its own
code path. Here they are declarations in SQL: each table's distribution,
stated where the table is created, which a developer or an agent can change
without touching the training step, while the planner chooses the
partitioning of everything in between, per join, from the whole step's
plan. That is torch-sql's
hypothesis H1: that choosing the partitioning this way distributes a model
like Samudra better than FSDP does.

### S7.2 Collectives as relational operators

| Collective | Relational form |
|---|---|
| all-reduce of gradients | partial `SUM` per partition, shuffle by key, final `SUM`: a reduce-scatter, with the gather left to whoever reads the result next |
| reduce-scatter | the same, without the gather |
| all-gather | a replicated table, or a broadcast join |
| broadcast | a broadcast join |
| all-to-all | a repartition |
| halo exchange | the rows of an index-table join that cross a tile's edge |
| send and receive | a shuffle between two partitions |

The relational form has two advantages. The planner writes it, so nothing
in the model names a collective, and it moves only the rows a join needs:
a halo exchange moves a tile's edge, and land cells never move.

It also has a disadvantage, which is unmeasured. NCCL's collectives are
tuned for each topology: rings and trees, and on NVLink switch systems
reductions done inside the switch. A shuffle-and-sum is bandwidth-optimal
on paper, but nobody has run it against NCCL on the same GPUs. Design spike
8 does. If it loses, the runtime adds a collective node that recognizes a
shuffle-and-sum and calls NCCL's reduce-scatter, which keeps the relational
form in the plan and NCCL's speed in the run.

### S7.3 Transport decides whether it's fast

datafusion-distributed moves Arrow record batches over gRPC, through host
memory. For queries over files that is the right default. For a training
step it puts a host round trip into every gradient reduction and every
halo exchange: the cost that S1.1 measured on one machine, now multiplied
by the network. A GB200 NVL72 rack joins its 72 GPUs with NVLink, so
within a rack the right transport never touches a host. That is why the
design lowers exchanges to the device's own transports as part of
distribution, not an optimization of it (S7.7,
[#6](https://github.com/xqlsystems/datafusion-accelerated/issues/6)),
and why the schedule records the topology: device copies within a node,
NVLink within a rack, the network across racks
([#8](https://github.com/xqlsystems/datafusion-accelerated/issues/8)).

### S7.4 Queries against training programs

Distributed query engines are built for queries that run once: plan,
ship stages, run, discard. A training step runs thousands of times, with
state that must stay where it was computed. Three things follow, none of
which distributed query engines provide today
([#7](https://github.com/xqlsystems/datafusion-accelerated/issues/7)):
- **stages deployed once**, which a run only triggers, as a compiled
  program does on one machine (design §5.4);
- **state resident on workers**, each partition of `w` on one worker's GPU,
  referred to from the coordinating session by a remote table handle;
- **placement by affinity**, so the task that updates `w[k]` runs where
  `w[k]` lives.

This extends the runtime's own layer, the program, across machines. It is
the largest piece of new design in distribution, and the one most likely
to need hooks from whichever distributed engine hosts it.

### S7.5 datafusion-distributed and Ballista

[Ballista](https://github.com/apache/datafusion-ballista) is Apache
DataFusion's own distributed engine, and the alternative to
datafusion-distributed (design D7).

| | datafusion-distributed | Ballista |
|---|---|---|
| Shape | a library on `SessionStateBuilder`; any node can coordinate | an engine: scheduler and executor processes |
| Data between stages | streamed over gRPC | staged mode writes shuffle files; pipelined mode (experimental) streams over Arrow Flight |
| Python | no binding ([#10](https://github.com/xqlsystems/datafusion-accelerated/issues/10)) | `BallistaSessionContext`, but no custom codecs from Python yet ([ballista#2618](https://github.com/apache/datafusion-ballista/issues/2618)) |
| Custom nodes | a codec registered on every session | extension codecs, in Rust |
| Failure | a failed task fails the query | staged mode re-runs failed tasks; pipelined mode fails the job |
| Processes | workers are servers, stateless per query | executors are long-lived, under a central scheduler |
| Published speed | its README reports beating Ballista on TPC-H | |

datafusion-distributed fits the design's shape better: it is a library,
"vanilla DataFusion plus nodes" as the runtime is (design R11), it streams
between stages instead of writing them to disk, and it composes with the
runtime on the same builder. Ballista's strengths are where training's
needs differ from queries': a central scheduler and long-lived executors
are a natural home for stages deployed once and for placement by affinity
(S7.4), its staged mode recovers from a failed task without restarting the
run, and its Python client exists. Neither has a device-aware transport.

So D7 stands, and design spike 9 checks it against a distributed training
step before M7 ([#13](https://github.com/xqlsystems/datafusion-accelerated/issues/13)).
Either way, the runtime's side of the contract should stay the same:
device nodes with a codec, and hooks for placement. Then the runtime
depends on neither engine's internals, and a training job could move from
one to the other.

### S7.6 Failure

A distributed training step is synchronous: every tile's gradient is
summed before the weights update. One slow GPU stalls every step, and one
failed worker fails the run, as with DDP. With state spread across
workers, R5's atomic publication would need every worker to agree before
any publishes. The simpler rule, which the design takes, is to restart
from the last checkpoint, which each worker writes its shards into
([#11](https://github.com/xqlsystems/datafusion-accelerated/issues/11)). Whether a
run can survive a lost worker without a restart, as Ballista's staged mode
allows for queries, is open.

### S7.7 Lowering exchanges to NCCL and NVLink

The relational layer decides *what* moves: the exchanges in a device plan
say which blocks go to which partition. The runtime decides *how*, by
lowering each exchange to the fastest transport that is exact for it
(design §10.5). Under that sits a standard split of distributed systems:
datafusion-distributed's gRPC as the *control plane*, carrying plans,
stages, task assignment, metrics and the bootstrap of each collective
library, and NCCL and NVLink as the *data plane*, carrying the bytes. The
SQL is the same, the plan has the same nodes, and `EXPLAIN` shows the
transport each exchange got.

**Exactness reduces to two conditions.** A shuffle by key between a partial
and a final `SUM` is a reduce-scatter only if each rank's share of the key
is a contiguous slice, which is why sharded state is declared `blocks(k)`
rather than `hash(k)`, and only if each partial is dense over the key, which
compact arrays are. A third subtlety is absence: in SQL a `SUM` over no rows
gives no row, where a dense collective gives zero. So the collective reduces
a count beside each sum, and a group that no partition reached stays absent.
This is einfold's three cell states (absent, NULL, a value) carried through
a collective.

**Speed comes from what DDP already does, plus what the design makes easy:**
- *Persistent communicators and registered buffers.* Communicators are
  created once per compiled program, and buffers are registered with NCCL
  once, because state is written in place and exchange buffers are
  allocated once. Registration enables zero-copy transfers, and on NVL72,
  NVLS, the NVLink switches' in-network reductions. NVLS wants buffers from
  CUDA's virtual-memory APIs, which may conflict with stream-ordered pools
  ([#19](https://github.com/xqlsystems/datafusion-accelerated/issues/19)).
- *Bucketing.* A step's many small reductions are fused into grouped calls
  over flat buffers, as DDP's gradient buckets do.
- *Overlap.* A layer's reduction starts on a communication stream while
  earlier layers' backward steps still compute, ordered by events
  ([#18](https://github.com/xqlsystems/datafusion-accelerated/issues/18)).
- *CUDA graphs.* NCCL calls can be captured, so a compiled step with stable
  buffers could replay whole, collectives included.
- *One-sided halos.* Within an NVL72 rack, GB200's fabric memory handles let
  a GPU map a neighbour's memory across nodes, so a halo can be written
  straight into the neighbour's buffer by an ordinary bound kernel.

**Composing with what is already in the process.** NCCL finds the topology
itself, so the runtime asks it rather than keeping a second model. As a
co-processor beside PyTorch, the runtime derives its communicator from one
the host exposes (`ncclCommSplit`) rather than starting a second world.
`ncclCommAbort` on a failed worker fits the design's restart rule (S7.6).
And since NCCL's reduction order depends on the algorithm it picks, a
session that requires reproducibility pins the algorithm.

### S7.8 The portability test: AMD and TPU clusters

A design that only fits NVIDIA's hardware would be a CUDA runtime with SQL
on top. So the design was tested against two other kinds of cluster:

| | NVIDIA GB200 | AMD MI300-class | Google TPU |
|---|---|---|---|
| Within a domain | NVLink through NVSwitches: 72 GPUs per rack, any to any | Infinity Fabric: 8 GPUs per node, a full mesh with no switch | ICI: a 2D or 3D torus with wraparound, up to thousands of chips per slice |
| Between domains | InfiniBand | Ethernet (RoCE) or InfiniBand | the data-centre network, between slices |
| Collectives | NCCL, called per operation on a stream | RCCL, which implements NCCL's API | compiled by XLA into the program; nothing to call on its own |
| Raw pointers and queues for users | yes | yes, through HIP | no: buffers belong to PJRT, inside compiled programs |
| Arrow and DLPack device type | `CUDA`, `CUDA_HOST` | `ROCM`, `ROCM_HOST` | none |
| Kernel languages | Triton, Gluon, Pallas, cuTile Rust, CUTLASS | Triton, HIP, CubeCL | Pallas (Mosaic TPU), inside XLA programs |

**The relational core passed unchanged.** Declared partitioning,
exchanges in the plan and collectives as relational operators assume no
topology, so they hold on all three. A torus suits domain decomposition
especially well: torus networks were built for stencils, every halo
exchange of a grid mapped onto the torus goes to a direct neighbour, and
Samudra's periodic longitude, which wraps around the globe, maps onto the
torus's own wraparound.

**AMD mapped almost one to one.** HIP mirrors CUDA's contexts, streams,
events and stream-ordered allocator, RCCL speaks NCCL's API, and Arrow and
DLPack have ROCm device types. What it tested was the topology: with no
72-GPU domain, Samudra's 36 GPUs span five nodes, so the design's "first
distribution stays within one rack" turned out to be NVIDIA-specific, and
placement has to keep most halos within a node from the start. With no
switch, there are no in-switch reductions; rings and trees do instead.

**The TPU found four places where the device layer was CUDA-shaped:**
1. **The kernel contract** handed kernels raw pointers, contexts and
   streams, which TPUs don't give users. It is now defined per device
   family: DLPack where a device has a type, the backend's native buffers
   otherwise, an execution context and queue rather than a CUDA context and
   stream, and bindings that are either callables or functions the backend
   compiles in
   ([#20](https://github.com/xqlsystems/datafusion-accelerated/issues/20)).
2. **Collectives were assumed callable.** On a TPU they exist only inside a
   compiled program, and speed comes from XLA scheduling them across the
   whole step. So transports are either eager or compiled, and regions may
   contain exchanges, which a compiled transport lowers with the region's
   computation. B-0, one node at a time, is therefore a poor fit for TPUs
   ([#21](https://github.com/xqlsystems/datafusion-accelerated/issues/21)).
3. **Tiers were named by Arrow device types**, and a TPU has none. They are
   now named by device family, and a TPU's tables cross into DataFusion only
   through the host.
4. **The topology was a list of tiers.** It is now a graph, covering
   switched domains, full meshes and tori, taken from each platform's own
   discovery, with placement as an embedding of the partition grid into it
   ([#8](https://github.com/xqlsystems/datafusion-accelerated/issues/8)).

None of these changed the design's shape, which is why it passed: the
generalizations sit in the backends, in how an exchange is lowered, how a
kernel binds and how partitions are placed. Spikes 10 and 11 test the
prediction on real AMD and TPU hardware
([#22](https://github.com/xqlsystems/datafusion-accelerated/issues/22),
[#23](https://github.com/xqlsystems/datafusion-accelerated/issues/23)).

---

## S8. Database theory for parallelism

The design's parallelism follows four decades of work on parallel
databases. This section maps that work onto it. Most of it confirms the
design. Five points changed it (design §10.5): placement through
DataFusion's physical properties, blocks rather than rows in exchanges,
tensor and pipeline parallelism, a cost model for the partitioning of
intermediates, and a stated rule for halos.

### S8.1 Exchange operators and physical properties

Graefe's *exchange* operator, in the Volcano system (SIGMOD 1990), put all
of a plan's parallelism into one operator, so that every other operator
runs unchanged on its partition. datafusion-distributed's network nodes
and the runtime's `CopyExec` are exchanges.

The Volcano optimizer generator and its successor Cascades then treated
partitioning as a *physical property*. Each operator requires a property of
its inputs and delivers one on its output, and the optimizer inserts an
*enforcer*, here an exchange, wherever the two differ. SCOPE, Microsoft's
distributed SQL system (Zhou, Bruno and Lin, ICDE 2010), and Greenplum's
Orca optimizer (Soliman et al., SIGMOD 2014) reason about distribution
this way, and Orca's *Motion* operators (redistribute, broadcast, gather)
are what the design calls shuffles.

DataFusion has the same framework: `Distribution` requirements,
`Partitioning`s that nodes deliver (hash, round robin, and, since
DataFusion 55, ranges), and the `ensure_requirements` rule that inserts
repartitions. That is why the design places tiles and shards through it,
rather than through a rule of the runtime's own.

### S8.2 Aggregates that can be summed in parts

Gray et al.'s data cube paper classifies aggregates as *distributive*,
such as `SUM`, *algebraic*, such as `AVG`, kept as a sum and a count, or
*holistic*, such as a median. Only the first two can be computed in parts
and combined, which is what makes "a partial sum per GPU, a shuffle, a
final sum" a correct all-reduce. Yan and Larson's *eager aggregation*
(VLDB 1995) is the general rule for moving an aggregate below a join. A
float `SUM` computed in parts adds in a different order, which the design's
floating-point bound covers (design §6.3).

Training as an aggregate has its own prior art. Bismarck (Feng, Kumar,
Recht and Ré, SIGMOD 2012) and MADlib (VLDB 2012) ran stochastic gradient
descent inside databases as user-defined aggregates. The parameter server
(Li et al., OSDI 2014) is weight state partitioned by key, pushed to and
pulled from: the design's `hash(k)` state.

### S8.3 Communication cost, and its lower bounds

The *massively parallel communication* model (Beame, Koutris and Suciu,
PODS 2013) measures a parallel join by the data each server receives. In
it, the *HyperCube* or *Shares* algorithm (Afrati and Ullman, EDBT 2010) is
optimal for joins done in one round, and the bounds depend on the join's
structure.

A matrix product is a join and a `SUM`. HyperCube applied to it is the
3D distributed matrix multiply, whose communication lower bounds are known
(Irony, Toledo and Tiskin, 2004), and which Solomonik and Demmel's 2.5D
algorithm (2011) trades against memory. Tensor parallelism, as Megatron
does it, and its 2D, 2.5D and 3D variants are therefore partitionings of a
join too, which the design's table of strategies now lists.

Two things follow:
- the cost model that chooses intermediates' partitionings (design §10.5)
  can use this communication load as its measure, as Alpa (Zheng et al.,
  OSDI 2022) and EinDecomp do with their own;
- spike 8 has a second target: the lower bound for each join, beside NCCL.

### S8.4 Band joins, and halos

A halo exchange is a *band join* on grid coordinates: each output cell
joins input cells within a fixed distance (DeWitt, Naughton and Schneider,
1991). Spatial joins handle it by replicating boundary tuples into
neighbouring partitions, as PBSM does (Patel and DeWitt, SIGMOD 1996), and
avoid duplicate results with the *reference point method* (Dittrich and
Seeger, 2000): each result is produced only by the partition that owns it.
The design's halo rule is that method: replicate to the band's width, and
let the owner compute.

### S8.5 Iterative computation

Training repeats one step thousands of times, and distributed systems for
iterative work keep data where it was computed:
- HaLoop (Bu et al., VLDB 2010) schedules each iteration's tasks where the
  previous iteration left its data, and caches what doesn't change;
- Spark (Zaharia et al., NSDI 2012) persists data across iterations, and
  recovers a lost partition from its lineage;
- Naiad (Murray et al., SOSP 2013) deploys a dataflow with loops once.

The design's distributed programs, with stages deployed once and state
resident on workers, follow them ([#7](https://github.com/xqlsystems/datafusion-accelerated/issues/7)). Synchronous steps are Valiant's
bulk-synchronous parallel model. Its relaxations, such as stale-synchronous
parallelism, would conflict with R5 and R9, so the design keeps steps
synchronous.

### S8.6 Tensor-relational systems, and ML's own sharding

The closest prior art to the whole design is Chris Jermaine's group:
SimSQL, which trained neural networks as relational computations; the
[tensor relational algebra](https://par.nsf.gov/biblio/10320005-tensor-relational-algebra-distributed-machine-learning-system-design)
(TRA, VLDB 2021), in which relations hold tiles of tensors rather than
single values, and an implementation algebra chooses how each is
partitioned; [relational auto-differentiation](https://arxiv.org/abs/2306.00088)
(ICML 2023), which is ddx's territory; and
[EinDecomp](https://arxiv.org/html/2410.02682v1), which chooses how to split
einsums for parallel execution automatically.

Their central lesson matches S1: at the grain of distribution, a tuple
should be a block, not a scalar. Long-format rows are the right model of an
array's meaning, and the wrong unit to send across a network. The design's
compact arrays are its blocks, so its exchanges move them whole.

On the ML side, Mesh-TensorFlow (Shazeer et al., 2018) maps named tensor
dimensions onto a mesh of devices, and GSPMD (Xu et al., 2021) propagates
sharding annotations through a program. Partitioning a table by named
columns is the same idea in relational terms.

### S8.7 Declaring distribution in SQL

MPP databases declare a table's distribution in its DDL, and plan every
query from it:
- Greenplum: `DISTRIBUTED BY (k)`, `DISTRIBUTED REPLICATED`;
- Amazon Redshift: `DISTKEY (k)`, `DISTSTYLE ALL`;
- Citus: `create_distributed_table` and `create_reference_table`;
- Hive and Spark: `CLUSTERED BY (k) INTO n BUCKETS`, and `DISTRIBUTE BY`
  on a query, which DataFusion also parses.

Declaring a base table's distribution, and letting the optimizer derive
the rest, is the design's rule (design D9).

### S8.8 Logical and physical

In Codd's terms, partitioning is *physical*: it doesn't change what a query
means, which is the guarantee data independence gives (design R7). It is
still declared in SQL, as indexes and `DISTRIBUTED BY` are, because SQL is
where a program's author states what they intend. One case looks like an
exception, and isn't. When partitioning would change a result, as batch
norm with statistics per device does, the dependence is part of the model,
and the model says it: a `GROUP BY` on a tile column. The two never mix:
declared distributions change nothing but speed, and anything that changes
a result is written as logic, where ddx differentiates it and checking mode
compares it.
