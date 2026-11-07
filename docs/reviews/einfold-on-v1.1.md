# einfold's review of the runtime design, draft v1.1

🪭 Claude (einfold), reviewing as einfold's approver. 2026-11-07.

**Verdict: design B accepted in principle, and the doc not yet approved as written.**
I want the runtime to exist, and I agree with its central split: einfold
translates, the runtime executes, and SQL's semantics live only in engines
and in einfold's proofs. Four changes block my approval (B1 to B4 below).
Each is small in text but changes what M2 means. Everything after them is
advice, and the authors may take it or leave it.

Whether einfold takes on design B is finally Alex's decision as einfold's
maintainer, not mine. It changes einfold's roadmap (B4), and einfold's
rules require a sketch in einfold's own design doc before code. What
follows is what I would propose there.

Sources: einfold's design ([`docs/design.md`](https://github.com/xqlsystems/einfold/blob/main/docs/design.md),
v20 on main), its code on main, and these open einfold PRs and issues:
- #155, convolutions;
- #156 and #157, the dense backend and `DenseProgram`;
- #39, xarray-sql's facts;
- #58, #81 and #145, the costs and gaps of the FFI;
- #79 and #80, memory and dictionary-encoded dimensions;
- #135, attention's fused backward pass.

---

## What blocks approval

### B1. einfold checks data at run time; it doesn't prove facts at compile time. The doc says the opposite.

§7.2 says einfold "already proves that a plan fragment computes the same
thing as an array algorithm, using facts about the inputs". §11.3 says
"einfold's fact table, through which frontends pass facts ... and hints
... reaches the runtime the same way". Neither is true of einfold today.

- **The fact table is designed but not built.** einfold's design §8.1
  describes it, and einfold's M5 builds it. No code on main reads a
  fact. There is no channel for hints at all. xarray-sql's side of the
  contract is written down in einfold#39, with exact row, key, extent and
  NULL guarantees, but nothing in einfold consumes it yet.
- **What einfold proves statically is the plan's structure.** For example,
  that this aggregate is a `SUM` of a product over an inner equi-join, that
  each factor reads one side, and that a group key is
  `x.lat - 2·(w.ky - 1)` (#155).
- **Everything about the data is checked at run time, on the rows that
  arrive.** That covers dense integer ranges, unique tuples, NULLs, holes,
  negative keys, and the finiteness that the stencil kernel and the chain
  kernel's float guard need. `EinFoldExec`, einfold's DataFusion operator,
  checks each one as it reads its inputs.
- **When a check fails, einfold runs the host's own plan.** Every detected
  node keeps that plan, unplanned until it is needed.

So einfold's guarantee is "the structure matches, and the data passed
checks on this run, or else the host's plan ran". For a CPU operator that
reads Arrow rows, that model is cheap and sound. For an array region with
state resident on a GPU, it is the hard part of design B, and the doc
needs to say so:

1. **Every check einfold does at run time becomes a data check of the
   region (§6.1).** On a device, a check is a reduction followed by a sync
   to the host. `TrainingSession`, the xarray-sql training loop of §3.1,
   declined to compile any step whose checks read device data. The way out
   is already in the doc. §3.1's resident step read back "one validation
   flag", and R5 publishes state only after the checks pass. So each of
   einfold's checks should compile to an on-device predicate, ANDed into
   the step's one flag and read once at the end. A run whose flag fails is
   a decline at run time (§7.3): it rolls back, and the region runs on the
   engine from its inputs.
2. **Facts make checks unnecessary.** An Exact fact from a frontend, such
   as torch-sql's "every table is complete" (§3.3) or xarray-sql's
   structural guarantees (einfold#39), lets einfold drop a check entirely.
   So the fact table moves onto design B's critical path. It is no longer
   a nice-to-have of einfold's M5.
3. **Some facts can't be compile-time facts.** einfold#39 found that
   xarray-sql's values change between queries without the table being
   registered again (its `Context.prepare` writes into the arrays). Facts
   about values, such as finiteness, are true for one run only. The
   stencil (#155) needs every weight finite before it accepts holes in the
   image, and a chain needs its float guard. So a compiled program always
   has some data checks. Treating "structural checks only" as the normal
   case (§8, on buffer donation) is optimistic.

**Ask:** rewrite §7.2's and §11.3's descriptions of einfold as above.
Then add, to §7.3, the rule that einfold's run-time checks become a
region's data checks, folded into R5's validation flag.

### B2. The portable program stays SQL; einfold's IR annotates it; no new array language

§7.2 lists elementwise expressions, `einsum`, reductions, gathers, scatters
and `where`. §17.2 Q2 asks whether to use the Python array API standard or
a subset of StableHLO, the operation set that JAX and XLA compile. Alex
rejects StableHLO as XQL's IR (2026-11-07): an array program written in
SQL is already the most portable IR there is, since every engine runs it.
einfold agrees, and would go one step further. There should be no
separate array language at all.

- **SQL is the program.** It is what every engine runs, what ddx
  differentiates, and what R1 calls the meaning.
- **einfold's IR is an annotation on that SQL, not a translation of it.**
  An einfold node says "this subplan is this fold" and carries the subplan
  with it. This is einfold's extension form as it already exists: every
  detected node keeps the plan it replaced. A runtime runs the node on
  arrays when it can, and runs the SQL when it can't. So a decline never
  needs a second program.
- **Lowering to an array library is private to each backend.** A backend
  may lower a fold to NumPy, MLX or `torch.compile`, or to StableHLO
  through JAX. None of that crosses an interface, so none of it needs
  agreement between projects.

A second program in StableHLO, the array API standard or a format of
einfold's own would have to be kept equal to the SQL forever. That is the
risk this doc rejects in design A. It would also lose two things.

- **Existence isn't a mask on values.** A compact array has three states
  per cell: absent (no row), NULL, or a value. einfold's kernels already
  carry exactly this, as per-row ABSENT, NULL and VALUED flags. #155's
  stencil kernel is the newest example. SQL's rules for these states don't
  reduce to `where`:
  - a group exists only if some row reached it;
  - `SUM` over no rows is no row, not `0`;
  - a left join from a domain turns that into NULL, or into a default such
    as ddx's `0.0`;
  - the sign of a zero sum follows the host (DataFusion's `SUM` starts
    from `+0.0`).

  einfold's property tests check each of these rules, bit for bit,
  against DataFusion. No existing array operation set states them. If the
  contract is primitive operations plus masks, then every backend
  reimplements these rules. That is design A's second copy of SQL,
  moved down one layer.
- **Primitive operations lose the structure that backends need for speed.**
  einfold's IR already has three nodes:
  - a *fold*, a sum of products over a join, with its domain, NULL rules
    and fallbacks;
  - a *pipeline*, a fold that reads another fold's result through an
    expression, which einfold fuses as FlashAttention does;
  - a fold with *shifted outputs*, a stencil (#155).

  Lowered to `einsum` + `exp` + `einsum`, attention materializes `N²`
  scores. Demo 1's fused kernel exists precisely to avoid that, running
  65,536 tokens in one tile per thread (einfold design §10.2, spike S27).
  Lowered to gathers, a convolution becomes an index table and a gather.
  The stencil is one GEMM per tap, with no indices at all.

**Ask:** in §7.2, replace the list of array operations with this: a region
is a SQL plan in which einfold has marked nodes (folds, pipelines, shifted
folds, gathers on value indexes, and the elementwise expressions between
them). Each node states its named axes and its three states, and carries
the subplan it stands for. The node types are versioned, and einfold
publishes them with a conformance suite, as einfold's design already plans
for its `Fold` Substrait relation (its §7.2). A backend that can't run a
node declines it, and that node's SQL runs. §18.1's choice of GPU library
then no longer depends on einfold's answer.

This has a cost, and the doc should state it. SQL's portability is a
portability of meaning, not of speed. Everything fast still depends on
einfold recognizing structure, and anything einfold doesn't recognize runs
at the engine's speed. That is the right failure mode, but it makes
detection's coverage the number to watch (spike 6, and hints in the
answer to Q5).

One consequence: the runtime depends on `einfold-ir`'s published types.
That direction is consistent with P-D, since the runtime implements
einfold's interfaces.

### B3. What to save, and where regions fall, are one decision

§7.4 and §7.5 let the runtime choose save points from a planning budget
and from memory, with einfold translating whatever regions remain. But
saving a value can destroy the structure einfold fuses:
- if the runtime saves attention's scores `S`, there is no pipeline left
  to fuse;
- einfold#135, attention's fused backward pass, depends on ddx
  *recomputing* the scores in each backward step instead of saving them
  (ddx#251).

The reverse also happens. Inlined views make plans einfold matches worse.
The Samudra demo as views is 3,862 operators, and each block's input is
recomputed (#155). The GPT's loss planned 2.7× longer with einfold on,
through 9,882 lines of inlined plan (#81), before #86 to #92 fixed most of
that.

**Ask:** in §7.5 and §8, say that the runtime chooses save points *with*
einfold. The simplest form: einfold reports, for a candidate save point,
whether it would split a node it can fuse, and the runtime prefers save
points that don't. A saved pipeline producer is allowed, but it is counted
as a decline with that reason (R8), so its cost is visible.

### B4. Stage design B, so M2 doesn't wait on einfold's whole program mode

As written, M2 needs einfold to translate *regions of steps*. einfold today
is a DataFusion optimizer rule, called once per query. ddx materializes
each step as a table, so einfold sees one step at a time (einfold's M3).
Seeing a whole program is einfold's "program mode", planned for its M7,
after DuckDB (M4), facts (M5) and sparse kernels (M6). Design B pulls M7
and part of M5 ahead of all of those, and that is a scope decision for
Alex.

I would make it, in stages, each measured and each useful alone:

- **B-0, one node per region.** einfold already detects the node. Instead
  of running it on Arrow rows in `EinFoldExec`, einfold hands the runtime
  three things:
  - the node's IR;
  - its run-time checks, as data checks;
  - the host's plan, for the fallback.

  The runtime runs the node on compact arrays it keeps resident. This
  replaces #156's per-call interface (below), and needs no new detection.
  It is the first runtime M2.
- **B-1, regions within one step.** einfold nodes connected through
  elementwise projections in one plan become one array program. This is
  most of a forward pass.
- **B-2, regions across steps.** This is einfold's program mode, and it
  needs ddx's step graph (ddx#157) and regions (ddx#251).

Spike 6 should also measure B-0's coverage before anyone builds it. For
demo 2, demo 4 and torch-sql's nanoGPT, run `EXPLAIN ANALYZE` with einfold
on, and report the share of each step's time inside einfold's nodes
against the share left on the host. That takes an afternoon, and it says
how far B-0 gets before B-1 is needed. I'll run einfold's side of it.

**Ask:** replace M2's "array regions" with B-0, and list B-1 and B-2 as
later milestones gated on their own measurements. Record in the decision
log that B requires an einfold design change approved by its maintainer.

---

## Answers to §17.2

1. **Does einfold want to own design B?** Yes, with B1 to B4. It fits
   einfold's architecture as a third *output form*, beside the relational
   form (rewritten SQL) and the extension form (`EinFoldExec`). It keeps
   einfold's founding rule that einfold "rewrites query plans; it never
   executes them". The runtime is the host.
2. **The operation set:** none of its own. The program stays SQL, and
   einfold's IR nodes annotate it (B2). StableHLO is rejected as XQL's IR
   (Alex, 2026-11-07). Like the array API standard, it may serve as one
   backend's private lowering target. Neither has a notion of a row's
   existence.
3. **`DenseProgram` (#157):** a prototype to learn from, not the place to
   start. It interprets DataFusion's logical operators one at a time. That
   is design A, in a different repository. It is 1,564 lines of Python with
   private types, which xarray-sql imports by name (§3.1, lesson 5), and
   einfold's maintainer hasn't reviewed it. Its value is twofold:
   - **as a corpus:** the operators and spellings it had to handle to
     train a 123.6M model are the best evidence we have for spike 6;
   - **as an oracle:** its outputs on those programs are a second check on
     einfold's translations.

   The translation itself belongs in Rust, beside detection. It must serve
   DuckDB too (einfold's M4), and einfold's proofs and property tests are
   in Rust.
4. **Facts for a region's proof:**
   - each dimension's extent, and that its keys are integer positions
     `0 .. n − 1`;
   - completeness (every coordinate tuple present) or a domain table;
   - unique tuples;
   - values never NULL;
   - for gathers, that every index lies in range;
   - for reordering sums, that values are finite and bounded (einfold's
     float guard, its design §8.6).

   torch-sql's §3.3 list covers the structural ones exactly, and
   einfold#39 states the same for xarray-sql. No fact table carries any of
   them yet (B1). Two notes:
   - The runtime should be a fact provider for the tables it writes. It
     knows what it wrote, and einfold's design already lists the runtime
     among its fact sources.
   - Exact facts change results if they're wrong, so checking mode (§7.3)
     should also verify the facts a region relied on, not only compare
     results.
5. **Checking a candidate rather than finding one:** yes to hints, but not
   as a way around detection. A hint ("this is `conv2d`, stride 1,
   dilation 2") tells einfold which spelling and which axes to try. It can
   never stand in for the proof. einfold's principle 8 is that users
   declare semantics and einfold proves optimizations. Hints have one big
   use: a hint einfold fails to confirm is a counted decline with a reason.
   That is the measure of detection's coverage we lack today.

---

## Advice, not blocking

- **Drop "through a kernel" (way 2 of §7.1) as an investment.** #156's
  interface packs each kernel's operands from Arrow and calls back into
  Python on every call. §3.1 measured that path at 164.0 s against
  163.6 s, and §3.1's own lesson 1 explains why. Keep #156 as a stopgap
  for large, compute-bound kernels. Let B-0 subsume it.
- **The FFI is part of Python first's cost.** einfold reaches
  datafusion-python's sessions through datafusion-ffi, DataFusion's
  interface for extensions built separately from it. Through it, steps
  einfold doesn't match run 1.25× slower and plan 2.9× slower than on
  DataFusion alone (einfold#58 and #81). §3.4 says planning already
  dominates small steps. The FFI also has gaps that einfold has filed as
  upstream bugs:
  - object stores don't cross it (einfold#149);
  - a DataFrame outlives its session badly (#145 and #146);
  - functions planned through it lose the session's configuration (#137).

  Spike 4 should measure the FFI's cost alongside the GIL, Python's
  global interpreter lock. A runtime with a Rust core links DataFusion
  and einfold directly and has neither cost.
- **Define "the stated floating-point bound" once.** R1, §4.2 and §18.5 say
  "stated bound" and "tolerance stated per backend". einfold found that a
  fixed tolerance is not enough. A float sum of `n` terms in any order is
  within `(n − 1)·u·Σ|terms|` of the exact sum, provided nothing overflows
  (einfold's design §8.6). Adopt that, with `u` set by the precision the
  backend uses. Then float32 on the GPU is an explicit, per-session change
  of `u`, as #156 already makes precision the host's choice. It is not a
  tolerance applied to an answer meant to be float64.
- **State a stance on determinism.** einfold's kernels give the same bits
  for any thread count (its spike S24), and einfold never adds
  nondeterminism. GPU backends with atomic adds don't give the same bits.
  The runtime should say whether it is deterministic by default, or by a
  setting, since checking mode and reproducing a training run both depend
  on it.
- **Make the runtime the memory pool.** `EinFoldExec` reserves its inputs
  and working arrays from DataFusion's memory pool, and runs the host's
  plan when the pool refuses (einfold#79). If the runtime gives the
  engine session its own pool, einfold's reservations count against the
  runtime's budget (§8) at no extra cost.
- **Convolutions need integer keys and affine spelling.** #155 (a draft)
  detects convolutions written as shifted group keys, as xarray-sql's
  Samudra demo writes them. It runs one dense matrix product per tap of
  the kernel. On the i7-8700 at two levels, with each layer saved
  ("eager"):
  - at 128 × 256, 0.74 s with einfold against 2.84 s on DataFusion alone;
  - as views at 64 × 128, 9.6 s against 11.2 s.

  These numbers come from one run each, and could update §3.2. Two
  findings for frontends:
  - **The demo's pooling makes coordinates `DOUBLE`.** `floor(lat / 2.0)`
    does this, and einfold's dense kernels take only integer keys.
    `lat / 2` on an integer column keeps them integers.
  - **torch-sql writes convolutions through index tables.** §7.2 then
    turns those into gathers, which run much slower than the stencil.
    Either torch-sql spells shifts as arithmetic on keys, or it states as
    a fact that an index table is affine, so einfold can read the shift
    from it.
- **Dictionary-encoded dimensions** (einfold#80) are the portable way to
  give keys as positions across engines and bridges. They belong in §6.2's
  compact-array placement, and in the facts that frontends state.
- **Small corrections:**
  - §2 calls einfold#156 "in review". #156 and #157 are drafts from a
    fork. 🌀 has commented on them, but einfold's maintainer hasn't
    reviewed them.
  - The runtime's AGENTS.md says spikes are in §15 and milestones in §14,
    but they are §16 and §15. It also still calls the name a placeholder,
    while P-F proposes `xql-runtime`.

## What I'd sign

Once B1 to B4 are in the doc, I'll approve. Alex has asked that no code
be written yet. When the doc settles, design B would go to einfold as a
design-doc pull request for Alex: B-0 first, with einfold's nodes as
annotations on SQL.

🪭 Claude (einfold)
