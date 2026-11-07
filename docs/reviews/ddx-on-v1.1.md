# ddx's review of the runtime design, draft v1.1

∂ Claude (ddx), reviewing the sections that touch ddx. 2026-11-07.

**Verdict: the runtime and design B accepted from ddx's side; the doc not
yet approved as written.** I want the runtime to exist. Its split matches
how ddx is built: ddx writes programs, and something else runs them
through ddx's `Backend` (§3.4). Four changes block my approval, D1 to D4.
- **D1 and D2 are about recomputing.** The doc makes recomputing the
  runtime's choice, but Alex decided for ddx that it must be opt-in. The
  doc also promises more than recomputing can keep.
- **D3 and D4 correct what the doc says ddx does.** ddx already
  differentiates through shared CTEs and through programs, and it already
  draws step boundaries for four reasons. That changes §7.4, §7.5 and P-I.

Everything after D4 is advice.

Sources: ddx `main` (ddx-ad 0.3.3, ddxdb 0.4.7, released 2026-11-06). Code:
- `ddx_ad::rewrite`, with its tests `sql_shared_ctes.rs`;
- `compose.rs`, with its test `shared_relations.rs`;
- `opt/split.rs`, `opt/inline.rs`, and the checks in `program.rs`;
- `python/ddxdb`.

Issues and PRs:
- ddx#168, #169, #203, #214, #231, #253 to #255 (merged);
- ddx#157, #251, #259, #260, and #261 to #265 (open).

Some ddx terms used below. A **step** is a plan whose rows ddx saves as a
table, which later steps read by name. A **saved aggregate** is an
aggregate's output that the backward pass reads, so ddx writes it as a
forward step. **`prepare`** builds a program once and runs it again on new
data, and a backend's **`compile`** keeps each step's physical plan for
those runs.

---

## What blocks approval

### D1. Recomputing is opt-in; R7 makes it the runtime's choice

R7 says "saving a value as a table, recomputing it, or keeping it on a
device is the runtime's choice". Goal 5 of §4.1, and §8, have the runtime
choose what to save and what to recompute from the memory available.

On 2026-11-01 Alex decided, for ddx#251, that recomputing is an explicit
policy that people opt into, as with `jax.checkpoint`. The engine must not
choose it on its own. ddx is being built to that rule: a program with no
policy is unchanged, byte for byte.

R7 joins two choices that differ:
- **Where a value is held**: a shared subplan saved as a table, or a table
  kept on a device. This changes where data sits, not what is computed. The
  runtime choosing it is fine.
- **Recomputing instead of saving.** This changes the work done and the
  memory used. On plain engines recomputing attention's scores computes
  them three times (ddx#251), which is a trade the user should choose.

**Ask:** split R7. Where a value is held is the runtime's choice. Whether a
value is recomputed is the user's, given as a policy (§8), and the runtime
applies a memory-budget policy only when the session turns one on. Goal 5
becomes "budget memory, under a policy the user chooses". Saving *more*
than a region asks, as §7.5's last paragraph allows, recomputes less, so it
stays the runtime's choice.

### D2. R7's guarantee holds only for plans that give the same rows twice

R7 says that saving or recomputing "changes neither results nor gradients".
Recomputing changes results when a value's plan doesn't give the same rows
each time it runs:
- **A different set of rows.** This happens with a volatile function such
  as `random()`, a `LIMIT` without a total order, or a rank over ties.
- **Different last bits.** A `SUM` adds DataFusion's partitions in the
  order they arrive.

ddx meets the first case already. ddx#254 reads a window aggregate as its
`GROUP BY` joined back, which computes the window's input twice. So it
refuses when that input calls a volatile function, cuts rows with a
`LIMIT`, or holds a window it can't rewrite. But its test for volatility is
a fixed list of function names. A user-defined function that isn't on the
list, a frontend's dropout say, is taken as deterministic.

**Ask:**
- **Qualify R7.** It should read "within R1's bound, for values whose plans
  give the same rows each time they run", and say that the runtime and ddx
  recompute nothing else.
- **Make the engine the source of volatility.** DataFusion's functions
  declare their volatility, so ddx should read it through its `Backend`
  rather than from a list. That is a ddx change, and I'd make it before
  ddx#251's recomputing lands.
- **Qualify §7.5's claim** that `torch.utils.checkpoint` "loses nothing".
  That holds only if the counter-based random function of §10 is declared
  deterministic. Dropout through `random()` can't be recomputed at all.

### D3. ddx already differentiates through shared CTEs and through programs

§7.5 says that ddx "differentiates the plan of the whole loss, which the
engine produces first", and that "by the time the runtime sees ddx's steps,
the growth has happened". That is true of a loss handed to ddx as one
Substrait plan. It isn't true of ddx's other two inputs.

- **SQL.** ddxdb's `sql_all`, `prepare` and `rewrite` take `grad(...)`
  written in SQL. Before the engine's producer runs, they plan each CTE that
  the objective reads more than once on its own. The pieces are joined into
  one plan that reads each piece through a `ReferenceRel`, Substrait's
  relation for a subtree that several parents share, and ddx lowers each
  piece once (ddx#168). A view the objective reads is written in as CTEs
  first (ddx#169).

  A thirty-layer chain, which written out holds 2^30 copies of its first
  layer, is differentiated from SQL in about a second on an i7-8700
  (ddx#168, test `a_chain_too_deep_to_write_out_is_differentiated_from_sql`).
  The gradients match the written-out query's on the chains small enough to
  write out (`a_cte_read_twice_is_planned_once_and_the_gradient_is_the_same`).
- **Programs.** `grad` and `vjp` take a program as well as a plan. ddx
  writes each step once among the plan's relations, and reads it through a
  `ReferenceRel` wherever another step reads it. So the plan is the size of
  the program, however often its steps read one another (`compose.rs`).

So the save points §7.5 asks for, values ddx computes as steps and
differentiates through, exist today. Each one is either a CTE read more than
once, or a step of a program handed to ddx. Two things follow:
- torch-sql's chain rule across segments (`torch_sql.program`) can be
  replaced now by handing ddx the segments as one program.
- §3.2's Samudra demo can keep its views. ddx differentiates through them,
  and plans each layer once when a layer is read twice.

What's missing is small:
- **Save points beyond "read more than once".** ddx leaves a CTE read once
  to the engine, so a budget that wants to cut one long chain can't. ddx
  would take the names of extra CTEs or views to plan alone, as an option.
- **DuckDB.** ddx#264 is a gap in this path on DuckDB.

§7.4's plan, "the runtime finds shared subplans in the program's plans, not
in SQL text", can't work for the case it targets. Neither engine's producer
emits a `ReferenceRel` (torch-sql's spike S3), so a plan arrives with its
copies already written out: 136,107 operators for three blocks (§3.3).
Sharing has to be found before the producer runs:
- in SQL text, as ddx#168 and `SharedExecutor` do;
- or in the frontend's own graph, handed over as a program.

Inside ddx's programs, nothing is left to find:
- ddx saves each aggregate once, by structural identity;
- it lowers each shared relation once;
- it computes a step read by one later step inside that step (ddx#231).

**Ask:**
- **§7.5.** Rewrite its first two paragraphs from the above. The order
  "the engine plans, then ddx differentiates, so the growth has happened"
  holds only for a single plan.
- **§7.4.** Take shared subplans from the frontend's program, or from SQL
  before it is planned. For plain queries, ddx can offer #168's split as a
  public function that doesn't differentiate. Then the runtime needs no
  second implementation, and calls ddx's public API (R2).
- **P-I.** Reword it as: "the runtime chooses save points beyond those ddx
  makes, and passes them to ddx. ddx differentiates through them, so that
  no frontend runs a chain rule of its own."
- **§18.4.** Name Substrait with `ReferenceRel` as how plans between layers
  carry a shared relation.

### D4. One owner for step boundaries

ddx already draws step boundaries, for four reasons:
1. **Saved aggregates.** Each aggregate the backward pass reads is saved as
   a step.
2. **Shared CTEs.** Each CTE read more than once is planned alone
   (ddx#168).
3. **Depth.** A step deeper than a consumer can decode is split every 32
   relations (`opt::split`). datafusion-python decodes Substrait with
   protobuf's default limit of 100 nested messages.
4. **Inlining.** A step that one later step reads, once, is computed inside
   that step (ddx#231). That isn't done if the reader would pass 2,000
   expression nodes or 32 relations.

The doc adds another chooser: the runtime, from a planning budget (400
`SELECT` blocks on DataFusion, as torch-sql uses) and from memory, with
regions from ddx#251 on top. Two choosers with separate budgets undo each
other's work:
- ddx folds a step into its reader that the runtime meant to save;
- or the runtime saves a value that the user asked ddx to recompute.

**Ask:** say in §7.4 or §8 that the runtime takes ddx's program as ddx
writes it. The runtime changes step boundaries only through ddx's inputs:
- save points (D3) and recompute policies (D1) are options to ddx;
- engine limits, such as the depth a consumer decodes and a planning
  budget, are facts the `Backend` reports. ddx reads them in place of its
  constants.

Each budget then has one owner. ddx's `prepare` cache also stays correct,
because a program remains a function of its plan and its options. That is
also why ddx should take save points from the caller rather than ask its
`Backend` for them (answer 4 below).

---

## Answers to §17.1

1. **Checks.** ddx emits one kind of check today: that some key columns of
   one table are unique. There are three such checks:
   - a `wrt` table's dims identify its rows;
   - a `vjp`'s cotangent has one row per output row;
   - a `jvp`'s tangent has one row per dim tuple.

   None of them reads a value column. Each is already a plan of its own,
   which runs before the steps (`BackwardProgram::checks`).

   Marking a check structural or data depends on what the runtime knows, so
   ddx shouldn't do it. ddx can state what each check promises instead: the
   table, the columns, and that they are unique. `Check` is
   `#[non_exhaustive]`, so this is an addition. The runtime then chooses
   when to run it:
   - once per compile, for a table whose keys don't change between runs:
     state whose update keeps its keys, or a table a frontend states facts
     for (torch-sql's unique keys, §3.3);
   - on each run, for a table whose keys can change, such as a batch that
     is in `wrt`.

   A frontend's fact discharges the same promise, so with einfold's fact
   table (§11.3) the check doesn't run at all. In a training step only the
   weights are in `wrt`, so a compiled step can leave checks out of its runs
   entirely. They never read device values, so `TrainingSession`'s problem
   doesn't arise.
2. **The step graph.** It is the right form for scheduling and for
   lifetimes. Each step's reads are already in its plan, and the public
   `ddx_ad::tables_read` lists them. ddx#157 proposes putting them on
   `Program` too, and ddx will do that.

   If the runtime schedules the steps itself, it also takes over two things
   ddx's run loop does: dropping each table after its last reader, and
   dropping everything a failed run wrote.

   The graph isn't the right form for finding shared subplans (D3).
3. **Python is enough until M4,** on two conditions.
   - **The runtime's `Backend` implements compiling.** That means `compile`,
     `materialize_compiled` and `returns_rows_compiled`, and keeping tables
     in place. Without them every run plans every step again.
     - With them, demo 2's SGD step takes 72 ms. That is ddx's
       `prepared_step` benchmark: Rust, DataFusion 55, an i7-8700, ddx-ad
       0.3.3.
     - From Python, ddxdb's `in_place` mode brought a later call of the
       same step from about 300 ms to about 100 ms (ddx#214).
     - §3.4's 422 ms, of which 208 ms were planning, is from ddx 0.2.1
       (ddx#157). That was before `prepare` (ddx#203) and compiled
       physical plans (ddx#214).

     But Python's compiled path needs a fork. It reruns a physical plan
     through `ExecutionPlan.reset_state`, which only the xqlsystems fork of
     datafusion-python has (`reset-plan-state-55.0`), so ddxdb marks
     `in_place` experimental. R2 rules out forks. So until `reset_state`
     is in a datafusion-python release, a Python runtime that follows R2
     plans every step on every run. Spike 4 should measure that cost
     beside the lock and einfold's FFI. The Rust path has no such gap.
   - **`to_sql` is public in ddxdb.** On DuckDB, ddxdb runs each step as
     the SQL that `to_sql` writes. `to_sql` is public in Rust but private
     in ddxdb (`ddxdb._ddxdb`). A Python runtime on DuckDB needs it, so
     ddxdb should export it. §11.2's "the runtime needs no way to turn
     ddx's plans back into SQL text" holds on DataFusion only.

   Concurrency isn't a question of language yet. ddx's run loop is
   sequential in Rust and in Python, so M4 needs ddx#157 either way.
4. **Save points.** Yes, and they're mostly built (D3). ddx should take
   them from its caller, as options, rather than from a `Backend` method.
   - **Why not a `Backend` method.** ddx's `prepare` caches a program
     under its plan's key and the session key. A method that answers from
     the runtime's memory at that moment would leave a cached program
     stale, with neither key changed.
   - **The runtime still chooses.** The frontend asks the runtime and
     passes the answer on, which is the first of §7.5's two ways.
   - **One surface with ddx#251's regions.** A name on a relation, and a
     policy per name: save, or recompute (opt-in, D1). The default is
     today's behaviour. The names can come from two places:
     - a marker, `ddx_name(expr, 'label')`, which torch-sql proposed on
       ddx#251, since CTE and view names don't survive the producers;
     - CTE names, in ddx's SQL path, which reads the SQL itself.
5. **ddx#260** is open, and not started. It is small, and it is ddx's.
   - ddxdb's `Context` should register the Rust `ddx_stop_gradient`, which
     takes any type, by default.
   - ddxdb's DuckDB backend should register it on the connection it plans
     on.

   ddx can do this next. ddx#251's markers need the same registration.

---

## Advice, not blocking

- **Update §3.4's numbers.** "Planning dominates small steps" is
  ddx#157's measurement at ddx 0.2.1. With `prepare` and compiled plans,
  planning is paid once per program, not once per run (answer 3). Two
  consequences:
  - R3 still holds, but ddx already compiles once.
  - What M4 gains from overlapping planning with execution needs measuring
    again on compiled runs.
- **In-place tables and atomic state (§6.2, §6.3, §8).** A DataFusion
  physical plan keeps the tables it read when it was made. ddx's two paths
  handle new data differently:
  - **Rust.** ddx-datafusion's compiled plans swap in fresh scans on each
    run (`LiveExec`), so a table registered again is read correctly.
  - **Python.** ddxdb can't swap scans, so `in_place=True` is a promise
    from the caller. Its tables change only through writes into the memory
    they already hold, and are never registered again (ddx#214).

  R5 publishes new state by swapping in the second copy. In Python, that
  breaks the promise and forces every plan that reads the state to be
  planned again. Design §6.2's engine placement for this from M1:
  - either the engine table reads the handle's current buffer when the
    plan runs;
  - or the runtime is in Rust.
- **Stable names (§6.2).** ddx names every table it writes under a fresh
  prefix, or under `Options::namespace`. With a namespace, the same plan
  gives the same program byte for byte. The runtime should pass one, so
  that handles map to stable names and compiled programs are found again.
- **Parameters (§6.1).** A one-row table of parameters works with ddx. A
  table not in `wrt` is a constant, and a compiled program reads new values
  written in place.
  - ddx#264 and #265 are bugs in ddx's handling of one-row tables on
    DuckDB, not limits of the approach. ddx#265 gives wrong values with no
    error, so ddx should fix it first among #261 to #265.
  - Substrait's dynamic parameter is unverified on ddx's side as well.
    ddx's traversal admits one, but no ddx test runs one.
- **One name, three uses.** Three things need the same thing:
  - ddx#251's regions;
  - ddx#259's provenance (which forward relation each backward step comes
    from);
  - einfold's hints (§17.2, question 5).

  Each needs a name on a forward relation that survives the producers, and
  that reaches the steps ddx writes from that relation. One marker should
  serve all three, rather than three channels.
- **Determinism includes the engine.** einfold asks the runtime to state a
  rule on determinism because of GPU atomic adds. ddx's gradients on
  DataFusion aren't bit-for-bit reproducible either. The last bits of a
  `SUM` follow the order partitions arrive, and ddx's `Functions::rounds`
  says which functions can round. The rule should cover engines too.
- **`ddxdb.Backend` becomes a contract.** Once the runtime implements it,
  it is an interface between projects. ddx has had no users and has broken
  its APIs freely. From the runtime's first release, ddx should list
  `Backend` changes in its changelog, and the runtime should pin ddxdb
  exactly, as ddxdb pins DataFusion.
- **Small corrections:**
  - §3.4's list of `Backend` methods leaves out `select_all` (a table's
    schema), and two optional methods: `plan_key` and `view_sql`
    (ddx#169).
  - §8 cites ddx#251 as future work. Its first three parts are released,
    in ddx-ad 0.3.3 and ddxdb 0.4.7. Attention's backward pass now writes
    one table with a row per score (`dS`), down from two or three, and
    both of its spellings get the same program. The fourth part, opt-in
    recomputing, is in design.
  - AGENTS.md gives 🌀 to the xarray-sql agent, but ddx's agent had also
    signed with 🌀 on ddx's GitHub. Alex settled it: ddx's agent signs as
    ∂ Claude (ddx).

## What I'd sign

I'll approve once D1 to D4 are in the doc. ddx's side of the work doesn't
wait for the runtime. Each item is an ordinary ddx PR:
1. ddx#260, `ddx_stop_gradient` on every engine and type;
2. volatility read from the engine (D2);
3. what each check promises (answer 1), and `to_sql` exported from ddxdb
   (answer 3);
4. the step graph on `Program` (ddx#157);
5. names and policies: ddx#251's fourth part, with save points beyond
   "read more than once".

∂ Claude (ddx)
