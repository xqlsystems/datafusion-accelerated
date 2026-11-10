<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike 4a: compiled plans from Python, without a fork

**Question** (design §12.3, spike 4a, D18). From Python, can the runtime's
Rust core plan a step once against a datafusion-python session, then run
it again on new rows, with no replanning? A physical plan that runs again
needs `ExecutionPlan.reset_state`, which only the xqlsystems fork of
datafusion-python has. The fork is ruled out by R2.

**Answer: yes, on the released datafusion-python, through the query planner
that `with_extensions` installs.** The runtime keeps its own copy of each
plan, in Rust, and resets and runs it there. Python never holds the plan, so
Python's missing `reset_state` doesn't matter. The fork isn't needed, and
neither is an upstream change.

On one of demo 2's steps, a layer norm and causal self-attention, a
compiled run took **21.5 ms**. Planning and running it each time in
datafusion-python took **48.9 ms**. The results matched datafusion-python's
own on every run, including after the batch changed its rows and its size.
No run called the planner.

That holds under four rules, which the spike found:

1. **Attach as a query planner, not as a physical optimizer rule.** A rule
   sees the plan as proxies into datafusion-python's library, and that
   library runs its operators on its own Tokio runtime, which datafusion-ffi
   doesn't pass to a rule. The first run of a plan kept by a rule aborted
   the process ("there is no reactor running"). A planner gets the plan as
   bytes, decoded in the runtime's library, so every operator is its own.
2. **The runtime's codec carries every table it can't serialize by
   reference.** datafusion-python serializes the logical plan to reach a
   foreign planner. A table registered from Python, such as one from
   `register_record_batches`, has no serialization. So once the planner is
   installed, every query over such a table fails, compiled or not, unless
   the runtime's codec carries it by an id that is valid within the process.
   With that, ordinary tables work, and the compiled run took 18.9 ms.
3. **A handle reports statistics, and never exact ones.** Without
   statistics, DataFusion made all 30 of the step's joins `Partitioned`,
   with 54 hash repartitions, and the compiled run took 37.4 ms. With row
   counts, it chose `CollectLeft` joins, as it does for in-memory tables.
   But a compiled plan outlives the rows it was planned on, and DataFusion
   answers `COUNT(*)` from exact statistics without reading a row. With
   exact statistics, a `COUNT(*)` compiled at 3 sequences returned the old
   count after the batch grew to 5. With inexact ones, it was right.
4. **Settings come in through `SessionConfig.with_extension`, before the
   session is built.** `with_extensions` installs codecs and planners only.
   `SET accelerated.label = ...` then reached the runtime's planner, through
   datafusion-ffi's extension options.

Status: done. Date: 2026-11-09. datafusion-python 55.0.0 from PyPI, which
is built on DataFusion 55.1.0, with the spike built on DataFusion 55.2.0.
Intel i7-8700, Linux. [results.md](results.md) has every run.

## How

[`src/lib.rs`](src/lib.rs) is a Python module, built with maturin, holding
miniatures of what M0 builds:
- `Accelerator`, an extension object for `with_extensions`. In phase one it
  contributes the runtime's codecs. In phase two it returns a planner that
  wraps the session's own, delegates to it, and keeps a copy of each
  physical plan. `compile(ctx, sql)` plans `sql` through the session and
  keeps the runtime's copy. `run(ctx, i)` takes a task context from the
  session's provider, resets the copy with DataFusion's
  `reset_plan_states`, and executes it.
- `HandleTable`, a table whose scan reads the handle's batches when it
  executes. Python can replace the batches between runs.
- `AcceleratedCodec`, which carries a handle and its scan by id, and,
  optionally, any other table or node by reference.
- `AcceleratedConfig`, the `accelerated` settings namespace.

It also implements the other attachment point, a physical optimizer rule
that keeps the plan, for the comparison in rule 1.

[`driver.py`](driver.py) builds the tables and runs the step. The step is
the first half of a block of demo 2, written as xarray-sql's
`benchmarks/ml/02_nanogpt.py` writes it: a layer norm, then the
projections, scores, softmax, heads and output projection of attention.
It runs at 2 sequences of 16 positions, 32 channels and 4 heads. The batch
`x` is a handle, replaced before each run, as a training loop replaces its
batch. The driver checks:
- that each compiled run matches the same SQL in a plain datafusion-python
  session, after new rows and after a batch of a different size;
- that no run calls the planner;
- that a compiled `COUNT(*)` stays right when the batch grows;
- that a `SET` made before and after compiling reaches the planner.

Then it times 100 runs of each path, with the batch replaced each time.

```sh
nix develop ~/git/ddx            # or any shell with cargo, maturin, uv and Python 3.12
uv venv .venv && . .venv/bin/activate && uv pip install "datafusion==55.0.0" pyarrow numpy
maturin develop --release
python driver.py planner 100     # every table a handle
python driver.py mixed 100       # the weights as datafusion-python's tables, carried by reference
python driver.py planner 100 nostats   # handles without statistics
python driver.py planner 3 exact       # the control: exact statistics
python driver.py rule 5          # the physical rule; aborts on the first run
```

## Results

| path | median ms per run |
|---|---|
| compiled through the runtime, every table a handle | 21.5 |
| compiled through the runtime, weights carried by reference | 18.9 |
| compiled through the runtime, handles without statistics | 37.4 |
| planned and run each time, datafusion-python alone | 41.0 to 48.9 |
| planned and run each time, through the runtime's planner | 132 to 171 |
| planning only, datafusion-python alone | 25.8 to 28.8 |
| planning only, through the runtime's planner | 114 to 131 |

Ranges are the same path measured in separate runs of the driver.

- **Planning costs more than running, at this size.** datafusion-python
  spent 26 to 29 ms of each 41 to 49 ms planning. Part of that is the SQL:
  DataFusion plans a CTE again at every read (xarray-sql#267), so the
  compiled plan holds 5 `power`s and 5 `sqrt`s where the SQL writes one
  of each.
- **Planning through the runtime's planner costs four or five times as
  much**, since each plan crosses the FFI four times: the logical plan
  into the runtime's library and back out to be planned, and the physical
  plan back in and out again.
  That is a cost of compiling, paid once per program, and of every
  uncompiled query in an accelerated session.
- **The functions in a compiled plan are proxies.** Each scalar function
  in the runtime's copy calls back into datafusion-python across the FFI.
  The same SQL compiled in a pure-Rust session of the spike's own, with no
  FFI, ran in 33.0 ms without statistics, against 40 to 45 ms through the
  proxies. So they cost something, but less than the missing statistics
  did.
- **Building a task context and resetting the plan are cheap**: 0.5 ms and
  0.15 ms per run.

## What it doesn't show

- **Speed at real sizes.** The step is small, so per-node overhead
  dominates. Demo 2 trains at larger sizes, and a full step has many more
  nodes.
- **ddx.** The step is a forward query, not a program from ddx. ddxdb's
  `Backend` from Python is the next thing to build on this path.
- **Lifetimes.** The registries of handles and of tables carried by
  reference only grow. The runtime must drop entries with the plans and
  sessions that hold them.
- **A memory pool.** A task context built across datafusion-ffi has a
  `RuntimeEnv` of the runtime's library, not the session's, so the
  runtime's reservations aren't counted in the session's pool. That
  matters for M1's device pools.
- **Another process.** Carrying tables by reference works only within one
  process, which is enough for one machine. Distribution needs real codecs.
- **User-defined functions** with the same names as built-ins. The spike
  left every function a proxy, so it didn't need to tell them apart.

## What changes in the design

- §5.1 gains rules 1, 2 and 4: from Python, the runtime attaches as a
  query planner through `with_extensions`, its codec carries what it can't
  serialize by reference, and its settings come in through
  `SessionConfig.with_extension`.
- §5.2 gains rule 3: runtime tables report inexact statistics.
- §12.3 marks spike 4a done. D18's upstream proposal for `reset_state` isn't
  needed.
- Two upstream requests become possible, neither needed for M0, each only
  with Alex's go-ahead:
  - datafusion-ffi could pass the host's runtime to a foreign physical
    optimizer rule, so that a plan a rule keeps can run;
  - datafusion-python could let `with_extensions` add settings namespaces,
    so the runtime attaches with one call.
