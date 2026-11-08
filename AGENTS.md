# Agent guidelines for DataFusion Accelerated

## Status

This project, DataFusion Accelerated (`datafusion-accelerated`), is an
extension of DataFusion, in Rust and Python, that runs DataFusion's
physical plans on GPUs and other devices, with XQL as its first user. It
keeps tables in device memory, runs kernels for array work and relational
operators, schedules steps, and distributes them through
datafusion-distributed. Its design is in [docs/design.md](docs/design.md).
Nothing is built.

Work proceeds in three stages: discussion, then a design doc, then
implementation. The project is in the design stage.

- **Spikes are allowed now.** The spikes listed in the design doc (§12.3)
  exist to settle its open questions. Keep each one small, keep it under
  `docs/spikes/`, and report what it found in the design doc, whichever way
  it went.
- **Milestone M0 and later wait for agreement.** Do not build the runtime
  itself until the design's proposals are settled in its decision log and
  Alex agrees to start.

## Who does what

- **Alex Merose** owns the project and reviews throughout.
- **The xarray-sql agent** (🌀 Claude) is the technical lead and lead
  author of the design. It decides scope, and what moves into the runtime
  from other projects.
- **The project's agent** (⚡ Claude (datafusion-accelerated)) implements
  this project, runs its spikes, and co-authors the design. Its sections
  are the interfaces (§10), the milestones (§12.2) and the spikes (§12.3).
  It began as the torch-sql agent, which wrote the first draft.
- **The torch-sql agent** (🔥 Claude (torch-sql)) speaks for torch-sql, the
  PyTorch frontend.
- **The einfold and ddx agents** review the sections that touch their
  projects.

Where authors or reviewers disagree, record the disagreement in the design
doc's open questions, with each position stated, until it is settled and
moves to the decision log. Do not overwrite another author's position.

## Read before starting

The supplement's S1 summarizes the evidence. Its sources are:
- [xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307),
  where the runtime was proposed;
- the nanoGPT performance PRs:
  - [xarray-sql#302](https://github.com/xqlsystems/xarray-sql/pull/302),
    [#303](https://github.com/xqlsystems/xarray-sql/pull/303) and
    [#305](https://github.com/xqlsystems/xarray-sql/pull/305);
  - [einfold#156](https://github.com/xqlsystems/einfold/pull/156) and
    [#157](https://github.com/xqlsystems/einfold/pull/157);
- [ddx#157](https://github.com/xqlsystems/ddx/issues/157), on running a
  program's steps concurrently, and `ddxdb.Backend` in ddxdb's
  `__init__.py`, the interface ddx runs programs through;
- torch-sql's [design doc](https://github.com/xqlsystems/torch-sql/blob/main/docs/design.md):
  - §5, on XQL's layers;
  - §9.6, on cut points;
  - §12, on the runtime contract it expects.

## The design and its supplement

[docs/design.md](docs/design.md) says what to build and how: rules,
interfaces, milestones with their exit criteria, and spikes. Keep it
prescriptive. Evidence, measurements, rejected alternatives and the
reasoning behind a rule go in [docs/supplement.md](docs/supplement.md),
and the design cites them by section ("(S3.4)"). When a rule changes,
update both.

## Writing rules

These apply to the design doc and to all public-facing writing (README,
docs, docstrings, issues, PR descriptions).

- **Introduce every concept before you use it.** When you mention a
  project, term or idea for the first time, first say what it is in a
  sentence or two. Examples include XQL, ddx, einfold, DataFusion,
  Substrait, compact arrays and the C Device Data Interface. Don't assume
  the reader already knows.
- When a later section depends on an earlier concept, link back to where it
  was introduced.
- Write plain prose. Don't use em dashes, and don't use evaluative words
  where a measurement would do.
- Every number comes with its source: a PR, an issue, a spike, or a
  benchmark with its hardware and versions.

## Engineering rules

- **Public interfaces only.** Depend on neighbouring projects' released,
  public APIs. No private imports across repositories, and no forks. When
  the runtime needs something from ddx, einfold, xarray-sql or an engine,
  the change goes in that project, through an issue or a PR there.
- **The SQL is the meaning** (principle R1 of the design). An engine's
  result is the reference. Test every substituted implementation (a kernel,
  an array program, a GPU path) against it, on more than one engine where
  it can run. torch-sql found silent disagreements between DataFusion and
  DuckDB this way.
- **One job** (principle R11). The runtime owns tables' lifetimes,
  placement, the schedule and bindings, and builds no compiler, allocator
  or communication library. If a feature could be a library of its own,
  used without the runtime, make it one, or use one that exists.
- **Vanilla DataFusion, plus nodes** (principle R11). Extend DataFusion
  only through its own extension points: builder methods, settings,
  `RuntimeEnv`, physical rules, `ExecutionPlan` nodes, metrics and codecs.
  Attach through the builder in Rust and `with_extensions` in Python, never
  through an installer or a subclass of `SessionContext`.
- **Rust and Python** (principle R12). Build every capability in Rust and
  bind it to Python. A Rust program must never need Python.
- **The core delegates to hardware** (principle R13). The core depends on
  no hardware or vendor library: no CUDA, ROCm, Metal, PJRT, NCCL or
  libcudf. Every integration is a separate crate or Python package that
  implements the core's interfaces (design §5.8), and the core's tests run
  on its CPU reference implementations, with no GPU.
- **No hidden decisions** (principle R10). Every physical choice the
  runtime makes goes in the schedule, with who made it. An override is
  binding, and one the runtime can't obey is an error, never ignored.
- **Cap memory in tests.** DataFusion can exhaust memory planning a deep
  query, which has crashed sessions on this machine. Run tests under a
  memory cap, for example
  `systemd-run --user --scope -q -p MemoryMax=8G uv run pytest`. Don't use
  `ulimit -v` for this, because it makes DuckDB abort.

## GitHub identity

Agents start every GitHub comment, issue and pull request description with
an emoji and a name, so the record shows who said what. These are taken:
- ⚡ Claude (datafusion-accelerated), this project's agent;
- 🔥 Claude (torch-sql);
- 🌀 Claude, used by the xarray-sql agent;
- 🪭 Claude (einfold);
- ∂ Claude (ddx).

Comment freely on xqlsystems repositories. Don't comment on upstream
projects (DataFusion, datafusion-contrib and datafusion-distributed, DuckDB,
Arrow, PyTorch, Sirius and so on) without asking Alex first. File issues
that need another project's work here, labelled `upstream`.
