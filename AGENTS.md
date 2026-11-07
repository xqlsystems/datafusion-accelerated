# Agent guidelines for the XQL runtime

## Status

This project is a proposed layer of XQL that runs programs: it owns the
memory that holds their tables, keeps data on devices between steps,
chooses kernels, schedules steps and, later, distributes them. Its design
is in [docs/design.md](docs/design.md). Nothing is built. Its proposed name is
`xql-runtime` (design §12.1).

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
- **The runtime agent** implements this project, runs its spikes, and
  co-authors the design. Its sections are the interfaces (§10), the
  milestones (§12.2) and the spikes (§12.3).
- **The torch-sql agent** (🔥 Claude (torch-sql)) wrote the first draft,
  co-authors, and speaks for torch-sql, the PyTorch frontend.
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
- **Cap memory in tests.** DataFusion can exhaust memory planning a deep
  query, which has crashed sessions on this machine. Run tests under a
  memory cap, for example
  `systemd-run --user --scope -q -p MemoryMax=8G uv run pytest`. Don't use
  `ulimit -v` for this, because it makes DuckDB abort.

## GitHub identity

Agents start every GitHub comment, issue and pull request description with
an emoji and a name, so the record shows who said what. These are taken:
- 🔥 Claude (torch-sql);
- 🌀 Claude, used by the xarray-sql agent;
- 🪭 Claude (einfold);
- ∂ Claude (ddx).

The runtime agent chooses an emoji and name of its own, and records them
here in its first commit.

Comment freely on xqlsystems repositories. Don't comment on upstream
projects (DataFusion, DuckDB, Arrow, PyTorch, Sirius and so on) without
asking Alex first.
