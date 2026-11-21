# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike 4a's driver: compile one of demo 2's steps through the runtime from
Python, run it again on new rows, and time it against planning every run.

The step is the first half of a block of demo 2 (xarray-sql's
``benchmarks/ml/02_nanogpt.py``): a layer norm, then causal self-attention,
written as that demo writes them. The batch ``x`` is a runtime handle table,
replaced between runs as a training loop replaces its batch. The weights are
ordinary tables registered with datafusion-python.

Two ways to attach, chosen by the first argument:
- ``rule``: a physical optimizer rule, added after DataFusion's own, that
  keeps the finished plan;
- ``planner``: a query planner for ``with_extensions`` that wraps the
  session's. datafusion-python serializes the logical plan to reach it, so
  every table must be serializable: here, every table is a handle.

- ``mixed``: the planner, with the weights left as datafusion-python's own
  tables, which the runtime's codec carries by reference within the process.

Run it with ``python driver.py planner 100`` after ``maturin develop --release``.
"""

import json
import statistics
import sys
import time

import numpy as np
import pyarrow as pa
from datafusion import SessionConfig, SessionContext

import s04a

SEQ, CONTEXT, EMBED, HEADS = 2, 16, 32, 4
HEAD = EMBED // HEADS
MODE = sys.argv[1] if len(sys.argv) > 1 else "planner"   # "planner", "mixed" or "rule"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 100

# --- demo 2's SQL, as 02_nanogpt.py writes it, for layer 0 -------------------

STEP = f"""
WITH
ln_mean AS (
  SELECT seq, pos, AVG(val) AS mean FROM x GROUP BY seq, pos),
ln_var AS (
  SELECT x.seq, x.pos, AVG(power(x.val - m.mean, 2)) AS var
  FROM x x JOIN ln_mean m ON x.seq = m.seq AND x.pos = m.pos
  GROUP BY x.seq, x.pos),
ln AS (
  SELECT x.seq, x.pos, x.e, (x.val - m.mean) / sqrt(v.var + 1e-5) * g.w AS val
  FROM x x
  JOIN ln_mean m ON x.seq = m.seq AND x.pos = m.pos
  JOIN ln_var v ON x.seq = v.seq AND x.pos = v.pos
  JOIN ln_gain g ON x.e = g.e),
{"".join(f'''
l0_{n} AS (
  SELECT x.seq, x.pos, p.h, p.c, SUM(x.val * p.w) AS val
  FROM ln x JOIN w{n} p ON x.e = p.e
  WHERE p.layer = 0
  GROUP BY x.seq, x.pos, p.h, p.c),''' for n in "qkv")}
l0_scores AS (
  SELECT q.seq, q.h, q.pos AS i, k.pos AS j,
         SUM(q.val * k.val) / {np.sqrt(HEAD)} AS s
  FROM l0_q q JOIN l0_k k ON q.seq = k.seq AND q.h = k.h AND q.c = k.c
  WHERE q.pos >= k.pos
  GROUP BY q.seq, q.h, q.pos, k.pos),
l0_weights AS (
  SELECT seq, h, i, j,
         exp(s - MAX(s) OVER (PARTITION BY seq, h, i)) AS p
  FROM l0_scores),
l0_norm AS (
  SELECT seq, h, i, SUM(p) AS z FROM l0_weights GROUP BY seq, h, i),
l0_heads AS (
  SELECT a.seq, a.i AS pos, a.h, v.c, SUM(a.p / n.z * v.val) AS val
  FROM l0_weights a
  JOIN l0_v v ON a.seq = v.seq AND a.h = v.h AND a.j = v.pos
  JOIN l0_norm n ON a.seq = n.seq AND a.h = n.h AND a.i = n.i
  GROUP BY a.seq, a.i, a.h, v.c),
out AS (
  SELECT a.seq, a.pos, p.e, SUM(a.val * p.w) AS val
  FROM l0_heads a JOIN wo p ON a.h = p.h AND a.c = p.c
  WHERE p.layer = 0
  GROUP BY a.seq, a.pos, p.e)
SELECT seq, pos, e, val FROM out ORDER BY seq, pos, e
"""

# --- tables ------------------------------------------------------------------

rng = np.random.default_rng(0)


def grid(**dims):
    """Long-format coordinates for every cell of the named dimensions."""
    names = list(dims)
    mesh = np.meshgrid(*(np.arange(n) for n in dims.values()), indexing="ij")
    return {n: m.ravel().astype(np.int64) for n, m in zip(names, mesh)}


def batch(seq=SEQ):
    cols = grid(seq=seq, pos=CONTEXT, e=EMBED)
    cols["val"] = rng.standard_normal(len(cols["e"]))
    return pa.table(cols)


WEIGHTS = {"ln_gain": pa.table({**grid(e=EMBED), "w": rng.uniform(0.5, 1.5, EMBED)})}
for n in "qkv":
    cols = grid(layer=1, e=EMBED, h=HEADS, c=HEAD)
    cols["w"] = rng.standard_normal(len(cols["e"])) / np.sqrt(EMBED)
    WEIGHTS[f"w{n}"] = pa.table(cols)
cols = grid(layer=1, h=HEADS, c=HEAD, e=EMBED)
cols["w"] = rng.standard_normal(len(cols["e"])) / np.sqrt(EMBED)
WEIGHTS["wo"] = pa.table(cols)


def plain_session(x):
    """The reference: datafusion-python alone, with no runtime."""
    ctx = SessionContext()
    ctx.register_record_batches("x", [x.to_batches()])
    for name, t in WEIGHTS.items():
        ctx.register_record_batches(name, [t.to_batches()])
    return ctx


def same(a: pa.Table, b: pa.Table) -> bool:
    if a.num_rows != b.num_rows or a.schema.names != b.schema.names:
        return False
    for name in ("seq", "pos", "e"):
        if a[name] != b[name]:
            return False
    return np.allclose(a["val"].to_numpy(), b["val"].to_numpy(), rtol=1e-12, atol=1e-12)


# --- the spike ---------------------------------------------------------------

# "stats" (inexact, the default), "exact" (the control) or "nostats".
STATS = sys.argv[3] if len(sys.argv) > 3 else "stats"
s04a.Accelerator.set_statistics(STATS != "nostats", exact=STATS == "exact")
report = {"datafusion-python": __import__("datafusion").__version__, "mode": MODE, "handle statistics": STATS}

s04a.Accelerator.set_by_reference(MODE == "mixed")
acc = s04a.Accelerator()
config = SessionConfig().with_extension(s04a.AcceleratedConfig())
if MODE == "rule":
    ctx = SessionContext(config)
    ctx.add_physical_optimizer_rule(acc)
else:
    ctx = SessionContext(config).with_extensions(acc)

x0 = batch()
x = s04a.HandleTable(x0)
ctx.register_table("x", x)
for name, t in WEIGHTS.items():
    if MODE in ("rule", "mixed"):
        ctx.register_record_batches(name, [t.to_batches()])
    else:
        ctx.register_table(name, s04a.HandleTable(t))

ctx.sql("SET accelerated.label = 'first'").collect()
index = acc.compile(ctx, STEP)
report["planner calls after compile"] = acc.planner_calls()
report["settings at compile"] = acc.settings_seen()[-1]
report["plan"] = acc.explain(index)

# 1. The compiled plan gives the session's own answer.
out, _ = acc.run(ctx, index)
report["run 1 matches a plain session"] = same(pa.table(out), plain_session(x0).sql(STEP).to_arrow_table())

# 2. New rows in the handle; the same compiled plan, no planning.
calls = acc.planner_calls()
x1 = batch()
x.replace(x1)
out, _ = acc.run(ctx, index)
report["run 2, new rows, matches"] = same(pa.table(out), plain_session(x1).sql(STEP).to_arrow_table())

# 3. A batch of a different size.
x2 = batch(seq=3)
x.replace(x2)
out, _ = acc.run(ctx, index)
report["run 3, 3 sequences, matches"] = same(pa.table(out), plain_session(x2).sql(STEP).to_arrow_table())
report["planner calls during runs 1 to 3"] = acc.planner_calls() - calls
COUNT = "SELECT COUNT(*) AS n FROM x"
count = acc.compile(ctx, COUNT)
calls = acc.planner_calls()
x.replace(batch(seq=5))
n = pa.table(acc.run(ctx, count)[0])["n"][0].as_py()
report["COUNT(*) compiled at 3 sequences, run at 5, is right"] = n == 5 * CONTEXT * EMBED
report["planner calls during the COUNT run"] = acc.planner_calls() - calls

# 4. Settings changed after compile reach the next plan.
ctx.sql("SET accelerated.label = 'second'").collect()
ctx.sql("SELECT 1").collect()
report["settings after SET"] = acc.settings_seen()[-1]

# 5. Timing, over RUNS runs each, with the batch replaced before every run.
x.replace(x0)
plain = plain_session(x0)


def median_ms(f):
    times = []
    for _ in range(RUNS):
        t = time.perf_counter()
        f()
        times.append((time.perf_counter() - t) * 1e3)
    return round(statistics.median(times), 2)


def compiled_run():
    x.replace(x0)
    pa.table(acc.run(ctx, index)[0])


report["timing, median ms per run"] = {
    "compiled through the runtime (replace + run + read)": median_ms(compiled_run),
    "planned every run, datafusion-python alone": median_ms(lambda: plain.sql(STEP).to_arrow_table()),
    "planned every run, through the runtime's planner": median_ms(lambda: ctx.sql(STEP).to_arrow_table()),
    "planning only, datafusion-python alone": median_ms(lambda: plain.sql(STEP).execution_plan()),
    "planning only, through the runtime's planner": median_ms(lambda: ctx.sql(STEP).execution_plan()),
}
report["handle scans executed"] = s04a.Accelerator.scans()
report["codec events"] = s04a.Accelerator.codec_events()
print(json.dumps(report, indent=2))
