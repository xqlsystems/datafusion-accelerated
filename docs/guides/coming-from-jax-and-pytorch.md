<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# Coming from PyTorch on CUDA, or JAX on TPU

_status_: Draft, for review. DataFusion Accelerated isn't built yet. Every
example says whether it runs **today** or is **proposed**.

This guide is for people who train models with PyTorch on NVIDIA GPUs, or
with JAX on TPUs, and want to know what the same work looks like in
DataFusion, and then in DataFusion Accelerated. It assumes you know your
own framework well, and introduces everything else.

- **[DataFusion](https://datafusion.apache.org/)** is a SQL query engine
  written in Rust, with a Python binding, designed to be extended. A
  program is a SQL query over tables, which DataFusion plans and runs.
- **DataFusion Accelerated** is a proposed extension of DataFusion that
  runs parts of a query on GPUs and other devices
  ([design](../design.md)). Hardware comes in through *integrations*,
  separate packages you mix in: one for CUDA, one for NCCL, one for TPUs,
  and so on.
- **[ddx](https://github.com/xqlsystems/ddx)** differentiates SQL. Its
  `grad` function, written inside a query, computes a gradient.
- **[einfold](https://github.com/xqlsystems/einfold)** is a query
  optimizer for DataFusion. It finds array computations in a query's plan,
  such as a sum of products over a join, which is a matrix product, and
  runs them with array algorithms instead of a join.

---

## The one idea that changes everything else

In PyTorch and JAX, an array is a dense block of memory with a shape. In
DataFusion, an array is a **table in long format**: one row per element,
with a column for each dimension's position and a column for the value.

```
a 2 × 3 matrix              the same matrix as a table

[[1., 2., 3.],              i  j  v
 [4., 5., 6.]]              0  0  1.0
                            0  1  2.0
                            0  2  3.0
                            1  0  4.0
                            ...
```

Operations on arrays become operations on tables:

| Array operation | Relational form |
|---|---|
| elementwise `a + b` | a join on the positions, then `a.v + b.v` |
| a reduction, `x.sum(dim=1)` | `GROUP BY i`, then `SUM(v)` |
| a matrix product, `x @ w` | a join on the shared dimension, then `GROUP BY` and `SUM` of the products |
| indexing, `wte[tokens]` | a join on the token's value |
| a mask | rows that aren't there |

The last row is the reason XQL does this. A table can be *partial*: an
ocean model's grid has no rows for land, so in this long format land is
never stored, computed or sent anywhere. A dense array has to hold it and
multiply it by zero. Long format is also slow where data is dense, so the
runtime trades some of this away for speed, as the end of this guide
explains.

This also means **you don't have to give up your framework.** If you'd
rather keep writing PyTorch, [torch-sql](https://github.com/xqlsystems/torch-sql)
turns PyTorch code on `device="sql"` into these queries for you (see the
end of the PyTorch half below). The rest of this guide shows the SQL
directly, so you can see what's underneath.

The table below maps the vocabulary. The design's
[supplement, S6.1](../supplement.md), has the full table.

| You know | In PyTorch | In JAX | In DataFusion Accelerated |
|---|---|---|---|
| an array | `torch.Tensor` | `jax.Array` | a table, in long format |
| a model's forward pass | Python calling ops | a traced function | a SQL query |
| autograd | `loss.backward()` | `jax.grad` | `grad(loss, v)` in SQL, through ddx |
| a compiled step | `torch.compile` | `jax.jit(...).lower().compile()` | `Program(...).lower().compile()` |
| putting data on a device | `.to("cuda")` | `jax.device_put` | memory tiers; `pin` in the schedule |
| a custom kernel | Triton, `torch.library` | Pallas, `jax.ffi` | a binding by label |
| distributing | DDP, FSDP | `NamedSharding`, `shard_map` | each table's distribution, declared in SQL |
| the hardware | CUDA, built in | PJRT plugins | integrations, mixed in |

---

## Part 1: coming from PyTorch on CUDA

The running example is linear regression, trained with SGD on one GPU:

```python
# PyTorch, today
import torch

x = torch.randn(1024, 16, device="cuda")
y = torch.randn(1024, device="cuda")
w = torch.zeros(16, device="cuda", requires_grad=True)
opt = torch.optim.SGD([w], lr=1e-2)

for step in range(100):
    loss = ((x @ w - y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
```

### 1.1 Tensors become tables

```python
# DataFusion, today
import numpy as np
import pyarrow as pa
from datafusion import SessionContext

X = np.random.randn(1024, 16)
Y = np.random.randn(1024)
i, k = np.meshgrid(np.arange(1024), np.arange(16), indexing="ij")

ctx = SessionContext()
ctx.from_pydict({"i": i.ravel(), "k": k.ravel(), "v": X.ravel()}, name="x")
ctx.from_pydict({"i": np.arange(1024), "v": Y}, name="y")
ctx.from_pydict({"k": np.arange(16), "v": np.zeros(16)}, name="w")
```

If your data is in an [Xarray](https://xarray.dev/) Dataset or in Zarr
files, [xarray-sql](https://github.com/xqlsystems/xarray-sql) registers it
for you, with the dimension names as columns:

```python
# xarray-sql, today
import xarray as xr
import xarray_sql as xql

ctx = xql.Context()
ctx.from_dataset("ocean", xr.open_zarr("om4.zarr"), chunks=dict(time=1))
```

### 1.2 The forward pass becomes a query

`x @ w - y`, squared and averaged, is a join and two aggregates:

```sql
-- DataFusion, today
WITH pred AS (
  SELECT x.i, SUM(x.v * w.v) AS v       -- x @ w
  FROM x JOIN w ON x.k = w.k
  GROUP BY x.i
)
SELECT AVG((pred.v - y.v) * (pred.v - y.v)) AS loss
FROM pred JOIN y ON pred.i = y.i
```

DataFusion plans this as a whole: it chooses the join's algorithm and
order, and you never write a loop.

### 1.3 `loss.backward()` becomes `grad` in SQL

With ddx, the gradient is part of the query. `grad(loss, v)`, in a query
over `w`, is the gradient of the CTE `loss` with respect to `w`'s column
`v`, at each of `w`'s rows:

```python
# ddx, today: ddxdb.Context is a SessionContext whose .sql() understands grad
from ddxdb import Context

ctx = Context()
# ... register x, y and w as above ...
GRAD = """
WITH pred AS (SELECT x.i, SUM(x.v * w.v) AS v FROM x JOIN w ON x.k = w.k GROUP BY x.i),
     loss AS (SELECT AVG((pred.v - y.v) * (pred.v - y.v)) AS l FROM pred JOIN y ON pred.i = y.i)
SELECT k, grad(loss, v) FROM w
"""
grads = ctx.sql(GRAD).to_arrow_table()       # columns k and v_grad
```

There is no tape to clear, so there's no `zero_grad()`: each query computes
its gradient from scratch. ddx writes the backward pass as SQL too, so it
runs wherever the forward pass runs.

[xarray-sql](https://github.com/xqlsystems/xarray-sql) does the same over
Xarray Datasets, with einfold planning the array work. Its `ml` extra
installs ddxdb, so its `Context.sql` understands `grad`. einfold is a
separate package that attaches as an extension. The released xarray-sql,
0.6.0, is built on DataFusion 54, and einfold needs DataFusion 55, so
until xarray-sql releases on 55
([xarray-sql#310](https://github.com/xqlsystems/xarray-sql/issues/310)),
install it from `main`:

```sh
pip install "xarray-sql[ml] @ git+https://github.com/xqlsystems/xarray-sql" "einfold[datafusion]"
```

Then the whole PyTorch loop above is a query and a loop:

```python
# xarray-sql, today, with the ml extra and einfold
import numpy as np
import xarray as xr
import xarray_sql as xql
from einfold.datafusion import EinFold

X = np.random.randn(1024, 16)
Y = np.random.randn(1024)

ctx = xql.Context()               # with the ml extra, .sql() understands grad
ctx.with_extensions(EinFold())    # einfold plans the matrix products
ctx.from_dataset("x", xr.Dataset({"v": (("i", "k"), X)}))   # columns i, k, v
ctx.from_dataset("y", xr.Dataset({"v": ("i", Y)}))
ctx.from_dataset("w", xr.Dataset({"v": ("k", np.zeros(16))}))

STEP = """
WITH pred AS (SELECT x.i, SUM(x.v * w.v) AS v FROM x JOIN w ON x.k = w.k GROUP BY x.i),
     loss AS (SELECT AVG((pred.v - y.v) * (pred.v - y.v)) AS l FROM pred JOIN y ON pred.i = y.i)
SELECT k, v - 1e-2 * grad(loss, v) AS v FROM w
"""
for step in range(100):
    w = ctx.sql(STEP).to_dataset()    # an xarray.Dataset with dimension k
    ctx.deregister_table("w")
    ctx.from_dataset("w", w)
```

`EXPLAIN` on the forward query shows einfold's node where DataFusion's
join and grouped sum were, `EinFoldExec: a,b * a -> b`, a matrix-vector
product. Each step runs twelve plans, since ddx writes the backward pass
as SQL queries of its own, and einfold took two of them in our run, with
DataFusion running the rest. Each step also reads `w` back into Python and
registers it again, which is the cost §1.4's program removes.

### 1.4 `opt.step()` becomes a program with state

Today you'd run the update as another query, and re-register `w`. That is
one round trip per step, which is what DataFusion Accelerated removes. A
*program* runs one or more queries many times, with *state* that persists
from run to run:

```python
# DataFusion Accelerated, proposed
from datafusion import SessionContext
from ddxdb.datafusion import Ddx                           # proposed: ddx as an extension
from datafusion_accelerated import Accelerator, Program
from datafusion_accelerated_cuda import Cuda

ctx = SessionContext().with_extensions(Ddx(), Accelerator(), Cuda(devices="local"))
# ... register x and y as above ...

STEP = """
WITH pred AS (SELECT x.i, SUM(x.v * w.v) AS v FROM x JOIN w ON x.k = w.k GROUP BY x.i),
     loss AS (SELECT AVG((pred.v - y.v) * (pred.v - y.v)) AS l FROM pred JOIN y ON pred.i = y.i)
SELECT k, v - $lr * grad(loss, v) AS v FROM w
"""
compiled = Program(ctx, STEP, state={"w": w0}, outputs=["loss"]).lower().compile()

for step in range(100):
    future = compiled.run(lr=1e-2)
print(future.result()["loss"])
```

- `state={"w": w0}` makes `w` a table that the program owns and keeps on the
  GPU. Each run replaces it with the query's result, but only once the run
  has succeeded, so a failed step leaves `w` as it was.
- `$lr` is a parameter: it can change every run without recompiling.
- `run` returns at once, like a CUDA launch. Only the final read waits.
- Adam, momentum and the rest are SQL too: their moments are more state
  tables. torch-sql lowers `torch.optim` into exactly that.

### 1.5 `.to("cuda")` becomes integrations and memory tiers

PyTorch builds CUDA in. DataFusion Accelerated builds none in. CUDA comes
from an integration you add, `Cuda()`, beside the accelerator itself. What
you get from it:

- **Where a table lives** is a *tier*, like a tensor's device: `CUDA`
  device memory, `CUDA_HOST` pinned memory, host memory, or files. The
  runtime chooses by default, and you can say otherwise:

  ```python
  # DataFusion Accelerated, proposed
  lowered = Program(ctx, STEP, state={"w": w0}).lower()
  schedule = lowered.schedule()
  schedule.table("x").pin("CUDA")          # keep x on the GPU across runs, like a resident tensor
  compiled = lowered.compile(schedule=schedule)
  ```

- **Every copy shows in the plan.** `EXPLAIN` shows each node's device and
  every copy between tiers, as nodes of their own. There's no hidden
  `.cpu()`.
- **Memory is shared.** The runtime draws GPU memory from a shared
  allocator, so it can share a GPU with PyTorch rather than compete with
  it.

### 1.6 A Triton kernel becomes a binding

You can attach your own kernel to any labelled part of a query. The SQL
stays the specification: if your kernel declines, or its assumptions fail
a check, the SQL runs instead, and checking mode can compare the two.

```sql
-- label the part of the query the kernel computes
WITH pred AS (SELECT x.i, ddx_name(SUM(x.v * w.v), 'predict') AS v
              FROM x JOIN w ON x.k = w.k GROUP BY x.i), ...
```

```python
# DataFusion Accelerated, proposed
import triton

runtime = Accelerator()

@runtime.kernel("predict", assumes=["complete", "no_nulls"])
def predict(args):
    x, w, out = args["x"], args["w"], args.out          # device arrays, named axes
    gemv[(triton.cdiv(x.extent("i"), 128),)](x, w, out, x.extent("k"), BLOCK=128)

ctx = SessionContext().with_extensions(Ddx(), runtime, Cuda(devices="local"))
```

You don't write a backward kernel, as you would with
`torch.autograd.Function`: ddx writes the backward pass in SQL, and you can
bind kernels to its steps too.

### 1.7 DDP and FSDP become declared distributions

In PyTorch, data parallelism and sharding are wrappers around the model.
Here they are declarations on tables, and the model's SQL doesn't change:

```sql
-- DataFusion Accelerated, proposed
CREATE EXTERNAL TABLE batch STORED AS ACCELERATED
  OPTIONS ('accelerated.distribution' 'hash(i)');       -- data parallelism, like DDP
CREATE EXTERNAL TABLE w STORED AS ACCELERATED
  OPTIONS ('accelerated.distribution' 'blocks(k)');     -- sharded weights and moments, like FSDP
```

The planner writes the collectives: a gradient's partial sums on each GPU,
then a shuffle by weight key and a final sum, which is a reduce-scatter.
With the NCCL integration, `Nccl()`, that runs as NCCL's `ReduceScatter`
over NVLink. Across machines, the runtime composes with
[datafusion-distributed](https://github.com/datafusion-contrib/datafusion-distributed).

### 1.8 `torch.compile` and CUDA graphs become `lower` and `compile`

A program is planned once, when you call `lower`, and fixed when you call
`compile`. After that, each run only executes. `explain()` shows what it
chose, and `report()` what it cost, as JSON an agent can read:

```python
# DataFusion Accelerated, proposed
lowered = Program(ctx, STEP, state={"w": w0}).lower()
print(lowered.explain())      # each step's plan: which node runs where, with which kernel, chosen by whom
compiled = lowered.compile()
compiled.run(lr=1e-2)
print(compiled.report())      # time per node, bytes moved per tier, declines
```

Nothing is chosen silently. If you override a choice and the runtime can't
obey it, compiling fails and says why.

### 1.9 Or keep writing PyTorch

torch-sql runs PyTorch code on a `device="sql"`, recording each operation
and running the whole step as SQL. It trains nanoGPT today, matching
PyTorch to 1e-11:

```python
# torch-sql, today (on DataFusion or DuckDB, on the CPU)
import torch
import torch_sql

with torch_sql.Environment(engine="datafusion"):
    model = torch.nn.Linear(16, 1, bias=False).to("sql")
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    for batch in loader:
        loss = ((model(batch.x).squeeze() - batch.y) ** 2).mean()
        loss.backward()                  # ddx writes the backward pass
        opt.step()
        opt.zero_grad()
```

Once DataFusion Accelerated exists, the same code runs its queries on the
GPU, with the whole step as one program.

---

## Part 2: coming from JAX on TPU

The running example is the same regression, in JAX, sharded over a TPU
slice:

```python
# JAX, today
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

mesh = Mesh(jax.devices(), ("data",))
x = jax.device_put(X, NamedSharding(mesh, P("data", None)))
y = jax.device_put(Y, NamedSharding(mesh, P("data")))
w = jnp.zeros(16)

def loss_fn(w, x, y):
    return jnp.mean((x @ w - y) ** 2)

@jax.jit
def step(w, x, y, lr):
    return w - lr * jax.grad(loss_fn)(w, x, y)

for _ in range(100):
    w = step(w, x, y, 1e-2)
```

### 2.1 Arrays become tables, and named axes become columns

JAX's named axes are already close to the relational form: a table's
columns *are* its named axes. The tables are those of 1.1.

### 2.2 `jax.grad` becomes `grad` in SQL

`jax.grad(loss_fn)` is a function transformation. ddx is the same idea
applied to a query: it transforms the query into one that computes the
gradient. See 1.3; the SQL is identical. Two differences from JAX:
- ddx differentiates with respect to *columns of tables*, named in the
  query, rather than positional arguments;
- the result is a table, with a gradient for each row of the table you
  differentiated with respect to.

### 2.3 `jit`, `lower` and `compile` keep their names

The program API copies JAX's on purpose:

| JAX | DataFusion Accelerated (proposed) |
|---|---|
| `jax.jit(step)` | `Program(ctx, STEP, state=..., outputs=...)` |
| `.lower(w, x, y)` | `.lower()` |
| `Lowered.as_text()` | `explain()` |
| `.compile()` | `.compile()` |
| `Compiled.cost_analysis()`, `.memory_analysis()` | `report()` |
| `donate_argnums=(0,)` | state tables, written in place once a run's checks pass |
| async dispatch, `block_until_ready()` | `run()` returns a future; `.result()` |

### 2.4 Sharding becomes declared distribution, and collectives come from the planner

In JAX you place arrays on a mesh with a `PartitionSpec`, and write
collectives in `shard_map` with `psum`. Here you declare each table's
distribution, and the planner derives everything between:

```sql
-- DataFusion Accelerated, proposed
CREATE EXTERNAL TABLE x STORED AS ACCELERATED
  OPTIONS ('accelerated.distribution' 'hash(i)');       -- P("data", None)
CREATE EXTERNAL TABLE w STORED AS ACCELERATED
  OPTIONS ('accelerated.distribution' 'replicated');    -- P(), on every chip
```

A `psum` over the data axis is what the planner writes for the gradient's
`SUM` over rows: partial sums on each chip, combined. You don't write it.
For a grid model, `tiles(lat, lon)` splits the grid into tiles, one per
chip. The planner maps the tiles onto the torus's axes, so each halo
exchange goes to a neighbouring chip, and a periodic longitude wraps around
the torus's own wraparound.

On a TPU, those collectives run as XLA's own. The TPU integration (proposed,
through PJRT, XLA's plugin interface to hardware) lowers each region of a
plan, with its collectives, into one XLA program, since TPU collectives
only exist inside compiled programs.

### 2.5 `checkpoint_name` and remat policies become labels and policies

JAX names values with `checkpoint_name` and chooses what to recompute with
`jax.checkpoint` policies. DataFusion Accelerated uses the same idea and
vocabulary:

```sql
-- a label in the query, like checkpoint_name
SELECT ddx_name(SUM(q.v * k.v), 'attn.scores') AS s FROM ...
```

```python
# DataFusion Accelerated, proposed
schedule.recompute(save_only_these_names("attn.out"))      # like jax.checkpoint_policies
```

Recomputing is never the runtime's own choice: you opt in, by label.

### 2.6 A Pallas kernel becomes a binding

A Pallas kernel binds to a label, as a Triton kernel does in 1.6. On a TPU,
it is compiled into the region's XLA program rather than launched on its
own:

```python
# DataFusion Accelerated, proposed
@runtime.kernel("attn.0")
def attention(args):
    q, k, v = (jax.dlpack.from_dlpack(args[n]) for n in ("q", "k", "v"))
    return flash_attention(q, k, v)       # a Pallas kernel
```

### 2.7 What's different on a TPU

- **Tables cross into DataFusion through the host.** Arrow, the columnar
  format DataFusion uses, can describe GPU memory but not TPU memory, so a
  TPU's tables stay on the chip until a query's result is read on the host.
- **The TPU integration comes later** than CUDA's, and needs a
  region-at-a-time execution stage that the CUDA path doesn't need first
  ([design §6.2](../design.md)).

---

## What's different, honestly

- **The SQL is the meaning.** Every fast path is checked against what
  DataFusion's own SQL computes. That makes your kernels safe to try, and
  wrong numbers loud.
- **Long format costs memory and time** where the data is dense. The
  runtime holds dense intermediates as *compact arrays* (coordinates once,
  values densely, masks for absent rows), which closed most of the gap in
  XQL's measurements. It doesn't close all of it yet. On Modded-NanoGPT, a
  124M-parameter GPT at context 1,024, the best SQL training step measured
  so far took 238.95 ms on an Apple M5 Ultra, against 185.40 ms for
  PyTorch on the same GPU: 1.29 times slower. That comparison is narrow.
  Both ran at batch 1 in float32, and PyTorch ran eagerly, without
  `torch.compile` and with unfused AdamW. A compiled PyTorch step is
  likely faster, and hasn't been measured. Memory is further off: an earlier run of the same workload
  peaked at 24.67 GB of process memory against PyTorch's 2.44 GB, though
  MLX and PyTorch count GPU memory differently
  ([supplement S1.1](../supplement.md)).
- **Absence is free only as rows.** Cells with no row, such as land in an
  ocean model, cost nothing in long format. A compact array stores every
  cell of its domain, so there an absent cell costs its storage and its
  arithmetic as a masked value, as in a dense array. The runtime holds a
  table compactly only when its domain is known and dense enough
  ([design §5.2](../design.md)), so a sparse table stays as rows.
- **Nothing here is built yet,** except DataFusion, ddx, xarray-sql and
  torch-sql, which run today on the CPU. The design's
  [milestones](../design.md) say what comes first.
