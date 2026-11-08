# Spike 7: can handle batches carry device data between DataFusion nodes?

**Question** (design §12.3, spike 7). DataFusion's record batches live in
host memory, so the design's device nodes pass *handle batches*: record
batches whose one column holds handles to buffers in a device's memory
(design §5.3). Do handle batches survive DataFusion's planner and its
optimizer rules, which insert, move and rewrite operators? Or must device
work stay inside single nodes?

**Answer: they survive, under three rules.** No DataFusion operator ever
read a handle batch as data, and all 136 queries gave the same rows as
without the device nodes. But DataFusion's optimizer does put its own
operators *between* device nodes unless the device nodes and the
runtime's rule are set up to prevent it, and when it does, handles move to
partitions (standing for devices) that don't own their buffers. The rules:

1. **Device nodes report that they don't benefit from more input
   partitions** (`benefits_from_input_partitioning` returns `false`).
   DataFusion's default is `true`. With it, `EnforceDistribution` inserted a
   round-robin `RepartitionExec` between `ToDeviceExec` and the first kernel
   in every one of 28 queries at 4 target partitions, and kernels then read
   144 buffers that another partition had written. With `false`, the
   repartition moved above `ToHostExec`, onto the host, and no buffer was
   read across partitions.
2. **The runtime's placement rule runs after DataFusion's rules.** A rule
   added with `SessionStateBuilder::with_physical_optimizer_rule` runs after
   all of DataFusion's defaults, `SanityCheckPlan` included. Placed there,
   nothing rewrote a device segment, whatever the knobs. Placed first, the
   later rules inserted the same round-robin repartition between device
   nodes.
3. **Inside a segment, the runtime owns every exchange.** A handle batch's
   schema hides the logical columns, so a device node can't state a
   partitioning or an ordering over them, and DataFusion can't reason about
   the data inside a segment. If a device operator needs its input
   partitioned by a key, as a device hash aggregate would, the exchange is
   the runtime's own node, placed by its rule, never DataFusion's
   `RepartitionExec`, which would hash the handles. `ToHostExec` restates
   the logical properties for the host plan above it.

Status: done. Date: 2026-11-07. DataFusion 55.2.0, Rust 1.99, on an x86-64
Linux machine. The device is a mock, so this measures plan mechanics, not
speed.

## How

[`src/main.rs`](src/main.rs) builds three nodes like the design's:
- `ToDeviceExec` copies each host batch into a pool and emits one handle;
- `KernelExec` reads each handle's buffer, computes, writes a new buffer;
- `ToHostExec` reads each buffer back as an ordinary batch.

The pool is host memory, and each buffer records the partition that wrote
it, which stands for a device. A kernel that reads a buffer written by
another partition counts as a cross-device read.

Two experiments let DataFusion build real plans around the nodes:
- **E1, a segment at the leaf.** A table whose scan is `ToHostExec(Kernel(+1,
  Kernel(×2, ToDeviceExec(scan))))`, queried with SQL through DataFusion's
  planner and its default physical optimizer rules. Seven queries: a
  `GROUP BY`, a filter with a sort and limit, a projection, `COUNT(*)`, a
  self join, a window function and a `UNION ALL`. Each runs at 1 and 4
  target partitions, over 1 and 3 source partitions, with
  `benefits_from_input_partitioning` and `maintains_input_order` each on and
  off: 112 runs. Each result is compared with the same query over a plain
  table holding `v * 2 + 1`.
- **E2, a segment placed mid-plan.** A rule wraps the input of the first
  partial aggregate in a segment with an identity kernel, as the runtime's
  rule would place device nodes inside a host plan. The rule runs first or
  last among DataFusion's rules, at 1 and 4 target partitions, with
  `benefits_from_input_partitioning` on and off, over three queries: 24
  runs, compared with the plan without the rule.

After each run, the spike walks the plan and records any DataFusion
operator whose output is a handle batch (it sits between device nodes), and
any whose input is a handle batch while its output isn't (it would read
handles as data).

Run it with `cargo run --release` in this directory. `--plans` prints the
plans below. [`results.md`](results.md) has every run.

## Results

| Experiment | Setting | Runs | DataFusion operator between device nodes | Cross-device reads | Results match |
|---|---|---|---|---|---|
| E1 | 1 target partition | 56 | none | 0 | all |
| E1 | 4 target partitions, `benefits_from_input_partitioning = true` (DataFusion's default) | 28 | round-robin `RepartitionExec` in all 28 | 144 | all |
| E1 | 4 target partitions, `benefits_from_input_partitioning = false` | 28 | none | 0 | all |
| E2 | rule first, 4 target partitions, default | 3 | round-robin `RepartitionExec` in 2 | 12 | all |
| E2 | rule first, any other setting | 9 | none | 0 | all |
| E2 | rule last, every setting | 12 | none | 0 | all |

In no run did a DataFusion operator read a handle batch as data, and no
buffer was left in the pool after a query. `maintains_input_order` changed
nothing: no `SortExec` was ever pushed into a segment, since a handle
batch has no column to sort by. No filter, projection or limit was pushed
into a segment either, because DataFusion's defaults for an unknown node
are conservative.

With DataFusion's default, the repartition lands inside the segment:

```
AggregateExec: mode=Partial, gby=[k@0 as k], aggr=[sum(d.v)]
  ToHostExec
    KernelExec: Plus1
      KernelExec: Times2
        RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3
          ToDeviceExec
            DataSourceExec: partitions=3, partition_sizes=[3, 3, 3]
```

With `benefits_from_input_partitioning = false`, it lands above the segment,
on the host:

```
AggregateExec: mode=Partial, gby=[k@0 as k], aggr=[sum(d.v)]
  RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3
    ToHostExec
      KernelExec: Plus1
        KernelExec: Times2
          ToDeviceExec
            DataSourceExec: partitions=3, partition_sizes=[3, 3, 3]
```

## What it doesn't show

- **Speed, and real device memory.** The pool is host memory.
- **Other rules.** Only DataFusion 55.2's default physical optimizer rules
  ran. datafusion-distributed's planner and einfold's rules are the next to
  check, since both rewrite physical plans.
- **A device operator that needs its input partitioned by key.** Rule 3
  follows from the schema, but no device exchange was built here.
- **Cancellation.** Every query ran to completion, so no buffer was ever
  orphaned by a stream dropped halfway.
- **Statistics and metrics.** A handle batch has one row per buffer, so
  DataFusion's row counts for operators that forward handle batches mean
  buffers, not rows. Device nodes should report logical statistics.
- **Serialization.** No codec was written for datafusion-proto.

## What changes in the design

§5.3 gains the three rules above, and P-Q's open question is answered:
device plans work as written. The follow-ups are datafusion-distributed's
and einfold's rules, a device exchange, cancellation and statistics, each
small enough for the start of M2.
