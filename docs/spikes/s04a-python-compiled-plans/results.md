# Spike 4a results

Raw results, written from the driver's JSON reports. The [README](README.md) reads them.

Intel i7-8700 (6 cores, 12 threads), Linux 6.12, datafusion-python 55.0.0 from PyPI (built on DataFusion 55.1.0), the spike built on DataFusion 55.2.0 with rustc 1.98.1, pyarrow 26.0.1, numpy 2.5.4. Medians of 100 runs, except the exact-statistics control, which ran 3.

## The planner hook, every table a handle

| check | inexact statistics (the default) | no statistics | exact statistics (the control) |
|---|---|---|---|
| planner calls after compile | 2 | 2 | 2 |
| settings at compile | ffi: strict=false label=first | ffi: strict=false label=first | ffi: strict=false label=first |
| run 1 matches a plain session | True | True | True |
| run 2, new rows, matches | True | True | True |
| run 3, 3 sequences, matches | True | True | True |
| planner calls during runs 1 to 3 | 0 | 0 | 0 |
| COUNT(*) compiled at 3 sequences, run at 5, is right | True | True | False |
| planner calls during the COUNT run | 0 | 0 | 0 |
| settings after SET | ffi: strict=false label=second | ffi: strict=false label=second | ffi: strict=false label=second |
| handle scans executed | 6294 | 6294 | 279 |

## Timing, median ms per run

| path | inexact statistics | no statistics |
|---|---|---|
| compiled through the runtime (replace + run + read) | 21.49 | 37.4 |
| planned every run, datafusion-python alone | 48.9 | 41.09 |
| planned every run, through the runtime's planner | 171.4 | 169.59 |
| planning only, datafusion-python alone | 28.75 | 25.78 |
| planning only, through the runtime's planner | 130.56 | 125.2 |

## The planner hook, with datafusion-python's tables carried by reference

The weights left as datafusion-python's in-memory tables, which the runtime's codec carries by reference within the process; the batch `x` a handle, with inexact statistics. Medians of 100 runs.

| check | result |
|---|---|
| planner calls after compile | 2 |
| settings at compile | ffi: strict=false label=first |
| run 1 matches a plain session | True |
| run 2, new rows, matches | True |
| run 3, 3 sequences, matches | True |
| planner calls during runs 1 to 3 | 0 |
| COUNT(*) compiled at 3 sequences, run at 5, is right | True |
| planner calls during the COUNT run | 0 |
| settings after SET | ffi: strict=false label=second |
| handle scans executed | 4061 |

| path | median ms |
|---|---|
| compiled through the runtime (replace + run + read) | 18.94 |
| planned every run, datafusion-python alone | 41.04 |
| planned every run, through the runtime's planner | 132.2 |
| planning only, datafusion-python alone | 25.75 |
| planning only, through the runtime's planner | 114.45 |

| codec event | count |
|---|---|
| logical decode: handle table | 8042 |
| logical decode: table by reference | 4422 |
| logical encode: handle table | 8042 |
| logical encode: other table | 4422 |
| logical encode: table by reference | 4422 |
| physical decode: handle scan | 8042 |
| physical encode: handle scan | 8042 |

The compiled plan: 30 `HashJoinExec: mode=CollectLeft`, 20 `HandleScanExec`s and 11 `DataSourceExec`s, the in-memory tables' scans, which DataFusion's own serialization carried with their rows.

## What crossed the codec, in the run with inexact statistics

| event | count |
|---|---|
| logical decode: handle table | 12464 |
| logical encode: handle table | 12464 |
| physical decode: handle scan | 12464 |
| physical encode: handle scan | 12464 |

## Break-down of one compiled run, without statistics

From a separate script on the same session, 30 runs: building the task context from the session's provider took 0.5 ms, resetting the plan 0.15 ms, executing it 40 to 45 ms. Every scalar function in the compiled plan was a proxy across datafusion-ffi: `power` 5 times, `sqrt` 5 and `exp` 2. The same SQL compiled in a session of the spike's own, over the same handles with no FFI, took 33.0 ms per compiled run and 39.2 ms to plan.

Join strategies in the compiled plan, without statistics: 30 `HashJoinExec: mode=Partitioned` and 54 hash `RepartitionExec`s. In datafusion-python's own plan over in-memory tables: 30 `mode=CollectLeft` and 26 hash `RepartitionExec`s.

## The physical-rule hook

Compiling worked: the rule received the finished plan and kept it. The first run aborted the process:

```
thread '<unnamed>' panicked at datafusion-common-runtime-55.2.0/src/common.rs:48:21:
there is no reactor running, must be called from the context of a Tokio 1.x runtime
thread caused non-unwinding panic. aborting.
```

## The planner hook, with datafusion-python's in-memory tables and nothing carried by reference

Planning failed before any run:

```
Error serializing custom table ... No installed extension codec handled a table provider: ... not the runtime's table: ln_gain; This feature is not implemented: LogicalExtensionCodec is not provided
```
