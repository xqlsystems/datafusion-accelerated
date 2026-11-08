# xql-runtime

XQL is a group of projects for computing on scientific arrays with
databases: arrays become tables, and array programs become SQL. This
project is a proposed layer of XQL that runs those programs on GPUs and
other devices. It is an extension of
[Apache DataFusion](https://datafusion.apache.org/), in Rust and Python,
that keeps tables on a device between steps, runs kernels for the array
work in a query, and composes with
[datafusion-distributed](https://github.com/datafusion-contrib/datafusion-distributed)
to go beyond one machine.

Nothing is built yet. The design is in [docs/design.md](docs/design.md),
and the evidence and reasoning behind it in
[docs/supplement.md](docs/supplement.md). It grew out of
[xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307).
