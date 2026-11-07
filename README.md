# xql-runtime

XQL is a group of projects for computing on scientific arrays with
databases: arrays become tables, and array programs become SQL. This
project is a proposed layer of XQL that runs those programs. It would own
the memory that holds their tables, keep data on a device such as a GPU
between steps, choose kernels, schedule steps and, later, distribute them.

Nothing is built yet. The design, which also proposes the name, is in
[docs/design.md](docs/design.md). It grew out of
[xarray-sql#307](https://github.com/xqlsystems/xarray-sql/issues/307).
