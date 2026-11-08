# DataFusion Accelerated

DataFusion Accelerated (`datafusion-accelerated`) runs
[Apache DataFusion](https://datafusion.apache.org/)'s physical plans on GPUs
and other devices. It's an extension of DataFusion, in Rust and Python,
that:
- keeps tables in device memory between operators and between runs;
- runs device kernels for any node of a plan, whether array work or
  DataFusion's own relational operators, with DataFusion's operators as the
  fallback and the reference;
- composes with
  [datafusion-distributed](https://github.com/datafusion-contrib/datafusion-distributed)
  to go beyond one machine.

It's an interstitial layer: it makes CUDA, ROCm, Metal, TPUs and the
libraries built on them easy to integrate, and integrates none of them
itself. Each integration is a separate crate or package, mixed in as an
extension of its own.

Its first user is [XQL](https://github.com/xqlsystems), whose projects
compute on scientific arrays with databases and train models such as the
ocean emulator Samudra in SQL.

Nothing is built yet. The design is in [docs/design.md](docs/design.md),
and the evidence and reasoning behind it in
[docs/supplement.md](docs/supplement.md).

DataFusion Accelerated is not part of the Apache DataFusion project.
"Apache DataFusion" is a trademark of the Apache Software Foundation.
