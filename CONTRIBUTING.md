<!--
SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors

SPDX-License-Identifier: Apache-2.0
-->

# Contributing

The design is in [docs/design.md](docs/design.md), and the rules for agents
and people working on the project are in [AGENTS.md](AGENTS.md).

## Tools

You need Rust 1.94 or newer, with clippy and rustfmt, and
[uv](https://docs.astral.sh/uv/) for Python. On Nix or NixOS,
`nix develop` (or `direnv allow`) gives a shell with all of them.

## Style

Rust follows the [Google Rust style guide](https://google.github.io/styleguide/rustguide.html).
`rustfmt.toml` holds its formatting settings, which need a nightly
rustfmt (`cargo +nightly fmt`; in the Nix shell, plain `cargo fmt` has
them). The workspace's lints in `Cargo.toml` hold the rules clippy can
check. The rest, such as naming each file after its main item and
explaining each assertion, are for review.

## Checks

CI runs these, and a pull request must pass them:

```sh
cargo +nightly fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
uvx --from 'reuse[charset-normalizer]' reuse lint
```

Run tests under a memory cap. DataFusion can exhaust memory planning a deep
query, which has crashed development machines. On Linux with systemd:

```sh
systemd-run --user --scope -q -p MemoryMax=8G cargo test --workspace
```

Don't use `ulimit -v` for this, because it makes DuckDB abort.

## Licensing

Every file carries an SPDX header, following the
[REUSE](https://reuse.software/) specification:

```sh
reuse annotate --copyright "Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors" \
  --year 2026 --license Apache-2.0 <file>
```

A file that can't carry a header, such as a lockfile, goes in `REUSE.toml`.
