# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

# A development shell for Nix and NixOS:
#
#   nix develop        # or, with direnv, `direnv allow` once, then `cd` in
#
# It provides what CONTRIBUTING.md asks for. CI doesn't use Nix.
{
  description = "DataFusion Accelerated: DataFusion's physical plans on GPUs and other devices";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            # Rust. nixpkgs' toolchain is newer than the 1.94 minimum in Cargo.toml.
            cargo
            rustc
            clippy
            # rustfmt.toml's Google style settings are nightly-only; this is
            # the same rustfmt with them enabled.
            (rustfmt.override { asNightly = true; })
            rust-analyzer
            # Python: uv manages the virtual environments, and maturin builds
            # the Python package's Rust extension.
            uv
            maturin
            python312
            # pyo3 and build scripts need a C compiler and pkg-config.
            pkg-config
            # ddx-ad, which the ddx integration depends on, generates Rust
            # types from .proto files at build time.
            protobuf
          ];

          # Lets rust-analyzer find the standard library's sources.
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          # uv's downloaded Pythons are generic Linux binaries that don't run
          # on NixOS, so use only the interpreter on PATH.
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON_PREFERENCE = "only-system";

          # Wheels from PyPI (datafusion, pyarrow, numpy) expect libstdc++ on
          # the loader path, which NixOS doesn't provide globally.
          shellHook = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });

      formatter = forAll (pkgs: pkgs.nixpkgs-fmt);
    };
}
