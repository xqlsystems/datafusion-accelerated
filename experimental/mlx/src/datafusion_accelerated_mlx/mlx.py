# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Apple silicon execution for SQL tensor computations.

[EinFold](https://github.com/xqlsystems/einfold) recognizes tensor operations
in DataFusion plans. This extension supplies its validated dense matrices to
[MLX](https://github.com/ml-explore/mlx), Apple's array framework. SQL parsing,
row existence, NULL handling and [ddx](https://github.com/xqlsystems/ddx)
gradient construction remain in their original engines.

It needs einfold 0.3.0rc1 or later, whose ``EinFold`` takes a dense backend
(einfold#167). Install it on a context before planning queries::

    # Launch Python with MLX_ENABLE_TF32=0 before importing MLX.
    from xarray_sql import Context
    from datafusion_accelerated_mlx.mlx import MLX

    accelerator = MLX(precision="float32")
    ctx = Context().with_extensions(accelerator)

``float32`` explicitly permits reduced precision for dense arithmetic on
the GPU, including arithmetic in ddx's forward and backward SQL programs.
The default ``float64`` uses NumPy's CPU matrix multiplication and EinFold's
CPU attention; MLX's Metal GPU does not implement double precision. Neither
mode replaces ddx with MLX autodiff. Unsupported shapes and exceptional
values retain EinFold's CPU implementation. No table values are cached.
"""

from __future__ import annotations

import os
import threading
from collections import Counter
from typing import Literal

import mlx.core as mx
import numpy as np
import pyarrow as pa
from einfold.datafusion import EinFold

__all__ = ["MLX"]


class MLX:
    """EinFold extension with an optional MLX GPU arithmetic backend.

    Args:
        precision: ``float64`` keeps double precision on the CPU.
            ``float32`` permits rounding inputs and dense results to single
            precision on the GPU. Results retain their SQL Arrow dtype.
        min_ops: Minimum scalar products before crossing the native/Python
            boundary. Smaller computations use the original CPU kernel.
        min_gpu_ops: Minimum scalar products for GPU execution. Smaller
            dense products use float64 CPU matrix multiplication; smaller
            attention uses EinFold's CPU tiles.
        max_working_bytes: Upper bound on estimated extra backend buffers.
            Larger computations retain the CPU kernel. This is separate
            from DataFusion's own memory reservation.
        query_block: Maximum queries in an attention block. Bounds mask
            storage independently of the number of queries.

    An einfold without the dense backend raises an actionable error
    instead of silently claiming GPU execution.
    """

    def __init__(
        self,
        *,
        precision: Literal["float64", "float32"] = "float64",
        min_ops: int = 1 << 20,
        min_gpu_ops: int = 1 << 33,
        max_working_bytes: int = 512 << 20,
        query_block: int = 1024,
    ) -> None:
        if precision not in ("float64", "float32"):
            raise ValueError("precision must be 'float64' or 'float32'")
        if precision == "float32" and os.environ.get("MLX_ENABLE_TF32") != "0":
            raise ValueError(
                "Launch Python with MLX_ENABLE_TF32=0 for full float32 "
                "arithmetic; MLX reads this process setting on first use"
            )
        if (
            min_ops < 0
            or min_gpu_ops < 0
            or max_working_bytes <= 0
            or not 1 <= query_block <= 1024
        ):
            raise ValueError("invalid operation, memory or block limit")
        self.precision = precision
        self._backend = _Backend(
            precision, min_ops, min_gpu_ops, max_working_bytes, query_block
        )
        try:
            self._extension = EinFold(
                dense_backend=self._backend.run,
                dense_min_ops=min_ops,
                dense_max_bytes=max_working_bytes,
            )
        except TypeError as exc:
            raise ImportError(
                "MLX needs an einfold whose EinFold takes a dense backend: "
                "einfold 0.3.0rc1 or later (einfold#167)"
            ) from exc

    def __datafusion_session_components__(self, ctx):
        return self._extension.__datafusion_session_components__(ctx)

    def __datafusion_session_planner__(self, ctx, fallback):
        return self._extension.__datafusion_session_planner__(ctx, fallback)

    def statistics(self) -> dict[str, int]:
        """Completed CPU/GPU kernels and declined backend requests.

        Counts describe execution, including repeated collections, rather
        than the presence of a node in an explained plan.
        """
        return self._backend.statistics()


class _Backend:
    def __init__(
        self,
        precision: Literal["float64", "float32"],
        min_ops: int,
        min_gpu_ops: int,
        max_working_bytes: int,
        query_block: int,
    ) -> None:
        self.precision = precision
        self.min_ops = min_ops
        self.min_gpu_ops = min_gpu_ops
        self.max_working_bytes = max_working_bytes
        self.query_block = query_block
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def statistics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def _count(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def _decline(self):
        self._count("declined")

    def run(self, kind, columns, shape, scale, ranges):
        # The native side has already checked positional keys, duplicates,
        # SQL NULLs and the domain. It also refuses non-finite factors.
        #
        # einfold#167's shapes: a matmul is (batches, rows, inner, columns);
        # attention is (batches, queries, features, keys, values), with one
        # range of keys per query shared by every batch. Ray's fork offered
        # attention unbatched, as (queries, features, keys, values).
        if kind == "matmul":
            w, x, z, y = shape
            ops = w * x * z * y
            output = w * x * y
        elif kind == "attention":
            n, w, x, z, y = shape
            ops = n * w * z * (x + y)
            output = n * w * y
        else:
            # A kind a later einfold offers, which this backend doesn't know.
            return self._decline()
        cells = sum(len(c) for c in columns)
        if ops < self.min_ops or (cells + output) * 32 > self.max_working_bytes:
            return self._decline()
        arrays = [c.to_numpy(zero_copy_only=True) for c in columns]
        if self.precision == "float64" or ops < self.min_gpu_ops:
            if kind != "matmul":
                return self._decline()
            a, b = arrays
            result = np.matmul(a.reshape(w, x, z), b.reshape(w, z, y))
            device = "cpu"
        else:
            # Keep overflow, underflow and exceptional SQL arithmetic on
            # the CPU. In particular, masked infinities must stay absent
            # rows instead of participating in a zero-weighted product.
            maximum = []
            for a in arrays:
                magnitudes = np.abs(a)
                hi = magnitudes.max(initial=0.0)
                if (
                    not np.isfinite(hi)
                    or hi > 1e15
                    or np.any((magnitudes > 0) & (magnitudes < 1e-15))
                ):
                    return self._decline()
                maximum.append(hi)
            if kind == "attention":
                mode, constant = scale
                multiplier = constant if mode == 1 else 1.0
                if mode == 2:
                    if constant == 0:
                        return self._decline()
                    multiplier = 1.0 / constant
                # Scores too close to single precision's range, including
                # infinity, keep the original double precision algorithm.
                bound = maximum[0] * maximum[1] * x * abs(multiplier)
                if not np.isfinite(bound) or bound > 1e30:
                    return self._decline()
            block_size = self.query_block
            if kind == "attention":
                available = self.max_working_bytes - (cells + output) * 32
                block_size = min(block_size, available // (n * z * 16))
                if block_size < 1:
                    return self._decline()
            with mx.stream(mx.gpu):
                inputs = [mx.array(a.astype(np.float32)) for a in arrays]
                if kind == "matmul":
                    a, b = inputs
                    result = mx.matmul(a.reshape(w, x, z), b.reshape(w, z, y))
                    mx.eval(result)
                    result = np.asarray(result).astype(np.float64)
                else:
                    q, kt, v = inputs
                    q, k, v = (
                        q.reshape(n, w, x),
                        kt.reshape(n, x, z).transpose(0, 2, 1),
                        v.reshape(n, z, y),
                    )
                    # Query tiles bound the mask and temporary score storage.
                    # Ranges encode all of EinFold's normalized key masks,
                    # including strict, offset and sliding-window masks.
                    result = np.zeros((n, w, y), dtype=np.float64)
                    for start in range(0, w, block_size):
                        stop = min(w, start + block_size)
                        lo, hi = np.asarray(ranges[start:stop]).T
                        nonempty = lo < hi
                        if not nonempty.any():
                            continue
                        key_start = int(lo[nonempty].min())
                        key_stop = int(hi[nonempty].max())
                        key_count = key_stop - key_start
                        causal_hi = np.clip(
                            np.arange(stop - start)
                            + key_count
                            - (stop - start)
                            + 1,
                            0,
                            key_count,
                        )
                        if np.all(lo == key_start) and np.all(hi == key_stop):
                            allowed = None
                        elif np.all(lo == key_start) and np.array_equal(
                            hi - key_start, causal_hi
                        ):
                            # MLX aligns a causal mask at the lower right.
                            # Its last query then sees this slice's last key.
                            allowed = "causal"
                        else:
                            keys = mx.arange(key_start, key_stop)[None, :]
                            allowed = (keys >= mx.array(lo[:, None])) & (
                                keys < mx.array(hi[:, None])
                            )
                        # Each batch is one head of MLX's (batch, head,
                        # length, features) layout; the mask is shared.
                        block = mx.fast.scaled_dot_product_attention(
                            q[:, None, start:stop, :],
                            k[:, None, key_start:key_stop, :],
                            v[:, None, key_start:key_stop, :],
                            scale=multiplier,
                            mask=allowed,
                        )
                        mx.eval(block)
                        result[:, start:stop] = np.asarray(block)[:, 0]
            device = "gpu"
        if not np.isfinite(result).all():
            return self._decline()
        self._count(f"{device}_{kind}")
        return pa.array(result.reshape(-1), type=pa.float64())
