# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A numerical host can lower reductions without changing relational rows."""

import datafusion
import numpy as np
import pyarrow as pa
import pytest
from datafusion_accelerated_mlx.dense_program import (
    _CACHE,
    _EXECUTION,
    DenseProgram,
    _Axis,
    _Column,
    _column,
    _ComputedCache,
    _mask,
    _Table,
)


class _NumpyHost:
    array = staticmethod(np.asarray)
    numpy = staticmethod(np.asarray)

    def owns(self, array):
        return False

    def pointwise(self, tensor):
        return None

    def reduction(self, tensor):
        info = tensor.reduction
        if (
            info is None
            or info.kind != "sum"
            or any(len(a) == 0 for a in info.kept)
        ):
            return None
        shape = tuple(len(a) for a in info.row_axes)

        def rows(node):
            ordered = tuple(a for a in info.row_axes if a in node.axes)
            data = np.asarray(node.get()).transpose(
                tuple(node.axes.index(a) for a in ordered)
            )
            return np.broadcast_to(
                data.reshape(
                    tuple(
                        len(a) if a in node.axes else 1 for a in info.row_axes
                    )
                ),
                shape,
            )

        contributions = np.where(rows(info.mask), rows(info.value), 0)
        result = np.sum(
            contributions,
            axis=tuple(
                i for i, a in enumerate(info.row_axes) if a not in info.kept
            ),
        )
        remaining = tuple(a for a in info.row_axes if a in info.kept)
        result = result.transpose(tuple(remaining.index(a) for a in info.kept))
        return result[
            tuple(slice(None) if a in tensor.axes else 0 for a in info.kept)
        ]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("host", [None, _NumpyHost()])
def test_reduction_host_keeps_nulls_empty_domains_and_axis_views(seed, host):
    rng = np.random.default_rng(seed)
    n, m = seed % 4, (seed // 2) % 4
    i, j = _Axis(np.arange(n)), _Axis(np.arange(m))
    data = rng.normal(size=(n, m))
    table = _Table(
        (i, j),
        [
            (("t", "i"), _column(i.values, (i,), i)),
            (("t", "j"), _column(j.values, (j,), j)),
            (
                ("t", "v"),
                _Column(
                    _column(data, (i, j)).tensor,
                    _mask(rng.random((n, m)) > 0.3, (i, j)),
                ),
            ),
        ],
    )
    schema = pa.schema(
        [("i", pa.int64()), ("j", pa.int64()), ("v", pa.float64())]
    )
    rows = table.arrow(schema)
    ctx = datafusion.SessionContext()
    batches = rows.to_batches() or [
        pa.record_batch(
            [pa.array([], type=f.type) for f in schema], schema=schema
        )
    ]
    ctx.register_record_batches("t", [batches])
    queries = [
        "SELECT SUM(v) AS s,AVG(v) AS a,COUNT(*) AS n,MIN(v) AS lo,MAX(v) AS hi FROM t",
        "SELECT i,SUM(v) AS s,AVG(v) AS a,COUNT(v) AS n,MIN(v) AS lo,MAX(v) AS hi FROM t WHERE v>0 GROUP BY i ORDER BY i",
        "SELECT i,SUM(2.0) AS s,COUNT(*) AS n FROM t GROUP BY i ORDER BY i",
        "WITH a AS (SELECT i,MAX(v) AS v FROM t GROUP BY i) SELECT a.i,b.i AS j,a.v-b.v AS d FROM a CROSS JOIN a b ORDER BY a.i,b.i",
        "WITH a AS (SELECT * FROM t UNION ALL SELECT * FROM t) SELECT i,SUM(v) AS s,COUNT(v) AS n FROM a GROUP BY i ORDER BY i",
    ]
    for query in queries:
        frame = ctx.sql(query)
        execution = _EXECUTION.set(host)
        cache = _CACHE.set(_ComputedCache(128))
        try:
            actual = (
                DenseProgram(frame.logical_plan(), frame.schema())
                .run(lambda _: table)
                .arrow(frame.schema())
            )
        finally:
            _CACHE.reset(cache)
            _EXECUTION.reset(execution)
        expected = frame.to_arrow_table()
        assert actual.schema == expected.schema
        assert actual.num_rows == expected.num_rows
        for a, e in zip(actual.columns, expected.columns):
            assert a.is_null().to_pylist() == e.is_null().to_pylist()
            np.testing.assert_allclose(
                a.to_numpy(), e.to_numpy(), atol=1e-12, equal_nan=True
            )
