# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Compact graph passes preserve SQL bags, NULLs and changing inputs."""

import datafusion
import numpy as np
import pyarrow as pa
import pytest
from datafusion_accelerated_mlx.dense_program import (
    _CACHE,
    DenseProgram,
    _Axis,
    _Column,
    _column,
    _ComputedCache,
    _mask,
    _Table,
)


@pytest.mark.parametrize("mode", ["off", "facts", "cse", "all"])
@pytest.mark.parametrize("seed", range(12))
def test_optimizer_matches_sql_with_empty_nullable_and_duplicate_rows(
    monkeypatch, mode, seed
):
    monkeypatch.setenv("EINFOLD_DENSE_OPTIMIZE", mode)
    rng = np.random.default_rng(seed)
    n, m = seed % 4, int(rng.integers(1, 5))
    i, j = _Axis(np.arange(n)), _Axis(np.arange(m))
    data = rng.normal(size=(n, m))
    valid = rng.random((n, m)) > 0.35
    table = _Table(
        (i, j),
        [
            (("t", "i"), _column(i.values, (i,), i)),
            (("t", "j"), _column(j.values, (j,), j)),
            (
                ("t", "v"),
                _Column(_column(data, (i, j)).tensor, _mask(valid, (i, j))),
            ),
        ],
    )
    schema = pa.schema(
        [("i", pa.int64()), ("j", pa.int64()), ("v", pa.float64())]
    )
    rows = table.arrow(schema)
    batches = rows.to_batches() or [
        pa.RecordBatch.from_arrays(
            [pa.array([], type=f.type) for f in schema], schema=schema
        )
    ]
    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [batches])
    queries = [
        "SELECT COUNT(*) AS n,COUNT(v) AS c,SUM(v) AS s,AVG(v) AS a FROM t",
        "SELECT i,COUNT(*) AS n,COUNT(v) AS c,SUM(v) AS s,AVG(v) AS a FROM t GROUP BY i ORDER BY i",
        "SELECT COUNT(*) AS n,SUM(1.0) AS s,AVG(1.0) AS a FROM t WHERE FALSE",
        "SELECT SUM(1.0)/2 AS half_count FROM t",
        "WITH u AS (SELECT * FROM t UNION ALL SELECT * FROM t) SELECT i,COUNT(*) AS n,COUNT(v) AS c,SUM(v) AS s FROM u GROUP BY i ORDER BY i",
        "SELECT i,j,v-1.0 AS a,1.0-v AS b,v*(-0.0) AS z FROM t ORDER BY i,j",
        "SELECT i,j,v*0.0 AS p,v*(-0.0) AS n FROM t ORDER BY i,j",
        "SELECT i,COUNT(*) AS n,COUNT(v) AS c FROM t WHERE v>0 GROUP BY i ORDER BY i",
    ]
    for query in queries:
        frame = ctx.sql(query)
        token = _CACHE.set(_ComputedCache(1 << 20))
        try:
            actual = (
                DenseProgram(frame.logical_plan(), frame.schema())
                .run(lambda _: table)
                .arrow(frame.schema())
            )
        finally:
            _CACHE.reset(token)
        expected = frame.to_arrow_table()
        assert actual.schema == expected.schema
        assert actual.num_rows == expected.num_rows
        for a, e in zip(actual.columns, expected.columns):
            assert a.is_null().to_pylist() == e.is_null().to_pylist()
            np.testing.assert_allclose(
                a.to_numpy(), e.to_numpy(), atol=1e-12, equal_nan=True
            )
            if pa.types.is_floating(a.type):
                zero = (e.to_numpy() == 0) & ~e.is_null().to_numpy()
                np.testing.assert_array_equal(
                    np.signbit(a.to_numpy()[zero]),
                    np.signbit(e.to_numpy()[zero]),
                )


@pytest.mark.parametrize("mode", ["off", "facts", "cse", "all"])
@pytest.mark.parametrize("cache_bytes", [1, 1 << 20])
def test_shared_reductions_keep_input_identity_and_operand_order(
    monkeypatch, mode, cache_bytes
):
    monkeypatch.setenv("EINFOLD_DENSE_OPTIMIZE", mode)
    axis = _Axis(np.arange(4))
    ctx = datafusion.SessionContext()
    ctx.register_record_batches(
        "t", [[pa.record_batch({"i": np.arange(4), "v": np.zeros(4)})]]
    )
    frame = ctx.sql(
        "WITH a AS (SELECT SUM(v*v) AS s FROM t), b AS (SELECT SUM(v*v) AS s FROM t) SELECT a.s+b.s AS doubled,a.s-b.s AS difference FROM a CROSS JOIN b"
    )
    program = DenseProgram(frame.logical_plan(), frame.schema())
    for offset in [0.0, 3.0, -2.0]:
        values = np.arange(4.0) + offset
        table = _Table(
            (axis,),
            [
                (("t", "i"), _column(axis.values, (axis,), axis)),
                (("t", "v"), _column(values, (axis,))),
            ],
        )
        token = _CACHE.set(_ComputedCache(cache_bytes))
        try:
            result = program.run(lambda _, current=table: current).arrow(
                frame.schema()
            )
        finally:
            _CACHE.reset(token)
        assert result.to_pydict() == {
            "doubled": [2 * np.sum(values**2)],
            "difference": [0.0],
        }


@pytest.mark.parametrize("seed", range(8))
def test_renamed_contractions_preserve_free_and_reduced_axes(monkeypatch, seed):
    monkeypatch.setenv("EINFOLD_DENSE_OPTIMIZE", "all")
    rng = np.random.default_rng(seed)
    i, j = _Axis(np.arange(3)), _Axis(np.arange(2))
    values = rng.normal(size=(3, 2))
    table = _Table(
        (i, j),
        [
            (("t", "i"), _column(i.values, (i,), i)),
            (("t", "j"), _column(j.values, (j,), j)),
            (("t", "v"), _column(values, (i, j))),
        ],
    )
    ctx = datafusion.SessionContext()
    ctx.register_record_batches(
        "t",
        [
            [
                pa.record_batch(
                    {
                        "i": np.repeat(i.values, 2),
                        "j": np.tile(j.values, 3),
                        "v": values.ravel(),
                    }
                )
            ]
        ],
    )
    query = """WITH a AS (
      SELECT x.i,y.i AS k,SUM(x.v*y.v) AS v FROM t x JOIN t y ON x.j=y.j GROUP BY x.i,y.i),
      b AS (SELECT x.i,y.i AS k,SUM(x.v*y.v) AS v FROM t x JOIN t y ON x.j=y.j GROUP BY x.i,y.i)
      SELECT a.i,a.k,a.v+b.v AS v FROM a JOIN b ON a.i=b.i AND a.k=b.k ORDER BY a.i,a.k"""
    frame = ctx.sql(query)
    token = _CACHE.set(_ComputedCache(128))
    try:
        actual = (
            DenseProgram(frame.logical_plan(), frame.schema())
            .run(lambda _: table)
            .arrow(frame.schema())
        )
    finally:
        _CACHE.reset(token)
    expected = frame.to_arrow_table()
    assert actual.select(["i", "k"]) == expected.select(["i", "k"])
    np.testing.assert_allclose(
        actual["v"].to_numpy(), expected["v"].to_numpy(), atol=1e-12
    )


@pytest.mark.parametrize("cache_bytes", [1, 1 << 20])
def test_cost_gated_reductions_keep_masks_and_refresh_inputs(
    monkeypatch, cache_bytes
):
    monkeypatch.setenv("EINFOLD_DENSE_REDUCTION_VALUES", "1")
    rng = np.random.default_rng(17)
    i, j = _Axis(np.arange(96)), _Axis(np.arange(96))
    ctx = datafusion.SessionContext()
    query = """WITH a AS (
      SELECT x.i,y.i AS k,SUM(x.v*y.v) AS v
      FROM t x JOIN t y ON x.j=y.j GROUP BY x.i,y.i),
    b AS (
      SELECT x.i,y.i AS k,SUM(x.v*y.v) AS v
      FROM t x JOIN t y ON x.j=y.j GROUP BY x.i,y.i),
    c AS (
      SELECT x.i,y.i AS k,SUM(CASE WHEN x.j<48 THEN x.v*y.v END) AS v
      FROM t x JOIN t y ON x.j=y.j GROUP BY x.i,y.i)
    SELECT a.i,a.k,a.v+b.v AS full,c.v AS masked FROM a
      JOIN b ON a.i=b.i AND a.k=b.k JOIN c ON a.i=c.i AND a.k=c.k
    ORDER BY a.i,a.k"""
    for offset in (0.0, 0.5):
        values = rng.normal(size=(96, 96)) + offset
        table = _Table(
            (i, j),
            [
                (("t", "i"), _column(i.values, (i,), i)),
                (("t", "j"), _column(j.values, (j,), j)),
                (("t", "v"), _column(values, (i, j))),
            ],
        )
        schema = pa.schema(
            [("i", pa.int64()), ("j", pa.int64()), ("v", pa.float64())]
        )
        ctx.deregister_table("t")
        ctx.register_record_batches("t", [table.arrow(schema).to_batches()])
        frame = ctx.sql(query)
        token = _CACHE.set(_ComputedCache(cache_bytes))
        try:
            actual = (
                DenseProgram(frame.logical_plan(), frame.schema())
                .run(lambda _, current=table: current)
                .arrow(frame.schema())
            )
        finally:
            _CACHE.reset(token)
        expected = values @ values.T
        masked = values[:, :48] @ values[:, :48].T
        np.testing.assert_allclose(actual["full"], (2 * expected).ravel())
        np.testing.assert_allclose(actual["masked"], masked.ravel())
