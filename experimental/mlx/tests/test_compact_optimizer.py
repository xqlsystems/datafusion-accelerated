# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
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


@pytest.mark.parametrize("seed", range(12))
def test_separable_join_masks_preserve_nullable_aggregates(monkeypatch, seed):
    monkeypatch.setenv("EINFOLD_DENSE_SYMBOLIC_MASKS", "1")
    monkeypatch.setenv("EINFOLD_DENSE_FACTOR_MASKS", "1")
    rng = np.random.default_rng(seed)
    i, r, s = (_Axis(np.arange(n)) for n in (3, seed % 4, (seed + 1) % 5))
    tables, schemas = {}, {}
    ctx = datafusion.SessionContext()
    for name, axis in [("a", r), ("b", s)]:
        shape = (len(i), len(axis))
        values = rng.normal(size=shape)
        valid = rng.random(shape) > 0.4
        table = _Table(
            (i, axis),
            [
                ((name, "i"), _column(i.values, (i,), i)),
                (
                    (name, "v"),
                    _Column(
                        _column(values, (i, axis)).tensor,
                        _mask(valid, (i, axis)),
                    ),
                ),
            ],
        )
        schema = pa.schema([("i", pa.int64()), ("v", pa.float64())])
        rows = table.arrow(schema)
        batches = rows.to_batches() or [
            pa.RecordBatch.from_arrays(
                [pa.array([], type=f.type) for f in schema], schema=schema
            )
        ]
        ctx.register_record_batches(name, [batches])
        tables[name] = table
        schemas[name] = schema
    for where in ["", " WHERE a.v>0 AND b.v<0"]:
        query = f"""SELECT a.i,COUNT(*) AS n,COUNT(a.v*b.v) AS c,
        SUM(a.v*b.v) AS total,AVG(a.v*b.v) AS mean
        FROM a JOIN b ON a.i=b.i{where} GROUP BY a.i ORDER BY a.i"""
        frame = ctx.sql(query)
        token = _CACHE.set(_ComputedCache(1 << 20))
        try:
            actual = (
                DenseProgram(frame.logical_plan(), frame.schema())
                .run(lambda name: tables[name])
                .arrow(frame.schema())
            )
        finally:
            _CACHE.reset(token)
        expected = frame.to_arrow_table()
        assert actual.num_rows == expected.num_rows
        for got, want in zip(actual.columns, expected.columns):
            assert got.is_null().to_pylist() == want.is_null().to_pylist()
            np.testing.assert_allclose(
                got.to_numpy(), want.to_numpy(), equal_nan=True, atol=1e-12
            )


@pytest.mark.parametrize("cache_bytes", [1, 1 << 20])
def test_positional_values_keep_consumer_labels_masks_and_changed_inputs(
    monkeypatch, cache_bytes
):
    monkeypatch.setenv("EINFOLD_DENSE_VALUE_IDENTITIES", "1")
    i, j, u, v = (
        _Axis(np.array(labels))
        for labels in [[10, 20], [2, 4, 8], [-1, 3], [11, 12, 13]]
    )
    programs, snapshots = {}, []
    for offset in [0.0, 7.0]:
        data = np.arange(6.0).reshape(2, 3) + offset
        source = _column(data, (i, j)).tensor
        tables = {
            "a": _Table(
                (i, j),
                [
                    (("a", "i"), _column(i.values, (i,), i)),
                    (("a", "j"), _column(j.values, (j,), j)),
                    (("a", "x"), _Column(source, _mask(data % 2 == 0, (i, j)))),
                ],
            ),
            "b": _Table(
                (u, v),
                [
                    (("b", "u"), _column(u.values, (u,), u)),
                    (("b", "v"), _column(v.values, (v,), v)),
                    (
                        ("b", "y"),
                        _Column(
                            source.rename({i: u, j: v}),
                            _mask(data % 2 != 0, (u, v)),
                        ),
                    ),
                ],
            ),
        }
        token = _CACHE.set(_ComputedCache(cache_bytes))
        try:
            for name, dims, column in [
                ("a", ("i", "j"), "x"),
                ("b", ("u", "v"), "y"),
            ]:
                schema = pa.schema(
                    [
                        (dims[0], pa.int64()),
                        (dims[1], pa.int64()),
                        (column, pa.float64()),
                    ]
                )
                ctx = datafusion.SessionContext()
                ctx.register_record_batches(
                    name, [tables[name].arrow(schema).to_batches()]
                )
                query = f"SELECT {','.join(dims)},sqrt({column}*{column}+1.0) AS z FROM {name} ORDER BY {','.join(dims)}"
                frame = ctx.sql(query)
                if name not in programs:
                    programs[name] = DenseProgram(
                        frame.logical_plan(), frame.schema()
                    )
                actual = (
                    programs[name]
                    .run(lambda key, current=tables: current[key])
                    .arrow(frame.schema())
                )
                expected = frame.to_arrow_table()
                for got, want in zip(actual.columns, expected.columns):
                    assert (
                        got.is_null().to_pylist() == want.is_null().to_pylist()
                    )
                    np.testing.assert_allclose(
                        got.to_numpy(), want.to_numpy(), equal_nan=True
                    )
                snapshots.append((actual, actual.to_pydict()))
        finally:
            _CACHE.reset(token)
    for snapshot, expected in snapshots:
        assert snapshot.to_pydict() == expected


@pytest.mark.parametrize("mode", ["off", "facts", "cse", "all"])
@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("symbolic", [False, True])
def test_optimizer_matches_sql_with_empty_nullable_and_duplicate_rows(
    monkeypatch, mode, seed, symbolic
):
    monkeypatch.setenv("EINFOLD_DENSE_OPTIMIZE", mode)
    monkeypatch.setenv("EINFOLD_DENSE_SYMBOLIC_MASKS", str(int(symbolic)))
    rng = np.random.default_rng(seed)
    n, m = seed % 4, int(rng.integers(1, 5))
    i, j = _Axis(np.arange(n)), _Axis(np.arange(m))
    data = rng.normal(size=(n, m))
    valid = rng.random((n, m)) > 0.35
    validity_axes = (i, j)
    if seed % 3 == 0:
        valid = valid[:, 0]
        validity_axes = (i,)
    table = _Table(
        (i, j),
        [
            (("t", "i"), _column(i.values, (i,), i)),
            (("t", "j"), _column(j.values, (j,), j)),
            (
                ("t", "v"),
                _Column(
                    _column(data, (i, j)).tensor, _mask(valid, validity_axes)
                ),
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
