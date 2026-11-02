# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Presence proofs agree with SQL for kept, reduced and empty domains."""

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


@pytest.mark.parametrize("mode", ["off", "boolean", "presence", "all"])
@pytest.mark.parametrize("scope", ["kept", "reduced", "both"])
@pytest.mark.parametrize("seed", range(8))
def test_presence_proofs_keep_nulls_filters_empty_groups_and_boolean_types(
    monkeypatch, mode, scope, seed
):
    monkeypatch.setenv("EINFOLD_DENSE_PREDICATES", mode)
    rng = np.random.default_rng(seed)
    n, m = seed % 4, (seed // 2) % 3
    i, j = (
        _Axis(np.arange(n, dtype=np.int64)),
        _Axis(np.arange(m, dtype=np.int64)),
    )
    local = {"kept": (i,), "reduced": (j,), "both": (i, j)}[scope]
    valid = rng.random(tuple(len(a) for a in local)) > 0.5
    table = _Table(
        (i, j),
        [
            (("t", "i"), _column(i.values, (i,), i)),
            (("t", "j"), _column(j.values, (j,), j)),
            (
                ("t", "v"),
                _Column(
                    _column(rng.normal(size=(n, m)), (i, j)).tensor,
                    _mask(valid, local),
                ),
            ),
        ],
    )
    schema = pa.schema(
        [("i", pa.int64()), ("j", pa.int64()), ("v", pa.float64())]
    )
    rows = table.arrow(schema)
    batches = rows.to_batches() or [
        pa.record_batch(
            [pa.array([], type=f.type) for f in schema], schema=schema
        )
    ]
    ctx = datafusion.SessionContext()
    ctx.register_record_batches("t", [batches])
    queries = [
        "SELECT COUNT(*)>0 AS rows,COUNT(v)>0 AS vals,NOT NOT(COUNT(v)>0) AS again FROM t",
        "WITH c AS (SELECT i,COUNT(v) AS n FROM t GROUP BY i) SELECT i,n>0 AS p,CASE WHEN NOT TRUE THEN FALSE ELSE n>0 END AS q,CASE WHEN n>0 THEN TRUE ELSE FALSE END AS r FROM c ORDER BY i",
        "SELECT i,COUNT(v)>0 AS p FROM t WHERE i%2=0 GROUP BY i ORDER BY i",
        "SELECT i,COUNT(v)>0 AS p FROM t WHERE v>0 GROUP BY i ORDER BY i",
        "WITH c AS (SELECT i,COUNT(v) AS n FROM t GROUP BY i) SELECT a.i,b.n>0 AS p FROM c a JOIN c b ON a.i=b.i ORDER BY a.i",
        "WITH c AS (SELECT i,COUNT(v) AS n FROM t GROUP BY i) SELECT COUNT(n)>0 AS p FROM c",
    ]
    for query in queries:
        frame = ctx.sql(query)
        token = _CACHE.set(_ComputedCache(256))
        try:
            result = (
                DenseProgram(frame.logical_plan(), frame.schema())
                .run(lambda _: table)
                .arrow(frame.schema())
            )
        finally:
            _CACHE.reset(token)
        assert result == frame.to_arrow_table()
