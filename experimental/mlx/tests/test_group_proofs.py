# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Group-presence proofs preserve bags, partial domains and NULL values."""

import datafusion
import numpy as np
import pyarrow as pa
import pytest
import xarray as xr
from datafusion_accelerated_mlx.dense_program import (
    _CACHE,
    DenseProgram,
    _ComputedCache,
    _read_dataset,
)


@pytest.mark.parametrize("proofs", ["0", "1"])
@pytest.mark.parametrize("seed", range(16))
def test_full_branch_presence_matches_sql_with_partial_nullable_branches(
    monkeypatch, proofs, seed
):
    monkeypatch.setenv("EINFOLD_DENSE_GROUP_PROOFS", proofs)
    rng = np.random.default_rng(seed)
    n, m = seed % 4, (seed // 2) % 4
    i, j = np.arange(n), np.arange(m)
    if seed % 2:
        i = i[::-1]
    datasets = {
        "a": xr.Dataset(
            {"v": (("i", "j"), rng.normal(size=(n, m)))},
            coords={"i": i, "j": j},
        ),
        "b": xr.Dataset(
            {"v": (("i", "j"), rng.normal(size=(len(i[::2]), m)))},
            coords={"i": i[::2], "j": j},
        ),
    }
    tables = {name: _read_dataset(name, ds) for name, ds in datasets.items()}
    ctx = datafusion.SessionContext()
    schema = pa.schema(
        [("i", pa.int64()), ("j", pa.int64()), ("v", pa.float64())]
    )
    for name, table in tables.items():
        rows = table.arrow(schema)
        batches = rows.to_batches() or [
            pa.record_batch(
                [pa.array([], type=f.type) for f in schema], schema=schema
            )
        ]
        ctx.register_record_batches(name, [batches])
    branches = [
        "SELECT i,j,v FROM a UNION ALL SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM b WHERE v>-1",
        "SELECT i,j,CASE WHEN v>0 THEN v END AS v FROM a UNION ALL SELECT i,j,v FROM b",
        "SELECT i,j,v FROM a UNION ALL SELECT i,j,v FROM b UNION ALL SELECT i,j,v FROM a",
    ]
    for branch in branches:
        for group in ["i,j", "i", ""]:
            prefix = group + "," if group else ""
            suffix = f" GROUP BY {group} ORDER BY {group}" if group else ""
            frame = ctx.sql(
                f"WITH u AS ({branch}) SELECT {prefix}SUM(v) AS s,COUNT(v) AS n,COUNT(*) AS rows FROM u{suffix}"
            )
            token = _CACHE.set(_ComputedCache(256))
            try:
                actual = (
                    DenseProgram(frame.logical_plan(), frame.schema())
                    .run(tables.__getitem__)
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
