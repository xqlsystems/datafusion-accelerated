# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Positional reshapes retain coordinate joins, counts and NULL groups."""

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


@pytest.mark.parametrize(
    "layout",
    ["ordered", "cyclic", "permuted", "repeated", "unmatched", "varying"],
)
@pytest.mark.parametrize("overlap", [False, True])
def test_coordinate_split_and_regroup_match_datafusion(
    monkeypatch, layout, overlap
):
    monkeypatch.setenv("EINFOLD_DENSE_LAYOUT", "extended")
    labels = np.array([10, 20, 40, 50, 60, 70])
    positions = np.arange(2)
    keys = np.broadcast_to(labels, (2, 6)).copy()
    if layout == "cyclic":
        keys = np.roll(keys.reshape(2, 2, 3), (-1, -2), axis=(1, 2)).reshape(
            2, 6
        )
    elif layout == "permuted":
        keys = keys[:, ::-1]
    elif layout == "repeated":
        keys[:, 0] = keys[:, 1]
    elif layout == "unmatched":
        keys[:, 0] = 999
    elif layout == "varying":
        keys[1] = keys[1, ::-1]
    if overlap:
        map_dims, map_keys = ("pos", "h", "c"), keys.reshape(2, 2, 3)
        map_coords = {"pos": positions, "h": np.arange(2), "c": np.arange(3)}
        predicate = "f.pos=m.pos AND f.e=m.e"
    else:
        map_dims, map_keys = ("h", "c"), keys[0].reshape(2, 3)
        map_coords = {"h": np.arange(2), "c": np.arange(3)}
        predicate = "f.e=m.e"
    datasets = {
        "flat": xr.Dataset(
            {"v": (("e", "pos"), np.arange(-5.0, 7).reshape(6, 2))},
            coords={"e": labels, "pos": positions},
        ),
        "mapping": xr.Dataset(
            {"e": (map_dims, map_keys)},
            coords=map_coords,
        ),
    }
    ctx = datafusion.SessionContext()
    tables = {name: _read_dataset(name, ds) for name, ds in datasets.items()}
    for name, columns in (
        ("flat", ("e", "pos", "v")),
        ("mapping", (*map_dims, "e")),
    ):
        schema = pa.schema(
            [
                (key, pa.float64() if key == "v" else pa.int64())
                for key in columns
            ]
        )
        ctx.register_record_batches(
            name, [tables[name].arrow(schema).to_batches()]
        )
    for select in (
        "SELECT pos,h,c,v FROM split ORDER BY pos,h,c",
        "SELECT pos,e,SUM(v) AS s,COUNT(v) AS n FROM split GROUP BY pos,e ORDER BY pos,e",
    ):
        frame = ctx.sql(f"""WITH split AS (
          SELECT f.pos,m.h,m.c,m.e,CASE WHEN f.v>=0 THEN f.v END AS v
          FROM flat f JOIN mapping m ON {predicate})
          {select}""")
        token = _CACHE.set(_ComputedCache(4096))
        try:
            actual = (
                DenseProgram(frame.logical_plan(), frame.schema())
                .run(tables.__getitem__)
                .arrow(frame.schema())
            )
        finally:
            _CACHE.reset(token)
        assert actual.equals(frame.to_arrow_table())
