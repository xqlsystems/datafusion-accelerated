# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""The native module, through datafusion-python's public API."""

import pyarrow as pa
import pytest
from datafusion import SessionContext

from datafusion_accelerated_mlx._native import _SortedArrowTable, shared_ctes


def _sorted(ctx):
    """``ORDER BY k`` over a sorted table in ``ctx``, which must outlive the
    frame: a table crossing datafusion-ffi reads through its session."""
    batch = pa.record_batch({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    ctx.register_table("t", _SortedArrowTable(batch, ["k"]))
    return ctx.sql("SELECT k, v FROM t ORDER BY k")


def test_a_sorted_table_registers_and_reads():
    ctx = SessionContext()
    frame = _sorted(ctx)
    assert frame.to_pydict() == {"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]}


@pytest.mark.xfail(
    strict=True,
    reason="the order is lost crossing datafusion-ffi "
    "(xqlsystems/datafusion-accelerated#55)",
)
def test_a_sorted_tables_order_skips_the_sort():
    # The table declares its order, so DataFusion should plan no sort for
    # it, as it doesn't in Rust (rust/sorted_arrow_table.rs).
    ctx = SessionContext()
    plan = _sorted(ctx).execution_plan().display_indent()
    assert "SortExec" not in plan, plan


def test_shared_ctes_gives_each_part_as_a_tuple():
    parts = shared_ctes("WITH h AS (SELECT exp(v) AS v FROM x) SELECT v FROM h")
    assert parts == [
        ("h", "SELECT exp(v) AS v FROM x", ["x"], ["exp"]),
        ("", "SELECT v FROM h", ["h"], []),
    ]
    with pytest.raises(ValueError, match="nested WITH"):
        shared_ctes("SELECT * FROM (WITH r AS (SELECT 1) SELECT * FROM r)")
