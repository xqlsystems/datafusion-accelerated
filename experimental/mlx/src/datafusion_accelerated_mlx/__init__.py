# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Experimental: compact-array execution of DataFusion plans, and MLX.

This package holds the contributor's MLX path, the speed track of open
question 14 (issue #41), until DataFusion Accelerated's milestones replace
it. Each part retires with the milestone named in the package's README.
Nothing outside this repository's ``experimental/`` directory and XQL's
demos may import it.
"""

from datafusion_accelerated_mlx.dense_program import DenseProgram

__all__ = ["DenseProgram"]
