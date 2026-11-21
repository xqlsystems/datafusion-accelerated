# SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Optional SQL mathematical primitives with equivalent array lowering.

DataFusion (https://datafusion.apache.org/) supplies most numerical SQL
functions. This module supplies the error function for exact GELU, the
activation used by nanoGPT (https://github.com/karpathy/nanoGPT).

Ported from xarray-sql's ``xarray_sql/math.py`` at ``a1d237a``, on Ray
Cromwell's branch ``codex/numerical-value-identities``.
"""

import math

import numpy as np
import pyarrow as pa
from datafusion import udf

try:
    from scipy.special import erf as _erf
except ImportError:
    _erf = np.vectorize(math.erf, otypes=[np.float64])

__all__ = ["register_erf", "trusted_functions"]


def _arrow_erf(values):
    return pa.array(
        _erf(values.to_numpy(zero_copy_only=False)),
        mask=values.is_null().to_numpy(zero_copy_only=False),
        type=pa.float64(),
    )


# The one erf this module registers. A context's erf is trusted only while
# it is this object, so registering another erf removes the trust.
_ERF = udf(_arrow_erf, [pa.float64()], pa.float64(), "immutable", name="erf")

# The functions this package may run as array primitives in place of the
# SQL function registered under their name, as ``(name, function)``.
_TRUSTED = (("erf", _ERF),)


def register_erf(ctx):
    """Install SQL ``erf(DOUBLE)`` and its equivalent CPU/GPU primitive.

    Differentiation also needs ddx's erf rule, in ddxdb 0.4.8 and later
    (ddx#256). Registering another ``erf`` on ``ctx`` removes its trusted
    status, so a function of the user's is never silently bypassed.
    """
    ctx.register_udf(_ERF)


def trusted_functions(ctx):
    """The lowered names of the functions registered on ``ctx``, an
    xarray-sql ``Context``, that this package may run as array primitives:
    those it registered itself, and that nothing has registered over since.
    """
    registered = ctx.registered_functions
    return {name for name, function in _TRUSTED if registered.get(name) is function}
