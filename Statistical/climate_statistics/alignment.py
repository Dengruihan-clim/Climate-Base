"""Explicit calendar-year alignment for annual or seasonal statistics.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
"""
from __future__ import annotations

import numpy as np


def align_years(*series):
    """Intersect strictly ordered year labels and align each time-first array.

    Each positional argument is ``(years, values)``. Return
    ``(common_years, tuple_of_aligned_arrays)``. At least two inputs and one
    common year are required. Years must be finite integers, unique, and
    strictly increasing. Masked data become NaN; infinities and complex data
    are rejected. No data are imputed and no
    index is standardized again after selecting the common years.
    """
    if len(series) < 2:
        raise ValueError("Provide at least two (years, values) pairs")
    prepared = []
    for item in series:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("Each input must be a (years, values) pair")
        try:
            for part in item:
                raw = np.ma.asarray(part)
                if (not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
                        or np.issubdtype(raw.dtype, np.bool_)):
                    raise ValueError("Years and values must be real numeric arrays")
            years = np.asarray(np.ma.filled(np.ma.asarray(item[0], dtype=float), np.nan))
            values = np.asarray(np.ma.filled(np.ma.asarray(item[1], dtype=float), np.nan))
        except (TypeError, ValueError) as error:
            raise ValueError("Years and values must be real numeric arrays") from error
        if (years.ndim != 1 or not len(years) or not np.isfinite(years).all()
                or np.any(years != np.floor(years)) or np.any(np.diff(years) <= 0)):
            raise ValueError("Years must be finite, integer, and strictly increasing")
        if values.ndim < 1 or values.shape[0] != len(years):
            raise ValueError("The first values dimension must match years")
        if np.isinf(values).any():
            raise ValueError("Values cannot contain infinities; use NaN for missing data")
        prepared.append((years, values))
    common = prepared[0][0]
    for years, _ in prepared[1:]:
        common = np.intersect1d(common, years, assume_unique=True)
    if not len(common):
        raise ValueError("The inputs have no common years")
    return common, tuple(values[np.searchsorted(years, common)] for years, values in prepared)
