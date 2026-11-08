"""Mapping between Python types in schemas and polars dtypes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Literal, get_args, get_origin

import polars as pl


@dataclass
class TypeChecker:
    """Type checker with both dtype-level and value-level validation."""

    dtype: Callable  # Check dtype
    value: Callable[[object], bool]  # Check individual values


def _is_datetime_dtype(dtype: Any) -> bool:
    """Check if dtype is a Datetime type (with any time unit)."""
    if dtype == pl.Datetime:
        return True
    if hasattr(dtype, "time_unit"):
        return True
    return False


TYPE_CHECKERS = {
    int: TypeChecker(
        dtype=lambda dtype: (
            dtype
            in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        ),
        value=lambda x: isinstance(x, int) and not isinstance(x, bool),
    ),
    float: TypeChecker(
        dtype=lambda dtype: dtype in (pl.Float32, pl.Float64),
        value=lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    ),
    str: TypeChecker(
        dtype=lambda dtype: dtype == pl.Utf8,
        value=lambda x: isinstance(x, str),
    ),
    bool: TypeChecker(
        dtype=lambda dtype: dtype == pl.Boolean,
        value=lambda x: isinstance(x, bool),
    ),
    datetime: TypeChecker(
        dtype=_is_datetime_dtype,
        value=lambda x: isinstance(x, datetime),
    ),
    date: TypeChecker(
        dtype=lambda dtype: dtype == pl.Date,
        value=lambda x: isinstance(x, date),
    ),
    timedelta: TypeChecker(
        dtype=lambda dtype: dtype == pl.Duration,
        value=lambda x: isinstance(x, timedelta),
    ),
    Any: TypeChecker(
        dtype=lambda _: True,
        value=lambda _: True,
    ),
}

# Mapping from Python type to polars dtype for creating empty DataFrames
TYPE_TO_DTYPE = {
    int: pl.Int64(),
    float: pl.Float64(),
    str: pl.Utf8(),
    bool: pl.Boolean(),
    datetime: pl.Datetime(),
    date: pl.Date(),
    timedelta: pl.Duration(),
    Any: pl.Utf8(),
}


def is_dtype_annotation(base_type: Any) -> bool:
    """Whether an annotation is a polars dtype, as a class (pl.Int64) or instance (pl.List(...))."""
    return isinstance(base_type, pl.DataType) or (
        isinstance(base_type, type) and issubclass(base_type, pl.DataType)
    )


def dtype_name(base_type: Any) -> str:
    """Name a Python type or polars dtype annotation for error messages."""
    return getattr(base_type, "__name__", str(base_type))


def dtype_for_type(base_type: Any) -> pl.DataType:
    """Get the polars dtype of an empty column for a scalar annotation."""
    # Union types (represented as tuple) use the first type
    if isinstance(base_type, tuple):
        base_type = base_type[0]
    if get_origin(base_type) is Literal:
        literal_values = get_args(base_type)
        if literal_values:
            base_type = type(literal_values[0])
    if isinstance(base_type, pl.DataType):
        return base_type
    if isinstance(base_type, type) and issubclass(base_type, pl.DataType):
        return base_type()
    return TYPE_TO_DTYPE.get(base_type, pl.Utf8())
