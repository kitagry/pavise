"""Validation of List and Struct columns for polars."""

from __future__ import annotations

from typing import Any, Literal, NoReturn, get_args, get_origin

import polars as pl

from pavise._annotations import (
    Field,
    ListOf,
    Scalar,
    StructOf,
    format_field,
    literal_base_type,
    nested_error,
    nested_path,
)
from pavise._polars.dtypes import TYPE_CHECKERS, dtype_for_type, dtype_name, is_dtype_annotation
from pavise._polars.validator_impl import MAX_SAMPLE_SIZE
from pavise.exceptions import ValidationError


def check_nested_dtype(field: Field, dtype: pl.DataType, column_name: str, strict: bool) -> None:
    """Check a List or Struct column's dtype against its annotation, level by level."""
    _check_dtype(field, dtype, column_name, column_name, strict)


def _check_dtype(
    field: Field, dtype: pl.DataType, column_name: str, path: str, strict: bool
) -> None:
    # polars infers Null from nulls and empty lists alone; whether nulls are allowed is
    # checked on the values
    if dtype == pl.Null:
        return

    node = field.node
    if isinstance(node, ListOf):
        if not isinstance(dtype, (pl.List, pl.Array)):
            raise nested_error(f"expected {format_field(field)}, got {dtype}", column_name, path)
        _check_dtype(node.element, dtype.inner, column_name, f"{path}[]", strict)  # type: ignore[arg-type]
    elif isinstance(node, StructOf):
        if not isinstance(dtype, pl.Struct):
            raise nested_error(f"expected {format_field(field)}, got {dtype}", column_name, path)
        actual = {f.name: f.dtype for f in dtype.fields}
        for name, sub in node.fields.items():
            if name not in actual:
                if sub.not_required:
                    continue
                raise nested_error(
                    f"missing struct field '{name}' (found fields: {list(actual)})",
                    column_name,
                    path,
                )
            _check_dtype(sub, actual[name], column_name, f"{path}.{name}", strict)  # type: ignore[arg-type]
        extra = sorted(set(actual) - set(node.fields))
        if strict and extra:
            raise nested_error(f"Strict mode: unexpected struct fields {extra}", column_name, path)
    else:
        _check_scalar_dtype(node, dtype, column_name, path)


def _check_scalar_dtype(node: Scalar, dtype: pl.DataType, column_name: str, path: str) -> None:
    expected = literal_base_type(node.type)
    if is_dtype_annotation(expected):
        matches = dtype == expected
    elif expected in TYPE_CHECKERS:
        matches = TYPE_CHECKERS[expected].dtype(dtype)
    else:
        raise nested_error(f"unsupported type: {dtype_name(expected)}", column_name, path)

    if not matches:
        raise nested_error(f"expected {dtype_name(expected)}, got {dtype}", column_name, path)


def check_nested_values(field: Field, series: pl.Series, column_name: str) -> None:
    """Reject nulls that are not Optional and values outside a Literal, at every level."""
    rows = pl.Series(range(series.len()), dtype=pl.UInt32)
    _check_values(field, series, rows, series, column_name, column_name)


def _raise_for_rows(
    message: str, rows: pl.Series, column: pl.Series, column_name: str, path: str
) -> NoReturn:
    """Raise for the values found in ``rows``, showing the whole rows as samples."""
    invalid_rows = rows.unique(maintain_order=True)
    sample_rows = invalid_rows.head(MAX_SAMPLE_SIZE)
    samples: list[tuple[Any, ...]] = list(
        zip(sample_rows.to_list(), column.gather(sample_rows).to_list())
    )
    raise ValidationError.new_with_samples(
        column_name, message, samples, invalid_rows.len(), path=nested_path(column_name, path)
    )


def _check_values(
    field: Field,
    values: pl.Series,
    rows: pl.Series,
    column: pl.Series,
    column_name: str,
    path: str,
) -> None:
    """
    Check nulls of ``values``, the values found at ``path``.

    ``rows`` holds the column row each value comes from, so errors can show the whole row.
    """
    is_null = values.is_null()
    if is_null.any():
        if not field.nullable:
            _raise_for_rows(
                "is non-optional but contains null values",
                rows.filter(is_null),
                column,
                column_name,
                path,
            )
        values = values.filter(~is_null)
        rows = rows.filter(~is_null)
    if values.dtype == pl.Null:
        return

    node = field.node
    if isinstance(node, ListOf):
        if isinstance(values.dtype, pl.Array):
            values = values.cast(pl.List(values.dtype.inner))  # type: ignore[arg-type]
        lengths = values.list.len()
        # Exploding an empty list yields a null, so leave empty lists out
        non_empty = lengths > 0
        values = values.filter(non_empty)
        rows = rows.filter(non_empty).repeat_by(lengths.filter(non_empty)).explode()
        _check_values(node.element, values.explode(), rows, column, column_name, f"{path}[]")
    elif isinstance(node, StructOf):
        present = {f.name for f in values.dtype.fields}  # type: ignore[attr-defined]
        for name, sub in node.fields.items():
            if name in present:
                _check_values(
                    sub, values.struct.field(name), rows, column, column_name, f"{path}.{name}"
                )
    elif get_origin(node.type) is Literal:
        allowed = get_args(node.type)
        invalid = ~values.is_in([v for v in allowed if v is not None])
        if invalid.any():
            _raise_for_rows(
                f"expected one of {allowed}, got invalid values",
                rows.filter(invalid),
                column,
                column_name,
                path,
            )


def nested_dtype(field: Field) -> pl.DataType:
    """Build the polars dtype of an empty List or Struct column."""
    node = field.node
    if isinstance(node, ListOf):
        return pl.List(nested_dtype(node.element))
    if isinstance(node, StructOf):
        return pl.Struct({name: nested_dtype(sub) for name, sub in node.fields.items()})
    return dtype_for_type(node.type)
