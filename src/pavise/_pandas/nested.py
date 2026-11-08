"""Validation of List and Struct columns for pandas, which supports them only as pd.ArrowDtype."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal, NoReturn, get_args, get_origin

import numpy as np
import pandas as pd

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
from pavise._pandas.validator_impl import MAX_SAMPLE_SIZE, _to_python_scalar
from pavise.exceptions import ValidationError

# pyarrow.types predicates accepted for each scalar type inside a List or Struct column. Names,
# not functions, so pyarrow is imported only when such a column is checked; a predicate missing
# from an older pyarrow simply never matches.
_ARROW_PREDICATES: dict[Any, tuple[str, ...]] = {
    int: ("is_integer",),
    float: ("is_floating",),
    str: ("is_string", "is_large_string", "is_string_view"),
    bool: ("is_boolean",),
    datetime: ("is_timestamp",),
    # Matches top-level date columns, which accept timestamps as well
    date: ("is_date", "is_timestamp"),
    timedelta: ("is_duration",),
}

_LIST_PREDICATES = (
    "is_list",
    "is_large_list",
    "is_fixed_size_list",
    "is_list_view",
    "is_large_list_view",
)

_ARROW_DTYPE_HINT = (
    "\n  (pavise checks list and struct columns only with pd.ArrowDtype, "
    "e.g. dtype=pd.ArrowDtype(pa.list_(pa.string())))"
)


def _matches(arrow_type: Any, predicate_names: tuple[str, ...]) -> bool:
    import pyarrow as pa

    return any(getattr(pa.types, name, lambda _: False)(arrow_type) for name in predicate_names)


def check_nested_column(field: Field, series: pd.Series, column_name: str, strict: bool) -> None:
    """Check a List or Struct column's dtype level by level, then its values."""
    arrow_type = getattr(series.dtype, "pyarrow_dtype", None)
    if arrow_type is None:
        raise ValidationError(
            f"expected {format_field(field)}, got {series.dtype}{_ARROW_DTYPE_HINT}",
            column_name=column_name,
        )
    _check_type(field, arrow_type, str(series.dtype), column_name, column_name, strict)

    import pyarrow as pa

    values = pa.array(series)
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    positions = np.arange(len(series))
    _check_values(field, values, positions, series, values, column_name, column_name)


def _check_type(
    field: Field, arrow_type: Any, shown: str, column_name: str, path: str, strict: bool
) -> None:
    """Compare ``arrow_type`` with ``field``; ``shown`` is how the type reads in errors."""
    import pyarrow as pa

    # Arrow infers null from nulls and empty lists alone; whether nulls are allowed is
    # checked on the values
    if pa.types.is_null(arrow_type):
        return

    node = field.node
    if isinstance(node, ListOf):
        if not _matches(arrow_type, _LIST_PREDICATES):
            raise nested_error(f"expected {format_field(field)}, got {shown}", column_name, path)
        element_type = arrow_type.value_type
        _check_type(node.element, element_type, str(element_type), column_name, f"{path}[]", strict)
    elif isinstance(node, StructOf):
        if not pa.types.is_struct(arrow_type):
            raise nested_error(f"expected {format_field(field)}, got {shown}", column_name, path)
        actual = {
            arrow_type.field(i).name: arrow_type.field(i).type for i in range(arrow_type.num_fields)
        }
        for name, sub in node.fields.items():
            if name not in actual:
                if sub.not_required:
                    continue
                raise nested_error(
                    f"missing struct field '{name}' (found fields: {list(actual)})",
                    column_name,
                    path,
                )
            sub_type = actual[name]
            _check_type(sub, sub_type, str(sub_type), column_name, f"{path}.{name}", strict)
        extra = sorted(set(actual) - set(node.fields))
        if strict and extra:
            raise nested_error(f"Strict mode: unexpected struct fields {extra}", column_name, path)
    else:
        _check_scalar_type(node, arrow_type, shown, column_name, path)


def _check_scalar_type(
    node: Scalar, arrow_type: Any, shown: str, column_name: str, path: str
) -> None:
    expected = literal_base_type(node.type)
    if expected is Any:
        return
    name = getattr(expected, "__name__", str(expected))
    if expected not in _ARROW_PREDICATES:
        raise nested_error(f"unsupported type: {name}", column_name, path)
    if not _matches(arrow_type, _ARROW_PREDICATES[expected]):
        raise nested_error(f"expected {name}, got {shown}", column_name, path)


def _raise_for_positions(
    message: str,
    positions: np.ndarray,
    series: pd.Series,
    column: Any,
    column_name: str,
    path: str,
) -> NoReturn:
    """Raise for the values found at row ``positions``, showing the whole rows as samples."""
    invalid_positions: list[int] = np.unique(positions).tolist()
    samples: list[tuple[Any, ...]] = [
        (_to_python_scalar(series.index[pos]), column[pos].as_py())
        for pos in invalid_positions[:MAX_SAMPLE_SIZE]
    ]
    raise ValidationError.new_with_samples(
        column_name, message, samples, len(invalid_positions), path=nested_path(column_name, path)
    )


def _check_values(
    field: Field,
    values: Any,
    positions: np.ndarray,
    series: pd.Series,
    column: Any,
    column_name: str,
    path: str,
) -> None:
    """
    Check nulls and Literal values of ``values``, the Arrow values found at ``path``.

    ``positions`` holds the row position each value comes from, so errors can show the whole
    row (``column`` is the column as an Arrow array, ``series`` gives the row labels).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    is_null = values.is_null().to_numpy(zero_copy_only=False)
    if is_null.any():
        if not field.nullable:
            _raise_for_positions(
                "is non-optional but contains null values",
                positions[is_null],
                series,
                column,
                column_name,
                path,
            )
        keep = ~is_null
        values = values.filter(pa.array(keep))
        positions = positions[keep]
    if pa.types.is_null(values.type):
        return

    node = field.node
    if isinstance(node, ListOf):
        parents = pc.call_function("list_parent_indices", [values])
        _check_values(
            node.element,
            pc.call_function("list_flatten", [values]),
            positions[parents.to_numpy(zero_copy_only=False)],
            series,
            column,
            column_name,
            f"{path}[]",
        )
    elif isinstance(node, StructOf):
        present = {values.type.field(i).name for i in range(values.type.num_fields)}
        for name, sub in node.fields.items():
            if name in present:
                _check_values(
                    sub,
                    pc.call_function(
                        "struct_field",
                        [values],
                        pc.StructFieldOptions([values.type.get_field_index(name)]),
                    ),
                    positions,
                    series,
                    column,
                    column_name,
                    f"{path}.{name}",
                )
    elif get_origin(node.type) is Literal:
        allowed = get_args(node.type)
        non_null = [v for v in allowed if v is not None]
        value_set = pa.array(non_null) if non_null else pa.array([], type=values.type)
        in_set = pc.call_function("is_in", [values], pc.SetLookupOptions(value_set))
        invalid = ~in_set.to_numpy(zero_copy_only=False)
        if invalid.any():
            _raise_for_positions(
                f"expected one of {allowed}, got invalid values",
                positions[invalid],
                series,
                column,
                column_name,
                path,
            )


def nested_dtype(field: Field) -> pd.ArrowDtype:
    """Build the pd.ArrowDtype of an empty List or Struct column."""
    return pd.ArrowDtype(_arrow_type(field))


def _arrow_type(field: Field) -> Any:
    import pyarrow as pa

    node = field.node
    if isinstance(node, ListOf):
        return pa.list_(_arrow_type(node.element))
    if isinstance(node, StructOf):
        return pa.struct([(name, _arrow_type(sub)) for name, sub in node.fields.items()])

    return {
        int: pa.int64(),
        float: pa.float64(),
        bool: pa.bool_(),
        datetime: pa.timestamp("ns"),
        date: pa.date32(),
        timedelta: pa.duration("ns"),
    }.get(literal_base_type(node.type), pa.string())
