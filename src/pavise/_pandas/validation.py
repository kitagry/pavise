"""Common validation functions for DataFrame schema checking."""

from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
import pandas as pd

from pavise.checks import DataFrameCheck, get_dataframe_checks
from pavise.exceptions import ValidationError
from pavise.types import NotRequiredColumn

# Special column name for index validation
INDEX_COLUMN_NAME = "__index__"

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5


@dataclass
class TypeChecker:
    """Type checker with both dtype-level and value-level validation."""

    dtype: Callable[[pd.Series | pd.Index], bool]  # Check dtype of entire Series
    value: Callable[[Any], bool]  # Check individual values


def type_check_str(df: pd.Series | pd.Index) -> bool:
    df = pd.Series(df)
    if pd.api.types.is_string_dtype(df):
        return True
    elif df.dtype == object:
        # infer_dtype scans in C; only fall back to the exact per-element check when it is unsure.
        if pd.api.types.infer_dtype(df, skipna=True) in ("string", "empty"):
            return True
        return not _invalid_value_mask(df, lambda x: isinstance(x, str), null_valid=True).any()
    return False


def type_check_date(df: pd.Series | pd.Index) -> bool:
    df = pd.Series(df)
    if pd.api.types.is_datetime64_any_dtype(df):
        return True
    elif df.dtype == object:
        if pd.api.types.infer_dtype(df, skipna=True) in ("date", "datetime", "empty"):
            return True
        return not _invalid_value_mask(df, lambda x: isinstance(x, date), null_valid=True).any()
    return False


def _invalid_value_mask(
    series: pd.Series, is_valid: Callable[[Any], bool], null_valid: bool
) -> pd.Series:
    """
    Return a boolean mask of values rejected by ``is_valid``, calling it as rarely as possible.

    ``is_valid`` must depend only on the value's type for non-null values (an isinstance check),
    which holds for every TYPE_CHECKERS value checker.
    """
    if isinstance(series.dtype, pd.SparseDtype):
        # SparseArray.map rejects mappers that send a stored value to the fill value's result,
        # and Sparse[object] may mix Python types, so check the dense values instead.
        series = series.sparse.to_dense()
    na = series.isna().to_numpy(dtype=bool)
    valid = np.full(len(series), null_valid, dtype=bool)
    nonnull = ~na
    if not nonnull.any():
        return pd.Series(~valid, index=series.index)

    if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
        cache: dict[type, bool] = {}

        def check(value: Any) -> bool:
            value_type = type(value)
            if value_type not in cache:
                cache[value_type] = bool(is_valid(value))
            return cache[value_type]

        valid[nonnull] = series.loc[nonnull].apply(check).to_numpy(dtype=bool)
    else:
        # Non-null values of any other dtype all share one Python type, so a single element
        # decides for the whole column. Slice + apply to box the value exactly as apply would.
        first = int(np.argmax(nonnull))
        valid[nonnull] = bool(series.iloc[first : first + 1].apply(is_valid).iloc[0])
    return pd.Series(~valid, index=series.index)


def _collect_samples(invalid: pd.Series) -> list[tuple[Any, Any]]:
    """Return up to MAX_SAMPLE_SIZE (index, value) pairs from the invalid values."""
    return [(idx, invalid[idx]) for idx in invalid.head(MAX_SAMPLE_SIZE).index]


TYPE_CHECKERS = {
    int: TypeChecker(
        dtype=pd.api.types.is_integer_dtype,
        value=lambda x: isinstance(x, (int, np.integer)) or pd.isna(x),
    ),
    float: TypeChecker(
        dtype=pd.api.types.is_float_dtype,
        value=lambda x: isinstance(x, (float, np.floating)) or pd.isna(x),
    ),
    str: TypeChecker(
        dtype=type_check_str,
        value=lambda x: isinstance(x, str) or pd.isna(x),
    ),
    bool: TypeChecker(
        dtype=pd.api.types.is_bool_dtype,
        value=lambda x: isinstance(x, (bool, np.bool_)) or pd.isna(x),
    ),
    datetime: TypeChecker(
        dtype=pd.api.types.is_datetime64_any_dtype,
        value=lambda x: isinstance(x, (pd.Timestamp, datetime)) or pd.isna(x),
    ),
    date: TypeChecker(
        dtype=type_check_date,
        value=lambda x: isinstance(x, (pd.Timestamp, datetime, date)) or pd.isna(x),
    ),
    timedelta: TypeChecker(
        dtype=pd.api.types.is_timedelta64_dtype,
        value=lambda x: isinstance(x, (pd.Timedelta, timedelta)) or pd.isna(x),
    ),
    Any: TypeChecker(
        dtype=lambda _: True,
        value=lambda _: True,
    ),
}

# Mapping from Python type to pandas dtype for creating empty DataFrames
TYPE_TO_DTYPE = {
    int: "int64",
    float: "float64",
    str: "object",
    bool: "bool",
    datetime: "datetime64[ns]",
    date: "datetime64[ns]",
    timedelta: "timedelta64[ns]",
    Any: "object",
}


def validate_dataframe(df: pd.DataFrame, schema: type, strict: bool = False) -> None:
    """
    Validate that a DataFrame conforms to a Protocol schema.

    Args:
        df: DataFrame to validate
        schema: Protocol type defining the expected schema
        strict: If True, raise error on extra columns not in schema

    Raises:
        ValueError: If a required column is missing or type is unsupported
        TypeError: If a column has the wrong type
    """
    expected_cols = get_type_hints(schema, include_extras=True)

    for col_name, col_type in expected_cols.items():
        if col_name == INDEX_COLUMN_NAME:
            _check_index_type(df, col_type)
        else:
            is_not_required = isinstance(col_type, type) and issubclass(col_type, NotRequiredColumn)
            if is_not_required and col_name not in df.columns:
                continue
            _check_column_exists(df, col_name)
            _check_column_type(df, col_name, col_type)

    if strict:
        schema_cols = set(expected_cols.keys())
        if INDEX_COLUMN_NAME in schema_cols:
            schema_cols.remove(INDEX_COLUMN_NAME)
        df_cols = set(df.columns)
        extra_cols = df_cols - schema_cols
        if extra_cols:
            raise ValidationError(f"Strict mode: unexpected columns {sorted(extra_cols)}")

    schema_cols = [c for c in expected_cols if c != INDEX_COLUMN_NAME and c in df.columns]
    for check in get_dataframe_checks(schema):
        _run_dataframe_check(df, check, schema_cols)


def _run_dataframe_check(df: pd.DataFrame, check: DataFrameCheck, schema_cols: list[str]) -> None:
    """Run a DataFrame-level check and raise with sample violating rows if it fails."""
    result = check.func(df)
    base_message = f"DataFrame check '{check.name}' failed"

    if isinstance(result, (bool, np.bool_)):
        if not result:
            raise ValidationError(base_message)
        return

    if not isinstance(result, pd.Series) or not pd.api.types.is_bool_dtype(result):
        raise ValidationError(
            f"DataFrame check '{check.name}' must return a boolean Series or bool, "
            f"got {type(result).__name__}"
        )

    # Null means the check could not be evaluated for the row, which counts as passing.
    invalid_mask = ~result.fillna(True).astype(bool)
    if not invalid_mask.any():
        return

    invalid_rows = df.loc[invalid_mask, schema_cols].head(MAX_SAMPLE_SIZE)
    samples: list[tuple[Any, ...]] = list(zip(invalid_rows.index, invalid_rows.to_dict("records")))
    raise ValidationError.new_with_samples(
        None, base_message, samples, int(invalid_mask.sum()), repr
    )


def _extract_type_and_validators(
    annotation: type,
) -> tuple[type | tuple[type, ...], list[Any], bool, bool]:
    """
    Extract base type, validators, nullable flag, and not-required flag from a type annotation.

    Args:
        annotation: Type annotation (e.g., int, Optional[int], NotRequiredColumn[int],
            or Annotated[int, Range(0, 100)])

    Returns:
        Tuple of (base_type, validators, is_optional, is_not_required)
        - For Annotated[int, Range(0, 100)]: (int, [Range(0, 100)], False, False)
        - For int: (int, [], False, False)
        - For Optional[int]: (int, [], True, False)
        - For NotRequiredColumn[int]: (int, [], False, True)
        - For NotRequiredColumn[Optional[int]]: (int, [], True, True)
        - For Union[int, str]: ((int, str), [], False, False)
        - For Union[int, str, None]: ((int, str), [], True, False)
    """
    validators = []
    is_optional = False
    is_not_required = False

    if isinstance(annotation, type) and issubclass(annotation, NotRequiredColumn):
        is_not_required = True
        annotation = getattr(annotation, "_inner_type", annotation)

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        validators = list(args[1:])
        annotation = base_type

    origin = get_origin(annotation)
    # Handle both Union[int, str] and int | str (PEP 604)
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        # Check if None is in the union
        if type(None) in args:
            is_optional = True
            # Remove None from the union
            non_none_types = tuple(arg for arg in args if arg is not type(None))
            # If only one type remains after removing None, return it as a single type
            if len(non_none_types) == 1:
                base_type = non_none_types[0]
                return base_type, validators, is_optional, is_not_required
            # Otherwise, return the tuple of types
            return non_none_types, validators, is_optional, is_not_required
        # Union without None - return all types as a tuple
        return args, validators, is_optional, is_not_required

    return annotation, validators, is_optional, is_not_required


def _extract_index_name_type_and_validators(
    annotation: type,
) -> tuple[type, str | tuple[str, ...] | None, list[Any], bool]:
    """
    Extract base type, index name, validators, and nullable flag from an index type annotation.

    Args:
        annotation: Type annotation for index (e.g., int, Annotated[int, "user_id"])

    Returns:
        Tuple of (base_type, index_name, validators, is_optional)
        - For Annotated[int, "user_id"]: (int, "user_id", [], False)
        - For Annotated[int, "user_id", Range(0, 100)]: (int, "user_id", [Range(0, 100)], False)
        - For Annotated[tuple[str, int], ("region", "user_id")]:
            (tuple[str, int], ("region", "user_id"), [], False)
        - For Annotated[int, Range(0, 100)]: (int, None, [Range(0, 100)], False)
        - For int: (int, None, [], False)
    """
    index_name = None
    validators = []
    is_optional = False

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        metadata = list(args[1:])

        if metadata and isinstance(metadata[0], (str, tuple)):
            index_name = metadata[0]
            validators = metadata[1:]
        else:
            validators = metadata

        annotation = base_type

    origin = get_origin(annotation)
    # Handle both Union[int, str] and int | str (PEP 604)
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            is_optional = True
            base_type = args[0] if args[1] is type(None) else args[1]
            return base_type, index_name, validators, is_optional

    return annotation, index_name, validators, is_optional


def _check_index_type(df: pd.DataFrame, expected_type: type) -> None:
    """Check if index has the expected type and name."""
    from pavise._pandas.validator_impl import apply_validator

    base_type, index_name, validators, is_optional = _extract_index_name_type_and_validators(
        expected_type
    )

    if get_origin(base_type) is tuple:
        if not (index_name is None or isinstance(index_name, tuple)):
            raise ValidationError("MultiIndex must have a tuple of names specified in Annotated")
        _check_multiindex_type(df, base_type, index_name, validators, is_optional)
    else:
        if index_name is not None:
            if df.index.name != index_name:
                raise ValidationError(f"Index name expected '{index_name}', got {df.index.name!r}")

        if base_type not in TYPE_CHECKERS:
            raise ValidationError(f"unsupported type: {base_type}")

        checker = TYPE_CHECKERS[base_type]
        if not checker.dtype(df.index):
            raise ValidationError(f"Index expected {base_type.__name__}, got {df.index.dtype}")

        index_series = pd.Series(df.index)
        for validator in validators:
            apply_validator(index_series, validator, "Index")


def _check_multiindex_type(
    df: pd.DataFrame,
    expected_types: type,
    index_names: tuple[str, ...] | None,
    validators: list[Any],
    is_optional: bool,
) -> None:
    """Check if MultiIndex has the expected types and names."""
    from pavise._pandas.validator_impl import apply_validator

    level_types = get_args(expected_types)

    if not isinstance(df.index, pd.MultiIndex):
        raise ValidationError(
            f"Expected MultiIndex with {len(level_types)} levels, got {type(df.index).__name__}"
        )

    if len(df.index.levels) != len(level_types):
        raise ValidationError(
            f"Expected MultiIndex with {len(level_types)} levels, got {len(df.index.levels)}"
        )

    if index_names is not None:
        if tuple(df.index.names) != index_names:
            raise ValidationError(
                f"Index names expected {index_names!r}, got {tuple(df.index.names)!r}"
            )

    for level_idx, level_type in enumerate(level_types):
        if level_type not in TYPE_CHECKERS:
            raise ValidationError(f"Unsupported type: {level_type}")

        checker = TYPE_CHECKERS[level_type]
        level_data = df.index.get_level_values(level_idx)

        if not checker.dtype(level_data):
            raise ValidationError(
                f"Index level {level_idx} expected {level_type.__name__}, got {level_data.dtype}"
            )

    if validators:
        for level_idx in range(len(level_types)):
            level_series = pd.Series(df.index.get_level_values(level_idx))
            for validator in validators:
                apply_validator(level_series, validator, f"Index level {level_idx}")


def _check_column_exists(df: pd.DataFrame, col_name: str) -> None:
    """Check if a column exists in the DataFrame."""
    if col_name not in df.columns:
        raise ValidationError("missing", column_name=col_name)


def _check_column_type(df: pd.DataFrame, col_name: str, expected_type: type) -> None:
    """Check if a column has the expected type and apply validators."""
    from pavise._pandas.validator_impl import apply_validator

    base_type, validators, is_optional, _is_not_required = _extract_type_and_validators(
        expected_type
    )
    col_series: pd.Series = df[col_name]  # type: ignore[assignment]

    if isinstance(base_type, type) and issubclass(base_type, pd.api.extensions.ExtensionDtype):
        col_dtype = col_series.dtype
        if type(col_dtype) is not base_type:
            base_tname = base_type.__name__
            col_tname = type(col_dtype).__name__
            raise ValidationError(
                f"expected {base_tname}, got {col_tname}",
                column_name=col_name,
            )
        for validator in validators:
            apply_validator(col_series, validator, col_name)
        return

    if get_origin(base_type) is Literal:
        allowed_values = get_args(base_type)
        invalid_mask = ~col_series.isin(allowed_values)
        if invalid_mask.any():
            invalid_df: pd.Series = col_series[invalid_mask]  # type: ignore[assignment]
            raise ValidationError.new_with_samples(
                col_name,
                f"expected one of {allowed_values}, got invalid values",
                _collect_samples(invalid_df),
                len(invalid_df),
                repr,
            )
        for validator in validators:
            apply_validator(col_series, validator, col_name)
        return

    # Handle Union types (represented as tuple)
    if isinstance(base_type, tuple):
        union_types = base_type
        # Check if all types in the union are supported
        for union_type in union_types:
            if union_type not in TYPE_CHECKERS:
                raise ValidationError(f"unsupported type: {union_type}", column_name=col_name)

        # Check if each value matches at least one of the union types
        invalid_mask = _invalid_value_mask(
            col_series,
            lambda value: any(TYPE_CHECKERS[t].value(value) for t in union_types),
            null_valid=is_optional,
        )
        if bool(invalid_mask.any()):
            invalid_df = col_series[invalid_mask]  # type: ignore[assignment]
            union_type_names = " | ".join(t.__name__ for t in union_types)
            raise ValidationError.new_with_samples(
                col_name,
                f"expected {union_type_names}, got {col_series.dtype}",
                _collect_samples(invalid_df),
                len(invalid_df),
                repr,
            )

        for validator in validators:
            apply_validator(col_series, validator, col_name)
        return

    if base_type not in TYPE_CHECKERS:
        raise ValidationError(f"unsupported type: {base_type}", column_name=col_name)

    checker = TYPE_CHECKERS[base_type]
    col_dtype = col_series.dtype

    if not is_optional and col_series.isna().any():
        raise ValidationError("is non-optional but contains null values", column_name=col_name)
    if not checker.dtype(col_series):
        invalid_mask = _invalid_value_mask(col_series, checker.value, null_valid=True)
        invalid_df = col_series[invalid_mask]  # type: ignore[assignment]
        raise ValidationError.new_with_samples(
            col_name,
            f"expected {base_type.__name__}, got {col_dtype}",
            _collect_samples(invalid_df),
            len(invalid_df),
            repr,
        )

    for validator in validators:
        apply_validator(col_series, validator, col_name)
