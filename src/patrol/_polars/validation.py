"""Common validation functions for Polars DataFrame schema checking."""

from datetime import date, datetime, timedelta
from typing import Annotated, Union, get_args, get_origin, get_type_hints

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install patrol[polars]")


def _is_datetime_dtype(dtype):
    """Check if dtype is a Datetime type (with any time unit)."""
    if dtype == pl.Datetime:
        return True
    if hasattr(dtype, "time_unit"):
        return True
    return False


TYPE_CHECKERS = {
    int: lambda dtype: dtype
    in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    float: lambda dtype: dtype in (pl.Float32, pl.Float64),
    str: lambda dtype: dtype == pl.Utf8,
    bool: lambda dtype: dtype == pl.Boolean,
    datetime: _is_datetime_dtype,
    date: lambda dtype: dtype == pl.Date,
    timedelta: lambda dtype: dtype == pl.Duration,
}


def validate_dataframe(df: pl.DataFrame, schema: type) -> None:
    """
    Validate that a Polars DataFrame conforms to a Protocol schema.

    Args:
        df: Polars DataFrame to validate
        schema: Protocol type defining the expected schema

    Raises:
        ValueError: If a required column is missing or type is unsupported
        TypeError: If a column has the wrong type
    """
    expected_cols = get_type_hints(schema, include_extras=True)

    for col_name, col_type in expected_cols.items():
        _check_column_exists(df, col_name)
        _check_column_type(df, col_name, col_type)


def _extract_type_and_validators(annotation: type) -> tuple[type, list, bool]:
    """
    Extract base type, validators, and nullable flag from a type annotation.

    Args:
        annotation: Type annotation (e.g., int, Optional[int], or Annotated[int, Range(0, 100)])

    Returns:
        Tuple of (base_type, validators, is_optional)
        - For Annotated[int, Range(0, 100)]: (int, [Range(0, 100)], False)
        - For int: (int, [], False)
        - For Optional[int]: (int, [], True)
        - For Annotated[Optional[int], Range(0, 100)]: (int, [Range(0, 100)], True)
    """
    validators = []
    is_optional = False

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        validators = list(args[1:])
        annotation = base_type

    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        if len(args) == 2 and type(None) in args:
            is_optional = True
            base_type = args[0] if args[1] is type(None) else args[1]
            return base_type, validators, is_optional

    return annotation, validators, is_optional


def _check_column_exists(df: pl.DataFrame, col_name: str) -> None:
    """Check if a column exists in the DataFrame."""
    if col_name not in df.columns:
        raise ValueError(f"Missing column: {col_name}")


def _check_column_type(df: pl.DataFrame, col_name: str, expected_type: type) -> None:
    """Check if a column has the expected type and apply validators."""
    from patrol._polars.validator_impl import apply_validator

    base_type, validators, is_optional = _extract_type_and_validators(expected_type)

    if isinstance(base_type, type) and issubclass(base_type, pl.DataType):
        col_dtype = df[col_name].dtype
        if col_dtype != base_type:
            raise TypeError(f"Column '{col_name}' expected {base_type.__name__}, got {col_dtype}")
        for validator in validators:
            apply_validator(df[col_name], validator, col_name)
        return

    if base_type not in TYPE_CHECKERS:
        raise ValueError(f"Unsupported type: {base_type}")

    type_checker = TYPE_CHECKERS[base_type]
    col_dtype = df[col_name].dtype
    if not type_checker(col_dtype):
        raise TypeError(f"Column '{col_name}' expected {base_type.__name__}, got {col_dtype}")

    if not is_optional and df[col_name].null_count() > 0:
        raise TypeError(f"Column '{col_name}' is non-optional but contains null values")

    for validator in validators:
        apply_validator(df[col_name], validator, col_name)
