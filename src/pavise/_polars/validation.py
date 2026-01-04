"""Common validation functions for Polars DataFrame schema checking."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Annotated, Callable, Literal, Union, get_args, get_origin, get_type_hints

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise.exceptions import ValidationError

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5
# Maximum number of rows to check for type validation (performance optimization)
MAX_CHECK_ROWS = 100


@dataclass
class TypeChecker:
    """Type checker with both dtype-level and value-level validation."""

    dtype: Callable  # Check dtype
    value: Callable[[object], bool]  # Check individual values


def _is_datetime_dtype(dtype):
    """Check if dtype is a Datetime type (with any time unit)."""
    if dtype == pl.Datetime:
        return True
    if hasattr(dtype, "time_unit"):
        return True
    return False


TYPE_CHECKERS = {
    int: TypeChecker(
        dtype=lambda dtype: dtype
        in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
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
}


def validate_dataframe(df: pl.DataFrame, schema: type, strict: bool = False) -> None:
    """
    Validate that a Polars DataFrame conforms to a Protocol schema.

    Args:
        df: Polars DataFrame to validate
        schema: Protocol type defining the expected schema
        strict: If True, raise error on extra columns not in schema

    Raises:
        ValueError: If a required column is missing or type is unsupported
        TypeError: If a column has the wrong type
    """
    expected_cols = get_type_hints(schema, include_extras=True)

    for col_name, col_type in expected_cols.items():
        _check_column_exists(df, col_name)
        _check_column_type(df, col_name, col_type)

    if strict:
        schema_cols = set(expected_cols.keys())
        df_cols = set(df.columns)
        extra_cols = df_cols - schema_cols
        if extra_cols:
            raise ValidationError(f"Strict mode: unexpected columns {sorted(extra_cols)}")


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


def _raise_type_error_with_samples(
    df: pl.DataFrame, col_name: str, checker: TypeChecker, expected_type: type, actual_dtype
) -> None:
    """Raise TypeError with sample invalid values."""
    invalid_mask = df[col_name].map_elements(lambda v: not checker.value(v))
    invalid_df = df[col_name].to_frame().with_row_index("__row__").filter(invalid_mask)
    samples = [
        (row["__row__"], row[col_name])
        for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
    ]
    total_invalid = int(invalid_mask.sum())

    raise ValidationError.new_with_samples(
        col_name,
        f"expected {expected_type.__name__}, got {actual_dtype}",
        samples,
        total_invalid,
    )


def _check_column_exists(df: pl.DataFrame, col_name: str) -> None:
    """Check if a column exists in the DataFrame."""
    if col_name not in df.columns:
        raise ValidationError("missing", column_name=col_name)


def _check_column_type(df: pl.DataFrame, col_name: str, expected_type: type) -> None:
    """Check if a column has the expected type and apply validators."""
    from pavise._polars.validator_impl import apply_validator

    base_type, validators, is_optional = _extract_type_and_validators(expected_type)

    if isinstance(base_type, type) and issubclass(base_type, pl.DataType):
        col_dtype = df[col_name].dtype
        if col_dtype != base_type:
            raise ValidationError(
                f"expected {base_type.__name__}, got {col_dtype}",
                column_name=col_name,
            )
        for validator in validators:
            apply_validator(df[col_name], validator, col_name)
        return

    if get_origin(base_type) is Literal:
        allowed_values = get_args(base_type)
        invalid_mask = ~df[col_name].is_in(allowed_values)
        if invalid_mask.any():
            invalid_df = df[col_name].to_frame().with_row_index("__row__").filter(invalid_mask)
            samples = [
                (row["__row__"], row[col_name])
                for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
            ]
            total_invalid = int(invalid_mask.sum())
            raise ValidationError.new_with_samples(
                col_name,
                f"expected one of {allowed_values}, got invalid values",
                samples,
                total_invalid,
                repr,
            )
        for validator in validators:
            apply_validator(df[col_name], validator, col_name)
        return

    if base_type not in TYPE_CHECKERS:
        raise ValidationError(f"unsupported type: {base_type}", column_name=col_name)

    checker = TYPE_CHECKERS[base_type]
    col_dtype = df[col_name].dtype
    if not checker.dtype(col_dtype):
        _raise_type_error_with_samples(df, col_name, checker, base_type, col_dtype)

    if not is_optional and df[col_name].null_count() > 0:
        raise ValidationError("is non-optional but contains null values", column_name=col_name)

    for validator in validators:
        apply_validator(df[col_name], validator, col_name)
