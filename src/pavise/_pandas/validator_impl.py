"""Pandas-specific validator implementations."""

import operator
from collections.abc import Callable
from typing import Any

import pandas as pd

from pavise.exceptions import ValidationError
from pavise.validators import Custom, Ge, Gt, In, Le, Lt, MaxLen, MinLen, Range, Regex, Unique

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5

# Maps each comparison validator to the operator that detects violations and its symbol
_COMPARISONS = {
    Ge: (operator.lt, ">="),
    Gt: (operator.le, ">"),
    Le: (operator.gt, "<="),
    Lt: (operator.ge, "<"),
}

_BOUND_VALIDATORS = (Range, *_COMPARISONS)

# infer_dtype kinds of object columns whose values can be ordered against a bound
_ORDERABLE_OBJECT_KINDS = frozenset(
    {
        "integer",
        "floating",
        "mixed-integer-float",
        "decimal",
        "date",
        "datetime",
        "datetime64",
        "timedelta",
        "timedelta64",
        "time",
    }
)


def apply_validator(series: pd.Series, validator: Any, col_name: str) -> None:
    """
    Apply a validator to a pandas Series.

    Args:
        series: pandas Series to validate
        validator: Validator instance (e.g., Range, Unique, In, Regex, MinLen, MaxLen)
        col_name: column name for error messages

    Raises:
        ValueError: if validation fails
    """
    if isinstance(validator, _BOUND_VALIDATORS):
        _require_orderable(series, validator, col_name)

    if isinstance(validator, Range):
        _validate_range(series, validator, col_name)
    elif isinstance(validator, (Ge, Gt, Le, Lt)):
        _validate_comparison(series, validator, col_name)
    elif isinstance(validator, Unique):
        _validate_unique(series, col_name)
    elif isinstance(validator, In):
        _validate_in(series, validator, col_name)
    elif isinstance(validator, Regex):
        _validate_regex(series, validator, col_name)
    elif isinstance(validator, MinLen):
        _validate_minlen(series, validator, col_name)
    elif isinstance(validator, MaxLen):
        _validate_maxlen(series, validator, col_name)
    elif isinstance(validator, Custom):
        _validate_custom(series, validator, col_name)
    else:
        raise ValidationError(f"Unknown validator type: {type(validator)}")


def _get_invalid_samples(
    series: pd.Series, invalid_mask: pd.Series
) -> tuple[list[tuple[Any, Any]], int]:
    """
    Extract invalid samples and total count from an invalid mask.

    Args:
        series: pandas Series being validated
        invalid_mask: Boolean mask indicating invalid values

    Returns:
        Tuple of (samples, total_invalid) where samples is a list of (index, value) tuples
    """
    invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE].tolist()  # type: ignore[index]
    samples: list[tuple[Any, Any]] = [(idx, series.loc[idx]) for idx in invalid_indices]
    total_invalid = invalid_mask.sum()
    return samples, total_invalid


def _require_orderable(series: pd.Series, validator: Any, col_name: str) -> None:
    """Reject columns that cannot be meaningfully ordered against a bound."""
    # bool counts as numeric in pandas but not in polars; exclude it so both backends agree
    if pd.api.types.is_bool_dtype(series):
        orderable = False
    elif (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    ):
        orderable = True
    elif series.dtype == object:
        # Any-typed or Decimal columns land in object dtype; polars accepts the equivalent dtypes
        orderable = pd.api.types.infer_dtype(series, skipna=True) in _ORDERABLE_OBJECT_KINDS
    else:
        orderable = False
    if not orderable:
        raise ValidationError(
            f"{type(validator).__name__} requires a numeric or temporal column, got {series.dtype}",
            column_name=col_name,
        )


def _compare(
    series: pd.Series, op: Callable[[Any, Any], Any], bound: Any, validator: Any, col_name: str
) -> pd.Series:
    """Compare series with a bound, reporting an incomparable bound as a ValidationError."""
    try:
        return op(series, bound)
    except TypeError as e:
        raise ValidationError(
            f"{type(validator).__name__} bound {bound!r} cannot be compared with "
            f"{series.dtype} values",
            column_name=col_name,
        ) from e


def _validate_range(series: pd.Series, validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    invalid_mask = _compare(series, operator.lt, validator.min, validator, col_name) | _compare(
        series, operator.gt, validator.max, validator, col_name
    )
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name,
        f"values must be in range [{validator.min}, {validator.max}]",
        samples,
        total_invalid,
        format_value=str,
    )


def _validate_comparison(series: pd.Series, validator: Ge | Gt | Le | Lt, col_name: str) -> None:
    """Validate that all values in series satisfy the comparison against the bound."""
    violates, symbol = next(
        entry for cls, entry in _COMPARISONS.items() if isinstance(validator, cls)
    )
    invalid_mask = _compare(series, violates, validator.value, validator, col_name)
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name,
        f"values must be {symbol} {validator.value}",
        samples,
        total_invalid,
        format_value=str,
    )


def _validate_unique(series: pd.Series, col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    duplicated_mask = series.duplicated(keep=False)
    if not duplicated_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, duplicated_mask)
    raise ValidationError.new_with_samples(
        col_name,
        "contains duplicate values",
        samples,
        total_invalid,
        format_value=repr,
    )


def _validate_in(series: pd.Series, validator: In, col_name: str) -> None:
    """Validate that all values in series are within the allowed set."""
    invalid_mask = ~series.isin(validator.allowed_values)
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name, "contains values not in allowed values", samples, total_invalid
    )


def _validate_regex(series: pd.Series, validator: Regex, col_name: str) -> None:
    """Validate that all values in series match the regex pattern."""
    # na=True keeps the mask boolean (nulls are checked by the optional check, not here);
    # with the default, object columns yield None and `~` raises TypeError
    invalid_mask = ~series.str.fullmatch(validator.pattern, na=True)
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name, "contains values that don't match the pattern", samples, total_invalid
    )


def _validate_minlen(series: pd.Series, validator: MinLen, col_name: str) -> None:
    """Validate that all string values have minimum length."""
    invalid_mask = series.str.len() < validator.min_length
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name,
        "contains strings shorter than minimum length",
        samples,
        total_invalid,
        format_value=lambda val: f"{repr(val)} (length: {len(val)})",
    )


def _validate_maxlen(series: pd.Series, validator: MaxLen, col_name: str) -> None:
    """Validate that all string values have maximum length."""
    invalid_mask = series.str.len() > validator.max_length
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name,
        "contains strings longer than maximum length",
        samples,
        total_invalid,
        format_value=lambda val: f"{repr(val)} (length: {len(val)})",
    )


def _validate_custom(series: pd.Series, validator: Custom, col_name: str) -> None:
    """Validate using a custom function."""
    invalid_mask: pd.Series = ~series.apply(validator.func)  # type: ignore[assignment]
    if not bool(invalid_mask.any()):
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(col_name, validator.message, samples, total_invalid)
