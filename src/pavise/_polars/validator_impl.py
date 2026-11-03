"""Polars-specific validator implementations."""

import operator
from collections.abc import Callable
from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

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

BOUND_VALIDATORS = (Range, *_COMPARISONS)


def apply_validator(series: "pl.Series", validator: Any, col_name: str) -> None:
    """
    Apply a validator to a polars Series.

    Args:
        series: polars Series to validate
        validator: Validator instance (e.g., Range, Unique, In, Regex, MinLen, MaxLen)
        col_name: column name for error messages

    Raises:
        ValueError: if validation fails
    """
    if isinstance(validator, BOUND_VALIDATORS):
        require_orderable_dtype(series.dtype, validator, col_name)

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
    series: "pl.Series", invalid_mask: "pl.Series"
) -> tuple[list[tuple[Any, Any]], int]:
    """
    Extract invalid samples and total count from an invalid mask.

    Args:
        series: polars Series being validated
        invalid_mask: Boolean mask indicating invalid values

    Returns:
        Tuple of (samples, total_invalid) where samples is a list of (index, value) tuples
    """
    invalid_df = series.to_frame().with_row_index("__row__").filter(invalid_mask)
    samples = [
        (row["__row__"], row[series.name])
        for row in invalid_df.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
    ]
    total_invalid = int(invalid_mask.sum())
    return samples, total_invalid


def require_orderable_dtype(dtype: "pl.DataType", validator: Any, col_name: str) -> None:
    """Reject dtypes that cannot be meaningfully ordered against a bound."""
    # Utf8 would otherwise be compared with the bound as strings and silently pass
    if dtype.is_numeric() or dtype.is_temporal():
        return
    raise ValidationError(
        f"{type(validator).__name__} requires a numeric or temporal column, got {dtype}",
        column_name=col_name,
    )


def _compare(
    series: "pl.Series",
    op: Callable[[Any, Any], Any],
    bound: Any,
    validator: Any,
    col_name: str,
) -> "pl.Series":
    """Compare series with a bound, reporting an incomparable bound as a ValidationError."""
    try:
        return op(series, bound)
    except (TypeError, NotImplementedError, pl.exceptions.PolarsError) as e:
        raise ValidationError(
            f"{type(validator).__name__} bound {bound!r} cannot be compared with "
            f"{series.dtype} values",
            column_name=col_name,
        ) from e


def _validate_range(series: "pl.Series", validator: Range, col_name: str) -> None:
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


def _validate_comparison(series: "pl.Series", validator: Ge | Gt | Le | Lt, col_name: str) -> None:
    """Validate that all values in series satisfy the comparison against the bound."""
    violates, symbol = next(
        entry for cls, entry in _COMPARISONS.items() if isinstance(validator, cls)
    )
    invalid_mask = _compare(series, violates, validator.value, validator, col_name)
    if series.dtype.is_float():
        # Polars orders NaN above every number, so only upper bounds would catch it
        invalid_mask = invalid_mask | series.is_nan()
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


def _validate_unique(series: "pl.Series", col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    duplicated_mask = series.is_duplicated()
    if not duplicated_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, duplicated_mask)
    raise ValidationError.new_with_samples(
        col_name,
        "contains duplicate values",
        samples,
        total_invalid=total_invalid,
        format_value=repr,
    )


def _validate_in(series: "pl.Series", validator: In, col_name: str) -> None:
    """Validate that all values in series are within the allowed set."""
    invalid_mask = ~series.is_in(validator.allowed_values)
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name, "contains values not in allowed values", samples, total_invalid
    )


def _validate_regex(series: "pl.Series", validator: Regex, col_name: str) -> None:
    """Validate that all values in series match the regex pattern."""
    # Group the pattern so top-level alternation stays inside the anchors
    invalid_mask = ~series.str.contains(f"^(?:{validator.pattern})$")
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(
        col_name, "contains values that don't match the pattern", samples, total_invalid
    )


def _validate_minlen(series: "pl.Series", validator: MinLen, col_name: str) -> None:
    """Validate that all string values have minimum length."""
    invalid_mask = series.str.len_chars() < validator.min_length
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


def _validate_maxlen(series: "pl.Series", validator: MaxLen, col_name: str) -> None:
    """Validate that all string values have maximum length."""
    invalid_mask = series.str.len_chars() > validator.max_length
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


def _validate_custom(series: "pl.Series", validator: Custom, col_name: str) -> None:
    """Validate using a custom function."""
    invalid_mask = ~series.map_elements(validator.func)
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(col_name, validator.message, samples, total_invalid)
