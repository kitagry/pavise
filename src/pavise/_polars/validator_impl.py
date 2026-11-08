"""Polars-specific validator implementations."""

import operator
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise._regex import fullmatch_pattern, invalid_pattern_error
from pavise._violation import Violation
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
        ValidationError: if validation fails
    """
    violation = find_violation(series, validator, col_name)
    if violation is None:
        return
    samples, total_invalid = get_invalid_samples(series, violation.mask)
    raise ValidationError.new_with_samples(
        col_name, violation.message, samples, total_invalid, violation.format_value
    )


def find_violation(series: "pl.Series", validator: Any, col_name: str) -> Violation | None:
    """
    Find the values a validator rejects, or None if it accepts them all.

    Raises:
        ValidationError: if the validator cannot be applied to the series at all
    """
    if isinstance(validator, BOUND_VALIDATORS):
        require_orderable_dtype(series.dtype, validator, col_name)

    if isinstance(validator, Range):
        violation = _find_range(series, validator, col_name)
    elif isinstance(validator, (Ge, Gt, Le, Lt)):
        violation = _find_comparison(series, validator, col_name)
    elif isinstance(validator, Unique):
        violation = Violation(series.is_duplicated(), "contains duplicate values")
    elif isinstance(validator, In):
        violation = Violation(
            ~series.is_in(validator.allowed_values), "contains values not in allowed values"
        )
    elif isinstance(validator, Regex):
        violation = _find_regex(series, validator, col_name)
    elif isinstance(validator, MinLen):
        violation = Violation(
            series.str.len_chars() < validator.min_length,
            "contains strings shorter than minimum length",
            _with_length,
        )
    elif isinstance(validator, MaxLen):
        violation = Violation(
            series.str.len_chars() > validator.max_length,
            "contains strings longer than maximum length",
            _with_length,
        )
    elif isinstance(validator, Custom):
        violation = Violation(~series.map_elements(validator.func), validator.message)
    else:
        raise ValidationError(f"Unknown validator type: {type(validator)}")

    if not violation.mask.any():
        return None
    return violation


def _with_length(value: Any) -> str:
    return f"{value!r} (length: {len(value)})"


def get_invalid_samples(
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
        if (
            isinstance(bound, date)
            and not isinstance(bound, datetime)
            and series.dtype == pl.Datetime
        ):
            # pandas stores date and datetime columns alike as datetime64 and so accepts a date
            # bound on a datetime column; read it as midnight here so both backends agree
            return op(series, datetime.combine(bound, time()))
        return op(series, bound)
    except (TypeError, NotImplementedError, pl.exceptions.PolarsError) as e:
        raise ValidationError(
            f"{type(validator).__name__} bound {bound!r} cannot be compared with "
            f"{series.dtype} values",
            column_name=col_name,
        ) from e


def _find_range(series: "pl.Series", validator: Range, col_name: str) -> Violation:
    """Find values outside the specified range."""
    invalid_mask = _compare(series, operator.lt, validator.min, validator, col_name) | _compare(
        series, operator.gt, validator.max, validator, col_name
    )
    return Violation(
        invalid_mask, f"values must be in range [{validator.min}, {validator.max}]", str
    )


def _find_comparison(series: "pl.Series", validator: Ge | Gt | Le | Lt, col_name: str) -> Violation:
    """Find values that do not satisfy the comparison against the bound."""
    violates, symbol = next(
        entry for cls, entry in _COMPARISONS.items() if isinstance(validator, cls)
    )
    invalid_mask = _compare(series, violates, validator.value, validator, col_name)
    if series.dtype.is_float():
        # Polars orders NaN above every number, so only upper bounds would catch it
        invalid_mask = invalid_mask | series.is_nan()
    return Violation(invalid_mask, f"values must be {symbol} {validator.value}", str)


def _find_regex(series: "pl.Series", validator: Regex, col_name: str) -> Violation:
    """Find values that do not match the regex pattern."""
    try:
        # Compile the pattern as written first: wrapping can make an invalid pattern valid
        # (`a)|(b` becomes `^(?:a)|(b)$`), and the engine's error then quotes the user's text
        pl.Series([""], dtype=pl.Utf8).str.contains(validator.pattern)
        invalid_mask = ~series.str.contains(fullmatch_pattern(validator.pattern))
    except pl.exceptions.ComputeError as e:
        raise invalid_pattern_error(validator.pattern, col_name, e) from e
    return Violation(invalid_mask, "contains values that don't match the pattern")
