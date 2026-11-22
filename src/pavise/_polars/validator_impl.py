"""Polars-specific validator implementations."""

import math
import operator
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, time
from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise._regex import fullmatch_pattern, invalid_pattern_error
from pavise._violation import Violation, with_length
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
        violation = _find_duplicates(series)
        if violation is None:
            return None
    elif isinstance(validator, In):
        violation = Violation(
            ~is_in_values(series, validator.allowed_values), "contains values not in allowed values"
        )
    elif isinstance(validator, Regex):
        violation = _find_regex(series, validator, col_name)
    elif isinstance(validator, MinLen):
        violation = Violation(
            series.str.len_chars() < validator.min_length,
            "contains strings shorter than minimum length",
            with_length,
        )
    elif isinstance(validator, MaxLen):
        violation = Violation(
            series.str.len_chars() > validator.max_length,
            "contains strings longer than maximum length",
            with_length,
        )
    elif isinstance(validator, Custom):
        violation = Violation(~series.map_elements(validator.func), validator.message)
    else:
        raise ValidationError(f"Unknown validator type: {type(validator)}")

    # Comparisons leave the mask null where they cannot decide, which is no violation
    mask = violation.mask.fill_null(False)
    if not mask.any():
        return None
    return Violation(mask, violation.message, violation.format_value)


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


_NAN = object()


def _find_duplicates(series: "pl.Series") -> Violation | None:
    message = "contains duplicate values"
    # polars cannot hash Object values, so union columns compare their Python values instead
    if series.dtype == pl.Object:
        mask = _object_duplicate_mask(series.to_list())
        if mask is None:
            return None
        return Violation(pl.Series(series.name, mask, dtype=pl.Boolean), message)
    # n_unique is far cheaper than building the per-row mask, which only the error needs.
    # For structs is_duplicated equates a null row with an all-null struct but n_unique
    # does not, so they skip the shortcut to keep is_duplicated the only judge.
    if series.dtype != pl.Struct and series.n_unique() == series.len():
        return None
    return Violation(series.is_duplicated(), message)


def _object_duplicate_mask(values: list[Any]) -> list[bool] | None:
    """Return which values occur more than once, or None if all are distinct."""
    # NaN is never equal to itself in Python; key it as one value, as polars and pandas do
    keys = [_NAN if isinstance(v, float) and math.isnan(v) else v for v in values]
    try:
        counts = Counter(keys)
    except TypeError:
        # Unhashable values such as lists and dicts can only be grouped by equality
        groups: list[Any] = []
        group_ids = []
        for key in keys:
            group_id = next((i for i, g in enumerate(groups) if g == key), len(groups))
            if group_id == len(groups):
                groups.append(key)
            group_ids.append(group_id)
        if len(groups) == len(keys):
            return None
        sizes = Counter(group_ids)
        return [sizes[i] > 1 for i in group_ids]
    if len(counts) == len(keys):
        return None
    return [counts[k] > 1 for k in keys]


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


def is_in_values(series: "pl.Series", values: Any) -> "pl.Series":
    """Return which values of ``series`` are in ``values``; nulls stay null, as with is_in."""
    if series.dtype != pl.Object:
        try:
            return series.is_in(values)
        except (TypeError, pl.exceptions.InvalidOperationError):
            # TypeError: values of several types cannot form the one polars Series is_in needs.
            # InvalidOperationError: is_in refuses values of another dtype than the column.
            pass
    # polars' is_in cannot check Object columns, so compare the Python values instead
    allowed = list(values)
    try:
        allowed_set: set | None = set(allowed)
    except TypeError:
        allowed_set = None

    def contains(value: Any) -> bool:
        # A set lookup keeps large allowed lists cheap; unhashable values need the list scan
        if allowed_set is not None:
            try:
                return value in allowed_set
            except TypeError:
                pass
        return value in allowed

    return pl.Series(
        series.name,
        [None if value is None else contains(value) for value in series.to_list()],
        dtype=pl.Boolean,
    )
