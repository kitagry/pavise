"""Pandas-specific validator implementations."""

from typing import Any

import pandas as pd

from pavise.exceptions import ValidationError
from pavise.validators import Custom, In, MaxLen, MinLen, Range, Regex, Unique

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5


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
    if isinstance(validator, Range):
        _validate_range(series, validator, col_name)
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


def _get_invalid_samples(series: pd.Series, invalid_mask: pd.Series) -> tuple[list[tuple], int]:
    """
    Extract invalid samples and total count from an invalid mask.

    Args:
        series: pandas Series being validated
        invalid_mask: Boolean mask indicating invalid values

    Returns:
        Tuple of (samples, total_invalid) where samples is a list of (index, value) tuples
    """
    invalid_indices = series.index[invalid_mask][:MAX_SAMPLE_SIZE]
    samples = [(idx, series.loc[idx]) for idx in invalid_indices]
    total_invalid = invalid_mask.sum()
    return samples, total_invalid


def _validate_range(series: pd.Series, validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    invalid_mask = (series < validator.min) | (series > validator.max)
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
    invalid_mask = ~series.str.match(validator.pattern)
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
    invalid_mask = ~series.apply(validator.func)
    if not invalid_mask.any():
        return

    samples, total_invalid = _get_invalid_samples(series, invalid_mask)
    raise ValidationError.new_with_samples(col_name, validator.message, samples, total_invalid)
