"""Polars-specific validator implementations."""

from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise.exceptions import ValidationError
from pavise.validators import Custom, In, MaxLen, MinLen, Range, Regex, Unique

# Maximum number of invalid sample values to show in error messages
MAX_SAMPLE_SIZE = 5


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


def _get_invalid_samples(series: "pl.Series", invalid_mask: "pl.Series") -> tuple[list[tuple], int]:
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


def _validate_range(series: "pl.Series", validator: Range, col_name: str) -> None:
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
    invalid_mask = ~series.str.contains(f"^{validator.pattern}$")
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
