"""Polars-specific validator implementations."""

from typing import Any

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install patrol[polars]")

from patrol.validators import Range, Unique


def apply_validator(series: "pl.Series", validator: Any, col_name: str) -> None:
    """
    Apply a validator to a polars Series.

    Args:
        series: polars Series to validate
        validator: Validator instance (e.g., Range, Unique)
        col_name: column name for error messages

    Raises:
        ValueError: if validation fails
    """
    if isinstance(validator, Range):
        _validate_range(series, validator, col_name)
    elif isinstance(validator, Unique):
        _validate_unique(series, col_name)
    else:
        raise ValueError(f"Unknown validator type: {type(validator)}")


def _validate_range(series: "pl.Series", validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    if (series < validator.min).any() or (series > validator.max).any():
        raise ValueError(
            f"Column '{col_name}': values must be in range [{validator.min}, {validator.max}]"
        )


def _validate_unique(series: "pl.Series", col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    if series.is_duplicated().any():
        raise ValueError(f"Column '{col_name}': contains duplicate values")
