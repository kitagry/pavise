"""Pandas-specific validator implementations."""

from typing import Any

import pandas as pd

from patrol.validators import Range, Unique


def apply_validator(series: pd.Series, validator: Any, col_name: str) -> None:
    """
    Apply a validator to a pandas Series.

    Args:
        series: pandas Series to validate
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


def _validate_range(series: pd.Series, validator: Range, col_name: str) -> None:
    """Validate that all values in series are within the specified range."""
    if (series < validator.min).any() or (series > validator.max).any():
        raise ValueError(
            f"Column '{col_name}': values must be in range [{validator.min}, {validator.max}]"
        )


def _validate_unique(series: pd.Series, col_name: str) -> None:
    """Validate that all values in series are unique (no duplicates)."""
    if series.duplicated().any():
        raise ValueError(f"Column '{col_name}': contains duplicate values")
