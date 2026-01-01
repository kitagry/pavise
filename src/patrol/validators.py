"""Validators for DataFrame column validation with Annotated types.

Validators are simple data classes that define validation rules.
The actual validation logic is implemented in backend-specific modules:
- _pandas.validator_impl for pandas
- _polars.validator_impl for polars
"""

from dataclasses import dataclass


@dataclass
class Range:
    """Validate that numeric values are within a specified range.

    Example:
        age: Annotated[int, Range(0, 150)]
    """

    min: float
    max: float


@dataclass
class Unique:
    """Validate that column values are unique (no duplicates).

    Example:
        user_id: Annotated[int, Unique()]
    """
