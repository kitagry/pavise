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


@dataclass
class In:
    """Validate that column values are within a set of allowed values.

    Example:
        status: Annotated[str, In(["pending", "approved", "rejected"])]
    """

    allowed_values: list


@dataclass
class Regex:
    r"""Validate that string values match a regular expression pattern.

    Example:
        email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
    """

    pattern: str


@dataclass
class MinLen:
    """Validate that string values have a minimum length.

    Example:
        username: Annotated[str, MinLen(3)]
    """

    min_length: int


@dataclass
class MaxLen:
    """Validate that string values have a maximum length.

    Example:
        username: Annotated[str, MaxLen(20)]
    """

    max_length: int
