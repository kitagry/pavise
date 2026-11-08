"""Validators for DataFrame column validation with Annotated types.

Validators are simple data classes that define validation rules.
The actual validation logic is implemented in backend-specific modules:
- _pandas.validator_impl for pandas
- _polars.validator_impl for polars
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, TypeAlias

# Validators are hashable because Python 3.10 hashes Annotated metadata when deduplicating
# the members of a union such as Annotated[int, Range(0, 10)] | None.

# Every bound type the backends can order a numeric or temporal column against
Bound: TypeAlias = float | Decimal | date | datetime | time | timedelta


@dataclass(unsafe_hash=True)
class Range:
    """Validate that numeric or temporal values are within a specified range.

    Example:
        age: Annotated[int, Range(0, 150)]
        joined: Annotated[date, Range(date(2020, 1, 1), date(2020, 12, 31))]
    """

    min: Bound
    max: Bound


@dataclass(unsafe_hash=True)
class Ge:
    """Validate that numeric or temporal values are greater than or equal to a bound.

    Example:
        price: Annotated[float, Ge(0)]
        created_at: Annotated[datetime, Ge(datetime(2020, 1, 1))]
    """

    value: Bound


@dataclass(unsafe_hash=True)
class Gt:
    """Validate that numeric or temporal values are strictly greater than a bound.

    Example:
        quantity: Annotated[int, Gt(0)]
    """

    value: Bound


@dataclass(unsafe_hash=True)
class Le:
    """Validate that numeric or temporal values are less than or equal to a bound.

    Example:
        percentage: Annotated[float, Le(100)]
    """

    value: Bound


@dataclass(unsafe_hash=True)
class Lt:
    """Validate that numeric or temporal values are strictly less than a bound.

    Example:
        temperature: Annotated[float, Lt(1000)]
    """

    value: Bound


@dataclass(unsafe_hash=True)
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

    def __hash__(self) -> int:
        return hash(tuple(self.allowed_values))


@dataclass(unsafe_hash=True)
class Regex:
    r"""Validate that string values match a regular expression pattern.

    The whole string must match (``re.fullmatch`` semantics), so ``^`` and ``$``
    anchors are optional.

    Example:
        email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
    """

    pattern: str


@dataclass(unsafe_hash=True)
class MinLen:
    """Validate that string values have a minimum length.

    Example:
        username: Annotated[str, MinLen(3)]
    """

    min_length: int


@dataclass(unsafe_hash=True)
class MaxLen:
    """Validate that string values have a maximum length.

    Example:
        username: Annotated[str, MaxLen(20)]
    """

    max_length: int


@dataclass(unsafe_hash=True)
class Custom:
    """Validate using a custom validation function.

    The function should accept a single value and return True if valid, False otherwise.

    Example:
        def is_positive(value) -> bool:
            return value > 0

        age: Annotated[int, Custom(is_positive, "must be positive")]
    """

    func: Callable[[Any], bool]
    message: str
