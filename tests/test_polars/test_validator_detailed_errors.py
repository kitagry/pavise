from typing import Annotated, Protocol

import polars as pl
import pytest

from patrol.exceptions import ValidationError
from patrol.polars import DataFrame
from patrol.validators import In, Range, Regex, Unique


class RangeSchema(Protocol):
    age: Annotated[int, Range(0, 150)]


def test_range_validator_error_shows_sample_invalid_values():
    """Range validator error should show sample values that are out of range"""
    df = pl.DataFrame({"age": [25, 200, 30, -5, 35, 300, 40]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[RangeSchema](df)

    error = exc_info.value

    # Check ValidationError attributes
    assert error.column_name == "age"
    assert error.invalid_samples == [(1, 200), (3, -5), (5, 300)]

    # Check error message - at least verify the first line exactly
    error_message = str(error)
    assert error_message.startswith("Column 'age': values must be in range [0, 150]")
    assert "Row 1: 200" in error_message
    assert "Row 3: -5" in error_message
    assert "Row 5: 300" in error_message


class UniqueSchema(Protocol):
    user_id: Annotated[int, Unique()]


def test_unique_validator_error_shows_duplicate_values():
    """Unique validator error should show which values are duplicated"""
    df = pl.DataFrame({"user_id": [1, 2, 3, 2, 4, 3, 5]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[UniqueSchema](df)

    error = exc_info.value

    # Check ValidationError attributes
    assert error.column_name == "user_id"
    # Unique validator doesn't use invalid_samples the same way
    # (it shows duplicate values, not row indices)

    # Check error message - at least verify the first line exactly
    error_message = str(error)
    assert error_message.startswith("Column 'user_id': contains duplicate values")

    # Should show the duplicate values (2 and 3)
    assert "2" in error_message
    assert "3" in error_message


class InSchema(Protocol):
    status: Annotated[str, In(["pending", "approved", "rejected"])]


def test_in_validator_error_shows_invalid_values():
    """In validator error should show values not in allowed set"""
    df = pl.DataFrame({"status": ["pending", "invalid", "approved", "bad", "rejected"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[InSchema](df)

    error = exc_info.value

    # Check ValidationError attributes
    assert error.column_name == "status"
    assert error.invalid_samples == [(1, "invalid"), (3, "bad")]

    # Check error message - at least verify the first line exactly
    error_message = str(error)
    assert error_message.startswith("Column 'status': contains values not in allowed values")
    assert "Row 1: 'invalid'" in error_message
    assert "Row 3: 'bad'" in error_message


class RegexSchema(Protocol):
    email: Annotated[str, Regex(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-z]+$")]


def test_regex_validator_error_shows_non_matching_values():
    """Regex validator error should show values that don't match the pattern"""
    df = pl.DataFrame({"email": ["valid@example.com", "invalid", "test@test.com", "bad@"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[RegexSchema](df)

    error = exc_info.value

    # Check ValidationError attributes
    assert error.column_name == "email"
    assert error.invalid_samples == [(1, "invalid"), (3, "bad@")]

    # Check error message - at least verify the first line exactly
    error_message = str(error)
    assert error_message.startswith("Column 'email': contains values that don't match the pattern")
    assert "Row 1: 'invalid'" in error_message
    assert "Row 3: 'bad@'" in error_message
