from typing import Protocol

import polars as pl
import pytest

from patrol.exceptions import ValidationError
from patrol.polars import DataFrame


class SimpleSchema(Protocol):
    age: int
    name: str


def test_type_error_shows_sample_invalid_values():
    """Type error should show sample invalid values with row numbers"""
    # Polars requires explicit dtype for mixed types
    df = pl.DataFrame(
        {
            "age": pl.Series(
                [25, "invalid", 30, None, 35, "bad", 40], dtype=pl.Object, strict=False
            ),
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[SimpleSchema](df)

    error_message = str(exc_info.value)

    # Error message should contain basic information
    assert "Column 'age'" in error_message
    assert "expected int" in error_message

    # Should show sample invalid values with row numbers
    assert "Row 1:" in error_message or "row 1" in error_message.lower()
    assert "invalid" in error_message or "str" in error_message


def test_type_error_limits_sample_size():
    """Type error should limit the number of sample values shown (e.g., first 5)"""
    # DataFrame with 20 invalid values
    invalid_values = ["bad"] * 20
    df = pl.DataFrame(
        {
            "age": pl.Series(invalid_values + [25, 30], dtype=pl.Object, strict=False),
            "name": ["Alice"] * 22,
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[SimpleSchema](df)

    error_message = str(exc_info.value)

    # Should indicate that samples are limited
    assert "showing" in error_message.lower() or "sample" in error_message.lower()
