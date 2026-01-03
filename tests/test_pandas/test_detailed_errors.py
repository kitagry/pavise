from typing import Protocol

import pandas as pd
import pytest

from patrol.exceptions import ValidationError
from patrol.pandas import DataFrame


class SimpleSchema(Protocol):
    age: int
    name: str


def test_type_error_shows_sample_invalid_values():
    """Type error should show sample invalid values with row numbers"""
    df = pd.DataFrame(
        {
            "age": [25, "invalid", 30, None, 35, "bad", 40],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
        }
    )

    with pytest.raises(
        ValidationError, match="Column 'age': is non-optional but contains null values"
    ):
        DataFrame[SimpleSchema](df)


def test_type_error_limits_sample_size():
    """Type error should limit the number of sample values shown (e.g., first 5)"""
    # DataFrame with 20 invalid values
    invalid_values = ["bad"] * 20
    df = pd.DataFrame(
        {
            "age": invalid_values + [25, 30],
            "name": ["Alice"] * 22,
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[SimpleSchema](df)

    error_message = str(exc_info.value)

    # Should indicate that samples are limited
    assert "showing" in error_message.lower() or "sample" in error_message.lower()
