"""Custom exceptions for patrol."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import Self


class PatrolError(Exception):
    """Base exception for all patrol errors."""

    pass


class ValidationError(PatrolError):
    """
    Raised when DataFrame validation fails.

    Attributes:
        message: Human-readable error message
        column_name: Name of the column that failed validation (if applicable)
        invalid_samples: List of (row_index, value) tuples showing sample invalid values
    """

    def __init__(
        self,
        base_message: str,
        column_name: str | None = None,
        invalid_samples: list[tuple] | None = None,
    ):
        msg = base_message
        if column_name:
            msg = f"Column '{column_name}': {msg}"
        super().__init__(msg)
        self.column_name = column_name
        self.invalid_samples = invalid_samples or []

    @classmethod
    def new_with_samples(
        cls,
        col_name: str,
        base_message: str,
        samples: list[tuple],
        total_invalid: int,
        format_value: Callable[[Any], str] = repr,
    ) -> Self:
        """
        Create a ValidationError with a formatted message including sample invalid values.
        Args:
            col_name: Column name
            base_message: Base error message (e.g., "values must be in range [0, 150]")
            samples: List of (index, value) tuples
            total_invalid: Total number of invalid values
            format_value: Optional function to format values (default: repr)

        Returns:
        Formatted error message
        """
        msg = base_message
        msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
        msg += "".join([f"\n  Row {idx}: {format_value(val)}" for idx, val in samples])
        return cls(msg, column_name=col_name, invalid_samples=samples)
