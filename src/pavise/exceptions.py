"""Custom exceptions for pavise."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import Self


class PaviseError(Exception):
    """Base exception for all pavise errors."""


# Kept for backward compatibility with code written against the old project name.
PatrolError = PaviseError


class ValidationError(PaviseError):
    """
    Raised when DataFrame validation fails.

    Attributes:
        message: Human-readable error message
        column_name: Name of the column that failed validation (if applicable)
        path: Where inside a List or Struct column the failure is, e.g. ``history[].zip``
            (None for a failure of the column itself)
        invalid_samples: List of (row_index, value) tuples showing sample invalid values
    """

    def __init__(
        self,
        base_message: str,
        column_name: str | None = None,
        invalid_samples: list[tuple[Any, ...]] | None = None,
        path: str | None = None,
    ):
        msg = base_message
        if path or column_name:
            msg = f"Column '{path or column_name}': {msg}"
        super().__init__(msg)
        self.column_name = column_name
        self.path = path
        self.invalid_samples = invalid_samples or []

    @classmethod
    def new_with_samples(
        cls,
        col_name: str | None,
        base_message: str,
        samples: list[tuple[Any, ...]],
        total_invalid: int,
        format_value: Callable[[Any], str] = repr,
        path: str | None = None,
    ) -> Self:
        """
        Create a ValidationError with a formatted message including sample invalid values.
        Args:
            col_name: Column name, or None for DataFrame-level checks
            base_message: Base error message (e.g., "values must be in range [0, 150]")
            samples: List of (index, value) tuples
            total_invalid: Total number of invalid values
            format_value: Optional function to format values (default: repr)
            path: Where inside a List or Struct column the invalid values are

        Returns:
        Formatted error message
        """
        msg = base_message
        # A dtype mismatch can reject a column whose values all pass the per-value check
        if total_invalid > 0:
            msg += f"\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
            msg += "".join([f"\n  Row {idx}: {format_value(val)}" for idx, val in samples])
        return cls(msg, column_name=col_name, invalid_samples=samples, path=path)

    @classmethod
    def new_with_duplicate_keys(
        cls,
        columns: tuple[str, ...],
        duplicates: list[tuple[dict[str, Any], list[Any], int]],
        total_keys: int,
    ) -> Self:
        """
        Create a ValidationError for a failed ``@unique`` check.

        Args:
            columns: Key columns of the check
            duplicates: Sample duplicate keys as (key, sample row labels, number of rows)
            total_keys: Total number of duplicate keys

        ``invalid_samples`` holds a (row label, key) tuple for each sample row.
        """
        noun = "key" if total_keys == 1 else "keys"
        msg = f"Columns {columns!r} must be unique, found {total_keys} duplicate {noun}"
        msg += f"\n\nSample duplicate keys (showing first {len(duplicates)} of {total_keys}):"
        samples: list[tuple[Any, ...]] = []
        for key, rows, row_count in duplicates:
            shown = ", ".join(repr(row) for row in rows)
            if row_count > len(rows):
                msg += f"\n  {key!r}: rows [{shown}, ...] ({row_count} rows)"
            else:
                msg += f"\n  {key!r}: rows [{shown}]"
            samples.extend((row, key) for row in rows)
        return cls(msg, invalid_samples=samples)
