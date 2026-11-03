"""Pin exact error messages and invalid samples of type checks.

These guard the dtype-based fast paths: they must report the same rows, values
and counts as a per-element check would.
"""

from datetime import date, datetime
from typing import Annotated, Any, Protocol

import polars as pl
import pytest

from pavise.exceptions import ValidationError
from pavise.polars import DataFrame
from pavise.validators import Unique


class IntSchema(Protocol):
    v: int


class FloatSchema(Protocol):
    v: float


class OptionalIntSchema(Protocol):
    v: int | None


class UnionSchema(Protocol):
    v: int | str


class OptionalUnionSchema(Protocol):
    v: int | str | None


class FloatOrStrSchema(Protocol):
    v: float | str


class DateOrStrSchema(Protocol):
    v: date | str


@pytest.mark.parametrize(
    ("schema", "df", "message", "samples"),
    [
        pytest.param(
            UnionSchema,
            pl.DataFrame({"v": [1.5, None, 2.5]}),
            "Column 'v': expected int | str, got Float64\n\n"
            "Sample invalid values (showing first 3 of 3):\n"
            "  Row 0: 1.5\n  Row 1: None\n  Row 2: 2.5",
            [(0, 1.5), (1, None), (2, 2.5)],
            id="union-got-float64-null-is-invalid",
        ),
        pytest.param(
            OptionalUnionSchema,
            pl.DataFrame({"v": [1.5, None, 2.5]}),
            "Column 'v': expected int | str, got Float64\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 0: 1.5\n  Row 2: 2.5",
            [(0, 1.5), (2, 2.5)],
            id="optional-union-got-float64",
        ),
        pytest.param(
            UnionSchema,
            pl.DataFrame({"v": [True, False]}),
            "Column 'v': expected int | str, got Boolean\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 0: True\n  Row 1: False",
            [(0, True), (1, False)],
            id="union-got-boolean",
        ),
        pytest.param(
            UnionSchema,
            pl.DataFrame({"v": [1, None, 3]}),
            "Column 'v': expected int | str, got Int64\n\n"
            "Sample invalid values (showing first 1 of 1):\n  Row 1: None",
            [(1, None)],
            id="union-int64-with-null",
        ),
        pytest.param(
            UnionSchema,
            pl.DataFrame({"v": [1.5] * 7}),
            "Column 'v': expected int | str, got Float64\n\n"
            "Sample invalid values (showing first 5 of 7):\n"
            "  Row 0: 1.5\n  Row 1: 1.5\n  Row 2: 1.5\n  Row 3: 1.5\n  Row 4: 1.5",
            [(i, 1.5) for i in range(5)],
            id="union-limits-samples",
        ),
        pytest.param(
            UnionSchema,
            pl.DataFrame({"v": pl.Series([1, "a", 2.5, None], dtype=pl.Object)}),
            "Column 'v': expected int | str, got Object\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 2: 2.5\n  Row 3: None",
            [(2, 2.5), (3, None)],
            id="union-object-mixed",
        ),
        pytest.param(
            IntSchema,
            pl.DataFrame({"v": ["a", None, "b", "c", "d", "e", "f"]}),
            "Column 'v': expected int, got String\n\n"
            "Sample invalid values (showing first 5 of 6):\n"
            "  Row 0: 'a'\n  Row 2: 'b'\n  Row 3: 'c'\n  Row 4: 'd'\n  Row 5: 'e'",
            [(0, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")],
            id="int-got-string-skips-nulls",
        ),
        pytest.param(
            FloatSchema,
            pl.DataFrame({"v": [1, 2]}),
            "Column 'v': expected float, got Int64\n\n"
            "Sample invalid values (showing first 0 of 0):",
            [],
            id="float-got-int64-values-pass",
        ),
        pytest.param(
            OptionalIntSchema,
            pl.DataFrame({"v": [1.5, None]}),
            "Column 'v': expected int, got Float64\n\n"
            "Sample invalid values (showing first 1 of 1):\n  Row 0: 1.5",
            [(0, 1.5)],
            id="int-got-float64",
        ),
        pytest.param(
            IntSchema,
            pl.DataFrame({"v": pl.Series([1, "a", None, 2.5], dtype=pl.Object)}),
            "Column 'v': expected int, got Object\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 1: 'a'\n  Row 3: 2.5",
            [(1, "a"), (3, 2.5)],
            id="int-got-object",
        ),
    ],
)
def test_error_reports_exact_samples(
    schema: type, df: pl.DataFrame, message: str, samples: list[tuple[Any, Any]]
):
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    assert str(exc_info.value) == message
    assert exc_info.value.invalid_samples == samples


@pytest.mark.parametrize(
    ("schema", "df"),
    [
        pytest.param(FloatOrStrSchema, pl.DataFrame({"v": [1, 2]}), id="float-union-int64"),
        pytest.param(
            DateOrStrSchema, pl.DataFrame({"v": [datetime(2024, 1, 1)]}), id="date-union-datetime"
        ),
        pytest.param(OptionalUnionSchema, pl.DataFrame({"v": [1, None, 3]}), id="optional-union"),
        pytest.param(UnionSchema, pl.DataFrame({"v": pl.Series([], dtype=pl.Float64)}), id="empty"),
        pytest.param(
            OptionalUnionSchema,
            pl.DataFrame({"v": [None, None]}, schema={"v": pl.Float64}),
            id="all-null",
        ),
    ],
)
def test_valid_dataframe_passes(schema: type, df: pl.DataFrame):
    DataFrame[schema](df)


class UniqueSchema(Protocol):
    v: Annotated[int, Unique()]


@pytest.mark.parametrize(
    ("schema", "df", "samples"),
    [
        pytest.param(
            UnionSchema,
            pl.DataFrame({"v": [1.5, 2.0, 1.5]}),
            [(0, 1.5), (1, 2.0), (2, 1.5)],
            id="type-check",
        ),
        pytest.param(
            UniqueSchema,
            pl.DataFrame({"v": [1, 2, 1, 3]}),
            [(0, 1), (2, 1)],
            id="validator",
        ),
    ],
)
def test_samples_are_positional_with_repeated_values(
    schema: type, df: pl.DataFrame, samples: list[tuple[Any, Any]]
):
    """Polars has no index labels; samples carry row positions and scalar values"""
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    assert exc_info.value.invalid_samples == samples
