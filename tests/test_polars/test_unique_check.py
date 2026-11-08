"""Tests for multi-column uniqueness checks (@unique) on the polars backend."""

from typing import Protocol

import pytest

pytest.importorskip("polars")

import polars as pl

from pavise.checks import dataframe_check, unique
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.types import NotRequiredColumn


@unique("user_id", "date")
class ActivitySchema(Protocol):
    user_id: int
    date: str
    steps: int


def test_unique_passes_when_key_combinations_are_distinct():
    df = pl.DataFrame({"user_id": [1, 1, 2], "date": ["d1", "d2", "d1"], "steps": [1, 2, 3]})

    result = DataFrame[ActivitySchema](df)

    assert result.height == 3


def test_unique_error_shows_duplicate_keys_and_their_rows():
    df = pl.DataFrame(
        {
            "user_id": [1, 2, 1, 2, 2, 3],
            "date": ["d1", "d2", "d1", "d2", "d2", "d1"],
            "steps": [1, 2, 3, 4, 5, 6],
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ActivitySchema](df)

    error = exc_info.value
    assert error.column_name is None
    assert error.invalid_samples == [
        (0, {"user_id": 1, "date": "d1"}),
        (2, {"user_id": 1, "date": "d1"}),
        (1, {"user_id": 2, "date": "d2"}),
        (3, {"user_id": 2, "date": "d2"}),
        (4, {"user_id": 2, "date": "d2"}),
    ]
    assert str(error) == (
        "Columns ('user_id', 'date') must be unique, found 2 duplicate keys"
        "\n\nSample duplicate keys (showing first 2 of 2):"
        "\n  {'user_id': 1, 'date': 'd1'}: rows [0, 2]"
        "\n  {'user_id': 2, 'date': 'd2'}: rows [1, 3, 4]"
    )


def test_unique_error_limits_keys_and_rows_per_key():
    df = pl.DataFrame(
        {
            "user_id": [0] * 7 + [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "date": ["d"] * 17,
            "steps": range(17),
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ActivitySchema](df)

    assert str(exc_info.value) == (
        "Columns ('user_id', 'date') must be unique, found 6 duplicate keys"
        "\n\nSample duplicate keys (showing first 5 of 6):"
        "\n  {'user_id': 0, 'date': 'd'}: rows [0, 1, 2, 3, 4, ...] (7 rows)"
        "\n  {'user_id': 1, 'date': 'd'}: rows [7, 8]"
        "\n  {'user_id': 2, 'date': 'd'}: rows [9, 10]"
        "\n  {'user_id': 3, 'date': 'd'}: rows [11, 12]"
        "\n  {'user_id': 4, 'date': 'd'}: rows [13, 14]"
    )


@unique("user_id", "date")
class OptionalDateSchema(Protocol):
    user_id: int
    date: str | None


def test_unique_treats_nulls_as_equal():
    df = pl.DataFrame(
        {"user_id": [1, 1, 2], "date": [None, None, None]}, schema_overrides={"date": pl.Utf8}
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[OptionalDateSchema](df)

    assert str(exc_info.value) == (
        "Columns ('user_id', 'date') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1, 'date': None}: rows [0, 1]"
    )


def test_unique_checks_single_column():
    @unique("user_id")
    class Schema(Protocol):
        user_id: int

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pl.DataFrame({"user_id": [1, 2, 1]}))

    assert str(exc_info.value) == (
        "Columns ('user_id',) must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1}: rows [0, 2]"
    )


def test_unique_and_dataframe_check_run_in_written_order():
    @unique("user_id", "date")
    @dataframe_check(lambda df: pl.col("steps") >= 0, name="steps_non_negative")
    class Schema(Protocol):
        user_id: int
        date: str
        steps: int

    df = pl.DataFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [-1, 1]})

    with pytest.raises(ValidationError, match="must be unique"):
        DataFrame[Schema](df)


def test_unique_is_skipped_when_not_required_key_column_is_absent():
    @unique("user_id", "date")
    class Schema(Protocol):
        user_id: int
        date: NotRequiredColumn[str]

    result = DataFrame[Schema](pl.DataFrame({"user_id": [1, 1]}))

    assert result.height == 2


def test_make_empty_passes_unique():
    result = DataFrame[ActivitySchema].make_empty()

    assert result.height == 0


def test_lazyframe_does_not_check_unique_at_construction():
    lf = pl.LazyFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [1, 2]})

    LazyFrame[ActivitySchema](lf)


def test_lazyframe_checks_unique_on_collect():
    lf = LazyFrame[ActivitySchema](
        pl.LazyFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [1, 2]})
    )

    with pytest.raises(ValidationError) as exc_info:
        lf.collect()

    assert str(exc_info.value) == (
        "Columns ('user_id', 'date') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1, 'date': 'd'}: rows [0, 1]"
    )


def test_unique_accepts_key_columns_with_any_name():
    @unique("row", "count")
    class Schema(Protocol):
        row: int
        count: int

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pl.DataFrame({"row": [1, 1], "count": [2, 2]}))

    assert str(exc_info.value) == (
        "Columns ('row', 'count') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'row': 1, 'count': 2}: rows [0, 1]"
    )
