"""Tests for multi-column uniqueness checks (@unique) on the pandas backend."""

from typing import Any, Protocol

import pytest

pytest.importorskip("pandas")

import pandas as pd

from pavise.checks import dataframe_check, unique
from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame
from pavise.types import NotRequiredColumn


@unique("user_id", "date")
class ActivitySchema(Protocol):
    user_id: int
    date: str
    steps: int


def test_unique_passes_when_key_combinations_are_distinct():
    df = pd.DataFrame({"user_id": [1, 1, 2], "date": ["d1", "d2", "d1"], "steps": [1, 2, 3]})

    result = DataFrame[ActivitySchema](df)

    assert len(result) == 3


def test_unique_error_shows_duplicate_keys_and_their_rows():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 2, 2, 3],
            "date": ["d1", "d2", "d1", "d2", "d2", "d1"],
            "steps": [1, 2, 3, 4, 5, 6],
        },
        index=pd.Index([10, 11, 12, 13, 14, 15]),
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ActivitySchema](df)

    error = exc_info.value
    assert error.column_name is None
    assert error.invalid_samples == [
        (10, {"user_id": 1, "date": "d1"}),
        (12, {"user_id": 1, "date": "d1"}),
        (11, {"user_id": 2, "date": "d2"}),
        (13, {"user_id": 2, "date": "d2"}),
        (14, {"user_id": 2, "date": "d2"}),
    ]
    assert str(error) == (
        "Columns ('user_id', 'date') must be unique, found 2 duplicate keys"
        "\n\nSample duplicate keys (showing first 2 of 2):"
        "\n  {'user_id': 1, 'date': 'd1'}: rows [10, 12]"
        "\n  {'user_id': 2, 'date': 'd2'}: rows [11, 13, 14]"
    )


def test_unique_error_limits_keys_and_rows_per_key():
    df = pd.DataFrame(
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


def test_unique_error_says_key_for_single_duplicate():
    df = pd.DataFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [1, 2]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ActivitySchema](df)

    assert str(exc_info.value).startswith(
        "Columns ('user_id', 'date') must be unique, found 1 duplicate key\n"
    )


@unique("user_id", "date")
class OptionalDateSchema(Protocol):
    user_id: int
    date: str | None


def test_unique_treats_nulls_as_equal():
    df = pd.DataFrame({"user_id": [1, 1, 2], "date": [None, None, None]})

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
        DataFrame[Schema](pd.DataFrame({"user_id": [1, 2, 1]}))

    assert str(exc_info.value) == (
        "Columns ('user_id',) must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1}: rows [0, 2]"
    )


def test_stacked_unique_checks_run_in_written_order():
    @unique("user_id")
    @unique("email")
    class Schema(Protocol):
        user_id: int
        email: str

    df = pd.DataFrame({"user_id": [1, 1], "email": ["a", "a"]})

    with pytest.raises(ValidationError, match=r"^Columns \('user_id',\)"):
        DataFrame[Schema](df)


def test_unique_and_dataframe_check_run_in_written_order():
    @dataframe_check(lambda df: df["steps"] >= 0, name="steps_non_negative")
    @unique("user_id", "date")
    class Schema(Protocol):
        user_id: int
        date: str
        steps: int

    df = pd.DataFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [-1, 1]})

    with pytest.raises(ValidationError, match="steps_non_negative"):
        DataFrame[Schema](df)


def test_unique_is_inherited_by_child_protocols():
    class ChildSchema(ActivitySchema, Protocol):
        note: str

    df = pd.DataFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [1, 2], "note": ["a", "b"]})

    with pytest.raises(ValidationError, match="must be unique"):
        DataFrame[ChildSchema](df)


def test_unique_is_skipped_when_not_required_key_column_is_absent():
    @unique("user_id", "date")
    class Schema(Protocol):
        user_id: int
        date: NotRequiredColumn[str]

    result = DataFrame[Schema](pd.DataFrame({"user_id": [1, 1]}))

    assert len(result) == 2


def test_make_empty_passes_unique():
    result = DataFrame[ActivitySchema].make_empty()

    assert len(result) == 0


def test_unique_works_when_index_shares_a_key_column_name():
    df = pd.DataFrame({"user_id": [1, 1], "date": ["d", "d"], "steps": [1, 2]})
    df = df.set_index("user_id", drop=False)

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ActivitySchema](df)

    assert str(exc_info.value) == (
        "Columns ('user_id', 'date') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1, 'date': 'd'}: rows [1, 1]"
    )


@pytest.mark.filterwarnings("error")
def test_unique_counts_only_observed_categories():
    @unique("a", "b")
    class Schema(Protocol):
        a: Any
        b: Any

    df = pd.DataFrame(
        {
            "a": pd.Categorical(["x", "x", "y"], categories=["x", "y", "z"]),
            "b": pd.Categorical(["p", "p", "q"], categories=["p", "q", "r"]),
        }
    )

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](df)

    assert str(exc_info.value) == (
        "Columns ('a', 'b') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'a': 'x', 'b': 'p'}: rows [0, 1]"
    )
