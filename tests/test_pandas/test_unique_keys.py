"""Tests for unique keys declared with the UniqueKeys marker on the pandas backend."""

from typing import Literal, Protocol

import pytest

pytest.importorskip("pandas")

import pandas as pd

from pavise.checks import Key, UniqueKeys
from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame


class ActivityColumns(Protocol):
    id: int
    user_id: int
    date: str


class Activity(ActivityColumns, UniqueKeys[Key[Literal["id"]]], Protocol): ...


class DailyActivity(ActivityColumns, UniqueKeys[Key[Literal["user_id", "date"]]], Protocol): ...


class StrictActivity(
    ActivityColumns,
    UniqueKeys[Key[Literal["id"]] | Key[Literal["date", "user_id"]]],
    Protocol,
): ...


def test_unique_keys_passes_when_keys_are_distinct():
    df = pd.DataFrame({"id": [1, 2], "user_id": [1, 1], "date": ["d1", "d2"]})

    result = DataFrame[StrictActivity](df, strict=True)

    assert len(result) == 2


def test_unique_keys_checks_single_column_key():
    df = pd.DataFrame({"id": [1, 1], "user_id": [1, 2], "date": ["d", "d"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Activity](df)

    assert str(exc_info.value) == (
        "Columns ('id',) must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'id': 1}: rows [0, 1]"
    )


def test_unique_keys_checks_composite_key():
    df = pd.DataFrame({"id": [1, 2], "user_id": [1, 1], "date": ["d", "d"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[DailyActivity](df)

    assert str(exc_info.value) == (
        "Columns ('user_id', 'date') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1, 'date': 'd'}: rows [0, 1]"
    )


def test_unique_keys_checks_every_key():
    df = pd.DataFrame({"id": [1, 2], "user_id": [1, 1], "date": ["d", "d"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[StrictActivity](df)

    assert str(exc_info.value).startswith("Columns ('user_id', 'date') must be unique")


def test_unique_keys_are_inherited_by_child_protocols():
    class Child(Activity, Protocol):
        steps: int

    df = pd.DataFrame({"id": [1, 1], "user_id": [1, 2], "date": ["d", "d"], "steps": [1, 2]})

    with pytest.raises(ValidationError, match=r"Columns \('id',\) must be unique"):
        DataFrame[Child](df)


def test_make_empty_passes_unique_keys():
    result = DataFrame[StrictActivity].make_empty()

    assert len(result) == 0
