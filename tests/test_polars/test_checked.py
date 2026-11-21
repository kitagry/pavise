"""Tests for DataFrame-level checks declared with the Checked marker on the polars backend."""

from typing import Literal, Protocol

import pytest

pytest.importorskip("polars")

import polars as pl

from pavise.checks import Check, Checked, Key, UniqueKeys
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame


class StartBeforeEnd(Check):
    def __call__(self, df):
        return pl.col("start_at") < pl.col("end_at")


class PositiveStart(Check):
    def __call__(self, df):
        return df["start_at"] > 0


class EventColumns(Protocol):
    id: int
    start_at: int
    end_at: int


class Event(EventColumns, Checked[StartBeforeEnd], Protocol): ...


class StrictEvent(
    EventColumns,
    UniqueKeys[Key[Literal["id"]]],
    Checked[StartBeforeEnd | PositiveStart],
    Protocol,
): ...


def test_checked_passes_when_rows_satisfy_the_checks():
    df = pl.DataFrame({"id": [1, 2], "start_at": [1, 2], "end_at": [2, 3]})

    result = DataFrame[StrictEvent](df, strict=True)

    assert result.height == 2


def test_checked_error_names_the_check_class():
    df = pl.DataFrame({"id": [1, 2], "start_at": [1, 5], "end_at": [2, 3]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Event](df)

    assert str(exc_info.value) == (
        "DataFrame check 'StartBeforeEnd' failed"
        "\n\nSample invalid values (showing first 1 of 1):"
        "\n  Row 1: {'id': 2, 'start_at': 5, 'end_at': 3}"
    )


@pytest.mark.parametrize(
    ("start_at", "end_at", "failing_check"),
    [([1, 5], [2, 3], "StartBeforeEnd"), ([-1, 1], [2, 3], "PositiveStart")],
)
def test_checked_runs_every_check(start_at: list[int], end_at: list[int], failing_check: str):
    df = pl.DataFrame({"id": [1, 2], "start_at": start_at, "end_at": end_at})

    with pytest.raises(ValidationError, match=f"DataFrame check '{failing_check}' failed"):
        DataFrame[StrictEvent](df)


def test_checked_runs_after_unique_keys():
    df = pl.DataFrame({"id": [1, 1], "start_at": [-1, 1], "end_at": [2, 3]})

    with pytest.raises(ValidationError, match=r"Columns \('id',\) must be unique"):
        DataFrame[StrictEvent](df)


def test_checked_is_inherited_by_child_protocols():
    class Child(Event, Protocol):
        label: str

    df = pl.DataFrame({"id": [1], "start_at": [5], "end_at": [3], "label": ["x"]})

    with pytest.raises(ValidationError, match="DataFrame check 'StartBeforeEnd' failed"):
        DataFrame[Child](df)


def test_make_empty_passes_checked():
    result = DataFrame[StrictEvent].make_empty()

    assert result.height == 0


def test_lazyframe_runs_checked_on_collect():
    lf = LazyFrame[Event](pl.LazyFrame({"id": [1], "start_at": [5], "end_at": [3]}))

    with pytest.raises(ValidationError, match="DataFrame check 'StartBeforeEnd' failed"):
        lf.collect()
