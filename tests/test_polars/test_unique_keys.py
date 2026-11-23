"""Tests for unique keys declared with the UniqueKeys marker on the polars backend."""

from dataclasses import InitVar, dataclass
from typing import ClassVar, Literal, Protocol

import pytest

pytest.importorskip("polars")

import future_annotations_schemas
import polars as pl

from pavise.checks import Key, UniqueKeys
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.types import NotRequiredColumn


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
    df = pl.DataFrame({"id": [1, 2], "user_id": [1, 1], "date": ["d1", "d2"]})

    result = DataFrame[StrictActivity](df, strict=True)

    assert result.height == 2


def test_unique_keys_checks_single_column_key():
    df = pl.DataFrame({"id": [1, 1], "user_id": [1, 2], "date": ["d", "d"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Activity](df)

    assert str(exc_info.value) == (
        "Columns ('id',) must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'id': 1}: rows [0, 1]"
    )


def test_unique_keys_checks_composite_key():
    df = pl.DataFrame({"id": [1, 2], "user_id": [1, 1], "date": ["d", "d"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[DailyActivity](df)

    assert str(exc_info.value) == (
        "Columns ('user_id', 'date') must be unique, found 1 duplicate key"
        "\n\nSample duplicate keys (showing first 1 of 1):"
        "\n  {'user_id': 1, 'date': 'd'}: rows [0, 1]"
    )


def test_unique_keys_checks_every_key():
    df = pl.DataFrame({"id": [1, 2], "user_id": [1, 1], "date": ["d", "d"]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[StrictActivity](df)

    assert str(exc_info.value).startswith("Columns ('user_id', 'date') must be unique")


def test_unique_keys_are_inherited_by_child_protocols():
    class Child(Activity, Protocol):
        steps: int

    df = pl.DataFrame({"id": [1, 1], "user_id": [1, 2], "date": ["d", "d"], "steps": [1, 2]})

    with pytest.raises(ValidationError, match=r"Columns \('id',\) must be unique"):
        DataFrame[Child](df)


def test_make_empty_passes_unique_keys():
    result = DataFrame[StrictActivity].make_empty()

    assert result.height == 0


def test_lazyframe_checks_unique_keys_on_collect():
    lf = LazyFrame[Activity](pl.LazyFrame({"id": [1, 1], "user_id": [1, 2], "date": ["d", "d"]}))

    with pytest.raises(ValidationError, match=r"Columns \('id',\) must be unique"):
        lf.collect()


@dataclass
class ClassVarKeyed(UniqueKeys[Key[Literal["kind"]]]):
    id: int
    kind: ClassVar[str] = "user"


@dataclass
class InitVarKeyed(UniqueKeys[Key[Literal["id", "check"]]]):
    id: int
    check: InitVar[bool] = False


class ClassVarKeyedProtocol(UniqueKeys[Key[Literal["kind"]]], Protocol):
    id: int
    kind: ClassVar[str]


class AnnotatedBase:
    id: int


@dataclass
class NonDataclassBaseKeyed(AnnotatedBase, UniqueKeys[Key[Literal["id"]]]):
    name: str


@pytest.mark.parametrize(
    ("schema", "column"),
    [
        pytest.param(ClassVarKeyed, "kind", id="ClassVar"),
        pytest.param(InitVarKeyed, "check", id="InitVar"),
        pytest.param(ClassVarKeyedProtocol, "kind", id="Protocol-ClassVar"),
        pytest.param(future_annotations_schemas.KindKeyedUserData, "kind", id="string-ClassVar"),
        pytest.param(NonDataclassBaseKeyed, "id", id="non-dataclass-base"),
    ],
)
def test_unique_keys_rejects_a_key_that_is_not_a_dataclass_column(schema: type, column: str):
    df = pl.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError) as exc_info:
        DataFrame[schema](df)

    assert str(exc_info.value) == (
        f"UniqueKeys: column {column!r} is not in schema {schema.__name__}; "
        "ClassVar, InitVar and annotations of non-dataclass bases are not columns"
    )


def test_lazyframe_rejects_a_key_that_is_not_a_dataclass_column():
    with pytest.raises(ValueError, match="UniqueKeys: column 'kind' is not in schema"):
        LazyFrame[ClassVarKeyed](pl.LazyFrame({"id": [1, 2]}))


@dataclass
class NotRequiredKeyed(UniqueKeys[Key[Literal["id", "kind"]]]):
    id: int
    kind: NotRequiredColumn[str]


def test_unique_keys_skips_a_dataclass_key_column_that_may_be_missing_and_is_absent():
    result = DataFrame[NotRequiredKeyed](pl.DataFrame({"id": [1, 1]}))

    assert len(result) == 2
