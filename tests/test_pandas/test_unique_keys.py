"""Tests for unique keys declared with the UniqueKeys marker on the pandas backend."""

from dataclasses import InitVar, dataclass
from typing import Any, ClassVar, Literal, Protocol

import pytest

pytest.importorskip("pandas")

import future_annotations_schemas
import numpy as np
import pandas as pd

from pavise.checks import Key, UniqueKeys
from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame
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


def test_unique_keys_reports_duplicates_beside_duplicated_column_outside_schema():
    df = pd.DataFrame(
        [[1, 1, "d", 0, 0], [2, 1, "d", 0, 0]],
        columns=pd.Index(["id", "user_id", "date", "x", "x"]),
    )

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


class NullableColumns(Protocol):
    a: str | None
    b: str | None


class NullableKey(NullableColumns, UniqueKeys[Key[Literal["a", "b"]]], Protocol): ...


def test_unique_keys_tells_null_apart_from_values_in_other_rows():
    df = pd.DataFrame({"a": [None, "x"], "b": ["q", None]})

    DataFrame[NullableKey](df)


class WideKeyColumns(Protocol):
    a: int
    b: int
    c: int
    d: int
    e: int


class WideKey(WideKeyColumns, UniqueKeys[Key[Literal["a", "b", "c", "d", "e"]]], Protocol): ...


def test_unique_keys_compares_every_column_of_a_key_with_more_combinations_than_int64():
    # 65,535 distinct values per column make 2**80 possible keys, more than int64 holds,
    # so the last row must not be mistaken for the first one, which differs only in "a".
    n = 2**16 - 1
    values = list(range(n))
    df = pd.DataFrame({c: [*values, 0] for c in "abcde"})
    df.loc[n, "a"] = 1

    DataFrame[WideKey](df)


class AnyKeyColumns(Protocol):
    a: int
    b: Any | None


class AnyKey(AnyKeyColumns, UniqueKeys[Key[Literal["a", "b"]]], Protocol): ...


def _arrow(values: list[Any], type_name: str) -> pd.Series:
    pa = pytest.importorskip("pyarrow")
    return pd.Series(values, dtype=pd.ArrowDtype(getattr(pa, type_name)()))


@pytest.mark.parametrize(
    "b",
    [
        pytest.param(lambda: [np.nan, np.nan], id="nan-nan"),
        pytest.param(lambda: [None, None], id="none-none"),
        pytest.param(lambda: pd.Series([None, np.nan], dtype=object), id="none-nan"),
        pytest.param(lambda: pd.Series([pd.NA, None], dtype=object), id="na-none"),
        pytest.param(lambda: pd.array([pd.NA, pd.NA], dtype="Int64"), id="Int64-na"),
        pytest.param(lambda: pd.array([pd.NA, 1], dtype="Int64"), id="Int64-na-value"),
        pytest.param(lambda: pd.to_datetime([None, None]), id="nat-nat"),
        pytest.param(lambda: pd.to_datetime(["2024-01-01", "2024-01-01"]), id="datetime"),
        pytest.param(
            lambda: pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC"),
            id="datetime-tz",
        ),
        pytest.param(lambda: pd.Series([1, "1"], dtype=object), id="int-str"),
        pytest.param(lambda: pd.Series([1, 1.0], dtype=object), id="int-float"),
        pytest.param(lambda: pd.Series([True, 1], dtype=object), id="bool-int"),
        pytest.param(lambda: pd.Series([False, 0.0], dtype=object), id="bool-float"),
        pytest.param(lambda: pd.Series([1.5, "1.5"], dtype=object), id="float-str"),
        pytest.param(lambda: pd.period_range("2024-01", periods=2, freq="M"), id="period"),
        pytest.param(
            lambda: pd.PeriodIndex(["2024-01", "2024-01"], freq="M"), id="period-duplicate"
        ),
        pytest.param(lambda: pd.IntervalIndex.from_breaks([0, 1, 1]), id="interval"),
        pytest.param(lambda: pd.Categorical(["x", "x"], categories=["x", "y"]), id="category"),
        pytest.param(lambda: pd.Categorical([None, None], categories=["x"]), id="category-na"),
        pytest.param(lambda: pd.Categorical(["x", "y"]), id="category-distinct"),
        pytest.param(lambda: _arrow(["x", "x"], "string"), id="arrow-string"),
        pytest.param(lambda: _arrow([None, None], "int64"), id="arrow-null"),
        pytest.param(lambda: _arrow([1, None], "int64"), id="arrow-null-value"),
        pytest.param(lambda: pd.array(["x", None], dtype="string"), id="string-na-value"),
        pytest.param(lambda: pd.array([None, None], dtype="string"), id="string-na"),
    ],
)
# pandas 3.1 warns when Series.values drops a time zone or boxes periods and intervals.
@pytest.mark.filterwarnings("error")
def test_unique_keys_finds_the_duplicates_pandas_finds(b: Any):
    df = pd.DataFrame({"a": [1, 1], "b": b()})

    if df.duplicated().any():
        with pytest.raises(ValidationError, match="must be unique"):
            DataFrame[AnyKey](df)
    else:
        DataFrame[AnyKey](df)


def test_unique_keys_are_inherited_by_child_protocols():
    class Child(Activity, Protocol):
        steps: int

    df = pd.DataFrame({"id": [1, 1], "user_id": [1, 2], "date": ["d", "d"], "steps": [1, 2]})

    with pytest.raises(ValidationError, match=r"Columns \('id',\) must be unique"):
        DataFrame[Child](df)


def test_make_empty_passes_unique_keys():
    result = DataFrame[StrictActivity].make_empty()

    assert len(result) == 0


@dataclass
class ClassVarKeyed(UniqueKeys[Key[Literal["kind"]]]):
    id: int
    kind: ClassVar[str] = "user"


@dataclass
class InitVarKeyed(UniqueKeys[Key[Literal["id", "check"]]]):
    id: int
    check: InitVar[bool] = False


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
        pytest.param(future_annotations_schemas.KindKeyedUserData, "kind", id="string-ClassVar"),
        pytest.param(NonDataclassBaseKeyed, "id", id="non-dataclass-base"),
    ],
)
def test_unique_keys_rejects_a_key_that_is_not_a_dataclass_column(schema: type, column: str):
    df = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError) as exc_info:
        DataFrame[schema](df)

    assert str(exc_info.value) == (
        f"UniqueKeys: column {column!r} is not in schema {schema.__name__}; "
        "ClassVar, InitVar and annotations of non-dataclass bases are not columns"
    )


@dataclass
class NotRequiredKeyed(UniqueKeys[Key[Literal["id", "kind"]]]):
    id: int
    kind: NotRequiredColumn[str]


def test_unique_keys_skips_a_dataclass_key_column_that_may_be_missing_and_is_absent():
    result = DataFrame[NotRequiredKeyed](pd.DataFrame({"id": [1, 1]}))

    assert len(result) == 2
