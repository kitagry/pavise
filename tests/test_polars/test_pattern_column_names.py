"""Columns named like a pattern that pl.col reads as a regex or wildcard (issue #121).

Each case is validated once with such a name and once with an ordinary one, and must give
the same result, with the name swapped in the error.
"""

import warnings
from typing import Annotated, Any, Literal, Protocol, TypedDict

import polars as pl
import pytest

from pavise.checks import Key, UniqueKeys, dataframe_check
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.validators import Custom, Ge, In, MaxLen, MinLen, Range, Regex, Unique

PLAIN = "plain"
# "^o.*$" also matches the other column of the frame
PATTERN_NAMES = ["^zz$", "^o.*$", "*"]


def _error(schema: type, df: pl.DataFrame, lazy: bool) -> str | None:
    try:
        with warnings.catch_warnings():
            # Custom validators run through map_elements, which polars warns about
            warnings.simplefilter("ignore", pl.exceptions.PolarsInefficientMapWarning)
            if lazy:
                LazyFrame[schema](df.lazy()).collect()
            else:
                DataFrame[schema](df)
    except ValidationError as e:
        return str(e)
    return None


def _schema(annotations: dict[str, Any]) -> type:
    # Protocol is a special form to type checkers, but a class at runtime
    bases: tuple[Any, ...] = (Protocol,)
    return type("Schema", bases, {"__annotations__": annotations})


def _assert_same_as_plain(name: str, annotation: Any, values: list[Any], lazy: bool) -> None:
    def error(column: str) -> str | None:
        schema = _schema({column: annotation, "other": int})
        df = pl.DataFrame({column: values, "other": range(len(values))}, strict=False)
        return _error(schema, df, lazy)

    expected = error(PLAIN)
    assert error(name) == (expected and expected.replace(PLAIN, name))


COLUMN_CASES = [
    pytest.param(Annotated[int, Range(0, 1)], [-1, 0, 5], id="range"),
    pytest.param(Annotated[int, Range(0, 1)], [0, 1], id="range-passes"),
    pytest.param(Annotated[float, Ge(0)], [-1.5, 0.5], id="ge"),
    pytest.param(Annotated[str, In(["x"])], ["y", "x"], id="in"),
    pytest.param(Annotated[str, Regex("x+")], ["y", "x"], id="regex"),
    pytest.param(Annotated[str, MinLen(2), MaxLen(3)], ["y", "xx", "xxxx"], id="length"),
    pytest.param(Annotated[int, Custom(lambda v: v > 0, "must be positive")], [-1, 1], id="custom"),
    pytest.param(Annotated[int, Unique()], [1, 1, 2], id="unique"),
    pytest.param(Literal["x", "y"], ["z", "x"], id="literal"),
    pytest.param(Annotated[Literal[1, 2], Ge(2)], [1, 2], id="literal-with-validator"),
    pytest.param(int | str, [1.5, 2.0], id="union-mismatch"),
    pytest.param(Annotated[int | None, Ge(0)], [-1, None], id="optional"),
    pytest.param(int, ["a", "b"], id="type-mismatch"),
    pytest.param(int, [1, None], id="null"),
    pytest.param(Annotated[pl.Int64, Ge(0)], [-1, 1], id="dtype-annotation"),
    pytest.param(list[Annotated[int, Ge(0)]], [[-1, 1], [2]], id="list"),
    pytest.param(list[Annotated[int, Ge(0)]], [[0], [2]], id="list-passes"),
]


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
@pytest.mark.parametrize("annotation,values", COLUMN_CASES)
@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_column_validates_like_an_ordinary_one(
    name: str, annotation: Any, values: list[Any], lazy: bool
) -> None:
    _assert_same_as_plain(name, annotation, values, lazy)


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
@pytest.mark.parametrize("values", [[{"v": -1}, {"v": 1}], [{"v": 1}]], ids=["fails", "passes"])
@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_struct_column_validates_like_an_ordinary_one(
    name: str, values: list[Any], lazy: bool
) -> None:
    class Inner(TypedDict):
        v: Annotated[int, Ge(0)]

    _assert_same_as_plain(name, Inner, values, lazy)


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
@pytest.mark.parametrize("values", [[-1, 1], [1]], ids=["fails", "passes"])
@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_struct_field_validates_like_an_ordinary_one(
    name: str, values: list[int], lazy: bool
) -> None:
    def error(field: str) -> str | None:
        # Type checkers only take literal field names, which the class syntax cannot spell
        typed_dict: Any = TypedDict
        inner = typed_dict("Inner", {field: Annotated[int, Ge(0)], "w": int})
        df = pl.DataFrame({"s": [{field: v, "w": 0} for v in values]})
        return _error(_schema({"s": inner}), df, lazy)

    expected = error(PLAIN)
    assert error(name) == (expected and expected.replace(PLAIN, name))


Columns = _schema({"^zz$": int, "b": int})


class CompositeKey(Columns, UniqueKeys[Key[Literal["^zz$", "b"]]], Protocol): ...


class SingleKey(Columns, UniqueKeys[Key[Literal["^zz$"]]], Protocol): ...


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
def test_composite_unique_key_has_no_false_duplicates(lazy: bool) -> None:
    df = pl.DataFrame({"^zz$": [1, 2], "b": [1, 1]})
    assert _error(CompositeKey, df, lazy) is None


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
@pytest.mark.parametrize(
    "schema,message",
    [
        (CompositeKey, "Columns ('^zz$', 'b') must be unique, found 1 duplicate key"),
        (SingleKey, "Columns ('^zz$',) must be unique, found 1 duplicate key"),
    ],
)
def test_unique_key_reports_duplicates(schema: type, message: str, lazy: bool) -> None:
    df = pl.DataFrame({"^zz$": [1, 2, 1], "b": [1, 1, 1]})
    error = _error(schema, df, lazy)
    assert error is not None
    assert error.startswith(message)
    assert "{'^zz$': 1" in error


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
def test_dataframe_check_samples_include_the_column(lazy: bool) -> None:
    schema = dataframe_check(lambda df: df["b"] > 0, name="positive")(
        _schema({"^zz$": int, "b": int})
    )
    df = pl.DataFrame({"^zz$": [7, 8], "b": [-1, 1]})
    error = _error(schema, df, lazy)
    assert error is not None
    assert "Row 0: {'^zz$': 7, 'b': -1}" in error


@pytest.mark.parametrize("lazy", [False, True], ids=["DataFrame", "LazyFrame"])
@pytest.mark.parametrize("name", PATTERN_NAMES)
def test_dataframe_check_returning_a_series_of_that_name_validates_like_an_ordinary_one(
    name: str, lazy: bool
) -> None:
    def error(column: str) -> str | None:
        # The mask the check returns is named after the column it compares
        schema = dataframe_check(lambda df: df[column] > 0, name="positive")(
            _schema({column: int, "other": int})
        )
        df = pl.DataFrame({column: [-1, 1], "other": [0, 1]})
        return _error(schema, df, lazy)

    expected = error(PLAIN)
    assert expected is not None
    assert error(name) == expected.replace(PLAIN, name)
