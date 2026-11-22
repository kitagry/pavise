"""Tests for subscripting DataFrame[Schema] and LazyFrame[Schema] more than once."""

import re
import sys
import types
from typing import Protocol, get_type_hints

import pytest

pytest.importorskip("polars")

import polars as pl

from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame


class User(Protocol):
    id: int


def test_dataframe_subscript_returns_same_class() -> None:
    assert DataFrame[User] is DataFrame[User]


def test_validated_dataframe_is_instance_of_subscripted_class() -> None:
    validated = DataFrame[User](pl.DataFrame({"id": [1]}))
    # A separate subscript, the way code elsewhere would write it
    expected: type = DataFrame[User]
    assert isinstance(validated, expected)


def test_lazyframe_subscript_returns_same_class() -> None:
    assert LazyFrame[User] is LazyFrame[User]


def test_validated_lazyframe_is_instance_of_subscripted_class() -> None:
    validated = LazyFrame[User](pl.LazyFrame({"id": [1]}))
    # A separate subscript, the way code elsewhere would write it
    expected: type = LazyFrame[User]
    assert isinstance(validated, expected)


def test_collected_lazyframe_is_instance_of_subscripted_dataframe() -> None:
    validated = LazyFrame[User](pl.LazyFrame({"id": [1]})).collect()
    # A separate subscript, the way code elsewhere would write it
    expected: type = DataFrame[User]
    assert isinstance(validated, expected)


def test_dataframe_classes_differ_by_schema() -> None:
    class Other(Protocol):
        id: int

    assert DataFrame[User] is not DataFrame[Other]


@pytest.fixture
def schema_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """A module whose schemas refer to a Point class that is not defined yet."""
    module = types.ModuleType("pavise_test_late_point")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(
        "from dataclasses import dataclass\n"
        "from typing import Optional, Protocol\n"
        "class ProtocolSchema(Protocol):\n"
        "    point: 'Point'\n"
        "@dataclass\n"
        "class DataclassSchema:\n"
        "    point: 'Point'\n"
        "@dataclass\n"
        "class OptionalDataclassSchema:\n"
        "    point: Optional['Point']\n",
        module.__dict__,
    )
    return module


def _define_point(module: types.ModuleType) -> None:
    exec("class Point(Protocol):\n    x: int\n", module.__dict__)


def test_protocol_schema_validates_once_forward_reference_is_defined(
    schema_module: types.ModuleType,
) -> None:
    df = pl.DataFrame({"point": [{"x": 1}]})
    with pytest.raises(NameError, match="Point"):
        DataFrame[schema_module.ProtocolSchema](df)

    _define_point(schema_module)

    DataFrame[schema_module.ProtocolSchema](df)
    LazyFrame[schema_module.ProtocolSchema](df.lazy()).collect()


def test_dataclass_schema_validates_once_forward_reference_is_defined(
    schema_module: types.ModuleType,
) -> None:
    df = pl.DataFrame({"point": [{"x": 1}]})
    with pytest.raises(ValidationError, match="cannot resolve forward reference 'Point'"):
        DataFrame[schema_module.DataclassSchema](df)

    _define_point(schema_module)

    DataFrame[schema_module.DataclassSchema](df)
    LazyFrame[schema_module.DataclassSchema](df.lazy()).collect()


def test_optional_dataclass_schema_validates_once_forward_reference_is_defined(
    schema_module: types.ModuleType,
) -> None:
    """Before Point exists the annotation stays a ForwardRef, which must not be kept"""
    df = pl.DataFrame({"point": [{"x": 1}]})
    with pytest.raises(ValidationError):
        DataFrame[schema_module.OptionalDataclassSchema](df)

    _define_point(schema_module)

    DataFrame[schema_module.OptionalDataclassSchema](df)
    LazyFrame[schema_module.OptionalDataclassSchema](df.lazy()).collect()


def test_string_schema_reports_the_error_of_reading_its_annotations() -> None:
    """A string is no schema; the error stays the one reading its annotations gives"""
    with pytest.raises(TypeError) as expected:
        get_type_hints("User")

    with pytest.raises(TypeError, match=re.escape(str(expected.value))):
        DataFrame["User"](pl.DataFrame({"id": [1]}))
