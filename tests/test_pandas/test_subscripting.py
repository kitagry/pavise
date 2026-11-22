"""Tests for subscripting DataFrame[Schema] more than once."""

import re
import sys
import types
from typing import Protocol, get_type_hints

import pandas as pd
import pytest

from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame


class User(Protocol):
    id: int


def test_dataframe_subscript_returns_same_class() -> None:
    assert DataFrame[User] is DataFrame[User]


def test_validated_dataframe_is_instance_of_subscripted_class() -> None:
    validated = DataFrame[User](pd.DataFrame({"id": [1]}))
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


def _point_frame() -> pd.DataFrame:
    pa = pytest.importorskip("pyarrow")
    point = pa.struct([("x", pa.int64())])
    return pd.DataFrame({"point": pd.Series([{"x": 1}], dtype=pd.ArrowDtype(point))})


def test_protocol_schema_validates_once_forward_reference_is_defined(
    schema_module: types.ModuleType,
) -> None:
    df = _point_frame()
    with pytest.raises(NameError, match="Point"):
        DataFrame[schema_module.ProtocolSchema](df)

    _define_point(schema_module)

    DataFrame[schema_module.ProtocolSchema](df)


def test_dataclass_schema_validates_once_forward_reference_is_defined(
    schema_module: types.ModuleType,
) -> None:
    df = _point_frame()
    with pytest.raises(ValidationError, match="cannot resolve forward reference 'Point'"):
        DataFrame[schema_module.DataclassSchema](df)

    _define_point(schema_module)

    DataFrame[schema_module.DataclassSchema](df)


def test_optional_dataclass_schema_validates_once_forward_reference_is_defined(
    schema_module: types.ModuleType,
) -> None:
    """Before Point exists the annotation stays a ForwardRef, which must not be kept"""
    df = _point_frame()
    with pytest.raises(ValidationError):
        DataFrame[schema_module.OptionalDataclassSchema](df)

    _define_point(schema_module)

    DataFrame[schema_module.OptionalDataclassSchema](df)


def test_string_schema_reports_the_error_of_reading_its_annotations() -> None:
    """A string is no schema; the error stays the one reading its annotations gives"""
    with pytest.raises(TypeError) as expected:
        get_type_hints("User")

    with pytest.raises(TypeError, match=re.escape(str(expected.value))):
        DataFrame["User"](pd.DataFrame({"id": [1]}))
