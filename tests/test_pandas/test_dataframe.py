from datetime import date, datetime, timedelta
from typing import Optional, Protocol

import pandas as pd
import pytest

from patrol.pandas import DataFrame


class SimpleSchema(Protocol):
    a: int


class MultiTypeSchema(Protocol):
    int_col: int
    float_col: float
    str_col: str
    bool_col: bool


class DatetimeSchema(Protocol):
    created_at: datetime
    event_date: date
    duration: timedelta


class OptionalSchema(Protocol):
    user_id: int
    email: Optional[str]
    age: Optional[int]


class PandasDtypeSchema(Protocol):
    category: pd.CategoricalDtype
    value: pd.Int64Dtype


def test_dataframe_class_getitem_returns_class():
    """DataFrame[Schema] returns a class"""
    type_of = DataFrame[SimpleSchema]
    assert isinstance(type_of, type)


def test_dataframe_with_schema_validates_correct_dataframe():
    """DataFrame[Schema](df) passes validation for correct DataFrame"""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = DataFrame[SimpleSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_schema_raises_on_missing_column():
    """DataFrame[Schema](df) raises error for missing column"""
    df = pd.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing column: a"):
        DataFrame[SimpleSchema](df)


def test_dataframe_with_schema_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong type"""
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="Column 'a' expected int"):
        DataFrame[SimpleSchema](df)


def test_dataframe_multiple_types():
    """Support multiple types"""
    df = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.5, 3.7],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
    result = DataFrame[MultiTypeSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_extra_columns():
    """Extra columns are ignored during validation"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = DataFrame[SimpleSchema](df)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_datetime_types():
    """Support datetime, date, and timedelta types"""
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-02 13:30:00"]),
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "duration": pd.to_timedelta(["1 days", "2 days 3 hours"]),
        }
    )
    result = DataFrame[DatetimeSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_datetime_type_raises_on_wrong_type():
    """DataFrame raises error when datetime column has wrong type"""
    df = pd.DataFrame(
        {
            "created_at": ["2024-01-01", "2024-01-02"],  # string instead of datetime
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "duration": pd.to_timedelta(["1 days", "2 days"]),
        }
    )
    with pytest.raises(TypeError, match="Column 'created_at' expected datetime"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_date_type_raises_on_wrong_type():
    """DataFrame raises error when date column has wrong type"""
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-02 13:30:00"]),
            "event_date": [1, 2],  # int instead of date
            "duration": pd.to_timedelta(["1 days", "2 days"]),
        }
    )
    with pytest.raises(TypeError, match="Column 'event_date' expected date"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_timedelta_type_raises_on_wrong_type():
    """DataFrame raises error when timedelta column has wrong type"""
    df = pd.DataFrame(
        {
            "created_at": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-02 13:30:00"]),
            "event_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "duration": [1.5, 2.5],  # float instead of timedelta
        }
    )
    with pytest.raises(TypeError, match="Column 'duration' expected timedelta"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_raises_on_null_values_in_int_column():
    """DataFrame raises error when int column contains null values"""
    df = pd.DataFrame({"a": [1, 2, None]})
    with pytest.raises(TypeError, match="Column 'a' expected int"):
        DataFrame[SimpleSchema](df)


def test_dataframe_raises_on_null_values_in_str_column():
    """DataFrame raises error when str column contains null values"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})

    class SchemaWithStr(Protocol):
        a: int
        b: str

    with pytest.raises(TypeError, match="Column 'b' expected str"):
        DataFrame[SchemaWithStr](df)


def test_dataframe_optional_int_accepts_null_values():
    """DataFrame with Optional[int] accepts null values"""
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "email": ["a@b.com", None, "c@d.com"], "age": [20, None, 30]}
    )
    result = DataFrame[OptionalSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_optional_type_raises_on_wrong_type():
    """DataFrame with Optional[int] still raises error for wrong type"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "email": ["a@b.com", "b@c.com", "c@d.com"],
            "age": ["20", "25", "30"],
        }
    )
    with pytest.raises(TypeError, match="Column 'age' expected int"):
        DataFrame[OptionalSchema](df)


def test_dataframe_pandas_categorical_dtype():
    """DataFrame accepts pandas CategoricalDtype"""
    df = pd.DataFrame(
        {
            "category": pd.Categorical(["A", "B", "A"]),
            "value": pd.array([1, 2, None], dtype=pd.Int64Dtype()),
        }
    )
    result = DataFrame[PandasDtypeSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_pandas_dtype_raises_on_wrong_type():
    """DataFrame raises error when pandas dtype doesn't match"""
    df = pd.DataFrame({"category": ["A", "B", "A"], "value": [1, 2, 3]})
    with pytest.raises(TypeError, match="Column 'category' expected CategoricalDtype"):
        DataFrame[PandasDtypeSchema](df)
