from datetime import date, datetime, timedelta
from typing import Optional, Protocol

import pytest

try:
    import polars as pl

    from patrol.polars import DataFrame

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


pytestmark = pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")


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


def test_dataframe_class_getitem_returns_class():
    """DataFrame[Schema] returns a class"""
    type_of = DataFrame[SimpleSchema]
    assert isinstance(type_of, type)


def test_dataframe_with_schema_validates_correct_dataframe():
    """DataFrame[Schema](df) passes validation for correct DataFrame"""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = DataFrame[SimpleSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_schema_raises_on_missing_column():
    """DataFrame[Schema](df) raises error for missing column"""
    df = pl.DataFrame({"b": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing column: a"):
        DataFrame[SimpleSchema](df)


def test_dataframe_with_schema_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong type"""
    df = pl.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(TypeError, match="Column 'a' expected int"):
        DataFrame[SimpleSchema](df)


def test_dataframe_multiple_types():
    """Support multiple types"""
    df = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.5, 3.7],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )
    result = DataFrame[MultiTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_extra_columns():
    """Extra columns are ignored during validation"""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = DataFrame[SimpleSchema](df)
    assert result.equals(df)


def test_dataframe_datetime_types():
    """Support datetime, date, and timedelta types"""
    df = pl.DataFrame(
        {
            "created_at": [
                datetime(2024, 1, 1, 12, 0, 0),
                datetime(2024, 1, 2, 13, 30, 0),
            ],
            "event_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "duration": [timedelta(days=1), timedelta(days=2, hours=3)],
        }
    )
    result = DataFrame[DatetimeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_datetime_type_raises_on_wrong_type():
    """DataFrame raises error when datetime column has wrong type"""
    df = pl.DataFrame(
        {
            "created_at": ["2024-01-01", "2024-01-02"],  # string instead of datetime
            "event_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "duration": [timedelta(days=1), timedelta(days=2)],
        }
    )
    with pytest.raises(TypeError, match="Column 'created_at' expected datetime"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_date_type_raises_on_wrong_type():
    """DataFrame raises error when date column has wrong type"""
    df = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 2, 13, 30, 0)],
            "event_date": [1, 2],  # int instead of date
            "duration": [timedelta(days=1), timedelta(days=2)],
        }
    )
    with pytest.raises(TypeError, match="Column 'event_date' expected date"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_timedelta_type_raises_on_wrong_type():
    """DataFrame raises error when timedelta column has wrong type"""
    df = pl.DataFrame(
        {
            "created_at": [datetime(2024, 1, 1, 12, 0, 0), datetime(2024, 1, 2, 13, 30, 0)],
            "event_date": [date(2024, 1, 1), date(2024, 1, 2)],
            "duration": [1.5, 2.5],  # float instead of timedelta
        }
    )
    with pytest.raises(TypeError, match="Column 'duration' expected timedelta"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_raises_on_null_values_in_int_column():
    """DataFrame raises error when non-optional int column contains null values"""
    df = pl.DataFrame({"a": [1, 2, None]})
    with pytest.raises(TypeError, match="Column 'a' is non-optional but contains null values"):
        DataFrame[SimpleSchema](df)


def test_dataframe_raises_on_null_values_in_str_column():
    """DataFrame raises error when non-optional str column contains null values"""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})

    class SchemaWithStr(Protocol):
        a: int
        b: str

    with pytest.raises(TypeError, match="Column 'b' is non-optional but contains null values"):
        DataFrame[SchemaWithStr](df)


def test_dataframe_optional_int_accepts_null_values():
    """DataFrame with Optional[int] accepts null values"""
    df = pl.DataFrame(
        {"user_id": [1, 2, 3], "email": ["a@b.com", None, "c@d.com"], "age": [20, None, 30]}
    )
    result = DataFrame[OptionalSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_optional_type_raises_on_wrong_type():
    """DataFrame with Optional[int] still raises error for wrong type"""
    df = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "email": ["a@b.com", "b@c.com", "c@d.com"],
            "age": ["20", "25", "30"],
        }
    )
    with pytest.raises(TypeError, match="Column 'age' expected int"):
        DataFrame[OptionalSchema](df)


class PolarsDtypeSchema(Protocol):
    category: pl.Categorical
    value: pl.Int64


def test_dataframe_polars_categorical_dtype():
    """DataFrame accepts polars Categorical dtype"""
    df = pl.DataFrame(
        {
            "category": pl.Series(["A", "B", "A"], dtype=pl.Categorical),
            "value": pl.Series([1, 2, None], dtype=pl.Int64),
        }
    )
    result = DataFrame[PolarsDtypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_polars_dtype_raises_on_wrong_type():
    """DataFrame raises error when polars dtype doesn't match"""
    df = pl.DataFrame({"category": ["A", "B", "A"], "value": [1, 2, 3]})
    with pytest.raises(TypeError, match="Column 'category' expected Categorical"):
        DataFrame[PolarsDtypeSchema](df)
