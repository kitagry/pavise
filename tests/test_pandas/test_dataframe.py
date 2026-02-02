from datetime import date, datetime, timedelta
from typing import Annotated, Any, Literal, Optional, Protocol

import pandas as pd
import pytest

from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame
from pavise.types import NotRequiredColumn
from pavise.validators import Range, Unique


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


class LiteralSchema(Protocol):
    status: Literal["pending", "approved", "rejected"]
    priority: Literal[1, 2, 3]


class NotRequiredSchema(Protocol):
    user_id: int
    name: str
    age: NotRequiredColumn[int]
    email: NotRequiredColumn[Optional[str]]


class DateSchema(Protocol):
    event_date: date


class NullableDateSchema(Protocol):
    event_date: Optional[date]


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
    with pytest.raises(ValidationError, match="Column 'a': missing"):
        DataFrame[SimpleSchema](df)


def test_dataframe_with_schema_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong type"""
    df = pd.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(ValidationError, match="Column 'a': expected int"):
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
    with pytest.raises(ValidationError, match="Column 'created_at': expected datetime"):
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
    with pytest.raises(ValidationError, match="Column 'event_date': expected date"):
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
    with pytest.raises(ValidationError, match="Column 'duration': expected timedelta"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_raises_on_null_values_in_int_column():
    """DataFrame raises error when int column contains null values"""
    df = pd.DataFrame({"a": [1, 2, None]})
    with pytest.raises(
        ValidationError, match="Column 'a': is non-optional but contains null values"
    ):
        DataFrame[SimpleSchema](df)


def test_dataframe_raises_on_null_values_in_str_column():
    """DataFrame raises error when str column contains null values"""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})

    class SchemaWithStr(Protocol):
        a: int
        b: str

    with pytest.raises(
        ValidationError, match="Column 'b': is non-optional but contains null values"
    ):
        DataFrame[SchemaWithStr](df)


def test_dataframe_optional_int_accepts_null_values():
    """DataFrame with Optional[int] accepts null values"""
    df = pd.DataFrame(
        {"user_id": [1, 2, 3], "email": ["a@b.com", None, "c@d.com"], "age": [20, None, 30]}
    ).astype({"age": "Int64"})
    result = DataFrame[OptionalSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_optional_str_contains_int():
    """DataFrame with Optional[int] accepts null values"""
    df = pd.DataFrame({"email": ["a@b.com", None, 2]})

    class OptionalStrSchema(Protocol):
        email: Optional[str]

    with pytest.raises(ValidationError, match="Column 'email': expected str, got object"):
        DataFrame[OptionalStrSchema](df)


def test_dataframe_optional_type_raises_on_wrong_type():
    """DataFrame with Optional[int] still raises error for wrong type"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "email": ["a@b.com", "b@c.com", "c@d.com"],
            "age": ["20", "25", "30"],
        }
    )
    with pytest.raises(ValidationError, match="Column 'age': expected int, got (object|str)"):
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
    with pytest.raises(ValidationError, match="Column 'category': expected CategoricalDtype"):
        DataFrame[PandasDtypeSchema](df)


class IndexIntSchema(Protocol):
    __index__: int
    value: float


def test_dataframe_with_index_type_validates_correct_index():
    """DataFrame[Schema](df) passes validation for correct index type"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    result = DataFrame[IndexIntSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_index_type_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong index type"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=["a", "b", "c"])
    with pytest.raises(ValidationError, match="Index expected int"):
        DataFrame[IndexIntSchema](df)


class IndexWithNameSchema(Protocol):
    __index__: Annotated[int, "user_id"]
    value: float


def test_dataframe_with_index_name_validates_correct_index():
    """DataFrame[Schema](df) passes validation for correct index name"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.Index([0, 1, 2], name="user_id"))
    result = DataFrame[IndexWithNameSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_index_name_raises_on_wrong_name():
    """DataFrame[Schema](df) raises error for wrong index name"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.Index([0, 1, 2], name="id"))
    with pytest.raises(ValidationError, match="Index name expected 'user_id', got 'id'"):
        DataFrame[IndexWithNameSchema](df)


def test_dataframe_with_index_name_raises_on_missing_name():
    """DataFrame[Schema](df) raises error when index name is None"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    with pytest.raises(ValidationError, match="Index name expected 'user_id', got None"):
        DataFrame[IndexWithNameSchema](df)


class MultiIndexSchema(Protocol):
    __index__: tuple[str, int]
    value: float


def test_dataframe_with_multiindex_validates_correct_types():
    """DataFrame[Schema](df) passes validation for correct MultiIndex types"""
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples([("a", 0), ("b", 1), ("c", 2)]),
    )
    result = DataFrame[MultiIndexSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_multiindex_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong MultiIndex type"""
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples([(0, 0), (1, 1), (2, 2)]),
    )
    with pytest.raises(ValidationError, match="Index level 0 expected str"):
        DataFrame[MultiIndexSchema](df)


def test_dataframe_with_multiindex_raises_on_single_index():
    """DataFrame[Schema](df) raises error when MultiIndex expected but got single index"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    with pytest.raises(ValidationError, match="Expected MultiIndex with 2 levels, got Index"):
        DataFrame[MultiIndexSchema](df)


class MultiIndexWithNamesSchema(Protocol):
    __index__: Annotated[tuple[str, int], ("region", "user_id")]
    value: float


def test_dataframe_with_multiindex_names_validates_correct_names():
    """DataFrame[Schema](df) passes validation for correct MultiIndex names"""
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [("a", 0), ("b", 1), ("c", 2)], names=["region", "user_id"]
        ),
    )
    result = DataFrame[MultiIndexWithNamesSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_multiindex_raises_on_wrong_names():
    """DataFrame[Schema](df) raises error for wrong MultiIndex names"""
    df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples([("a", 0), ("b", 1), ("c", 2)], names=["area", "id"]),
    )
    with pytest.raises(
        ValidationError,
        match="Index names expected \\('region', 'user_id'\\), got \\('area', 'id'\\)",
    ):
        DataFrame[MultiIndexWithNamesSchema](df)


class IndexWithValidatorSchema(Protocol):
    __index__: Annotated[int, Range(0, 10)]
    value: float


def test_dataframe_with_index_validator_passes_validation():
    """DataFrame[Schema](df) passes validation when index values satisfy validator"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[0, 5, 10])
    result = DataFrame[IndexWithValidatorSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_index_validator_raises_on_invalid_value():
    """DataFrame[Schema](df) raises error when index value violates validator"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[0, 5, 15])
    with pytest.raises(ValidationError, match="values must be in"):
        DataFrame[IndexWithValidatorSchema](df)


class IndexWithNameAndValidatorSchema(Protocol):
    __index__: Annotated[int, "user_id", Range(0, 100), Unique()]
    value: float


def test_dataframe_with_index_name_and_validator_passes():
    """DataFrame[Schema](df) passes validation with both name and validators"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.Index([0, 50, 100], name="user_id"))
    result = DataFrame[IndexWithNameAndValidatorSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_index_unique_validator_raises_on_duplicates():
    """DataFrame[Schema](df) raises error when index has duplicate values"""
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=pd.Index([0, 50, 50], name="user_id"))
    with pytest.raises(ValidationError, match="contains duplicate values"):
        DataFrame[IndexWithNameAndValidatorSchema](df)


def test_dataframe_ignores_extra_columns_by_default():
    """DataFrame[Schema](df) ignores extra columns by default (strict=False)"""
    df = pd.DataFrame({"a": [1, 2, 3], "extra": ["x", "y", "z"]})
    result = DataFrame[SimpleSchema](df)
    assert isinstance(result, pd.DataFrame)
    assert "extra" in result.columns


def test_dataframe_strict_mode_raises_on_extra_columns():
    """DataFrame[Schema](df, strict=True) raises error when extra columns exist"""
    df = pd.DataFrame({"a": [1, 2, 3], "extra": ["x", "y", "z"]})
    with pytest.raises(ValidationError, match="unexpected columns"):
        DataFrame[SimpleSchema](df, strict=True)


def test_dataframe_strict_mode_passes_with_exact_columns():
    """DataFrame[Schema](df, strict=True) passes when columns exactly match schema"""
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = DataFrame[SimpleSchema](df, strict=True)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_literal_type_validates_correct_values():
    """DataFrame[Schema] with Literal type validates correct values"""
    df = pd.DataFrame(
        {"status": ["pending", "approved", "rejected", "pending"], "priority": [1, 2, 3, 1]}
    )
    result = DataFrame[LiteralSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_literal_type_raises_error_for_invalid_values():
    """DataFrame[Schema] with Literal type raises ValidationError for invalid values"""
    df = pd.DataFrame({"status": ["pending", "invalid", "approved"], "priority": [1, 2, 3]})
    with pytest.raises(ValidationError, match="status"):
        DataFrame[LiteralSchema](df)


def test_dataframe_with_literal_type_raises_error_for_wrong_type():
    """DataFrame[Schema] with Literal type raises ValidationError for wrong type values"""
    df = pd.DataFrame(
        {
            "status": ["pending", "approved", "rejected"],
            "priority": [1, 99, 3],  # 99 is not in Literal[1, 2, 3]
        }
    )
    with pytest.raises(ValidationError, match="priority"):
        DataFrame[LiteralSchema](df)


def test_dataframe_with_notrequired_missing_column():
    """DataFrame[Schema] with NotRequired passes when optional column is missing"""
    df = pd.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    result = DataFrame[NotRequiredSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_notrequired_present_and_valid():
    """DataFrame[Schema] with NotRequired validates type when column is present"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )
    result = DataFrame[NotRequiredSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_notrequired_present_and_invalid():
    """DataFrame[Schema] with NotRequired raises error when column present but wrong type"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": ["25", "30", "35"],  # str instead of int
        }
    )
    with pytest.raises(ValidationError, match="age"):
        DataFrame[NotRequiredSchema](df)


def test_dataframe_with_notrequired_optional_combination():
    """DataFrame[Schema] with NotRequired[Optional[T]] allows None when column is present"""
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@example.com", None, "charlie@example.com"],
        }
    )
    result = DataFrame[NotRequiredSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_empty_creates_empty_dataframe_with_basic_types():
    """DataFrame.make_empty() creates an empty DataFrame with basic types"""
    result = DataFrame[MultiTypeSchema].make_empty()

    expected = pd.DataFrame(
        {
            "int_col": pd.Series([], dtype="int64"),
            "float_col": pd.Series([], dtype="float64"),
            "str_col": pd.Series([], dtype="object"),
            "bool_col": pd.Series([], dtype="bool"),
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_empty_creates_empty_dataframe_with_optional_types():
    """DataFrame.make_empty() creates an empty DataFrame with Optional types"""
    result = DataFrame[OptionalSchema].make_empty()

    expected = pd.DataFrame(
        {
            "user_id": pd.Series([], dtype="int64"),
            "email": pd.Series([], dtype="object"),
            "age": pd.Series(
                [], dtype="int64"
            ),  # Empty DataFrame can use int64 even for Optional[int]
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_empty_creates_empty_dataframe_with_notrequired_types():
    """DataFrame.make_empty() creates an empty DataFrame including NotRequired columns"""
    result = DataFrame[NotRequiredSchema].make_empty()

    # NotRequired columns should be included in the empty DataFrame
    expected = pd.DataFrame(
        {
            "user_id": pd.Series([], dtype="int64"),
            "name": pd.Series([], dtype="object"),
            "age": pd.Series([], dtype="int64"),
            "email": pd.Series([], dtype="object"),
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_empty_creates_empty_dataframe_with_literal_types():
    """DataFrame.make_empty() creates an empty DataFrame with Literal types (using base type)"""
    result = DataFrame[LiteralSchema].make_empty()

    expected = pd.DataFrame(
        {
            "status": pd.Series([], dtype="object"),  # Literal["a", "b"] -> str -> object
            "priority": pd.Series([], dtype="int64"),  # Literal[1, 2, 3] -> int -> int64
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_empty_creates_empty_dataframe_with_annotated_types():
    """DataFrame.make_empty() creates an empty DataFrame with Annotated types (using base type)"""

    class AnnotatedSchema(Protocol):
        age: Annotated[int, Range(0, 150)]
        score: Annotated[float, Unique()]

    result = DataFrame[AnnotatedSchema].make_empty()

    expected = pd.DataFrame(
        {
            "age": pd.Series([], dtype="int64"),
            "score": pd.Series([], dtype="float64"),
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_empty_creates_empty_dataframe_with_datetime_types():
    """DataFrame.make_empty() creates an empty DataFrame with datetime types"""
    result = DataFrame[DatetimeSchema].make_empty()

    expected = pd.DataFrame(
        {
            "created_at": pd.Series([], dtype="datetime64[ns]"),
            "event_date": pd.Series([], dtype="datetime64[ns]"),
            "duration": pd.Series([], dtype="timedelta64[ns]"),
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_datetime_schema_date_column():
    """DataFrame[DateSchema] correctly validates date column"""
    df = pd.DataFrame(
        {
            "event_date": [
                date(2024, 1, 1),
                pd.to_datetime("2024-01-02").date(),
                datetime(2024, 1, 3),  # datetime is acceptable as date
            ],
        }
    )
    result = DataFrame[DateSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_datetime_schema_date_column_nullable():
    """DataFrame[NullableDateSchema] correctly validates nullable date column"""
    df = pd.DataFrame(
        {
            "event_date": [
                date(2024, 1, 1),
                None,  # Nullable date accepts None
                pd.NaT,  # Nullable date accepts pd.NaT
            ],
        }
    )
    result = DataFrame[NullableDateSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_empty_creates_empty_dataframe_with_index_name():
    """DataFrame.make_empty() creates an empty DataFrame with index name"""
    result = DataFrame[IndexWithNameSchema].make_empty()

    assert result.index.name == "user_id"
    assert len(result) == 0
    assert "value" in result.columns


def test_empty_creates_empty_dataframe_with_multiindex_names():
    """DataFrame.make_empty() creates an empty DataFrame with MultiIndex names"""
    result = DataFrame[MultiIndexWithNamesSchema].make_empty()

    assert isinstance(result.index, pd.MultiIndex)
    assert result.index.names == ["region", "user_id"]
    assert len(result) == 0
    assert "value" in result.columns


class AnyTypeSchema(Protocol):
    any_col: Any
    user_id: int


def test_dataframe_with_any_type_accepts_various_types():
    """DataFrame[Schema] with Any type accepts columns with various types of values"""
    df = pd.DataFrame(
        {
            "any_col": [1, "string", 3.14, True, False],
            "user_id": [1, 2, 3, 4, 5],
        }
    )
    result = DataFrame[AnyTypeSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_any_type_accepts_int_column():
    """DataFrame[Schema] with Any type accepts int column"""
    df = pd.DataFrame(
        {
            "any_col": [1, 2, 3],
            "user_id": [1, 2, 3],
        }
    )
    result = DataFrame[AnyTypeSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)


def test_dataframe_with_any_type_accepts_str_column():
    """DataFrame[Schema] with Any type accepts str column"""
    df = pd.DataFrame(
        {
            "any_col": ["a", "b", "c"],
            "user_id": [1, 2, 3],
        }
    )
    result = DataFrame[AnyTypeSchema](df)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, df)
