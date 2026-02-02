from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Annotated, Any, Literal, Optional, Protocol, Union

import pytest

try:
    import polars as pl

    from pavise.exceptions import ValidationError
    from pavise.polars import DataFrame
    from pavise.types import NotRequiredColumn
    from pavise.validators import Range, Unique

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


class LiteralSchema(Protocol):
    status: Literal["pending", "approved", "rejected"]
    priority: Literal[1, 2, 3]


class NotRequiredSchema(Protocol):
    user_id: int
    name: str
    age: NotRequiredColumn[int]
    email: NotRequiredColumn[Optional[str]]


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
    with pytest.raises(ValidationError, match="Column 'a': missing"):
        DataFrame[SimpleSchema](df)


def test_dataframe_with_schema_raises_on_wrong_type():
    """DataFrame[Schema](df) raises error for wrong type"""
    df = pl.DataFrame({"a": ["x", "y", "z"]})
    with pytest.raises(ValidationError, match="Column 'a': expected int"):
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
    with pytest.raises(ValidationError, match="Column 'created_at': expected datetime"):
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
    with pytest.raises(ValidationError, match="Column 'event_date': expected date"):
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
    with pytest.raises(ValidationError, match="Column 'duration': expected timedelta"):
        DataFrame[DatetimeSchema](df)


def test_dataframe_raises_on_null_values_in_int_column():
    """DataFrame raises error when non-optional int column contains null values"""
    df = pl.DataFrame({"a": [1, 2, None]})
    with pytest.raises(
        ValidationError, match="Column 'a': is non-optional but contains null values"
    ):
        DataFrame[SimpleSchema](df)


def test_dataframe_raises_on_null_values_in_str_column():
    """DataFrame raises error when non-optional str column contains null values"""
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"]})

    class SchemaWithStr(Protocol):
        a: int
        b: str

    with pytest.raises(
        ValidationError, match="Column 'b': is non-optional but contains null values"
    ):
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
    with pytest.raises(ValidationError, match="Column 'age': expected int"):
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
    with pytest.raises(
        ValidationError, match="Column 'category': expected Categorical, got String"
    ):
        DataFrame[PolarsDtypeSchema](df)


def test_dataframe_ignores_extra_columns_by_default():
    """DataFrame[Schema](df) ignores extra columns by default (strict=False)"""
    df = pl.DataFrame({"a": [1, 2, 3], "extra": ["x", "y", "z"]})
    result = DataFrame[SimpleSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert "extra" in result.columns


def test_dataframe_strict_mode_raises_on_extra_columns():
    """DataFrame[Schema](df, strict=True) raises error when extra columns exist"""
    df = pl.DataFrame({"a": [1, 2, 3], "extra": ["x", "y", "z"]})
    with pytest.raises(ValidationError, match="unexpected columns"):
        DataFrame[SimpleSchema](df, strict=True)


def test_dataframe_strict_mode_passes_with_exact_columns():
    """DataFrame[Schema](df, strict=True) passes when columns exactly match schema"""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = DataFrame[SimpleSchema](df, strict=True)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_literal_type_validates_correct_values():
    """DataFrame[Schema] with Literal type validates correct values"""
    df = pl.DataFrame(
        {"status": ["pending", "approved", "rejected", "pending"], "priority": [1, 2, 3, 1]}
    )
    result = DataFrame[LiteralSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_literal_type_raises_error_for_invalid_values():
    """DataFrame[Schema] with Literal type raises ValidationError for invalid values"""
    df = pl.DataFrame({"status": ["pending", "invalid", "approved"], "priority": [1, 2, 3]})
    with pytest.raises(ValidationError, match="status"):
        DataFrame[LiteralSchema](df)


def test_dataframe_with_literal_type_raises_error_for_wrong_type():
    """DataFrame[Schema] with Literal type raises ValidationError for wrong type values"""
    df = pl.DataFrame(
        {
            "status": ["pending", "approved", "rejected"],
            "priority": [1, 99, 3],  # 99 is not in Literal[1, 2, 3]
        }
    )
    with pytest.raises(ValidationError, match="priority"):
        DataFrame[LiteralSchema](df)


def test_dataframe_with_notrequired_missing_column():
    """DataFrame[Schema] with NotRequired passes when optional column is missing"""
    df = pl.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
    result = DataFrame[NotRequiredSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_notrequired_present_and_valid():
    """DataFrame[Schema] with NotRequired validates type when column is present"""
    df = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )
    result = DataFrame[NotRequiredSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_notrequired_present_and_invalid():
    """DataFrame[Schema] with NotRequired raises error when column present but wrong type"""
    df = pl.DataFrame(
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
    df = pl.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "email": ["alice@example.com", None, "charlie@example.com"],
        }
    )
    result = DataFrame[NotRequiredSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_empty_creates_empty_dataframe_with_basic_types():
    """DataFrame.make_empty() creates an empty DataFrame with basic types"""
    result = DataFrame[MultiTypeSchema].make_empty()

    expected = pl.DataFrame(
        {
            "int_col": pl.Series([], dtype=pl.Int64),
            "float_col": pl.Series([], dtype=pl.Float64),
            "str_col": pl.Series([], dtype=pl.Utf8),
            "bool_col": pl.Series([], dtype=pl.Boolean),
        }
    )
    assert result.equals(expected)


def test_empty_creates_empty_dataframe_with_optional_types():
    """DataFrame.make_empty() creates an empty DataFrame with Optional types"""
    result = DataFrame[OptionalSchema].make_empty()

    expected = pl.DataFrame(
        {
            "user_id": pl.Series([], dtype=pl.Int64),
            "email": pl.Series([], dtype=pl.Utf8),
            "age": pl.Series(
                [], dtype=pl.Int64
            ),  # Empty DataFrame can use Int64 even for Optional[int]
        }
    )
    assert result.equals(expected)


def test_empty_creates_empty_dataframe_with_notrequired_types():
    """DataFrame.make_empty() creates an empty DataFrame including NotRequired columns"""
    result = DataFrame[NotRequiredSchema].make_empty()

    # NotRequired columns should be included in the empty DataFrame
    expected = pl.DataFrame(
        {
            "user_id": pl.Series([], dtype=pl.Int64),
            "name": pl.Series([], dtype=pl.Utf8),
            "age": pl.Series([], dtype=pl.Int64),
            "email": pl.Series([], dtype=pl.Utf8),
        }
    )
    assert result.equals(expected)


def test_empty_creates_empty_dataframe_with_literal_types():
    """DataFrame.make_empty() creates an empty DataFrame with Literal types (using base type)"""
    result = DataFrame[LiteralSchema].make_empty()

    expected = pl.DataFrame(
        {
            "status": pl.Series([], dtype=pl.Utf8),  # Literal["a", "b"] -> str -> Utf8
            "priority": pl.Series([], dtype=pl.Int64),  # Literal[1, 2, 3] -> int -> Int64
        }
    )
    assert result.equals(expected)


def test_empty_creates_empty_dataframe_with_annotated_types():
    """DataFrame.make_empty() creates an empty DataFrame with Annotated types (using base type)"""

    class AnnotatedSchema(Protocol):
        age: Annotated[int, Range(0, 150)]
        score: Annotated[float, Unique()]

    result = DataFrame[AnnotatedSchema].make_empty()

    expected = pl.DataFrame(
        {
            "age": pl.Series([], dtype=pl.Int64),
            "score": pl.Series([], dtype=pl.Float64),
        }
    )
    assert result.equals(expected)


def test_empty_creates_empty_dataframe_with_datetime_types():
    """DataFrame.make_empty() creates an empty DataFrame with datetime types"""
    result = DataFrame[DatetimeSchema].make_empty()

    expected = pl.DataFrame(
        {
            "created_at": pl.Series([], dtype=pl.Datetime),
            "event_date": pl.Series([], dtype=pl.Date),
            "duration": pl.Series([], dtype=pl.Duration),
        }
    )
    assert result.equals(expected)


class AnyTypeSchema(Protocol):
    any_col: Any
    user_id: int


def test_dataframe_with_any_type_accepts_various_types():
    """DataFrame[Schema] with Any type accepts columns with object/string dtype"""
    df = pl.DataFrame(
        {
            "any_col": ["1", "string", "3.14", "True", "False"],
            "user_id": [1, 2, 3, 4, 5],
        }
    )
    result = DataFrame[AnyTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_any_type_accepts_int_column():
    """DataFrame[Schema] with Any type accepts int column"""
    df = pl.DataFrame(
        {
            "any_col": [1, 2, 3],
            "user_id": [1, 2, 3],
        }
    )
    result = DataFrame[AnyTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_any_type_accepts_str_column():
    """DataFrame[Schema] with Any type accepts str column"""
    df = pl.DataFrame(
        {
            "any_col": ["a", "b", "c"],
            "user_id": [1, 2, 3],
        }
    )
    result = DataFrame[AnyTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


class UnionTypeSchema(Protocol):
    code: Union[int, str]
    value: float


class UnionMultiTypeSchema(Protocol):
    mixed: Union[int, str, float]
    value: int


class UnionWithNoneSchema(Protocol):
    code: Union[int, str, None]
    name: str


# PEP 604 Union operator (str | int) test schemas
class Pep604UnionTypeSchema(Protocol):
    code: str | int
    value: float


class Pep604UnionMultiTypeSchema(Protocol):
    mixed: int | str | float
    value: int


class Pep604UnionWithNoneSchema(Protocol):
    code: int | str | None
    name: str


def test_dataframe_with_union_type_accepts_int():
    """DataFrame[Schema] with Union[int, str] accepts int values"""
    df = pl.DataFrame({"code": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    result = DataFrame[UnionTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_union_type_accepts_str():
    """DataFrame[Schema] with Union[int, str] accepts str values"""
    df = pl.DataFrame({"code": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]})
    result = DataFrame[UnionTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_union_type_accepts_mixed():
    """DataFrame[Schema] with Union[int, str] accepts mixed int/str values"""
    df = pl.DataFrame(
        {"code": pl.Series([1, "B", 3, "D"], dtype=pl.Object), "value": [1.0, 2.0, 3.0, 4.0]}
    )
    result = DataFrame[UnionTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_union_type_raises_on_wrong_type():
    """DataFrame[Schema] with Union[int, str] raises error for invalid type"""
    df = pl.DataFrame({"code": pl.Series([1, 2.5, 3], dtype=pl.Object), "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError, match="Column 'code': expected int \\| str"):
        DataFrame[UnionTypeSchema](df)


def test_dataframe_with_union_multi_type_accepts_all_types():
    """DataFrame[Schema] with Union[int, str, float] accepts all union types"""
    df = pl.DataFrame(
        {"mixed": pl.Series([1, "text", 3.14, 42], dtype=pl.Object), "value": [1, 2, 3, 4]}
    )
    result = DataFrame[UnionMultiTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_union_and_none_accepts_none():
    """DataFrame[Schema] with Union[int, str, None] accepts None values"""
    df = pl.DataFrame(
        {"code": pl.Series([1, "B", None, 4], dtype=pl.Object), "name": ["a", "b", "c", "d"]}
    )
    result = DataFrame[UnionWithNoneSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_union_and_none_accepts_mixed():
    """DataFrame[Schema] with Union[int, str, None] accepts mixed values"""
    df = pl.DataFrame({"code": pl.Series([1, None, "C"], dtype=pl.Object), "name": ["a", "b", "c"]})
    result = DataFrame[UnionWithNoneSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_union_and_none_raises_on_invalid_type():
    """DataFrame[Schema] with Union[int, str, None] raises error for invalid type"""
    df = pl.DataFrame({"code": pl.Series([1, "B", 3.14], dtype=pl.Object), "name": ["a", "b", "c"]})
    with pytest.raises(ValidationError, match="Column 'code': expected int \\| str"):
        DataFrame[UnionWithNoneSchema](df)


def test_empty_creates_empty_dataframe_with_union_types():
    """DataFrame.make_empty() creates an empty DataFrame with Union types"""
    result = DataFrame[UnionTypeSchema].make_empty()

    expected = pl.DataFrame(
        {
            "code": pl.Series([], dtype=pl.Int64),  # Use first type in Union
            "value": pl.Series([], dtype=pl.Float64),
        }
    )
    assert result.equals(expected)


# PEP 604 Union operator (str | int) tests
def test_dataframe_with_pep604_union_type_accepts_int():
    """DataFrame[Schema] with str | int accepts int values"""
    df = pl.DataFrame({"code": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    result = DataFrame[Pep604UnionTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_pep604_union_type_accepts_str():
    """DataFrame[Schema] with str | int accepts str values"""
    df = pl.DataFrame({"code": ["A", "B", "C"], "value": [1.0, 2.0, 3.0]})
    result = DataFrame[Pep604UnionTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    assert result.equals(df)


def test_dataframe_with_pep604_union_type_accepts_mixed():
    """DataFrame[Schema] with str | int accepts mixed int/str values"""
    df = pl.DataFrame(
        {"code": pl.Series([1, "B", 3, "D"], dtype=pl.Object), "value": [1.0, 2.0, 3.0, 4.0]}
    )
    result = DataFrame[Pep604UnionTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_pep604_union_type_raises_on_wrong_type():
    """DataFrame[Schema] with str | int raises error for invalid type"""
    df = pl.DataFrame({"code": pl.Series([1, 2.5, 3], dtype=pl.Object), "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValidationError, match="Column 'code': expected str \\| int"):
        DataFrame[Pep604UnionTypeSchema](df)


def test_dataframe_with_pep604_union_multi_type_accepts_all_types():
    """DataFrame[Schema] with int | str | float accepts all union types"""
    df = pl.DataFrame(
        {"mixed": pl.Series([1, "text", 3.14, 42], dtype=pl.Object), "value": [1, 2, 3, 4]}
    )
    result = DataFrame[Pep604UnionMultiTypeSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_pep604_union_and_none_accepts_none():
    """DataFrame[Schema] with int | str | None accepts None values"""
    df = pl.DataFrame(
        {"code": pl.Series([1, "B", None, 4], dtype=pl.Object), "name": ["a", "b", "c", "d"]}
    )
    result = DataFrame[Pep604UnionWithNoneSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_pep604_union_and_none_accepts_mixed():
    """DataFrame[Schema] with int | str | None accepts mixed values"""
    df = pl.DataFrame({"code": pl.Series([1, None, "C"], dtype=pl.Object), "name": ["a", "b", "c"]})
    result = DataFrame[Pep604UnionWithNoneSchema](df)
    assert isinstance(result, pl.DataFrame)
    # Validation passed - check that data is preserved
    assert result.shape == df.shape
    assert result.columns == df.columns


def test_dataframe_with_pep604_union_and_none_raises_on_invalid_type():
    """DataFrame[Schema] with int | str | None raises error for invalid type"""
    df = pl.DataFrame({"code": pl.Series([1, "B", 3.14], dtype=pl.Object), "name": ["a", "b", "c"]})
    with pytest.raises(ValidationError, match="Column 'code': expected int \\| str"):
        DataFrame[Pep604UnionWithNoneSchema](df)


def test_empty_creates_empty_dataframe_with_pep604_union_types():
    """DataFrame.make_empty() creates an empty DataFrame with PEP 604 Union types"""
    result = DataFrame[Pep604UnionTypeSchema].make_empty()

    expected = pl.DataFrame(
        {
            "code": pl.Series([], dtype=pl.Object),  # Use first type in Union
            "value": pl.Series([], dtype=pl.Float64),
        }
    )
    assert result.equals(expected)
