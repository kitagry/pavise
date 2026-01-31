"""Tests for LazyFrame[Schema] support."""

from typing import Literal, Optional, Protocol

import pytest

try:
    import polars as pl

    from pavise.exceptions import ValidationError
    from pavise.polars import DataFrame, LazyFrame

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


pytestmark = pytest.mark.skipif(not POLARS_AVAILABLE, reason="Polars not installed")


class SimpleSchema(Protocol):
    a: int


class MultiColumnSchema(Protocol):
    user_id: int
    name: str
    age: Optional[int]


class LiteralSchema(Protocol):
    status: Literal["pending", "approved", "rejected"]
    priority: Literal[1, 2, 3]


class TestLazyFrameBasic:
    """Basic tests for LazyFrame[Schema]"""

    def test_lazyframe_class_getitem_returns_class(self):
        """LazyFrame[Schema] returns a class"""
        typed = LazyFrame[SimpleSchema]
        assert isinstance(typed, type)

    def test_lazyframe_with_schema_validates_correct_lazyframe(self):
        """LazyFrame[Schema](lf) passes validation for correct schema"""
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = LazyFrame[SimpleSchema](lf)
        assert isinstance(result, pl.LazyFrame)

    def test_lazyframe_with_schema_raises_on_missing_column(self):
        """LazyFrame[Schema](lf) raises error for missing column"""
        lf = pl.LazyFrame({"b": [1, 2, 3]})
        with pytest.raises(ValidationError, match="Column 'a': missing"):
            LazyFrame[SimpleSchema](lf)

    def test_lazyframe_with_schema_raises_on_wrong_type(self):
        """LazyFrame[Schema](lf) raises error for wrong column type"""
        lf = pl.LazyFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValidationError, match="Column 'a': expected int"):
            LazyFrame[SimpleSchema](lf)


class TestLazyFrameCollect:
    """Tests for LazyFrame.collect()"""

    def test_collect_returns_typed_dataframe(self):
        """collect() returns DataFrame[Schema]"""
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        typed_lf = LazyFrame[SimpleSchema](lf)
        result = typed_lf.collect()

        assert isinstance(result, DataFrame)
        assert isinstance(result, pl.DataFrame)
        assert result["a"].to_list() == [1, 2, 3]

    def test_collect_validates_literal_values(self):
        """LazyFrame passes schema check but collect() fails on invalid Literal values"""
        # LazyFrame accepts this because str type matches Literal[str, ...]
        lf = pl.LazyFrame(
            {
                "status": ["pending", "invalid_value"],  # "invalid_value" is not in Literal
                "priority": [1, 2],
            }
        )
        typed_lf = LazyFrame[LiteralSchema](lf)  # Schema-level OK (str type)

        # But collect() fails because value validation happens
        with pytest.raises(ValidationError, match="status"):
            typed_lf.collect()


class TestLazyFrameMakeEmpty:
    """Tests for LazyFrame.make_empty()"""

    def test_make_empty_creates_empty_lazyframe(self):
        """make_empty() creates an empty LazyFrame with correct columns"""
        result = LazyFrame[MultiColumnSchema].make_empty()

        assert isinstance(result, LazyFrame)
        assert isinstance(result, pl.LazyFrame)

        df = result.collect()
        assert len(df) == 0
        assert set(df.columns) == {"user_id", "name", "age"}

    def test_make_empty_has_correct_dtypes(self):
        """make_empty() creates LazyFrame with correct dtypes"""
        result = LazyFrame[MultiColumnSchema].make_empty()
        schema = result.collect_schema()

        assert schema["user_id"] == pl.Int64
        assert schema["name"] == pl.Utf8
        assert schema["age"] == pl.Int64


class TestLazyFrameLiteralType:
    """Tests for Literal type validation at schema level"""

    def test_literal_str_type_raises_on_wrong_base_type(self):
        """Literal[str] should raise error when column has int type"""
        lf = pl.LazyFrame({"status": [1, 2, 3], "priority": [1, 2, 3]})
        with pytest.raises(ValidationError, match="Column 'status': expected str"):
            LazyFrame[LiteralSchema](lf)

    def test_literal_int_type_raises_on_wrong_base_type(self):
        """Literal[int] should raise error when column has str type"""
        lf = pl.LazyFrame({"status": ["pending", "approved"], "priority": ["a", "b"]})
        with pytest.raises(ValidationError, match="Column 'priority': expected int"):
            LazyFrame[LiteralSchema](lf)

    def test_literal_type_passes_with_correct_base_type(self):
        """Literal type passes when base type matches"""
        lf = pl.LazyFrame({"status": ["pending", "approved"], "priority": [1, 2]})
        result = LazyFrame[LiteralSchema](lf)
        assert isinstance(result, pl.LazyFrame)
