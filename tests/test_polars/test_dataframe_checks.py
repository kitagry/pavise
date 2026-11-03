"""Tests for DataFrame-level checks (@dataframe_check) on the polars backend."""

from typing import Protocol

import pytest

pytest.importorskip("polars")

import polars as pl

from pavise.checks import dataframe_check
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame


@dataframe_check(lambda df: pl.col("start_at") < pl.col("end_at"), name="start_before_end")
class ExprEventSchema(Protocol):
    start_at: int
    end_at: int


@dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
class SeriesEventSchema(Protocol):
    start_at: int
    end_at: int


@pytest.mark.parametrize("schema", [ExprEventSchema, SeriesEventSchema])
def test_row_check_passes_when_all_rows_satisfy_it(schema: type):
    df = pl.DataFrame({"start_at": [1, 2], "end_at": [2, 3]})

    result = DataFrame[schema](df)

    assert result.height == 2


@pytest.mark.parametrize("schema", [ExprEventSchema, SeriesEventSchema])
def test_row_check_error_shows_violating_rows(schema: type):
    df = pl.DataFrame({"start_at": [1, 5, 2, 9], "end_at": [2, 3, 3, 1], "note": list("abcd")})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    error = exc_info.value
    assert error.column_name is None
    assert error.invalid_samples == [
        (1, {"start_at": 5, "end_at": 3}),
        (3, {"start_at": 9, "end_at": 1}),
    ]
    assert str(error) == (
        "DataFrame check 'start_before_end' failed"
        "\n\nSample invalid values (showing first 2 of 2):"
        "\n  Row 1: {'start_at': 5, 'end_at': 3}"
        "\n  Row 3: {'start_at': 9, 'end_at': 1}"
    )


def test_row_check_error_samples_are_limited_to_max_sample_size():
    df = pl.DataFrame({"start_at": [9] * 7, "end_at": [1] * 7})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ExprEventSchema](df)

    assert len(exc_info.value.invalid_samples) == 5
    assert "showing first 5 of 7" in str(exc_info.value)


@dataframe_check(lambda df: pl.col("start_at") < pl.col("end_at"), name="start_before_end")
class OptionalEventSchema(Protocol):
    start_at: int
    end_at: int | None


def test_row_check_treats_null_result_as_passing():
    df = pl.DataFrame({"start_at": [1, 2], "end_at": [2, None]})

    result = DataFrame[OptionalEventSchema](df)

    assert result.height == 2


@dataframe_check(lambda df: df["share"].sum() == 100, name="shares_sum_to_100")
class ShareSchema(Protocol):
    share: int


def test_scalar_check_passes_when_true():
    result = DataFrame[ShareSchema](pl.DataFrame({"share": [60, 40]}))

    assert result.height == 2


def test_scalar_check_error_has_no_samples():
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ShareSchema](pl.DataFrame({"share": [60, 30]}))

    assert str(exc_info.value) == "DataFrame check 'shares_sum_to_100' failed"
    assert exc_info.value.invalid_samples == []


def total_matches_parts(df: pl.DataFrame) -> pl.Series:
    return df["total"] == df["a"] + df["b"]


@dataframe_check(lambda df: pl.col("a") >= 0, name="a_non_negative")
@dataframe_check(total_matches_parts)
class TotalSchema(Protocol):
    a: int
    b: int
    total: int


def test_check_name_defaults_to_function_name():
    df = pl.DataFrame({"a": [1], "b": [2], "total": [4]})

    with pytest.raises(ValidationError, match="DataFrame check 'total_matches_parts' failed"):
        DataFrame[TotalSchema](df)


def test_checks_run_from_top_decorator_to_bottom():
    df = pl.DataFrame({"a": [-1], "b": [2], "total": [4]})

    with pytest.raises(ValidationError, match="DataFrame check 'a_non_negative' failed"):
        DataFrame[TotalSchema](df)


class ChildEventSchema(ExprEventSchema, Protocol):
    label: str


def test_checks_are_inherited_by_child_protocols():
    df = pl.DataFrame({"start_at": [5], "end_at": [3], "label": ["x"]})

    with pytest.raises(ValidationError, match="DataFrame check 'start_before_end' failed"):
        DataFrame[ChildEventSchema](df)


@dataframe_check(lambda df: "not a mask", name="bad_return")
class BadReturnSchema(Protocol):
    a: int


def test_check_returning_unsupported_type_raises_validation_error():
    with pytest.raises(
        ValidationError,
        match="DataFrame check 'bad_return' must return a boolean Series, Expr or bool",
    ):
        DataFrame[BadReturnSchema](pl.DataFrame({"a": [1]}))


def test_checks_run_with_strict_mode():
    df = pl.DataFrame({"start_at": [5], "end_at": [3]})

    with pytest.raises(ValidationError, match="DataFrame check 'start_before_end' failed"):
        DataFrame[ExprEventSchema](df, strict=True)


def test_make_empty_passes_row_checks():
    result = DataFrame[ExprEventSchema].make_empty()

    assert result.height == 0


def test_lazyframe_does_not_run_checks_at_construction():
    lf = pl.LazyFrame({"start_at": [5], "end_at": [3]})

    LazyFrame[ExprEventSchema](lf)


def test_lazyframe_runs_checks_on_collect():
    lf = LazyFrame[ExprEventSchema](pl.LazyFrame({"start_at": [5], "end_at": [3]}))

    with pytest.raises(ValidationError, match="DataFrame check 'start_before_end' failed"):
        lf.collect()


@dataframe_check(lambda df: pl.col("share").sum() == 100, name="shares_sum_to_100")
class AggregateExprShareSchema(Protocol):
    share: int


def test_aggregate_expr_check_is_treated_as_single_bool():
    df = pl.DataFrame({"share": [1, 2, 3]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[AggregateExprShareSchema](df)

    assert str(exc_info.value) == "DataFrame check 'shares_sum_to_100' failed"
    assert exc_info.value.invalid_samples == []


def test_aggregate_expr_check_passes_when_true():
    df = pl.DataFrame({"share": [60, 40, 0]})

    result = DataFrame[AggregateExprShareSchema](df)

    assert result.height == 3


@dataframe_check(lambda df: pl.col("a", "b") > 0, name="positive")
class MultiColumnExprSchema(Protocol):
    a: int
    b: int


def test_multi_column_expr_check_raises_validation_error():
    df = pl.DataFrame({"a": [1], "b": [-1]})

    with pytest.raises(ValidationError, match="single column, got 2 columns"):
        DataFrame[MultiColumnExprSchema](df)


@dataframe_check(lambda df: df["__row__"] > 0, name="positive_row")
class RowNamedColumnSchema(Protocol):
    __row__: int


def test_row_check_works_with_column_named_like_internal_row_index():
    df = pl.DataFrame({"__row__": [1, -1]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[RowNamedColumnSchema](df)

    assert exc_info.value.invalid_samples == [(1, {"__row__": -1})]
