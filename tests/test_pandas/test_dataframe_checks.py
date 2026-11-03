"""Tests for DataFrame-level checks (@dataframe_check) on the pandas backend."""

from typing import Protocol

import pandas as pd
import pytest

from pavise.checks import dataframe_check
from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame


@dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
class EventSchema(Protocol):
    start_at: int
    end_at: int


def test_row_check_passes_when_all_rows_satisfy_it():
    df = pd.DataFrame({"start_at": [1, 2], "end_at": [2, 3]})

    result = DataFrame[EventSchema](df)

    assert len(result) == 2


def test_row_check_error_shows_violating_rows():
    df = pd.DataFrame({"start_at": [1, 5, 2, 9], "end_at": [2, 3, 3, 1], "note": list("abcd")})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[EventSchema](df)

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
    df = pd.DataFrame({"start_at": [9] * 7, "end_at": [1] * 7})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[EventSchema](df)

    assert len(exc_info.value.invalid_samples) == 5
    assert "showing first 5 of 7" in str(exc_info.value)


@dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
class OptionalEventSchema(Protocol):
    start_at: float
    end_at: float | None


def test_row_check_treats_na_result_as_passing():
    # Nullable dtypes propagate NA through comparisons, unlike numpy NaN which yields False.
    df = pd.DataFrame({"start_at": [1.0, 2.0], "end_at": pd.array([2.0, None], dtype="Float64")})

    result = DataFrame[OptionalEventSchema](df)

    assert len(result) == 2


@dataframe_check(lambda df: bool(df["share"].sum() == 100), name="shares_sum_to_100")
class ShareSchema(Protocol):
    share: int


def test_scalar_check_passes_when_true():
    df = pd.DataFrame({"share": [60, 40]})

    result = DataFrame[ShareSchema](df)

    assert len(result) == 2


def test_scalar_check_error_has_no_samples():
    df = pd.DataFrame({"share": [60, 30]})

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[ShareSchema](df)

    assert str(exc_info.value) == "DataFrame check 'shares_sum_to_100' failed"
    assert exc_info.value.invalid_samples == []


def total_matches_parts(df: pd.DataFrame) -> pd.Series:
    return df["total"] == df["a"] + df["b"]


@dataframe_check(lambda df: df["a"] >= 0, name="a_non_negative")
@dataframe_check(total_matches_parts)
class TotalSchema(Protocol):
    a: int
    b: int
    total: int


def test_check_name_defaults_to_function_name():
    df = pd.DataFrame({"a": [1], "b": [2], "total": [4]})

    with pytest.raises(ValidationError, match="DataFrame check 'total_matches_parts' failed"):
        DataFrame[TotalSchema](df)


def test_checks_run_from_top_decorator_to_bottom():
    df = pd.DataFrame({"a": [-1], "b": [2], "total": [4]})

    with pytest.raises(ValidationError, match="DataFrame check 'a_non_negative' failed"):
        DataFrame[TotalSchema](df)


class ChildEventSchema(EventSchema, Protocol):
    label: str


def test_checks_are_inherited_by_child_protocols():
    df = pd.DataFrame({"start_at": [5], "end_at": [3], "label": ["x"]})

    with pytest.raises(ValidationError, match="DataFrame check 'start_before_end' failed"):
        DataFrame[ChildEventSchema](df)


@dataframe_check(lambda df: df["end_at"] < 100, name="end_before_100")
class ChildWithOwnCheckSchema(EventSchema, Protocol):
    pass


def test_child_protocol_runs_parent_checks_before_its_own():
    df = pd.DataFrame({"start_at": [5], "end_at": [300]})

    DataFrame[EventSchema](df)
    with pytest.raises(ValidationError, match="DataFrame check 'end_before_100' failed"):
        DataFrame[ChildWithOwnCheckSchema](df)
    with pytest.raises(ValidationError, match="DataFrame check 'start_before_end' failed"):
        DataFrame[ChildWithOwnCheckSchema](pd.DataFrame({"start_at": [500], "end_at": [300]}))


def test_parent_is_not_affected_by_child_checks():
    df = pd.DataFrame({"start_at": [5], "end_at": [300]})

    result = DataFrame[EventSchema](df)

    assert len(result) == 1


@dataframe_check(lambda df: "not a mask", name="bad_return")
class BadReturnSchema(Protocol):
    a: int


def test_check_returning_unsupported_type_raises_validation_error():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(
        ValidationError, match="DataFrame check 'bad_return' must return a boolean Series or bool"
    ):
        DataFrame[BadReturnSchema](df)


def test_checks_run_after_column_validation():
    df = pd.DataFrame({"start_at": [1]})

    with pytest.raises(ValidationError, match="Column 'end_at': missing"):
        DataFrame[EventSchema](df)


def test_checks_run_with_strict_mode():
    df = pd.DataFrame({"start_at": [5], "end_at": [3]})

    with pytest.raises(ValidationError, match="DataFrame check 'start_before_end' failed"):
        DataFrame[EventSchema](df, strict=True)


class StructuralEventSchema(Protocol):
    start_at: int
    end_at: int
    label: str


def count_events(df: DataFrame[EventSchema]) -> int:
    return len(df)


def test_checks_do_not_break_structural_subtyping():
    # pyright verifies this call: a decorated Protocol must still accept
    # structurally compatible schemas that do not inherit from it.
    df = DataFrame[StructuralEventSchema](
        pd.DataFrame({"start_at": [5], "end_at": [3], "label": ["x"]})
    )

    assert count_events(df) == 1


def test_make_empty_passes_row_checks():
    result = DataFrame[EventSchema].make_empty()

    assert len(result) == 0


@dataframe_check(lambda df: df.groupby("g")["a"].sum() > 0, name="group_sum_positive")
class GroupedResultSchema(Protocol):
    g: str
    a: int


def test_check_returning_unaligned_series_raises_validation_error():
    df = pd.DataFrame({"g": ["x", "x", "y"], "a": [1, -5, 2]})

    with pytest.raises(ValidationError, match="aligned with the DataFrame's index"):
        DataFrame[GroupedResultSchema](df)


@dataframe_check(lambda df: df.apply(lambda row: row["a"] > 0, axis=1), name="a_positive")
class ApplyRowSchema(Protocol):
    a: int


def test_make_empty_passes_row_wise_apply_checks():
    result = DataFrame[ApplyRowSchema].make_empty()

    assert len(result) == 0


def test_row_check_samples_are_positional_with_duplicate_index():
    """Duplicate index labels must not pull valid rows into the samples"""
    df = pd.DataFrame({"start_at": [1, 5], "end_at": [2, 3]}, index=pd.Index([0, 0]))

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[EventSchema](df)

    assert exc_info.value.invalid_samples == [(0, {"start_at": 5, "end_at": 3})]
