"""Pin exact error messages and invalid samples of type checks.

These guard the vectorized fast paths: they must report the same rows, values
and counts as a per-element check would.
"""

from datetime import date, datetime
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pandas as pd
import pytest

from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame
from pavise.validators import Custom, MinLen, Range, Unique


class StrSchema(Protocol):
    v: str


class OptionalStrSchema(Protocol):
    v: str | None


class IntSchema(Protocol):
    v: int


class FloatSchema(Protocol):
    v: float


class DateSchema(Protocol):
    v: date


class OptionalDateSchema(Protocol):
    v: date | None


class UnionSchema(Protocol):
    v: int | str


class OptionalUnionSchema(Protocol):
    v: int | str | None


class LiteralSchema(Protocol):
    v: Literal["a", "b"]


class RangeSchema(Protocol):
    v: Annotated[int, Range(0, 10)]


class OptionalRangeSchema(Protocol):
    v: Annotated[int | None, Range(0, 10)]


class OptionalMinLenSchema(Protocol):
    v: Annotated[str | None, MinLen(3)]


class UniqueSchema(Protocol):
    v: Annotated[int, Unique()]


class CustomSchema(Protocol):
    v: Annotated[int, Custom(lambda x: x > 0, "must be positive")]


class StrIndexSchema(Protocol):
    __index__: str


@pytest.mark.parametrize(
    ("schema", "df", "message", "samples"),
    [
        pytest.param(
            OptionalStrSchema,
            pd.DataFrame({"v": ["a", 1, None, "b", 2.5]}),
            "Column 'v': expected str, got object\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 1: 1\n  Row 4: 2.5",
            [(1, 1), (4, 2.5)],
            id="object-str-with-non-str",
        ),
        pytest.param(
            StrSchema,
            pd.DataFrame({"v": np.arange(8)}),
            "Column 'v': expected str, got int64\n\n"
            "Sample invalid values (showing first 5 of 8):\n"
            "  Row 0: 0\n  Row 1: 1\n  Row 2: 2\n"
            "  Row 3: 3\n  Row 4: 4",
            [(i, np.int64(i)) for i in range(5)],
            id="str-got-int64",
        ),
        pytest.param(
            IntSchema,
            pd.DataFrame({"v": ["x", "y", 3]}, index=pd.Index(["r0", "r1", "r2"])),
            "Column 'v': expected int, got object\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row r0: 'x'\n  Row r1: 'y'",
            [("r0", "x"), ("r1", "y")],
            id="int-got-object-with-label-index",
        ),
        pytest.param(
            FloatSchema,
            pd.DataFrame({"v": np.arange(3)}),
            "Column 'v': expected float, got int64\n\n"
            "Sample invalid values (showing first 3 of 3):\n"
            "  Row 0: 0\n  Row 1: 1\n  Row 2: 2",
            [(i, np.int64(i)) for i in range(3)],
            id="float-got-int64",
        ),
        pytest.param(
            DateSchema,
            pd.DataFrame({"v": [date(2024, 1, 1), "x", 3]}),
            "Column 'v': expected date, got object\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 1: 'x'\n  Row 2: 3",
            [(1, "x"), (2, 3)],
            id="object-date-with-non-date",
        ),
        pytest.param(
            UnionSchema,
            pd.DataFrame({"v": [1, "a", 2.5, None, True]}),
            "Column 'v': expected int | str, got object\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 2: 2.5\n  Row 3: None",
            [(2, 2.5), (3, None)],
            id="union-object-mixed",
        ),
        pytest.param(
            OptionalUnionSchema,
            pd.DataFrame({"v": [1.0, np.nan, 2.0]}),
            "Column 'v': expected int | str, got float64\n\n"
            "Sample invalid values (showing first 2 of 2):\n"
            "  Row 0: 1.0\n  Row 2: 2.0",
            [(0, np.float64(1.0)), (2, np.float64(2.0))],
            id="optional-union-got-float64",
        ),
        pytest.param(
            UnionSchema,
            pd.DataFrame({"v": np.arange(7) * 1.5}),
            "Column 'v': expected int | str, got float64\n\n"
            "Sample invalid values (showing first 5 of 7):\n"
            "  Row 0: 0.0\n  Row 1: 1.5\n  Row 2: 3.0\n"
            "  Row 3: 4.5\n  Row 4: 6.0",
            [(i, np.float64(i * 1.5)) for i in range(5)],
            id="union-got-float64-limits-samples",
        ),
        pytest.param(
            UnionSchema,
            pd.DataFrame({"v": pd.Categorical([1, "a", 2.5])}),
            "Column 'v': expected int | str, got category\n\n"
            "Sample invalid values (showing first 1 of 1):\n  Row 2: 2.5",
            [(2, 2.5)],
            id="union-categorical-mixed",
        ),
        pytest.param(
            UnionSchema,
            pd.DataFrame({"v": pd.arrays.SparseArray([1, "a", 2.5], dtype=pd.SparseDtype(object))}),
            "Column 'v': expected int | str, got Sparse[object, nan]\n\n"
            "Sample invalid values (showing first 1 of 1):\n  Row 2: 2.5",
            [(2, 2.5)],
            id="union-sparse-object-mixed",
        ),
        pytest.param(
            LiteralSchema,
            pd.DataFrame({"v": ["a", "x", "y", "z", "w", "v", "u", "b"]}),
            "Column 'v': expected one of ('a', 'b'), got invalid values\n\n"
            "Sample invalid values (showing first 5 of 6):\n"
            "  Row 1: 'x'\n  Row 2: 'y'\n  Row 3: 'z'\n  Row 4: 'w'\n  Row 5: 'v'",
            [(1, "x"), (2, "y"), (3, "z"), (4, "w"), (5, "v")],
            id="literal-limits-samples",
        ),
        pytest.param(
            RangeSchema,
            pd.DataFrame({"v": [-1, -2, -3, -4, -5, -6, 5]}),
            "Column 'v': values must be in range [0, 10]\n\n"
            "Sample invalid values (showing first 5 of 6):\n"
            "  Row 0: -1\n  Row 1: -2\n  Row 2: -3\n  Row 3: -4\n  Row 4: -5",
            [(i, np.int64(-(i + 1))) for i in range(5)],
            id="validator-limits-samples",
        ),
        pytest.param(
            UniqueSchema,
            pd.DataFrame({"v": [1, 2, 5, 2, 5]}),
            "Column 'v': contains duplicate values\n\n"
            "Sample invalid values (showing first 4 of 4):\n"
            "  Row 1: 2\n  Row 2: 5\n  Row 3: 2\n  Row 4: 5",
            [(1, 2), (2, 5), (3, 2), (4, 5)],
            id="unique-shows-python-scalars",
        ),
        pytest.param(
            CustomSchema,
            pd.DataFrame({"v": [1, -100, 3, 0]}),
            "Column 'v': must be positive\n\n"
            "Sample invalid values (showing first 2 of 2):\n  Row 1: -100\n  Row 3: 0",
            [(1, -100), (3, 0)],
            id="custom-shows-python-scalars",
        ),
    ],
)
def test_error_reports_exact_samples(
    schema: type, df: pd.DataFrame, message: str, samples: list[tuple[Any, Any]]
):
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    assert str(exc_info.value) == message
    assert exc_info.value.invalid_samples == samples


@pytest.mark.parametrize(
    ("schema", "df"),
    [
        pytest.param(
            OptionalStrSchema,
            pd.DataFrame({"v": ["a", None, np.nan, pd.NA, "b"]}),
            id="object-str-with-nulls",
        ),
        pytest.param(StrSchema, pd.DataFrame({"v": pd.Categorical(["a", "b"])}), id="str-category"),
        pytest.param(
            OptionalDateSchema,
            pd.DataFrame({"v": [date(2024, 1, 1), None, pd.NaT, datetime(2024, 1, 2)]}),
            id="object-date-with-nulls-and-datetime",
        ),
        pytest.param(UnionSchema, pd.DataFrame({"v": np.arange(3)}), id="union-int64"),
        pytest.param(
            UnionSchema,
            pd.DataFrame({"v": pd.arrays.SparseArray([1, "a"], dtype=pd.SparseDtype(object))}),
            id="union-sparse-object",
        ),
        pytest.param(
            OptionalUnionSchema,
            pd.DataFrame({"v": [1, "a", None, np.nan, True]}),
            id="optional-union-object-with-nulls",
        ),
    ],
)
def test_valid_dataframe_passes(schema: type, df: pd.DataFrame):
    DataFrame[schema](df)


def test_str_index_with_non_str_value_raises():
    df = pd.DataFrame({"v": [1, 2]}, index=pd.Index(["a", 1], dtype=object))

    with pytest.raises(ValidationError, match="^Index expected str, got object$"):
        DataFrame[StrIndexSchema](df)


@pytest.mark.parametrize(
    ("schema", "df", "samples"),
    [
        pytest.param(
            IntSchema,
            pd.DataFrame({"v": ["a", "x", "y"]}, index=pd.Index([0, 0, 0])),
            [(0, "a"), (0, "x"), (0, "y")],
            id="type-check",
        ),
        pytest.param(
            RangeSchema,
            pd.DataFrame({"v": [5, 20, 3, 30]}, index=pd.Index([0, 0, 1, 1])),
            [(0, 20), (1, 30)],
            id="validator",
        ),
        pytest.param(
            LiteralSchema,
            pd.DataFrame({"v": ["a", "z", "w"]}, index=pd.Index(["k", "k", "k"])),
            [("k", "z"), ("k", "w")],
            id="literal",
        ),
    ],
)
def test_samples_are_positional_with_duplicate_index(
    schema: type, df: pd.DataFrame, samples: list[tuple[Any, Any]]
):
    """Duplicate index labels must not pull in every row sharing the label"""
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    assert exc_info.value.invalid_samples == samples


@pytest.mark.parametrize(
    ("schema", "df", "samples"),
    [
        pytest.param(
            OptionalRangeSchema,
            pd.DataFrame({"v": pd.array([1, None, 20], dtype="Int64")}),
            [(2, 20)],
            id="nullable-int",
        ),
        pytest.param(
            OptionalMinLenSchema,
            pd.DataFrame({"v": pd.array(["abcd", None, "a"], dtype="string")}),
            [(2, "a")],
            id="nullable-string",
        ),
    ],
)
def test_samples_skip_na_in_nullable_mask(
    schema: type, df: pd.DataFrame, samples: list[tuple[Any, Any]]
):
    """Nullable dtypes yield NA in the invalid mask; NA rows are not violations"""
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    assert exc_info.value.invalid_samples == samples


@pytest.mark.parametrize(
    ("schema", "df", "samples"),
    [
        pytest.param(
            IntSchema, pd.DataFrame({"v": ["invalid"]}), [(0, "invalid")], id="type-check"
        ),
        pytest.param(RangeSchema, pd.DataFrame({"v": [5, 20]}), [(1, 20)], id="validator"),
        pytest.param(LiteralSchema, pd.DataFrame({"v": ["a", "z"]}), [(1, "z")], id="literal"),
        pytest.param(
            RangeSchema,
            pd.DataFrame({"v": [5, 20]}, index=pd.Index([0.5, 1.5])),
            [(1.5, 20)],
            id="float-index",
        ),
        pytest.param(
            RangeSchema,
            pd.DataFrame({"v": [5, 20]}, index=pd.Index(["x", "y"])),
            [("y", 20)],
            id="str-index",
        ),
        pytest.param(
            RangeSchema,
            pd.DataFrame({"v": [5, 20]}, index=pd.date_range("2024-01-01", periods=2)),
            [(pd.Timestamp("2024-01-02"), 20)],
            id="datetime-index",
        ),
    ],
)
def test_sample_row_labels_are_python_scalars(
    schema: type, df: pd.DataFrame, samples: list[tuple[Any, Any]]
):
    """Row labels must not leak numpy scalars (np.int64(0)), matching the message and polars"""
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)

    actual = exc_info.value.invalid_samples
    assert actual == samples
    # == alone passes for np.int64(0) == 0, so pin the label types too
    assert [type(label) for label, _ in actual] == [type(label) for label, _ in samples]
