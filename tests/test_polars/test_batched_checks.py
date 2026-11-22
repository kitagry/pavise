"""Checks evaluated together in one select must report exactly what checking each alone does."""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol

import polars as pl
import pytest

from pavise._polars import validation
from pavise._polars.validator_impl import apply_validator
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.types import NotRequiredColumn
from pavise.validators import Custom, Ge, Gt, In, Le, Lt, MaxLen, MinLen, Range, Regex, Unique


@pytest.fixture(params=["in-memory", "streaming"], autouse=True)
def engine(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every test on both engines the batched checks can pick."""
    if request.param == "in-memory":
        monkeypatch.setattr(validation, "_STREAMING_MIN_ROWS", 10**18)
    else:
        monkeypatch.setattr(validation, "_STREAMING_MIN_ROWS", 0)
    return request.param


def _schema(annotation: Any) -> type:
    return type("Schema", (), {"__annotations__": {"c": annotation}})


def _expected_error(series: pl.Series, validator: Any) -> Exception | None:
    """The error checking the validator alone on the column raises, if any."""
    try:
        apply_validator(series, validator, "c")
    except Exception as e:  # some combinations raise polars' own errors
        return e
    return None


def _assert_raises_like(expected: Exception, frame: pl.DataFrame, schema: type) -> None:
    with pytest.raises(type(expected)) as excinfo:
        DataFrame[schema](frame)
    assert str(excinfo.value) == str(expected)


CASES: list[tuple[list[Any], pl.DataType, Any]] = [
    ([1, 2, None], pl.Int64(), Range(0, 2)),
    ([1, 2, 3], pl.Int64(), Range(0, 2)),
    ([0, 1], pl.UInt8(), Ge(-1)),
    ([0, 1], pl.UInt8(), Lt(-1)),
    ([0, 255], pl.UInt8(), Gt(300)),
    ([0, 255], pl.UInt8(), Le(300)),
    ([0, 255, None], pl.UInt8(), Range(-5, 300)),
    ([1, None], pl.Int8(), Ge(200)),
    ([1], pl.Int64(), Le(2**70)),
    ([1], pl.UInt64(), Lt(-(2**70))),
    ([1, 2], pl.Int64(), Ge(1.5)),
    ([1, 2], pl.Int64(), Gt(0.5)),
    ([1.0, float("nan")], pl.Float64(), Ge(0)),
    ([1.0, float("nan")], pl.Float64(), Le(10)),
    ([1.0, float("nan")], pl.Float64(), Range(0, 10)),
    ([1.0, None], pl.Float32(), Lt(2)),
    ([datetime(2024, 1, 1)], pl.Datetime("us"), Ge(date(2024, 1, 1))),
    ([datetime(2024, 1, 1)], pl.Datetime("ns"), Gt(date(2024, 1, 1))),
    ([datetime(2024, 1, 1)], pl.Datetime("ms", "UTC"), Ge(datetime(2023, 1, 1))),
    ([date(2024, 1, 1)], pl.Date(), Ge(date(2024, 1, 2))),
    ([date(2024, 1, 1)], pl.Date(), Le(date(2024, 1, 2))),
    ([timedelta(0), timedelta(1)], pl.Duration("us"), Ge(timedelta(0))),
    ([time(13)], pl.Time(), Le(time(12))),
    ([Decimal("1.5")], pl.Decimal(10, 2), Range(Decimal(0), Decimal(1))),
    ([Decimal("0.5")], pl.Decimal(10, 2), Range(Decimal(0), Decimal(1))),
    ([1, 2], pl.Int64(), Ge(date(2024, 1, 1))),
    (["a", "b"], pl.Utf8(), Ge(1)),
    (["a", "b", None], pl.Utf8(), In(["a", "b"])),
    (["a", "c"], pl.Utf8(), In(["a", "b"])),
    (["a", "c"], pl.Utf8(), In(["a", None])),
    (["a", "b"], pl.Categorical(), In(["a", "b"])),
    (["a", "b"], pl.Categorical(), In(["a"])),
    (["a", "b"], pl.Enum(["a", "b", "c"]), In(["a", "c"])),
    ([1, 2], pl.Int64(), In([1, 2])),
    ([1, 3], pl.Int32(), In([1, 2])),
    # Int16: Annotated caches by equality, and In([1.0, 2.0]) == In([1, 2])
    ([1, 2], pl.Int16(), In([1.0, 2.0])),
    ([1, 2], pl.Int64(), In([1, "a"])),
    (["ab", "abc", None], pl.Utf8(), MinLen(2)),
    (["ab", "abc"], pl.Utf8(), MinLen(3)),
    (["日本語", "ab"], pl.Utf8(), MaxLen(3)),
    (["日本語x", "ab"], pl.Utf8(), MaxLen(3)),
    (["abc", None], pl.Utf8(), Regex(r"[a-z]+")),
    (["abc", "ab1"], pl.Utf8(), Regex(r"[a-z]+")),
    (["abc"], pl.Utf8(), Regex(r"a)|(b")),
    (["abc"], pl.Utf8(), Regex(r"(")),
    ([1, 2, None], pl.Int64(), Unique()),
    ([1, None, None], pl.Int64(), Unique()),
    ([1, 1], pl.Int64(), Unique()),
    ([1.0, float("nan"), float("nan")], pl.Float64(), Unique()),
    ([True, False], pl.Boolean(), Unique()),
    (["a", "a"], pl.Categorical(), Unique()),
    ([{"x": 1}, {"x": 1}], pl.Struct({"x": pl.Int64()}), Unique()),
    ([{"x": 1}, None], pl.Struct({"x": pl.Int64()}), Unique()),
    ([[1], [1]], pl.List(pl.Int64()), Unique()),
    ([], pl.Int64(), Range(0, 1)),
    ([], pl.Utf8(), Regex("a")),
    ([], pl.Int64(), Unique()),
    ([None, None], pl.Int64(), Range(0, 1)),
    ([None, None], pl.Int64(), Unique()),
    ([None, None], pl.Utf8(), MinLen(1)),
]


@pytest.mark.parametrize(("values", "dtype", "validator"), CASES)
def test_dtype_annotation_reports_what_the_validator_alone_does(
    values: list[Any], dtype: pl.DataType, validator: Any
) -> None:
    series = pl.Series("c", values, dtype=dtype)
    expected = _expected_error(series, validator)
    # The class, since Python 3.10 accepts only types in Annotated; it equals every instance
    schema = _schema(Annotated[type(dtype), validator])

    if expected is None:
        DataFrame[schema](series.to_frame())
    else:
        _assert_raises_like(expected, series.to_frame(), schema)


PYTHON_TYPE_CASES: list[tuple[list[Any], pl.DataType, Any, Any]] = [
    ([1, 2, None], pl.Int64(), int | None, Range(0, 2)),
    ([1, 5], pl.UInt16(), int, Range(0, 2)),
    ([1.0, float("nan")], pl.Float32(), float, Le(2)),
    (["a", "bb", None], pl.Utf8(), str | None, MaxLen(1)),
    ([datetime(2024, 1, 1)], pl.Datetime("ns"), datetime, Lt(date(2024, 1, 1))),
    ([timedelta(1)], pl.Duration("ms"), timedelta, Gt(timedelta(0))),
    ([None, None], pl.Null(), int | None, Range(0, 1)),
    ([1, 2], pl.Int64(), Any, Range(0, 1)),
    (["a", "a"], pl.Utf8(), Any, Unique()),
    ([[1], [1, 2]], pl.List(pl.Int64()), Any, MinLen(2)),
]


@pytest.mark.parametrize(("values", "dtype", "annotation", "validator"), PYTHON_TYPE_CASES)
def test_python_type_reports_what_the_validator_alone_does(
    values: list[Any], dtype: pl.DataType, annotation: Any, validator: Any
) -> None:
    series = pl.Series("c", values, dtype=dtype)
    if dtype == pl.Null:
        series = series.cast(pl.Int64)
    expected = _expected_error(series, validator)
    schema = _schema(Annotated[annotation, validator])

    frame = pl.Series("c", values, dtype=dtype).to_frame()
    if expected is None:
        DataFrame[schema](frame)
    else:
        _assert_raises_like(expected, frame, schema)


@pytest.mark.parametrize(
    ("values", "annotation", "message"),
    [
        (["a", "b", None], Literal["a", "b"] | None, None),
        ([1, 2], Literal[1, 2], None),
        (["a", "c"], Literal["a", "b"], r"expected one of \('a', 'b'\), got invalid values"),
        ([1, 3], Literal[1, 2], r"expected one of \(1, 2\), got invalid values"),
    ],
)
def test_literal(values: list[Any], annotation: Any, message: str | None) -> None:
    frame = pl.DataFrame({"c": values})
    schema = _schema(annotation)

    if message is None:
        DataFrame[schema](frame)
    else:
        with pytest.raises(ValidationError, match=message):
            DataFrame[schema](frame)


class Ordered(Protocol):
    a: Annotated[int, Range(0, 1), Unique()]
    b: Annotated[str, Regex("[a-z]+"), MaxLen(2)]
    c: Annotated[int, Custom(lambda v: v > 0, "must be positive")]
    d: NotRequiredColumn[Annotated[int, Ge(0)]]


def test_reports_the_first_column_that_fails() -> None:
    df = pl.DataFrame({"a": [0, 1], "b": ["1", "abc"], "c": [0, 1], "d": [-1, 0]})
    with pytest.raises(ValidationError, match="Column 'b': contains values that don't match"):
        DataFrame[Ordered](df)


def test_reports_the_first_validator_that_fails_in_a_column() -> None:
    df = pl.DataFrame({"a": [5, 5], "b": ["a", "b"], "c": [1, 1]})
    with pytest.raises(ValidationError, match=r"Column 'a': values must be in range \[0, 1\]"):
        DataFrame[Ordered](df)


def test_reports_a_failing_validator_before_a_later_missing_column() -> None:
    df = pl.DataFrame({"a": [0, 0], "c": [1, 1]})
    with pytest.raises(ValidationError, match="Column 'a': contains duplicate values"):
        DataFrame[Ordered](df)


def test_reports_a_failing_validator_before_a_later_wrong_type() -> None:
    df = pl.DataFrame({"a": [0, 1], "b": ["abc", "a"], "c": ["x", "y"]})
    with pytest.raises(ValidationError, match="Column 'b': contains strings longer"):
        DataFrame[Ordered](df)


def test_runs_custom_after_earlier_columns_pass() -> None:
    df = pl.DataFrame({"a": [0, 1], "b": ["a", "b"], "c": [0, 1], "d": [-1, 0]})
    with pytest.raises(ValidationError, match="Column 'c': must be positive"):
        DataFrame[Ordered](df)


def test_checks_a_not_required_column_when_present() -> None:
    df = pl.DataFrame({"a": [0, 1], "b": ["a", "b"], "c": [1, 1], "d": [-1, 0]})
    with pytest.raises(ValidationError, match="Column 'd': values must be >= 0"):
        DataFrame[Ordered](df)


def test_reports_a_column_before_a_failing_bound_on_a_later_column() -> None:
    """A check the engine cannot evaluate must not hide an earlier column's violation."""

    class Schema(Protocol):
        a: Annotated[int, Range(0, 1)]
        b: Annotated[int, Ge(datetime(2024, 1, 1))]

    df = pl.DataFrame({"a": [5], "b": [1]})
    with pytest.raises(ValidationError, match=r"Column 'a': values must be in range"):
        DataFrame[Schema](df)


def test_lazyframe_collect_reports_the_first_column_that_fails() -> None:
    lf = pl.LazyFrame({"a": [0, 1], "b": ["1", "abc"], "c": [0, 1]})
    with pytest.raises(ValidationError, match="Column 'b': contains values that don't match"):
        LazyFrame[Ordered](lf).collect()


def test_passes_a_frame_with_every_check_satisfied() -> None:
    df = pl.DataFrame({"a": [0, 1], "b": ["a", "bc"], "c": [1, 2], "d": [0, 3]})
    assert DataFrame[Ordered](df).shape == (2, 4)


def test_reuses_a_schema_on_frames_of_other_dtypes() -> None:
    """Each set of column dtypes gets its own batched checks, not those of an earlier frame."""

    class Schema(Protocol):
        a: Annotated[Any, Range(0, 1)]

    DataFrame[Schema](pl.DataFrame({"a": [0, 1]}))
    with pytest.raises(ValidationError, match="Range requires a numeric or temporal column"):
        DataFrame[Schema](pl.DataFrame({"a": ["z"]}))
    with pytest.raises(ValidationError, match=r"values must be in range \[0, 1\]"):
        DataFrame[Schema](pl.DataFrame({"a": [0.5, 5.0]}))
    DataFrame[Schema](pl.DataFrame({"a": [0, 1]}, schema={"a": pl.UInt8}))


@pytest.mark.parametrize("name", ["^a$", "^zz$", "*"])
def test_leaves_columns_polars_reads_as_patterns_out_of_the_batch(name: str) -> None:
    """pl.col reads such a name as a regex or wildcard, so it would check other columns."""
    schema = type(
        "Schema", (), {"__annotations__": {name: Annotated[int, Ge(0)], "b": Annotated[int, Ge(0)]}}
    )
    df = pl.DataFrame({name: [-1], "a": [5], "b": [1]})
    plan = validation._batch_plan(
        df,
        schema,
        validation.dataframe_columns(schema),
        validation.get_dataframe_checks(schema),
    )
    assert validation._batch_passed(df, plan) == {("b", 0)}
