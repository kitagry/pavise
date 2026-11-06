from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Annotated, Any, Protocol

import pandas as pd
import pytest

from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame
from pavise.validators import Custom, Ge, Gt, In, Le, Lt, MaxLen, MinLen, Range, Regex, Unique


class AgeSchema(Protocol):
    age: Annotated[int, Range(0, 150)]


class UserIdSchema(Protocol):
    user_id: Annotated[int, Unique()]


class StatusSchema(Protocol):
    status: Annotated[str, In(["pending", "approved", "rejected"])]


class EmailSchema(Protocol):
    email: Annotated[str, Regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")]


class UsernameMinLenSchema(Protocol):
    username: Annotated[str, MinLen(3)]


class UsernameMaxLenSchema(Protocol):
    username: Annotated[str, MaxLen(20)]


def test_range_validator_accepts_values_within_range():
    """Range validator accepts values within the specified range"""
    df = pd.DataFrame({"age": [25, 30, 45, 100]})
    result = DataFrame[AgeSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_range_validator_rejects_values_below_minimum():
    """Range validator rejects values below the minimum"""
    df = pd.DataFrame({"age": [25, -5, 30]})
    with pytest.raises(ValidationError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_range_validator_rejects_values_above_maximum():
    """Range validator rejects values above the maximum"""
    df = pd.DataFrame({"age": [25, 200, 30]})
    with pytest.raises(ValidationError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_unique_validator_accepts_unique_values():
    """Unique validator accepts columns with all unique values"""
    df = pd.DataFrame({"user_id": [1, 2, 3, 4]})
    result = DataFrame[UserIdSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_unique_validator_rejects_duplicate_values():
    """Unique validator rejects columns with duplicate values"""
    df = pd.DataFrame({"user_id": [1, 2, 2, 3]})
    with pytest.raises(ValidationError, match="user_id.*duplicate"):
        DataFrame[UserIdSchema](df)


def test_in_validator_accepts_allowed_values():
    """In validator accepts values within the allowed set"""
    df = pd.DataFrame({"status": ["pending", "approved", "rejected", "pending"]})
    result = DataFrame[StatusSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_in_validator_rejects_disallowed_values():
    """In validator rejects values not in the allowed set"""
    df = pd.DataFrame({"status": ["pending", "invalid", "approved"]})
    with pytest.raises(ValidationError, match="status.*allowed values"):
        DataFrame[StatusSchema](df)


def test_regex_validator_accepts_matching_values():
    """Regex validator accepts values that match the pattern"""
    df = pd.DataFrame({"email": ["user@example.com", "test@test.org", "admin@company.co.jp"]})
    result = DataFrame[EmailSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_regex_validator_rejects_non_matching_values():
    """Regex validator rejects values that don't match the pattern"""
    df = pd.DataFrame({"email": ["user@example.com", "invalid-email", "test@test.org"]})
    with pytest.raises(ValidationError, match="email.*pattern"):
        DataFrame[EmailSchema](df)


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"[a-z]+", "abc123"),
        (r"[a-z]+", "123abc"),
        (r"a|b", "ab"),
    ],
)
def test_regex_validator_rejects_partial_match(pattern: str, value: str):
    """Regex validator requires the whole string to match the pattern"""

    class Schema(Protocol):
        v: Annotated[str, Regex(pattern)]

    df = pd.DataFrame({"v": [value]})
    with pytest.raises(ValidationError, match="v.*pattern"):
        DataFrame[Schema](df)


@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"[a-z]+", "abc"),
        (r"a|b", "b"),
        (r"^[a-z]+$", "abc"),
    ],
)
def test_regex_validator_accepts_full_match(pattern: str, value: str):
    """Regex validator accepts values whose whole string matches the pattern"""

    class Schema(Protocol):
        v: Annotated[str, Regex(pattern)]

    df = pd.DataFrame({"v": [value]})
    DataFrame[Schema](df)


def test_regex_validator_ignores_null_in_optional_column():
    """Regex validator skips nulls in Optional columns like the comparison validators do"""

    class Schema(Protocol):
        v: Annotated[str | None, Regex(r"[a-z]+")]

    df = pd.DataFrame({"v": ["abc", None]})
    DataFrame[Schema](df)


def test_minlen_validator_accepts_valid_length():
    """MinLen validator accepts strings with sufficient length"""
    df = pd.DataFrame({"username": ["alice", "bob123", "charlie"]})
    result = DataFrame[UsernameMinLenSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_minlen_validator_rejects_short_strings():
    """MinLen validator rejects strings that are too short"""
    df = pd.DataFrame({"username": ["alice", "ab", "charlie"]})
    with pytest.raises(ValidationError, match="username.*length"):
        DataFrame[UsernameMinLenSchema](df)


def test_maxlen_validator_accepts_valid_length():
    """MaxLen validator accepts strings within length limit"""
    df = pd.DataFrame({"username": ["alice", "bob", "verylongusername123"]})
    result = DataFrame[UsernameMaxLenSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_maxlen_validator_rejects_long_strings():
    """MaxLen validator rejects strings that are too long"""
    df = pd.DataFrame({"username": ["alice", "thisusernameiswaytoolongtobevalid", "bob"]})
    with pytest.raises(ValidationError, match="username.*length"):
        DataFrame[UsernameMaxLenSchema](df)


def is_positive(value: Any) -> bool:
    """Custom validator function for testing."""
    return value > 0


class CustomValidatorSchema(Protocol):
    value: Annotated[int, Custom(is_positive, "must be positive")]


def test_custom_validator_accepts_valid_values():
    """Custom validator accepts values that pass the validation function"""
    df = pd.DataFrame({"value": [1, 2, 3, 100]})
    result = DataFrame[CustomValidatorSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_custom_validator_rejects_invalid_values():
    """Custom validator rejects values that fail the validation function"""
    df = pd.DataFrame({"value": [1, -5, 3]})
    with pytest.raises(ValidationError, match="value.*must be positive"):
        DataFrame[CustomValidatorSchema](df)


class GeSchema(Protocol):
    score: Annotated[int, Ge(0)]


class GtSchema(Protocol):
    score: Annotated[int, Gt(0)]


class LeSchema(Protocol):
    score: Annotated[int, Le(100)]


class LtSchema(Protocol):
    score: Annotated[int, Lt(100)]


class OptionalGeSchema(Protocol):
    score: Annotated[float | None, Ge(0)]


@pytest.mark.parametrize(
    "schema, values",
    [
        (GeSchema, [0, 1, 1000]),
        (GtSchema, [1, 1000]),
        (LeSchema, [-1000, 99, 100]),
        (LtSchema, [-1000, 99]),
    ],
)
def test_comparison_validator_accepts_values_satisfying_bound(
    schema: type[Any], values: list[int]
) -> None:
    """Comparison validators accept values that satisfy the bound, including inclusive edges"""
    df = pd.DataFrame({"score": values})
    result = DataFrame[schema](df)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "schema, values, expected_message",
    [
        (GeSchema, [5, -1], "score.*>= 0"),
        (GtSchema, [5, 0], "score.*> 0"),
        (LeSchema, [5, 101], "score.*<= 100"),
        (LtSchema, [5, 100], "score.*< 100"),
    ],
)
def test_comparison_validator_rejects_values_violating_bound(
    schema: type[Any], values: list[int], expected_message: str
) -> None:
    """Comparison validators reject values that violate the bound, including exclusive edges"""
    df = pd.DataFrame({"score": values})
    with pytest.raises(ValidationError, match=expected_message):
        DataFrame[schema](df)


def test_comparison_validator_ignores_null_in_optional_column():
    """Comparison validators skip nulls in Optional columns like Range does"""
    df = pd.DataFrame({"score": [1.5, None, 0.0]})
    result = DataFrame[OptionalGeSchema](df)
    assert isinstance(result, pd.DataFrame)


class StrictGe(Ge):
    pass


class SubclassGeSchema(Protocol):
    score: Annotated[int, StrictGe(0)]


def test_comparison_validator_subclass_is_applied():
    """Subclasses of comparison validators are dispatched like the base class"""
    df = pd.DataFrame({"score": [1, -1]})
    with pytest.raises(ValidationError, match="score.*>= 0"):
        DataFrame[SubclassGeSchema](df)


@pytest.mark.parametrize(
    "validator, expected_name",
    [
        (Range(0, 10), "Range"),
        (Ge(0), "Ge"),
        (Gt(0), "Gt"),
        (Le(0), "Le"),
        (Lt(0), "Lt"),
    ],
)
def test_numeric_validator_rejects_string_column(validator: Any, expected_name: str) -> None:
    """Numeric validators raise ValidationError on a non-numeric column instead of comparing"""

    class Schema(Protocol):
        v: Annotated[str, validator]

    df = pd.DataFrame({"v": ["a", "b"]})
    with pytest.raises(
        ValidationError, match=f"v.*{expected_name} requires a numeric or temporal column"
    ):
        DataFrame[Schema](df)


def test_numeric_validator_rejects_bool_column() -> None:
    """bool is not treated as numeric, so both backends reject it the same way"""

    class Schema(Protocol):
        v: Annotated[bool, Ge(0)]

    df = pd.DataFrame({"v": [True, False]})
    with pytest.raises(ValidationError, match="v.*Ge requires a numeric or temporal column"):
        DataFrame[Schema](df)


def test_numeric_validator_error_sets_column_name() -> None:
    """The dtype rejection is attributed to the column like other validation errors"""

    class Schema(Protocol):
        v: Annotated[str, Ge(0)]

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pd.DataFrame({"v": ["a"]}))
    assert exc_info.value.column_name == "v"


def test_bound_validator_accepts_datetime_column() -> None:
    """Datetime columns are orderable, so datetime bounds keep working"""

    class Schema(Protocol):
        v: Annotated[datetime, Ge(datetime(2020, 1, 1))]

    DataFrame[Schema](pd.DataFrame({"v": [datetime(2021, 1, 1)]}))
    with pytest.raises(ValidationError, match="v.*>= 2020-01-01"):
        DataFrame[Schema](pd.DataFrame({"v": [datetime(2019, 1, 1)]}))


DATE_BOUND_CASES = [
    pytest.param(
        Range(date(2020, 1, 1), date(2020, 12, 31)), date(2020, 6, 1), date(2021, 1, 1), id="Range"
    ),
    pytest.param(Ge(date(2020, 1, 1)), date(2020, 1, 1), date(2019, 12, 31), id="Ge"),
    pytest.param(Gt(date(2020, 1, 1)), date(2020, 1, 2), date(2020, 1, 1), id="Gt"),
    pytest.param(Le(date(2020, 1, 1)), date(2020, 1, 1), date(2020, 1, 2), id="Le"),
    pytest.param(Lt(date(2020, 1, 1)), date(2019, 12, 31), date(2020, 1, 1), id="Lt"),
]


@pytest.mark.parametrize("validator, valid, invalid", DATE_BOUND_CASES)
@pytest.mark.parametrize(
    "to_series",
    [
        pytest.param(lambda d: pd.Series([d], dtype=object), id="object"),
        # date columns are coerced to datetime64[ns], so a date bound must compare with Timestamps
        pytest.param(lambda d: pd.Series(pd.to_datetime([d])), id="datetime64"),
    ],
)
def test_bound_validator_accepts_date_bound_on_date_column(
    validator: Any, valid: date, invalid: date, to_series: Callable[[date], pd.Series]
) -> None:
    """date bounds apply to date columns whichever storage pandas uses for them"""

    class Schema(Protocol):
        v: Annotated[date, validator]

    DataFrame[Schema](pd.DataFrame({"v": to_series(valid)}))
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pd.DataFrame({"v": to_series(invalid)}))
    assert exc_info.value.column_name == "v"


def test_incomparable_date_bound_reports_the_bound_as_written() -> None:
    """The error names the user's date, not the Timestamp it was converted to"""

    class Schema(Protocol):
        v: Annotated[datetime, Ge(date(2020, 1, 1))]

    df = pd.DataFrame({"v": pd.to_datetime(["2021-01-01"]).tz_localize("UTC")})
    with pytest.raises(ValidationError, match=r"datetime\.date\(2020, 1, 1\) cannot be compared"):
        DataFrame[Schema](df)


def test_range_accepts_datetime_bounds() -> None:
    """Range takes datetime bounds the same way as the one-sided validators"""

    class Schema(Protocol):
        v: Annotated[datetime, Range(datetime(2020, 1, 1), datetime(2020, 12, 31))]

    DataFrame[Schema](pd.DataFrame({"v": [datetime(2020, 6, 1)]}))
    with pytest.raises(ValidationError, match="v.*in range \\[2020-01-01 00:00:00, 2020-12-31"):
        DataFrame[Schema](pd.DataFrame({"v": [datetime(2021, 1, 1)]}))


def test_bound_validator_accepts_decimal_bounds() -> None:
    """Decimal bounds compare exactly against Decimal values"""

    class Schema(Protocol):
        v: Annotated[Any, Range(Decimal("0.1"), Decimal("0.3"))]

    DataFrame[Schema](pd.DataFrame({"v": [Decimal("0.1"), Decimal("0.3")]}))
    with pytest.raises(ValidationError, match="v.*in range \\[0.1, 0.3\\]"):
        DataFrame[Schema](pd.DataFrame({"v": [Decimal("0.30001")]}))


def test_bound_validator_accepts_timedelta_bound() -> None:
    """timedelta bounds apply to timedelta columns"""

    class Schema(Protocol):
        v: Annotated[timedelta, Lt(timedelta(hours=1))]

    DataFrame[Schema](pd.DataFrame({"v": [timedelta(minutes=59)]}))
    with pytest.raises(ValidationError, match="v.*< 1:00:00"):
        DataFrame[Schema](pd.DataFrame({"v": [timedelta(hours=1)]}))


def test_bound_validator_accepts_time_bound() -> None:
    """time bounds apply to time-of-day values"""

    class Schema(Protocol):
        v: Annotated[Any, Lt(time(12))]

    DataFrame[Schema](pd.DataFrame({"v": [time(11, 59)]}))
    with pytest.raises(ValidationError, match="v.*< 12:00:00"):
        DataFrame[Schema](pd.DataFrame({"v": [time(12)]}))


def test_bound_validator_accepts_object_decimal_column() -> None:
    """Decimal values live in object dtype in pandas but are numeric, matching polars Decimal"""

    class Schema(Protocol):
        v: Annotated[Any, Ge(0)]

    DataFrame[Schema](pd.DataFrame({"v": [Decimal("1.5")]}))
    with pytest.raises(ValidationError, match="v.*>= 0"):
        DataFrame[Schema](pd.DataFrame({"v": [Decimal("-1.5")]}))


ARROW_TEMPORAL_BOUND_CASES = [
    pytest.param("date32", Ge(date(2020, 1, 1)), date(2020, 1, 1), date(2019, 12, 31), id="date32"),
    pytest.param("date64", Lt(date(2020, 1, 1)), date(2019, 12, 31), date(2020, 1, 1), id="date64"),
    pytest.param(
        "timestamp",
        Range(datetime(2020, 1, 1), datetime(2020, 12, 31)),
        datetime(2020, 6, 1),
        datetime(2021, 1, 1),
        id="timestamp",
    ),
    pytest.param(
        "timestamp",
        Ge(date(2020, 1, 1)),
        datetime(2020, 1, 1),
        datetime(2019, 12, 31, 23),
        id="timestamp-date-bound",
    ),
    pytest.param("time64", Le(time(12)), time(12), time(12, 1), id="time64"),
    pytest.param("duration", Gt(timedelta(0)), timedelta(seconds=1), timedelta(0), id="duration"),
]


@pytest.mark.parametrize("arrow_type, validator, valid, invalid", ARROW_TEMPORAL_BOUND_CASES)
def test_bound_validator_accepts_pyarrow_temporal_column(
    arrow_type: str, validator: Any, valid: Any, invalid: Any
) -> None:
    """pyarrow-backed temporal columns are orderable, matching polars temporal dtypes"""
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype requires pandas 1.5+")
    dtype = pd.ArrowDtype(
        {
            "date32": pa.date32(),
            "date64": pa.date64(),
            "timestamp": pa.timestamp("us"),
            "time64": pa.time64("us"),
            "duration": pa.duration("us"),
        }[arrow_type]
    )

    class Schema(Protocol):
        v: Annotated[Any | None, validator]

    DataFrame[Schema](pd.DataFrame({"v": pd.array([valid, None], dtype=dtype)}))
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pd.DataFrame({"v": pd.array([invalid], dtype=dtype)}))
    assert exc_info.value.column_name == "v"
    assert "requires a numeric or temporal column" not in str(exc_info.value)


def test_incomparable_bound_on_pyarrow_timestamp_column() -> None:
    """A tz-aware bound on a naive pyarrow timestamp column is a ValidationError, not TypeError"""
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype requires pandas 1.5+")

    class Schema(Protocol):
        v: Annotated[Any, Ge(datetime(2020, 1, 1, tzinfo=timezone.utc))]

    df = pd.DataFrame(
        {"v": pd.array([datetime(2021, 1, 1)], dtype=pd.ArrowDtype(pa.timestamp("us")))}
    )
    with pytest.raises(ValidationError, match="v.*Ge bound .* cannot be compared"):
        DataFrame[Schema](df)


def test_bound_validator_rejects_pyarrow_non_temporal_column() -> None:
    """Only temporal pyarrow types are let through; strings are still rejected"""
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype requires pandas 1.5+")

    class Schema(Protocol):
        v: Annotated[Any, Ge(0)]

    df = pd.DataFrame({"v": pd.array(["a"], dtype=pd.ArrowDtype(pa.string()))})
    with pytest.raises(ValidationError, match="v.*Ge requires a numeric or temporal column"):
        DataFrame[Schema](df)


def test_bound_validator_rejects_incomparable_bound() -> None:
    """A bound of the wrong kind raises ValidationError instead of a bare TypeError"""

    class Schema(Protocol):
        v: Annotated[datetime, Ge(0)]

    with pytest.raises(ValidationError, match="v.*Ge bound 0 cannot be compared"):
        DataFrame[Schema](pd.DataFrame({"v": [datetime(2021, 1, 1)]}))


STRING_DTYPES = [
    pytest.param("object", id="object"),
    pytest.param("string[python]", id="string-python"),
    pytest.param("string[pyarrow]", id="string-pyarrow"),
    pytest.param("large_string[pyarrow]", id="arrow-dtype"),
]


def _string_series(values: list[str], dtype: str) -> pd.Series:
    if "pyarrow" in dtype:
        pytest.importorskip("pyarrow")
    return pd.Series(values, dtype=dtype)


@pytest.mark.parametrize("dtype", STRING_DTYPES)
@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"[a-z]+", "abc123"),
        (r"a|b", "ab"),
        (r"^a|b$", "ab"),
        (r"(?m)abc", "x\nabc"),
    ],
)
def test_regex_validator_rejects_partial_match_for_every_string_dtype(
    pattern: str, value: str, dtype: str
):
    """Full-match semantics must not depend on how the string column is stored"""

    class Schema(Protocol):
        v: Annotated[str, Regex(pattern)]

    df = pd.DataFrame({"v": _string_series([value], dtype)})
    with pytest.raises(ValidationError, match="v.*pattern"):
        DataFrame[Schema](df)


@pytest.mark.parametrize("dtype", STRING_DTYPES)
@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"[a-z]+", "abc"),
        (r"^a|b$", "b"),
        (r"(?i)abc", "ABC"),
    ],
)
def test_regex_validator_accepts_full_match_for_every_string_dtype(
    pattern: str, value: str, dtype: str
):
    """Full matches, including leading inline flags, pass on every string dtype"""

    class Schema(Protocol):
        v: Annotated[str, Regex(pattern)]

    df = pd.DataFrame({"v": _string_series([value], dtype)})
    DataFrame[Schema](df)


@pytest.mark.parametrize("dtype", STRING_DTYPES)
def test_regex_validator_reports_invalid_pattern_as_validation_error(dtype: str):
    """A pattern the engine cannot compile raises ValidationError, not a raw backend error"""

    class Schema(Protocol):
        v: Annotated[str, Regex("(")]

    df = pd.DataFrame({"v": _string_series(["abc"], dtype)})
    with pytest.raises(ValidationError, match=r"v.*invalid regex pattern '\('") as exc_info:
        DataFrame[Schema](df)
    assert exc_info.value.column_name == "v"


@pytest.mark.parametrize(
    "dtype",
    [d for d in STRING_DTYPES if "pyarrow" in str(d.values[0])],
)
def test_regex_validator_reports_re2_unsupported_syntax_as_validation_error(dtype: str):
    """RE2 has no verbose mode; pyarrow's ArrowInvalid surfaces as ValidationError"""

    class Schema(Protocol):
        v: Annotated[str, Regex(r"(?x)a")]

    df = pd.DataFrame({"v": _string_series(["a"], dtype)})
    with pytest.raises(ValidationError, match=r"v.*invalid regex pattern"):
        DataFrame[Schema](df)


@pytest.mark.parametrize("dtype", STRING_DTYPES)
def test_regex_validator_rejects_pattern_that_only_compiles_once_wrapped(dtype: str):
    """`a)|(b` is invalid, but wrapped as `^(?:a)|(b)$` it would compile and match prefixes"""

    class Schema(Protocol):
        v: Annotated[str, Regex("a)|(b")]

    df = pd.DataFrame({"v": _string_series(["axyz"], dtype)})
    with pytest.raises(ValidationError, match=r"v.*invalid regex pattern"):
        DataFrame[Schema](df)


@pytest.mark.parametrize(
    "dtype",
    [d for d in STRING_DTYPES if "pyarrow" in str(d.values[0])],
)
def test_regex_validator_reports_re2_unsupported_syntax_on_empty_column(dtype: str):
    """The pattern error must not depend on whether the column has data yet"""

    # pandas 3 falls back to Python re for some RE2-unsupported syntax (e.g. lookbehind),
    # but not for verbose mode, so (?x) fails on every supported pandas version
    class Schema(Protocol):
        v: Annotated[str, Regex(r"(?x)a")]

    df = pd.DataFrame({"v": _string_series([], dtype)})
    with pytest.raises(ValidationError, match=r"v.*invalid regex pattern"):
        DataFrame[Schema](df)


# RE2, used for pyarrow-backed columns, has no verbose mode at all
@pytest.mark.parametrize("dtype", ["object", "string[python]"])
def test_regex_validator_accepts_verbose_pattern_ending_in_comment(dtype: str):
    """Verbose patterns ending in a comment behave as on polars (these paths use re.fullmatch)"""

    class Schema(Protocol):
        v: Annotated[str, Regex(r"(?x)[a-z]+  # letters")]

    DataFrame[Schema](pd.DataFrame({"v": _string_series(["abc"], dtype)}))
    with pytest.raises(ValidationError, match="v.*pattern"):
        DataFrame[Schema](pd.DataFrame({"v": _string_series(["abc1"], dtype)}))
