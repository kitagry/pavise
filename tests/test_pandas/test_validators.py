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
