from typing import Annotated, Protocol

import polars as pl
import pytest

from patrol.exceptions import ValidationError
from patrol.polars import DataFrame
from patrol.validators import In, MaxLen, MinLen, Range, Regex, Unique


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
    df = pl.DataFrame({"age": [25, 30, 45, 100]})
    result = DataFrame[AgeSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_range_validator_rejects_values_below_minimum():
    """Range validator rejects values below the minimum"""
    df = pl.DataFrame({"age": [25, -5, 30]})
    with pytest.raises(ValidationError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_range_validator_rejects_values_above_maximum():
    """Range validator rejects values above the maximum"""
    df = pl.DataFrame({"age": [25, 200, 30]})
    with pytest.raises(ValidationError, match="age.*range"):
        DataFrame[AgeSchema](df)


def test_unique_validator_accepts_unique_values():
    """Unique validator accepts columns with all unique values"""
    df = pl.DataFrame({"user_id": [1, 2, 3, 4]})
    result = DataFrame[UserIdSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_unique_validator_rejects_duplicate_values():
    """Unique validator rejects columns with duplicate values"""
    df = pl.DataFrame({"user_id": [1, 2, 2, 3]})
    with pytest.raises(ValidationError, match="user_id.*duplicate"):
        DataFrame[UserIdSchema](df)


def test_in_validator_accepts_allowed_values():
    """In validator accepts values within the allowed set"""
    df = pl.DataFrame({"status": ["pending", "approved", "rejected", "pending"]})
    result = DataFrame[StatusSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_in_validator_rejects_disallowed_values():
    """In validator rejects values not in the allowed set"""
    df = pl.DataFrame({"status": ["pending", "invalid", "approved"]})
    with pytest.raises(ValidationError, match="status.*allowed values"):
        DataFrame[StatusSchema](df)


def test_regex_validator_accepts_matching_values():
    """Regex validator accepts values that match the pattern"""
    df = pl.DataFrame({"email": ["user@example.com", "test@test.org", "admin@company.co.jp"]})
    result = DataFrame[EmailSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_regex_validator_rejects_non_matching_values():
    """Regex validator rejects values that don't match the pattern"""
    df = pl.DataFrame({"email": ["user@example.com", "invalid-email", "test@test.org"]})
    with pytest.raises(ValidationError, match="email.*pattern"):
        DataFrame[EmailSchema](df)


def test_minlen_validator_accepts_valid_length():
    """MinLen validator accepts strings with sufficient length"""
    df = pl.DataFrame({"username": ["alice", "bob123", "charlie"]})
    result = DataFrame[UsernameMinLenSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_minlen_validator_rejects_short_strings():
    """MinLen validator rejects strings that are too short"""
    df = pl.DataFrame({"username": ["alice", "ab", "charlie"]})
    with pytest.raises(ValidationError, match="username.*length"):
        DataFrame[UsernameMinLenSchema](df)


def test_maxlen_validator_accepts_valid_length():
    """MaxLen validator accepts strings within length limit"""
    df = pl.DataFrame({"username": ["alice", "bob", "verylongusername123"]})
    result = DataFrame[UsernameMaxLenSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_maxlen_validator_rejects_long_strings():
    """MaxLen validator rejects strings that are too long"""
    df = pl.DataFrame({"username": ["alice", "thisusernameiswaytoolongtobevalid", "bob"]})
    with pytest.raises(ValidationError, match="username.*length"):
        DataFrame[UsernameMaxLenSchema](df)
