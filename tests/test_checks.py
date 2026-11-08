from typing import Protocol, runtime_checkable

import pytest

from pavise.checks import dataframe_check, unique
from pavise.types import NotRequiredColumn


@runtime_checkable
@dataframe_check(lambda df: df["a"] > 0, name="a_positive")
class RuntimeCheckableSchema(Protocol):
    a: int


class HasA:
    a = 1


def test_dataframe_check_does_not_add_protocol_members():
    assert isinstance(HasA(), RuntimeCheckableSchema)


class UserSchema(Protocol):
    user_id: int
    date: str


def test_unique_rejects_column_not_in_schema():
    with pytest.raises(ValueError) as exc_info:
        unique("user_id", "dat")(UserSchema)

    assert str(exc_info.value) == "unique: column 'dat' is not in schema UserSchema"


def test_unique_accepts_columns_inherited_from_base_protocol():
    class EventSchema(UserSchema, Protocol):
        event: str

    unique("user_id", "event")(EventSchema)


def test_unique_accepts_not_required_columns():
    class Schema(Protocol):
        user_id: int
        date: NotRequiredColumn[str]

    unique("user_id", "date")(Schema)


def test_unique_accepts_annotations_that_cannot_be_evaluated():
    class Schema(Protocol):
        user_id: "UndefinedType"  # noqa: F821  # pyright: ignore[reportUndefinedVariable]

    unique("user_id")(Schema)


def test_unique_rejects_index():
    class Schema(Protocol):
        __index__: int
        user_id: int

    with pytest.raises(ValueError) as exc_info:
        unique("__index__", "user_id")(Schema)

    assert str(exc_info.value) == "unique: column '__index__' is not in schema Schema"


def test_unique_requires_at_least_one_column():
    with pytest.raises(ValueError) as exc_info:
        unique()(UserSchema)

    assert str(exc_info.value) == "unique: at least one column is required"


def test_unique_rejects_repeated_column():
    with pytest.raises(ValueError) as exc_info:
        unique("user_id", "user_id")(UserSchema)

    assert str(exc_info.value) == "unique: column 'user_id' is given more than once"
