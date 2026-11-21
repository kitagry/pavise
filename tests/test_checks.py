from typing import Literal, Protocol, TypeVar, get_type_hints, runtime_checkable

import pytest
from typing_extensions import TypeAliasType

from pavise.checks import (
    Key,
    UniqueCheck,
    UniqueKeys,
    dataframe_check,
    get_dataframe_checks,
    unique,
)
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


class UserColumns(Protocol):
    id: int
    email: str


def test_unique_keys_rejects_bare_literal():
    with pytest.raises(TypeError) as exc_info:

        class Schema(UserColumns, UniqueKeys[Literal["id"]], Protocol):  # pyright: ignore[reportInvalidTypeArguments]
            pass

    assert str(exc_info.value) == (
        "UniqueKeys: wrap each key in Key[...], e.g. UniqueKeys[Key[Literal['id']]], "
        "got typing.Literal['id'] in Schema"
    )


def test_unique_keys_rejects_key_without_literal_columns():
    with pytest.raises(TypeError) as exc_info:

        class Schema(UserColumns, UniqueKeys[Key[str]], Protocol):
            pass

    assert str(exc_info.value) == (
        "Key: list the key columns as a Literal, e.g. Key[Literal['user_id', 'date']], "
        "got <class 'str'> in Schema"
    )


def test_unique_keys_rejects_column_not_in_schema():
    with pytest.raises(ValueError) as exc_info:

        class Schema(UserColumns, UniqueKeys[Key[Literal["id", "mail"]]], Protocol):
            pass

    assert str(exc_info.value) == "UniqueKeys: column 'mail' is not in schema Schema"


def test_unique_keys_rejects_index():
    with pytest.raises(ValueError) as exc_info:

        class Schema(UniqueKeys[Key[Literal["__index__"]]], Protocol):
            __index__: int

    assert str(exc_info.value) == "UniqueKeys: column '__index__' is not in schema Schema"


def test_unique_keys_accepts_annotations_that_cannot_be_evaluated():
    class Schema(UniqueKeys[Key[Literal["user_id"]]], Protocol):
        user_id: "UndefinedType"  # noqa: F821  # pyright: ignore[reportUndefinedVariable]


class User(UserColumns, UniqueKeys[Key[Literal["id"]]], Protocol): ...


def test_unique_keys_rejects_adding_keys_to_a_schema_that_has_keys():
    with pytest.raises(TypeError) as exc_info:

        class StrictUser(User, UniqueKeys[Key[Literal["id"]] | Key[Literal["email"]]], Protocol):
            pass

    assert str(exc_info.value) == (
        "UniqueKeys: StrictUser gets different unique keys from more than one place "
        "(User, StrictUser itself), and type checkers keep only one of them. "
        "Protocols are structural, so StrictUser can be used where User is expected "
        "without inheriting from it: combine a Protocol of only the columns with the keys, "
        "e.g. class StrictUser(Columns, UniqueKeys[...], Protocol), "
        "or declare the columns in StrictUser itself"
    )


def test_unique_keys_rejects_combining_schemas_with_different_keys():
    class ContactableUser(UserColumns, UniqueKeys[Key[Literal["email"]]], Protocol): ...

    with pytest.raises(TypeError) as exc_info:

        class Both(User, ContactableUser, Protocol):  # pyright: ignore[reportGeneralTypeIssues]
            pass

    assert str(exc_info.value).startswith(
        "UniqueKeys: Both gets different unique keys from more than one place "
        "(User, ContactableUser)"
    )


def test_unique_keys_accepts_combining_schemas_with_the_same_keys():
    class SameKeys(UserColumns, UniqueKeys[Key[Literal["id"]]], Protocol): ...

    class Both(User, SameKeys, Protocol):
        pass


def test_unique_keys_does_not_make_annotations_columns():
    class Child(User, Protocol):
        name: str

    assert get_type_hints(Child) == {"id": int, "email": str, "name": str}


T_co = TypeVar("T_co", covariant=True)


def test_unique_keys_are_inherited_through_a_subscripted_generic_base():
    class GenericUser(UserColumns, UniqueKeys[Key[Literal["id"]]], Protocol[T_co]): ...

    class Child(GenericUser[int], Protocol): ...

    assert get_dataframe_checks(Child) == [UniqueCheck(columns=("id",))]


IdKey = Key[Literal["id"]]
EmailColumns = TypeAliasType("EmailColumns", Literal["email"])
EmailKey = TypeAliasType("EmailKey", Key[EmailColumns])
UserKeys = TypeAliasType("UserKeys", IdKey | EmailKey)


def test_unique_keys_accepts_type_aliases():
    class Schema(UserColumns, UniqueKeys[UserKeys], Protocol): ...

    assert get_dataframe_checks(Schema) == [
        UniqueCheck(columns=("id",)),
        UniqueCheck(columns=("email",)),
    ]


def test_unique_is_deprecated_in_favor_of_unique_keys():
    with pytest.warns(DeprecationWarning) as record:
        unique("user_id", "date")

    assert len(record) == 1
    assert str(record[0].message) == (
        "unique() is deprecated and will be removed in the next minor release. "
        "Declare the key in the schema type instead, e.g. "
        "class Schema(UniqueKeys[Key[Literal['user_id', 'date']]], Protocol)"
    )
    assert record[0].filename == __file__
