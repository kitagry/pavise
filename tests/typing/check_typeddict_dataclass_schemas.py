# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Type checking cases for TypedDicts and dataclasses as DataFrame schemas, verified by
pyright and never run.

A line expected to fail carries ``# pyright: ignore[<rule>]``; if pyright stops reporting
the error there, the ignore becomes unnecessary and pyright fails.
"""

from dataclasses import dataclass
from typing import Literal, Protocol

from typing_extensions import NotRequired, TypedDict

from pavise.checks import Key, UniqueKeys
from pavise.pandas import DataFrame


class UserDict(TypedDict):
    id: int


class LabeledUserDict(TypedDict):
    id: int
    label: str


class MaybeIdDict(TypedDict):
    id: NotRequired[int]


class UserColumns(Protocol):
    id: int


@dataclass
class UserData:
    id: int


@dataclass
class OtherUserData:
    id: int


@dataclass
class LabeledUserData(UserData):
    label: str


@dataclass
class KeyedUserData(UniqueKeys[Key[Literal["id"]]]):
    id: int


def labeled_user_dicts() -> DataFrame[LabeledUserDict]: ...
def user_dicts() -> DataFrame[UserDict]: ...
def user_data() -> DataFrame[UserData]: ...
def other_user_data() -> DataFrame[OtherUserData]: ...
def labeled_user_data() -> DataFrame[LabeledUserData]: ...
def keyed_user_data() -> DataFrame[KeyedUserData]: ...
def needs_user_dicts(df: DataFrame[UserDict]) -> None: ...
def needs_maybe_id_dicts(df: DataFrame[MaybeIdDict]) -> None: ...
def needs_columns(df: DataFrame[UserColumns]) -> None: ...
def needs_user_data(df: DataFrame[UserData]) -> None: ...
def needs_keyed_columns(df: DataFrame[UniqueKeys[Key[Literal["id"]]]]) -> None: ...


# TypedDicts are structural: one with more keys can be used where fewer are expected.
needs_user_dicts(labeled_user_dicts())
# A required key does not match one that may be missing, which could be written as absent.
needs_maybe_id_dicts(user_dicts())  # pyright: ignore[reportArgumentType]
# A TypedDict has no attributes, so it does not match a Protocol of the same columns.
needs_columns(user_dicts())  # pyright: ignore[reportArgumentType]

# Dataclasses are nominal, but their fields are attributes that match a Protocol.
needs_columns(user_data())
needs_user_data(labeled_user_data())
needs_user_data(other_user_data())  # pyright: ignore[reportArgumentType]

# A dataclass can declare unique keys; a TypedDict cannot have such a base.
needs_columns(keyed_user_data())
needs_keyed_columns(keyed_user_data())


class KeyedUserDict(TypedDict, UniqueKeys[Key[Literal["id"]]]):  # pyright: ignore[reportGeneralTypeIssues]
    id: int
