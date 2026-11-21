# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Type checking cases for structural subtyping between schemas, verified by pyright and never
run.

A line expected to fail carries ``# pyright: ignore[<rule>]``; if pyright stops reporting
the error there, the ignore becomes unnecessary and pyright fails.
"""

from dataclasses import dataclass
from typing import Protocol, TypedDict, cast

from pavise.pandas import DataFrame


class Address(Protocol):
    city: str


class AddressWithZip(Protocol):
    city: str
    zip: str


class SameAddress(Protocol):
    city: str


class AddressDict(TypedDict):
    city: str


class SameAddressDict(TypedDict):
    city: str


@dataclass
class AddressData:
    city: str


@dataclass
class SameAddressData:
    city: str


class User(Protocol):
    id: int
    score: int
    address: Address
    history: list[Address]


class UserWithName(Protocol):
    id: int
    score: int
    address: Address
    history: list[Address]
    name: str


class HasAddress(Protocol):
    address: Address


class HasHistory(Protocol):
    history: list[Address]


class FloatScore(Protocol):
    score: float


class OptionalScore(Protocol):
    score: int | None


class ZipUser(Protocol):
    address: AddressWithZip


class SameShapeUser(Protocol):
    address: SameAddress


class ZipHistoryUser(Protocol):
    history: list[AddressWithZip]


class DictUser(Protocol):
    address: AddressDict


class SameDictUser(Protocol):
    address: SameAddressDict


class DictHistoryUser(Protocol):
    history: list[AddressDict]


class SameDictHistoryUser(Protocol):
    history: list[SameAddressDict]


class DataUser(Protocol):
    address: AddressData


class SameDataUser(Protocol):
    address: SameAddressData


def users() -> DataFrame[User]: ...
def users_with_name() -> DataFrame[UserWithName]: ...
def zip_users() -> DataFrame[ZipUser]: ...
def zip_history_users() -> DataFrame[ZipHistoryUser]: ...
def dict_users() -> DataFrame[DictUser]: ...
def dict_history_users() -> DataFrame[DictHistoryUser]: ...
def data_users() -> DataFrame[DataUser]: ...
def needs_user(df: DataFrame[User]) -> None: ...
def needs_float_score(df: DataFrame[FloatScore]) -> None: ...
def needs_optional_score(df: DataFrame[OptionalScore]) -> None: ...
def needs_same_shape(df: DataFrame[SameShapeUser]) -> None: ...
def needs_address(df: DataFrame[HasAddress]) -> None: ...
def needs_history(df: DataFrame[HasHistory]) -> None: ...
def needs_same_dict(df: DataFrame[SameDictUser]) -> None: ...
def needs_same_dict_history(df: DataFrame[SameDictHistoryUser]) -> None: ...
def needs_same_data(df: DataFrame[SameDataUser]) -> None: ...


# A schema with extra columns can be used where fewer columns are expected.
needs_user(users_with_name())
needs_address(users())

# Columns are mutable attributes, so their types must match exactly.
needs_float_score(users())  # pyright: ignore[reportArgumentType]
needs_optional_score(users())  # pyright: ignore[reportArgumentType]

# The same goes for struct columns: extra fields and another Protocol of the same shape are
# not accepted, nor is a list of structs with extra fields (list is invariant too).
needs_address(zip_users())  # pyright: ignore[reportArgumentType]
needs_same_shape(users())  # pyright: ignore[reportArgumentType]
needs_history(zip_history_users())  # pyright: ignore[reportArgumentType]

# Dataclasses are nominal: another dataclass with the same fields is not accepted.
needs_same_data(data_users())  # pyright: ignore[reportArgumentType]

# Two TypedDicts with the same keys count as the same type.
needs_same_dict(dict_users())
needs_same_dict_history(dict_history_users())

# cast tells the type checker that the columns are compatible.
needs_address(cast(DataFrame[HasAddress], zip_users()))
