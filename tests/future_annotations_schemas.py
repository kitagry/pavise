"""Schemas whose annotations are strings, for tests of how they are read."""

from __future__ import annotations

import dataclasses
import typing

import typing_extensions

from pavise.checks import Key, UniqueKeys
from pavise.types import NotRequiredColumn


class AddressDict(typing.TypedDict):
    city: str
    zip: typing_extensions.NotRequired[str]


class PartialAddressDict(typing_extensions.TypedDict, total=False):
    city: typing_extensions.Required[str]
    zip: str


@dataclasses.dataclass
class AddressData:
    city: str
    zip: str = ""
    kind: typing.ClassVar[str] = "address"
    check: dataclasses.InitVar[bool] = False


@dataclasses.dataclass
class LabeledData:
    label: typing.Literal["home", "work"]


class UserDict(typing.TypedDict):
    id: int
    email: typing_extensions.NotRequired[str]


@dataclasses.dataclass
class UserData:
    id: int
    email: NotRequiredColumn[str]
    kind: typing.ClassVar[str] = "user"
    check: dataclasses.InitVar[bool] = False


@dataclasses.dataclass
class KindKeyedUserData(UniqueKeys[Key[typing.Literal["kind"]]]):
    id: int
    kind: typing.ClassVar[str] = "user"
