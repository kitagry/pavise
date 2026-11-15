"""Struct schemas whose annotations are strings, for tests of how they are read."""

from __future__ import annotations

import dataclasses
import typing

import typing_extensions


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
