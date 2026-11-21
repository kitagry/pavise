"""TypedDicts and dataclasses as the schema of a whole DataFrame."""

from dataclasses import InitVar, dataclass
from typing import Annotated, Any, ClassVar, Literal, Protocol

import future_annotations_schemas
import pytest
from typing_extensions import NotRequired, Required, TypedDict

pytest.importorskip("polars")

import polars as pl

from pavise.checks import Check, Checked, Key, UniqueKeys, dataframe_check
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.types import NotRequiredColumn
from pavise.validators import Regex


class UserDict(TypedDict):
    id: int
    email: NotRequired[Annotated[str, Regex(r".+@.+")]]


class PartialUserDict(TypedDict, total=False):
    id: Required[int]
    email: Annotated[str, Regex(r".+@.+")]


@dataclass
class UserData:
    id: int
    email: NotRequiredColumn[Annotated[str, Regex(r".+@.+")]]
    kind: ClassVar[str] = "user"
    check: InitVar[bool] = False


USER_SCHEMAS = [
    UserDict,
    PartialUserDict,
    UserData,
    future_annotations_schemas.UserDict,
    future_annotations_schemas.UserData,
]


@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_columns_that_may_be_missing_can_be_absent(schema: type):
    DataFrame[schema](pl.DataFrame({"id": [1]}))


@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_columns_that_may_be_missing_are_checked_when_present(schema: type):
    DataFrame[schema](pl.DataFrame({"id": [1], "email": ["a@example.com"]}))
    with pytest.raises(ValidationError, match=r"^Column 'email': expected str, got Int64"):
        DataFrame[schema](pl.DataFrame({"id": [1], "email": [2]}))


@pytest.mark.parametrize("schema", [UserDict, PartialUserDict, UserData])
def test_validators_run_on_columns_that_may_be_missing(schema: type):
    with pytest.raises(
        ValidationError, match=r"^Column 'email': contains values that don't match the pattern"
    ):
        DataFrame[schema](pl.DataFrame({"id": [1], "email": ["example.com"]}))


@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_required_columns_must_be_present(schema: type):
    with pytest.raises(ValidationError, match=r"^Column 'id': missing$"):
        DataFrame[schema](pl.DataFrame({"email": ["a@example.com"]}))


@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_strict_accepts_only_schema_columns(schema: type):
    DataFrame[schema](pl.DataFrame({"id": [1], "email": ["a@example.com"]}), strict=True)
    with pytest.raises(ValidationError, match=r"unexpected columns \['check', 'kind'\]"):
        DataFrame[schema](pl.DataFrame({"id": [1], "kind": ["a"], "check": [True]}), strict=True)


@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_make_empty_builds_schema_columns(schema: type):
    assert DataFrame[schema].make_empty().schema == pl.Schema({"id": pl.Int64, "email": pl.String})
    assert LazyFrame[schema].make_empty().collect_schema() == pl.Schema(
        {"id": pl.Int64, "email": pl.String}
    )


@pytest.mark.parametrize("schema", USER_SCHEMAS)
def test_lazyframe_reads_columns_like_dataframe(schema: type):
    LazyFrame[schema](pl.LazyFrame({"id": [1]}))
    with pytest.raises(ValidationError, match=r"^Column 'id': missing$"):
        LazyFrame[schema](pl.LazyFrame({"email": ["a@example.com"]}))
    with pytest.raises(ValidationError, match=r"unexpected columns \['kind'\]"):
        LazyFrame[schema](pl.LazyFrame({"id": [1], "kind": ["a"]}), strict=True)


@dataclass
class EventData:
    id: int
    note: str | None = None


def test_dataclass_columns_with_defaults_must_be_present():
    """Defaults are for building instances, and rows built from them always have the field"""
    with pytest.raises(ValidationError, match=r"^Column 'note': missing$"):
        DataFrame[EventData](pl.DataFrame({"id": [1]}))


def test_not_required_column_in_typeddict_is_rejected():
    class WithNotRequiredColumn(TypedDict):
        id: int
        email: NotRequiredColumn[str]

    with pytest.raises(
        ValidationError,
        match=r"^Column 'email': NotRequiredColumn cannot be used in a TypedDict; use NotRequired$",
    ):
        DataFrame[WithNotRequiredColumn](pl.DataFrame({"id": [1]}))


def test_not_required_in_protocol_is_rejected():
    class WithNotRequired(Protocol):
        id: int
        email: NotRequired[str]  # pyright: ignore[reportInvalidTypeForm]

    with pytest.raises(
        ValidationError, match=r"^Column 'email': NotRequired can only be used in a TypedDict"
    ):
        DataFrame[WithNotRequired](pl.DataFrame({"id": [1], "email": ["a@example.com"]}))


@dataframe_check(lambda df: df["id"] > 0, name="positive_id")
class CheckedUserDict(TypedDict):
    id: int


def test_dataframe_check_runs_on_typeddict():
    with pytest.raises(ValidationError, match="positive_id"):
        DataFrame[CheckedUserDict](pl.DataFrame({"id": [-1]}))


@dataclass
class KeyedUserData(UniqueKeys[Key[Literal["id"]]]):
    id: int


def test_unique_keys_on_dataclass_are_checked():
    with pytest.raises(ValidationError, match=r"must be unique"):
        DataFrame[KeyedUserData](pl.DataFrame({"id": [1, 1]}))


class PositiveId(Check):
    def __call__(self, df: Any) -> Any:
        return df["id"] > 0


@dataclass
class CheckedUserData(Checked[PositiveId]):
    id: int


def test_checked_on_dataclass_is_checked():
    with pytest.raises(ValidationError, match="PositiveId"):
        DataFrame[CheckedUserData](pl.DataFrame({"id": [-1]}))
