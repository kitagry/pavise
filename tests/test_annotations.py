from dataclasses import InitVar, dataclass, field
from typing import (
    Annotated,
    ClassVar,
    ForwardRef,
    Generic,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
)

import future_annotations_schemas
import pytest
import typing_extensions
from typing_extensions import NotRequired, ReadOnly, Required

from pavise._annotations import (
    Field,
    ListOf,
    Scalar,
    StructOf,
    format_field,
    is_nested_type,
    parse_annotation,
    parse_nested_column,
)
from pavise.exceptions import ValidationError
from pavise.types import NotRequiredColumn
from pavise.validators import Custom, In, MaxLen, MinLen, Range, Regex, Unique


class Address(Protocol):
    city: str
    zip: Annotated[str, Regex(r"^\d{3}-\d{4}$")]


class Node(Protocol):
    children: list["Node"]


def test_parse_scalar():
    assert parse_annotation(int, "a") == Field(Scalar(int))


def test_parse_list_of_scalar():
    assert parse_annotation(list[str], "a") == Field(ListOf(Field(Scalar(str))))


def test_parse_validators_on_list_elements():
    field = parse_annotation(list[Annotated[int, Range(0, 100)]], "a")
    assert field == Field(ListOf(Field(Scalar(int), validators=(Range(0, 100),))))


def test_parse_validators_on_list_column():
    field = parse_annotation(Annotated[list[int], MinLen(1)], "a")
    assert field == Field(ListOf(Field(Scalar(int))), validators=(MinLen(1),))


def test_parse_protocol_as_struct():
    field = parse_annotation(Address, "a")
    assert field == Field(
        StructOf(
            Address,
            {
                "city": Field(Scalar(str)),
                "zip": Field(Scalar(str), validators=(Regex(r"^\d{3}-\d{4}$"),)),
            },
        )
    )


def test_parse_list_of_struct():
    field = parse_annotation(list[Address], "a")
    assert isinstance(field.node, ListOf)
    assert isinstance(field.node.element.node, StructOf)


def test_parse_optional_list_of_optional_elements():
    field = parse_annotation(list[str | None] | None, "a")
    assert field == Field(ListOf(Field(Scalar(str), nullable=True)), nullable=True)


def test_parse_optional_outside_annotated_element():
    """Annotated[T, ...] | None and Annotated[T | None, ...] give the same element"""
    outside = parse_annotation(list[Annotated[int, Range(0, 1)] | None], "a")
    inside = parse_annotation(list[Annotated[int | None, Range(0, 1)]], "a")
    assert outside == inside


def test_parse_not_required_column():
    field = parse_annotation(NotRequiredColumn[list[int]], "a")
    assert field == Field(ListOf(Field(Scalar(int))), not_required=True)


def test_parse_not_required_struct_field():
    class WithOptionalField(Protocol):
        zip: NotRequiredColumn[str]

    field = parse_annotation(WithOptionalField, "a")
    assert isinstance(field.node, StructOf)
    assert field.node.fields["zip"] == Field(Scalar(str), not_required=True)


def test_parse_literal_element_is_scalar():
    assert parse_annotation(list[Literal["a", "b"]], "a") == Field(
        ListOf(Field(Scalar(Literal["a", "b"])))
    )


@pytest.mark.parametrize(
    "annotation, message",
    [
        (list, "list needs an element type"),
        (list[int | str], "union element types are not supported"),
        (list[NotRequiredColumn[int]], "NotRequiredColumn is only allowed"),
        (Node, "recursive schema is not supported: Node"),
    ],
)
def test_parse_rejects_unsupported_annotations(annotation: object, message: str):
    with pytest.raises(ValidationError, match=f"Column 'col': .*{message}"):
        parse_annotation(annotation, "col")


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (list[str], "list[str]"),
        (list[str | None], "list[str | None]"),
        (list[list[int]], "list[list[int]]"),
        (Address, "Address"),
        (list[Address], "list[Address]"),
        (list[Literal["a"]], "list[Literal['a']]"),
    ],
)
def test_format_field(annotation: object, expected: str):
    assert format_field(parse_annotation(annotation, "a")) == expected


class Holder(Protocol):
    history: list["Address"]


def test_parse_resolves_forward_reference_inside_list():
    """Python 3.10's get_type_hints leaves strings inside list[...] unresolved"""
    field = parse_annotation(list["Address"], "history", owner=Holder)
    assert isinstance(field.node, ListOf)
    assert isinstance(field.node.element.node, StructOf)
    assert field.node.element.node.schema is Address


def test_parse_rejects_unresolvable_forward_reference():
    with pytest.raises(
        ValidationError, match="Column 'a': cannot resolve forward reference 'Nope'"
    ):
        parse_annotation(list[ForwardRef("Nope")], "a", owner=Holder)


class UniqueCity(Protocol):
    city: Annotated[str, Unique()]


@pytest.mark.parametrize(
    "annotation",
    [
        Annotated[list[int], MinLen(1), MaxLen(3), Custom(bool, "must not be empty")],
        list[Annotated[int, Range(0, 100)]],
        list[Annotated[str, Regex("a+"), In(["a"]), MinLen(1)]],
        Annotated[Address, Custom(bool, "must not be empty")],
        list[Annotated[list[int], MinLen(1)]],
    ],
)
def test_parse_nested_column_accepts_validators_on_their_targets(annotation: object):
    parse_nested_column(annotation, "col", owner=Holder)


@pytest.mark.parametrize(
    "annotation, message",
    [
        (
            Annotated[list[int], Range(0, 1)],
            r"^Column 'col': Range cannot be used on a list; use MinLen, MaxLen or Custom$",
        ),
        (
            list[Annotated[list[int], In([[1]])]],
            r"^Column 'col\[\]': In cannot be used on a list; use MinLen, MaxLen or Custom$",
        ),
        (
            Annotated[Address, MinLen(1)],
            r"^Column 'col': MinLen cannot be used on a struct; use Custom$",
        ),
        (
            list[Annotated[int, Unique()]],
            r"^Column 'col\[\]': Unique cannot be used inside list and struct columns$",
        ),
        (
            UniqueCity,
            r"^Column 'col.city': Unique cannot be used inside list and struct columns$",
        ),
        (
            Annotated[list[int], Unique()],
            r"^Column 'col': Unique cannot be used on a list; use MinLen, MaxLen or Custom$",
        ),
    ],
)
def test_parse_nested_column_rejects_validators_without_meaning(annotation: object, message: str):
    with pytest.raises(ValidationError, match=message):
        parse_nested_column(annotation, "col", owner=Holder)


class AddressDict(TypedDict):
    city: str
    zip: NotRequired[Annotated[str, Regex(r"^\d{3}-\d{4}$")]]


class PartialAddressDict(TypedDict, total=False):
    city: Required[str]
    zip: str


class LabeledAddressDict(PartialAddressDict):
    label: str


class AnnotatedOutsideDict(TypedDict):
    zip: Annotated[NotRequired[str], Regex("a")]


class ReadOnlyDict(TypedDict):
    city: ReadOnly[str]
    zip: ReadOnly[NotRequired[str]]


def test_parse_typeddict_as_struct():
    field = parse_annotation(AddressDict, "a")
    assert field == Field(
        StructOf(
            AddressDict,
            {
                "city": Field(Scalar(str)),
                "zip": Field(Scalar(str), validators=(Regex(r"^\d{3}-\d{4}$"),), not_required=True),
            },
        )
    )


@pytest.mark.parametrize(
    "schema, expected",
    [
        (PartialAddressDict, {"city": False, "zip": True}),
        (LabeledAddressDict, {"city": False, "zip": True, "label": False}),
        (ReadOnlyDict, {"city": False, "zip": True}),
    ],
)
def test_parse_typeddict_not_required_keys(schema: type, expected: dict[str, bool]):
    field = parse_annotation(schema, "a")
    assert isinstance(field.node, StructOf)
    assert {name: sub.not_required for name, sub in field.node.fields.items()} == expected
    assert all(sub.node == Scalar(str) for name, sub in field.node.fields.items())


def test_parse_typeddict_validators_outside_not_required():
    field = parse_annotation(AnnotatedOutsideDict, "a")
    assert isinstance(field.node, StructOf)
    assert field.node.fields["zip"] == Field(
        Scalar(str), validators=(Regex("a"),), not_required=True
    )


def test_parse_rejects_not_required_column_in_typeddict():
    class WithNotRequiredColumn(TypedDict):
        zip: NotRequiredColumn[str]

    with pytest.raises(
        ValidationError,
        match=r"^Column 'col': NotRequiredColumn cannot be used in a TypedDict; use NotRequired$",
    ):
        parse_annotation(WithNotRequiredColumn, "col")


@pytest.mark.parametrize(
    "schema",
    [future_annotations_schemas.AddressDict, future_annotations_schemas.PartialAddressDict],
)
def test_parse_typeddict_with_string_annotations(schema: type):
    field = parse_annotation(schema, "a")
    assert isinstance(field.node, StructOf)
    assert field.node.fields == {
        "city": Field(Scalar(str)),
        "zip": Field(Scalar(str), not_required=True),
    }


@dataclass
class AddressData:
    city: str
    label: NotRequiredColumn[str]
    zip: str | None = None
    tags: list[str] = field(default_factory=list)
    kind: ClassVar[str] = "address"
    check: InitVar[bool] = False


def test_parse_dataclass_as_struct():
    """Defaults are for building instances, so fields with defaults must still be present"""
    field = parse_annotation(AddressData, "a")
    assert field == Field(
        StructOf(
            AddressData,
            {
                "city": Field(Scalar(str)),
                "label": Field(Scalar(str), not_required=True),
                "zip": Field(Scalar(str), nullable=True),
                "tags": Field(ListOf(Field(Scalar(str)))),
            },
        )
    )


def test_parse_dataclass_with_string_annotations():
    field = parse_annotation(future_annotations_schemas.AddressData, "a")
    assert isinstance(field.node, StructOf)
    assert field.node.fields == {"city": Field(Scalar(str)), "zip": Field(Scalar(str))}


@dataclass
class CityLabeledData(future_annotations_schemas.LabeledData):
    city: str = ""


def test_parse_dataclass_resolves_inherited_string_annotations_where_written():
    field = parse_annotation(CityLabeledData, "a")
    assert isinstance(field.node, StructOf)
    assert field.node.fields == {
        "label": Field(Scalar(Literal["home", "work"])),
        "city": Field(Scalar(str)),
    }


@dataclass
class NotRequiredData:
    zip: NotRequired[str]  # pyright: ignore[reportInvalidTypeForm]


class ReadOnlyProtocol(Protocol):
    zip: ReadOnly[str]  # pyright: ignore[reportInvalidTypeForm]


@pytest.mark.parametrize(
    "schema, message",
    [
        (
            NotRequiredData,
            "NotRequired can only be used in a TypedDict; "
            "use NotRequiredColumn for a field that may be absent",
        ),
        (ReadOnlyProtocol, "ReadOnly can only be used in a TypedDict"),
    ],
)
def test_parse_rejects_typeddict_qualifiers_outside_typeddict(schema: type, message: str):
    with pytest.raises(ValidationError, match=f"^Column 'col': {message}$"):
        parse_annotation(schema, "col")


T = TypeVar("T")


class AddressTuple(NamedTuple):
    city: str


# Stands in for pydantic.BaseModel, which pavise does not depend on
PydanticBaseModel = type("BaseModel", (), {"__module__": "pydantic.main"})


class AddressModel(PydanticBaseModel):
    city: str


# typing.TypedDict cannot be generic on Python 3.10
class Box(typing_extensions.TypedDict, Generic[T]):
    value: T


@dataclass
class DataBox(Generic[T]):
    value: T


@pytest.mark.parametrize(
    "annotation, message",
    [
        (
            AddressTuple,
            r"unsupported type: AddressTuple "
            r"\(struct schemas must be a Protocol, TypedDict or dataclass\)",
        ),
        (
            AddressModel,
            r"unsupported type: AddressModel "
            r"\(struct schemas must be a Protocol, TypedDict or dataclass\)",
        ),
        (Box[int], r"unsupported type: Box\[int\] \(generic struct schemas are not supported\)"),
        (
            list[DataBox[int]],
            r"unsupported type: DataBox\[int\] \(generic struct schemas are not supported\)",
        ),
    ],
)
def test_parse_rejects_unsupported_struct_schemas(annotation: object, message: str):
    with pytest.raises(ValidationError, match=f"^Column 'col': {message}$"):
        parse_annotation(annotation, "col")


@pytest.mark.parametrize("annotation", [AddressTuple, AddressModel, Box[int]])
def test_unsupported_struct_schemas_are_nested_types(annotation: object):
    """So a column of one reports why instead of a generic unsupported type error"""
    assert is_nested_type(annotation)
