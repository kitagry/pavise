from typing import Annotated, ForwardRef, Literal, Protocol

import pytest

from pavise._annotations import (
    Field,
    ListOf,
    Scalar,
    StructOf,
    format_field,
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
