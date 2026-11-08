from typing import Annotated, Literal, Protocol

import pytest

from pavise._annotations import Field, ListOf, Scalar, StructOf, format_field, parse_annotation
from pavise.exceptions import ValidationError
from pavise.types import NotRequiredColumn
from pavise.validators import MinLen, Range, Regex


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
