"""Backend-independent parsing of schema annotations.

Column annotations are unwrapped one level at a time (NotRequiredColumn, Annotated, Optional).
List and Struct columns (``list[X]`` and nested Protocols) are parsed into a tree of
``Field`` nodes, which each backend compares with its nested dtypes.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from typing_extensions import is_protocol

from pavise.exceptions import ValidationError
from pavise.types import NotRequiredColumn


def _extract_type_and_validators(
    annotation: type,
) -> tuple[type | tuple[type, ...], list[Any], bool, bool]:
    """
    Extract base type, validators, nullable flag, and not-required flag from a type annotation.

    Args:
        annotation: Type annotation (e.g., int, Optional[int], NotRequiredColumn[int],
            or Annotated[int, Range(0, 100)])

    Returns:
        Tuple of (base_type, validators, is_optional, is_not_required)
        - For Annotated[int, Range(0, 100)]: (int, [Range(0, 100)], False, False)
        - For int: (int, [], False, False)
        - For Optional[int]: (int, [], True, False)
        - For Annotated[int, Range(0, 100)] | None: (int, [Range(0, 100)], True, False)
        - For NotRequiredColumn[int]: (int, [], False, True)
        - For NotRequiredColumn[Optional[int]]: (int, [], True, True)
        - For Union[int, str]: ((int, str), [], False, False)
        - For Union[int, str, None]: ((int, str), [], True, False)
    """
    validators = []
    is_optional = False
    is_not_required = False

    if isinstance(annotation, type) and issubclass(annotation, NotRequiredColumn):
        is_not_required = True
        annotation = getattr(annotation, "_inner_type", annotation)

    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        base_type = args[0]
        validators = list(args[1:])
        annotation = base_type

    origin = get_origin(annotation)
    # Handle both Union[int, str] and int | str (PEP 604)
    if origin is Union or isinstance(annotation, types.UnionType):
        args = get_args(annotation)
        # Check if None is in the union
        if type(None) in args:
            is_optional = True
            # Remove None from the union
            non_none_types = tuple(arg for arg in args if arg is not type(None))
            # If only one type remains after removing None, return it as a single type
            if len(non_none_types) == 1:
                base_type = non_none_types[0]
                # Annotated[T, ...] | None puts the validators inside the union
                if get_origin(base_type) is Annotated:
                    base_type, *inner_validators = get_args(base_type)
                    validators.extend(inner_validators)
                return base_type, validators, is_optional, is_not_required
            # Otherwise, return the tuple of types
            return non_none_types, validators, is_optional, is_not_required
        # Union without None - return all types as a tuple
        return args, validators, is_optional, is_not_required

    return annotation, validators, is_optional, is_not_required


@dataclass(frozen=True)
class Scalar:
    """A leaf value: a Python type, a Literal, or a backend dtype."""

    type: Any


@dataclass(frozen=True)
class ListOf:
    """A list column or value (``list[X]``)."""

    element: Field


@dataclass(frozen=True)
class StructOf:
    """A struct column or value, described by a nested Protocol."""

    schema: type
    fields: dict[str, Field]


@dataclass(frozen=True)
class Field:
    """One place a value can be: a column, a list element, or a struct field."""

    node: Scalar | ListOf | StructOf
    validators: tuple[Any, ...] = ()
    nullable: bool = False
    not_required: bool = False

    def has_validators(self) -> bool:
        """Whether this field or anything nested in it has validators."""
        if self.validators:
            return True
        if isinstance(self.node, ListOf):
            return self.node.element.has_validators()
        if isinstance(self.node, StructOf):
            return any(sub.has_validators() for sub in self.node.fields.values())
        return False


def is_nested_type(base_type: Any) -> bool:
    """Whether a base type from _extract_type_and_validators is a List or Struct annotation."""
    return base_type is list or get_origin(base_type) is list or _is_struct_schema(base_type)


def _is_struct_schema(base_type: Any) -> bool:
    return isinstance(base_type, type) and is_protocol(base_type)


def parse_annotation(annotation: Any, column_name: str) -> Field:
    """Parse a column annotation into a Field tree, rejecting what no backend can check."""
    return _parse(annotation, column_name, allow_not_required=True, seen=frozenset())


def _parse(
    annotation: Any, column_name: str, *, allow_not_required: bool, seen: frozenset[type]
) -> Field:
    base_type, validators, nullable, not_required = _extract_type_and_validators(annotation)
    if not_required and not allow_not_required:
        raise ValidationError(
            "NotRequiredColumn is only allowed on columns and struct fields",
            column_name=column_name,
        )

    node: Scalar | ListOf | StructOf
    if base_type is list:
        raise ValidationError("list needs an element type, e.g. list[int]", column_name=column_name)
    if get_origin(base_type) is list:
        (element,) = get_args(base_type)
        node = ListOf(_parse(element, column_name, allow_not_required=False, seen=seen))
    elif _is_struct_schema(base_type):
        assert isinstance(base_type, type)
        if base_type in seen:
            raise ValidationError(
                f"recursive schema is not supported: {base_type.__name__}",
                column_name=column_name,
            )
        hints = get_type_hints(base_type, include_extras=True)
        node = StructOf(
            base_type,
            {
                name: _parse(hint, column_name, allow_not_required=True, seen=seen | {base_type})
                for name, hint in hints.items()
            },
        )
    elif isinstance(base_type, tuple):
        raise ValidationError(
            f"unsupported type: {' | '.join(_type_name(t) for t in base_type)} "
            "(union element types are not supported)",
            column_name=column_name,
        )
    else:
        node = Scalar(base_type)
    return Field(node, tuple(validators), nullable, not_required)


def format_field(field: Field) -> str:
    """Render a Field the way it is written in a schema, e.g. ``list[str | None]``."""
    node = field.node
    if isinstance(node, ListOf):
        element = format_field(node.element)
        if node.element.nullable:
            element += " | None"
        return f"list[{element}]"
    if isinstance(node, StructOf):
        return node.schema.__name__
    return _type_name(node.type)


def _type_name(t: Any) -> str:
    if get_origin(t) is Literal:
        return f"Literal[{', '.join(repr(v) for v in get_args(t))}]"
    return getattr(t, "__name__", str(t))
