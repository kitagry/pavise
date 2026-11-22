"""Backend-independent parsing of schema annotations.

Column annotations are unwrapped one level at a time (NotRequiredColumn, Annotated, Optional).
List and Struct columns (``list[X]`` and nested Protocols, TypedDicts and dataclasses) are parsed
into a tree of ``Field`` nodes, which each backend compares with its nested dtypes.
"""

from __future__ import annotations

import dataclasses
import sys
import types
import typing
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    ForwardRef,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from typing_extensions import is_protocol, is_typeddict

from pavise.exceptions import ValidationError
from pavise.types import NotRequiredColumn
from pavise.validators import Custom, MaxLen, MinLen, Unique


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
    """A struct column or value, described by a nested Protocol, TypedDict or dataclass."""

    schema: type
    fields: dict[str, Field]


@dataclass(frozen=True)
class Field:
    """One place a value can be: a column, a list element, or a struct field."""

    node: Scalar | ListOf | StructOf
    validators: tuple[Any, ...] = ()
    nullable: bool = False
    not_required: bool = False


def is_nested_type(base_type: Any) -> bool:
    """
    Whether a base type from _extract_type_and_validators is a List or Struct annotation.

    Struct schemas pavise does not support count too, so parsing them reports why.
    """
    return (
        base_type is list
        or get_origin(base_type) is list
        or _is_struct_schema(base_type)
        or _unsupported_struct_reason(base_type) is not None
    )


def _is_struct_schema(base_type: Any) -> bool:
    return isinstance(base_type, type) and (
        is_protocol(base_type) or is_typeddict(base_type) or dataclasses.is_dataclass(base_type)
    )


def _unsupported_struct_reason(base_type: Any) -> str | None:
    """Why a type written as a struct schema cannot be one, or None if it is no such type."""
    if _is_struct_schema(get_origin(base_type)):
        return "generic struct schemas are not supported"
    if not isinstance(base_type, type):
        return None
    is_named_tuple = issubclass(base_type, tuple) and hasattr(base_type, "_fields")
    # Detected by module so that pavise does not need to import pydantic
    is_pydantic = any(c.__module__.split(".")[0] == "pydantic" for c in base_type.__mro__)
    if is_named_tuple or is_pydantic:
        return "struct schemas must be a Protocol, TypedDict or dataclass"
    return None


def _qualifiers(name: str) -> frozenset[Any]:
    # Older typing_extensions define their own forms even where typing has them
    return frozenset(getattr(m, name) for m in (typing, typing_extensions) if hasattr(m, name))


_REQUIRED = _qualifiers("Required")
_NOT_REQUIRED = _qualifiers("NotRequired")
_READ_ONLY = _qualifiers("ReadOnly")


def _typeddict_qualifier_name(hint: Any) -> str | None:
    origin = get_origin(hint)
    for name, forms in (
        ("Required", _REQUIRED),
        ("NotRequired", _NOT_REQUIRED),
        ("ReadOnly", _READ_ONLY),
    ):
        if origin in forms:
            return name
    return None


def schema_columns(schema: type) -> dict[str, tuple[Any, bool]]:
    """
    Each column's annotation and whether the column may be missing.

    ``schema`` is a Protocol, TypedDict or dataclass, used as the schema of a DataFrame or of
    a struct. TypedDict qualifiers are removed from the annotations; NotRequiredColumn is kept.
    """
    if dataclasses.is_dataclass(schema):
        return _dataclass_columns(schema)
    if is_typeddict(schema):
        return _typeddict_columns(schema)
    return _protocol_columns(schema)


def _protocol_columns(schema: type) -> dict[str, tuple[Any, bool]]:
    hints = get_type_hints(schema, include_extras=True)
    return {name: _column(name, hint) for name, hint in hints.items()}


def _dataclass_columns(schema: type) -> dict[str, tuple[Any, bool]]:
    # fields() leaves out ClassVar and InitVar. get_type_hints resolves string types in the
    # module of the class that declares each field, but fails on a string InitVar on
    # Python 3.10 and on names local to a function, so those are resolved like other
    # forward references. They are resolved here rather than when parsed, because a
    # DataFrame's columns are read as types, and NotRequiredColumn must be seen to let a
    # column be missing.
    try:
        hints = get_type_hints(schema, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    columns = {}
    for f in dataclasses.fields(schema):
        hint = hints.get(f.name, f.type)
        if isinstance(hint, str):
            hint = _resolve_forward_reference(hint, f.name, schema)
        columns[f.name] = _column(f.name, hint)
    return columns


def _typeddict_columns(schema: type) -> dict[str, tuple[Any, bool]]:
    columns = {}
    for name, hint in get_type_hints(schema, include_extras=True).items():
        hint, not_required = _strip_typeddict_qualifiers(hint)
        if _is_not_required_column(hint):
            raise ValidationError(
                "NotRequiredColumn cannot be used in a TypedDict; use NotRequired",
                column_name=name,
            )
        if not_required is None:
            # __optional_keys__ misses NotRequired written as a string (from __future__ import
            # annotations, or typing.TypedDict on Python 3.10), but always reflects total=False
            not_required = name in schema.__optional_keys__
        columns[name] = (hint, not_required)
    return columns


def _column(name: str, hint: Any) -> tuple[Any, bool]:
    """A column of a schema other than a TypedDict, where TypedDict qualifiers have no meaning."""
    base_type = _extract_type_and_validators(hint)[0]
    if (qualifier := _typeddict_qualifier_name(base_type)) is not None:
        raise ValidationError(_misplaced_qualifier_message(qualifier), column_name=name)
    return hint, _is_not_required_column(hint)


def _misplaced_qualifier_message(qualifier: str) -> str:
    message = f"{qualifier} can only be used in a TypedDict"
    if qualifier != "ReadOnly":
        message += "; use NotRequiredColumn for a field that may be absent"
    return message


def _is_not_required_column(hint: Any) -> bool:
    return isinstance(hint, type) and issubclass(hint, NotRequiredColumn)


def _strip_typeddict_qualifiers(hint: Any) -> tuple[Any, bool | None]:
    """Remove Required, NotRequired and ReadOnly, returning whether NotRequired was given."""
    origin = get_origin(hint)
    if origin is Annotated:
        inner, *metadata = get_args(hint)
        inner, not_required = _strip_typeddict_qualifiers(inner)
        return Annotated[(inner, *metadata)], not_required
    qualifier = _typeddict_qualifier_name(hint)
    if qualifier is not None:
        (inner,) = get_args(hint)
        inner, not_required = _strip_typeddict_qualifiers(inner)
        if qualifier == "ReadOnly":
            return inner, not_required
        return inner, qualifier == "NotRequired"
    return hint, None


def parse_annotation(annotation: Any, column_name: str, owner: type | None = None) -> Field:
    """
    Parse a column annotation into a Field tree, rejecting what no backend can check.

    ``owner`` is the schema the annotation is written in, used to resolve forward references.
    """
    return _parse(annotation, column_name, owner, allow_not_required=True, seen=frozenset())


def parse_nested_column(annotation: Any, column_name: str, owner: type) -> Field:
    """Parse a List or Struct column, rejecting validators that have no meaning where written."""
    field = parse_annotation(annotation, column_name, owner)
    _check_validator_targets(field, column_name, column_name)
    return field


@dataclass(frozen=True)
class SchemaColumn:
    """A column of a DataFrame schema, with its annotation unwrapped."""

    annotation: Any
    missing_ok: bool
    base_type: Any
    validators: tuple[Any, ...]
    nullable: bool
    nested: bool


# Reading a schema's annotations costs more than checking a small DataFrame, so it is done
# once per schema. Only results are kept, never errors: a forward reference that cannot be
# resolved yet may be defined later.
_DATAFRAME_COLUMNS: weakref.WeakKeyDictionary[type, Mapping[str, SchemaColumn]] = (
    weakref.WeakKeyDictionary()
)
_NESTED_FIELDS: weakref.WeakKeyDictionary[type, dict[str, Field]] = weakref.WeakKeyDictionary()


def dataframe_columns(schema: type) -> Mapping[str, SchemaColumn]:
    """The columns of a DataFrame schema, read once per schema."""
    try:
        columns = _DATAFRAME_COLUMNS.get(schema)
    except TypeError:
        # No class, such as a string: reading it reports what is wrong with it
        return _read_dataframe_columns(schema)
    if columns is None:
        columns = _read_dataframe_columns(schema)
        # A dataclass whose type hints do not resolve yet keeps ForwardRefs, which would
        # stay wrong after the name is defined
        if not any(_is_unresolved(c.base_type) for c in columns.values()):
            _DATAFRAME_COLUMNS[schema] = columns
    return columns


def _is_unresolved(base_type: Any) -> bool:
    members = base_type if isinstance(base_type, tuple) else (base_type,)
    return any(isinstance(m, (str, ForwardRef)) for m in members)


def _read_dataframe_columns(schema: type) -> Mapping[str, SchemaColumn]:
    columns = {}
    for name, (annotation, missing_ok) in schema_columns(schema).items():
        base_type, validators, nullable, _ = _extract_type_and_validators(annotation)
        columns[name] = SchemaColumn(
            annotation,
            missing_ok,
            base_type,
            tuple(validators),
            nullable,
            is_nested_type(base_type),
        )
    return types.MappingProxyType(columns)


def nested_column_field(schema: type, column_name: str, column: SchemaColumn) -> Field:
    """
    The parsed annotation of a List or Struct column of ``schema``, parsed once.

    Parsed only when the column is checked, so that a column that may be missing and is
    absent never reports errors in its annotation.
    """
    fields = _NESTED_FIELDS.setdefault(schema, {})
    field = fields.get(column_name)
    if field is None:
        field = parse_nested_column(column.annotation, column_name, schema)
        fields[column_name] = field
    return field


def _check_validator_targets(field: Field, column_name: str, path: str) -> None:
    node = field.node
    for validator in field.validators:
        name = type(validator).__name__
        if isinstance(node, ListOf) and not isinstance(validator, (MinLen, MaxLen, Custom)):
            raise nested_error(
                f"{name} cannot be used on a list; use MinLen, MaxLen or Custom", column_name, path
            )
        if isinstance(node, StructOf) and not isinstance(validator, Custom):
            raise nested_error(f"{name} cannot be used on a struct; use Custom", column_name, path)
        # Whether values must be unique per list, per column or per struct has no single answer
        if isinstance(validator, Unique):
            raise nested_error(
                f"{name} cannot be used inside list and struct columns", column_name, path
            )

    if isinstance(node, ListOf):
        _check_validator_targets(node.element, column_name, f"{path}[]")
    elif isinstance(node, StructOf):
        for name, sub in node.fields.items():
            _check_validator_targets(sub, column_name, f"{path}.{name}")


def literal_base_type(t: Any) -> Any:
    """The type of a Literal's values (str for ``Literal["a"]``), or ``t`` if it is no Literal."""
    if get_origin(t) is not Literal:
        return t
    values = [v for v in get_args(t) if v is not None]
    return type(values[0]) if values else Any


def nested_path(column_name: str, path: str) -> str | None:
    """The path to report, or None when the column itself failed."""
    return path if path != column_name else None


def nested_violation_error(
    message: str,
    samples: list[tuple[Any, Any, Any]],
    total_invalid: int,
    format_value: Callable[[Any], str],
    column_name: str,
    path: str,
) -> ValidationError:
    """
    Report values a validator rejected at ``path``, given as (row, value, row value) samples.

    Inside a List or Struct column, each sample also shows the whole row it was found in.
    """
    at_column = path == column_name
    msg = f"{message}\n\nSample invalid values (showing first {len(samples)} of {total_invalid}):"
    for row, value, row_value in samples:
        msg += f"\n  Row {row}: {format_value(value)}"
        if not at_column:
            msg += f" (in {row_value!r})"
    return ValidationError(
        msg,
        column_name=column_name,
        invalid_samples=[(row, value) for row, value, _ in samples],
        path=nested_path(column_name, path),
    )


def nested_error(message: str, column_name: str, path: str) -> ValidationError:
    return ValidationError(message, column_name=column_name, path=nested_path(column_name, path))


def _resolve_forward_reference(annotation: str, column_name: str, owner: type | None) -> Any:
    """Evaluate a string annotation in the namespace of the schema it is written in."""
    module = sys.modules.get(owner.__module__) if owner is not None else None
    globalns = dict(vars(module)) if module is not None else {}
    localns = {owner.__name__: owner} if owner is not None else {}
    try:
        return eval(annotation, globalns, localns)
    except NameError as e:
        raise ValidationError(
            f"cannot resolve forward reference {annotation!r}", column_name=column_name
        ) from e


def _parse(
    annotation: Any,
    column_name: str,
    owner: type | None,
    *,
    allow_not_required: bool,
    seen: frozenset[type],
) -> Field:
    base_type, validators, nullable, not_required = _extract_type_and_validators(annotation)
    if isinstance(base_type, ForwardRef):
        base_type = base_type.__forward_arg__
    if isinstance(base_type, str):
        # Python 3.10's get_type_hints leaves strings inside list[...] or Optional[...] as is
        resolved = _resolve_forward_reference(base_type, column_name, owner)
        field = _parse(
            resolved, column_name, owner, allow_not_required=allow_not_required, seen=seen
        )
        return Field(
            field.node,
            tuple(validators) + field.validators,
            nullable or field.nullable,
            not_required or field.not_required,
        )
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
        node = ListOf(_parse(element, column_name, owner, allow_not_required=False, seen=seen))
    elif _is_struct_schema(base_type):
        assert isinstance(base_type, type)
        if base_type in seen:
            raise ValidationError(
                f"recursive schema is not supported: {base_type.__name__}",
                column_name=column_name,
            )
        try:
            columns = schema_columns(base_type)
        except ValidationError as e:
            # Reported at the struct column, like other errors in the struct's schema
            raise ValidationError(e._base_message, column_name=column_name) from None
        fields = {}
        for name, (hint, missing_ok) in columns.items():
            field = _parse(
                hint, column_name, base_type, allow_not_required=True, seen=seen | {base_type}
            )
            fields[name] = dataclasses.replace(field, not_required=field.not_required or missing_ok)
        node = StructOf(base_type, fields)
    elif (reason := _unsupported_struct_reason(base_type)) is not None:
        raise ValidationError(
            f"unsupported type: {_type_name(base_type)} ({reason})", column_name=column_name
        )
    elif isinstance(base_type, tuple):
        raise ValidationError(
            f"unsupported type: {' | '.join(_type_name(t) for t in base_type)} "
            "(union element types are not supported)",
            column_name=column_name,
        )
    elif (qualifier := _typeddict_qualifier_name(base_type)) is not None:
        raise ValidationError(_misplaced_qualifier_message(qualifier), column_name=column_name)
    else:
        node = Scalar(base_type)
        # A top-level Literal column accepts nulls when None is one of its values
        if get_origin(base_type) is Literal and None in get_args(base_type):
            nullable = True
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
    if _is_struct_schema(get_origin(t)):
        return f"{get_origin(t).__name__}[{', '.join(_type_name(a) for a in get_args(t))}]"
    return getattr(t, "__name__", str(t))
