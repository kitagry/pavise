"""DataFrame-level checks that validate relationships across columns.

Checks are attached to a schema Protocol with the ``@dataframe_check`` and ``@unique``
class decorators.
They are stored outside the Protocol's annotations and members, so they do not affect
structural subtyping: ``DataFrame[Child]`` stays compatible with ``DataFrame[Parent]``.

Unique keys can instead be declared in the type with the ``UniqueKeys`` marker, so that a
type checker tells a schema with the keys apart from one with the same columns but no keys.

The actual evaluation is implemented in backend-specific modules:
- _pandas.validation for pandas
- _polars.validation for polars
"""

import sys
import types
import typing
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import typing_extensions

T = TypeVar("T", bound=type)


@dataclass
class DataFrameCheck:
    """A check evaluated against the whole DataFrame after column validation.

    ``func`` receives the backend's native DataFrame and returns either a boolean
    per row (pandas Series, polars Series or polars Expr) or a single bool.
    Null results are treated as passing.
    """

    func: Callable[[Any], Any]
    name: str


@dataclass
class UniqueCheck:
    """A check that the combination of ``columns`` is unique across rows.

    Nulls compare equal to each other, as in the column-level ``Unique()`` validator.
    """

    columns: tuple[str, ...]


# Kept outside the class namespace: on Python < 3.12 any class attribute set on a Protocol
# becomes a protocol member, which would break isinstance() on runtime_checkable schemas.
_CHECKS: "weakref.WeakKeyDictionary[type, list[DataFrameCheck | UniqueCheck]]" = (
    weakref.WeakKeyDictionary()
)


def _attach(schema: type, check: DataFrameCheck | UniqueCheck) -> None:
    own_checks = _CHECKS.setdefault(schema, [])
    # Decorators apply bottom-up; prepend so checks run in the order they are written.
    own_checks.insert(0, check)


ColumnsT = TypeVar("ColumnsT", bound=str)


class Key(Generic[ColumnsT]):
    """A unique key for ``UniqueKeys``: one column or a set of columns, as a ``Literal``.

    ``Key[Literal["user_id", "date"]]`` means the combination of ``user_id`` and ``date``
    is unique. The order of the columns does not matter.
    """


KeyT_contra = TypeVar("KeyT_contra", bound="Key[Any]", contravariant=True)


class UniqueKeys(Protocol[KeyT_contra]):
    """Declare the unique keys of a schema in its type.

    The type parameter is a ``Key`` or a union of them; each key is checked to be unique
    across rows when a DataFrame is validated. Rows whose key columns hold nulls in the same
    places count as duplicates. A schema with more keys can be used where a schema with
    fewer of them is expected, but not the other way around.

    Declare the columns in a Protocol of their own and combine it with the keys, so that
    schemas with different keys share the columns::

        class UserColumns(Protocol):
            id: int
            email: str

        class User(UserColumns, UniqueKeys[Key[Literal["id"]]], Protocol): ...

        class StrictUser(
            UserColumns, UniqueKeys[Key[Literal["id"]] | Key[Literal["email"]]], Protocol
        ): ...
    """

    # The parameter carries the keys for type checkers. Taking it as an argument makes a
    # schema with more keys a subtype of one with fewer.
    def __pavise_unique__(self, key: KeyT_contra, /) -> None: ...

    # Hidden from type checkers, which would otherwise count it as a protocol member.
    if not TYPE_CHECKING:

        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            _register_unique_keys(cls)


# Unique keys of each UniqueKeys subclass, including the keys it inherits.
_UNIQUE_KEYS: "weakref.WeakKeyDictionary[type, tuple[tuple[str, ...], ...]]" = (
    weakref.WeakKeyDictionary()
)


def _register_unique_keys(schema: type) -> None:
    sources: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    # __orig_bases__ is only in the class's own namespace when one of its bases is subscripted.
    for base in schema.__dict__.get("__orig_bases__", schema.__bases__):
        origin = get_origin(base)
        if origin is UniqueKeys:
            sources.append((f"{schema.__name__} itself", _parse_unique_keys(schema, base)))
            continue
        # A subscripted generic base such as Base[int] is registered as Base.
        base_class = origin or base
        if base_class in _UNIQUE_KEYS:
            sources.append((base_class.__name__, _UNIQUE_KEYS[base_class]))
    if not sources:
        return

    if len({frozenset(frozenset(key) for key in keys) for _, keys in sources}) > 1:
        names = ", ".join(name for name, _ in sources)
        raise TypeError(
            f"UniqueKeys: {schema.__name__} gets different unique keys from more than one "
            f"place ({names}), and type checkers only keep the first. Declare the columns "
            "in a Protocol without keys and combine it with the keys in one place, e.g. "
            f"class {schema.__name__}(Columns, UniqueKeys[...], Protocol)"
        )
    _UNIQUE_KEYS[schema] = sources[0][1]


def _parse_unique_keys(schema: type, marker: Any) -> tuple[tuple[str, ...], ...]:
    schema_columns = [c for c in _annotated_names(schema) if c != "__index__"]
    keys = []
    for key in _union_members(get_args(marker)[0]):
        if get_origin(key) is not Key:
            raise TypeError(
                "UniqueKeys: wrap each key in Key[...], e.g. UniqueKeys[Key[Literal['id']]], "
                f"got {key!r} in {schema.__name__}"
            )
        columns = _resolve_type_alias(get_args(key)[0])
        if get_origin(columns) is not Literal:
            raise TypeError(
                "Key: list the key columns as a Literal, e.g. "
                f"Key[Literal['user_id', 'date']], got {columns!r} in {schema.__name__}"
            )
        for column in get_args(columns):
            if column not in schema_columns:
                raise ValueError(
                    f"UniqueKeys: column {column!r} is not in schema {schema.__name__}"
                )
        # Equal Literals share a cache entry whatever their order, so follow the schema's.
        keys.append(tuple(c for c in schema_columns if c in get_args(columns)))
    return tuple(keys)


def _union_members(tp: Any) -> tuple[Any, ...]:
    tp = _resolve_type_alias(tp)
    if get_origin(tp) in (Union, types.UnionType):
        return tuple(_resolve_type_alias(member) for member in get_args(tp))
    return (tp,)


# Older typing_extensions define their own TypeAliasType even where typing has one
_TYPE_ALIAS_TYPES = tuple(
    m.TypeAliasType for m in (typing, typing_extensions) if hasattr(m, "TypeAliasType")
)


def _resolve_type_alias(tp: Any) -> Any:
    """Follow ``type X = ...`` aliases, which type checkers see through but get_origin does not."""
    while isinstance(tp, _TYPE_ALIAS_TYPES):
        tp = tp.__value__
    return tp


def dataframe_check(func: Callable[[Any], Any], *, name: str | None = None) -> Callable[[T], T]:
    """Attach a DataFrame-level check to a schema Protocol.

    Example::

        @dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
        class EventSchema(Protocol):
            start_at: datetime
            end_at: datetime
    """
    check = DataFrameCheck(func=func, name=name if name is not None else func.__name__)

    def decorator(schema: T) -> T:
        _attach(schema, check)
        return schema

    return decorator


def unique(*columns: str) -> Callable[[T], T]:
    """Require the combination of ``columns`` to be unique across rows.

    Rows whose key columns hold nulls in the same places count as duplicates, as with the
    column-level ``Unique()`` validator. Column names are checked against the schema when
    the decorator is applied.

    Example::

        @unique("user_id", "date")
        class DailyActivitySchema(Protocol):
            user_id: int
            date: date
            steps: int
    """
    if not columns:
        raise ValueError("unique: at least one column is required")
    for i, column in enumerate(columns):
        if column in columns[:i]:
            raise ValueError(f"unique: column {column!r} is given more than once")
    check = UniqueCheck(columns=columns)

    def decorator(schema: T) -> T:
        schema_columns = set(_annotated_names(schema)) - {"__index__"}
        for column in columns:
            if column not in schema_columns:
                raise ValueError(f"unique: column {column!r} is not in schema {schema.__name__}")
        _attach(schema, check)
        return schema

    return decorator


def _annotated_names(schema: type) -> list[str]:
    """Names annotated on ``schema`` and its bases, base classes first, without evaluating
    the annotations.

    Evaluating them could fail on forward references that only resolve after the
    decorator has run.
    """
    names: dict[str, None] = {}
    for cls in reversed(schema.__mro__):
        if sys.version_info >= (3, 14):
            import annotationlib

            names.update(
                dict.fromkeys(
                    annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF)
                )
            )
        else:
            names.update(dict.fromkeys(cls.__dict__.get("__annotations__", {})))
    return list(names)


def get_dataframe_checks(schema: type) -> list[DataFrameCheck | UniqueCheck]:
    """Collect the keys declared with ``UniqueKeys``, then the checks attached to a schema
    and its base Protocols, base classes first."""
    checks: list[DataFrameCheck | UniqueCheck] = [
        UniqueCheck(columns=key) for key in _UNIQUE_KEYS.get(schema, ())
    ]
    for cls in reversed(schema.__mro__):
        checks.extend(_CHECKS.get(cls, []))
    return checks
