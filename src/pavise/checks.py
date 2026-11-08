"""DataFrame-level checks that validate relationships across columns.

Checks are attached to a schema Protocol with the ``@dataframe_check`` and ``@unique``
class decorators.
They are stored outside the Protocol's annotations and members, so they do not affect
structural subtyping: ``DataFrame[Child]`` stays compatible with ``DataFrame[Parent]``.

The actual evaluation is implemented in backend-specific modules:
- _pandas.validation for pandas
- _polars.validation for polars
"""

import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

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
        schema_columns = _annotated_names(schema) - {"__index__"}
        for column in columns:
            if column not in schema_columns:
                raise ValueError(f"unique: column {column!r} is not in schema {schema.__name__}")
        _attach(schema, check)
        return schema

    return decorator


def _annotated_names(schema: type) -> set[str]:
    """Names annotated on ``schema`` and its bases, without evaluating the annotations.

    Evaluating them could fail on forward references that only resolve after the
    decorator has run.
    """
    names: set[str] = set()
    for cls in schema.__mro__:
        if sys.version_info >= (3, 14):
            import annotationlib

            names.update(annotationlib.get_annotations(cls, format=annotationlib.Format.FORWARDREF))
        else:
            names.update(cls.__dict__.get("__annotations__", {}))
    return names


def get_dataframe_checks(schema: type) -> list[DataFrameCheck | UniqueCheck]:
    """Collect checks of a schema and its base Protocols, base classes first."""
    checks: list[DataFrameCheck | UniqueCheck] = []
    for cls in reversed(schema.__mro__):
        checks.extend(_CHECKS.get(cls, []))
    return checks
