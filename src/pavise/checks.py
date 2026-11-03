"""DataFrame-level checks that validate relationships across columns.

Checks are attached to a schema Protocol with the ``@dataframe_check`` class decorator.
They are stored outside the Protocol's annotations and members, so they do not affect
structural subtyping: ``DataFrame[Child]`` stays compatible with ``DataFrame[Parent]``.

The actual evaluation is implemented in backend-specific modules:
- _pandas.validation for pandas
- _polars.validation for polars
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

_CHECKS_ATTR = "__pavise_checks__"


@dataclass
class DataFrameCheck:
    """A check evaluated against the whole DataFrame after column validation.

    ``func`` receives the backend's native DataFrame and returns either a boolean
    per row (pandas Series, polars Series or polars Expr) or a single bool.
    Null results are treated as passing.
    """

    func: Callable[[Any], Any]
    name: str


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
        own_checks: list[DataFrameCheck] = vars(schema).get(_CHECKS_ATTR, [])
        # Decorators apply bottom-up; prepend so checks run in the order they are written.
        setattr(schema, _CHECKS_ATTR, [check, *own_checks])
        return schema

    return decorator


def get_dataframe_checks(schema: type) -> list[DataFrameCheck]:
    """Collect checks of a schema and its base Protocols, base classes first."""
    checks: list[DataFrameCheck] = []
    for cls in reversed(schema.__mro__):
        checks.extend(vars(cls).get(_CHECKS_ATTR, []))
    return checks
