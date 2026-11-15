# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Type checking cases for Checked, verified by pyright and never run.

A line expected to fail carries ``# pyright: ignore[<rule>]``; if pyright stops reporting
the error there, the ignore becomes unnecessary and pyright fails.
"""

from typing import Any, Literal, Protocol

from pavise.checks import Check, Checked, Key, UniqueKeys
from pavise.pandas import DataFrame


class StartBeforeEnd(Check):
    def __call__(self, df: Any) -> Any:
        return df["start_at"] < df["end_at"]


class PositiveStart(Check):
    def __call__(self, df: Any) -> Any:
        return df["start_at"] > 0


class EventColumns(Protocol):
    id: int
    start_at: int
    end_at: int


class Event(EventColumns, Checked[StartBeforeEnd], Protocol): ...


class StrictEvent(
    EventColumns,
    UniqueKeys[Key[Literal["id"]]],
    Checked[StartBeforeEnd | PositiveStart],
    Protocol,
): ...


class ConsumerEvent(Checked[StartBeforeEnd], Protocol):
    """Declared by a consumer without importing the producer's schema."""

    start_at: int
    end_at: int


def events() -> DataFrame[EventColumns]: ...
def checked_events() -> DataFrame[Event]: ...
def strict_events() -> DataFrame[StrictEvent]: ...
def needs_columns(df: DataFrame[EventColumns]) -> None: ...
def needs_event(df: DataFrame[Event]) -> None: ...
def needs_strict_event(df: DataFrame[StrictEvent]) -> None: ...
def needs_consumer_event(df: DataFrame[ConsumerEvent]) -> None: ...


# A schema with checks can be used where fewer checks or none are expected.
needs_columns(checked_events())
needs_event(strict_events())
needs_consumer_event(strict_events())

# A schema without the expected checks cannot.
needs_event(events())  # pyright: ignore[reportArgumentType]
needs_strict_event(checked_events())  # pyright: ignore[reportArgumentType]


# Checked takes Check subclasses only.
class NotACheck(Checked[int], Protocol):  # pyright: ignore[reportInvalidTypeArguments]
    id: int
