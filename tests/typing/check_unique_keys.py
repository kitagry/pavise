# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Type checking cases for UniqueKeys, verified by pyright and never run.

A line expected to fail carries ``# pyright: ignore[<rule>]``; if pyright stops reporting
the error there, the ignore becomes unnecessary and pyright fails.
"""

from typing import Literal, Protocol

from pavise.checks import Key, UniqueKeys
from pavise.pandas import DataFrame


class UserColumns(Protocol):
    id: int
    email: str


class User(UserColumns, UniqueKeys[Key[Literal["id"]]], Protocol): ...


class StrictUser(UserColumns, UniqueKeys[Key[Literal["id"]] | Key[Literal["email"]]], Protocol): ...


class EmailUser(UserColumns, UniqueKeys[Key[Literal["email"]]], Protocol): ...


class IdEmailUser(UserColumns, UniqueKeys[Key[Literal["id", "email"]]], Protocol): ...


class EmailIdUser(UserColumns, UniqueKeys[Key[Literal["email", "id"]]], Protocol): ...


class ConsumerUser(UniqueKeys[Key[Literal["id"]]], Protocol):
    """Declared by a consumer without importing the producer's schema."""

    id: int


def users() -> DataFrame[UserColumns]: ...
def unique_users() -> DataFrame[User]: ...
def strict_users() -> DataFrame[StrictUser]: ...
def id_email_users() -> DataFrame[IdEmailUser]: ...
def needs_columns(df: DataFrame[UserColumns]) -> None: ...
def needs_user(df: DataFrame[User]) -> None: ...
def needs_strict_user(df: DataFrame[StrictUser]) -> None: ...
def needs_email_user(df: DataFrame[EmailUser]) -> None: ...
def needs_email_id_user(df: DataFrame[EmailIdUser]) -> None: ...
def needs_consumer_user(df: DataFrame[ConsumerUser]) -> None: ...


# A schema with keys can be used where fewer keys or none are expected.
needs_columns(unique_users())
needs_user(strict_users())
needs_email_user(strict_users())
needs_consumer_user(strict_users())

# A schema without the expected keys cannot.
needs_user(users())  # pyright: ignore[reportArgumentType]
needs_strict_user(unique_users())  # pyright: ignore[reportArgumentType]
needs_email_user(unique_users())  # pyright: ignore[reportArgumentType]

# A composite key matches regardless of column order, and is neither implied by nor implies
# its columns being unique one by one.
needs_email_id_user(id_email_users())
needs_email_id_user(strict_users())  # pyright: ignore[reportArgumentType]
needs_user(id_email_users())  # pyright: ignore[reportArgumentType]


# Keys must be wrapped in Key[Literal[...]].
class BareLiteral(UniqueKeys[Literal["id"]], Protocol):  # pyright: ignore[reportInvalidTypeArguments]
    id: int


class BareLiteralInUnion(UniqueKeys[Key[Literal["id"]] | Literal["email"]], Protocol):  # pyright: ignore[reportInvalidTypeArguments]
    id: int
    email: str


class NotColumnNames(UniqueKeys[Key[int]], Protocol):  # pyright: ignore[reportInvalidTypeArguments]
    id: int
