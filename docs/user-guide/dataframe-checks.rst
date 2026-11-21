DataFrame-Level Checks
======================

Column validators check each column on its own. To validate relationships across
columns, such as ``start_at < end_at`` or ``total == a + b``, attach a DataFrame-level
check to the schema with the ``@dataframe_check`` class decorator.

Basic Usage
-----------

.. example-setup

   import pandas as pd

   df = pd.DataFrame({
       "start_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
       "end_at": pd.to_datetime(["2024-01-02", "2024-01-03"]),
   })

.. code-block:: python

   from datetime import datetime
   from typing import Protocol

   import pandas as pd
   from pavise.checks import dataframe_check
   from pavise.pandas import DataFrame

   @dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
   class EventSchema(Protocol):
       start_at: datetime
       end_at: datetime

   validated_df = DataFrame[EventSchema](df)

Multiple decorators can be stacked. Checks run from the top decorator to the bottom.

.. code-block:: python

   @dataframe_check(lambda df: df["a"] >= 0, name="a_non_negative")
   @dataframe_check(lambda df: df["total"] == df["a"] + df["b"], name="total_is_sum")
   class TotalSchema(Protocol):
       a: int
       b: int
       total: int

``name`` is used in error messages. When omitted, the function's ``__name__`` is used,
so pass ``name`` when using a lambda.

Check Functions
---------------

A check function receives the backend's native DataFrame and returns one of:

- **A boolean per row**: rows where it is ``False`` are reported as violations.

  - pandas: ``pd.Series`` of bools
  - polars: ``pl.Series`` of bools or a ``pl.Expr`` evaluating to bools

- **A single bool**: for constraints on the whole DataFrame, such as an aggregate.
  With polars, an ``Expr`` that aggregates to a single value (e.g.
  ``pl.col("share").sum() == 100``) is treated the same way.

.. code-block:: python

   import polars as pl
   from pavise.polars import DataFrame

   # polars: return an expression
   @dataframe_check(lambda df: pl.col("total") == pl.col("a") + pl.col("b"), name="total_is_sum")
   class TotalSchema(Protocol):
       a: int
       b: int
       total: int

   # Whole-DataFrame check
   @dataframe_check(lambda df: df["share"].sum() == 100, name="shares_sum_to_100")
   class ShareSchema(Protocol):
       share: int

Any other return value raises ``ValidationError``.

Null Handling
~~~~~~~~~~~~~

Rows where the check result is null are treated as passing, like SQL ``CHECK``
constraints. Use ``Optional`` columns or a separate check to reject missing values.

.. note::

   With numpy-backed pandas columns, comparisons involving ``NaN`` return ``False``
   rather than null, so those rows are reported as violations. Use nullable dtypes
   (e.g. ``Float64``) or allow missing values explicitly:
   ``lambda df: (df["start_at"] < df["end_at"]) | df["end_at"].isna()``.

Error Messages
--------------

When a row check fails, the error shows up to five violating rows with the schema's
columns (extra columns in the DataFrame are omitted):

.. code-block:: text

   ValidationError: DataFrame check 'start_before_end' failed

   Sample invalid values (showing first 1 of 1):
     Row 1: {'start_at': Timestamp('2024-01-05 00:00:00'), 'end_at': Timestamp('2024-01-03 00:00:00')}

The row label is the pandas index, or the row number for polars. The samples are also
available as ``error.invalid_samples``, and ``error.column_name`` is ``None``.
A failing single-bool check only reports the check name.

Unique Column Combinations
--------------------------

.. warning::

   ``@unique`` is deprecated and will be removed in the next minor release. Declare the keys
   in the schema type with ``UniqueKeys`` instead (see `Unique Keys in the Type`_):
   ``@unique("user_id", "date")`` becomes
   ``class DailyStepsSchema(UniqueKeys[Key[Literal["user_id", "date"]]], Protocol)``.
   Using ``@unique`` emits a ``DeprecationWarning``.

``@unique(*columns)`` requires the combination of the given columns to be unique across
rows, like a composite unique key. For a single column, the ``Unique()`` validator does
the same.

.. example-raises: ValidationError

.. code-block:: python

   from datetime import date
   from typing import Protocol

   import pandas as pd
   from pavise.checks import unique
   from pavise.pandas import DataFrame

   @unique("user_id", "date")
   class DailyStepsSchema(Protocol):
       user_id: int
       date: date
       steps: int

   df = pd.DataFrame({
       "user_id": [1, 2, 1],
       "date": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
       "steps": [3000, 5000, 4000],
   })
   DataFrame[DailyStepsSchema](df)

.. example-output

.. code-block:: text

   ValidationError: Columns ('user_id', 'date') must be unique, found 1 duplicate key

   Sample duplicate keys (showing first 1 of 1):
     {'user_id': 1, 'date': datetime.date(2024, 1, 1)}: rows [0, 2]

The error shows up to five duplicate keys, each with up to five of its rows (pandas
index labels, or row numbers for polars). ``error.invalid_samples`` holds a
``(row, key)`` tuple for each shown row.

- **Column names are checked when the decorator is applied**, so a typo raises
  ``ValueError`` at import time instead of when data is validated. Columns inherited
  from a base Protocol can be used.
- **Nulls are equal to each other**: rows whose key columns hold nulls in the same places
  are duplicates. This matches the ``Unique()`` validator, pandas ``duplicated()`` and
  polars ``is_duplicated()``, and differs from SQL ``UNIQUE``, which lets nulls repeat.
  Declare the key columns without ``Optional`` to reject nulls.
- ``@unique`` can be stacked with other ``@unique`` and ``@dataframe_check`` decorators,
  runs in the same order and is inherited the same way.
- If a ``NotRequiredColumn`` key column is absent, the check is skipped.

.. note::

   Like ``@dataframe_check`` and column validators, ``@unique`` is not part of the type.
   It is only checked at runtime, by ``DataFrame[Schema](df)`` or by ``collect()`` on a
   polars ``LazyFrame``. ``@unique`` returns the class unchanged and the type parameter
   of ``DataFrame`` is covariant, so a type checker accepts a ``DataFrame`` of another
   Protocol with the same columns where ``DataFrame[DailyStepsSchema]`` is expected, and
   the other way around. To make the keys part of the type, declare them with
   ``UniqueKeys`` (see `Unique Keys in the Type`_).

Unique Keys in the Type
-----------------------

``UniqueKeys`` declares the unique keys of a schema in its type. Each key is wrapped in
``Key[Literal[...]]``, and Pavise checks every key when the DataFrame is validated, in the
same way as ``@unique``.

.. example-raises: ValidationError

.. code-block:: python

   from typing import Literal, Protocol

   import pandas as pd
   from pavise.checks import Key, UniqueKeys
   from pavise.pandas import DataFrame

   class User(UniqueKeys[Key[Literal["id"]]], Protocol):
       id: int
       email: str

   df = pd.DataFrame({"id": [1, 2, 1], "email": ["a@example.com", "b@example.com", "c@example.com"]})
   DataFrame[User](df)

.. example-output

.. code-block:: text

   ValidationError: Columns ('id',) must be unique, found 1 duplicate key

   Sample duplicate keys (showing first 1 of 1):
     {'id': 1}: rows [0, 2]

A union declares several keys, and a ``Literal`` with several columns declares a composite
key. The order of the columns in a key does not matter.

.. code-block:: python

   from datetime import date

   # id is unique, and email is unique
   class StrictUser(UniqueKeys[Key[Literal["id"]] | Key[Literal["email"]]], Protocol):
       id: int
       email: str

   # The combination of user_id and date is unique
   class DailySteps(UniqueKeys[Key[Literal["user_id", "date"]]], Protocol):
       user_id: int
       date: date
       steps: int

Type aliases keep long keys short:

.. code-block:: python

   IdKey = Key[Literal["id"]]
   EmailKey = Key[Literal["email"]]

   class StrictUser(UniqueKeys[IdKey | EmailKey], Protocol):
       id: int
       email: str

On Python 3.12 and later, ``type IdKey = Key[Literal["id"]]`` works as well.

Type Checking
~~~~~~~~~~~~~

The keys are part of the schema's type, so a type checker rejects a DataFrame whose schema
does not declare them, even when it has the same columns. A schema with more keys is
accepted where fewer of them are expected.

.. code-block:: python

   class UserRow(Protocol):
       id: int
       email: str

   def send_welcome_mail(users: DataFrame[User]) -> None: ...

   send_welcome_mail(DataFrame[StrictUser](df.drop_duplicates("id")))  # OK: also unique on id
   # send_welcome_mail(DataFrame[UserRow](df))  # type error: id is not declared unique

Since the schema is still a ``Protocol``, a module that only consumes the data can declare
its own ``class User(UniqueKeys[Key[Literal["id"]]], Protocol)`` with the columns it needs,
without importing the producer's schema. Whichever schema the DataFrame was validated
against, its keys were checked.

Keys must be written as ``Key[Literal[...]]``: a bare ``Literal`` such as
``UniqueKeys[Literal["id"]]`` is a type error, and raises ``TypeError`` when the class is
defined. A column that is not in the schema raises ``ValueError``.

Same Columns, Different Keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Protocols are structural, so a schema with other keys does not need to inherit from
``User`` to be accepted where ``User`` is expected. Combine a ``Protocol`` of only the columns
with the keys of each schema, or declare the columns again:

.. code-block:: python

   class UserColumns(Protocol):
       id: int
       email: str

   class User(UserColumns, UniqueKeys[Key[Literal["id"]]], Protocol): ...

   # Combines the columns with other keys
   class StrictUser(UserColumns, UniqueKeys[Key[Literal["id"]] | Key[Literal["email"]]], Protocol): ...

   # Declares the columns again
   class ContactableUser(UniqueKeys[Key[Literal["id"]] | Key[Literal["email"]]], Protocol):
       id: int
       email: str

   def send_welcome_mail(users: DataFrame[User]) -> None: ...

   send_welcome_mail(DataFrame[StrictUser](df.drop_duplicates("id")))  # OK
   send_welcome_mail(DataFrame[ContactableUser](df.drop_duplicates("id")))  # OK

Inheriting from ``User`` and adding keys, as in
``class StrictUser(User, UniqueKeys[...], Protocol)``, raises ``TypeError``: type checkers
see the keys of only one of the bases.

A child ``Protocol`` that inherits from a schema with keys inherits its keys. As with
``@unique``, the keys of a schema used as a struct column are not checked on the struct
values. A ``@runtime_checkable`` schema with ``UniqueKeys`` requires ``isinstance()``
targets to have the marker's ``__pavise_unique__`` method as well.

Checks in the Type
------------------

``Checked`` declares DataFrame-level checks in the schema's type, as ``UniqueKeys`` does for
unique keys. A check is a subclass of ``Check`` whose ``__call__`` returns what a
``@dataframe_check`` function returns, and its class name is the check's name in errors.

.. example-raises: ValidationError

.. code-block:: python

   from datetime import datetime
   from typing import Literal, Protocol

   import pandas as pd
   from pavise.checks import Check, Checked, Key, UniqueKeys
   from pavise.pandas import DataFrame

   class StartBeforeEnd(Check):
       def __call__(self, df):
           return df["start_at"] < df["end_at"]

   class Event(Checked[StartBeforeEnd], Protocol):
       start_at: datetime
       end_at: datetime

   df = pd.DataFrame({
       "start_at": pd.to_datetime(["2024-01-01", "2024-01-05"]),
       "end_at": pd.to_datetime(["2024-01-02", "2024-01-03"]),
   })
   DataFrame[Event](df)

.. example-output

.. code-block:: text

   ValidationError: DataFrame check 'StartBeforeEnd' failed

   Sample invalid values (showing first 1 of 1):
     Row 1: {'start_at': Timestamp('2024-01-05 00:00:00'), 'end_at': Timestamp('2024-01-03 00:00:00')}

A union declares several checks, and ``Checked`` can be combined with ``UniqueKeys``.
Unique keys are checked before the checks.

.. code-block:: python

   class PositiveDuration(Check):
       def __call__(self, df):
           return (df["end_at"] - df["start_at"]).dt.total_seconds() > 0

   class StrictEvent(
       UniqueKeys[Key[Literal["id"]]],
       Checked[StartBeforeEnd | PositiveDuration],
       Protocol,
   ):
       id: int
       start_at: datetime
       end_at: datetime

As with keys, a type checker rejects a DataFrame whose schema does not declare the expected
checks, and accepts a schema with more checks where fewer are expected. To give a schema
other checks, do not inherit from a schema with checks: combine a ``Protocol`` of only the
columns with the checks, or declare the columns again (see `Same Columns, Different Keys`_).
Inheriting from a schema with checks and adding checks raises ``TypeError``.

Use ``Checked`` for constraints that should be part of the type, and ``@dataframe_check`` for
quick checks that only need to run at runtime.

A check cannot take arguments, since only its class appears in the type. Write a separate
``Check`` subclass for each variant. Subclassing a ``Check`` that already implements
``__call__`` raises ``TypeError``, because a type checker would accept a schema that only
ran the parent's check where the subclass is expected. A subclass of ``Check`` that leaves
``__call__`` abstract can be shared as a base.

When Checks Run
---------------

- After all column type checks, column validators and strict mode pass, so check
  functions can rely on the declared columns and types. ``NotRequiredColumn`` columns
  may still be absent, so guard for them (e.g. ``"col" in df.columns``).
- ``make_empty()`` also runs the checks, so whole-DataFrame checks that fail on an empty
  DataFrame (such as ``sum() == 100``) make ``make_empty()`` raise.
- Validation stops at the first failing check.
- ``polars.LazyFrame[Schema](lf)`` only checks the schema at construction; DataFrame-level
  checks run on ``collect()``.

Inheritance and Structural Subtyping
------------------------------------

A Protocol that inherits from a decorated Protocol inherits its checks. The parent's
checks run before the child's own checks.

.. example-setup

   from pavise.pandas import DataFrame

   df = pd.DataFrame({
       "start_at": pd.to_datetime(["2024-01-01"]),
       "end_at": pd.to_datetime(["2024-01-02"]),
       "label": ["launch"],
   })

.. code-block:: python

   class LabeledEventSchema(EventSchema, Protocol):
       label: str

   DataFrame[LabeledEventSchema](df)  # runs 'start_before_end'

The decorator stores checks outside the Protocol's annotations and members, so it does
not affect static type checking. A schema that is structurally compatible but does not
inherit from ``EventSchema`` can still be passed where ``DataFrame[EventSchema]`` is
expected. Its own checks, not ``EventSchema``'s, run when it is validated.
