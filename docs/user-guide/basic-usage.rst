Basic Usage
===========

This page covers the fundamental concepts and usage patterns of Pavise.

Design Philosophy
-----------------

Pavise is designed with these principles:

1. **Type-first design**: Leverage Python's type system for DataFrame validation
2. **Structural subtyping**: Use Protocol for flexible schema definitions
3. **Optional runtime validation**: Type checking is free, validation is opt-in
4. **Detailed error messages**: Help users quickly identify and fix issues

Type Checking vs Runtime Validation
------------------------------------

Type Checking Only
~~~~~~~~~~~~~~~~~~

For internal functions, use type annotations without runtime overhead:

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame

   class UserSchema(Protocol):
       user_id: int
       name: str

   def internal_processing(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
       # No validation, just type hints
       # Type checker ensures schema compliance
       return df

Each annotation of the ``Protocol`` is a column, except a ``ClassVar``, which is not a column
(as in a :ref:`dataclass schema <typeddict-and-dataclass-schemas>`).

Runtime Validation
~~~~~~~~~~~~~~~~~~

At system boundaries (loading from CSV, database, API), validate explicitly:

.. example-setup

   with open("users.csv", "w") as f:
       f.write("user_id,name\n1,Alice\n2,Bob\n")

.. code-block:: python

   import pandas as pd
   from pavise.pandas import DataFrame

   # Load data from external source
   raw_df = pd.read_csv("users.csv")

   # Validate at boundary
   validated_df = DataFrame[UserSchema](raw_df)

   # Now pass to internal functions with confidence
   result = internal_processing(validated_df)

Covariance and Structural Subtyping
------------------------------------

Pavise uses covariant type parameters, allowing schemas with more columns to be used where fewer are expected:

.. code-block:: python

   class MinimalSchema(Protocol):
       user_id: int

   class ExtendedSchema(Protocol):
       user_id: int
       name: str
       age: int

   def process_minimal(df: DataFrame[MinimalSchema]) -> None:
       pass

   extended_df: DataFrame[ExtendedSchema] = ...
   process_minimal(extended_df)  # OK: ExtendedSchema is compatible

.. _column-types-must-match-exactly:

Column Types Must Match Exactly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extra columns are fine, but the types of the shared columns must be the same. A column is a
mutable attribute of the ``Protocol``, and type checkers compare mutable attributes as
invariant: something that can be assigned ``float`` cannot stand in for an ``int`` attribute,
and the other way around. So none of these are accepted:

* ``int`` where ``float`` is expected
* ``int`` where ``int | None`` is expected
* A struct with extra fields where a struct with fewer fields is expected, or, with pyright,
  another ``Protocol`` with the same fields (see :doc:`nested-types`)
* ``list[X]`` where ``list[Y]`` is expected, unless ``X`` and ``Y`` are the same type (``list``
  is invariant too)

.. code-block:: python

   class IntScore(Protocol):
       score: int

   class OptionalScore(Protocol):
       score: int | None

   def process_scores(df: DataFrame[OptionalScore]) -> None:
       pass

   int_scores: DataFrame[IntScore] = ...
   process_scores(int_scores)  # Type error: "score" is invariant because it is mutable

When the schemas come from different places, either use the same schema on both sides
(import it from the module that produces the DataFrame), or tell the type checker with
``typing.cast`` that the columns are compatible:

.. code-block:: python

   from typing import cast

   process_scores(cast(DataFrame[OptionalScore], int_scores))

``cast`` does not validate anything; validate the DataFrame at runtime if you are not sure.

Backend Selection
-----------------

pandas Backend
~~~~~~~~~~~~~~

.. example-setup

   import polars as pl

   pandas_df = pd.DataFrame({"user_id": [1], "name": ["Alice"]})
   polars_df = pl.DataFrame({"user_id": [1], "name": ["Alice"]})

.. code-block:: python

   from pavise.pandas import DataFrame

   validated_df = DataFrame[UserSchema](pandas_df)

polars Backend
~~~~~~~~~~~~~~

.. code-block:: python

   from pavise.polars import DataFrame

   validated_df = DataFrame[UserSchema](polars_df)

The API is identical across backends, but they validate against their respective type systems.

Handling Optional Columns
--------------------------

Nullable Columns
~~~~~~~~~~~~~~~~

Use ``Optional[T]`` for nullable columns:

.. code-block:: python

   from typing import Optional

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: Optional[int]  # Allows None values

Note: In pandas, nullable integers are stored as ``float64`` when they contain nulls.
In polars, all types are nullable by default.

Optional Columns
~~~~~~~~~~~~~~~~

Use ``NotRequiredColumn[T]`` for columns that may not exist in the DataFrame:

.. example-raises: ValidationError

.. code-block:: python

   from typing import Optional
   from pavise.pandas import DataFrame, NotRequiredColumn

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: NotRequiredColumn[int]  # Column can be missing
       email: NotRequiredColumn[Optional[str]]  # Column can be missing, or contain None

   # Valid: age column is missing
   df1 = pd.DataFrame({"user_id": [1], "name": ["Alice"]})
   validated_df1 = DataFrame[UserSchema](df1)  # OK

   # Valid: age column is present
   df2 = pd.DataFrame({"user_id": [1], "name": ["Alice"], "age": [25]})
   validated_df2 = DataFrame[UserSchema](df2)  # OK

   # Invalid: age column is present but has wrong type
   df3 = pd.DataFrame({"user_id": [1], "name": ["Alice"], "age": ["invalid"]})
   DataFrame[UserSchema](df3)  # ValidationError

Key differences:

* ``Optional[T]``: Column must exist, but can contain ``None`` values
* ``NotRequiredColumn[T]``: Column can be missing, but if present, must have type ``T`` (no ``None`` allowed)
* ``NotRequiredColumn[Optional[T]]``: Column can be missing, and if present, can contain ``None`` values

.. _typeddict-and-dataclass-schemas:

TypedDict and Dataclass Schemas
-------------------------------

A schema can also be a ``TypedDict`` or a dataclass, so that the type of a row
(``df.to_dict("records")`` in pandas, ``df.to_dicts()`` in polars) can describe the whole
DataFrame. They follow the same rules as :ref:`TypedDict and dataclass structs
<typeddict-and-dataclass-structs>`:

* In a ``TypedDict``, a key marked with ``NotRequired`` (or left unmarked in a ``total=False``
  ``TypedDict``) is a column that may be missing. ``NotRequiredColumn`` cannot be used in a
  ``TypedDict``.
* The columns of a dataclass are its fields: ``ClassVar`` and ``InitVar`` are not columns.
  Every field must be present, including fields with a default value. ``NotRequiredColumn``
  marks a column that may be missing, as in a ``Protocol``.

.. code-block:: python

   from dataclasses import dataclass
   from typing import ClassVar, TypedDict

   import pandas as pd
   from typing_extensions import NotRequired

   from pavise.pandas import DataFrame

   class UserRow(TypedDict):
       user_id: int
       email: NotRequired[str]   # may be missing

   @dataclass
   class Event:
       event_id: int
       note: str | None = None   # must be present, may be null
       source: ClassVar[str] = "web"   # not a column

   DataFrame[UserRow](pd.DataFrame({"user_id": [1]}))
   DataFrame[Event](pd.DataFrame({"event_id": [1], "note": [None]}))

A dataclass can declare unique keys and checks with ``UniqueKeys`` and ``Checked`` (see
:doc:`dataframe-checks`). A ``TypedDict`` cannot have them as bases, so use
``@dataframe_check`` for a ``TypedDict``, or a ``Protocol`` or dataclass when the constraints
should be part of the type.

Type checkers compare these schemas by their own rules:

* A ``TypedDict`` with more keys is accepted where one with fewer keys is expected, but a
  required key does not match a ``NotRequired`` one, and a ``TypedDict`` does not match a
  ``Protocol``.
* A dataclass is accepted where one of its base classes or a ``Protocol`` with the same
  columns is expected, but not where another dataclass with the same fields is expected.

Supported Types
---------------

Basic Types
~~~~~~~~~~~

* ``int``: Integer values
* ``float``: Floating point values
* ``str``: String values
* ``bool``: Boolean values

Datetime Types
~~~~~~~~~~~~~~

* ``datetime.datetime``: Date and time values
* ``datetime.date``: Date-only values
* ``datetime.timedelta``: Time duration values

Generic Types
~~~~~~~~~~~~~

* ``Optional[T]``: Nullable types (see "Handling Optional Columns" above)
* ``Literal[...]``: Restricts values to specific literals

The ``Literal`` type is useful for columns that should only contain specific values:

.. example-raises: ValidationError

.. code-block:: python

   from typing import Literal, Protocol

   class OrderSchema(Protocol):
       order_id: int
       status: Literal["pending", "approved", "rejected"]
       priority: Literal[1, 2, 3]

   # Valid data
   df = pd.DataFrame({
       "order_id": [1, 2, 3],
       "status": ["pending", "approved", "rejected"],
       "priority": [1, 2, 3]
   })
   validated_df = DataFrame[OrderSchema](df)  # OK

   # Invalid data
   df_invalid = pd.DataFrame({
       "order_id": [1],
       "status": ["invalid"],  # Not in Literal values
       "priority": [1]
   })
   DataFrame[OrderSchema](df_invalid)  # ValidationError

pandas ExtensionDtype
~~~~~~~~~~~~~~~~~~~~~

pandas-specific extension dtypes can be used directly:

.. code-block:: python

   import pandas as pd

   class Schema(Protocol):
       category: pd.CategoricalDtype
       value: pd.Int64Dtype

polars DataType
~~~~~~~~~~~~~~~

polars-specific data types can be used directly:

.. code-block:: python

   import polars as pl

   class Schema(Protocol):
       category: pl.Categorical
       value: pl.Int64

Creating Empty DataFrames
--------------------------

You can create an empty DataFrame that conforms to your schema using the ``make_empty()`` class method.
This is useful for initializing DataFrames, creating templates, or testing.

pandas Backend
~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: int

   # Create an empty DataFrame with the correct schema
   empty_df = DataFrame[UserSchema].make_empty()

   # Result: Empty DataFrame with columns [user_id, name, age]
   # - len(empty_df) == 0
   # - Columns have correct dtypes (int64, object, int64)

polars Backend
~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Protocol
   from pavise.polars import DataFrame

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: int

   # Create an empty DataFrame with the correct schema
   empty_df = DataFrame[UserSchema].make_empty()

   # Result: Empty DataFrame with columns [user_id, name, age]
   # - len(empty_df) == 0
   # - Columns have correct dtypes (Int64, Utf8, Int64)

Supported Features
~~~~~~~~~~~~~~~~~~

The ``make_empty()`` method supports all schema features:

* Basic types (int, float, str, bool)
* Datetime types (datetime, date, timedelta)
* Optional types (``Optional[T]``)
* NotRequired columns (``NotRequiredColumn[T]`` - included as empty columns)
* Literal types (uses base type)
* Annotated types with validators (uses base type, validators not applied)
* Backend-specific types (pandas ExtensionDtype, polars DataType)
