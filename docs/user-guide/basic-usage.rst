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

Runtime Validation
~~~~~~~~~~~~~~~~~~

At system boundaries (loading from CSV, database, API), validate explicitly:

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

Backend Selection
-----------------

pandas Backend
~~~~~~~~~~~~~~

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
