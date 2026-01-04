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

Use ``Optional[T]`` for nullable columns:

.. code-block:: python

   from typing import Optional

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: Optional[int]  # Allows None values

Note: In pandas, nullable integers are stored as ``float64`` when they contain nulls.
In polars, all types are nullable by default.

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
