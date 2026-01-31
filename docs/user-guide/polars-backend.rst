polars Backend
==============

The polars backend provides validation for polars DataFrames.

Installation
------------

.. code-block:: bash

   pip install pavise[polars]

Basic Usage
-----------

.. code-block:: python

   from typing import Protocol
   from pavise.polars import DataFrame
   import polars as pl

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: int

   # Create a polars DataFrame
   df = pl.DataFrame({
       "user_id": [1, 2, 3],
       "name": ["Alice", "Bob", "Charlie"],
       "age": [25, 30, 35]
   })

   # Validate
   validated_df = DataFrame[UserSchema](df)

Type Mapping
------------

Pavise maps Python types to polars dtypes:

================  =====================
Python Type       polars dtype
================  =====================
``int``           Int64
``float``         Float64
``str``           Utf8
``bool``          Boolean
``datetime``      Datetime
``date``          Date
``timedelta``     Duration
``Optional[T]``   Nullable version of T
================  =====================

polars DataType
---------------

You can use polars data types directly:

.. code-block:: python

   import polars as pl

   class Schema(Protocol):
       category: pl.Categorical
       value: pl.Int64
       text: pl.Utf8

   validated_df = DataFrame[Schema](df)

This gives you precise control over the polars dtype.

Nullable Types
--------------

Unlike pandas, polars types are nullable by default:

.. code-block:: python

   from typing import Optional

   class Schema(Protocol):
       value: Optional[int]  # Allows null values

   df = pl.DataFrame({"value": [1, 2, None]})  # dtype: Int64 (nullable)
   validated_df = DataFrame[Schema](df)

For non-nullable columns, don't use Optional:

.. code-block:: python

   class Schema(Protocol):
       value: int  # No nulls allowed

   df = pl.DataFrame({"value": [1, 2, None]})
   validated_df = DataFrame[Schema](df)  # Raises ValueError

Performance Considerations
--------------------------

polars is designed for performance, and Pavise validation is fast on polars DataFrames.
However, the same principles apply:

1. Validate once at system boundaries
2. Use type annotations without validation for internal functions
3. Trust the type system after initial validation

.. code-block:: python

   # Validate once
   validated_df = DataFrame[UserSchema](raw_df)

   # No validation overhead in internal functions
   def process(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
       return df

   result = process(validated_df)

LazyFrame Support
-----------------

Pavise also supports polars LazyFrame for lazy evaluation workflows:

.. code-block:: python

   from pavise.polars import LazyFrame, DataFrame

   class UserSchema(Protocol):
       user_id: int
       name: str

   # Wrap a LazyFrame with schema validation
   lf = pl.scan_csv("users.csv")
   validated_lf = LazyFrame[UserSchema](lf)

   # Schema is validated immediately (column existence and types)
   # Value-based validators are applied on collect()
   df: DataFrame[UserSchema] = validated_lf.collect()

LazyFrame Validation Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LazyFrame validation happens in two stages:

1. **On construction**: Schema-level validation (column existence and types) using ``collect_schema()``
2. **On collect()**: Full validation including value-based validators (Range, Unique, etc.)

.. code-block:: python

   from typing import Annotated
   from pavise.validators import Range

   class UserSchema(Protocol):
       user_id: int
       age: Annotated[int, Range(0, 150)]

   lf = pl.LazyFrame({"user_id": [1, 2], "age": [25, 200]})

   # Schema validation passes (types are correct)
   validated_lf = LazyFrame[UserSchema](lf)

   # Range validation fails on collect()
   df = validated_lf.collect()  # Raises ValidationError: age out of range

Creating Empty LazyFrames
~~~~~~~~~~~~~~~~~~~~~~~~~

Like DataFrame, LazyFrame supports ``make_empty()``:

.. code-block:: python

   empty_lf = LazyFrame[UserSchema].make_empty()
   # Returns LazyFrame with correct schema but no rows

Differences from pandas Backend
--------------------------------

1. **Nullable types**: polars types are nullable by default, pandas are not
2. **Type system**: polars has a richer type system (e.g., Categorical, Utf8)
3. **Performance**: polars validation is generally faster
4. **Index**: polars doesn't have an index concept, so ``__index__`` validation is not supported
5. **LazyFrame**: polars backend supports LazyFrame for lazy evaluation workflows

Method Chaining
---------------

polars preserves immutability, but type information is still lost:

.. code-block:: python

   validated_df = DataFrame[UserSchema](df)

   # Type information is lost after polars operations
   result = validated_df.group_by("age").mean()  # result is not DataFrame[UserSchema]

   # Re-validate if needed
   revalidated = DataFrame[ResultSchema](result)
