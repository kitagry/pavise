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

Differences from pandas Backend
--------------------------------

1. **Nullable types**: polars types are nullable by default, pandas are not
2. **Type system**: polars has a richer type system (e.g., Categorical, Utf8)
3. **Performance**: polars validation is generally faster
4. **Index**: polars doesn't have an index concept, so ``__index__`` validation is not supported

Method Chaining
---------------

polars preserves immutability, but type information is still lost:

.. code-block:: python

   validated_df = DataFrame[UserSchema](df)

   # Type information is lost after polars operations
   result = validated_df.group_by("age").mean()  # result is not DataFrame[UserSchema]

   # Re-validate if needed
   revalidated = DataFrame[ResultSchema](result)
