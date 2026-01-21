pandas Backend
==============

The pandas backend provides validation for pandas DataFrames.

Installation
------------

.. code-block:: bash

   pip install pavise[pandas]

Basic Usage
-----------

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame
   import pandas as pd

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: int

   # Create a pandas DataFrame
   df = pd.DataFrame({
       "user_id": [1, 2, 3],
       "name": ["Alice", "Bob", "Charlie"],
       "age": [25, 30, 35]
   })

   # Validate
   validated_df = DataFrame[UserSchema](df)

Type Mapping
------------

Pavise maps Python types to pandas dtypes:

================  =====================
Python Type       pandas dtype
================  =====================
``int``           int64
``float``         float64
``str``           object (str)
``bool``          bool
``datetime``      datetime64[ns]
``date``          datetime64[ns]
``timedelta``     timedelta64[ns]
``Optional[T]``   Nullable version of T
================  =====================

pandas ExtensionDtype
---------------------

You can use pandas extension dtypes directly:

.. code-block:: python

   import pandas as pd

   class Schema(Protocol):
       category: pd.CategoricalDtype
       nullable_int: pd.Int64Dtype
       string: pd.StringDtype

   validated_df = DataFrame[Schema](df)

This gives you more control over the exact dtype used.

Index Validation
----------------

Validate the index type using the special ``__index__`` attribute:

.. code-block:: python

   from typing import Protocol

   class Schema(Protocol):
       __index__: int  # Validates index is int64
       value: float

   # Create DataFrame with integer index
   df = pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=[0, 1, 2])

   validated_df = DataFrame[Schema](df)

Named Index Validation
^^^^^^^^^^^^^^^^^^^^^^

Use ``Annotated`` to validate both the index type and name:

.. code-block:: python

   from typing import Protocol, Annotated

   class UserSchema(Protocol):
       __index__: Annotated[int, "user_id"]  # Validates type AND name
       username: str
       score: float

   # Create DataFrame with named index
   df = pd.DataFrame(
       {"username": ["alice", "bob"], "score": [95.0, 87.0]},
       index=pd.Index([1, 2], name="user_id")
   )

   validated_df = DataFrame[UserSchema](df)

   # This will fail - wrong index name
   df_wrong = pd.DataFrame(
       {"username": ["alice"], "score": [95.0]},
       index=pd.Index([1], name="id")  # Expected "user_id"
   )
   # ValidationError: Index name expected 'user_id', got 'id'

MultiIndex Validation
^^^^^^^^^^^^^^^^^^^^^

For MultiIndex, use a tuple of types with a tuple of names:

.. code-block:: python

   from typing import Protocol, Annotated

   class RegionalSalesSchema(Protocol):
       __index__: Annotated[tuple[str, int], ("region", "user_id")]
       sales: float
       quantity: int

   # Create DataFrame with MultiIndex
   df = pd.DataFrame(
       {"sales": [100.0, 200.0, 150.0], "quantity": [5, 10, 7]},
       index=pd.MultiIndex.from_tuples(
           [("East", 1), ("East", 2), ("West", 1)],
           names=["region", "user_id"]
       )
   )

   validated_df = DataFrame[RegionalSalesSchema](df)

Nullable Types
--------------

pandas handles nullable integers specially:

.. code-block:: python

   from typing import Optional

   class Schema(Protocol):
       value: Optional[int]

   # pandas converts int to float64 when there are nulls
   df = pd.DataFrame({"value": [1, 2, None]})  # dtype: float64
   validated_df = DataFrame[Schema](df)

For true nullable integers, use ``pd.Int64Dtype``:

.. code-block:: python

   class Schema(Protocol):
       value: pd.Int64Dtype

   df = pd.DataFrame({"value": pd.array([1, 2, None], dtype=pd.Int64Dtype())})
   validated_df = DataFrame[Schema](df)

Method Chaining
---------------

Note: pandas method chaining may lose Pavise type information:

.. code-block:: python

   validated_df = DataFrame[UserSchema](df)

   # Type information is lost after pandas operations
   result = validated_df.groupby("age").mean()  # result is not DataFrame[UserSchema]

   # Re-validate if needed
   revalidated = DataFrame[ResultSchema](result)

Performance Considerations
--------------------------

Validation checks all rows for type correctness, which can be slow for large DataFrames.
For performance-critical code:

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
