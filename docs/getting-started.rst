Getting Started
===============

Installation
------------

Pavise requires Python 3.9 or later. Install it via pip with your preferred backend:

For pandas backend:

.. code-block:: bash

   pip install pavise[pandas]

For polars backend:

.. code-block:: bash

   pip install pavise[polars]

For both backends:

.. code-block:: bash

   pip install pavise[all]

Basic Usage
-----------

Define a Schema
~~~~~~~~~~~~~~~

Define your DataFrame schema using Python's Protocol:

.. code-block:: python

   from typing import Protocol

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: int
       email: str

Static Type Checking
~~~~~~~~~~~~~~~~~~~~

Use the schema for static type checking with mypy, pyright, or other type checkers:

.. code-block:: python

   from pavise.pandas import DataFrame

   def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
       # Type checker validates the schema
       # No runtime overhead
       return df

Runtime Validation
~~~~~~~~~~~~~~~~~~

Validate DataFrames at runtime, typically at system boundaries:

.. code-block:: python

   import pandas as pd
   from pavise.pandas import DataFrame
   from pavise.exceptions import ValidationError

   # Load data from external source
   raw_df = pd.read_csv("users.csv")

   # Validate at system boundary
   try:
       validated_df = DataFrame[UserSchema](raw_df)
   except ValidationError as e:
       print(f"Validation failed: {e}")

If validation fails, you'll get a detailed error message from ``ValidationError``:

.. code-block:: text

   Validation failed: Column 'age': expected int, got object

   Sample invalid values (showing first 3 of 10):
     Row 1: 'invalid' (str)
     Row 5: None (NoneType)
     Row 8: 200.5 (float)

Using Validators
~~~~~~~~~~~~~~~~

Add validators using ``typing.Annotated``:

.. code-block:: python

   from typing import Annotated
   from pavise.validators import Range, Regex

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: Annotated[int, Range(0, 150)]
       email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]

   # Runtime validation with validators
   validated_df = DataFrame[UserSchema](raw_df)

Using Literal Types
~~~~~~~~~~~~~~~~~~~

Restrict column values to specific literals using ``Literal``:

.. code-block:: python

   from typing import Literal, Protocol

   class OrderSchema(Protocol):
       order_id: int
       status: Literal["pending", "approved", "rejected"]
       priority: Literal[1, 2, 3]

   # Only these exact values are allowed
   validated_df = DataFrame[OrderSchema](raw_df)

Optional Columns
~~~~~~~~~~~~~~~~

Use ``NotRequiredColumn[T]`` for columns that may not exist in the DataFrame:

.. code-block:: python

   from typing import Optional, Protocol
   from pavise.pandas import DataFrame, NotRequiredColumn

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: NotRequiredColumn[int]  # Column can be missing
       email: NotRequiredColumn[Optional[str]]  # Column can be missing, or contain None

   # Both of these are valid
   df1 = pd.DataFrame({"user_id": [1], "name": ["Alice"]})  # age and email missing
   df2 = pd.DataFrame({"user_id": [1], "name": ["Alice"], "age": [25]})  # only email missing

   validated_df1 = DataFrame[UserSchema](df1)  # OK
   validated_df2 = DataFrame[UserSchema](df2)  # OK

Note: ``NotRequiredColumn[T]`` means the column is optional, while ``Optional[T]`` means the column can contain ``None`` values. Use ``NotRequiredColumn[Optional[T]]`` for columns that are both optional and nullable.

Next Steps
----------

* Learn about :doc:`user-guide/validators` for data quality checks
* Explore :doc:`user-guide/strict-mode` to reject extra columns
* Check :doc:`user-guide/polars-backend` for polars DataFrame and LazyFrame support
* Check :doc:`examples/index` for real-world use cases
