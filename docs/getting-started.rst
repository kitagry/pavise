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

Next Steps
----------

* Learn about :doc:`user-guide/validators` for data quality checks
* Explore :doc:`user-guide/strict-mode` to reject extra columns
* Check :doc:`examples/index` for real-world use cases
