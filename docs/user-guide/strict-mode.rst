Strict Mode
===========

By default, Pavise ignores extra columns not defined in the schema. This follows the philosophy of structural subtyping: a DataFrame with more columns can be used where fewer are expected.

However, in some cases you may want to reject DataFrames with unexpected columns. This is where strict mode comes in.

Enabling Strict Mode
--------------------

Pass ``strict=True`` to the DataFrame constructor:

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame

   class UserSchema(Protocol):
       user_id: int
       name: str

   # This will fail if df has columns other than user_id and name
   validated_df = DataFrame[UserSchema](df, strict=True)

Error Message
-------------

If the DataFrame contains extra columns, you'll get a clear error:

.. code-block:: text

   ValueError: Strict mode: unexpected columns ['age', 'email', 'address']

Use Cases
---------

Strict mode is useful when:

1. **Enforcing exact schemas**: You want to ensure the DataFrame has exactly the columns you expect
2. **Detecting typos**: Extra columns might indicate typos in column names
3. **API contracts**: You're receiving data from an external source and want to enforce a strict contract

Example: API Data Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame
   import pandas as pd

   class APIResponseSchema(Protocol):
       id: int
       timestamp: datetime.datetime
       value: float

   # Validate API response strictly
   response_data = fetch_from_api()
   df = pd.DataFrame(response_data)

   # Fail if API returns unexpected columns
   validated_df = DataFrame[APIResponseSchema](df, strict=True)

When Not to Use Strict Mode
----------------------------

Avoid strict mode when:

1. **Internal processing**: For internal functions, extra columns are usually harmless
2. **Data pipelines**: Intermediate steps may add temporary columns
3. **Flexibility needed**: You want to allow DataFrames to have additional context

Example: Flexible Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class MinimalSchema(Protocol):
       user_id: int
       name: str

   # Allow extra columns for flexibility
   def process_users(df: DataFrame[MinimalSchema]) -> DataFrame[MinimalSchema]:
       # df might have age, email, etc. - that's OK
       # We only care about user_id and name
       return df

   # Don't use strict=True here
   validated_df = DataFrame[MinimalSchema](df)  # Extra columns are ignored
   result = process_users(validated_df)

Combining with Validators
--------------------------

Strict mode works with validators:

.. code-block:: python

   from typing import Annotated
   from pavise.validators import Range

   class StrictSchema(Protocol):
       age: Annotated[int, Range(0, 150)]
       score: Annotated[float, Range(0.0, 100.0)]

   # Both type validation, validators, and column strictness are enforced
   validated_df = DataFrame[StrictSchema](df, strict=True)
