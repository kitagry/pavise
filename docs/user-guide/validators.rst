Validators
==========

Pavise provides validators to enforce data quality constraints beyond type checking.
Validators are attached to column types using ``typing.Annotated``.

Available Validators
--------------------

Range
~~~~~

Validates that numeric values are within a specified range.

.. code-block:: python

   from typing import Annotated, Protocol
   from pavise.validators import Range
   from pavise.pandas import DataFrame

   class Schema(Protocol):
       age: Annotated[int, Range(0, 150)]
       score: Annotated[float, Range(0.0, 100.0)]

   validated_df = DataFrame[Schema](df)

Error message example:

.. code-block:: text

   ValueError: Column 'age': values must be in range [0, 150]

   Sample invalid values (showing first 3 of 5):
     Row 1: 200
     Row 3: -5
     Row 5: 300

Unique
~~~~~~

Validates that column values are unique (no duplicates).

.. code-block:: python

   from pavise.validators import Unique

   class Schema(Protocol):
       user_id: Annotated[int, Unique()]
       email: Annotated[str, Unique()]

Error message example:

.. code-block:: text

   ValueError: Column 'user_id': contains duplicate values

   Sample duplicate values (showing first 2):
     Value 2 at rows: [1, 3]
     Value 5 at rows: [2, 4]

In
~~

Validates that column values are within a set of allowed values.

.. code-block:: python

   from pavise.validators import In

   class Schema(Protocol):
       status: Annotated[str, In(["pending", "approved", "rejected"])]
       priority: Annotated[int, In([1, 2, 3, 4, 5])]

Error message example:

.. code-block:: text

   ValueError: Column 'status': contains values not in allowed values

   Sample invalid values (showing first 2 of 3):
     Row 1: 'invalid'
     Row 3: 'bad'

Regex
~~~~~

Validates that string values match a regular expression pattern.

.. code-block:: python

   from pavise.validators import Regex

   class Schema(Protocol):
       email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
       phone: Annotated[str, Regex(r'^\d{3}-\d{4}-\d{4}$')]

Error message example:

.. code-block:: text

   ValueError: Column 'email': contains values that don't match the pattern

   Sample invalid values (showing first 2 of 4):
     Row 1: 'invalid'
     Row 3: 'bad@'

MinLen / MaxLen
~~~~~~~~~~~~~~~

Validates string length constraints.

.. code-block:: python

   from pavise.validators import MinLen, MaxLen

   class Schema(Protocol):
       username: Annotated[str, MinLen(3), MaxLen(20)]
       password: Annotated[str, MinLen(8)]

Error message example:

.. code-block:: text

   ValueError: Column 'username': contains strings shorter than minimum length

   Sample invalid values (showing first 2 of 3):
     Row 0: 'ab' (length: 2)
     Row 2: 'x' (length: 1)

Custom
~~~~~~

Create custom validators for business-specific logic.

.. code-block:: python

   from pavise.validators import Custom

   def is_positive(value) -> bool:
       return value > 0

   def is_business_day(value) -> bool:
       return value.weekday() < 5

   class Schema(Protocol):
       amount: Annotated[int, Custom(is_positive, "must be positive")]
       date: Annotated[datetime.date, Custom(is_business_day, "must be a business day")]

Error message example:

.. code-block:: text

   ValueError: Column 'amount': must be positive

   Sample invalid values (showing first 3 of 5):
     Row 1: -100
     Row 3: 0
     Row 7: -50

Combining Multiple Validators
------------------------------

You can attach multiple validators to a single column:

.. code-block:: python

   from typing import Annotated, Protocol
   from pavise.validators import Range, Custom

   def is_even(value) -> bool:
       return value % 2 == 0

   class Schema(Protocol):
       score: Annotated[int, Range(0, 100), Custom(is_even, "must be even")]

All validators are checked in order, and the first failure is reported.

Performance Considerations
--------------------------

Validators are only executed during runtime validation:

.. code-block:: python

   # No validation, no performance cost
   def process(df: DataFrame[Schema]) -> DataFrame[Schema]:
       return df

   # Validation happens here
   validated_df = DataFrame[Schema](raw_df)

   # No further validation cost
   result = process(validated_df)

For large DataFrames, consider validating only at system boundaries and relying on type checking for internal functions.
