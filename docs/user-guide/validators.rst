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

   ValidationError: Column 'age': values must be in range [0, 150]

   Sample invalid values (showing first 3 of 3):
     Row 1: 200
     Row 3: -5
     Row 5: 300

Ge / Gt / Le / Lt
~~~~~~~~~~~~~~~~~

Validate one-sided numeric bounds. Use these instead of ``Range(0, float("inf"))``.

- ``Ge(value)``: values must be greater than or equal to ``value``
- ``Gt(value)``: values must be strictly greater than ``value``
- ``Le(value)``: values must be less than or equal to ``value``
- ``Lt(value)``: values must be strictly less than ``value``

.. note::

   ``Range``, ``Ge``, ``Gt``, ``Le`` and ``Lt`` require a numeric (integer, float, Decimal) or
   temporal (date, datetime, timedelta) column. Applying them to any other column, including
   ``bool`` and string columns, raises ``ValidationError``
   (for example ``Column 'name': Ge requires a numeric or temporal column, got object``) on both
   backends. A bound that cannot be compared with the column, such as ``Ge(0)`` on a datetime
   column, also raises ``ValidationError``.

.. code-block:: python

   from typing import Annotated, Optional, Protocol
   from pavise.validators import Ge, Gt, Le, Lt
   from pavise.pandas import DataFrame

   class Schema(Protocol):
       price: Annotated[float, Ge(0)]
       quantity: Annotated[int, Gt(0)]
       discount: Annotated[float, Ge(0), Lt(1)]
       percentage: Annotated[Optional[float], Le(100)]

   validated_df = DataFrame[Schema](df)

As with ``Range``, null values in ``Optional`` columns are not checked.

Error message example:

.. code-block:: text

   ValidationError: Column 'price': values must be >= 0

   Sample invalid values (showing first 2 of 2):
     Row 1: -1.0
     Row 3: -20.0

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

   ValidationError: Column 'user_id': contains duplicate values

   Sample invalid values (showing first 4 of 4):
     Row 1: np.int64(2)
     Row 2: np.int64(5)
     Row 3: np.int64(2)
     Row 4: np.int64(5)

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

   ValidationError: Column 'status': contains values not in allowed values

   Sample invalid values (showing first 3 of 3):
     Row 1: 'invalid'
     Row 3: 'bad'
     Row 4: 'nope'

Regex
~~~~~

Validates that string values match a regular expression pattern.

The **whole string** must match the pattern (equivalent to Python's ``re.fullmatch``),
on both the pandas and polars backends, including pyarrow-backed pandas string columns.
``^`` and ``$`` anchors are therefore optional: ``Regex(r'[a-z]+')`` rejects ``'abc123'``.

The regex syntax follows the engine that runs it: Python's ``re`` for object and
``string[python]`` columns, RE2 for pyarrow-backed pandas columns, and the Rust ``regex``
crate for polars. Features such as lookarounds and backreferences are therefore only
available on Python ``re``.

.. note::

   In pavise 0.1.7 and earlier, the pandas backend only required the pattern to match at the
   start of the string (``re.match``), so ``Regex(r'[a-z]+')`` accepted ``'abc123'``.
   The polars backend anchored the pattern without grouping it, so a top-level alternation
   such as ``Regex(r'a|b')`` accepted ``'ab'``.
   If you relied on prefix matching, append ``.*`` to the pattern (e.g. ``Regex(r'[a-z]+.*')``).

.. code-block:: python

   from pavise.validators import Regex

   class Schema(Protocol):
       email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
       phone: Annotated[str, Regex(r'^\d{3}-\d{4}-\d{4}$')]

Error message example:

.. code-block:: text

   ValidationError: Column 'email': contains values that don't match the pattern

   Sample invalid values (showing first 2 of 2):
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

   ValidationError: Column 'username': contains strings shorter than minimum length

   Sample invalid values (showing first 2 of 2):
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

   ValidationError: Column 'amount': must be positive

   Sample invalid values (showing first 3 of 3):
     Row 1: np.int64(-100)
     Row 3: np.int64(0)
     Row 7: np.int64(-50)

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
