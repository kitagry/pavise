Error Messages
==============

Pavise provides detailed error messages to help you quickly identify and fix validation issues.

Type Errors
-----------

When a column has the wrong type, Pavise shows:

* Expected type and actual type
* Sample invalid values (first 5 maximum)
* Row numbers for each invalid value

If the column's dtype is rejected but every value would pass on its own (for
example, an ``object`` column holding only ints for an ``int`` annotation), the
error has no sample section (see the second example below).

Example
~~~~~~~

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame
   from pavise.exceptions import ValidationError
   import pandas as pd

   class Schema(Protocol):
       age: int

   df = pd.DataFrame({"age": [25, "invalid", 30, 35, "bad", 40]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'age': expected int, got object

   Sample invalid values (showing first 2 of 2):
     Row 1: 'invalid'
     Row 4: 'bad'

When every value has the expected type but the dtype does not match, only the
first line is shown:

.. code-block:: python

   df = pd.DataFrame({"age": [25, 30, 35]}, dtype=object)
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'age': expected int, got object

Missing Columns
---------------

When a required column is missing:

.. code-block:: python

   class Schema(Protocol):
       user_id: int
       name: str

   df = pd.DataFrame({"user_id": [1, 2, 3]})  # Missing 'name'
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'name': missing

Validator Errors
----------------

Each validator provides detailed error messages.

Range Validator
~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Annotated
   from pavise.validators import Range

   class Schema(Protocol):
       age: Annotated[int, Range(0, 150)]

   df = pd.DataFrame({"age": [25, 200, 30, -5, 35, 300]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'age': values must be in range [0, 150]

   Sample invalid values (showing first 3 of 3):
     Row 1: 200
     Row 3: -5
     Row 5: 300

Comparison Validators (Ge / Gt / Le / Lt)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from typing import Annotated
   from pavise.validators import Ge

   class Schema(Protocol):
       price: Annotated[int, Ge(0)]

   df = pd.DataFrame({"price": [10, -1, 30, -20]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'price': values must be >= 0

   Sample invalid values (showing first 2 of 2):
     Row 1: -1
     Row 3: -20

``Gt``, ``Le`` and ``Lt`` report ``values must be > {value}``, ``<= {value}`` and ``< {value}`` respectively.

Unique Validator
~~~~~~~~~~~~~~~~

.. code-block:: python

   from pavise.validators import Unique

   class Schema(Protocol):
       user_id: Annotated[int, Unique()]

   df = pd.DataFrame({"user_id": [1, 2, 2, 3, 5, 5, 5]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'user_id': contains duplicate values

   Sample invalid values (showing first 5 of 5):
     Row 1: 2
     Row 2: 2
     Row 4: 5
     Row 5: 5
     Row 6: 5

Every row holding a duplicated value is reported, including the first occurrence.

In Validator
~~~~~~~~~~~~

.. code-block:: python

   from pavise.validators import In

   class Schema(Protocol):
       status: Annotated[str, In(["pending", "approved", "rejected"])]

   df = pd.DataFrame({"status": ["pending", "invalid", "approved", "bad"]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'status': contains values not in allowed values

   Sample invalid values (showing first 2 of 2):
     Row 1: 'invalid'
     Row 3: 'bad'

Regex Validator
~~~~~~~~~~~~~~~

.. code-block:: python

   from pavise.validators import Regex

   class Schema(Protocol):
       email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]

   df = pd.DataFrame({"email": ["alice@example.com", "invalid", "bob@test.com", "bad@"]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'email': contains values that don't match the pattern

   Sample invalid values (showing first 2 of 2):
     Row 1: 'invalid'
     Row 3: 'bad@'

MinLen/MaxLen Validators
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pavise.validators import MinLen

   class Schema(Protocol):
       username: Annotated[str, MinLen(3)]

   df = pd.DataFrame({"username": ["alice", "ab", "bob", "x"]})
   try:
       validated_df = DataFrame[Schema](df)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Column 'username': contains strings shorter than minimum length

   Sample invalid values (showing first 2 of 2):
     Row 1: 'ab' (length: 2)
     Row 3: 'x' (length: 1)

Strict Mode Errors
------------------

When strict mode is enabled and extra columns are present:

.. code-block:: python

   class Schema(Protocol):
       user_id: int
       name: str

   df = pd.DataFrame({
       "user_id": [1, 2, 3],
       "name": ["Alice", "Bob", "Charlie"],
       "age": [25, 30, 35],  # Extra column
       "email": ["a@test.com", "b@test.com", "c@test.com"]  # Extra column
   })

   try:
       validated_df = DataFrame[Schema](df, strict=True)
   except ValidationError as e:
       print(e)

Output:

.. code-block:: text

   Strict mode: unexpected columns ['age', 'email']

Performance Notes
-----------------

To avoid overwhelming output and maintain performance:

* Error messages show at most 5 sample invalid values, along with the total
  number of invalid values
* Duplicate detection counts each row holding a duplicated value, so the 5
  samples may come from fewer than 5 distinct values

For large DataFrames, consider sampling before validation during development.
