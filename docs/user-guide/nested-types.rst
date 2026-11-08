Nested Types
============

Columns can hold lists and structs. Write a list column as ``list[X]`` and a struct column
as a nested ``Protocol``:

.. code-block:: python

   from typing import Protocol

   import polars as pl
   from pavise.polars import DataFrame

   class Address(Protocol):
       city: str
       zip: str

   class UserSchema(Protocol):
       tags: list[str]                     # List
       address: Address                    # Struct
       history: list[Address]              # List of Structs
       nickname: list[str | None] | None   # Optional elements in an Optional column

   df = pl.DataFrame({
       "tags": [["admin", "staff"], []],
       "address": [{"city": "Tokyo", "zip": "100-0001"}, {"city": "Osaka", "zip": "530-0001"}],
       "history": [[], [{"city": "Kyoto", "zip": "600-8001"}]],
       "nickname": [["Al", None], None],
   })
   validated_df = DataFrame[UserSchema](df)

Pavise checks each level of a nested column:

* The dtype of the column, the list elements and the struct fields
* That every field of the nested ``Protocol`` exists in the struct. Extra struct fields are
  allowed, unless ``strict=True`` is passed (see :doc:`strict-mode`)
* That the column, the list elements and the struct fields hold no nulls unless they are
  ``Optional``. A null struct row is checked as the column's value only, so the fields of a
  null struct are not reported as null
* That list elements and struct fields annotated with ``Literal`` hold only its values

Mark a struct field that may be absent with ``NotRequiredColumn``:

.. code-block:: python

   from pavise.types import NotRequiredColumn

   class PartialAddress(Protocol):
       city: str
       zip: NotRequiredColumn[str]

Validators
----------

Validators apply to the type they are written on. ``list[Annotated[int, Range(0, 100)]]``
checks every element, ``Annotated[list[int], MinLen(1)]`` checks the list itself, and a
validator on a field of the nested ``Protocol`` checks that field of every struct:

.. code-block:: python

   from typing import Annotated
   from pavise.exceptions import ValidationError
   from pavise.validators import Custom, MinLen, Range, Regex

   class CheckedAddress(Protocol):
       city: str
       zip: Annotated[str, Regex(r"\d{3}-\d{4}")]

   class OrderSchema(Protocol):
       scores: list[Annotated[int, Range(0, 100)]]   # each element
       items: Annotated[list[int], MinLen(1)]        # the list's length
       history: list[CheckedAddress]                 # each struct's zip

   df = pl.DataFrame({
       "scores": [[50], [50, 120]],
       "items": [[1], [2, 3]],
       "history": [[{"city": "Tokyo", "zip": "100-0001"}], []],
   })
   try:
       DataFrame[OrderSchema](df)
   except ValidationError as e:
       print(e)

Output:

.. example-output

.. code-block:: text

   Column 'scores[]': values must be in range [0, 100]

   Sample invalid values (showing first 1 of 1):
     Row 1: 120 (in [50, 120])

Each sample shows the rejected value and, in parentheses, the whole row it was found in.

On a list itself, only ``MinLen`` and ``MaxLen`` (which check the number of elements) and
``Custom`` can be used. On a struct itself, only ``Custom`` can be used. ``Custom`` receives the
row's value as a Python ``list`` or ``dict``:

.. code-block:: python

   class TotalSchema(Protocol):
       items: Annotated[list[int], Custom(lambda items: sum(items) <= 10, "sum must be <= 10")]

``Unique`` cannot be used inside list and struct columns, since it would be unclear whether
values must be unique within a list or across rows. Pavise raises ``ValidationError`` for these
when it checks the schema, instead of skipping them silently.

``@dataframe_check`` checks run on whole DataFrames, so checks attached to a ``Protocol`` used as
a struct are not run on the struct values. They still run when the ``Protocol`` is used as a
DataFrame schema itself.

Supported Dtypes
----------------

.. list-table::
   :header-rows: 1

   * - Annotation
     - polars
     - pandas
   * - ``list[X]``
     - ``pl.List`` or ``pl.Array``
     - ``pd.ArrowDtype`` of a pyarrow list, large list or fixed size list
   * - Nested ``Protocol``
     - ``pl.Struct``
     - ``pd.ArrowDtype`` of a pyarrow struct

A list whose element dtype is ``Null`` (for example ``pl.List(pl.Null)``, which polars infers
from empty lists) is accepted for any element type.

pandas supports nested columns only as ``pd.ArrowDtype`` (pyarrow is required). An ``object``
column holding Python lists or dicts is rejected:

.. example-setup

   import pandas as pd

.. code-block:: python

   from pavise.pandas import DataFrame as PandasDataFrame
   from pavise.exceptions import ValidationError

   class TagsSchema(Protocol):
       tags: list[str]

   try:
       PandasDataFrame[TagsSchema](pd.DataFrame({"tags": [["a"], ["b"]]}))
   except ValidationError as e:
       print(e)

Output:

.. example-output

.. code-block:: text

   Column 'tags': expected list[str], got object
     (pavise checks list and struct columns only with pd.ArrowDtype, e.g. dtype=pd.ArrowDtype(pa.list_(pa.string())))

Store the column as ``pd.ArrowDtype`` instead:

.. code-block:: python

   import pyarrow as pa

   df = pd.DataFrame(
       {"tags": pd.Series([["a"], ["b"]], dtype=pd.ArrowDtype(pa.list_(pa.string())))}
   )
   validated_df = PandasDataFrame[TagsSchema](df)

Error Messages
--------------

An error inside a nested column names where it is, with ``[]`` for list elements and
``.name`` for struct fields. ``ValidationError.column_name`` stays the column's name, and
``ValidationError.path`` holds the path:

.. code-block:: python

   class HistorySchema(Protocol):
       history: list[Address]

   df = pl.DataFrame({"history": [[{"city": "Tokyo", "zip": 1000001}]]})
   try:
       DataFrame[HistorySchema](df)
   except ValidationError as e:
       print(e)
       print(e.column_name, e.path)

Output:

.. example-output

.. code-block:: text

   Column 'history[].zip': expected str, got Int64
   history history[].zip

Null values show the whole row they are found in:

.. code-block:: python

   df = pl.DataFrame({"tags": [["a", None], ["b"], [None]]})
   try:
       DataFrame[UserSchema](df.with_columns(
           address=pl.lit({"city": "Tokyo", "zip": "100-0001"}),
           history=pl.lit([], dtype=pl.List(pl.Struct({"city": pl.Utf8, "zip": pl.Utf8}))),
           nickname=pl.lit(None),
       ))
   except ValidationError as e:
       print(e)

Output:

.. example-output

.. code-block:: text

   Column 'tags[]': is non-optional but contains null values

   Sample invalid values (showing first 2 of 2):
     Row 0: ['a', None]
     Row 2: [None]

A missing struct field:

.. code-block:: python

   class AddressSchema(Protocol):
       address: Address

   df = pl.DataFrame({"address": [{"city": "Tokyo"}]})
   try:
       DataFrame[AddressSchema](df)
   except ValidationError as e:
       print(e)

Output:

.. example-output

.. code-block:: text

   Column 'address': missing struct field 'zip' (found fields: ['city'])

Dtype Annotations
-----------------

Nested dtypes can also be written as dtype instances, which must then match exactly:

.. code-block:: python

   class DtypeSchema(Protocol):
       tags: pl.List(pl.Utf8)
       pair: pl.Array(pl.Int64, 2)

   df = pl.DataFrame({
       "tags": [["a"]],
       "pair": pl.Series([[1, 2]], dtype=pl.Array(pl.Int64, 2)),
   })
   validated_df = DataFrame[DtypeSchema](df)

For pandas, use ``pd.ArrowDtype(...)`` instances such as
``pd.ArrowDtype(pa.list_(pa.string()))``.
