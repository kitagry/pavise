DataFrame-Level Checks
======================

Column validators check each column on its own. To validate relationships across
columns, such as ``start_at < end_at`` or ``total == a + b``, attach a DataFrame-level
check to the schema with the ``@dataframe_check`` class decorator.

Basic Usage
-----------

.. code-block:: python

   from datetime import datetime
   from typing import Protocol

   import pandas as pd
   from pavise.checks import dataframe_check
   from pavise.pandas import DataFrame

   @dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
   class EventSchema(Protocol):
       start_at: datetime
       end_at: datetime

   validated_df = DataFrame[EventSchema](df)

Multiple decorators can be stacked. Checks run from the top decorator to the bottom.

.. code-block:: python

   @dataframe_check(lambda df: df["a"] >= 0, name="a_non_negative")
   @dataframe_check(lambda df: df["total"] == df["a"] + df["b"], name="total_is_sum")
   class TotalSchema(Protocol):
       a: int
       b: int
       total: int

``name`` is used in error messages. When omitted, the function's ``__name__`` is used,
so pass ``name`` when using a lambda.

Check Functions
---------------

A check function receives the backend's native DataFrame and returns one of:

- **A boolean per row**: rows where it is ``False`` are reported as violations.

  - pandas: ``pd.Series`` of bools
  - polars: ``pl.Series`` of bools or a ``pl.Expr`` evaluating to bools

- **A single bool**: for constraints on the whole DataFrame, such as an aggregate.

.. code-block:: python

   import polars as pl
   from pavise.polars import DataFrame

   # polars: return an expression
   @dataframe_check(lambda df: pl.col("total") == pl.col("a") + pl.col("b"), name="total_is_sum")
   class TotalSchema(Protocol):
       a: int
       b: int
       total: int

   # Whole-DataFrame check
   @dataframe_check(lambda df: df["share"].sum() == 100, name="shares_sum_to_100")
   class ShareSchema(Protocol):
       share: int

Any other return value raises ``ValidationError``.

Null Handling
~~~~~~~~~~~~~

Rows where the check result is null are treated as passing, like SQL ``CHECK``
constraints. Use ``Optional`` columns or a separate check to reject missing values.

.. note::

   With numpy-backed pandas columns, comparisons involving ``NaN`` return ``False``
   rather than null, so those rows are reported as violations. Use nullable dtypes
   (e.g. ``Float64``) or allow missing values explicitly:
   ``lambda df: (df["start_at"] < df["end_at"]) | df["end_at"].isna()``.

Error Messages
--------------

When a row check fails, the error shows up to five violating rows with the schema's
columns (extra columns in the DataFrame are omitted):

.. code-block:: text

   ValidationError: DataFrame check 'start_before_end' failed

   Sample invalid values (showing first 1 of 1):
     Row 1: {'start_at': Timestamp('2024-01-05 00:00:00'), 'end_at': Timestamp('2024-01-03 00:00:00')}

The row label is the pandas index, or the row number for polars. The samples are also
available as ``error.invalid_samples``, and ``error.column_name`` is ``None``.
A failing single-bool check only reports the check name.

When Checks Run
---------------

- After all column type checks, column validators and strict mode pass, so check
  functions can rely on the declared columns and types.
- Validation stops at the first failing check.
- ``polars.LazyFrame[Schema](lf)`` only checks the schema at construction; DataFrame-level
  checks run on ``collect()``.

Inheritance and Structural Subtyping
------------------------------------

A Protocol that inherits from a decorated Protocol inherits its checks. The parent's
checks run before the child's own checks.

.. code-block:: python

   class LabeledEventSchema(EventSchema, Protocol):
       label: str

   DataFrame[LabeledEventSchema](df)  # runs 'start_before_end'

The decorator stores checks outside the Protocol's annotations and members, so it does
not affect static type checking. A schema that is structurally compatible but does not
inherit from ``EventSchema`` can still be passed where ``DataFrame[EventSchema]`` is
expected. Its own checks, not ``EventSchema``'s, run when it is validated.
