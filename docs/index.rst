Pavise Documentation
====================

Pavise is a DataFrame validation library using Python Protocol for structural subtyping.
It supports both pandas and polars backends as optional dependencies.

Features
--------

* **Type-safe DataFrame validation** using Python's Protocol and structural subtyping
* **Multiple backends**: Support for both pandas and polars
* **Runtime validation** with detailed error messages
* **Annotated validators**: Attach validators to column types using ``typing.Annotated``
* **Strict mode**: Optionally reject DataFrames with extra columns
* **Covariant type parameters**: DataFrames with more columns can be used where fewer are expected

Quick Start
-----------

Installation:

.. code-block:: bash

   # For pandas backend
   pip install pavise[pandas]

   # For polars backend
   pip install pavise[polars]

   # For both backends
   pip install pavise[all]

Basic usage:

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame

   class UserSchema(Protocol):
       user_id: int
       name: str
       age: int

   # Type checking only (no runtime overhead)
   def process(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
       return df

   # Runtime validation at system boundaries
   validated_df = DataFrame[UserSchema](raw_df)

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting-started
   user-guide/index
   api-reference/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
