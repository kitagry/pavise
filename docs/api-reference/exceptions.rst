pavise.exceptions
=================

PaviseError
-----------

Base class for all exceptions raised by Pavise. ``PatrolError`` is kept as an
alias for backward compatibility.

.. autoclass:: pavise.exceptions.PaviseError
   :show-inheritance:

ValidationError
---------------

The main exception raised by Pavise when DataFrame validation fails.

.. autoclass:: pavise.exceptions.ValidationError
   :members:
   :show-inheritance:
   :special-members: __init__

Usage
~~~~~

.. code-block:: python

   from pavise.pandas import DataFrame
   from pavise.exceptions import ValidationError

   try:
       validated_df = DataFrame[Schema](raw_df)
   except ValidationError as e:
       print(f"Validation failed: {e}")
       print(f"Column: {e.column_name}")
       print(f"Invalid samples: {e.invalid_samples}")
