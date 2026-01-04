Data Validation Examples
========================

This page shows practical examples of data validation with Pavise.

CSV Data Validation
-------------------

Validate data loaded from CSV files:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.pandas import DataFrame
   from pavise.exceptions import ValidationError
   from pavise.validators import Range, Regex
   import pandas as pd

   class UserDataSchema(Protocol):
       user_id: int
       name: str
       age: Annotated[int, Range(0, 150)]
       email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]

   # Load from CSV
   raw_df = pd.read_csv("users.csv")

   # Validate
   try:
       validated_df = DataFrame[UserDataSchema](raw_df)
       print("Data is valid!")
   except ValidationError as e:
       print(f"Validation failed: {e}")

Database Query Validation
--------------------------

Validate data from database queries:

.. code-block:: python

   from typing import Protocol, Annotated, Optional
   from pavise.pandas import DataFrame
   from pavise.validators import In
   import pandas as pd
   import sqlalchemy

   class OrderSchema(Protocol):
       order_id: int
       customer_id: int
       status: Annotated[str, In(["pending", "processing", "shipped", "delivered"])]
       amount: Annotated[float, Range(0.0, float('inf'))]
       created_at: datetime.datetime

   # Query database
   engine = sqlalchemy.create_engine("sqlite:///orders.db")
   query = "SELECT * FROM orders WHERE created_at > '2024-01-01'"
   raw_df = pd.read_sql(query, engine)

   # Validate
   validated_df = DataFrame[OrderSchema](raw_df)

API Response Validation
-----------------------

Validate data from external APIs:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.polars import DataFrame
   from pavise.exceptions import ValidationError
   from pavise.validators import Unique, Range
   import polars as pl
   import requests

   class APIProductSchema(Protocol):
       id: Annotated[int, Unique()]
       name: str
       price: Annotated[float, Range(0.0, float('inf'))]
       in_stock: bool

   # Fetch from API
   response = requests.get("https://api.example.com/products")
   data = response.json()

   # Convert to polars DataFrame
   df = pl.DataFrame(data)

   # Validate with strict mode (no extra fields allowed)
   try:
       validated_df = DataFrame[APIProductSchema](df, strict=True)
   except ValidationError as e:
       print(f"API contract violation: {e}")

ETL Pipeline Validation
-----------------------

Validate data at different stages of an ETL pipeline:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.pandas import DataFrame
   from pavise.validators import Range, MinLen
   import pandas as pd

   # Raw data schema
   class RawDataSchema(Protocol):
       id: int
       raw_value: str

   # Cleaned data schema
   class CleanedDataSchema(Protocol):
       id: int
       value: Annotated[str, MinLen(1)]
       normalized_value: Annotated[float, Range(0.0, 1.0)]

   # Extract
   raw_df = pd.read_csv("raw_data.csv")
   validated_raw = DataFrame[RawDataSchema](raw_df)

   # Transform
   def clean_data(df: DataFrame[RawDataSchema]) -> DataFrame[CleanedDataSchema]:
       cleaned = df.copy()
       cleaned["value"] = cleaned["raw_value"].str.strip()
       cleaned["normalized_value"] = (
           cleaned["value"].str.len() / cleaned["value"].str.len().max()
       )
       cleaned = cleaned.drop(columns=["raw_value"])
       return DataFrame[CleanedDataSchema](cleaned)

   # Load
   cleaned_df = clean_data(validated_raw)
   cleaned_df.to_csv("cleaned_data.csv", index=False)

Handling Validation Errors
---------------------------

Gracefully handle validation errors in production:

.. code-block:: python

   from typing import Protocol
   from pavise.pandas import DataFrame
   from pavise.exceptions import ValidationError
   import pandas as pd
   import logging

   logger = logging.getLogger(__name__)

   class TransactionSchema(Protocol):
       transaction_id: int
       amount: float
       timestamp: datetime.datetime

   def process_transactions(file_path: str) -> None:
       try:
           raw_df = pd.read_csv(file_path)
           validated_df = DataFrame[TransactionSchema](raw_df)

           # Process valid data
           process_valid_transactions(validated_df)

       except ValidationError as e:
           logger.error(f"Validation failed in {file_path}: {e}")
           # Handle validation error - maybe clean and retry

       except Exception as e:
           logger.error(f"Unexpected error in {file_path}: {e}")
           raise

   def process_valid_transactions(df: DataFrame[TransactionSchema]) -> None:
       # Process validated data
       pass
