Custom Validator Examples
=========================

Examples of creating and using custom validators for domain-specific validation logic.

Business Rules Validation
--------------------------

Validate business-specific constraints:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.pandas import DataFrame
   from pavise.validators import Custom, Range
   import pandas as pd

   def is_business_day(date_value) -> bool:
       """Check if date is a business day (Monday-Friday)."""
       return date_value.weekday() < 5

   def is_positive(value) -> bool:
       """Check if value is positive."""
       return value > 0

   class FinancialDataSchema(Protocol):
       date: Annotated[datetime.date, Custom(is_business_day, "must be a business day")]
       amount: Annotated[float, Custom(is_positive, "must be positive")]
       profit_margin: Annotated[float, Range(0.0, 1.0)]

   # Validate financial data
   df = pd.DataFrame({
       "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date,
       "amount": [1000.0, 2000.0, -500.0],  # -500.0 will fail
       "profit_margin": [0.15, 0.20, 0.18]
   })

   try:
       validated_df = DataFrame[FinancialDataSchema](df)
   except ValueError as e:
       print(f"Validation failed: {e}")

Domain-Specific Validators
---------------------------

Create validators for domain-specific rules:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.validators import Custom
   import re

   def is_valid_isbn(isbn: str) -> bool:
       """Validate ISBN-10 or ISBN-13 format."""
       isbn = isbn.replace("-", "").replace(" ", "")
       if len(isbn) == 10:
           return bool(re.match(r'^\d{9}[\dX]$', isbn))
       elif len(isbn) == 13:
           return bool(re.match(r'^\d{13}$', isbn))
       return False

   def is_valid_price(price: float) -> bool:
       """Price must be positive and have at most 2 decimal places."""
       if price <= 0:
           return False
       return round(price, 2) == price

   class BookSchema(Protocol):
       isbn: Annotated[str, Custom(is_valid_isbn, "must be valid ISBN-10 or ISBN-13")]
       title: str
       price: Annotated[float, Custom(is_valid_price, "must be positive with max 2 decimals")]

Cross-Field Validation
----------------------

Validate relationships between fields (note: this requires accessing the full row):

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.pandas import DataFrame
   from pavise.validators import Custom
   import pandas as pd

   def is_valid_discount(discount: float) -> bool:
       """Discount must be between 0% and 100%."""
       return 0.0 <= discount <= 1.0

   class ProductSchema(Protocol):
       original_price: float
       discount: Annotated[float, Custom(is_valid_discount, "must be between 0.0 and 1.0")]
       final_price: float

   # After validation, verify cross-field constraint manually
   def validate_pricing(df: DataFrame[ProductSchema]) -> DataFrame[ProductSchema]:
       expected_price = df["original_price"] * (1 - df["discount"])
       if not (df["final_price"] == expected_price).all():
           raise ValueError("final_price must equal original_price * (1 - discount)")
       return df

   df = pd.DataFrame({
       "original_price": [100.0, 200.0],
       "discount": [0.1, 0.2],
       "final_price": [90.0, 160.0]
   })

   validated_df = DataFrame[ProductSchema](df)
   validated_df = validate_pricing(validated_df)

Combining Multiple Custom Validators
-------------------------------------

Use multiple custom validators on a single field:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.validators import Custom, MinLen

   def no_special_chars(value: str) -> bool:
       """Check if string contains only alphanumeric characters and spaces."""
       return value.replace(" ", "").isalnum()

   def no_leading_trailing_spaces(value: str) -> bool:
       """Check if string has no leading or trailing spaces."""
       return value == value.strip()

   class UserInputSchema(Protocol):
       username: Annotated[
           str,
           MinLen(3),
           Custom(no_special_chars, "must contain only letters, numbers, and spaces"),
           Custom(no_leading_trailing_spaces, "must not have leading or trailing spaces")
       ]

Complex Validation Logic
-------------------------

Implement complex business logic in custom validators:

.. code-block:: python

   from typing import Protocol, Annotated
   from pavise.validators import Custom
   import re

   def is_strong_password(password: str) -> bool:
       """
       Validate password strength:
       - At least 8 characters
       - Contains uppercase and lowercase
       - Contains at least one digit
       - Contains at least one special character
       """
       if len(password) < 8:
           return False
       if not re.search(r'[A-Z]', password):
           return False
       if not re.search(r'[a-z]', password):
           return False
       if not re.search(r'\d', password):
           return False
       if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
           return False
       return True

   def is_valid_email_domain(email: str) -> bool:
       """Only allow specific email domains."""
       allowed_domains = ["example.com", "test.com"]
       domain = email.split("@")[-1]
       return domain in allowed_domains

   class SecureUserSchema(Protocol):
       email: Annotated[
           str,
           Custom(is_valid_email_domain, "must be from allowed domains")
       ]
       password: Annotated[
           str,
           Custom(is_strong_password, "must meet password strength requirements")
       ]
