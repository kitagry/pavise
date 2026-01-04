"""Pandas backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import Any, Generic, Optional, TypeVar

import pandas as pd

from pavise._pandas.validation import validate_dataframe

SchemaT_co = TypeVar("SchemaT_co", covariant=True)


class DataFrame(pd.DataFrame, Generic[SchemaT_co]):
    """
    Type-parameterized DataFrame with runtime validation for pandas.

    Usage::

        # Static type checking only
        def process(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
            return df

        # Runtime validation
        validated = DataFrame[UserSchema](raw_df)

    The type parameter is covariant, allowing structural subtyping.
    DataFrame[ChildSchema] is compatible with DataFrame[ParentSchema]
    when ChildSchema has all columns of ParentSchema.
    """

    _schema: Optional[type] = None

    def __class_getitem__(cls, schema: type):
        """Create a new DataFrame class with schema validation."""

        class TypedDataFrame(DataFrame):
            _schema = schema

        return TypedDataFrame

    def __new__(cls, data: Any = None, *args: Any, strict: bool = False, **kwargs: Any):
        """Create a new DataFrame instance."""
        return super().__new__(cls)

    def __init__(self, data: Any = None, *args: Any, strict: bool = False, **kwargs: Any) -> None:
        """
        Initialize DataFrame with optional schema validation.

        Args:
            data: Data to create DataFrame from
            *args: Additional arguments passed to pd.DataFrame
            strict: If True, raise error on extra columns not in schema
            **kwargs: Additional keyword arguments passed to pd.DataFrame

        Raises:
            ValueError: If required column is missing
            TypeError: If column has wrong type
        """
        pd.DataFrame.__init__(self, data, *args, **kwargs)  # type: ignore[misc]
        if self._schema is not None:
            validate_dataframe(self, self._schema, strict=strict)
