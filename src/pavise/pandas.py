"""Pandas backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import pandas as pd

from pavise._pandas.validation import INDEX_COLUMN_NAME, validate_dataframe
from pavise.types import NotRequiredColumn

__all__ = ["DataFrame", "NotRequiredColumn"]

SchemaT_co = TypeVar("SchemaT_co", covariant=True)


def _get_dtype_for_type(base_type: type) -> str | pd.api.extensions.ExtensionDtype:
    """
    Get pandas dtype for a given Python type.

    Args:
        base_type: Python type (int, str, float, bool, datetime, date, timedelta)

    Returns:
        String representation of pandas dtype
    """
    from pavise._pandas.validation import TYPE_TO_DTYPE

    if isinstance(base_type, type) and issubclass(base_type, pd.api.extensions.ExtensionDtype):
        return base_type()

    return TYPE_TO_DTYPE.get(base_type, "object")


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

    _schema: type | None = None

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

    @classmethod
    def make_empty(cls):
        """
        Create an empty DataFrame with columns from the schema.

        Returns:
            DataFrame: Empty DataFrame with correct column types
        """
        if cls._schema is None:
            return cls({})

        from pavise._pandas.validation import (
            _extract_index_name_type_and_validators,
            _extract_type_and_validators,
        )

        type_hints = get_type_hints(cls._schema, include_extras=True)
        columns = {}
        index_name = None
        index_base_type = None

        for col_name, col_type in type_hints.items():
            if col_name == INDEX_COLUMN_NAME:
                # Extract index name from schema
                index_base_type, index_name, _validators, _is_optional = (
                    _extract_index_name_type_and_validators(col_type)
                )
                continue

            base_type, _validators, is_optional, _is_not_required = _extract_type_and_validators(
                col_type
            )

            # Handle Union types (represented as tuple) - use first type
            if isinstance(base_type, tuple):
                base_type = base_type[0]

            if get_origin(base_type) is Literal:
                literal_values = get_args(base_type)
                if literal_values:
                    first_value = literal_values[0]
                    base_type = type(first_value)

            dtype = _get_dtype_for_type(base_type)
            columns[col_name] = pd.Series([], dtype=dtype)

        # Create pandas DataFrame first, set index name, then convert to typed DataFrame
        raw_df = pd.DataFrame(columns)
        if index_name is not None:
            if isinstance(index_name, tuple):
                # MultiIndex: create empty MultiIndex with names
                level_types = get_args(index_base_type)
                level_arrays = [
                    pd.array([], dtype=_get_dtype_for_type(level_type))
                    for level_type in level_types
                ]
                raw_df.index = pd.MultiIndex.from_arrays(level_arrays, names=list(index_name))
            else:
                # Single index: set index name
                raw_df.index.name = index_name
        return cls(raw_df)
