"""Polars backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import Generic, Literal, Optional, TypeVar, get_args, get_origin, get_type_hints

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise._polars.validation import validate_dataframe
from pavise.types import NotRequiredColumn

__all__ = ["DataFrame", "NotRequiredColumn"]

SchemaT_co = TypeVar("SchemaT_co", covariant=True)


def _get_dtype_for_type(base_type: type) -> pl.DataType:
    """
    Get polars dtype for a given Python type.

    Args:
        base_type: Python type (int, str, float, bool, datetime, date, timedelta)

    Returns:
        Polars DataType
    """
    from pavise._polars.validation import TYPE_TO_DTYPE

    if isinstance(base_type, type) and issubclass(base_type, pl.DataType):
        return base_type()

    return TYPE_TO_DTYPE.get(base_type, pl.Utf8())


class DataFrame(pl.DataFrame, Generic[SchemaT_co]):
    """
    Type-parameterized DataFrame with runtime validation for Polars.

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

    def __init__(self, data, *args, strict=False, **kwargs):
        """
        Initialize DataFrame with optional schema validation.

        Args:
            data: Data to create DataFrame from (pl.DataFrame or dict/list)
            *args: Additional arguments passed to pl.DataFrame
            strict: If True, raise error on extra columns not in schema
            **kwargs: Additional keyword arguments passed to pl.DataFrame

        Raises:
            ValueError: If required column is missing
            TypeError: If column has wrong type
        """
        pl.DataFrame.__init__(self, data, *args, **kwargs)  # type: ignore[misc]
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

        from pavise._polars.validation import _extract_type_and_validators

        type_hints = get_type_hints(cls._schema, include_extras=True)
        columns = {}

        for col_name, col_type in type_hints.items():
            base_type, _validators, is_optional, _is_not_required = _extract_type_and_validators(
                col_type
            )

            if get_origin(base_type) is Literal:
                literal_values = get_args(base_type)
                if literal_values:
                    first_value = literal_values[0]
                    base_type = type(first_value)

            dtype = _get_dtype_for_type(base_type)
            columns[col_name] = pl.Series([], dtype=dtype)

        return cls(columns)
