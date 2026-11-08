"""Polars backend for type-parameterized DataFrame with Protocol-based schema validation."""

from typing import Any, Generic, TypeVar, get_type_hints

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise._annotations import _extract_type_and_validators, is_nested_type, parse_annotation
from pavise._polars.dtypes import dtype_for_type
from pavise._polars.nested import nested_dtype
from pavise._polars.validation import validate_dataframe, validate_lazyframe_schema
from pavise.types import NotRequiredColumn

__all__ = ["DataFrame", "LazyFrame", "NotRequiredColumn"]

SchemaT_co = TypeVar("SchemaT_co", covariant=True)


def _build_empty_columns(schema: type) -> dict[str, pl.Series]:
    """
    Build empty column dict from a Protocol schema.

    Used by both DataFrame.make_empty() and LazyFrame.make_empty().
    """
    type_hints = get_type_hints(schema, include_extras=True)
    columns = {}

    for col_name, col_type in type_hints.items():
        base_type, _, _, _ = _extract_type_and_validators(col_type)
        if is_nested_type(base_type):
            dtype = nested_dtype(parse_annotation(col_type, col_name, schema))
        else:
            dtype = dtype_for_type(base_type)
        columns[col_name] = pl.Series([], dtype=dtype)

    return columns


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

    _schema: type | None = None

    def __class_getitem__(cls, schema: type):
        """Create a new DataFrame class with schema validation."""

        class TypedDataFrame(DataFrame):
            _schema = schema

        return TypedDataFrame

    def __init__(self, data: Any, *args: Any, strict: bool = False, **kwargs: Any):
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
    def make_empty(cls) -> "DataFrame[SchemaT_co]":
        """
        Create an empty DataFrame with columns from the schema.

        Returns:
            DataFrame: Empty DataFrame with correct column types
        """
        if cls._schema is None:
            return cls({})

        return cls(_build_empty_columns(cls._schema))


class LazyFrame(pl.LazyFrame, Generic[SchemaT_co]):
    """
    Type-parameterized LazyFrame with runtime schema validation for Polars.

    Schema validation happens at construction time using collect_schema().
    Value-based validators (Range, Unique, etc.) are only checked on collect().
    """

    _schema: type | None = None

    def __class_getitem__(cls, schema: type):
        """Create a new LazyFrame class with schema validation."""

        class TypedLazyFrame(LazyFrame):
            _schema = schema

        return TypedLazyFrame

    def __new__(cls, data: pl.LazyFrame, strict: bool = False):  # noqa: ARG003
        """Create LazyFrame instance by copying internal state from source."""
        instance = object.__new__(cls)
        instance._ldf = data._ldf
        return instance

    def __init__(self, _data: pl.LazyFrame, strict: bool = False):
        """
        Initialize LazyFrame with schema validation.

        Args:
            data: Polars LazyFrame to wrap
            strict: If True, raise error on extra columns not in schema
        """
        if self._schema is not None:
            validate_lazyframe_schema(self, self._schema, strict=strict)

    def collect(self) -> "DataFrame[SchemaT_co]":  # type: ignore[override]
        """
        Collect LazyFrame into DataFrame with full validation.

        Returns:
            DataFrame[Schema] with all validators applied
        """
        df = pl.LazyFrame.collect(self)
        if self._schema is not None:
            return DataFrame[self._schema](df)  # type: ignore[valid-type]
        return DataFrame(df)

    @classmethod
    def make_empty(cls) -> "LazyFrame[SchemaT_co]":
        """
        Create an empty LazyFrame with columns from the schema.

        Returns:
            LazyFrame: Empty LazyFrame with correct column types
        """
        if cls._schema is None:
            return cls(pl.LazyFrame({}))

        return cls(pl.DataFrame(_build_empty_columns(cls._schema)).lazy())
