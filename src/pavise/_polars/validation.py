"""Common validation functions for Polars DataFrame schema checking."""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Any,
    Literal,
    get_args,
    get_origin,
    get_type_hints,
)

from pavise.types import NotRequiredColumn

try:
    import polars as pl
except ImportError:
    raise ImportError("Polars is not installed. Install it with: pip install pavise[polars]")

from pavise._annotations import (
    _extract_type_and_validators,
    is_nested_type,
    parse_nested_column,
)
from pavise._polars.dtypes import (
    TYPE_CHECKERS,
    TYPE_TO_DTYPE,
    TypeChecker,
    dtype_name,
    is_dtype_annotation,
)
from pavise._polars.nested import check_nested_dtype, check_nested_values
from pavise._polars.validator_impl import (
    BOUND_VALIDATORS,
    MAX_SAMPLE_SIZE,
    apply_validator,
    get_invalid_samples,
    require_orderable_dtype,
)
from pavise.checks import DataFrameCheck, UniqueCheck, get_dataframe_checks
from pavise.exceptions import ValidationError


def validate_dataframe(df: pl.DataFrame, schema: type, strict: bool = False) -> None:
    """
    Validate that a Polars DataFrame conforms to a Protocol schema.

    Args:
        df: Polars DataFrame to validate
        schema: Protocol type defining the expected schema
        strict: If True, raise error on extra columns not in schema

    Raises:
        ValueError: If a required column is missing or type is unsupported
        TypeError: If a column has the wrong type
    """
    expected_cols = get_type_hints(schema, include_extras=True)

    for col_name, col_type in expected_cols.items():
        is_not_required = isinstance(col_type, type) and issubclass(col_type, NotRequiredColumn)
        if is_not_required and col_name not in df.columns:
            continue
        _check_column_exists(df, col_name)
        _check_column_type(df, col_name, col_type, strict)

    schema_cols = [c for c in expected_cols if c in df.columns]
    if strict:
        extra_cols = set(df.columns) - set(schema_cols)
        if extra_cols:
            raise ValidationError(f"Strict mode: unexpected columns {sorted(extra_cols)}")

    for check in get_dataframe_checks(schema):
        if isinstance(check, UniqueCheck):
            _run_unique_check(df, check)
        else:
            _run_dataframe_check(df, check, schema_cols)


def _run_unique_check(df: pl.DataFrame, check: UniqueCheck) -> None:
    """Raise with sample duplicate keys and their rows if the key columns are not unique."""
    columns = list(check.columns)
    # An absent NotRequiredColumn leaves no complete key to compare.
    if not all(c in df.columns for c in columns):
        return

    duplicated = df.select(pl.struct(columns).is_duplicated()).to_series()
    if not duplicated.any():
        return

    # Nesting the key columns in a struct keeps their names from clashing with the helpers.
    groups = (
        df.select(pl.struct(columns).alias("key"), pl.int_range(pl.len()).alias("row"))
        .filter(duplicated)
        .group_by("key", maintain_order=True)
        .agg(pl.col("row").head(MAX_SAMPLE_SIZE), pl.len().alias("count"))
    )
    duplicates = [
        (group["key"], group["row"], group["count"])
        for group in groups.head(MAX_SAMPLE_SIZE).iter_rows(named=True)
    ]
    raise ValidationError.new_with_duplicate_keys(check.columns, duplicates, groups.height)


def _run_dataframe_check(df: pl.DataFrame, check: DataFrameCheck, schema_cols: list[str]) -> None:
    """Run a DataFrame-level check and raise with sample violating rows if it fails."""
    result = check.func(df)
    base_message = f"DataFrame check '{check.name}' failed"

    if isinstance(result, pl.Expr):
        evaluated = df.select(result)
        if evaluated.width != 1:
            raise ValidationError(
                f"DataFrame check '{check.name}' must return an Expr producing a single column, "
                f"got {evaluated.width} columns"
            )
        result = evaluated.to_series()
        # Aggregations like pl.col("a").sum() == 100 evaluate to one value for the whole frame.
        if result.len() == 1 and df.height != 1:
            result = result.item()
            if result is None:
                return

    if isinstance(result, bool):
        if not result:
            raise ValidationError(base_message)
        return

    if not isinstance(result, pl.Series) or result.dtype != pl.Boolean:
        if isinstance(result, pl.Series):
            type_name = f"Series of dtype {result.dtype}"
        else:
            type_name = type(result).__name__
        raise ValidationError(
            f"DataFrame check '{check.name}' must return a boolean Series, Expr or bool, "
            f"got {type_name}"
        )

    if result.len() != df.height:
        raise ValidationError(
            f"DataFrame check '{check.name}' must return one value per row, "
            f"got {result.len()} values for {df.height} rows"
        )

    # Null means the check could not be evaluated for the row, which counts as passing.
    invalid_mask = ~result.fill_null(True)
    total_invalid = int(invalid_mask.sum())
    if total_invalid == 0:
        return

    row_numbers = invalid_mask.arg_true().head(MAX_SAMPLE_SIZE).to_list()
    invalid_rows = df.select(schema_cols).filter(invalid_mask).head(MAX_SAMPLE_SIZE)
    samples: list[tuple[Any, ...]] = list(zip(row_numbers, invalid_rows.iter_rows(named=True)))
    raise ValidationError.new_with_samples(None, base_message, samples, total_invalid)


def validate_lazyframe_schema(lf: pl.LazyFrame, schema: type, strict: bool = False) -> None:
    """
    Validate that a Polars LazyFrame conforms to a Protocol schema.

    Only validates schema (column existence and types) without collecting.
    Value-based validators are NOT applied here.

    Args:
        lf: Polars LazyFrame to validate
        schema: Protocol type defining the expected schema
        strict: If True, raise error on extra columns not in schema
    """
    lf_schema = lf.collect_schema()
    expected_cols = get_type_hints(schema, include_extras=True)

    for col_name, col_type in expected_cols.items():
        is_not_required = isinstance(col_type, type) and issubclass(col_type, NotRequiredColumn)
        if is_not_required and col_name not in lf_schema.names():
            continue
        _check_lazyframe_column_exists(lf_schema, col_name)
        _check_lazyframe_column_type(lf_schema, col_name, col_type, strict)

    if strict:
        schema_cols = set(expected_cols.keys())
        lf_cols = set(lf_schema.names())
        extra_cols = lf_cols - schema_cols
        if extra_cols:
            raise ValidationError(f"Strict mode: unexpected columns {sorted(extra_cols)}")


def _check_lazyframe_column_exists(lf_schema: pl.Schema, col_name: str) -> None:
    """Check if a column exists in the LazyFrame schema."""
    if col_name not in lf_schema.names():
        raise ValidationError("missing", column_name=col_name)


def _check_lazyframe_column_type(
    lf_schema: pl.Schema, col_name: str, expected_type: type, strict: bool
) -> None:
    """Check if a column has the expected type (schema-level only, no value validation)."""
    base_type, validators, _is_optional, _is_not_required = _extract_type_and_validators(
        expected_type
    )

    col_dtype = lf_schema[col_name]

    if is_nested_type(base_type):
        field = parse_nested_column(expected_type, col_name)
        check_nested_dtype(field, col_dtype, col_name, strict)
        return

    # Orderability depends only on the dtype, so reject it here instead of waiting for collect()
    for validator in validators:
        if isinstance(validator, BOUND_VALIDATORS):
            require_orderable_dtype(col_dtype, validator, col_name)

    if is_dtype_annotation(base_type):
        if col_dtype != base_type:
            raise ValidationError(
                f"expected {dtype_name(base_type)}, got {col_dtype}",
                column_name=col_name,
            )
        return

    if get_origin(base_type) is Literal:
        # Check base type of Literal (e.g., str for Literal["a", "b"], int for Literal[1, 2])
        literal_args = get_args(base_type)
        if literal_args:
            literal_base_type = type(literal_args[0])
            if literal_base_type in TYPE_CHECKERS:
                checker = TYPE_CHECKERS[literal_base_type]
                if not checker.dtype(col_dtype):
                    raise ValidationError(
                        f"expected {literal_base_type.__name__}, got {col_dtype}",
                        column_name=col_name,
                    )
        return

    # Handle Union types (represented as tuple) - for LazyFrame,
    # we can only check dtype compatibility
    if isinstance(base_type, tuple):
        # For Union types in LazyFrame, we can't validate mixed values at schema level
        # Just check if the dtype is object (which can hold multiple types)
        # Full validation happens on collect()
        return

    if base_type not in TYPE_CHECKERS:
        raise ValidationError(f"unsupported type: {base_type}", column_name=col_name)

    checker = TYPE_CHECKERS[base_type]
    if not checker.dtype(col_dtype):
        raise ValidationError(
            f"expected {base_type.__name__}, got {col_dtype}",
            column_name=col_name,
        )


def _invalid_value_mask(
    series: pl.Series, is_valid: Callable[[Any], bool], null_valid: bool
) -> pl.Series:
    """
    Return a boolean mask of values rejected by ``is_valid``, calling it as rarely as possible.

    ``is_valid`` must depend only on the value's type for non-null values (an isinstance check),
    which holds for every TYPE_CHECKERS value checker.
    """
    if series.dtype == pl.Object:
        cache: dict[type, bool] = {}

        def is_invalid(value: Any) -> bool:
            if value is None:
                return not null_valid
            value_type = type(value)
            if value_type not in cache:
                cache[value_type] = not is_valid(value)
            return cache[value_type]

        return series.map_elements(is_invalid, return_dtype=pl.Boolean, skip_nulls=False)

    # Non-null values of any non-Object dtype all share one Python type, so a single element
    # decides for the whole column.
    is_null = series.is_null()
    null_count = series.null_count()
    invalid_non_null = False
    if null_count < series.len():
        first = 0 if null_count == 0 else int(is_null.arg_min())  # type: ignore[arg-type]
        invalid_non_null = not is_valid(series[first])
    return (is_null & (not null_valid)) | (~is_null & invalid_non_null)


def _raise_type_error_with_samples(
    df: pl.DataFrame, col_name: str, checker: TypeChecker, expected_type: type, actual_dtype: Any
) -> None:
    """Raise TypeError with sample invalid values."""
    invalid_mask = _invalid_value_mask(df[col_name], checker.value, null_valid=True)
    samples, total_invalid = get_invalid_samples(df[col_name], invalid_mask)

    raise ValidationError.new_with_samples(
        col_name,
        f"expected {expected_type.__name__}, got {actual_dtype}",
        samples,
        total_invalid,
    )


def _check_column_exists(df: pl.DataFrame, col_name: str) -> None:
    """Check if a column exists in the DataFrame."""
    if col_name not in df.columns:
        raise ValidationError("missing", column_name=col_name)


def _check_column_type(df: pl.DataFrame, col_name: str, expected_type: type, strict: bool) -> None:
    """Check if a column has the expected type and apply validators."""
    base_type, validators, is_optional, _is_not_required = _extract_type_and_validators(
        expected_type
    )

    if is_nested_type(base_type):
        field = parse_nested_column(expected_type, col_name)
        check_nested_dtype(field, df[col_name].dtype, col_name, strict)
        check_nested_values(field, df[col_name], col_name)
        return

    if is_dtype_annotation(base_type):
        col_dtype = df[col_name].dtype
        if col_dtype != base_type:
            raise ValidationError(
                f"expected {dtype_name(base_type)}, got {col_dtype}",
                column_name=col_name,
            )
        for validator in validators:
            apply_validator(df[col_name], validator, col_name)
        return

    if get_origin(base_type) is Literal:
        allowed_values = get_args(base_type)
        if not is_optional and None not in allowed_values and df[col_name].null_count() > 0:
            raise ValidationError("is non-optional but contains null values", column_name=col_name)
        # is_in propagates nulls, so nulls never count as invalid values here
        invalid_mask = ~df[col_name].is_in(allowed_values)
        if invalid_mask.any():
            samples, total_invalid = get_invalid_samples(df[col_name], invalid_mask)
            raise ValidationError.new_with_samples(
                col_name,
                f"expected one of {allowed_values}, got invalid values",
                samples,
                total_invalid,
                repr,
            )
        for validator in validators:
            apply_validator(df[col_name], validator, col_name)
        return

    # Handle Union types (represented as tuple)
    if isinstance(base_type, tuple):
        union_types = base_type
        # Check if all types in the union are supported
        for union_type in union_types:
            if union_type not in TYPE_CHECKERS:
                raise ValidationError(f"unsupported type: {union_type}", column_name=col_name)

        # Check if each value matches at least one of the union types
        invalid_mask = _invalid_value_mask(
            df[col_name],
            lambda value: any(TYPE_CHECKERS[t].value(value) for t in union_types),
            null_valid=is_optional,
        )
        if invalid_mask.any():
            samples, total_invalid = get_invalid_samples(df[col_name], invalid_mask)
            union_type_names = " | ".join(t.__name__ for t in base_type)
            raise ValidationError.new_with_samples(
                col_name,
                f"expected {union_type_names}, got {df[col_name].dtype}",
                samples,
                total_invalid,
                repr,
            )

        for validator in validators:
            apply_validator(df[col_name], validator, col_name)
        return

    if base_type not in TYPE_CHECKERS:
        raise ValidationError(f"unsupported type: {base_type}", column_name=col_name)

    checker = TYPE_CHECKERS[base_type]
    col_series = df[col_name]
    col_dtype = col_series.dtype
    if not checker.dtype(col_dtype):
        # polars builds an all-null column as pl.Null since it has no values to infer a dtype
        # from. Validators reject pl.Null, while nulls of the declared type are skipped by them.
        if not (is_optional and col_dtype == pl.Null):
            _raise_type_error_with_samples(df, col_name, checker, base_type, col_dtype)
        col_series = col_series.cast(TYPE_TO_DTYPE[base_type])

    if not is_optional and col_series.null_count() > 0:
        raise ValidationError("is non-optional but contains null values", column_name=col_name)

    for validator in validators:
        apply_validator(col_series, validator, col_name)
