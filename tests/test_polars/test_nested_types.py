from typing import Annotated, Any, Literal, Protocol

import polars as pl
import pytest

from pavise.checks import dataframe_check
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.types import NotRequiredColumn
from pavise.validators import MinLen, Range


class Address(Protocol):
    city: str
    zip: str


class TagsSchema(Protocol):
    tags: list[str]


class OptionalTagsSchema(Protocol):
    tags: list[str | None] | None


class AddressSchema(Protocol):
    address: Address


class HistorySchema(Protocol):
    history: list[Address]


class MatrixSchema(Protocol):
    matrix: list[list[int]]


ADDRESS = {"city": "Tokyo", "zip": "100-0001"}


@pytest.mark.parametrize(
    "schema, data",
    [
        (TagsSchema, {"tags": [["a", "b"], []]}),
        (AddressSchema, {"address": [ADDRESS]}),
        (HistorySchema, {"history": [[ADDRESS, ADDRESS], []]}),
        (MatrixSchema, {"matrix": [[[1, 2], [3]], []]}),
        (OptionalTagsSchema, {"tags": [["a", None], None]}),
    ],
)
def test_nested_column_accepts_matching_dtype(schema: type, data: dict[str, Any]) -> None:
    result = DataFrame[schema](pl.DataFrame(data))
    assert isinstance(result, pl.DataFrame)


def test_list_accepts_array_column() -> None:
    """pl.Array is a list of fixed length, so list[X] accepts it"""
    df = pl.DataFrame(
        {"matrix": [[[1, 2], [3, 4]]]}, schema={"matrix": pl.List(pl.Array(pl.Int64, 2))}
    )
    result = DataFrame[MatrixSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_list_accepts_list_of_null_dtype() -> None:
    """polars infers List(Null) from empty lists, which fit any element type"""
    df = pl.DataFrame({"tags": [[], []]})
    assert df.schema["tags"] == pl.List(pl.Null)
    result = DataFrame[TagsSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_struct_ignores_field_order_and_extra_fields() -> None:
    df = pl.DataFrame({"address": [{"zip": "100-0001", "country": "JP", "city": "Tokyo"}]})
    result = DataFrame[AddressSchema](df)
    assert isinstance(result, pl.DataFrame)


@pytest.mark.parametrize(
    "schema, data, message",
    [
        (TagsSchema, {"tags": ["a"]}, r"Column 'tags': expected list\[str\], got String"),
        (TagsSchema, {"tags": [[1]]}, r"Column 'tags\[\]': expected str, got Int64"),
        (AddressSchema, {"address": ["Tokyo"]}, r"Column 'address': expected Address, got String"),
        (
            AddressSchema,
            {"address": [{"city": "Tokyo", "zip": 1000001}]},
            r"Column 'address.zip': expected str, got Int64",
        ),
        (
            HistorySchema,
            {"history": [[{"city": "Tokyo", "zip": 1000001}]]},
            r"Column 'history\[\].zip': expected str, got Int64",
        ),
        (
            MatrixSchema,
            {"matrix": [["a"]]},
            r"Column 'matrix\[\]': expected list\[int\], got String",
        ),
    ],
)
def test_nested_column_rejects_mismatched_dtype(
    schema: type, data: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=f"^{message}$"):
        DataFrame[schema](pl.DataFrame(data))


def test_type_error_inside_list_sets_column_name_and_path() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[HistorySchema](pl.DataFrame({"history": [[{"city": 1, "zip": "x"}]]}))
    assert exc_info.value.column_name == "history"
    assert exc_info.value.path == "history[].city"


def test_struct_rejects_missing_field() -> None:
    df = pl.DataFrame({"address": [{"city": "Tokyo"}]})
    with pytest.raises(
        ValidationError,
        match=r"^Column 'address': missing struct field 'zip' \(found fields: \['city'\]\)$",
    ):
        DataFrame[AddressSchema](df)


def test_struct_accepts_missing_not_required_field() -> None:
    class PartialAddress(Protocol):
        city: str
        zip: NotRequiredColumn[str]

    class Schema(Protocol):
        address: PartialAddress

    result = DataFrame[Schema](pl.DataFrame({"address": [{"city": "Tokyo"}]}))
    assert isinstance(result, pl.DataFrame)


def test_strict_rejects_extra_struct_fields() -> None:
    df = pl.DataFrame({"address": [{**ADDRESS, "country": "JP"}]})
    with pytest.raises(
        ValidationError,
        match=r"^Column 'address': Strict mode: unexpected struct fields \['country'\]$",
    ):
        DataFrame[AddressSchema](df, strict=True)


def test_strict_rejects_extra_struct_fields_inside_list() -> None:
    df = pl.DataFrame({"history": [[{**ADDRESS, "country": "JP"}]]})
    with pytest.raises(ValidationError, match=r"^Column 'history\[\]': Strict mode"):
        DataFrame[HistorySchema](df, strict=True)


def test_list_column_rejects_null_rows() -> None:
    df = pl.DataFrame({"tags": [["a"], None]})
    with pytest.raises(
        ValidationError, match=r"Column 'tags': is non-optional but contains null values"
    ):
        DataFrame[TagsSchema](df)


def test_list_rejects_null_elements_with_samples() -> None:
    df = pl.DataFrame({"tags": [["a", None], ["b"], [], None, [None]]})

    class Schema(Protocol):
        tags: list[str] | None

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](df)
    assert str(exc_info.value) == (
        "Column 'tags[]': is non-optional but contains null values\n\n"
        "Sample invalid values (showing first 2 of 2):\n"
        "  Row 0: ['a', None]\n"
        "  Row 4: [None]"
    )
    assert exc_info.value.path == "tags[]"


def test_struct_rejects_null_field_values() -> None:
    df = pl.DataFrame(
        {"address": [ADDRESS, {"city": None, "zip": "100-0002"}]},
    )
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[AddressSchema](df)
    assert str(exc_info.value) == (
        "Column 'address.city': is non-optional but contains null values\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 1: {'city': None, 'zip': '100-0002'}"
    )


def test_null_struct_row_is_not_reported_as_null_fields() -> None:
    """A null struct row has null fields too, but only the row itself is checked"""

    class Schema(Protocol):
        address: Address | None

    df = pl.DataFrame({"address": [ADDRESS, None]})
    result = DataFrame[Schema](df)
    assert isinstance(result, pl.DataFrame)


def test_null_check_reports_row_of_deeply_nested_null() -> None:
    df = pl.DataFrame(
        {"history": [[ADDRESS], [], [ADDRESS, {"city": "Osaka", "zip": None}]]},
    )
    with pytest.raises(
        ValidationError, match=r"Column 'history\[\].zip'.*\n.*\n.*1 of 1.*\n  Row 2"
    ):
        DataFrame[HistorySchema](df)


def test_null_check_inside_array_column() -> None:
    df = pl.DataFrame({"matrix": [[[1, None]]]}, schema={"matrix": pl.List(pl.Array(pl.Int64, 2))})
    with pytest.raises(ValidationError, match=r"Column 'matrix\[\]\[\]': is non-optional"):
        DataFrame[MatrixSchema](df)


def test_optional_list_accepts_all_null_column() -> None:
    df = pl.DataFrame({"tags": [None, None]})
    assert df.schema["tags"] == pl.Null
    result = DataFrame[OptionalTagsSchema](df)
    assert isinstance(result, pl.DataFrame)


def test_validators_inside_nested_columns_are_not_supported_yet() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int, Range(0, 100)]]

    with pytest.raises(ValidationError, match=r"Column 'scores': validators .* not supported yet"):
        DataFrame[Schema](pl.DataFrame({"scores": [[1]]}))


def test_validators_on_list_column_are_not_supported_yet() -> None:
    class Schema(Protocol):
        items: Annotated[list[int], MinLen(1)]

    with pytest.raises(ValidationError, match=r"Column 'items': validators .* not supported yet"):
        DataFrame[Schema](pl.DataFrame({"items": [[1]]}))


def test_lazyframe_rejects_mismatched_nested_dtype() -> None:
    lf = pl.LazyFrame({"history": [[{"city": "Tokyo", "zip": 1}]]})
    with pytest.raises(ValidationError, match=r"Column 'history\[\].zip': expected str"):
        LazyFrame[HistorySchema](lf)


def test_lazyframe_checks_nulls_on_collect() -> None:
    lf = LazyFrame[TagsSchema](pl.LazyFrame({"tags": [["a", None]]}))
    with pytest.raises(ValidationError, match=r"Column 'tags\[\]': is non-optional"):
        lf.collect()


def test_lazyframe_strict_rejects_extra_struct_fields() -> None:
    lf = pl.LazyFrame({"address": [{**ADDRESS, "country": "JP"}]})
    with pytest.raises(ValidationError, match="unexpected struct fields"):
        LazyFrame[AddressSchema](lf, strict=True)


@pytest.mark.parametrize(
    "schema, expected",
    [
        (TagsSchema, pl.List(pl.Utf8)),
        (AddressSchema, pl.Struct({"city": pl.Utf8, "zip": pl.Utf8})),
        (HistorySchema, pl.List(pl.Struct({"city": pl.Utf8, "zip": pl.Utf8}))),
    ],
)
def test_make_empty_builds_nested_dtypes(schema: type, expected: pl.DataType) -> None:
    (column,) = DataFrame[schema].make_empty().schema.values()
    assert column == expected
    (column,) = LazyFrame[schema].make_empty().collect_schema().values()
    assert column == expected


class DtypeInstanceSchema(Protocol):
    tags: pl.List(pl.Utf8)  # type: ignore[valid-type]
    address: pl.Struct({"city": pl.Utf8})  # type: ignore[valid-type]
    pair: pl.Array(pl.Int64, 2)  # type: ignore[valid-type]


def _dtype_instance_df(**overrides: pl.Series) -> pl.DataFrame:
    columns = {
        "tags": pl.Series([["a"]]),
        "address": pl.Series([{"city": "Tokyo"}]),
        "pair": pl.Series([[1, 2]], dtype=pl.Array(pl.Int64, 2)),
    }
    return pl.DataFrame({**columns, **overrides})


def test_dtype_instance_annotation_accepts_equal_dtype() -> None:
    result = DataFrame[DtypeInstanceSchema](_dtype_instance_df())
    assert isinstance(result, pl.DataFrame)


def test_dtype_instance_annotation_rejects_different_dtype() -> None:
    df = _dtype_instance_df(tags=pl.Series([[1]]))
    with pytest.raises(
        ValidationError, match=r"^Column 'tags': expected List\(String\), got List\(Int64\)$"
    ):
        DataFrame[DtypeInstanceSchema](df)


def test_dtype_instance_annotation_in_lazyframe_and_make_empty() -> None:
    with pytest.raises(ValidationError, match=r"Column 'pair': expected Array"):
        LazyFrame[DtypeInstanceSchema](
            _dtype_instance_df(pair=pl.Series([[1, 2, 3]], dtype=pl.Array(pl.Int64, 3))).lazy()
        )
    assert DataFrame[DtypeInstanceSchema].make_empty().schema["pair"] == pl.Array(pl.Int64, 2)


def test_dtype_instance_as_list_element() -> None:
    class Schema(Protocol):
        values: list[pl.Int32]  # type: ignore[valid-type]

    df = pl.DataFrame({"values": [[1]]}, schema={"values": pl.List(pl.Int64)})
    with pytest.raises(ValidationError, match=r"Column 'values\[\]': expected Int32, got Int64"):
        DataFrame[Schema](df)


class RolesSchema(Protocol):
    roles: list[Literal["admin", "staff"]]


def test_literal_inside_list_rejects_values_outside_literal() -> None:
    df = pl.DataFrame({"roles": [["admin"], ["staff", "guest"]]})
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[RolesSchema](df)
    assert str(exc_info.value) == (
        "Column 'roles[]': expected one of ('admin', 'staff'), got invalid values\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 1: ['staff', 'guest']"
    )


def test_literal_inside_list_accepts_values_in_literal() -> None:
    result = DataFrame[RolesSchema](pl.DataFrame({"roles": [["admin"], ["staff"], []]}))
    assert isinstance(result, pl.DataFrame)


def test_literal_with_none_inside_list_accepts_nulls() -> None:
    class Schema(Protocol):
        roles: list[Literal["admin", None]]

    result = DataFrame[Schema](pl.DataFrame({"roles": [["admin", None]]}))
    assert isinstance(result, pl.DataFrame)


def test_dataframe_check_on_nested_protocol_is_ignored() -> None:
    """dataframe_check runs on whole DataFrames, so a Protocol used as a struct skips it"""

    @dataframe_check(lambda df: df["city"] != "", name="city_not_empty")
    class CheckedAddress(Protocol):
        city: str

    class Schema(Protocol):
        address: CheckedAddress

    result = DataFrame[Schema](pl.DataFrame({"address": [{"city": ""}]}))
    assert isinstance(result, pl.DataFrame)
