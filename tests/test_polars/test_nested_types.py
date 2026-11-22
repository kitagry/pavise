from dataclasses import dataclass
from typing import Annotated, Any, Literal, NamedTuple, Protocol, TypedDict

import polars as pl
import pytest
from typing_extensions import NotRequired

from pavise.checks import dataframe_check
from pavise.exceptions import ValidationError
from pavise.polars import DataFrame, LazyFrame
from pavise.types import NotRequiredColumn
from pavise.validators import Custom, Ge, MaxLen, MinLen, Range, Regex


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
        values: list[pl.Int32]

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


def test_mixed_literal_inside_list_checks_elements() -> None:
    """Literal values of several types cannot form one polars Series"""

    class Schema(Protocol):
        codes: list[Literal[1, "a"]]

    DataFrame[Schema](pl.DataFrame({"codes": [[1], [1, 1]]}))
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pl.DataFrame({"codes": [[1], [1, 2]]}))
    assert exc_info.value.invalid_samples == [(1, [1, 2])]


def test_dataframe_check_on_nested_protocol_is_ignored() -> None:
    """dataframe_check runs on whole DataFrames, so a Protocol used as a struct skips it"""

    @dataframe_check(lambda df: df["city"] != "", name="city_not_empty")
    class CheckedAddress(Protocol):
        city: str

    class Schema(Protocol):
        address: CheckedAddress

    result = DataFrame[Schema](pl.DataFrame({"address": [{"city": ""}]}))
    assert isinstance(result, pl.DataFrame)


class ForwardRefHistorySchema(Protocol):
    history: list["Address"]


def test_list_of_forward_referenced_struct() -> None:
    """Python 3.10's get_type_hints leaves the string inside list[...] unresolved"""
    df = pl.DataFrame({"history": [[{"city": "Tokyo", "zip": 1}]]})
    with pytest.raises(ValidationError, match=r"Column 'history\[\].zip': expected str"):
        DataFrame[ForwardRefHistorySchema](df)


def _validation_message(df: pl.DataFrame, schema: type) -> str:
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)
    return str(exc_info.value)


def test_element_validator_reports_value_and_row() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int, Range(0, 100)]]

    df = pl.DataFrame({"scores": [[50], [50, 120], [], [-3]]})
    assert _validation_message(df, Schema) == (
        "Column 'scores[]': values must be in range [0, 100]\n\n"
        "Sample invalid values (showing first 2 of 2):\n"
        "  Row 1: 120 (in [50, 120])\n"
        "  Row 3: -3 (in [-3])"
    )


def test_element_validator_error_sets_column_name_path_and_samples() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int, Range(0, 100)]]

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pl.DataFrame({"scores": [[1], [101]]}))
    assert exc_info.value.column_name == "scores"
    assert exc_info.value.path == "scores[]"
    assert exc_info.value.invalid_samples == [(1, 101)]


def test_element_validator_accepts_valid_values_and_skips_nulls() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int | None, Range(0, 100)]] | None

    result = DataFrame[Schema](pl.DataFrame({"scores": [[0, None, 100], [], None]}))
    assert isinstance(result, pl.DataFrame)


def test_struct_field_validator_inside_list() -> None:
    class CheckedAddress(Protocol):
        city: str
        zip: Annotated[str, Regex(r"\d{3}-\d{4}")]

    class Schema(Protocol):
        history: list[CheckedAddress]

    df = pl.DataFrame({"history": [[ADDRESS], [{"city": "Osaka", "zip": "abc"}]]})
    assert _validation_message(df, Schema) == (
        "Column 'history[].zip': contains values that don't match the pattern\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 1: 'abc' (in [{'city': 'Osaka', 'zip': 'abc'}])"
    )


def test_element_string_length_validator() -> None:
    class Schema(Protocol):
        tags: list[Annotated[str, MinLen(2)]]

    df = pl.DataFrame({"tags": [["a", "bb"]]})
    assert _validation_message(df, Schema).endswith("Row 0: 'a' (length: 1) (in ['a', 'bb'])")


def test_validator_on_deeply_nested_elements() -> None:
    class Schema(Protocol):
        matrix: list[list[Annotated[int, Ge(0)]]]

    df = pl.DataFrame({"matrix": [[[1], []], [[2, -1]]]})
    message = _validation_message(df, Schema)
    assert message.startswith("Column 'matrix[][]': values must be >= 0")
    assert message.endswith("Row 1: -1 (in [[2, -1]])")


@pytest.mark.parametrize(
    "validator, data, message",
    [
        (
            MinLen(1),
            [[1], []],
            "Column 'items': contains lists shorter than minimum length\n\n"
            "Sample invalid values (showing first 1 of 1):\n"
            "  Row 1: [] (length: 0)",
        ),
        (
            MaxLen(2),
            [[1, 2, 3], [1]],
            "Column 'items': contains lists longer than maximum length\n\n"
            "Sample invalid values (showing first 1 of 1):\n"
            "  Row 0: [1, 2, 3] (length: 3)",
        ),
    ],
)
def test_list_length_validators(validator: Any, data: list[Any], message: str) -> None:
    class Schema(Protocol):
        items: Annotated[list[int], validator]

    assert _validation_message(pl.DataFrame({"items": data}), Schema) == message


def test_list_length_validator_on_array_column() -> None:
    class Schema(Protocol):
        pair: Annotated[list[int], MaxLen(1)]

    df = pl.DataFrame({"pair": [[1, 2]]}, schema={"pair": pl.Array(pl.Int64, 2)})
    assert "contains lists longer than maximum length" in _validation_message(df, Schema)


def test_list_length_validator_on_list_elements() -> None:
    class Schema(Protocol):
        groups: list[Annotated[list[int], MinLen(1)]]

    df = pl.DataFrame({"groups": [[[1], []]]})
    assert _validation_message(df, Schema) == (
        "Column 'groups[]': contains lists shorter than minimum length\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 0: [] (length: 0) (in [[1], []])"
    )


def test_custom_on_list_column_receives_python_lists() -> None:
    seen: list[Any] = []

    def small_sum(value: Any) -> bool:
        seen.append(value)
        return sum(value) < 10

    class Schema(Protocol):
        items: Annotated[list[int], Custom(small_sum, "sum must be below 10")]

    message = _validation_message(pl.DataFrame({"items": [[1, 2], [5, 6]]}), Schema)
    assert message.startswith("Column 'items': sum must be below 10")
    assert message.endswith("Row 1: [5, 6]")
    assert seen == [[1, 2], [5, 6]]


def test_custom_on_struct_column_receives_dicts() -> None:
    class Schema(Protocol):
        address: Annotated[Address, Custom(lambda a: a["city"] != a["zip"], "city equals zip")]

    df = pl.DataFrame({"address": [ADDRESS, {"city": "x", "zip": "x"}]})
    assert _validation_message(df, Schema).endswith("Row 1: {'city': 'x', 'zip': 'x'}")


def test_bound_validator_on_string_elements_names_the_path() -> None:
    class Schema(Protocol):
        tags: list[Annotated[str, Range(0, 1)]]

    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](pl.DataFrame({"tags": [["a"]]}))
    assert str(exc_info.value) == (
        "Column 'tags[]': Range requires a numeric or temporal column, got String"
    )
    assert exc_info.value.column_name == "tags"


def test_lazyframe_rejects_bound_validator_on_string_elements() -> None:
    class Schema(Protocol):
        tags: list[Annotated[str, Range(0, 1)]]

    with pytest.raises(ValidationError, match=r"Column 'tags\[\]': Range requires"):
        LazyFrame[Schema](pl.LazyFrame({"tags": [["a"]]}))


def test_lazyframe_checks_element_validators_on_collect() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int, Range(0, 100)]]

    lf = LazyFrame[Schema](pl.LazyFrame({"scores": [[101]]}))
    with pytest.raises(ValidationError, match=r"Column 'scores\[\]': values must be in range"):
        lf.collect()


def _validate_frame(kind: str, schema: type, df: pl.DataFrame) -> None:
    if kind == "DataFrame":
        DataFrame[schema](df)
    else:
        LazyFrame[schema](df.lazy()).collect()


@pytest.mark.parametrize("kind", ["DataFrame", "LazyFrame"])
def test_element_errors_map_values_to_rows_past_nulls_and_empty_lists(kind: str) -> None:
    """Null rows, null inner lists and empty lists before an invalid value keep its row"""

    class Schema(Protocol):
        matrix: list[list[Annotated[int, Ge(0)]] | None] | None

    df = pl.DataFrame({"matrix": [None, [[1], [], None], [], [[2, -1], None, [-2]], [[-3]]]})
    with pytest.raises(ValidationError) as exc_info:
        _validate_frame(kind, Schema, df)
    assert str(exc_info.value) == (
        "Column 'matrix[][]': values must be >= 0\n\n"
        "Sample invalid values (showing first 3 of 3):\n"
        "  Row 3: -1 (in [[2, -1], None, [-2]])\n"
        "  Row 3: -2 (in [[2, -1], None, [-2]])\n"
        "  Row 4: -3 (in [[-3]])"
    )


@pytest.mark.parametrize("kind", ["DataFrame", "LazyFrame"])
def test_null_elements_map_to_rows_past_null_rows_and_empty_lists(kind: str) -> None:
    class Schema(Protocol):
        tags: list[str] | None

    df = pl.DataFrame({"tags": [None, [], ["a"], [None, "b", None], [], [None]]})
    with pytest.raises(ValidationError) as exc_info:
        _validate_frame(kind, Schema, df)
    assert str(exc_info.value) == (
        "Column 'tags[]': is non-optional but contains null values\n\n"
        "Sample invalid values (showing first 2 of 2):\n"
        "  Row 3: [None, 'b', None]\n"
        "  Row 5: [None]"
    )


@pytest.mark.parametrize("kind", ["DataFrame", "LazyFrame"])
def test_element_errors_map_values_to_rows_without_empty_lists(kind: str) -> None:
    class Schema(Protocol):
        history: list[Annotated[str, MinLen(2)]]

    df = pl.DataFrame({"history": [["ab", "cd"], ["x"], ["ef", "y", "z"]]})
    with pytest.raises(ValidationError) as exc_info:
        _validate_frame(kind, Schema, df)
    assert exc_info.value.invalid_samples == [(1, "x"), (2, "y"), (2, "z")]


@pytest.mark.parametrize("kind", ["DataFrame", "LazyFrame"])
def test_element_errors_map_values_to_rows_in_array_column(kind: str) -> None:
    class Schema(Protocol):
        pairs: list[Annotated[int, Ge(0)]]

    df = pl.DataFrame(
        {"pairs": [[1, 2], [3, -4], [-5, 6]]}, schema={"pairs": pl.Array(pl.Int64, 2)}
    )
    with pytest.raises(ValidationError) as exc_info:
        _validate_frame(kind, Schema, df)
    assert exc_info.value.invalid_samples == [(1, -4), (2, -5)]


def test_custom_on_list_does_not_see_null_elements_the_annotation_rejects() -> None:
    class Schema(Protocol):
        items: Annotated[list[int], Custom(lambda v: sum(v) < 10, "sum must be below 10")]

    message = _validation_message(pl.DataFrame({"items": [[1, None]]}), Schema)
    assert message.startswith("Column 'items[]': is non-optional but contains null values")


def test_custom_on_list_accepts_column_with_only_nulls() -> None:
    class Schema(Protocol):
        items: Annotated[list[int], Custom(lambda v: sum(v) < 10, "sum must be below 10")] | None

    DataFrame[Schema](pl.DataFrame({"items": pl.Series([None], dtype=pl.List(pl.Int64))}))


def test_custom_on_elements_accepts_column_with_only_empty_lists() -> None:
    class Schema(Protocol):
        items: list[Annotated[int, Custom(lambda v: v > 0, "must be positive")]]

    DataFrame[Schema](pl.DataFrame({"items": pl.Series([[], []], dtype=pl.List(pl.Int64))}))


class AddressDict(TypedDict):
    city: str
    zip: NotRequired[Annotated[str, Regex(r"^\d{3}-\d{4}$")]]


@dataclass
class AddressData:
    city: str
    zip: str | None = None


class AddressDictSchema(Protocol):
    address: AddressDict


class AddressDataSchema(Protocol):
    address: AddressData


@pytest.mark.parametrize(
    "schema, data",
    [
        (AddressDictSchema, {"address": [ADDRESS]}),
        (AddressDictSchema, {"address": [{"city": "Tokyo"}]}),
        (AddressDataSchema, {"address": [ADDRESS, {"city": "Osaka", "zip": None}]}),
    ],
)
def test_typeddict_and_dataclass_structs_accept_matching_data(
    schema: type, data: dict[str, Any]
) -> None:
    result = DataFrame[schema](pl.DataFrame(data))
    assert isinstance(result, pl.DataFrame)


@pytest.mark.parametrize(
    "schema, data, message",
    [
        (
            AddressDictSchema,
            {"address": [{"zip": "100-0001"}]},
            r"Column 'address': missing struct field 'city' \(found fields: \['zip'\]\)",
        ),
        (
            AddressDictSchema,
            {"address": [{"city": "Tokyo", "zip": 1000001}]},
            r"Column 'address.zip': expected str, got Int64",
        ),
        # A default is for building instances, not a field that may be missing
        (
            AddressDataSchema,
            {"address": [{"city": "Tokyo"}]},
            r"Column 'address': missing struct field 'zip' \(found fields: \['city'\]\)",
        ),
    ],
)
def test_typeddict_and_dataclass_structs_reject_invalid_data(
    schema: type, data: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=f"^{message}$"):
        DataFrame[schema](pl.DataFrame(data))


class Visit(TypedDict):
    place: AddressData | None
    tags: NotRequired[list[str]]


@dataclass
class Customer:
    visits: list[Visit]


class CustomerSchema(Protocol):
    customer: Customer | None


def test_protocol_typeddict_and_dataclass_structs_nest_in_each_other() -> None:
    # Tags keep the struct holding a null place from being all nulls, which polars 1.0 reads as
    # a null struct
    visits = [{"place": ADDRESS, "tags": []}, {"place": None, "tags": ["x"]}]
    df = pl.DataFrame({"customer": [{"visits": visits}, None]})
    assert isinstance(DataFrame[CustomerSchema](df), pl.DataFrame)

    df = pl.DataFrame({"customer": [{"visits": [{"place": {"city": 1, "zip": None}}]}]})
    with pytest.raises(ValidationError, match=r"^Column 'customer.visits\[\].place.city': "):
        DataFrame[CustomerSchema](df)


@pytest.mark.parametrize(
    "schema, expected",
    [
        (AddressDictSchema, pl.Struct({"city": pl.Utf8, "zip": pl.Utf8})),
        (AddressDataSchema, pl.Struct({"city": pl.Utf8, "zip": pl.Utf8})),
    ],
)
def test_make_empty_builds_typeddict_and_dataclass_structs(
    schema: type, expected: pl.DataType
) -> None:
    (column,) = DataFrame[schema].make_empty().schema.values()
    assert column == expected


def test_custom_on_dataclass_struct_receives_dicts() -> None:
    received: list[Any] = []

    class Schema(Protocol):
        address: Annotated[AddressData, Custom(lambda v: received.append(v) is None, "x")]

    DataFrame[Schema](pl.DataFrame({"address": [ADDRESS]}))
    assert received == [ADDRESS]


class AddressTuple(NamedTuple):
    city: str


def test_named_tuple_column_reports_supported_struct_schemas() -> None:
    class Schema(Protocol):
        address: AddressTuple

    with pytest.raises(
        ValidationError,
        match=r"^Column 'address': unsupported type: AddressTuple "
        r"\(struct schemas must be a Protocol, TypedDict or dataclass\)$",
    ):
        DataFrame[Schema](pl.DataFrame({"address": [ADDRESS]}))


class SameShapedAddressDict(TypedDict):
    city: str
    zip: NotRequired[Annotated[str, Regex(r"^\d{3}-\d{4}$")]]


class SameShapedAddressDictSchema(Protocol):
    address: SameShapedAddressDict


def count_addresses(df: DataFrame[AddressDictSchema]) -> int:
    return len(df)


def test_same_shaped_typeddict_structs_are_interchangeable() -> None:
    # pyright verifies this call: TypedDicts with the same keys are the same type to it
    df = DataFrame[SameShapedAddressDictSchema](pl.DataFrame({"address": [ADDRESS]}))
    assert count_addresses(df) == 1
