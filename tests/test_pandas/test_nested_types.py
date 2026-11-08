from typing import Annotated, Any, Literal, Protocol

import pandas as pd
import pytest

from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame
from pavise.types import NotRequiredColumn
from pavise.validators import Custom, In, MaxLen, MinLen, Range, Regex

pa = pytest.importorskip("pyarrow")


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


ADDRESS = {"city": "Tokyo", "zip": "100-0001"}
ADDRESS_TYPE = pa.struct([("city", pa.string()), ("zip", pa.string())])


def arrow_frame(
    column: str, values: list[Any], arrow_type: Any, index: list[Any] | None = None
) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(values, dtype=pd.ArrowDtype(arrow_type), index=index)})


@pytest.mark.parametrize(
    "schema, df",
    [
        (TagsSchema, arrow_frame("tags", [["a", "b"], []], pa.list_(pa.string()))),
        (TagsSchema, arrow_frame("tags", [["a"]], pa.large_list(pa.large_string()))),
        (TagsSchema, arrow_frame("tags", [["a", "b"]], pa.list_(pa.string(), 2))),
        (TagsSchema, arrow_frame("tags", [[]], pa.list_(pa.null()))),
        (AddressSchema, arrow_frame("address", [ADDRESS], ADDRESS_TYPE)),
        (HistorySchema, arrow_frame("history", [[ADDRESS], []], pa.list_(ADDRESS_TYPE))),
        (OptionalTagsSchema, arrow_frame("tags", [["a", None], None], pa.list_(pa.string()))),
    ],
)
def test_nested_column_accepts_matching_arrow_dtype(schema: type, df: pd.DataFrame) -> None:
    result = DataFrame[schema](df)
    assert isinstance(result, pd.DataFrame)


def test_nested_date_accepts_arrow_timestamp() -> None:
    """Like a top-level date column, a nested date accepts timestamps"""
    from datetime import date, datetime

    class Schema(Protocol):
        days: list[date]

    df = arrow_frame("days", [[datetime(2020, 1, 1)]], pa.list_(pa.timestamp("us")))
    result = DataFrame[Schema](df)
    assert isinstance(result, pd.DataFrame)


def test_struct_ignores_field_order_and_extra_fields() -> None:
    struct = pa.struct([("zip", pa.string()), ("country", pa.string()), ("city", pa.string())])
    df = arrow_frame("address", [{**ADDRESS, "country": "JP"}], struct)
    result = DataFrame[AddressSchema](df)
    assert isinstance(result, pd.DataFrame)


@pytest.mark.parametrize(
    "schema, df, message",
    [
        (
            TagsSchema,
            arrow_frame("tags", ["a"], pa.string()),
            r"Column 'tags': expected list\[str\], got string\[pyarrow\]",
        ),
        (
            TagsSchema,
            arrow_frame("tags", [[1]], pa.list_(pa.int64())),
            r"Column 'tags\[\]': expected str, got int64",
        ),
        (
            AddressSchema,
            arrow_frame(
                "address",
                [{"city": "Tokyo", "zip": 1}],
                pa.struct([("city", pa.string()), ("zip", pa.int64())]),
            ),
            r"Column 'address.zip': expected str, got int64",
        ),
        (
            HistorySchema,
            arrow_frame(
                "history",
                [[{"city": 1, "zip": "x"}]],
                pa.list_(pa.struct([("city", pa.int64()), ("zip", pa.string())])),
            ),
            r"Column 'history\[\].city': expected str, got int64",
        ),
    ],
)
def test_nested_column_rejects_mismatched_arrow_dtype(
    schema: type, df: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValidationError, match=f"^{message}$"):
        DataFrame[schema](df)


def test_object_column_of_lists_explains_arrow_dtype() -> None:
    df = pd.DataFrame({"tags": [["a"], ["b"]]})
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[TagsSchema](df)
    assert str(exc_info.value) == (
        "Column 'tags': expected list[str], got object\n"
        "  (pavise checks list and struct columns only with pd.ArrowDtype, "
        "e.g. dtype=pd.ArrowDtype(pa.list_(pa.string())))"
    )


def test_optional_nested_column_accepts_untyped_all_null_column() -> None:
    df = pd.DataFrame({"tags": [None, None]})
    result = DataFrame[OptionalTagsSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_struct_rejects_missing_field() -> None:
    df = arrow_frame("address", [{"city": "Tokyo"}], pa.struct([("city", pa.string())]))
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

    df = arrow_frame("address", [{"city": "Tokyo"}], pa.struct([("city", pa.string())]))
    result = DataFrame[Schema](df)
    assert isinstance(result, pd.DataFrame)


def test_strict_rejects_extra_struct_fields() -> None:
    struct = pa.struct([("city", pa.string()), ("zip", pa.string()), ("country", pa.string())])
    df = arrow_frame("address", [{**ADDRESS, "country": "JP"}], struct)
    with pytest.raises(
        ValidationError,
        match=r"^Column 'address': Strict mode: unexpected struct fields \['country'\]$",
    ):
        DataFrame[AddressSchema](df, strict=True)


def test_list_rejects_null_elements_with_row_labels() -> None:
    class Schema(Protocol):
        tags: list[str] | None

    df = arrow_frame(
        "tags",
        [["a", None], ["b"], [], None, [None]],
        pa.list_(pa.string()),
        index=["v", "w", "x", "y", "z"],
    )
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](df)
    assert str(exc_info.value) == (
        "Column 'tags[]': is non-optional but contains null values\n\n"
        "Sample invalid values (showing first 2 of 2):\n"
        "  Row v: ['a', None]\n"
        "  Row z: [None]"
    )
    assert exc_info.value.path == "tags[]"


def test_list_column_rejects_null_rows() -> None:
    df = arrow_frame("tags", [["a"], None], pa.list_(pa.string()))
    with pytest.raises(ValidationError, match=r"Column 'tags': is non-optional but contains null"):
        DataFrame[TagsSchema](df)


def test_struct_rejects_null_field_values_but_not_null_rows() -> None:
    class Schema(Protocol):
        address: Address | None

    df = arrow_frame("address", [ADDRESS, None, {"city": None, "zip": "100-0002"}], ADDRESS_TYPE)
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](df)
    assert str(exc_info.value) == (
        "Column 'address.city': is non-optional but contains null values\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 2: {'city': None, 'zip': '100-0002'}"
    )


def test_null_check_reports_row_of_deeply_nested_null() -> None:
    df = arrow_frame(
        "history",
        [[ADDRESS], [], [ADDRESS, {"city": "Osaka", "zip": None}]],
        pa.list_(ADDRESS_TYPE),
    )
    with pytest.raises(ValidationError, match=r"Column 'history\[\].zip'(.|\n)*Row 2: "):
        DataFrame[HistorySchema](df)


@pytest.mark.parametrize(
    "schema, expected",
    [
        (TagsSchema, pa.list_(pa.string())),
        (AddressSchema, ADDRESS_TYPE),
        (HistorySchema, pa.list_(ADDRESS_TYPE)),
    ],
)
def test_make_empty_builds_arrow_dtypes(schema: type, expected: Any) -> None:
    (dtype,) = DataFrame[schema].make_empty().dtypes
    assert dtype == pd.ArrowDtype(expected)


class ArrowInstanceSchema(Protocol):
    tags: pd.ArrowDtype(pa.list_(pa.string()))  # type: ignore[valid-type]


def test_arrow_dtype_instance_annotation_accepts_equal_dtype() -> None:
    df = arrow_frame("tags", [["a"]], pa.list_(pa.string()))
    result = DataFrame[ArrowInstanceSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_arrow_dtype_instance_annotation_rejects_different_dtype() -> None:
    df = arrow_frame("tags", [[1]], pa.list_(pa.int64()))
    with pytest.raises(
        ValidationError,
        match=r"^Column 'tags': expected list<item: string>\[pyarrow\], "
        r"got list<item: int64>\[pyarrow\]$",
    ):
        DataFrame[ArrowInstanceSchema](df)


def test_arrow_dtype_instance_annotation_in_make_empty() -> None:
    (dtype,) = DataFrame[ArrowInstanceSchema].make_empty().dtypes
    assert dtype == pd.ArrowDtype(pa.list_(pa.string()))


class RolesSchema(Protocol):
    roles: list[Literal["admin", "staff"]]


def test_literal_inside_list_rejects_values_outside_literal() -> None:
    df = arrow_frame("roles", [["admin"], ["staff", "guest"]], pa.list_(pa.string()))
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[RolesSchema](df)
    assert str(exc_info.value) == (
        "Column 'roles[]': expected one of ('admin', 'staff'), got invalid values\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 1: ['staff', 'guest']"
    )


def test_literal_inside_list_accepts_values_in_literal() -> None:
    df = arrow_frame("roles", [["admin"], ["staff"], []], pa.list_(pa.string()))
    result = DataFrame[RolesSchema](df)
    assert isinstance(result, pd.DataFrame)


def test_literal_inside_list_matches_narrow_integer_type() -> None:
    class Schema(Protocol):
        codes: list[Literal[1, 2]]

    df = arrow_frame("codes", [[1], [3]], pa.list_(pa.int8()))
    with pytest.raises(ValidationError, match=r"Column 'codes\[\]': expected one of \(1, 2\)"):
        DataFrame[Schema](df)


def test_literal_with_none_inside_list_accepts_nulls() -> None:
    class Schema(Protocol):
        roles: list[Literal["admin", None]]

    df = arrow_frame("roles", [["admin", None]], pa.list_(pa.string()))
    result = DataFrame[Schema](df)
    assert isinstance(result, pd.DataFrame)


class ForwardRefHistorySchema(Protocol):
    history: list["Address"]


def test_list_of_forward_referenced_struct() -> None:
    """Python 3.10's get_type_hints leaves the string inside list[...] unresolved"""
    df = arrow_frame("history", [[ADDRESS]], pa.list_(ADDRESS_TYPE))
    result = DataFrame[ForwardRefHistorySchema](df)
    assert isinstance(result, pd.DataFrame)
    (dtype,) = DataFrame[ForwardRefHistorySchema].make_empty().dtypes
    assert dtype == pd.ArrowDtype(pa.list_(ADDRESS_TYPE))


def _validation_message(df: pd.DataFrame, schema: type) -> str:
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[schema](df)
    return str(exc_info.value)


def test_element_validator_reports_value_and_row_label() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int, Range(0, 100)]]

    df = arrow_frame(
        "scores", [[50], [50, 120], [], [-3]], pa.list_(pa.int64()), index=["a", "b", "c", "d"]
    )
    assert _validation_message(df, Schema) == (
        "Column 'scores[]': values must be in range [0, 100]\n\n"
        "Sample invalid values (showing first 2 of 2):\n"
        "  Row b: 120 (in [50, 120])\n"
        "  Row d: -3 (in [-3])"
    )


def test_element_validator_accepts_valid_values_and_skips_nulls() -> None:
    class Schema(Protocol):
        scores: list[Annotated[int | None, Range(0, 100)]] | None

    df = arrow_frame("scores", [[0, None, 100], [], None], pa.list_(pa.int64()))
    result = DataFrame[Schema](df)
    assert isinstance(result, pd.DataFrame)


def test_struct_field_validator_inside_list() -> None:
    class CheckedAddress(Protocol):
        city: str
        zip: Annotated[str, Regex(r"\d{3}-\d{4}")]

    class Schema(Protocol):
        history: list[CheckedAddress]

    df = arrow_frame(
        "history", [[ADDRESS], [{"city": "Osaka", "zip": "abc"}]], pa.list_(ADDRESS_TYPE)
    )
    assert _validation_message(df, Schema) == (
        "Column 'history[].zip': contains values that don't match the pattern\n\n"
        "Sample invalid values (showing first 1 of 1):\n"
        "  Row 1: 'abc' (in [{'city': 'Osaka', 'zip': 'abc'}])"
    )


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
        items: Annotated[list[int], validator]  # type: ignore[valid-type]

    df = arrow_frame("items", data, pa.list_(pa.int64()))
    assert _validation_message(df, Schema) == message


def test_list_length_validator_on_fixed_size_list() -> None:
    class Schema(Protocol):
        pair: Annotated[list[int], MaxLen(1)]

    df = arrow_frame("pair", [[1, 2]], pa.list_(pa.int64(), 2))
    assert "contains lists longer than maximum length" in _validation_message(df, Schema)


def test_custom_on_list_and_struct_columns_receives_python_values() -> None:
    class Schema(Protocol):
        items: Annotated[list[int], Custom(lambda v: sum(v) < 10, "sum must be below 10")]

    df = arrow_frame("items", [[1, 2], [5, 6]], pa.list_(pa.int64()))
    assert _validation_message(df, Schema).endswith("Row 1: [5, 6]")

    class AddressSchema(Protocol):
        address: Annotated[Address, Custom(lambda a: a["city"] != a["zip"], "city equals zip")]

    df = arrow_frame("address", [ADDRESS, {"city": "x", "zip": "x"}], ADDRESS_TYPE)
    assert _validation_message(df, AddressSchema).endswith("Row 1: {'city': 'x', 'zip': 'x'}")


def test_element_string_validators() -> None:
    class Schema(Protocol):
        tags: list[Annotated[str, MinLen(2), In(["bb", "a"])]]

    df = arrow_frame("tags", [["a", "bb"]], pa.list_(pa.string()))
    assert _validation_message(df, Schema).endswith("Row 0: 'a' (length: 1) (in ['a', 'bb'])")


def test_bound_validator_on_string_elements_names_the_path() -> None:
    class Schema(Protocol):
        tags: list[Annotated[str, Range(0, 1)]]

    df = arrow_frame("tags", [["a"]], pa.list_(pa.string()))
    with pytest.raises(ValidationError) as exc_info:
        DataFrame[Schema](df)
    assert str(exc_info.value).startswith("Column 'tags[]': Range requires a numeric or temporal")
    assert exc_info.value.column_name == "tags"


def test_custom_on_list_does_not_see_null_elements_the_annotation_rejects() -> None:
    class Schema(Protocol):
        items: Annotated[list[int], Custom(lambda v: sum(v) < 10, "sum must be below 10")]

    df = arrow_frame("items", [[1, None]], pa.list_(pa.int64()))
    message = _validation_message(df, Schema)
    assert message.startswith("Column 'items[]': is non-optional but contains null values")
