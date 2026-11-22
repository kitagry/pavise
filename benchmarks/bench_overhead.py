"""Measure how much time pavise validation adds compared with doing the same checks by hand.

Not collected by pytest. See benchmarks/README.md for how to read the results. Run with:

    uv run python benchmarks/bench_overhead.py --rows 10000 --rows 1000000
    # only some scenarios and backends, e.g. for the before/after table of a PR:
    uv run python benchmarks/bench_overhead.py --backend polars --scenario Unique
    # compare with pandera too (installed only for this run):
    uv run --with 'pandera[polars]' python benchmarks/bench_overhead.py
    # list the scenarios:
    uv run python benchmarks/bench_overhead.py --list

Each scenario is timed three ways on the same frame:

- hand: the checks pavise performs, written directly with the backend's API (dtype comparison,
  null count, vectorized comparisons). It is the cost the checks themselves have, so
  ``pavise - hand`` is what pavise adds on top.
- pavise: ``DataFrame[Schema](frame)`` (or ``LazyFrame[Schema](lf)``), with the class
  subscripted once outside the loop.
- pandera: the equivalent ``DataFrameSchema.validate(frame)``, when pandera is importable.

Scenarios named "error: ..." time how long pavise takes to reject an invalid column.

Prints one Markdown table per row count with the best per-call time.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import timeit
from collections.abc import Callable
from datetime import date, datetime
from typing import Annotated, Any, Literal, Protocol

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from typing_extensions import TypedDict

from pavise.checks import Check, Checked, Key, UniqueKeys
from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame as PdDataFrame
from pavise.polars import DataFrame as PlDataFrame
from pavise.polars import LazyFrame as PlLazyFrame
from pavise.types import NotRequiredColumn
from pavise.validators import In, MaxLen, MinLen, Range, Regex, Unique

BACKENDS = ("pandas", "polars", "lazy")
EAGER = ("pandas", "polars")

# --- schemas ---------------------------------------------------------------------------------


class IntS(Protocol):
    id: int


class FloatS(Protocol):
    f: float


class StrS(Protocol):
    s: str


class OptStrS(Protocol):
    opt_s: str | None


class DatetimeS(Protocol):
    dt: datetime


class DateS(Protocol):
    d: date


class OptIntS(Protocol):
    oi: int | None


class LiteralS(Protocol):
    lit: Literal["a", "b", "c"]


class UnionIntColumnS(Protocol):
    id: int | str


class UnionMixedS(Protocol):
    mixed: int | str


class OptUnionMixedS(Protocol):
    opt_mixed: int | str | None


class RangeS(Protocol):
    i: Annotated[int, Range(0, 99)]


class RangeUniqueS(Protocol):
    id: Annotated[int, Range(0, 10**12), Unique()]


class StrValidatorsS(Protocol):
    s: Annotated[str, MinLen(1), MaxLen(20), Regex(r"user\d+")]


class InS(Protocol):
    lit: Annotated[str, In(["a", "b", "c"])]


class ListS(Protocol):
    lst: list[int]


class PointP(Protocol):
    x: int
    y: str


class PointTD(TypedDict):
    x: int
    y: str


@dataclasses.dataclass
class PointDC:
    x: int
    y: str


class StructProtocolS(Protocol):
    st: PointP


class StructTypedDictS(Protocol):
    st: PointTD


class StructDataclassS(Protocol):
    st: PointDC


class KeyColumns(Protocol):
    id: int
    i: int


class UniqueKeysS(KeyColumns, UniqueKeys[Key[Literal["id", "i"]]], Protocol): ...


class OrderColumns(Protocol):
    id: int
    id2: int


class IdBeforeId2(Check):
    def __call__(self, df: Any) -> Any:
        return df["id"] < df["id2"]


class CheckedS(OrderColumns, Checked[IdBeforeId2], Protocol): ...


class NotRequiredPresentS(Protocol):
    id: int
    oi: NotRequiredColumn[int | None]


class NotRequiredAbsentS(Protocol):
    id: int
    missing: NotRequiredColumn[int]


class WideS(Protocol):
    id: int
    i: Annotated[int, Range(0, 99)]
    f: float
    s: str
    dt: datetime
    oi: int | None
    lit: Literal["a", "b", "c"]


WIDE_COLUMNS = ["id", "i", "f", "s", "dt", "oi", "lit"]


class StrGotIntS(Protocol):
    id: str


class IntGotStrS(Protocol):
    s: int


class UnionGotFloatS(Protocol):
    f: int | str


# --- scenarios -------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    columns: list[str]
    schema: type
    strict: bool = False
    expect_error: bool = False
    backends: tuple[str, ...] = BACKENDS


SCENARIOS = [
    Scenario("int", ["id"], IntS),
    Scenario("float", ["f"], FloatS),
    Scenario("str", ["s"], StrS),
    Scenario("Optional[str]", ["opt_s"], OptStrS),
    Scenario("datetime", ["dt"], DatetimeS),
    Scenario("date", ["d"], DateS),
    Scenario("Optional[int]", ["oi"], OptIntS),
    Scenario("Literal", ["lit"], LiteralS),
    Scenario("Union[int, str] (int column)", ["id"], UnionIntColumnS),
    # pl.Object columns cannot go through a LazyFrame plan reliably, so they stay eager
    Scenario("Union[int, str] (mixed)", ["mixed"], UnionMixedS, backends=EAGER),
    Scenario("Union[int, str, None] (mixed)", ["opt_mixed"], OptUnionMixedS, backends=EAGER),
    Scenario("Range", ["i"], RangeS),
    Scenario("Range+Unique", ["id"], RangeUniqueS),
    Scenario("MinLen+MaxLen+Regex", ["s"], StrValidatorsS),
    Scenario("In", ["lit"], InS),
    Scenario("list[int]", ["lst"], ListS),
    Scenario("struct (Protocol)", ["st"], StructProtocolS),
    Scenario("struct (TypedDict)", ["st"], StructTypedDictS),
    Scenario("struct (dataclass)", ["st"], StructDataclassS),
    Scenario("UniqueKeys[id, i]", ["id", "i"], UniqueKeysS),
    Scenario("Checked[id < id2]", ["id", "id2"], CheckedS),
    Scenario("NotRequiredColumn (present)", ["id", "oi"], NotRequiredPresentS),
    Scenario("NotRequiredColumn (absent)", ["id"], NotRequiredAbsentS),
    Scenario("wide 7 cols", WIDE_COLUMNS, WideS),
    Scenario("wide 7 cols strict", WIDE_COLUMNS, WideS, strict=True),
    # A LazyFrame reports these only on collect(), which the eager rows already measure
    Scenario("error: str got int", ["id"], StrGotIntS, expect_error=True, backends=EAGER),
    Scenario("error: int got str", ["s"], IntGotStrS, expect_error=True, backends=EAGER),
    Scenario("error: Union got float", ["f"], UnionGotFloatS, expect_error=True, backends=EAGER),
]


# --- data ------------------------------------------------------------------------------------


def _arrow_columns(n: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    ids = np.arange(n, dtype="int64")
    lengths = rng.integers(0, 5, n)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype("int32")
    every_tenth = ids % 10 == 0
    return {
        "id": pa.array(ids),
        "id2": pa.array(ids + 1),
        "i": pa.array(rng.integers(0, 100, n)),
        "f": pa.array(rng.random(n)),
        "s": pa.array([f"user{k % 1000}" for k in range(n)]),
        "opt_s": pa.array([None if k % 10 == 0 else f"user{k % 1000}" for k in range(n)]),
        "dt": pa.array(
            np.datetime64("2000-01-01") + ids.astype("timedelta64[s]"), pa.timestamp("us")
        ),
        "d": pa.array(np.datetime64("2000-01-01") + (ids % 3650).astype("timedelta64[D]")),
        "oi": pa.array(ids, mask=every_tenth),
        "lit": pa.array(np.array(["a", "b", "c"])[ids % 3]),
        "lst": pa.ListArray.from_arrays(
            pa.array(offsets), pa.array(rng.integers(0, 100, int(offsets[-1])))
        ),
        "st": pa.StructArray.from_arrays(
            [pa.array(ids), pa.array([f"p{k % 100}" for k in range(n)])], names=["x", "y"]
        ),
    }


def _mixed(n: int, with_nulls: bool) -> list[Any]:
    return [None if with_nulls and k % 10 == 0 else (k if k % 2 else str(k)) for k in range(n)]


def build_pandas(cols: dict[str, Any], n: int) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            name: cols[name].to_numpy(zero_copy_only=False)
            for name in ["id", "id2", "i", "f", "dt", "lit"]
        }
    )
    # Built from Python objects so that strings get pandas' default dtype (object on pandas 2,
    # str on pandas 3); dates stay object and nullable integers are Int64
    for name in ["s", "opt_s", "d"]:
        df[name] = np.array(cols[name].to_pylist(), dtype=object)
    df["oi"] = pd.array(cols["oi"].to_pandas(), dtype="Int64")
    df["mixed"] = np.array(_mixed(n, with_nulls=False), dtype=object)
    df["opt_mixed"] = np.array(_mixed(n, with_nulls=True), dtype=object)
    # pavise checks list and struct columns only as pd.ArrowDtype
    df["lst"] = pd.Series(cols["lst"], dtype=pd.ArrowDtype(cols["lst"].type))
    df["st"] = pd.Series(cols["st"], dtype=pd.ArrowDtype(cols["st"].type))
    return df


def build_polars(cols: dict[str, Any], n: int) -> pl.DataFrame:
    df = pl.from_arrow(pa.table(cols))
    assert isinstance(df, pl.DataFrame)
    return df.with_columns(
        pl.Series("mixed", _mixed(n, with_nulls=False), dtype=pl.Object),
        pl.Series("opt_mixed", _mixed(n, with_nulls=True), dtype=pl.Object),
    )


# --- hand-written checks ---------------------------------------------------------------------
#
# Each does what pavise does for the scenario and raises AssertionError on failure.

HandCheck = Callable[[Any], object]


def _ok(cond: Any) -> None:
    if not cond:
        raise AssertionError


def _pd_no_null(s: Any) -> None:
    _ok(not s.isna().any())


def _pd_int(df: Any, col: str = "id") -> None:
    _ok(pd.api.types.is_integer_dtype(df[col]))
    _pd_no_null(df[col])


def _pd_str(df: Any, col: str = "s") -> None:
    _pd_no_null(df[col])
    _ok(pd.api.types.infer_dtype(df[col], skipna=True) == "string")


def _pd_types_in(s: Any, types: tuple[type, ...], null_ok: bool) -> None:
    if not null_ok:
        _pd_no_null(s)
    _ok(s.dropna().map(type).isin(types).all())


def _pd_wide(df: Any) -> None:
    _pd_int(df, "id")
    _pd_int(df, "i")
    _ok(not ((df["i"] < 0) | (df["i"] > 99)).any())
    _ok(pd.api.types.is_float_dtype(df["f"]))
    _pd_no_null(df["f"])
    _pd_str(df, "s")
    _ok(pd.api.types.is_datetime64_any_dtype(df["dt"]))
    _pd_no_null(df["dt"])
    _ok(pd.api.types.is_integer_dtype(df["oi"]))
    _pd_no_null(df["lit"])
    _ok(df["lit"].isin(["a", "b", "c"]).all())


def _pd_arrow_type(df: Any, col: str, expected: Any) -> None:
    _ok(df[col].dtype == pd.ArrowDtype(expected))
    _pd_no_null(df[col])


_PD_POINT = pa.struct([("x", pa.int64()), ("y", pa.string())])
_STRUCT_SCENARIOS = ("struct (Protocol)", "struct (TypedDict)", "struct (dataclass)")

PD_HAND: dict[str, HandCheck] = {
    "int": _pd_int,
    "float": lambda df: (_ok(pd.api.types.is_float_dtype(df["f"])), _pd_no_null(df["f"])),
    "str": _pd_str,
    "Optional[str]": lambda df: _ok(pd.api.types.infer_dtype(df["opt_s"], skipna=True) == "string"),
    "datetime": lambda df: (
        _ok(pd.api.types.is_datetime64_any_dtype(df["dt"])),
        _pd_no_null(df["dt"]),
    ),
    "date": lambda df: (
        _pd_no_null(df["d"]),
        _ok(pd.api.types.infer_dtype(df["d"], skipna=True) == "date"),
    ),
    "Optional[int]": lambda df: _ok(pd.api.types.is_integer_dtype(df["oi"])),
    "Literal": lambda df: (_pd_no_null(df["lit"]), _ok(df["lit"].isin(["a", "b", "c"]).all())),
    "Union[int, str] (int column)": _pd_int,
    "Union[int, str] (mixed)": lambda df: _pd_types_in(df["mixed"], (int, str), null_ok=False),
    "Union[int, str, None] (mixed)": lambda df: _pd_types_in(
        df["opt_mixed"], (int, str), null_ok=True
    ),
    "Range": lambda df: (_pd_int(df, "i"), _ok(not ((df["i"] < 0) | (df["i"] > 99)).any())),
    "Range+Unique": lambda df: (
        _pd_int(df, "id"),
        _ok(not ((df["id"] < 0) | (df["id"] > 10**12)).any()),
        _ok(not df["id"].duplicated().any()),
    ),
    "MinLen+MaxLen+Regex": lambda df: (
        _pd_str(df),
        _ok(df["s"].str.len().between(1, 20).all()),
        _ok(df["s"].str.fullmatch(r"user\d+").all()),
    ),
    "In": lambda df: (_pd_str(df, "lit"), _ok(df["lit"].isin(["a", "b", "c"]).all())),
    "list[int]": lambda df: _pd_arrow_type(df, "lst", pa.list_(pa.int64())),
    **dict.fromkeys(_STRUCT_SCENARIOS, lambda df: _pd_arrow_type(df, "st", _PD_POINT)),
    "UniqueKeys[id, i]": lambda df: (
        _pd_int(df, "id"),
        _pd_int(df, "i"),
        _ok(not df.duplicated(["id", "i"]).any()),
    ),
    "Checked[id < id2]": lambda df: (
        _pd_int(df, "id"),
        _pd_int(df, "id2"),
        _ok((df["id"] < df["id2"]).all()),
    ),
    "NotRequiredColumn (present)": lambda df: (
        _pd_int(df),
        _ok(pd.api.types.is_integer_dtype(df["oi"])),
    ),
    "NotRequiredColumn (absent)": lambda df: (_pd_int(df), _ok("missing" not in df.columns)),
    "wide 7 cols": _pd_wide,
    "wide 7 cols strict": lambda df: (_pd_wide(df), _ok(set(df.columns) == set(WIDE_COLUMNS))),
}


def _pl_one_pass(df: pl.DataFrame, dtypes: dict[str, Any], exprs: list[pl.Expr]) -> None:
    """Compare dtypes, then evaluate every value check in one select."""
    for col, dtype in dtypes.items():
        _ok(df.schema[col] == dtype)
    if exprs:
        _ok(all(df.select(e.alias(str(k)) for k, e in enumerate(exprs)).row(0)))


def _pl_no_null(*cols: str) -> list[pl.Expr]:
    return [pl.col(c).null_count() == 0 for c in cols]


def _pl_types_in(s: pl.Series, types: tuple[type, ...], null_ok: bool) -> None:
    _ok(all((v is None and null_ok) or isinstance(v, types) for v in s.to_list()))


_PL_POINT = pl.Struct({"x": pl.Int64, "y": pl.String})
_PL_POINT_EXPRS = [
    *_pl_no_null("st"),
    pl.col("st").struct.field("x").null_count() == 0,
    pl.col("st").struct.field("y").null_count() == 0,
]
_PL_WIDE_DTYPES = {
    "id": pl.Int64,
    "i": pl.Int64,
    "f": pl.Float64,
    "s": pl.String,
    "dt": pl.Datetime("us"),
    "oi": pl.Int64,
    "lit": pl.String,
}
_PL_WIDE_EXPRS = [
    *_pl_no_null("id", "i", "f", "s", "dt", "lit"),
    pl.col("i").is_between(0, 99).all(),
    pl.col("lit").is_in(["a", "b", "c"]).all(),
]


def _pl_lit(df: pl.DataFrame) -> None:
    _pl_one_pass(
        df, {"lit": pl.String}, [*_pl_no_null("lit"), pl.col("lit").is_in(["a", "b", "c"]).all()]
    )


PL_HAND: dict[str, HandCheck] = {
    "int": lambda df: _pl_one_pass(df, {"id": pl.Int64}, _pl_no_null("id")),
    "float": lambda df: _pl_one_pass(df, {"f": pl.Float64}, _pl_no_null("f")),
    "str": lambda df: _pl_one_pass(df, {"s": pl.String}, _pl_no_null("s")),
    "Optional[str]": lambda df: _pl_one_pass(df, {"opt_s": pl.String}, []),
    "datetime": lambda df: _pl_one_pass(df, {"dt": pl.Datetime("us")}, _pl_no_null("dt")),
    "date": lambda df: _pl_one_pass(df, {"d": pl.Date}, _pl_no_null("d")),
    "Optional[int]": lambda df: _pl_one_pass(df, {"oi": pl.Int64}, []),
    "Literal": _pl_lit,
    "Union[int, str] (int column)": lambda df: _pl_one_pass(
        df, {"id": pl.Int64}, _pl_no_null("id")
    ),
    "Union[int, str] (mixed)": lambda df: _pl_types_in(df["mixed"], (int, str), null_ok=False),
    "Union[int, str, None] (mixed)": lambda df: _pl_types_in(
        df["opt_mixed"], (int, str), null_ok=True
    ),
    "Range": lambda df: _pl_one_pass(
        df, {"i": pl.Int64}, [*_pl_no_null("i"), pl.col("i").is_between(0, 99).all()]
    ),
    "Range+Unique": lambda df: _pl_one_pass(
        df,
        {"id": pl.Int64},
        [
            *_pl_no_null("id"),
            pl.col("id").is_between(0, 10**12).all(),
            ~pl.col("id").is_duplicated().any(),
        ],
    ),
    "MinLen+MaxLen+Regex": lambda df: _pl_one_pass(
        df,
        {"s": pl.String},
        [
            *_pl_no_null("s"),
            pl.col("s").str.len_chars().is_between(1, 20).all(),
            pl.col("s").str.contains(r"^(?:user\d+)$").all(),
        ],
    ),
    "In": _pl_lit,
    "list[int]": lambda df: _pl_one_pass(
        df,
        {"lst": pl.List(pl.Int64)},
        [
            *_pl_no_null("lst"),
            # polars 1.x explodes an empty list to a null, so empty lists are skipped
            pl.col("lst").filter(pl.col("lst").list.len() > 0).explode().null_count() == 0,
        ],
    ),
    **dict.fromkeys(
        _STRUCT_SCENARIOS, lambda df: _pl_one_pass(df, {"st": _PL_POINT}, _PL_POINT_EXPRS)
    ),
    "UniqueKeys[id, i]": lambda df: _pl_one_pass(
        df,
        {"id": pl.Int64, "i": pl.Int64},
        [*_pl_no_null("id", "i"), ~pl.struct("id", "i").is_duplicated().any()],
    ),
    "Checked[id < id2]": lambda df: _pl_one_pass(
        df,
        {"id": pl.Int64, "id2": pl.Int64},
        [*_pl_no_null("id", "id2"), (pl.col("id") < pl.col("id2")).all()],
    ),
    "NotRequiredColumn (present)": lambda df: _pl_one_pass(
        df, {"id": pl.Int64, "oi": pl.Int64}, _pl_no_null("id")
    ),
    "NotRequiredColumn (absent)": lambda df: (
        _pl_one_pass(df, {"id": pl.Int64}, _pl_no_null("id")),
        _ok("missing" not in df.columns),
    ),
    "wide 7 cols": lambda df: _pl_one_pass(df, _PL_WIDE_DTYPES, _PL_WIDE_EXPRS),
    "wide 7 cols strict": lambda df: (
        _pl_one_pass(df, _PL_WIDE_DTYPES, _PL_WIDE_EXPRS),
        _ok(set(df.columns) == set(WIDE_COLUMNS)),
    ),
}


# --- pandera ---------------------------------------------------------------------------------


def _import_pandera(backend: str) -> Any:
    """pandera's module for the backend, or None when pandera is not installed."""
    # Imported by name so that pandera stays out of the project's dependencies
    module = "pandera.pandas" if backend == "pandas" else "pandera.polars"
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def pandera_schemas(backend: str) -> dict[str, Any]:
    """pandera schemas equivalent to the scenarios it can express, by scenario name."""
    m = _import_pandera(backend)
    if m is None:
        return {}
    is_pandas = backend == "pandas"
    int_: Any = "int64" if is_pandas else pl.Int64
    opt_int: Any = "Int64" if is_pandas else pl.Int64
    float_: Any = "float64" if is_pandas else pl.Float64
    str_: Any = str if is_pandas else pl.String
    dt: Any = "datetime64[ns]" if is_pandas else pl.Datetime("us")
    column = m.Column
    check = m.Check

    def schema(columns: dict[str, Any], **kwargs: Any) -> Any:
        return m.DataFrameSchema(columns, **kwargs)

    if is_pandas:
        id_before_id2 = check(lambda df: df["id"] < df["id2"])
    else:
        id_before_id2 = check(lambda data: data.lazyframe.select(pl.col("id") < pl.col("id2")))

    wide = {
        "id": column(int_),
        "i": column(int_, check.in_range(0, 99)),
        "f": column(float_),
        "s": column(str_),
        "dt": column(dt),
        "oi": column(opt_int, nullable=True),
        "lit": column(str_, check.isin(["a", "b", "c"])),
    }
    lit = schema({"lit": column(str_, check.isin(["a", "b", "c"]))})
    schemas = {
        "int": schema({"id": column(int_)}),
        "float": schema({"f": column(float_)}),
        "str": schema({"s": column(str_)}),
        "Optional[str]": schema({"opt_s": column(str_, nullable=True)}),
        "datetime": schema({"dt": column(dt)}),
        "Optional[int]": schema({"oi": column(opt_int, nullable=True)}),
        "Literal": lit,
        "Range": schema({"i": column(int_, check.in_range(0, 99))}),
        "Range+Unique": schema({"id": column(int_, check.in_range(0, 10**12), unique=True)}),
        "MinLen+MaxLen+Regex": schema(
            {"s": column(str_, [check.str_length(1, 20), check.str_matches(r"user\d+$")])}
        ),
        "In": lit,
        "UniqueKeys[id, i]": schema({"id": column(int_), "i": column(int_)}, unique=["id", "i"]),
        "Checked[id < id2]": schema(
            {"id": column(int_), "id2": column(int_)}, checks=id_before_id2
        ),
        "NotRequiredColumn (present)": schema(
            {"id": column(int_), "oi": column(opt_int, nullable=True, required=False)}
        ),
        "NotRequiredColumn (absent)": schema(
            {"id": column(int_), "missing": column(int_, required=False)}
        ),
        "wide 7 cols": schema(wide),
        "wide 7 cols strict": schema(wide, strict=True),
    }
    if not is_pandas:
        # pandas keeps dates as Python objects, which pandera has no plain dtype for
        schemas["date"] = schema({"d": column(pl.Date)})
    return schemas


# --- timing ----------------------------------------------------------------------------------


def best_time(fn: Callable[[], object], repeat: int) -> float:
    """Best per-call seconds; each sample runs long enough (>= 0.2s) to be measurable."""
    fn()
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(repeat, number)) / number


def _expecting_error(fn: Callable[[], object]) -> Callable[[], object]:
    def run() -> None:
        try:
            fn()
        except ValidationError:
            return
        raise AssertionError("expected ValidationError")

    return run


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def ratio(a: float | None, b: float | None) -> str:
    if a is None or b is None or b == 0:
        return "-"
    return f"{a / b:.1f}x"


def _difference(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _print_row(
    backend: str, name: str, hand: float | None, pavise: float, pandera: float | None
) -> None:
    print(
        f"| {backend} | {name} | {fmt(hand)} | {fmt(pavise)} | {fmt(pandera)} "
        f"| {fmt(_difference(pavise, hand))} | {ratio(pavise, hand)} | {ratio(pandera, hand)} |",
        flush=True,
    )


def _time_with(fn: Callable[[Any], object] | None, arg: Any, repeat: int) -> float | None:
    """Time ``fn(arg)``, or None when there is no ``fn`` to compare with."""
    if fn is None:
        return None
    return best_time(lambda: fn(arg), repeat)


def run_eager(
    backend: str,
    frame: Any,
    scenarios: list[Scenario],
    pandera: dict[str, Any],
    repeat: int,
) -> None:
    pavise_df: Any = PdDataFrame if backend == "pandas" else PlDataFrame
    hand_checks = PD_HAND if backend == "pandas" else PL_HAND
    for sc in scenarios:
        sub = frame[sc.columns] if backend == "pandas" else frame.select(sc.columns)
        cls = pavise_df[sc.schema]
        strict = sc.strict

        def validate(sub: Any = sub, cls: Any = cls, strict: bool = strict) -> object:
            return cls(sub, strict=strict)

        if sc.expect_error:
            _print_row(backend, sc.name, None, best_time(_expecting_error(validate), repeat), None)
            continue
        hand = hand_checks.get(sc.name)
        model = pandera.get(sc.name)
        t_hand = _time_with(hand, sub, repeat)
        t_pavise = best_time(validate, repeat)
        t_pandera = _time_with(None if model is None else model.validate, sub, repeat)
        _print_row(backend, sc.name, t_hand, t_pavise, t_pandera)


def _after_collect(check: HandCheck | None) -> HandCheck | None:
    """Run a hand-written check on what a LazyFrame collects to."""
    if check is None:
        return None
    return lambda lf: check(lf.collect())


def _then_collect(validate: Callable[[Any], Any]) -> HandCheck:
    return lambda lf: validate(lf).collect()


def run_lazy(
    frame: pl.DataFrame, scenarios: list[Scenario], pandera: dict[str, Any], repeat: int
) -> None:
    """LazyFrame[Schema](lf) checks only the schema; collect() validates the values too."""
    for sc in scenarios:
        # A filter keeps the plan from being a bare in-memory scan. "id" has no nulls, so the
        # filter keeps every row, including the nulls of the Optional scenarios.
        lf = frame.lazy().filter(pl.col("id").is_not_null()).select(sc.columns)
        cls: Any = PlLazyFrame[sc.schema]
        hand = PL_HAND.get(sc.name)
        model = pandera.get(sc.name)
        strict = sc.strict

        t_hand = best_time(lambda: lf.collect_schema(), repeat)
        t_pavise = best_time(lambda: cls(lf, strict=strict), repeat)
        # pandera checks only a LazyFrame's schema by default, like pavise's constructor
        t_pandera = _time_with(None if model is None else model.validate, lf, repeat)
        _print_row("polars lazy (schema)", sc.name, t_hand, t_pavise, t_pandera)

        t_hand = _time_with(_after_collect(hand), lf, repeat)
        t_pavise = best_time(lambda: cls(lf, strict=strict).collect(), repeat)
        t_pandera = _time_with(None if model is None else _then_collect(model.validate), lf, repeat)
        _print_row("polars lazy (collect)", sc.name, t_hand, t_pavise, t_pandera)


def select_scenarios(backend: str, patterns: list[str] | None) -> list[Scenario]:
    """Scenarios for the backend whose names contain any of the patterns (case-insensitive)."""
    return [
        sc
        for sc in SCENARIOS
        if backend in sc.backends
        and (not patterns or any(p.lower() in sc.name.lower() for p in patterns))
    ]


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", type=_positive_int, action="append", help="row count; repeatable (default: 10000)"
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        action="append",
        help="repeatable (default: all); lazy is polars.LazyFrame",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        metavar="TEXT",
        help="run scenarios whose name contains TEXT (case-insensitive); repeatable",
    )
    parser.add_argument(
        "--repeat", type=_positive_int, default=3, help="samples per timing (best is kept)"
    )
    parser.add_argument(
        "--no-pandera", action="store_true", help="skip pandera even when it is installed"
    )
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    args = parser.parse_args()
    backends: list[str] = args.backend or list(BACKENDS)

    if args.list:
        for sc in SCENARIOS:
            runs_on = [b for b in sc.backends if b in backends]
            if runs_on:
                print(f"{sc.name}  ({', '.join(runs_on)})")
        return

    selected = {b: select_scenarios(b, args.scenario) for b in backends}
    if not any(selected.values()):
        parser.error("no scenario matches; see --list")

    for n in args.rows or [10_000]:
        cols = _arrow_columns(n)
        print(f"\n### rows={n:,} (best of {args.repeat}, per call)\n")
        print(
            "| backend | scenario | hand | pavise | pandera | pavise − hand "
            "| pavise / hand | pandera / hand |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|")
        if selected.get("pandas"):
            pandera = {} if args.no_pandera else pandera_schemas("pandas")
            run_eager("pandas", build_pandas(cols, n), selected["pandas"], pandera, args.repeat)
        if selected.get("polars") or selected.get("lazy"):
            frame = build_polars(cols, n)
            pandera = {} if args.no_pandera else pandera_schemas("polars")
            if selected.get("polars"):
                run_eager("polars", frame, selected["polars"], pandera, args.repeat)
            if selected.get("lazy"):
                run_lazy(frame, selected["lazy"], pandera, args.repeat)


if __name__ == "__main__":
    main()
