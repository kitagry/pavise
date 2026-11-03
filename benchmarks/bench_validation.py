"""Benchmark pavise validation on large DataFrames.

Not collected by pytest. Run with:

    uv run python benchmarks/bench_validation.py [--rows 1000000] [--repeat 3]

Prints a Markdown table of the best wall-clock time (seconds) per scenario.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Annotated, Literal, Protocol

import numpy as np
import pandas as pd
import polars as pl

from pavise.exceptions import ValidationError
from pavise.pandas import DataFrame as PdDataFrame
from pavise.polars import DataFrame as PlDataFrame
from pavise.validators import In, MaxLen, MinLen, Range, Regex, Unique


class IntSchema(Protocol):
    v: int


class FloatSchema(Protocol):
    v: float


class StrSchema(Protocol):
    v: str


class DatetimeSchema(Protocol):
    v: datetime


class DateSchema(Protocol):
    v: date


class OptionalIntSchema(Protocol):
    v: int | None


class OptionalStrSchema(Protocol):
    v: str | None


class UnionSchema(Protocol):
    v: int | str


class OptionalUnionSchema(Protocol):
    v: int | str | None


class LiteralSchema(Protocol):
    v: Literal["a", "b", "c"]


class NumericValidatorsSchema(Protocol):
    v: Annotated[int, Range(0, 10_000_000), Unique()]


class StrValidatorsSchema(Protocol):
    v: Annotated[str, MinLen(1), MaxLen(20), Regex(r"s\d+"), In(["s1", "s2", "s3"])]


Scenario = tuple[str, Callable[[], object], bool]


def _pandas_scenarios(n: int) -> list[Scenario]:
    rng = np.random.default_rng(0)
    ints = pd.DataFrame({"v": np.arange(n, dtype="int64")})
    floats = pd.DataFrame({"v": rng.random(n)})
    strs = pd.DataFrame({"v": np.array([f"s{i % 3 + 1}" for i in range(n)], dtype=object)})
    opt_strs = strs.copy()
    opt_strs.loc[::10, "v"] = None
    datetimes = pd.DataFrame({"v": pd.date_range("2000-01-01", periods=n, freq="s")})
    base = date(2000, 1, 1)
    dates = pd.DataFrame({"v": np.array([base + timedelta(days=i % 3650) for i in range(n)])})
    opt_ints = pd.DataFrame({"v": pd.array(np.arange(n), dtype="Int64")})
    opt_ints.loc[::10, "v"] = pd.NA
    mixed = pd.DataFrame({"v": np.array([i if i % 2 else str(i) for i in range(n)], dtype=object)})
    opt_mixed = mixed.copy()
    opt_mixed.loc[::10, "v"] = None
    literals = pd.DataFrame({"v": np.array(["a", "b", "c"] * (n // 3) + ["a"] * (n % 3))})
    bad_str = pd.DataFrame({"v": np.arange(n, dtype="int64")})
    bad_union = pd.DataFrame({"v": rng.random(n)})

    return [
        ("int", lambda: PdDataFrame[IntSchema](ints), False),
        ("float", lambda: PdDataFrame[FloatSchema](floats), False),
        ("str (object)", lambda: PdDataFrame[StrSchema](strs), False),
        ("Optional[str] (object)", lambda: PdDataFrame[OptionalStrSchema](opt_strs), False),
        ("datetime", lambda: PdDataFrame[DatetimeSchema](datetimes), False),
        ("date (object)", lambda: PdDataFrame[DateSchema](dates), False),
        ("Optional[int] (Int64)", lambda: PdDataFrame[OptionalIntSchema](opt_ints), False),
        ("Union[int, str]", lambda: PdDataFrame[UnionSchema](mixed), False),
        ("Union[int, str, None]", lambda: PdDataFrame[OptionalUnionSchema](opt_mixed), False),
        ("Literal", lambda: PdDataFrame[LiteralSchema](literals), False),
        ("Range+Unique", lambda: PdDataFrame[NumericValidatorsSchema](ints), False),
        ("MinLen+MaxLen+Regex+In", lambda: PdDataFrame[StrValidatorsSchema](strs), False),
        ("error: str got int", lambda: PdDataFrame[StrSchema](bad_str), True),
        ("error: Union got float", lambda: PdDataFrame[UnionSchema](bad_union), True),
    ]


def _polars_scenarios(n: int) -> list[Scenario]:
    rng = np.random.default_rng(0)
    ints = pl.DataFrame({"v": np.arange(n, dtype="int64")})
    floats = pl.DataFrame({"v": rng.random(n)})
    strs = pl.DataFrame({"v": [f"s{i % 3 + 1}" for i in range(n)]})
    opt_strs = strs.with_columns(
        pl.when(pl.int_range(pl.len()) % 10 == 0).then(None).otherwise(pl.col("v")).alias("v")
    )
    datetimes = pl.DataFrame(
        {
            "v": pl.datetime_range(
                datetime(2000, 1, 1),
                datetime(2000, 1, 1) + timedelta(seconds=n - 1),
                "1s",
                eager=True,
            )
        }
    )
    dates = datetimes.select(pl.col("v").dt.date())
    opt_ints = ints.with_columns(
        pl.when(pl.col("v") % 10 == 0).then(None).otherwise(pl.col("v")).alias("v")
    )
    union_ints = ints
    mixed = pl.DataFrame(
        {"v": pl.Series([i if i % 2 else str(i) for i in range(n)], dtype=pl.Object)}
    )
    literals = pl.DataFrame({"v": ["a", "b", "c"] * (n // 3) + ["a"] * (n % 3)})
    bad_int = pl.DataFrame({"v": [str(i) for i in range(n)]})
    bad_union = pl.DataFrame({"v": rng.random(n)})

    return [
        ("int", lambda: PlDataFrame[IntSchema](ints), False),
        ("float", lambda: PlDataFrame[FloatSchema](floats), False),
        ("str", lambda: PlDataFrame[StrSchema](strs), False),
        ("Optional[str]", lambda: PlDataFrame[OptionalStrSchema](opt_strs), False),
        ("datetime", lambda: PlDataFrame[DatetimeSchema](datetimes), False),
        ("date", lambda: PlDataFrame[DateSchema](dates), False),
        ("Optional[int]", lambda: PlDataFrame[OptionalIntSchema](opt_ints), False),
        ("Union[int, str] (Int64)", lambda: PlDataFrame[UnionSchema](union_ints), False),
        ("Union[int, str] (Object)", lambda: PlDataFrame[UnionSchema](mixed), False),
        ("Literal", lambda: PlDataFrame[LiteralSchema](literals), False),
        ("Range+Unique", lambda: PlDataFrame[NumericValidatorsSchema](ints), False),
        ("MinLen+MaxLen+Regex+In", lambda: PlDataFrame[StrValidatorsSchema](strs), False),
        ("error: int got str", lambda: PlDataFrame[IntSchema](bad_int), True),
        ("error: Union got float", lambda: PlDataFrame[UnionSchema](bad_union), True),
    ]


def _measure(fn: Callable[[], object], expect_error: bool, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        try:
            fn()
        except ValidationError:
            if not expect_error:
                raise
        else:
            if expect_error:
                raise AssertionError("expected ValidationError")
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--backend", choices=["pandas", "polars", "all"], default="all")
    args = parser.parse_args()

    backends: list[tuple[str, Callable[[int], list[Scenario]]]] = []
    if args.backend in ("pandas", "all"):
        backends.append(("pandas", _pandas_scenarios))
    if args.backend in ("polars", "all"):
        backends.append(("polars", _polars_scenarios))

    print(f"rows={args.rows:,} repeat={args.repeat} (best of, seconds)\n")
    print("| backend | scenario | time (s) |")
    print("|---|---|---:|")
    for backend, build in backends:
        for name, fn, expect_error in build(args.rows):
            elapsed = _measure(fn, expect_error, args.repeat)
            print(f"| {backend} | {name} | {elapsed:.4f} |", flush=True)


if __name__ == "__main__":
    main()
