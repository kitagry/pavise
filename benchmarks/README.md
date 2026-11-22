# Benchmarks

`bench_overhead.py` measures how much time pavise validation adds on top of the checks
themselves. It is not collected by pytest and does not run in CI, since timings on shared
runners vary too much to catch regressions. Use it locally, for example to put a before/after
table in a pull request.

## Running

```sh
# every scenario on pandas, polars and polars.LazyFrame at 10,000 rows
uv run python benchmarks/bench_overhead.py

# several row counts
uv run python benchmarks/bench_overhead.py --rows 10 --rows 10000 --rows 1000000

# only the scenarios a change touches (case-insensitive substring; repeatable)
uv run python benchmarks/bench_overhead.py --backend polars --scenario Unique --rows 1000000

# list the scenarios and which backends run them
uv run python benchmarks/bench_overhead.py --list
```

| Option | Meaning |
|---|---|
| `--rows N` | Row count. Repeatable; default 10000. |
| `--backend {pandas,polars,lazy}` | Backends to run. Repeatable; default all. `lazy` is `polars.LazyFrame`. |
| `--scenario TEXT` | Run only scenarios whose name contains `TEXT`. Repeatable. |
| `--repeat N` | Samples per timing; the best one is reported. Default 3. |
| `--no-pandera` | Skip pandera even when it is installed. |
| `--list` | List the scenarios and exit. |

### Comparing with pandera

pandera is not a dependency of pavise. The pandera columns are filled only when it can be
imported, so install it for a single run with:

```sh
uv run --with 'pandera[polars]' python benchmarks/bench_overhead.py
```

pandera validates only the schema of a polars `LazyFrame` by default, so its
`polars lazy (collect)` numbers do not include value checks.

## Reading the results

Each scenario is timed three ways on the same frame, and the best per-call time is printed:

- **hand**: the checks pavise performs for the scenario (dtype comparison, null count,
  vectorized comparisons), written directly with the backend's API. This is what the checks
  themselves cost.
- **pavise**: `DataFrame[Schema](frame)`, with the class subscripted once outside the loop.
  For `lazy`, `polars lazy (schema)` is `LazyFrame[Schema](lf)`, which checks only the schema,
  and `polars lazy (collect)` adds `.collect()`, which validates the values too.
- **pandera**: the equivalent `DataFrameSchema.validate(frame)`.

`pavise − hand` is the time pavise adds. With few rows it is the fixed cost of a validation
(reading the schema, per-column calls); with many rows it is the work that grows with the
data. Running with `--rows 10` and a large `--rows` separates the two.

Scenarios named `error: ...` time how long pavise takes to reject an invalid column, so they
have no hand or pandera numbers.

The frames use what each backend gives by default: pandas strings get pandas' default string
dtype (`object` on pandas 2, `str` on pandas 3), dates are Python objects in `object` columns,
nullable integers are `Int64`, and list and struct columns are `pd.ArrowDtype`.

Measure on an otherwise idle machine and compare numbers only from the same machine.
