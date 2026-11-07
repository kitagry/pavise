<div align="center">

<img alt="pavise logo: a striped shield guarding a panda and a polar bear" src="https://raw.githubusercontent.com/kitagry/pavise/main/assets/logo.svg" width="128" height="128">

<h1>pavise</h1>

<p><strong>Type-safe DataFrame validation with Python Protocols.</strong><br>
Describe your columns once with <code>typing.Protocol</code>, and get static type checking and runtime validation for pandas and polars.</p>

<p>
  <a href="https://pypi.org/project/pavise/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/pavise?logo=pypi&logoColor=white"></a>
  <a href="https://pypi.org/project/pavise/"><img alt="Python versions" src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fkitagry%2Fpavise%2Fmain%2Fpyproject.toml&logo=python&logoColor=white"></a>
  <a href="https://github.com/kitagry/pavise/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/kitagry/pavise/actions/workflows/test.yml/badge.svg?branch=main"></a>
  <a href="https://pavise.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://img.shields.io/readthedocs/pavise/latest"></a>
  <a href="https://github.com/kitagry/pavise/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/kitagry/pavise"></a>
</p>

<p>
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-installation">Installation</a> ·
  <a href="#-usage">Usage</a> ·
  <a href="#-supported-types">Supported Types</a> ·
  <a href="https://pavise.readthedocs.io/">Documentation</a>
</p>

</div>

## 🚀 Quick Start

```bash
pip install "pavise[pandas]"   # or "pavise[polars]"
```

Define a schema as a `Protocol`, validate at the boundary, and pass `DataFrame[Schema]` around with full type checking:

```python
from typing import Annotated, Literal, Protocol

import pandas as pd

from pavise.pandas import DataFrame
from pavise.validators import Range, Unique


class UserSchema(Protocol):
    user_id: Annotated[int, Unique()]
    name: str
    age: Annotated[int, Range(0, 150)]
    plan: Literal["free", "pro"]


def count_pro_users(df: DataFrame[UserSchema]) -> int:
    return int((df["plan"] == "pro").sum())


raw_df = pd.DataFrame(
    {
        "user_id": [1, 2],
        "name": ["Alice", "Bob"],
        "age": [30, 25],
        "plan": ["free", "pro"],
    }
)
users = DataFrame[UserSchema](raw_df)  # validated at runtime
count_pro_users(users)  # type-checked by mypy / pyright
```

Bad data fails fast, with the rows that broke the rule:

<!-- example-raises: ValidationError -->
```python
DataFrame[UserSchema](
    pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "age": [30, -5, 200],
            "plan": ["free", "pro", "pro"],
        }
    )
)
```

<!-- example-output -->
```text
ValidationError: Column 'age': values must be in range [0, 150]

Sample invalid values (showing first 2 of 2):
  Row 1: -5
  Row 2: 200
```

## ✨ Features

| | Feature | What you get |
| :---: | --- | --- |
| 🧩 | **Protocol schemas** | Schemas are plain `typing.Protocol` classes. No base class to inherit, no custom DSL |
| 🔍 | **Static type checking** | `DataFrame[Schema]` works with mypy and pyright, so a wrong DataFrame is caught before you run anything |
| 🦆 | **Structural subtyping** | Only the declared columns are checked and extra columns are ignored. `DataFrame[Child]` is accepted wherever `DataFrame[Parent]` is expected |
| ✅ | **Runtime validation** | Check column types and values once at system boundaries, such as CSV files, databases and APIs |
| 📏 | **Validators** | `Range`, `Ge` / `Gt` / `Le` / `Lt`, `Unique`, `In`, `Regex`, `MinLen` / `MaxLen` and `Custom`, attached with `typing.Annotated` |
| 🔗 | **DataFrame-level checks** | `@dataframe_check` for rules across columns, such as `start_at < end_at` |
| 🔒 | **Strict mode** | `strict=True` rejects columns that are not in the schema |
| 🐼 | **pandas & polars** | The same schema works on both backends, including polars `LazyFrame` |
| 💬 | **Readable errors** | Errors name the column and show sample rows with the invalid values |

### Why "pavise"?

A **pavise** was a large shield used by medieval crossbowmen, big enough to cover the whole body.
Like its namesake, this library is a shield for your data: whether your datasets are small or big, pavise protects your code with type safety and validation.
The logo shows that shield, striped and split in two, with a panda on one half and a polar bear on the other: it guards your pandas and polars DataFrames.

## 📦 Installation

```bash
# For pandas support
pip install "pavise[pandas]"

# For polars support
pip install "pavise[polars]"

# For both
pip install "pavise[all]"
```

pavise supports Python 3.10 and later.

## 📖 Usage

The full guide is at **[pavise.readthedocs.io](https://pavise.readthedocs.io/en/latest/index.html)**.

### pandas and polars

Import `DataFrame` from `pavise.pandas` or `pavise.polars`. The schema is the same for both:

```python
from typing import Protocol

import pandas as pd
import polars as pl

from pavise.pandas import DataFrame as PandasDataFrame
from pavise.polars import DataFrame as PolarsDataFrame


class UserSchema(Protocol):
    name: str
    age: int


pandas_users = PandasDataFrame[UserSchema](
    pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 17]})
)
polars_users = PolarsDataFrame[UserSchema](
    pl.DataFrame({"name": ["Alice", "Bob"], "age": [30, 17]})
)


def adults(df: PolarsDataFrame[UserSchema]) -> PolarsDataFrame[UserSchema]:
    return PolarsDataFrame[UserSchema](df.filter(pl.col("age") >= 18))


adults(polars_users)
```

For polars, `LazyFrame[Schema]` checks column names and types when it is created and checks values on `collect()`.
See the [pandas](https://pavise.readthedocs.io/en/latest/user-guide/pandas-backend.html) and [polars](https://pavise.readthedocs.io/en/latest/user-guide/polars-backend.html) backend guides.

### Structural subtyping

A schema only lists the columns a function needs. A DataFrame with more columns can be passed in:

```python
from typing import Protocol

import pandas as pd

from pavise.pandas import DataFrame


class UserSchema(Protocol):
    name: str


class UserWithEmailSchema(Protocol):
    name: str
    email: str


def greet(df: DataFrame[UserSchema]) -> None:
    for name in df["name"]:
        print(f"Hello, {name}!")


df = DataFrame[UserWithEmailSchema](
    pd.DataFrame({"name": ["Alice"], "email": ["alice@example.com"]})
)
greet(df)  # OK: UserWithEmailSchema has every column of UserSchema
```

Pass `strict=True` to reject extra columns instead:

<!-- example-raises: ValidationError -->
```python
DataFrame[UserSchema](df, strict=True)
# ValidationError: Strict mode: unexpected columns ['email']
```

### Validators

Attach validators to a column with `typing.Annotated`:

<!-- example-raises: ValidationError -->
```python
from typing import Annotated, Protocol

import pandas as pd

from pavise.pandas import DataFrame
from pavise.validators import Ge, Gt, Range, Regex


class OrderSchema(Protocol):
    customer_email: Annotated[str, Regex(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")]
    price: Annotated[float, Ge(0)]
    quantity: Annotated[int, Gt(0)]
    discount: Annotated[float, Range(0, 1)]


DataFrame[OrderSchema](
    pd.DataFrame(
        {
            "customer_email": ["alice@example.com"],
            "price": [9.5],
            "quantity": [0],  # must be > 0
            "discount": [0.1],
        }
    )
)
# ValidationError: Column 'quantity': values must be > 0
```

| Validator | Description |
|-----------|-------------|
| `Range(min, max)` | Values within `[min, max]` |
| `Ge(value)` / `Gt(value)` | Values `>= value` / `> value` |
| `Le(value)` / `Lt(value)` | Values `<= value` / `< value` |
| `Unique()` | No duplicate values |
| `In(values)` | Values within an allowed set |
| `Regex(pattern)` | Strings fully matching a regular expression (`re.fullmatch`) |
| `MinLen(n)` / `MaxLen(n)` | String length bounds |
| `Custom(func, message)` | Custom validation function |

`Range`, `Ge`, `Gt`, `Le` and `Lt` accept numbers as well as `date`, `datetime`, `time`, `timedelta` and `Decimal` bounds.
See the [validators guide](https://pavise.readthedocs.io/en/latest/user-guide/validators.html) for details.

### DataFrame-level checks

Use the `@dataframe_check` class decorator for rules that span several columns:

<!-- example-raises: ValidationError -->
```python
from datetime import datetime
from typing import Protocol

import pandas as pd

from pavise.checks import dataframe_check
from pavise.pandas import DataFrame


@dataframe_check(lambda df: df["start_at"] < df["end_at"], name="start_before_end")
class EventSchema(Protocol):
    start_at: datetime
    end_at: datetime


df = pd.DataFrame(
    {
        "start_at": pd.to_datetime(["2024-01-01", "2024-01-05"]),
        "end_at": pd.to_datetime(["2024-01-02", "2024-01-03"]),
    }
)
DataFrame[EventSchema](df)
# ValidationError: DataFrame check 'start_before_end' failed
#
# Sample invalid values (showing first 1 of 1):
#   Row 1: {'start_at': Timestamp('2024-01-05 00:00:00'), 'end_at': Timestamp('2024-01-03 00:00:00')}
```

The check function receives the backend's native DataFrame and returns a boolean per row (`pd.Series`, `pl.Series` or `pl.Expr`) or a single `bool`.
Checks run after column validation, and `polars.LazyFrame` runs them on `collect()`.
The decorator does not add members to the Protocol, so structural subtyping keeps working.

### Optional values, unions and literals

```python
from typing import Literal, Optional, Protocol, Union

import pandas as pd

from pavise.pandas import DataFrame, NotRequiredColumn


class TicketSchema(Protocol):
    code: Union[int, str]  # int or str
    status: Literal["open", "closed"]  # only these values
    assignee: Optional[str]  # may contain None
    note: NotRequiredColumn[str]  # the column may be absent


df = pd.DataFrame(
    {
        "code": [1, "B-2"],
        "status": ["open", "closed"],
        "assignee": ["alice", None],
    }
)
DataFrame[TicketSchema](df)  # OK: 'note' is not required
```

## 🧾 Supported Types

| Kind | Types |
|------|-------|
| Basic | `int`, `float`, `str`, `bool` |
| Date / time | `datetime`, `date`, `timedelta` |
| Generic | `Optional[T]`, `Union[T1, T2, ...]` (`Union[int, str, None]` for nullable unions), `Literal[...]`, `NotRequiredColumn[T]` |
| pandas | `pd.CategoricalDtype`, `pd.Int64Dtype` and other extension dtypes |
| polars | `pl.Categorical`, `pl.Int64` and other polars data types |

## 🛠️ Development

```bash
# Install with dev dependencies (includes both pandas and polars)
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run tests for specific backend
uv run pytest tests/test_pandas/
uv run pytest tests/test_polars/
```

### Documentation examples

`tests/test_docs_examples.py` runs every Python code block in `README.md` and `docs/**/*.rst`.
Blocks on the same page share one namespace, top to bottom.
Put one of these comments right before a block to change how it is run or checked (they are not rendered):

| reStructuredText | Markdown | Effect |
|------------------|----------|--------|
| `.. example-setup` + indented code | `<!-- example-setup` ... `-->` | Hidden code run at that point (e.g. to define `df`) |
| `.. example-raises: ValidationError` | `<!-- example-raises: ValidationError -->` | The next block must raise that exception |
| `.. example-skip` | `<!-- example-skip -->` | Do not run the next block (e.g. it needs a database or network) |
| `.. example-output` + `text` block | `<!-- example-output -->` + `text` block | The text block must equal what the previous block prints, or `ExceptionName: message` for an `example-raises` block |

### Testing with tox

```bash
# Run tests for all Python versions and backends
tox

# Run tests for specific environment
tox -e py312-pandas    # Test pandas backend with Python 3.12
tox -e py312-polars    # Test polars backend with Python 3.12
tox -e py312-all       # Test both backends with Python 3.12

# Run linting
tox -e lint

# Run type checking
tox -e type

# Available Python versions: py310, py311, py312, py313, py314
# Available backends: pandas, polars, all
```
