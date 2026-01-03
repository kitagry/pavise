# Pavise

DataFrame validation library using Python Protocol for structural subtyping.

## Features

- Use Python Protocol to define DataFrame schemas
- `DataFrame[Schema]` type annotation for static type checking
- Structural subtyping: validate only required columns, ignore extra columns
- Covariant type parameters: `DataFrame[ChildSchema]` is compatible with `DataFrame[ParentSchema]`
- Optional runtime validation
- No inheritance required
- Support for both pandas and polars backends

## Installation

```bash
# For pandas support
pip install pavise[pandas]

# For polars support
pip install pavise[polars]

# For both
pip install pavise[all]
```

## Usage

### Pandas Backend

#### Static Type Checking Only (Recommended)

```python
from typing import Protocol
from pavise.pandas import DataFrame

class UserSchema(Protocol):
    name: str
    age: int

def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
    # mypy/pyrefly will check types, no runtime validation
    return df[df['age'] >= 18]

# Use regular pandas DataFrame
import pandas as pd
df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 17]})
result = process_users(df)
```

#### Runtime Validation (Explicit)

```python
from typing import Protocol
import pandas as pd
from pavise.pandas import DataFrame

class UserSchema(Protocol):
    name: str
    age: int

def load_users(raw_df: pd.DataFrame) -> DataFrame[UserSchema]:
    # Validate at runtime when needed
    return DataFrame[UserSchema](raw_df)

raw_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 17]})
validated_df = load_users(raw_df)  # Runtime validation occurs here
```

### Polars Backend

#### Static Type Checking Only (Recommended)

```python
from typing import Protocol
from pavise.polars import DataFrame

class UserSchema(Protocol):
    name: str
    age: int

def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
    # mypy/pyrefly will check types, no runtime validation
    return df.filter(df['age'] >= 18)

# Use regular polars DataFrame
import polars as pl
df = pl.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 17]})
result = process_users(df)
```

#### Runtime Validation (Explicit)

```python
from typing import Protocol
import polars as pl
from pavise.polars import DataFrame

class UserSchema(Protocol):
    name: str
    age: int

def load_users(raw_df: pl.DataFrame) -> DataFrame[UserSchema]:
    # Validate at runtime when needed
    return DataFrame[UserSchema](raw_df)

raw_df = pl.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 17]})
validated_df = load_users(raw_df)  # Runtime validation occurs here
```

### Structural Subtyping

```python
from typing import Protocol
import pandas as pd
from pavise.pandas import DataFrame

class UserSchema(Protocol):
    name: str

class UserWithEmailSchema(Protocol):
    name: str
    email: str

def process_user(df: DataFrame[UserSchema]) -> None:
    print(df['name'])

# This works! UserWithEmailSchema has all required columns of UserSchema
df: DataFrame[UserWithEmailSchema] = pd.DataFrame({
    'name': ['Alice'],
    'email': ['alice@example.com']
})
process_user(df)  # OK - covariant type parameter
```

### Extra Columns are Ignored

```python
from typing import Protocol
import pandas as pd
from pavise.pandas import DataFrame

class SimpleSchema(Protocol):
    a: int

# Extra columns are ignored during validation
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': ['x', 'y', 'z'],  # Extra column - ignored
    'c': [10.0, 20.0, 30.0]  # Extra column - ignored
})

validated = DataFrame[SimpleSchema](df)  # OK
```

## Supported Types

- `int`
- `float`
- `str`
- `bool`

## Development

```bash
# Install with dev dependencies (includes both pandas and polars)
uv pip install -e ".[dev]"

# Run all tests
uv run pytest

# Run tests for specific backend
uv run pytest tests/test_pandas.py
uv run pytest tests/test_polars.py
```

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

# Available Python versions: py39, py310, py311, py312
# Available backends: pandas, polars, all
```
