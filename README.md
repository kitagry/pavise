# Pavise

[![Documentation Status](https://readthedocs.org/projects/pavise/badge/?version=latest)](https://pavise.readthedocs.io/en/latest/?badge=latest)

DataFrame validation library using Python Protocol for structural subtyping.

## About the Name

A **pavise** was a large shield used by medieval crossbowmen, big enough to cover the entire body and provide strong protection.

Like its namesake, this library serves as a shield for your data. Whether you're working with small datasets or big data, pavise protects your code with type safety and validation.

## Features

- Use Python Protocol to define DataFrame schemas
- `DataFrame[Schema]` type annotation for static type checking
- Structural subtyping: validate only required columns, ignore extra columns
- Covariant type parameters: `DataFrame[ChildSchema]` is compatible with `DataFrame[ParentSchema]`
- Optional runtime validation
- No inheritance required
- Support for both pandas and polars backends

## Documentation

Full documentation is available at [https://pavise.readthedocs.io/](https://pavise.readthedocs.io/en/latest/index.html)

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

```python
from typing import Protocol
import pandas as pd
from pavise.pandas import DataFrame

class UserSchema(Protocol):
    name: str
    age: int

# Runtime validation when creating DataFrame[Schema]
raw_df = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 17]})
validated_df = DataFrame[UserSchema](raw_df)  # Validates column types at runtime

# Type hints work with static type checkers (mypy, pyright, etc.)
def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
    return df[df['age'] >= 18]

result = process_users(validated_df)
```

### Polars Backend

```python
from typing import Protocol
import polars as pl
from pavise.polars import DataFrame

class UserSchema(Protocol):
    name: str
    age: int

# Runtime validation when creating DataFrame[Schema]
raw_df = pl.DataFrame({'name': ['Alice', 'Bob'], 'age': [30, 17]})
validated_df = DataFrame[UserSchema](raw_df)  # Validates column types at runtime

# Type hints work with static type checkers (mypy, pyright, etc.)
def process_users(df: DataFrame[UserSchema]) -> DataFrame[UserSchema]:
    return df.filter(df['age'] >= 18)

result = process_users(validated_df)
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
df = DataFrame[UserWithEmailSchema](pd.DataFrame({
    'name': ['Alice'],
    'email': ['alice@example.com']
}))
process_user(df)  # OK - covariant type parameter
```

### Using Validators

Add validators using `typing.Annotated` to enforce data quality constraints:

```python
from typing import Annotated, Protocol
import pandas as pd
from pavise.pandas import DataFrame
from pavise.validators import Range, Regex

class UserSchema(Protocol):
    name: str
    age: Annotated[int, Range(0, 150)]
    email: Annotated[str, Regex(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]

# Valid data passes validation
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30],
    'email': ['alice@example.com', 'bob@example.com']
})
validated_df = DataFrame[UserSchema](df)  # OK

# Invalid data raises ValidationError
invalid_df = pd.DataFrame({
    'name': ['Charlie'],
    'age': [200],  # Exceeds maximum age
    'email': ['invalid-email']  # Invalid email format
})
DataFrame[UserSchema](invalid_df)  # ValidationError
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

### Basic Types
- `int` - Integer values
- `float` - Floating point values
- `str` - String values
- `bool` - Boolean values

### Date/Time Types
- `datetime` - Date and time values
- `date` - Date-only values
- `timedelta` - Time duration values

### Generic Types
- `Optional[T]` - Nullable types (e.g., `Optional[int]`, `Optional[str]`)
- `Literal[...]` - Specific literal values (e.g., `Literal["a", "b", "c"]`, `Literal[1, 2, 3]`)
- `NotRequiredColumn[T]` - Optional columns (e.g., `NotRequiredColumn[int]`, `NotRequiredColumn[Optional[str]]`)

### Backend-Specific Types
- **pandas**: `pd.CategoricalDtype`, `pd.Int64Dtype`, and other Extension dtypes
- **polars**: `pl.Categorical`, `pl.Int64`, and other polars DataTypes

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
