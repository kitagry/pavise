"""Backend-independent description of values rejected by a validator."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Violation:
    """Values a validator rejected, before they are turned into a ValidationError.

    Keeping the mask separate from raising lets List and Struct columns run validators on
    flattened values and report the rows the values come from.
    """

    mask: Any  # Boolean Series of the backend, True where a value is rejected
    message: str
    format_value: Callable[[Any], str] = repr
