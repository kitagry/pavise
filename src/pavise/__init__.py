"""Pavise: DataFrame validation library using Python Protocol for structural subtyping."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pavise")
except PackageNotFoundError:
    # Imported straight from src/ without installing the package.
    __version__ = "0.0.0"

__all__ = []
