"""CLI for building AI Functions with pluggable evaluation backends."""

from importlib.metadata import PackageNotFoundError, version

from .capture import Capture
from .runtime import aligned

__all__ = ["Capture", "aligned", "__version__"]

try:
    __version__ = version("jev-align")
except PackageNotFoundError:
    __version__ = "0+unknown"
