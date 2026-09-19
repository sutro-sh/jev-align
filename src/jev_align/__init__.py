"""CLI for building AI Functions with pluggable evaluation backends."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jev-align")
except PackageNotFoundError:
    __version__ = "0+unknown"
