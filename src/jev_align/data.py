from __future__ import annotations

import csv
import hashlib
import html
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from .models import Story


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            self.parts.append(value)

    def text(self) -> str:
        return "\n".join(self.parts)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    parser.close()
    return parser.text()


SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".parquet"}
IGNORED_DIRECTORIES = {
    ".git",
    ".jev-align",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}


def discover_datasets(root: Path, limit: int = 200) -> list[Path]:
    found: list[Path] = []
    for directory, names, files in os.walk(root):
        names[:] = [
            name
            for name in names
            if name not in IGNORED_DIRECTORIES and not name.startswith(".")
        ]
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(path)
    found.sort(
        key=lambda path: (
            len(path.relative_to(root).parts),
            -path.stat().st_mtime,
            str(path).lower(),
        )
    )
    return found[:limit]


def dataset_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            columns = csv.DictReader(handle).fieldnames
    elif suffix == ".jsonl":
        columns = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
                columns.extend(key for key in value if key not in columns)
                if index >= 99:
                    break
    elif suffix == ".parquet":
        import pyarrow.parquet as pq

        columns = pq.ParquetFile(path).schema_arrow.names
    else:
        raise ValueError("dataset must be CSV, Parquet, or JSONL")
    if not columns:
        raise ValueError("dataset has no columns")
    return list(columns)


def dataset_row_count(path: Path) -> int:
    """Return the number of data rows without materializing the dataset."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows
    raise ValueError("dataset must be CSV, Parquet, or JSONL")


def _rows(path: Path, columns: list[str]) -> Iterator[Mapping[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            yield from csv.DictReader(handle)
        return
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
                yield value
        return
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            yield from batch.to_pylist()
        return
    raise ValueError("dataset must be CSV, Parquet, or JSONL")


def _field_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stories(
    path: Path,
    limit: int = 1000,
    columns: Iterable[str] | None = None,
    *,
    concatenate: bool = False,
) -> list[Story]:
    available = dataset_columns(path)
    selected = list(columns) if columns is not None else available
    if not selected:
        raise ValueError("at least one column must be selected")
    missing = [column for column in selected if column not in available]
    if missing:
        raise ValueError(f"dataset does not contain columns: {', '.join(missing)}")
    stories: list[Story] = []
    for index, row in enumerate(_rows(path, selected), start=1):
        if len(stories) >= limit:
            break
        values = {column: _field_text(row.get(column)) for column in selected}
        fields = {"content": "\n".join(values.values())} if concatenate else values
        stories.append(Story(id=f"row-{index:06d}", row_number=index, fields=fields))
    if not stories:
        raise ValueError("dataset contains no data rows")
    return stories
