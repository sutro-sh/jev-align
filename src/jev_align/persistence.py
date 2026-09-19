from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import LabelRecord, RunState


class RunStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.state_path = directory / "state.json"
        self.labels_path = directory / "labels.jsonl"
        self.gepa_output_dir = directory / "gepa-output"
        self.gepa_run_dir = directory / "gepa-runs"

    def initialize(self, state: RunState) -> None:
        self.directory.mkdir(parents=True, exist_ok=False)
        self.gepa_output_dir.mkdir()
        self.gepa_run_dir.mkdir()
        self.save_state(state)

    def save_state(self, state: RunState) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = state.model_dump_json(indent=2) + "\n"
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory, delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, self.state_path)

    def load_state(self) -> RunState:
        return RunState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def append_label(self, record: LabelRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.labels_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def pop_last_label(self, *, story_id: str, round_number: int) -> LabelRecord:
        records = self.load_labels()
        if not records:
            raise ValueError("there is no label to undo")
        removed = records[-1]
        if removed.story_id != story_id or removed.round_number != round_number:
            raise ValueError("the last saved label is not the requested batch item")
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory, delete=False
        ) as handle:
            for record in records[:-1]:
                handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, self.labels_path)
        return removed

    def replace_labels(self, records: list[LabelRecord]) -> None:
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.directory, delete=False
        ) as handle:
            for record in records:
                handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, self.labels_path)

    def archive_rewind(self, from_round: int) -> Path:
        """Preserve state and move invalidated artifacts out of the live run."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        archive = (
            self.directory / "rewinds" / f"{timestamp}-from-round-{from_round:04d}"
        )
        archive.mkdir(parents=True)
        for path in (self.state_path, self.labels_path):
            if path.exists():
                shutil.copy2(path, archive / path.name)
        for cache_name in (
            "pool-predictions-current.json",
            "pool-predictions-pending.json",
        ):
            path = self.directory / cache_name
            if path.exists():
                shutil.move(str(path), archive / cache_name)
        for report in self.directory.glob("round-*-report.json"):
            round_number = int(report.stem.split("-")[1])
            if round_number >= from_round:
                shutil.move(str(report), archive / report.name)
        for parent in (self.gepa_output_dir, self.gepa_run_dir):
            for round_directory in parent.glob("round-*"):
                round_number = int(round_directory.name.split("-")[1])
                if round_number >= from_round:
                    destination = archive / parent.name
                    destination.mkdir(exist_ok=True)
                    shutil.move(
                        str(round_directory), destination / round_directory.name
                    )
        return archive

    def load_labels(self) -> list[LabelRecord]:
        if not self.labels_path.exists():
            return []
        records: list[LabelRecord] = []
        with self.labels_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(LabelRecord.model_validate_json(line))
        return records

    def write_json(self, name: str, payload: object) -> Path:
        path = self.directory / name
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path
