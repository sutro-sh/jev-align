"""Build the implementation and installable CLI aliases at the same version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
ALIASES = ("decision-align", "decal")


def alias_metadata(name: str, project: dict) -> str:
    """Generate a standalone project using the root package's version."""
    quote = json.dumps
    return f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = {quote(name)}
version = {quote(project["version"])}
description = {quote(f'Installation alias for jev-align: {project["description"]}')}
readme = "README.md"
requires-python = {quote(project["requires-python"])}
license = {quote(project["license"])}
dependencies = [{quote('jev-align==' + project['version'])}]

[project.scripts]
{name} = "jev_align.cli:app"

[project.urls]
Repository = {quote(project["urls"]["Repository"])}

[tool.hatch.build.targets.wheel]
bypass-selection = true

[tool.hatch.build.targets.sdist]
include = ["/pyproject.toml", "/README.md"]
"""


def build(out_dir: Path) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        f"{name.replace('-', '_')}-{project['version']}{suffix}"
        for name in (project["name"], *ALIASES)
        for suffix in (".tar.gz", "-py3-none-any.whl")
    }
    artifacts = {*out_dir.glob("*.whl"), *out_dir.glob("*.tar.gz")}
    if any(path.name not in expected for path in artifacts):
        raise SystemExit(f"Distribution directory contains unrelated artifacts: {out_dir}")
    subprocess.run(
        ["uv", "build", "--out-dir", str(out_dir)], cwd=ROOT, check=True
    )
    for name in ALIASES:
        with tempfile.TemporaryDirectory(prefix=f"{name}-build-") as temporary:
            directory = Path(temporary)
            (directory / "pyproject.toml").write_text(alias_metadata(name, project))
            (directory / "README.md").write_text(
                f"# {name}\n\n"
                "An installation alias for [jev-align]"
                "(https://pypi.org/project/jev-align/), the CLI for building "
                "AI Functions with decision models and GEPA.\n\n"
                f"Install with `uv tool install {name}` or `pip install {name}`, "
                f"then run `{name}`. This package provides the `{name}` command "
                "and depends on the matching jev-align release; "
                "the implementation and Python imports remain in `jev_align`.\n"
            )
            subprocess.run(
                ["uv", "build", "--out-dir", str(out_dir)],
                cwd=directory,
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    build(parser.parse_args().out_dir)
