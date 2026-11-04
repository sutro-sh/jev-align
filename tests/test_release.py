"""Keep installation aliases tied to the canonical release without copying code."""

from pathlib import Path
import runpy
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = runpy.run_path(str(ROOT / "scripts" / "build_release.py"))


@pytest.mark.parametrize("name", BUILD["ALIASES"])
@pytest.mark.parametrize("version", ["0.1.7", "1.2.0rc1"])
def test_alias_release_metadata(name, version) -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    project["version"] = version

    alias = tomllib.loads(BUILD["alias_metadata"](name, project))

    assert alias["project"]["name"] == name
    assert alias["project"]["version"] == version
    assert alias["project"]["dependencies"] == [f"jev-align=={version}"]
    assert alias["project"]["scripts"] == {name: project["scripts"][name]}
    assert alias["project"]["requires-python"] == project["requires-python"]


def test_release_refuses_stale_distributions(tmp_path) -> None:
    stale = tmp_path / "old-0.1.0-py3-none-any.whl"
    stale.touch()
    with pytest.raises(SystemExit, match="unrelated artifacts"):
        BUILD["build"](tmp_path)
    assert stale.exists()
