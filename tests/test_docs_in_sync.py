"""Guards the docs the repo's own rules say must track the code.

CLAUDE.md: "MUST update AGENTS.md, README.md, and docs/DESIGN.md when adding,
removing, or changing MCP tools." That rule was enforced by discipline alone,
and it drifted — the README documented 8 of the 10 tools the server actually
advertises, missing `search_models` and `list_model_sources`. A test costs
less than the discipline does.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "src" / "model_atlas" / "server.py"


def _declared_tools() -> list[str]:
    return re.findall(r"@mcp\.tool\(\)\s*\ndef (\w+)", SERVER.read_text())


def test_every_mcp_tool_is_documented_in_the_readme():
    readme = (ROOT / "README.md").read_text()
    missing = [t for t in _declared_tools() if f"`{t}`" not in readme]
    assert not missing, f"MCP tools missing from README.md tool table: {missing}"


def test_readme_documents_no_tool_that_does_not_exist():
    """The other direction: a tool row left behind after a rename sends the
    caller after something the server will reject."""
    readme = (ROOT / "README.md").read_text()
    declared = set(_declared_tools())
    table_rows = re.findall(r"^\| `(\w+)` \|", readme, re.MULTILINE)
    stale = [t for t in table_rows if t not in declared]
    assert not stale, f"README documents tools that no longer exist: {stale}"


def test_corpus_download_url_points_at_a_pinned_release():
    """`releases/latest/download/network.db` 404s whenever the newest release
    ships without the asset — which is exactly how the install broke after
    v0.4.1. curl then writes the error page over network.db."""
    readme = (ROOT / "README.md").read_text()
    assert "releases/latest/download/network.db" not in readme, (
        "README points at releases/latest, which silently 404s when the newest "
        "release has no network.db asset. Pin the version."
    )
    assert re.search(r"releases/download/v[\d.]+/network\.db", readme), (
        "README should download network.db from a pinned, versioned release URL"
    )


def test_package_version_strings_agree():
    """`pyproject.toml` said 0.2.0 and `__init__.py` said 0.1.0 while the
    project shipped v0.4.1 — so `pip show` and `model_atlas.__version__` both
    reported a version that had not existed for three releases."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    init = (ROOT / "src" / "model_atlas" / "__init__.py").read_text()
    proj_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    pkg_version = re.search(r'^__version__ = "([^"]+)"', init, re.MULTILINE).group(1)
    assert proj_version == pkg_version, (
        f"pyproject.toml says {proj_version}, __init__.py says {pkg_version}"
    )


def test_pinned_download_url_matches_the_package_version():
    """The README's pinned corpus URL must name the release being cut, or the
    quick start points users at an older asset than the code they cloned."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
    readme = (ROOT / "README.md").read_text()
    assert f"releases/download/v{version}/network.db" in readme, (
        f"README should pin the corpus download to v{version}"
    )
