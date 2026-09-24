"""Run the internal-reference guard over every tracked text file."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = {".png", ".pdf", ".svg", ".gif", ".jpg", ".zip"}
EXCLUDED = {"CHANGELOG.md", "tools/check_internal_refs.py"}


def test_no_internal_research_references_in_tracked_files():
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    files = [
        str(ROOT / f)
        for f in tracked
        if Path(f).suffix not in BINARY and f not in EXCLUDED
    ]
    assert files
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_internal_refs.py"), *files],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout
