"""The git commit a script ran from, for run records."""

from __future__ import annotations

import subprocess
from pathlib import Path


def code_version(script: Path) -> str:
    """The commit `script` ran from, marked dirty for a changed tree and untracked for a script outside git."""
    here = script.resolve().parent
    found = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True,
                           text=True, check=False, cwd=here)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", script.name],
                             capture_output=True, check=False, cwd=here).returncode == 0
    version = found.stdout.strip()
    return "unknown" if not version else version if tracked else f"{version}-untracked"
