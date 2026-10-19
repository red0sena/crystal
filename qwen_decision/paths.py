"""Project-root and source-file locations shared by the packaged modules.

Run manifests and audits key code hashes by the bare file names the modules had
when they lived in the project root (e.g. "scorer.py"). code_path keeps those
keys valid by resolving a bare name to the module's current location.
Standard library only, so restore/verify can still run with a plain python3.
"""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (ROOT / "qwen_decision", ROOT / "tests")


def code_path(name: str) -> Path:
    """Return the current path of a formerly root-level module given its bare file name."""
    matches = [path for directory in SOURCE_DIRS for path in directory.rglob(name)]
    if len(matches) != 1:
        raise FileNotFoundError(f"{name}: expected exactly one source file, found {matches}")
    return matches[0]
