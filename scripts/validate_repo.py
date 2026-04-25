#!/usr/bin/env python3
"""Lightweight artifact validation for the DyT composition repository."""

from __future__ import annotations

import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTERNAL_PATTERNS = [
    r"/mnt/",
    r"/home/",
    r"wandb\.ai/",
]
INTERNAL_PATTERNS.extend(
    p for p in os.environ.get("DYT_EXTRA_INTERNAL_PATTERNS", "").splitlines() if p.strip()
)


def iter_files(suffix: str):
    ignored = {".git", ".venv", "__pycache__"}
    for path in ROOT.rglob(f"*{suffix}"):
        if any(part in ignored for part in path.parts):
            continue
        yield path


def check_python() -> int:
    count = 0
    for path in iter_files(".py"):
        py_compile.compile(str(path), doraise=True)
        count += 1
    print(f"python_ok {count}")
    return count


def check_json() -> int:
    count = 0
    for path in iter_files(".json"):
        with path.open() as f:
            json.load(f)
        count += 1
    print(f"json_ok {count}")
    return count


def check_shell() -> int:
    count = 0
    for path in iter_files(".sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
        count += 1
    print(f"shell_ok {count}")
    return count


def check_internal_fingerprints() -> None:
    offenders: list[tuple[str, int, str]] = []
    compiled = [(p, re.compile(p, re.IGNORECASE)) for p in INTERNAL_PATTERNS]
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, regex in compiled:
                if regex.search(line):
                    offenders.append((str(path.relative_to(ROOT)), lineno, pattern))
    if offenders:
        for rel, lineno, pattern in offenders[:50]:
            print(f"internal_fingerprint {rel}:{lineno} pattern={pattern}", file=sys.stderr)
        raise SystemExit(1)
    print("fingerprints_ok")


def main() -> None:
    check_python()
    check_json()
    check_shell()
    check_internal_fingerprints()


if __name__ == "__main__":
    main()
