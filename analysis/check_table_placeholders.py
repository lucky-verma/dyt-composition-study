#!/usr/bin/env python3
"""Fail on unresolved table placeholders and stale manually typed table values.

This is intentionally narrow: it catches the class of errors where completed
run data exists in machine-readable sources, but the manuscript still shows a
manual placeholder such as "~---".
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


def uncommented_text(tex: str) -> str:
    lines: list[str] = []
    for line in tex.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        lines.append(line)
    return "\n".join(lines)


def first_numeric_token(value: str) -> str | None:
    match = re.search(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)", value)
    return match.group(0) if match else None


def numeric_variants(token: str) -> set[str]:
    variants = {token}
    try:
        val = float(token)
    except ValueError:
        return variants
    for ndigits in (4, 3, 2, 1):
        variants.add(f"{val:.{ndigits}f}".rstrip("0").rstrip("."))
    return {v for v in variants if v}


def check_manifest_values(paper_dir: Path, tex: str) -> list[str]:
    sources_path = paper_dir / "docs" / "paper_sources.json"
    if not sources_path.exists():
        return []
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for table_name, table_data in sources.get("tables", {}).items():
        if not isinstance(table_data, dict):
            continue
        for cell_name, cell_data in table_data.get("cells", {}).items():
            if not isinstance(cell_data, dict):
                continue
            value = cell_data.get("value")
            if not isinstance(value, str):
                continue
            lower = value.lower()
            if "%" in value or any(tok in lower for tok in ("pending", "superseded")):
                continue
            token = first_numeric_token(value)
            if not token:
                continue
            if not any(variant in tex for variant in numeric_variants(token)):
                problems.append(
                    f"paper_sources value not found in main.tex: {table_name}::{cell_name} = {value}"
                )
    return problems


def sig_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    mod = {"dyt": "DyT", "diffattn": "DiffAttn"}.get(str(row.get("mod")), str(row.get("mod")))
    return str(row.get("cell", "")), str(row.get("data", "")), mod


def check_sig_tests_table(paper_dir: Path, tex: str) -> list[str]:
    sig_path = paper_dir / "docs" / "sig_tests.json"
    if not sig_path.exists():
        return []
    sig = json.loads(sig_path.read_text(encoding="utf-8"))
    table_match = re.search(
        r"\\label\{tab:sig_tests\}(?P<body>.*?)\\end\{table\}",
        tex,
        flags=re.DOTALL,
    )
    if not table_match:
        return ["tab:sig_tests missing from main.tex"]
    table_body = table_match.group("body")
    rows: dict[tuple[str, str, str], str] = {}
    for line in table_body.splitlines():
        if not line.lstrip().startswith("S"):
            continue
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 5:
            continue
        cell_label = parts[0].split("(", 1)[0].strip().replace(" ", "_")
        size_match = re.search(r"\(([^)]+)\)", parts[0])
        if size_match:
            cell_label = f"{cell_label}_{size_match.group(1)}"
        key = (cell_label, parts[1], parts[2])
        rows[key] = line

    problems: list[str] = []
    for row in sig.get("cells", []):
        key = sig_row_key(row)
        line = rows.get(key)
        if line is None:
            problems.append(f"tab:sig_tests row missing: {key}")
            continue
        parts = [p.strip() for p in line.split("&")]
        displayed = {"van_mean": parts[3], "mod_mean": parts[4]}
        for field, shown in displayed.items():
            shown_match = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", shown)
            if not shown_match:
                problems.append(f"tab:sig_tests {key} {field} missing numeric value: {line.strip()}")
                continue
            shown_val = float(shown_match.group(0))
            expected_val = float(row[field])
            if abs(shown_val - expected_val) > 0.0015:
                problems.append(
                    f"tab:sig_tests {key} {field} expected approximately "
                    f"{expected_val:.3f}, got {shown_val:.3f}: {line.strip()}"
                )
    return problems


def main() -> int:
    paper_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if paper_dir is None or not (paper_dir / "main.tex").exists():
        print("usage: check-table-placeholders.py <paper-dir>", file=sys.stderr)
        return 2

    tex = uncommented_text((paper_dir / "main.tex").read_text(encoding="utf-8"))
    problems: list[str] = []
    unresolved = re.findall(r"~---|TODO_TABLE|PENDING_TABLE|TBD_TABLE", tex)
    if unresolved:
        problems.append(f"unresolved table placeholder token(s): {', '.join(sorted(set(unresolved)))}")

    problems.extend(check_sig_tests_table(paper_dir, tex))
    problems.extend(check_manifest_values(paper_dir, tex))

    if problems:
        print("table placeholder/value check failed:")
        for problem in problems[:50]:
            print(f"  - {problem}")
        if len(problems) > 50:
            print(f"  ... {len(problems) - 50} more")
        return 1

    print("table placeholder/value check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
