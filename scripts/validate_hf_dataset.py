#!/usr/bin/env python3
"""Validate the public Hugging Face artifact dataset wiring."""
from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "lucky-verma/dyt-composition-artifacts"
GITHUB_URL = "https://github.com/lucky-verma/dyt-composition-study"
ARXIV_ID = "2604.23434"
DOI = "10.48550/arXiv.2604.23434"
TITLE = (
    "When Does Removing LayerNorm Help? "
    "Activation Bounding as a Regime-Dependent Implicit Regularizer"
)

REQUIRED_REMOTE_FILES = {
    "README.md",
    "CITATION.bib",
    "PROVENANCE.json",
    "SHA256SUMS.txt",
    "metadata/validation_report.md",
    "data/artifact_index.jsonl",
}
FORBIDDEN_REMOTE_FILES = {"VALIDATION_REPORT.md"}

FORBIDDEN_PATH_RE = re.compile(
    r"(DONE\.txt$|partial_killed|\.pt$|\.pth$|\.ckpt$|\.safetensors$|"
    r"\.bin$|\.pkl$|\.pickle$|\.log$|\.out$|\.err$|\.pdf$|\.png$|"
    r"\.jpe?g$|\.zip$|\.tar$|\.gz$|(^|/)\.cache(/|$)|wandb)",
    re.IGNORECASE,
)


def fetch_text(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} while fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not fetch {url}: {exc.reason}") from exc


def fetch_json(url: str) -> dict:
    return json.loads(fetch_text(url))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def local_contains(path: str, expected: str, failures: list[str]) -> None:
    full_path = REPO_ROOT / path
    require(full_path.exists(), f"missing local file: {path}", failures)
    if full_path.exists():
        text = full_path.read_text(encoding="utf-8")
        require(expected in text, f"{path} does not contain {expected}", failures)


def main() -> int:
    failures: list[str] = []
    api = fetch_json(f"https://huggingface.co/api/datasets/{DATASET_ID}")
    siblings = {item["rfilename"] for item in api.get("siblings", [])}
    tags = set(api.get("tags", []))

    require(api.get("id") == DATASET_ID, "HF dataset id mismatch", failures)
    require(api.get("private") is False, "HF dataset is not public", failures)
    require(f"arxiv:{ARXIV_ID}" in tags, f"missing HF arXiv tag arxiv:{ARXIV_ID}", failures)
    require("license:cc-by-4.0" in tags, "missing HF CC-BY-4.0 license tag", failures)

    missing = sorted(REQUIRED_REMOTE_FILES - siblings)
    require(not missing, f"missing HF metadata files: {missing}", failures)
    stale = sorted(FORBIDDEN_REMOTE_FILES & siblings)
    require(not stale, f"stale root-level HF metadata files remain: {stale}", failures)

    bad = sorted(path for path in siblings if FORBIDDEN_PATH_RE.search(path))
    require(not bad, f"forbidden HF paths remain: {bad[:12]}", failures)

    validation = fetch_text(
        f"https://huggingface.co/datasets/{DATASET_ID}/raw/main/metadata/validation_report.md"
    )
    require("- Status: PASS" in validation, "HF validation report is not PASS", failures)
    require(f"- arXiv: {ARXIV_ID}" in validation, "HF validation report arXiv mismatch", failures)
    require(
        "Forbidden raw/checkpoint/log/image/archive files: 0" in validation,
        "HF validation report still has forbidden files",
        failures,
    )

    provenance = fetch_json(
        f"https://huggingface.co/datasets/{DATASET_ID}/raw/main/PROVENANCE.json"
    )
    paper = provenance.get("paper", {})
    source = provenance.get("source_repo", {})
    require(paper.get("title") == TITLE, "HF provenance title mismatch", failures)
    require(paper.get("arxiv_id") == ARXIV_ID, "HF provenance arXiv mismatch", failures)
    require(paper.get("doi") == DOI, "HF provenance DOI mismatch", failures)
    require(source.get("remote") == f"{GITHUB_URL}.git", "HF provenance source repo mismatch", failures)

    local_contains("README.md", f"https://huggingface.co/datasets/{DATASET_ID}", failures)
    local_contains("README.md", f"https://arxiv.org/abs/{ARXIV_ID}", failures)
    local_contains("README.md", "metadata/validation_report.md", failures)
    local_contains("CITATION.cff", f"https://huggingface.co/datasets/{DATASET_ID}", failures)
    local_contains("CITATION.cff", DOI, failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(f"HF dataset OK: {DATASET_ID}")
    print(f"Remote commit: {api.get('sha')}")
    print(f"Files: {len(siblings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
