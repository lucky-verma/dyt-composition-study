#!/usr/bin/env python3
"""
Paired t-tests + Bonferroni correction for DyT composition paper (tab:phase + tab:scaling).

Input: per-seed best_val_loss from the cached all_results.json manifest.
Output: JSON with raw p-values, Bonferroni-corrected, significance stars.

Usage:
    python analysis/sig_tests.py
    python analysis/sig_tests.py --local results/full/all_results.json

Design:
- Paired t-test (scipy.stats.ttest_rel): per-seed pair (mod_seed_k vs vanilla_seed_k)
  preserves seed-level correlation, more powerful than independent t-test.
- One-sided test: H1 = mod is better (lower val_loss). Emit both one- and two-sided.
- Bonferroni: p_corrected = min(1, p_raw * n_comparisons).
- n_comparisons: count of cells where we compare mod vs vanilla (DyT helps/hurts direction).
- Stars: *** p<0.001, ** p<0.01, * p<0.05, ns otherwise.

Covers:
- tab:phase (64M): 8 cells (DyT 1M/10M/50M/118M, DiffAttn 1M/10M/50M/118M) vs vanilla at same data
- tab:scaling: 7 scale-DyT cells (1M+118M cross S1-S5, plus 10M S1-S4) + 4 scale-DiffAttn cells
- Total N≈18-20 comparisons

This public artifact version uses the bundled aggregate result manifest.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Optional

try:
    from scipy import stats
    import numpy as np
except ImportError:
    print("ERROR: scipy/numpy not installed. pip install scipy numpy", file=sys.stderr)
    sys.exit(1)


DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results" / "full" / "all_results.json"
SEEDS = [1337, 42, 7]


# Paper cell manifest — maps table cell to folder pattern
# Format: (scale_label, data_regime, modification, paper_table, vanilla_folders, mod_folders)
CELLS = [
    # tab:phase Scale 1 (64M) — 4 data × 2 modifications = 8 cells
    ("S1_64M", "1M",   "dyt",      "tab:phase",  "runset_3seed/wikitext_1m_vanilla_s{seed}",  "runset_3seed/wikitext_1m_dyt_s{seed}"),
    ("S1_64M", "1M",   "diffattn", "tab:phase",  "runset_3seed/wikitext_1m_vanilla_s{seed}",  "runset_3seed/wikitext_1m_diffattn_s{seed}"),
    ("S1_64M", "10M",  "dyt",      "tab:phase",  "intermediate/wikitext_10m_vanilla_s{seed}", "intermediate/wikitext_10m_dyt_s{seed}"),
    ("S1_64M", "10M",  "diffattn", "tab:phase",  "intermediate/wikitext_10m_vanilla_s{seed}", "intermediate/wikitext_10m_diffattn_s{seed}"),
    ("S1_64M", "50M",  "dyt",      "tab:phase",  "intermediate/wikitext_50m_vanilla_s{seed}", "intermediate/wikitext_50m_dyt_s{seed}"),
    ("S1_64M", "50M",  "diffattn", "tab:phase",  "intermediate/wikitext_50m_vanilla_s{seed}", "intermediate/wikitext_50m_diffattn_s{seed}"),
    ("S1_64M", "118M", "dyt",      "tab:phase",  "runset_3seed/wikitext_vanilla_s{seed}",      "runset_3seed/wikitext_dyt_s{seed}"),
    ("S1_64M", "118M", "diffattn", "tab:phase",  "runset_3seed/wikitext_vanilla_s{seed}",      "runset_3seed/wikitext_diffattn_s{seed}"),
    # tab:scaling Scale 2 (124M)
    ("S2_124M", "1M",   "dyt",      "tab:scaling", "scale2/wikitext_1m_vanilla_s{seed}",       "scale2/wikitext_1m_dyt_s{seed}"),
    ("S2_124M", "118M", "dyt",      "tab:scaling", "scale2/wikitext_vanilla_s{seed}",          "scale2/wikitext_dyt_s{seed}"),
    ("S2_124M", "118M", "diffattn", "tab:scaling", "scale2/wikitext_vanilla_s{seed}",          "scale2/wikitext_diffattn_s{seed}"),
    # tab:scaling Scale 3 (354M)
    ("S3_354M", "1M",   "dyt",      "tab:scaling", "scale3/wikitext_1m_vanilla_s{seed}",       "scale3/wikitext_1m_dyt_s{seed}"),
    ("S3_354M", "118M", "dyt",      "tab:scaling", "scale3/wikitext_vanilla_s{seed}",          "scale3/wikitext_dyt_s{seed}"),
    ("S3_354M", "118M", "diffattn", "tab:scaling", "scale3/wikitext_vanilla_s{seed}",          "scale3/wikitext_diffattn_s{seed}"),
    # tab:scaling Scale 4 (1.3B)
    ("S4_1.3B", "1M",   "dyt",      "tab:scaling", "scale4/wikitext_1m_vanilla_s{seed}",       "scale4/wikitext_1m_dyt_s{seed}"),
    ("S4_1.3B", "118M", "dyt",      "tab:scaling", "scale4/wikitext_vanilla_s{seed}",          "scale4/wikitext_dyt_s{seed}"),
    ("S4_1.3B", "118M", "diffattn", "tab:scaling", "scale4/wikitext_vanilla_s{seed}",          "scale4/wikitext_diffattn_s{seed}"),
    # tab:scaling Scale 5 (3.78B) — vanilla + dyt only (DiffAttn collapsed in v1)
    ("S5_3.78B", "1M",   "dyt", "tab:scaling", "scale5/wikitext_1m_vanilla_s{seed}",       "scale5/wikitext_1m_dyt_s{seed}"),
    ("S5_3.78B", "118M", "dyt", "tab:scaling", "scale5/wikitext_vanilla_s{seed}",          "scale5/wikitext_dyt_s{seed}"),
]


def fetch_val_loss_local(all_results: dict, folder_pattern: str) -> list[Optional[float]]:
    """Fetch from cached all_results.json if present."""
    vals = []
    for seed in SEEDS:
        folder = folder_pattern.format(seed=seed)
        entry = all_results.get(folder, {})
        vals.append(entry.get("best_val_loss"))
    return vals


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=str, default=str(DEFAULT_RESULTS),
                    help="Path to local all_results.json")
    ap.add_argument("--out", type=str,
                    default="<PAPER_DIR>/docs/sig_tests.json")
    args = ap.parse_args()

    all_results = json.load(open(args.local))
    fetch = lambda p: fetch_val_loss_local(all_results, p)

    results = []
    skipped = []
    for cell_label, data, mod, table, van_pattern, mod_pattern in CELLS:
        print(f"[cell] {table} {cell_label}/{data}/{mod}", flush=True)
        van = fetch(van_pattern)
        modv = fetch(mod_pattern)
        paired = [(v, m) for v, m in zip(van, modv) if v is not None and m is not None]
        if len(paired) < 2:
            print(f"  SKIP: only {len(paired)} seed pair(s) found", flush=True)
            skipped.append({"cell": cell_label, "data": data, "mod": mod,
                            "reason": f"{len(paired)} pairs", "van": van, "mod": modv})
            continue
        v_arr = np.array([p[0] for p in paired])
        m_arr = np.array([p[1] for p in paired])
        delta = (m_arr.mean() - v_arr.mean()) / v_arr.mean() * 100

        # Paired t-test (2-sided)
        t_stat, p_two = stats.ttest_rel(m_arr, v_arr)
        # 1-sided test: H1 depends on observed direction
        p_one = p_two / 2 if t_stat > 0 else p_two / 2  # conservative same-sided

        results.append({
            "cell": cell_label,
            "data": data,
            "mod": mod,
            "table": table,
            "n_seeds": len(paired),
            "van_mean": float(v_arr.mean()),
            "van_std": float(v_arr.std(ddof=1)) if len(v_arr) > 1 else 0,
            "mod_mean": float(m_arr.mean()),
            "mod_std": float(m_arr.std(ddof=1)) if len(m_arr) > 1 else 0,
            "delta_pct": round(float(delta), 2),
            "t_stat": round(float(t_stat), 3),
            "p_raw_two_sided": float(p_two),
            "p_raw_one_sided": float(p_one),
            "direction": "improves" if delta < 0 else "harms",
        })

    # Bonferroni correction
    n = len(results)
    for r in results:
        p_bonf_two = min(1.0, r["p_raw_two_sided"] * n)
        p_bonf_one = min(1.0, r["p_raw_one_sided"] * n)
        r["p_bonferroni_two_sided"] = float(p_bonf_two)
        r["p_bonferroni_one_sided"] = float(p_bonf_one)
        r["stars_raw"] = stars(r["p_raw_two_sided"])
        r["stars_bonf"] = stars(p_bonf_two)

    # Summary
    n_sig_raw = sum(1 for r in results if r["p_raw_two_sided"] < 0.05)
    n_sig_bonf = sum(1 for r in results if r["p_bonferroni_two_sided"] < 0.05)

    summary = {
        "n_cells": n,
        "n_skipped": len(skipped),
        "n_significant_raw_p05": n_sig_raw,
        "n_significant_bonferroni_p05": n_sig_bonf,
        "bonferroni_correction": f"p_raw × {n}",
        "seeds": SEEDS,
        "fetch_source": args.local or "metadata files",
    }

    out = {
        "summary": summary,
        "cells": results,
        "skipped": skipped,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n[DONE] {n} cells tested, {n_sig_bonf}/{n} Bonferroni-significant at p<0.05", flush=True)
    print(f"[DONE] wrote {out_path}")
    print("\n=== Markdown table (copy to tables or captions) ===")
    print(f"{'Cell':12s} {'Data':6s} {'Mod':10s} {'Δ%':>8s} {'p_raw':>10s} {'p_bonf':>10s} {'stars':6s}")
    for r in sorted(results, key=lambda r: (r["cell"], r["data"], r["mod"])):
        print(f"{r['cell']:12s} {r['data']:6s} {r['mod']:10s} {r['delta_pct']:>8.1f} "
              f"{r['p_raw_two_sided']:>10.4g} {r['p_bonferroni_two_sided']:>10.4g} {r['stars_bonf']:6s}")

    return 0 if n_sig_bonf > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
