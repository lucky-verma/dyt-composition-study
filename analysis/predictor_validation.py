#!/usr/bin/env python3
"""predictor_validation — leave-one-scale-out CV + AUC for the saturation heuristic.

Per Codex review (NeurIPS rigor): replace in-sample 75% claim with proper
held-out validation. Outputs:
  * Per-cell predictions (n=12 pre-Scale-5 calibration cells, plus 2 Scale-5 stress cells)
  * Leave-one-scale-out (LOSO) CV pooled accuracy + balanced accuracy + AUC
  * S1+S2 train / S3+S4+S5 test (extrapolation stress test, secondary metric)
  * Llama 3 cells held-out OOD transfer (separate, "directional transfer")
  * Threshold from train folds only (no test-set leakage)
  * ROC-AUC over saturation score (sat ∈ [0,1])

Output: docs/predictor_validation.json + paper-ready table snippet.

Usage:
    python3 tools/paper-hygiene/predictor_validation.py <PAPER_DIR>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Cell definition: (scale, data, sat_3seed_mean, delta_pct)
# Saturation from data/saturation_results.json (final-ckpt mean across 3 seeds).
# Delta from docs/paper_sources.json tab:phase + tab:scaling (M24 corrected).
# ─────────────────────────────────────────────────────────────────────────

GPT2_CELLS = [
    # (scale_id, scale_M, data_M, sat_mean, delta_pct, condition_label)
    ("S1", 64, 1, 0.4931, -27.3, "S1/1M"),
    ("S1", 64, 10, 0.4135, +5.9, "S1/10M"),
    ("S1", 64, 50, 0.2368, +19.7, "S1/50M"),
    ("S1", 64, 118, 0.2343, +18.8, "S1/118M"),
    ("S2", 124, 1, 0.4662, -9.6, "S2/1M"),
    ("S2", 124, 10, 0.2925, -12.3, "S2/10M"),  # M24 corrected
    ("S2", 124, 118, 0.1931, +12.8, "S2/118M"),
    ("S3", 354, 1, 0.4902, +4.3, "S3/1M"),
    ("S3", 354, 10, 0.3688, -24.1, "S3/10M"),
    ("S3", 354, 118, 0.3271, +13.4, "S3/118M"),
    ("S4", 1300, 1, 0.3934, +2.1, "S4/1M"),
    ("S4", 1300, 118, 0.2376, +10.4, "S4/118M"),
    ("S5", 3780, 1, 0.501, +1.7, "S5/1M"),    # from Table 10
    ("S5", 3780, 118, 0.803, +27.9, "S5/118M"),  # anomaly (saturation INVERSION)
]

# The paper reports 75% / AUC 0.75 on the pre-Scale-5 calibration set:
# S1 has 1M/10M/50M/118M; S2-S3 have 1M/10M/118M; S4 has 1M/118M.
# Scale 5 is a stress test and should be reported separately.
CALIBRATION_CELLS = [c for c in GPT2_CELLS if c[0] != "S5"]

LLAMA_CELLS = [
    # held-out OOD test
    ("L_S1", 64, 1, 0.536, -25.6, "Llama_S1/1M"),
    ("L_S1", 64, 118, 0.326, +59.1, "Llama_S1/118M"),
    ("L_S2", 124, 1, 0.452, -7.1, "Llama_S2/1M"),
]


# ─────────────────────────────────────────────────────────────────────────
# Predictor: rule = "DyT helps (label=1) iff sat > threshold"
# helps = delta_pct < 0 (negative delta = improvement)
# ─────────────────────────────────────────────────────────────────────────

def label(delta_pct: float) -> int:
    """1 if DyT helps (delta < 0), 0 otherwise."""
    return 1 if delta_pct < 0 else 0


def threshold_search(train_cells: list[tuple]) -> tuple[float, float]:
    """Find threshold maximizing accuracy on training cells. Returns (best_thr, best_acc)."""
    thresholds = sorted(set(c[3] for c in train_cells))
    # Test midpoints between unique sat values + boundaries
    test_thrs = [thresholds[0] - 0.01]
    for i in range(len(thresholds) - 1):
        test_thrs.append((thresholds[i] + thresholds[i + 1]) / 2)
    test_thrs.append(thresholds[-1] + 0.01)

    best_thr = 0.43
    best_acc = 0.0
    for thr in test_thrs:
        correct = sum(
            1 for (_, _, _, sat, dl, _) in train_cells
            if (sat > thr) == (label(dl) == 1)
        )
        acc = correct / len(train_cells)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def predict(sat: float, threshold: float) -> int:
    """1 = DyT helps; 0 = DyT hurts."""
    return 1 if sat > threshold else 0


def evaluate(cells: list[tuple], threshold: float) -> dict:
    """Return acc, balanced_acc, confusion matrix."""
    n = len(cells)
    tp = fn = fp = tn = 0
    correct = 0
    details = []
    for (sc, sm, dm, sat, dl, name) in cells:
        true = label(dl)
        pred = predict(sat, threshold)
        ok = (true == pred)
        correct += ok
        if true == 1 and pred == 1: tp += 1
        elif true == 1 and pred == 0: fn += 1
        elif true == 0 and pred == 1: fp += 1
        else: tn += 1
        details.append({
            "cell": name, "sat": round(sat, 3), "delta_pct": dl,
            "true_helps": bool(true), "pred_helps": bool(pred), "correct": ok
        })
    acc = correct / n if n else 0
    sens = tp / (tp + fn) if (tp + fn) else 0  # recall on helps
    spec = tn / (tn + fp) if (tn + fp) else 0  # recall on hurts
    bal_acc = (sens + spec) / 2
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "n": n,
        "details": details,
    }


def auc_score(cells: list[tuple]) -> tuple[float, list[dict]]:
    """Compute AUC over saturation threshold sweep. Higher sat → predict helps."""
    sorted_cells = sorted(cells, key=lambda c: c[3])  # by saturation asc
    pos = sum(1 for c in cells if label(c[4]) == 1)
    neg = len(cells) - pos
    if pos == 0 or neg == 0:
        return float("nan"), []

    # Mann-Whitney U-based AUC
    rank_sum_pos = 0
    sorted_by_sat = sorted(cells, key=lambda c: c[3])
    for i, c in enumerate(sorted_by_sat):
        if label(c[4]) == 1:
            rank_sum_pos += i + 1
    auc = (rank_sum_pos - pos * (pos + 1) / 2) / (pos * neg)

    # ROC sweep
    all_thrs = sorted(set(c[3] for c in cells))
    roc = []
    for thr in [0] + all_thrs + [1]:
        tp = sum(1 for c in cells if label(c[4]) == 1 and c[3] > thr)
        fn = pos - tp
        fp = sum(1 for c in cells if label(c[4]) == 0 and c[3] > thr)
        tn = neg - fp
        tpr = tp / pos if pos else 0
        fpr = fp / neg if neg else 0
        roc.append({"threshold": round(thr, 3), "tpr": round(tpr, 3), "fpr": round(fpr, 3)})
    return round(auc, 3), roc


# ─────────────────────────────────────────────────────────────────────────
# Validation experiments
# ─────────────────────────────────────────────────────────────────────────

def in_sample_baseline(cells):
    """In-sample acc at canonical threshold 0.43 (baseline reported in paper)."""
    return evaluate(cells, 0.43)


def loso_cv(cells):
    """Leave-one-scale-out: train threshold on 4 scales, test on 5th."""
    scales = sorted(set(c[0] for c in cells))
    fold_results = []
    pooled_correct = 0
    pooled_n = 0
    pooled_tp = pooled_fn = pooled_fp = pooled_tn = 0
    for held_scale in scales:
        train = [c for c in cells if c[0] != held_scale]
        test = [c for c in cells if c[0] == held_scale]
        thr, train_acc = threshold_search(train)
        test_eval = evaluate(test, thr)
        fold_results.append({
            "held_scale": held_scale,
            "train_threshold": round(thr, 3),
            "train_acc": round(train_acc, 3),
            "test_n": test_eval["n"],
            "test_acc": round(test_eval["accuracy"], 3),
            "test_balanced_acc": round(test_eval["balanced_accuracy"], 3),
        })
        pooled_correct += sum(d["correct"] for d in test_eval["details"])
        pooled_n += test_eval["n"]
        pooled_tp += test_eval["tp"]
        pooled_fn += test_eval["fn"]
        pooled_fp += test_eval["fp"]
        pooled_tn += test_eval["tn"]
    pooled_acc = pooled_correct / pooled_n if pooled_n else 0
    sens = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) else 0
    spec = pooled_tn / (pooled_tn + pooled_fp) if (pooled_tn + pooled_fp) else 0
    pooled_bal_acc = (sens + spec) / 2
    return {
        "folds": fold_results,
        "pooled_held_out_accuracy": round(pooled_acc, 3),
        "pooled_balanced_accuracy": round(pooled_bal_acc, 3),
        "pooled_tp": pooled_tp, "pooled_fn": pooled_fn,
        "pooled_fp": pooled_fp, "pooled_tn": pooled_tn,
        "pooled_n": pooled_n,
    }


def stress_test(cells):
    """S1+S2 train / S3+S4+S5 test — Codex's harsher extrapolation stress test."""
    train = [c for c in cells if c[0] in ("S1", "S2")]
    test = [c for c in cells if c[0] in ("S3", "S4", "S5")]
    thr, train_acc = threshold_search(train)
    test_eval = evaluate(test, thr)
    return {
        "train_scales": ["S1", "S2"],
        "test_scales": ["S3", "S4", "S5"],
        "train_n": len(train),
        "test_n": len(test),
        "train_threshold": round(thr, 3),
        "train_acc": round(train_acc, 3),
        "test_acc": round(test_eval["accuracy"], 3),
        "test_balanced_acc": round(test_eval["balanced_accuracy"], 3),
        "details": test_eval["details"],
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0, center - half), 3), round(min(1, center + half), 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paper_dir")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    paper = Path(args.paper_dir).resolve()

    print(f"Predictor validation — {paper}")
    print(
        f"GPT-2 calibration cells: {len(CALIBRATION_CELLS)}, "
        f"GPT-2 all/stress cells: {len(GPT2_CELLS)}, "
        f"Llama held-out: {len(LLAMA_CELLS)}"
    )

    # Baseline (in-sample @ 0.43) on the calibration set reported in the paper.
    baseline = in_sample_baseline(CALIBRATION_CELLS)
    baseline_all = in_sample_baseline(GPT2_CELLS)
    print(
        f"\n[Calibration baseline @ thr=0.43] accuracy = "
        f"{baseline['accuracy']:.3f} ({baseline['n']} cells)"
    )
    print(
        f"[All GPT-2 cells incl. Scale 5 @ thr=0.43] accuracy = "
        f"{baseline_all['accuracy']:.3f} ({baseline_all['n']} cells)"
    )

    # LOSO CV
    cv = loso_cv(CALIBRATION_CELLS)
    auc, roc = auc_score(CALIBRATION_CELLS)
    auc_all, roc_all = auc_score(GPT2_CELLS)
    print(f"\n[LOSO-CV] pooled held-out accuracy = {cv['pooled_held_out_accuracy']:.3f}")
    print(f"          balanced accuracy = {cv['pooled_balanced_accuracy']:.3f}")
    p_loso = cv['pooled_held_out_accuracy']
    n_loso = cv['pooled_n']
    ci = wilson_ci(p_loso, n_loso)
    print(f"          Wilson 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}] (n={n_loso})")
    print(f"          AUC over saturation = {auc:.3f}")
    for f in cv['folds']:
        print(f"   fold held={f['held_scale']}  thr={f['train_threshold']}  "
              f"test_acc={f['test_acc']:.3f}  bal_acc={f['test_balanced_acc']:.3f} (n={f['test_n']})")

    # Stress test
    stress = stress_test(GPT2_CELLS)
    print(f"\n[Stress test S1+S2 train / S3+S4+S5 test]")
    print(f"  train_thr = {stress['train_threshold']} (train_acc={stress['train_acc']:.3f})")
    print(f"  test_acc = {stress['test_acc']:.3f}, balanced = {stress['test_balanced_acc']:.3f}")

    # Llama OOD
    llama_eval = evaluate(LLAMA_CELLS, threshold=0.43)
    print(f"\n[Llama OOD (held-out architecture)] thr=0.43, n={llama_eval['n']}")
    print(f"  accuracy = {llama_eval['accuracy']:.3f} ({sum(1 for d in llama_eval['details'] if d['correct'])}/{llama_eval['n']})")
    for d in llama_eval['details']:
        print(f"   {d['cell']}: sat={d['sat']:.3f} delta={d['delta_pct']:+.1f}% "
              f"pred_helps={d['pred_helps']}, true_helps={d['true_helps']} "
              f"{'✓' if d['correct'] else '✗'}")

    payload = {
        "summary": {
            "in_sample_baseline_acc": baseline["accuracy"],
            "in_sample_baseline_n": baseline["n"],
            "auc_over_saturation": auc,
            "all_gpt2_in_sample_acc": baseline_all["accuracy"],
            "all_gpt2_n": baseline_all["n"],
            "all_gpt2_auc_over_saturation": auc_all,
            "loso_pooled_held_out_acc": cv["pooled_held_out_accuracy"],
            "loso_pooled_balanced_acc": cv["pooled_balanced_accuracy"],
            "loso_wilson_95_ci": ci,
            "stress_test_acc": stress["test_acc"],
            "llama_ood_acc": llama_eval["accuracy"],
            "llama_ood_correct_of_total": f"{sum(1 for d in llama_eval['details'] if d['correct'])}/{llama_eval['n']}",
        },
        "in_sample_baseline": baseline,
        "all_gpt2_in_sample_baseline": baseline_all,
        "loso_cv": cv,
        "stress_test": stress,
        "llama_ood": llama_eval,
        "roc": roc,
        "all_gpt2_roc": roc_all,
        "cells": {
            "gpt2_calibration": [
                {"name": c[5], "scale": c[0], "scale_M": c[1], "data_M": c[2],
                 "sat": c[3], "delta_pct": c[4], "true_helps": bool(label(c[4]))}
                for c in CALIBRATION_CELLS
            ],
            "gpt2": [
                {"name": c[5], "scale": c[0], "scale_M": c[1], "data_M": c[2],
                 "sat": c[3], "delta_pct": c[4], "true_helps": bool(label(c[4]))}
                for c in GPT2_CELLS
            ],
            "llama_held_out": [
                {"name": c[5], "scale": c[0], "scale_M": c[1], "data_M": c[2],
                 "sat": c[3], "delta_pct": c[4], "true_helps": bool(label(c[4]))}
                for c in LLAMA_CELLS
            ],
        },
    }

    out_path = Path(args.json_out) if args.json_out else paper / "docs" / "predictor_validation.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nJSON → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
