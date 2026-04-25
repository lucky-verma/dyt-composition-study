#!/usr/bin/env python3
"""Instantiate the main architecture variants and run one CPU forward pass."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from model import GPT, GPTConfig  # noqa: E402


def main() -> None:
    base = dict(
        vocab_size=128,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
    )
    variants = [
        ("vanilla", {}),
        ("dyt", {"use_dyt": True}),
        ("hardtanh", {"use_hardtanh": True}),
        ("rmsnorm", {"use_rmsnorm": True}),
        ("diffattn_v1", {"use_diff_attn": True}),
        ("diffattn_v2", {"use_diff_attn": True, "diff_attn_v2": True}),
        ("gated_attn", {"use_gated_attn": True}),
    ]
    x = torch.randint(0, 128, (2, 8))
    y = torch.randint(0, 128, (2, 8))
    for name, flags in variants:
        model = GPT(GPTConfig(**base, **flags))
        logits, loss = model(x, y)
        assert logits.shape == (2, 8, 128), (name, logits.shape)
        assert loss is not None and torch.isfinite(loss), name
        print(f"{name}: ok loss={loss.item():.4f}")
    print("model_smoke_ok")


if __name__ == "__main__":
    main()

