"""
Wave 3 T4 — Cross-Domain Evaluation

Load wikitext-trained Scale 1 (64M) checkpoints and compute OWT val perplexity.
Addresses reviewer R1 objection #5: no downstream eval. Provides cross-distribution
perplexity evidence (wikitext → OpenWebText) without additional training.

Audit A3: NO gradient updates. torch.no_grad() enforced. eval_iters=100.
Output: out/cross_domain_eval.json
"""
import torch
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, "<CODE_ROOT>")
from model import GPT, GPTConfig  # noqa

DEVICE = "cuda"
DTYPE = torch.bfloat16
EVAL_ITERS = 100
BATCH_SIZE = 64
BLOCK_SIZE = 512
OWT_VAL = "<CODE_ROOT>/data/openwebtext/val.bin"
OUT_JSON = "<CODE_ROOT>/out/cross_domain_eval.json"

# Ckpts to evaluate (canonical eff=64 from runset_3seed)
CKPTS = {
    # Scale 1 / 1M (overparameterized regime)
    "s1_1m_vanilla_s1337": "out/runset_3seed/wikitext_1m_vanilla_s1337/ckpt.pt",
    "s1_1m_vanilla_s42":   "out/runset_3seed/wikitext_1m_vanilla_s42/ckpt.pt",
    "s1_1m_vanilla_s7":    "out/runset_3seed/wikitext_1m_vanilla_s7/ckpt.pt",
    "s1_1m_dyt_s1337":     "out/runset_3seed/wikitext_1m_dyt_s1337/ckpt.pt",
    "s1_1m_dyt_s42":       "out/runset_3seed/wikitext_1m_dyt_s42/ckpt.pt",
    "s1_1m_dyt_s7":        "out/runset_3seed/wikitext_1m_dyt_s7/ckpt.pt",
    # Scale 1 / 118M (data-rich regime)
    "s1_118m_vanilla_s1337": "out/runset_3seed/wikitext_vanilla_s1337/ckpt.pt",
    "s1_118m_vanilla_s42":   "out/runset_3seed/wikitext_vanilla_s42/ckpt.pt",
    "s1_118m_vanilla_s7":    "out/runset_3seed/wikitext_vanilla_s7/ckpt.pt",
    "s1_118m_dyt_s1337":     "out/runset_3seed/wikitext_dyt_s1337/ckpt.pt",
    "s1_118m_dyt_s42":       "out/runset_3seed/wikitext_dyt_s42/ckpt.pt",
    "s1_118m_dyt_s7":        "out/runset_3seed/wikitext_dyt_s7/ckpt.pt",
}

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

@torch.no_grad()
def estimate_loss(model, data, n_batches):
    model.eval()
    losses = torch.zeros(n_batches)
    for k in range(n_batches):
        X, Y = get_batch(data, BATCH_SIZE, BLOCK_SIZE)
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            _, loss = model(X, Y)
        losses[k] = loss.item()
    return losses.mean().item()

def main():
    # Load OWT val.bin
    owt_val = np.memmap(OWT_VAL, dtype=np.uint16, mode="r")
    print(f"Loaded OWT val: {len(owt_val):,} tokens ({len(owt_val)/1e6:.1f}M)")

    results = {}
    for name, ckpt_path in CKPTS.items():
        full = f"<CODE_ROOT>/{ckpt_path}" if not ckpt_path.startswith("/") else ckpt_path
        if not Path(full).exists():
            print(f"SKIP missing: {full}")
            continue

        print(f"Loading {name} from {ckpt_path}")
        ck = torch.load(full, map_location=DEVICE, weights_only=False)
        
        # Audit A3: verify eff_batch=64 on source ckpt
        cfg = ck.get("config") or {}
        bs = cfg.get("batch_size", -1)
        ga = cfg.get("gradient_accumulation_steps", -1)
        eff = bs * ga if bs > 0 and ga > 0 else -1
        assert eff == 64, f"AUDIT FAIL: {name} ckpt has eff_batch={eff}, expected 64"
        
        model_args = ck["model_args"]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf).to(DEVICE)
        
        state_dict = ck["model"]
        # strip torch.compile prefix if present
        unwanted = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted):
                state_dict[k[len(unwanted):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        owt_val_loss = estimate_loss(model, owt_val, EVAL_ITERS)
        wikitext_val_loss = ck.get("best_val_loss")
        if hasattr(wikitext_val_loss, "item"):
            wikitext_val_loss = wikitext_val_loss.item()
        
        results[name] = {
            "owt_val_loss": owt_val_loss,
            "owt_val_ppl": float(np.exp(owt_val_loss)),
            "wikitext_val_loss": wikitext_val_loss,
            "wikitext_val_ppl": float(np.exp(wikitext_val_loss)) if wikitext_val_loss else None,
            "eff_batch": eff,
            "ckpt": ckpt_path,
        }
        print(f"  OWT val_loss={owt_val_loss:.4f} ppl={np.exp(owt_val_loss):.2f}")
        del model, ck
        torch.cuda.empty_cache()
    
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON} with {len(results)} entries")

if __name__ == "__main__":
    main()
