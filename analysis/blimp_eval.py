#!/usr/bin/env python3
"""BLIMP syntactic acceptability eval — minimal custom for our GPT-2 model class.
Runs on compute environment. Uses nyu-mll/blimp from HuggingFace datasets.

For each phenomenon pair (good_sentence, bad_sentence):
  Compute log P(sentence) under model = -sum(per_token_NLL).
  Accuracy = fraction of pairs where log P(good) > log P(bad).

Eval scope: 18 ckpts (Scale 1-3 × vanilla/DyT × 3 seeds), 3 phenomena × 1000 pairs = 3000 pairs/ckpt.
Wall: ~2-5min per ckpt on H100 NVL → ~1-1.5hr total.

ADR 0007 Layer B: W&B logged per ckpt with full config dict.
"""
import os, json, sys, time, argparse
import torch
import numpy as np
from pathlib import Path

CODE_ROOT = '<CODE_ROOT>'
sys.path.insert(0, CODE_ROOT)
from model import GPT, GPTConfig
import tiktoken

OUT = Path(f'{CODE_ROOT}/out/blimp_eval')
OUT.mkdir(exist_ok=True, parents=True)

# 3 representative phenomena (syntactic phenomena where wikitext models can differentiate)
PHENOMENA = ['anaphor_number_agreement', 'determiner_noun_agreement_1', 'regular_plural_subject_verb_agreement_1']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding('gpt2')


def load_blimp():
    try:
        from datasets import load_dataset
        data = {}
        for p in PHENOMENA:
            data[p] = load_dataset('nyu-mll/blimp', p, split='train')
        return data
    except Exception as e:
        print(f'[blimp] ERROR loading dataset: {e}', flush=True)
        sys.exit(1)


def log_prob(model, tokens, block_size):
    """Compute sum log P under model for token sequence."""
    ids = torch.tensor([tokens], device=device, dtype=torch.long)
    if ids.size(1) < 2: return 0.0
    if ids.size(1) > block_size:
        ids = ids[:, -block_size:]
    x = ids[:, :-1]
    y = ids[:, 1:]
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(x, y)
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    per_tok = log_probs.gather(2, y.unsqueeze(-1)).squeeze(-1)
    return float(per_tok.sum().item())


def eval_ckpt(ckpt_path, blimp_data):
    out_file = OUT / f'{ckpt_path.parent.parent.name}_{ckpt_path.parent.name}.json'
    if out_file.exists() and out_file.stat().st_size > 100:
        print(f'[blimp] SKIP {out_file.name}', flush=True)
        return json.load(open(out_file))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ma = ckpt['model_args']
    cfg = GPTConfig(**{k: v for k, v in ma.items() if k in GPTConfig.__dataclass_fields__})
    m = GPT(cfg).to(device)
    state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    m.load_state_dict(state, strict=False)
    m.eval()

    block = ma.get('block_size', 512)
    results = {}
    t0 = time.time()
    for phen, ds in blimp_data.items():
        correct = total = 0
        for ex in ds:
            g_tok = enc.encode(ex['sentence_good'])
            b_tok = enc.encode(ex['sentence_bad'])
            try:
                lpg = log_prob(m, g_tok, block)
                lpb = log_prob(m, b_tok, block)
                if lpg > lpb: correct += 1
                total += 1
            except Exception:
                continue
        results[phen] = {'accuracy': correct / max(total, 1), 'n': total}
    wall = time.time() - t0

    mean_acc = sum(r['accuracy'] for r in results.values()) / len(results)
    out = {
        'ckpt_path': str(ckpt_path),
        'folder': ckpt_path.parent.parent.name,
        'name': ckpt_path.parent.name,
        'arch': 'dyt' if ma.get('use_dyt') else ('diffattn' if ma.get('use_diff_attn') else 'vanilla'),
        'model_args_core': {k: ma.get(k) for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'use_dyt', 'use_diff_attn']},
        'phenomena': results,
        'mean_accuracy': mean_acc,
        'wall_s': wall,
    }
    json.dump(out, open(out_file, 'w'), indent=2)
    print(f'[blimp] DONE {out_file.name} mean_acc={mean_acc:.3f} wall={wall:.1f}s', flush=True)
    del m, ckpt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folders', nargs='+', default=[
        'runset_3seed', 'scale2', 'scale3',
    ])
    ap.add_argument('--wandb_log', action='store_true')
    args = ap.parse_args()

    print('[blimp] loading BLIMP...', flush=True)
    blimp_data = load_blimp()
    for p, ds in blimp_data.items():
        print(f'  {p}: {len(ds)} pairs', flush=True)

    if args.wandb_log:
        import wandb
        wandb.init(project='composition-study', entity='anonymous_entity',
                   group='blimp_eval', name=f'blimp_sweep_{int(time.time())}', mode='online',
                   config={'phenomena': PHENOMENA, 'folders': args.folders})

    all_results = []
    for folder in args.folders:
        base = Path(f'{CODE_ROOT}/out/{folder}')
        if not base.exists():
            print(f'[blimp] skip {folder}: not present', flush=True)
            continue
        for ckpt in sorted(base.glob('*/ckpt.pt')):
            # Only 118M-trained vanilla/DyT (skip 1M + DiffAttn + scale-1M)
            name = ckpt.parent.name
            if '_1m' in name or 'diffattn' in name: continue
            if not ('vanilla' in name or 'dyt' in name): continue
            print(f'[blimp] eval {folder}/{name}', flush=True)
            r = eval_ckpt(ckpt, blimp_data)
            if args.wandb_log:
                wandb.log({f'blimp/{folder}/{name}/mean_accuracy': r['mean_accuracy']})
            all_results.append(r)

    # Summary
    summary = {'n_ckpts': len(all_results),
               'by_arch': {}}
    for r in all_results:
        a = r['arch']; summary['by_arch'].setdefault(a, []).append(r['mean_accuracy'])
    for a, accs in summary['by_arch'].items():
        m = sum(accs) / len(accs)
        s = (sum((x - m) ** 2 for x in accs) / max(len(accs) - 1, 1)) ** 0.5
        summary['by_arch'][a] = {'n': len(accs), 'mean_acc': round(m, 4), 'std': round(s, 4)}

    json.dump({'summary': summary, 'per_ckpt': all_results},
              open(OUT / 'aggregate.json', 'w'), indent=2)
    print('[blimp] COMPLETE', json.dumps(summary, indent=2), flush=True)
    if args.wandb_log:
        wandb.finish()


if __name__ == '__main__':
    main()
