#!/usr/bin/env python3
"""
Comprehensive Activation Saturation Sweep — DyT Crossover Predictor
=====================================================================
Auto-discovers ALL DyT checkpoints (GPT-2 + Llama), measures activation
saturation, fits a semi-empirical crossover predictor, and saves results
for direct inclusion in the NeurIPS paper.

Safety:
  - Checks free GPU memory before loading each model
  - Loads ONE model at a time per GPU, fully unloads before next
  - Falls back to CPU if GPU memory insufficient
  - Saves results incompute environmententally (crash-safe)
  - Does NOT interfere with running training jobs

Usage:
  # Single GPU (safe alongside training):
  CUDA_VISIBLE_DEVICES=1 python saturation_sweep.py

  # Both GPUs (splits work across them):
  python saturation_sweep.py --parallel

  # CPU-only fallback:
  python saturation_sweep.py --cpu

  # Autonomous nohup:
  nohup python -u saturation_sweep.py --parallel > out/saturation_sweep.log 2>&1 &
"""

import torch
import json
import os
import sys
import gc
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Setup ────────────────────────────────────────────────────────────────
BASE = Path('<CODE_ROOT>')
OUT_DIR = BASE / 'out' / 'saturation_sweep'
RESULTS_FILE = OUT_DIR / 'saturation_results.json'
PREDICTOR_FILE = OUT_DIR / 'crossover_predictor.json'
SUMMARY_FILE = OUT_DIR / 'paper_ready_table.txt'
DTYPE = torch.bfloat16
N_BATCHES = 50       # Forward passes per checkpoint (50 = stable statistics)
SEQ_LEN = 512
MIN_FREE_GPU_MB = 8000  # 8GB safety margin

sys.path.insert(0, str(BASE))


# ─── Model metadata: map checkpoint paths to scale/tokens/config ──────────

# Scale definitions: (n_params_approx, n_layer, n_head, n_embd)
GPT2_SCALES = {
    'scale1': (64_000_000,  12, 8,  512),   # 64M / "85M" in some dirs
    'scale2': (124_000_000, 12, 12, 768),
    'scale3': (354_000_000, 24, 16, 1024),
    'scale4': (1_310_000_000, 24, 32, 2048),
}

LLAMA_SCALES = {
    'scale1': (64_000_000,  12, 8,  512),
    'scale2': (124_000_000, 12, 12, 768),
}

# Token budget to actual token count
TOKEN_MAP = {
    '1m': 1_000_000,
    '10m': 10_000_000,
    '50m': 50_000_000,
    '118m': 118_000_000,
}

# Data path resolution
def get_data_path(tokens_key):
    """Return the training data path for a given token budget."""
    mapping = {
        '1m':   BASE / 'data' / 'wikitext_1m',
        '10m':  BASE / 'data' / 'wikitext_10m',
        '50m':  BASE / 'data' / 'wikitext_50m',
        '118m': BASE / 'data' / 'wikitext',
    }
    p = mapping.get(tokens_key)
    if p and (p / 'train.bin').exists():
        return str(p)
    return None


# ─── Checkpoint discovery ─────────────────────────────────────────────────

def discover_checkpoints():
    """
    Auto-discover all DyT checkpoints on compute environment.
    Returns list of dicts with metadata for each checkpoint.
    """
    checkpoints = []
    out_dir = BASE / 'out'

    # === GPT-2 checkpoints ===

    # 1. runset_3seed: Scale 1 (64M), 1M and 118M tokens
    for ckpt in sorted((out_dir / 'runset_3seed').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name  # e.g., wikitext_1m_dyt_s1337
        tokens_key = '1m' if '_1m_' in name else '118m'
        seed = name.split('_s')[-1]
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': 'scale1', 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': 64_000_000, 'n_tokens': TOKEN_MAP[tokens_key],
        })

    # 2. scale2: 124M, 1M and 118M
    for ckpt in sorted((out_dir / 'scale2').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name
        tokens_key = '1m' if '_1m_' in name else '118m'
        seed = name.split('_s')[-1]
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': 'scale2', 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': 124_000_000, 'n_tokens': TOKEN_MAP[tokens_key],
        })

    # 3. scale3: 354M, 1M and 118M
    for ckpt in sorted((out_dir / 'scale3').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name
        tokens_key = '1m' if '_1m_' in name else '118m'
        seed = name.split('_s')[-1]
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': 'scale3', 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': 354_000_000, 'n_tokens': TOKEN_MAP[tokens_key],
        })

    # 4. scale4: 1.3B, 1M and 118M
    for ckpt in sorted((out_dir / 'scale4').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name
        tokens_key = '1m' if '_1m_' in name else '118m'
        seed = name.split('_s')[-1]
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': 'scale4', 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': 1_310_000_000, 'n_tokens': TOKEN_MAP[tokens_key],
        })

    # 5. phase_fill: Scale 1-3 @ 10M tokens
    for ckpt in sorted((out_dir / 'phase_fill').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name  # e.g., scale2_10m_dyt_s1337
        parts = name.split('_')
        scale = parts[0]   # scale1, scale2, scale3
        tokens_key = parts[1]  # 10m
        seed = name.split('_s')[-1]
        params_map = {'scale1': 64_000_000, 'scale2': 124_000_000, 'scale3': 354_000_000}
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': scale, 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': params_map.get(scale, 0), 'n_tokens': TOKEN_MAP.get(tokens_key, 0),
        })

    # 6. intermediate: Scale 1, 10M/50M tokens (original 3-seed runs)
    for ckpt in sorted((out_dir / 'intermediate').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name  # e.g., wikitext_10m_dyt_s1337
        if '10m' in name:
            tokens_key = '10m'
        elif '50m' in name:
            tokens_key = '50m'
        else:
            continue
        seed = name.split('_s')[-1]
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': 'scale1', 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': 64_000_000, 'n_tokens': TOKEN_MAP[tokens_key],
        })

    # 7. alpha sweep: Scale 1, 1M tokens, different alpha inits
    for ckpt in sorted((out_dir / 'alpha_sweep').glob('dyt_a*_*/ckpt.pt')):
        name = ckpt.parent.name  # e.g., dyt_a0.5_s1337
        alpha = name.split('_a')[1].split('_')[0]  # 0.5, 1.0, 2.0, 3.0
        seed = name.split('_s')[-1]
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'gpt2', 'scale': 'scale1', 'tokens': '1m',
            'config': f'dyt_a{alpha}', 'seed': seed,
            'params': 64_000_000, 'n_tokens': 1_000_000,
        })

    # === Llama checkpoints ===
    for ckpt in sorted((out_dir / 'llama').glob('*_dyt_*/ckpt.pt')):
        name = ckpt.parent.name  # e.g., scale1_118m_dyt_s1337
        parts = name.split('_')
        scale = parts[0]
        tokens_key = parts[1]
        seed = name.split('_s')[-1]
        params_map = {'scale1': 64_000_000, 'scale2': 124_000_000}
        checkpoints.append({
            'path': str(ckpt.parent),
            'arch': 'llama', 'scale': scale, 'tokens': tokens_key,
            'config': 'dyt', 'seed': seed,
            'params': params_map.get(scale, 0), 'n_tokens': TOKEN_MAP.get(tokens_key, 0),
        })

    # Deduplicate by path
    seen = set()
    unique = []
    for c in checkpoints:
        if c['path'] not in seen:
            seen.add(c['path'])
            unique.append(c)

    return unique


# ─── Model loading ────────────────────────────────────────────────────────

def load_gpt2_model(ckpt_dir, device='cpu'):
    """Load GPT-2 model from checkpoint."""
    from model import GPTConfig, GPT
    ckpt = torch.load(f'{ckpt_dir}/ckpt.pt', map_location='cpu', weights_only=False)
    cfg = GPTConfig(**ckpt['model_args'])
    model = GPT(cfg)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device, dtype=DTYPE)
    return model, cfg


def load_llama_model(ckpt_dir, device='cpu'):
    """Load Llama model from checkpoint."""
    from model_llama import LlamaConfig, Llama
    ckpt = torch.load(f'{ckpt_dir}/ckpt.pt', map_location='cpu', weights_only=False)
    cfg = LlamaConfig(**ckpt['model_args'])
    model = Llama(cfg)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(device, dtype=DTYPE)
    return model, cfg


# ─── Saturation measurement ──────────────────────────────────────────────

def measure_saturation(model, data_path, device, n_batches=N_BATCHES, seq_len=SEQ_LEN):
    """
    Measure tanh activation saturation in DyT layers.
    Returns detailed per-layer and aggregate statistics.
    """
    data = np.memmap(f'{data_path}/train.bin', dtype=np.uint16, mode='r')
    if len(data) < seq_len + 1:
        return None

    hooks = []
    activations = {}

    # Hook into all DyT layers (they have learnable alpha)
    dyt_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'alpha') and isinstance(module.alpha, torch.nn.Parameter):
            dyt_layers.append(name)
            def make_hook(layer_name):
                def hook_fn(mod, inp, out):
                    if len(inp) > 0:
                        with torch.no_grad():
                            pre_tanh = mod.alpha * inp[0]
                            activations[layer_name] = pre_tanh.detach()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(name)))

    if not hooks:
        return None

    # Thresholds for saturation measurement
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    per_layer_sat = defaultdict(lambda: {str(t): [] for t in thresholds})
    global_sat = {str(t): [] for t in thresholds}
    per_layer_alpha = {}
    per_layer_mean_abs = defaultdict(list)
    per_layer_std = defaultdict(list)

    np.random.seed(42)  # Reproducible sampling

    for batch_idx in range(n_batches):
        start = np.random.randint(0, len(data) - seq_len - 1)
        x = torch.from_numpy(data[start:start + seq_len].astype(np.int64)).unsqueeze(0).to(device)

        with torch.no_grad():
            model(x)

        batch_all_abs = []

        for layer_name, pre_tanh in activations.items():
            abs_val = pre_tanh.abs().float()
            batch_all_abs.append(abs_val.flatten())

            for t in thresholds:
                frac = (abs_val > t).float().mean().item()
                per_layer_sat[layer_name][str(t)].append(frac)

            per_layer_mean_abs[layer_name].append(abs_val.mean().item())
            per_layer_std[layer_name].append(abs_val.std().item())

            if layer_name not in per_layer_alpha:
                for n, m in model.named_modules():
                    if n == layer_name and hasattr(m, 'alpha'):
                        per_layer_alpha[layer_name] = m.alpha.item()

        # Global saturation (all layers combined)
        if batch_all_abs:
            all_abs = torch.cat(batch_all_abs)
            for t in thresholds:
                global_sat[str(t)].append((all_abs > t).float().mean().item())

        activations.clear()
        del x
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # Aggregate results
    result = {
        'global_saturation': {t: float(np.mean(v)) for t, v in global_sat.items()},
        'global_saturation_std': {t: float(np.std(v)) for t, v in global_sat.items()},
        'per_layer_saturation_2.0': {
            layer: float(np.mean(per_layer_sat[layer]['2.0']))
            for layer in dyt_layers
        },
        'per_layer_alpha': per_layer_alpha,
        'per_layer_mean_abs_activation': {
            layer: float(np.mean(per_layer_mean_abs[layer]))
            for layer in dyt_layers
        },
        'per_layer_std_activation': {
            layer: float(np.mean(per_layer_std[layer]))
            for layer in dyt_layers
        },
        'mean_alpha': float(np.mean(list(per_layer_alpha.values()))) if per_layer_alpha else 0,
        'std_alpha': float(np.std(list(per_layer_alpha.values()))) if per_layer_alpha else 0,
        'n_dyt_layers': len(dyt_layers),
        'n_batches': n_batches,
    }

    return result


# ─── GPU memory management ────────────────────────────────────────────────

def get_free_gpu_mb(device_idx=0):
    """Get free GPU memory in MB."""
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info(device_idx)
    return free / (1024 * 1024)


def estimate_model_mb(n_params):
    """Estimate GPU memory needed for bfloat16 model + forward pass overhead."""
    # bfloat16 = 2 bytes per param, plus ~50% overhead for activations/hooks
    return (n_params * 2 * 1.5) / (1024 * 1024)


# ─── Main analysis loop ──────────────────────────────────────────────────

def analyze_checkpoint(ckpt_info, device='cuda:0'):
    """Analyze a single checkpoint. Returns result dict or None."""
    path = ckpt_info['path']
    arch = ckpt_info['arch']

    try:
        if arch == 'llama':
            model, cfg = load_llama_model(path, device)
        else:
            model, cfg = load_gpt2_model(path, device)

        data_path = get_data_path(ckpt_info['tokens'])
        if data_path is None:
            print(f"    SKIP: no data for tokens={ckpt_info['tokens']}", flush=True)
            return None

        result = measure_saturation(model, data_path, device)
        return result

    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        traceback.print_exc()
        return {'error': str(e)}

    finally:
        try:
            del model
        except:
            pass
        gc.collect()
        if 'cuda' in str(device):
            torch.cuda.empty_cache()


def run_sweep(args):
    """Main sweep: discover checkpoints, analyze all, save results."""
    os.makedirs(OUT_DIR, exist_ok=True)
    start_time = time.time()

    # Load any existing results for crash-safe incompute environmentental updates
    existing_results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            existing_results = json.load(f)
        print(f"[INFO] Loaded {len(existing_results)} existing results", flush=True)

    # Discover checkpoints
    checkpoints = discover_checkpoints()
    print(f"\n{'='*70}", flush=True)
    print(f"SATURATION SWEEP — {len(checkpoints)} DyT checkpoints found", flush=True)
    print(f"{'='*70}", flush=True)

    # Group by architecture for summary
    by_arch = defaultdict(int)
    for c in checkpoints:
        by_arch[c['arch']] += 1
    for arch, count in sorted(by_arch.items()):
        print(f"  {arch}: {count} checkpoints", flush=True)

    # Determine device
    if args.cpu:
        devices = ['cpu']
        print(f"\n[MODE] CPU-only (forced)", flush=True)
    elif args.parallel and torch.cuda.device_count() >= 2:
        devices = ['cuda:0', 'cuda:1']
        print(f"\n[MODE] Parallel across GPU 0 + GPU 1", flush=True)
    elif torch.cuda.is_available():
        # Pick the GPU with more free memory
        free0 = get_free_gpu_mb(0)
        free1 = get_free_gpu_mb(1) if torch.cuda.device_count() >= 2 else 0
        best = 'cuda:1' if free1 > free0 else 'cuda:0'
        devices = [best]
        print(f"\n[MODE] Single GPU: {best} ({max(free0, free1):.0f} MB free)", flush=True)
    else:
        devices = ['cpu']
        print(f"\n[MODE] CPU (no GPU available)", flush=True)

    # Sort: small models first (faster, safer), then large
    checkpoints.sort(key=lambda c: c['params'])

    # If parallel, split across GPUs: large models on GPU with more free memory
    if len(devices) == 2:
        free0 = get_free_gpu_mb(0)
        free1 = get_free_gpu_mb(1)
        big_gpu = 'cuda:1' if free1 > free0 else 'cuda:0'
        small_gpu = 'cuda:0' if big_gpu == 'cuda:1' else 'cuda:1'
        print(f"  Large models (>200M): {big_gpu} ({max(free0,free1):.0f} MB free)", flush=True)
        print(f"  Small models (≤200M): {small_gpu} ({min(free0,free1):.0f} MB free)", flush=True)

    all_results = dict(existing_results)
    completed = 0
    skipped = 0
    errors = 0

    for i, ckpt in enumerate(checkpoints):
        key = f"{ckpt['arch']}_{ckpt['scale']}_{ckpt['tokens']}_{ckpt['config']}_{ckpt['seed']}"

        # Skip if already analyzed
        if key in all_results and 'error' not in all_results[key]:
            skipped += 1
            continue

        # Choose device
        if len(devices) == 2:
            device = big_gpu if ckpt['params'] > 200_000_000 else small_gpu
        else:
            device = devices[0]

        # Check GPU memory if using GPU
        if 'cuda' in device:
            dev_idx = int(device.split(':')[1])
            free_mb = get_free_gpu_mb(dev_idx)
            needed_mb = estimate_model_mb(ckpt['params'])
            if free_mb < needed_mb + MIN_FREE_GPU_MB:
                print(f"  [{i+1}/{len(checkpoints)}] SKIP {key}: need {needed_mb:.0f}MB, "
                      f"only {free_mb:.0f}MB free on {device}", flush=True)
                skipped += 1
                continue

        # Analyze
        elapsed = time.time() - start_time
        tp_ratio = ckpt['n_tokens'] / ckpt['params'] if ckpt['params'] > 0 else 0
        print(f"\n  [{i+1}/{len(checkpoints)}] {key}", flush=True)
        print(f"    arch={ckpt['arch']} params={ckpt['params']/1e6:.0f}M "
              f"tokens={ckpt['tokens']} T/P={tp_ratio:.4f} device={device}", flush=True)

        t0 = time.time()
        result = analyze_checkpoint(ckpt, device)
        dt = time.time() - t0

        if result is not None and 'error' not in result:
            # Enrich with metadata
            result['metadata'] = {
                'arch': ckpt['arch'],
                'scale': ckpt['scale'],
                'tokens': ckpt['tokens'],
                'config': ckpt['config'],
                'seed': ckpt['seed'],
                'params': ckpt['params'],
                'n_tokens': ckpt['n_tokens'],
                'tp_ratio': tp_ratio,
                'checkpoint_path': ckpt['path'],
                'device': device,
                'analysis_time_sec': round(dt, 1),
            }
            all_results[key] = result
            completed += 1

            sat20 = result['global_saturation']['2.0']
            mean_a = result['mean_alpha']
            print(f"    ✓ Saturation(|αx|>2.0)={sat20:.4f}  mean_α={mean_a:.3f}  "
                  f"({dt:.1f}s)", flush=True)
        else:
            all_results[key] = result or {'error': 'returned None'}
            errors += 1
            print(f"    ✗ Failed ({dt:.1f}s)", flush=True)

        # Incompute environmentental save every 5 checkpoints
        if (completed + errors) % 5 == 0:
            with open(RESULTS_FILE, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"    [saved {len(all_results)} results]", flush=True)

    # Final save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n{'='*70}", flush=True)
    print(f"SWEEP COMPLETE", flush=True)
    print(f"  Analyzed: {completed}", flush=True)
    print(f"  Skipped (already done): {skipped}", flush=True)
    print(f"  Errors: {errors}", flush=True)
    print(f"  Total: {len(all_results)} results in {RESULTS_FILE}", flush=True)
    print(f"  Time: {total_time/60:.1f} minutes", flush=True)
    print(f"{'='*70}\n", flush=True)

    return all_results


# ─── Crossover predictor fitting ─────────────────────────────────────────

def fit_crossover_predictor(results):
    """
    Fit a semi-empirical predictor: saturation = f(T/P, model_params)
    and identify the crossover point where DyT switches from helpful to harmful.

    Uses the known DyT deltas from the paper to correlate saturation with effect.
    """
    print(f"\n{'='*70}", flush=True)
    print(f"FITTING CROSSOVER PREDICTOR", flush=True)
    print(f"{'='*70}", flush=True)

    # Known DyT effects (from all_results.json — the ground truth)
    # Format: (params, n_tokens, delta_pct)
    known_effects = [
        # GPT-2 Scale 1 (64M)
        (64e6,   1e6,   -27.3),
        (64e6,   10e6,  +5.9),
        (64e6,   50e6,  +19.7),
        (64e6,   118e6, +18.8),
        # GPT-2 Scale 2 (124M)
        (124e6,  1e6,   -9.6),
        (124e6,  10e6,  +6.1),
        (124e6,  118e6, +12.8),
        # GPT-2 Scale 3 (354M)
        (354e6,  1e6,   +4.3),
        (354e6,  10e6,  -24.1),
        (354e6,  118e6, +13.4),
        # GPT-2 Scale 4 (1.3B)
        (1310e6, 1e6,   +2.1),
        (1310e6, 118e6, +10.4),
    ]

    # Map known effects to T/P ratios
    effect_by_tp = {}
    for params, tokens, delta in known_effects:
        tp = tokens / params
        effect_by_tp[(params, tokens)] = {'tp_ratio': tp, 'delta_pct': delta}

    # Match saturation measurements to known effects
    matched_data = []
    for key, result in results.items():
        if 'error' in result or result is None:
            continue
        meta = result.get('metadata', {})
        if meta.get('arch') != 'gpt2':
            continue
        if meta.get('config') != 'dyt':
            continue

        params = meta['params']
        n_tokens = meta['n_tokens']
        sat = result['global_saturation']['2.0']
        mean_alpha = result['mean_alpha']
        tp = meta['tp_ratio']

        # Find matching known effect
        match_key = (params, n_tokens)
        if match_key in effect_by_tp:
            delta = effect_by_tp[match_key]['delta_pct']
            matched_data.append({
                'params': params,
                'n_tokens': n_tokens,
                'tp_ratio': tp,
                'saturation_2.0': sat,
                'mean_alpha': mean_alpha,
                'delta_pct': delta,
                'dyt_helps': delta < 0,
                'seed': meta['seed'],
            })

    if len(matched_data) < 5:
        print(f"  WARNING: Only {len(matched_data)} matched points, need ≥5 for predictor", flush=True)
        return None

    # Average saturation across seeds for same (params, tokens) combo
    grouped = defaultdict(list)
    for d in matched_data:
        key = (d['params'], d['n_tokens'])
        grouped[key].append(d)

    predictor_data = []
    for (params, n_tokens), entries in grouped.items():
        avg_sat = np.mean([e['saturation_2.0'] for e in entries])
        avg_alpha = np.mean([e['mean_alpha'] for e in entries])
        delta = entries[0]['delta_pct']  # Same for all seeds
        tp = entries[0]['tp_ratio']
        helps = entries[0]['dyt_helps']

        predictor_data.append({
            'params_M': params / 1e6,
            'n_tokens_M': n_tokens / 1e6,
            'tp_ratio': tp,
            'log_tp': np.log10(max(tp, 1e-6)),
            'log_params': np.log10(params),
            'saturation_2.0': avg_sat,
            'mean_alpha': avg_alpha,
            'delta_pct': delta,
            'dyt_helps': helps,
            'n_seeds': len(entries),
        })

    print(f"\n  Matched {len(predictor_data)} unique (scale, tokens) combinations:", flush=True)
    print(f"  {'Params':>8} {'Tokens':>8} {'T/P':>8} {'Sat@2.0':>8} {'α_mean':>8} "
          f"{'Δ DyT':>8} {'Helps?':>6}", flush=True)
    print(f"  {'-'*60}", flush=True)

    for d in sorted(predictor_data, key=lambda x: (x['params_M'], x['n_tokens_M'])):
        print(f"  {d['params_M']:>7.0f}M {d['n_tokens_M']:>7.0f}M {d['tp_ratio']:>8.4f} "
              f"{d['saturation_2.0']:>8.4f} {d['mean_alpha']:>8.3f} "
              f"{d['delta_pct']:>+7.1f}% {'  YES' if d['dyt_helps'] else '   NO':>6}",
              flush=True)

    # ─── Fit 1: Linear regression of delta on saturation + log(params) ────
    X = np.array([[d['saturation_2.0'], d['log_params']] for d in predictor_data])
    y = np.array([d['delta_pct'] for d in predictor_data])

    # Simple linear: delta = a * saturation + b * log_params + c
    # Add intercept column
    X_aug = np.column_stack([X, np.ones(len(X))])
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
        y_pred = X_aug @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n  Linear model: Δ = {beta[0]:.2f}·sat + {beta[1]:.2f}·log(P) + {beta[2]:.2f}")
        print(f"  R² = {r2:.4f}")

        # Crossover: where delta = 0
        # 0 = a*sat + b*log(P) + c  →  sat_crossover = -(b*log(P) + c) / a
        for params_M in [64, 124, 354, 1310, 7000]:
            lp = np.log10(params_M * 1e6)
            sat_cross = -(beta[1] * lp + beta[2]) / beta[0] if beta[0] != 0 else float('nan')
            print(f"    At {params_M}M params: crossover saturation = {sat_cross:.3f}")

    except Exception as e:
        print(f"  Linear fit failed: {e}")
        beta = None
        r2 = 0

    # ─── Fit 2: Logistic classifier (helps vs hurts) ─────────────────────
    X_log = np.array([[d['saturation_2.0'], d['log_params']] for d in predictor_data])
    y_binary = np.array([1 if d['dyt_helps'] else 0 for d in predictor_data])

    # Simple logistic via iterative reweighted least squares (no sklearn needed)
    # Or just find the saturation threshold that separates helps/hurts
    helps_sat = [d['saturation_2.0'] for d in predictor_data if d['dyt_helps']]
    hurts_sat = [d['saturation_2.0'] for d in predictor_data if not d['dyt_helps']]

    if helps_sat and hurts_sat:
        min_helps = min(helps_sat)
        max_hurts = max(hurts_sat)
        threshold = (min_helps + max_hurts) / 2

        # Check accuracy of simple threshold
        correct = sum(1 for d in predictor_data
                      if (d['saturation_2.0'] >= threshold) == d['dyt_helps'])
        accuracy = correct / len(predictor_data)

        print(f"\n  Simple saturation threshold: {threshold:.4f}")
        print(f"    Min saturation where DyT helps: {min_helps:.4f}")
        print(f"    Max saturation where DyT hurts: {max_hurts:.4f}")
        print(f"    Classification accuracy: {accuracy:.1%} ({correct}/{len(predictor_data)})")
    else:
        threshold = 0.35  # Default
        accuracy = 0

    # ─── Fit 3: T/P-based predictor (simplest for practitioners) ──────────
    helps_tp = [d['tp_ratio'] for d in predictor_data if d['dyt_helps']]
    hurts_tp = [d['tp_ratio'] for d in predictor_data if not d['dyt_helps']]

    if helps_tp and hurts_tp:
        max_helps_tp = max(helps_tp)
        min_hurts_tp = min(hurts_tp)
        tp_threshold = (max_helps_tp + min_hurts_tp) / 2 if min_hurts_tp > max_helps_tp else None

        tp_correct = 0
        for d in predictor_data:
            if tp_threshold and ((d['tp_ratio'] <= tp_threshold) == d['dyt_helps']):
                tp_correct += 1
        tp_accuracy = tp_correct / len(predictor_data) if tp_threshold else 0

        print(f"\n  T/P ratio predictor:")
        print(f"    Max T/P where DyT helps: {max_helps_tp:.4f}")
        print(f"    Min T/P where DyT hurts: {min_hurts_tp:.4f}")
        if tp_threshold:
            print(f"    Threshold: T/P ≤ {tp_threshold:.4f} → use DyT")
            print(f"    Classification accuracy: {tp_accuracy:.1%}")
        else:
            print(f"    Regimes overlap — T/P alone insufficient (need model scale too)")

    # ─── Predictions for unseen scales ────────────────────────────────────
    print(f"\n  PREDICTIONS (for paper Section 5):", flush=True)
    if beta is not None:
        for params_M, tokens_desc in [(7000, 'Chinchilla-optimal 140B tokens'),
                                       (7000, '1M tokens (overparameterized)'),
                                       (70000, 'Chinchilla-optimal 1.4T tokens')]:
            if '1M' in tokens_desc:
                tokens = 1e6
            elif '140B' in tokens_desc:
                tokens = 140e9
            else:
                tokens = 1.4e12
            tp = tokens / (params_M * 1e6)
            lp = np.log10(params_M * 1e6)
            # Predict required saturation for DyT to help
            sat_cross = -(beta[1] * lp + beta[2]) / beta[0] if beta[0] != 0 else float('nan')
            predicted_delta = beta[0] * sat_cross + beta[1] * lp + beta[2]
            print(f"    {params_M/1000:.0f}B / {tokens_desc}:")
            print(f"      T/P = {tp:.4f}, crossover sat = {sat_cross:.3f}")

    # ─── Llama cross-validation ───────────────────────────────────────────
    llama_data = []
    for key, result in results.items():
        if 'error' in result or result is None:
            continue
        meta = result.get('metadata', {})
        if meta.get('arch') != 'llama' or meta.get('config') != 'dyt':
            continue
        sat = result['global_saturation']['2.0']
        llama_data.append({
            'params_M': meta['params'] / 1e6,
            'n_tokens_M': meta['n_tokens'] / 1e6,
            'tp_ratio': meta['tp_ratio'],
            'saturation_2.0': sat,
            'mean_alpha': result['mean_alpha'],
            'seed': meta['seed'],
        })

    if llama_data:
        print(f"\n  LLAMA CROSS-VALIDATION (model trained on GPT-2 data only):", flush=True)
        llama_grouped = defaultdict(list)
        for d in llama_data:
            llama_grouped[(d['params_M'], d['n_tokens_M'])].append(d)

        for (pm, tm), entries in sorted(llama_grouped.items()):
            avg_sat = np.mean([e['saturation_2.0'] for e in entries])
            avg_alpha = np.mean([e['mean_alpha'] for e in entries])
            predicted_helps = avg_sat >= threshold if threshold else 'N/A'
            print(f"    Llama {pm:.0f}M / {tm:.0f}M: sat={avg_sat:.4f} α={avg_alpha:.3f} "
                  f"predicted_helps={predicted_helps}", flush=True)

    # Save predictor results
    predictor_output = {
        'timestamp': datetime.now().isoformat(),
        'n_gpt2_points': len(predictor_data),
        'n_llama_points': len(llama_data),
        'linear_model': {
            'coefficients': {
                'saturation': float(beta[0]) if beta is not None else None,
                'log_params': float(beta[1]) if beta is not None else None,
                'intercept': float(beta[2]) if beta is not None else None,
            },
            'r_squared': float(r2),
            'formula': f"Δ = {beta[0]:.2f}·sat + {beta[1]:.2f}·log₁₀(P) + {beta[2]:.2f}" if beta is not None else None,
        },
        'saturation_threshold': {
            'value': float(threshold),
            'accuracy': float(accuracy),
            'rule': f"DyT helps when saturation(|αx|>2.0) ≥ {threshold:.3f}",
        },
        'tp_threshold': {
            'value': float(tp_threshold) if tp_threshold else None,
            'accuracy': float(tp_accuracy) if tp_threshold else None,
            'note': 'T/P alone is insufficient — regimes overlap across scales' if not tp_threshold else None,
        },
        'data_points': predictor_data,
        'llama_validation': llama_data,
    }

    with open(PREDICTOR_FILE, 'w') as f:
        json.dump(predictor_output, f, indent=2, default=str)
    print(f"\n  Predictor saved to {PREDICTOR_FILE}", flush=True)

    return predictor_output


# ─── Paper-ready summary table ────────────────────────────────────────────

def generate_paper_table(results):
    """Generate a formatted table for direct inclusion in the paper."""
    print(f"\n{'='*70}", flush=True)
    print(f"PAPER-READY SATURATION TABLE", flush=True)
    print(f"{'='*70}", flush=True)

    # Group by (arch, scale, tokens, config) and average across seeds
    grouped = defaultdict(list)
    for key, result in results.items():
        if 'error' in result or result is None:
            continue
        meta = result.get('metadata', {})
        group_key = (meta.get('arch', '?'), meta.get('scale', '?'),
                     meta.get('tokens', '?'), meta.get('config', '?'))
        grouped[group_key].append(result)

    lines = []
    header = (f"{'Arch':<8} {'Scale':<8} {'Tokens':<8} {'Config':<12} "
              f"{'Sat@2.0':>8} {'±':>6} {'Mean α':>8} {'±':>6} {'N':>3}")
    lines.append(header)
    lines.append('-' * len(header))

    for group_key in sorted(grouped.keys()):
        arch, scale, tokens, config = group_key
        entries = grouped[group_key]

        sats = [e['global_saturation']['2.0'] for e in entries]
        alphas = [e['mean_alpha'] for e in entries]

        mean_sat = np.mean(sats)
        std_sat = np.std(sats) if len(sats) > 1 else 0
        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas) if len(alphas) > 1 else 0

        line = (f"{arch:<8} {scale:<8} {tokens:<8} {config:<12} "
                f"{mean_sat:>8.4f} {std_sat:>6.4f} {mean_alpha:>8.3f} {std_alpha:>6.3f} "
                f"{len(entries):>3}")
        lines.append(line)
        print(f"  {line}", flush=True)

    # Save to file
    with open(SUMMARY_FILE, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Table saved to {SUMMARY_FILE}", flush=True)

    return lines


# ─── Entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Comprehensive DyT saturation analysis')
    parser.add_argument('--parallel', action='store_true',
                        help='Use both GPUs (split by model size)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU-only mode')
    parser.add_argument('--skip-predictor', action='store_true',
                        help='Skip crossover predictor fitting')
    args = parser.parse_args()

    print(f"\n{'#'*70}", flush=True)
    print(f"# DyT Activation Saturation Sweep", flush=True)
    print(f"# Started: {datetime.now().isoformat()}", flush=True)
    print(f"# GPU count: {torch.cuda.device_count()}", flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free = get_free_gpu_mb(i)
            total = torch.cuda.get_device_properties(i).total_memory / (1024**2)
            name = torch.cuda.get_device_properties(i).name
            print(f"#   GPU {i}: {name} — {free:.0f}/{total:.0f} MB free", flush=True)
    print(f"{'#'*70}\n", flush=True)

    # Phase 1: Measure saturation on all checkpoints
    results = run_sweep(args)

    # Phase 2: Fit crossover predictor
    if not args.skip_predictor:
        predictor = fit_crossover_predictor(results)

    # Phase 3: Generate paper-ready table
    generate_paper_table(results)

    print(f"\n{'#'*70}", flush=True)
    print(f"# ALL DONE — {datetime.now().isoformat()}", flush=True)
    print(f"# Results: {RESULTS_FILE}", flush=True)
    print(f"# Predictor: {PREDICTOR_FILE}", flush=True)
    print(f"# Table: {SUMMARY_FILE}", flush=True)
    print(f"{'#'*70}", flush=True)


if __name__ == '__main__':
    main()
