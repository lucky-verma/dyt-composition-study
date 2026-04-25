"""
Activation Saturation Analysis for DyT Paper
Measures tanh saturation fraction across checkpoints to prove implicit regularization mechanism.

OOM Safety:
- Loads ONE checkpoint at a time
- Uses batch_size=1 for forward pass
- Clears GPU between each checkpoint
- Max memory: ~2GB for 354M model (96GB H100 = safe)
"""
import torch
import json
import os
import sys
import gc
import numpy as np
from pathlib import Path

from model import GPTConfig, GPT
import tiktoken

DEVICE = 'cuda:0'
DTYPE = torch.bfloat16
enc = tiktoken.get_encoding('gpt2')

def load_model(ckpt_dir):
    """Load model with OOM safety -- CPU first, then GPU."""
    ckpt = torch.load(f'{ckpt_dir}/ckpt.pt', map_location='cpu', weights_only=False)
    cfg = GPTConfig(**ckpt['model_args'])
    model = GPT(cfg)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval()
    model.to(DEVICE, dtype=DTYPE)
    return model, ckpt.get('iter_num', 0), cfg

def analyze_saturation(model, data_path, n_batches=50, seq_len=512):
    """
    Measure tanh saturation in DyT layers.
    Returns per-layer saturation fractions at different thresholds.
    """
    # Load a small chunk of data for forward passes
    data = np.memmap(f'{data_path}/train.bin', dtype=np.uint16, mode='r')

    saturation_stats = {}
    hooks = []
    activations = {}

    # Register hooks on DyT layers to capture pre-tanh activations
    for name, module in model.named_modules():
        if hasattr(module, 'alpha'):  # DyT layers have learnable alpha
            layer_name = name
            def hook_fn(mod, inp, out, name=layer_name):
                # Capture alpha * input (pre-tanh value)
                if len(inp) > 0:
                    with torch.no_grad():
                        pre_tanh = mod.alpha * inp[0]
                        activations[name] = pre_tanh.detach()
            hooks.append(module.register_forward_hook(hook_fn))

    if not hooks:
        print("  No DyT layers found (vanilla model). Skipping.", flush=True)
        return None

    # Collect saturation stats over multiple batches
    all_saturations = {t: [] for t in [1.0, 1.5, 2.0, 2.5, 3.0]}
    alpha_values = {}

    for batch_idx in range(n_batches):
        # Random position in data
        start = np.random.randint(0, len(data) - seq_len - 1)
        x = torch.from_numpy(data[start:start+seq_len].astype(np.int64)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            model(x)

        # Compute saturation at each threshold
        for layer_name, pre_tanh in activations.items():
            abs_val = pre_tanh.abs().float()
            for threshold in all_saturations:
                frac = (abs_val > threshold).float().mean().item()
                all_saturations[threshold].append(frac)

            # Record alpha values
            if layer_name not in alpha_values:
                for name, module in model.named_modules():
                    if name == layer_name and hasattr(module, 'alpha'):
                        alpha_values[layer_name] = module.alpha.item()

        activations.clear()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Compute mean saturation per threshold
    result = {
        'saturation': {str(t): np.mean(v) for t, v in all_saturations.items()},
        'alpha_values': alpha_values,
        'n_dyt_layers': len(alpha_values),
    }

    return result

# --- Checkpoints to analyze ---
# Update these paths to match your checkpoint locations
CHECKPOINTS = {
    # Scale 1 (85M) -- core comparison
    '85M_1M_vanilla':   ('out/scale1_3seed/wikitext_1m_vanilla_s1337', 'data/wikitext_1m'),
    '85M_1M_dyt':       ('out/scale1_3seed/wikitext_1m_dyt_s1337', 'data/wikitext_1m'),
    '85M_118M_vanilla': ('out/scale1_3seed/wikitext_vanilla_s1337', 'data/wikitext'),
    '85M_118M_dyt':     ('out/scale1_3seed/wikitext_dyt_s1337', 'data/wikitext'),
    # Scale 2 (124M)
    '124M_1M_dyt':      ('out/scale2/wikitext_1m_dyt_s1337', 'data/wikitext_1m'),
    '124M_118M_dyt':    ('out/scale2/wikitext_dyt_s1337', 'data/wikitext'),
    # Scale 3 (354M)
    '354M_1M_dyt':      ('out/scale3/wikitext_1m_dyt_s1337', 'data/wikitext_1m'),
    '354M_118M_dyt':    ('out/scale3/wikitext_dyt_s1337', 'data/wikitext'),
}

results = {}
os.makedirs('out/activation_analysis', exist_ok=True)

for name, (ckpt_path, data_path) in CHECKPOINTS.items():
    print(f'\n[{name}]', flush=True)

    if not os.path.exists(f'{ckpt_path}/ckpt.pt'):
        print(f'  SKIP: checkpoint not found', flush=True)
        continue

    try:
        model, step, cfg = load_model(ckpt_path)
        mem = torch.cuda.memory_allocated() / 1e9
        print(f'  Loaded. GPU: {mem:.2f}GB. Step: {step}', flush=True)

        sat = analyze_saturation(model, data_path, n_batches=50)
        if sat:
            results[name] = sat
            print(f'  Saturation (|ax|>2.0): {sat["saturation"]["2.0"]:.4f}', flush=True)
            print(f'  Saturation (|ax|>3.0): {sat["saturation"]["3.0"]:.4f}', flush=True)
            print(f'  Alpha values: {list(sat["alpha_values"].values())[:3]}...', flush=True)

    except Exception as e:
        print(f'  ERROR: {e}', flush=True)
        import traceback; traceback.print_exc()
        results[name] = {'error': str(e)}
    finally:
        try: del model
        except: pass
        gc.collect()
        torch.cuda.empty_cache()
        print(f'  Cleared.', flush=True)

# Save results
with open('out/activation_analysis/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary table
print('\n' + '='*70)
print('SATURATION SUMMARY (fraction of |ax| > threshold)')
print('='*70)
print(f'{"Checkpoint":<25} {"a>1.0":>8} {"a>2.0":>8} {"a>3.0":>8} {"Mean a":>8}')
print('-'*60)
for name, data in results.items():
    if 'error' in data or data is None:
        continue
    s = data['saturation']
    alphas = list(data['alpha_values'].values())
    mean_alpha = np.mean(alphas) if alphas else 0
    print(f'{name:<25} {s["1.0"]:>8.4f} {s["2.0"]:>8.4f} {s["3.0"]:>8.4f} {mean_alpha:>8.4f}')

print('\n=== ANALYSIS COMPLETE ===')
