"""
Extract train/val gap data from all experiment logs.
No GPU needed -- just parsing text files.
Outputs data for the train/val gap figure.
"""
import json, os, re

LOG_FILES = {
    'scale1_3seed': 'out/scale1_3seed_master.log',
    'scale2': 'out/queue_manager.log',
    'intermediate': 'out/intermediate_log.log',
    'alpha_sweep': 'out/alpha_sweep_log.log',
}

# Parse step-by-step training curves from logs
results = {}

for group, logfile in LOG_FILES.items():
    if not os.path.exists(logfile):
        print(f"SKIP: {logfile}")
        continue

    with open(logfile) as f:
        text = f.read()

    # Find experiment blocks: === [N/M] dataset / config / seed=S ===
    blocks = re.split(r'=== \[(\d+)/\d+\] (.+?) ===', text)

    for i in range(1, len(blocks)-1, 3):
        run_num = blocks[i]
        run_desc = blocks[i+1].strip()
        run_log = blocks[i+2]

        # Parse step data
        steps = re.findall(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', run_log)
        if steps:
            key = f"{group}/{run_desc}"
            results[key] = {
                'steps': [int(s[0]) for s in steps],
                'train_loss': [float(s[1]) for s in steps],
                'val_loss': [float(s[2]) for s in steps],
                'final_train': float(steps[-1][1]),
                'final_val': float(steps[-1][2]),
                'train_val_gap': float(steps[-1][2]) - float(steps[-1][1]),
            }

# Summary: train/val gaps for key conditions
print("="*70)
print("TRAIN/VAL GAP ANALYSIS")
print("="*70)
print(f"{'Condition':<50} {'Train':>8} {'Val':>8} {'Gap':>8}")
print("-"*76)

for key in sorted(results.keys()):
    r = results[key]
    print(f"{key:<50} {r['final_train']:>8.4f} {r['final_val']:>8.4f} {r['train_val_gap']:>8.4f}")

# Save for plotting
os.makedirs('out/train_val_gap', exist_ok=True)
with open('out/train_val_gap/data.json', 'w') as f:
    json.dump(results, f, indent=2)

# Key comparisons for the figure
print("\n" + "="*70)
print("KEY COMPARISONS FOR FIGURE")
print("="*70)

key_pairs = [
    ('85M 1M vanilla', 'scale1_3seed/wikitext_1m / vanilla / seed=1337'),
    ('85M 1M DyT', 'scale1_3seed/wikitext_1m / dyt / seed=1337'),
    ('85M 118M vanilla', 'scale1_3seed/wikitext / vanilla / seed=1337'),
    ('85M 118M DyT', 'scale1_3seed/wikitext / dyt / seed=1337'),
]

for label, key in key_pairs:
    if key in results:
        r = results[key]
        print(f"  {label}: train={r['final_train']:.4f}, val={r['final_val']:.4f}, gap={r['train_val_gap']:.4f}")

print("\n=== EXTRACTION COMPLETE ===")
