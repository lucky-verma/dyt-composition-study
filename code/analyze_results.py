"""
Analyze normalization-removal study results and generate figures.

Usage:
    python analyze_results.py --dataset shakespeare_char --scale small
"""

import argparse
import json
import os
from pathlib import Path


def load_results(results_dir):
    """Load all result.json files from experiment directories."""
    results = []
    for config_dir in sorted(Path(results_dir).iterdir()):
        result_file = config_dir / "result.json"
        if result_file.exists():
            with open(result_file) as f:
                results.append(json.load(f))
    return results


def print_summary_table(results):
    """Print a summary table of all results."""
    print(f"\n{'='*80}")
    print(f"{'Config':<30} {'Status':<10} {'Val Loss':<12} {'Params (M)':<12} {'Time (s)':<10}")
    print(f"{'='*80}")

    # Sort by val_loss (best first)
    sorted_results = sorted(results, key=lambda r: r.get('final_val_loss', 999))

    for r in sorted_results:
        config = r['config']
        status = r['status']
        val_loss = f"{r['final_val_loss']:.4f}" if 'final_val_loss' in r else 'N/A'
        elapsed = f"{r.get('elapsed_seconds', 0):.0f}"

        # Extract param count from stdout
        params = 'N/A'
        if 'stdout_tail' in r:
            for line in r['stdout_tail'].split('\n'):
                if 'number of parameters' in line:
                    try:
                        params = line.split(':')[1].strip().replace('M', '')
                    except:
                        pass

        print(f"{config:<30} {status:<10} {val_loss:<12} {params:<12} {elapsed:<10}")

    print(f"{'='*80}")

    # Interaction analysis for mechanisms reported in the paper.
    if len(results) >= 4:
        print("\n=== INTERACTION ANALYSIS ===")
        by_config = {r['config']: r.get('final_val_loss', 999) for r in results}

        vanilla = by_config.get('vanilla', None)
        if vanilla:
            print(f"\nVanilla baseline: {vanilla:.4f}")
            print(f"\nIndividual improvements over vanilla:")
            for config in ['dyt', 'diffattn', 'hardtanh', 'rmsnorm']:
                if config in by_config:
                    delta = by_config[config] - vanilla
                    direction = "better" if delta < 0 else "worse"
                    print(f"  {config:<20} {by_config[config]:.4f} ({delta:+.4f}, {direction})")

            print(f"\nReported mechanism pairs:")
            pairs = ['dyt+diffattn']
            for pair in pairs:
                if pair in by_config:
                    parts = pair.split('+')
                    expected = sum(by_config.get(p, 0) - vanilla for p in parts) + vanilla
                    actual = by_config[pair]
                    interaction = actual - expected
                    comp_type = "super-additive" if interaction < -0.01 else "sub-additive" if interaction > 0.01 else "additive"
                    print(f"  {pair:<25} actual={actual:.4f} expected={expected:.4f} interaction={interaction:+.4f} ({comp_type})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--scale", default="small")
    args = parser.parse_args()

    results_dir = f"out/{args.dataset}"
    if not os.path.exists(results_dir):
        print(f"No results directory: {results_dir}")
        return

    results = load_results(results_dir)
    if not results:
        print("No results found yet.")
        return

    print(f"\nLoaded {len(results)} results from {results_dir}")
    print_summary_table(results)


if __name__ == "__main__":
    main()
