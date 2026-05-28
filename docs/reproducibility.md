# Reproducibility Notes

This repository is organized so that readers can verify the artifact before launching expensive experiments.

## Artifact Scope

The public artifact is designed for three levels of reuse:

1. inspect the exact code/config/result surface behind the paper;
2. run CPU-safe integrity checks and small model-construction smoke tests;
3. rerun selected training cells on appropriate GPU hardware.

It is not a one-command full reproduction of the complete training campaign.
Large checkpoints and raw datasets are intentionally excluded from Git.

## Validation First

Run:

```bash
make validate
make smoke-model
```

These commands do not train models. They check syntax, parse JSON artifacts,
validate shell launchers, and instantiate the main architectural variants.
`make validate` is dependency-light; `make smoke-model` requires PyTorch.

## Data

The text experiments use WikiText-103 subsets prepared by:

```bash
cd code
python prepare_wikitext.py
```

Equivalent shortcut from the repository root:

```bash
make data
```

The preparation script creates:

- `data/wikitext_1m/`
- `data/wikitext_10m/`
- `data/wikitext_50m/`
- `data/wikitext/`

Raw dataset files are not committed to Git.

## Training Runs

Launchers are in `code/`:

- `run_3seed.sh`: primary 64M 3-seed phase diagram.
- `run_scaling.sh`: multi-scale GPT-2-family experiments.
- `run_scale5.sh`: 3.78B Scale 5 stress tests.
- `run_alpha_sweep.sh`: DyT alpha sweep.
- `run_rmsnorm.sh`: RMSNorm baseline.
- `run_10k_convergence.sh`: extended convergence checks.

The full Scale 5 launcher requires high-memory GPU hardware and is not intended as a laptop command.

## Result Verification

The paper-level result metadata is in `results/full/`:

- `manifests/paper_sources.json`
- `manifests/sig_tests.json`
- `manifests/predictor_validation.json`
- `scale5_metadata_check.json`

These files are intended to make table values auditable without requiring reviewers to rerun every training job.

## Claims This Artifact Supports

- DyT behaves as a regime-dependent intervention in the studied
  compute-limited Transformer settings.
- The paper's reported aggregate table and figure values can be traced through
  shipped JSON manifests.
- The included launchers/configs define the experiment surface used for
  selected reruns.

## Claims This Artifact Does Not Support By Itself

- a universal recommendation to replace LayerNorm with DyT;
- Chinchilla-optimal scaling conclusions;
- proof that the same sign pattern holds outside the paper's token-to-parameter
  regimes;
- laptop-scale rerun of the full 3.78B stress-test campaign.
