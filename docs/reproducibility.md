# Reproducibility Notes

This repository is organized so that readers can verify the artifact before launching expensive experiments.

## Validation First

Run:

```bash
python scripts/validate_repo.py
python scripts/smoke_model.py
```

These commands do not train models. They check syntax, parse JSON artifacts, validate shell launchers, and instantiate the main architectural variants.

## Data

The text experiments use WikiText-103 subsets prepared by:

```bash
cd code
python prepare_wikitext.py
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

