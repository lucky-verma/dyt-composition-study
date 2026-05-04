# When Does Removing LayerNorm Help?

[arXiv:2604.23434](https://arxiv.org/abs/2604.23434) | [Citation](#citation) | [MIT License](LICENSE)

Code and artifacts for the empirical study:

**When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer**

This repository studies when Dynamic Tanh (DyT) helps or hurts Transformer training, with controlled GPT-2-family scaling experiments, Llama-style checks, ViT checks, ablations, downstream evaluations, and machine-readable result manifests.

## What Is Included

- `code/`: GPT-2-family, Llama-style, and ViT experiment code.
- `configs/`: scale-specific model/training configurations, including the 3.78B Scale 5 stress-test config.
- `analysis/`: saturation measurement, significance tests, predictor validation, LAMBADA/BLIMP/cross-domain evaluation utilities, and table checks.
- `results/`: aggregate JSON result files and provenance manifests used to generate the paper tables and figures.
- `docs/`: artifact notes, reproducibility instructions, and release checklist.
- `.github/workflows/validate.yml`: lightweight validation for Python syntax, JSON validity, shell launchers, and model smoke tests.

Large raw datasets and model checkpoints are not stored directly in Git. The code uses public datasets, and checkpoint/data mirrors may be linked separately when available.

## Headline Scope

The paper is an empirical regime study, not a theory paper. It asks whether replacing LayerNorm with activation bounding behaves like an implicit regularizer whose sign depends on the token-to-parameter regime.

The artifact supports:

- 5 GPT-2-family model scales from 64M to 3.78B parameters;
- 1M and 118M token regimes, with intermediate-data experiments where used in the paper;
- DyT, LayerNorm, RMSNorm, HardTanh, DiffAttn V1, the V2-inspired sigmoid-lambda ablation, and gated-attention controls;
- activation saturation measurements;
- statistical tests and calibration-heuristic validation;
- LAMBADA, BLIMP, OpenWebText, Llama-style, and ViT checks.

## Quick Start

```bash
git clone https://github.com/lucky-verma/dyt-composition-study.git
cd dyt-composition-study

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/validate_repo.py
python scripts/smoke_model.py
```

Prepare WikiText-103 data:

```bash
cd code
python prepare_wikitext.py
```

Run a small single-GPU smoke experiment:

```bash
cd code
python train.py --dataset=wikitext_1m \
  --n_layer=2 --n_head=2 --n_embd=128 --block_size=128 \
  --batch_size=4 --gradient_accumulation_steps=16 \
  --max_iters=20 --eval_interval=10 --eval_iters=2 \
  --learning_rate=3e-4 --compile=False --device=cpu \
  --wandb_log=False --out_dir=out/smoke
```

Run the full experiment launchers only on appropriate GPU hardware:

```bash
cd code
bash run_3seed.sh
bash run_scaling.sh
bash run_scale5.sh   # 3.78B Scale 5; requires high-memory GPU hardware
```

## Result Files

The main machine-readable artifacts are under `results/full/`:

- `all_results.json`: aggregate training losses and per-run metadata.
- `fast_extract.json`: compact table-source extract.
- `saturation_results.json`: activation saturation measurements.
- `scale5_metadata_check.json`: config verification summary for the Scale 5 stress-test runs.
- `manifests/paper_sources.json`: table/source manifest for paper claims.
- `manifests/sig_tests.json`: paired tests and Bonferroni corrections.
- `manifests/predictor_validation.json`: calibration-heuristic validation results.

See `docs/reproducibility.md` and `docs/artifact_manifest.md` for details.

## Validation

The repo-level validator checks:

- Python syntax for all `.py` files;
- JSON parseability for all `.json` files;
- shell syntax for all `.sh` launchers;
- absence of internal path/ops fingerprints;
- model construction for the main architectural variants.

```bash
python scripts/validate_repo.py
python scripts/smoke_model.py
```

## Citation

If you use this code, result manifest, or experiment setup, please cite the associated arXiv preprint:

```bibtex
@misc{verma2026dytcomposition,
  title        = {When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer},
  author       = {Verma, Lucky},
  year         = {2026},
  publisher    = {arXiv},
  doi          = {10.48550/arXiv.2604.23434},
  url          = {https://arxiv.org/abs/2604.23434},
  eprint       = {2604.23434},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

GitHub also reads the repository-level `CITATION.cff` file. A plain BibTeX copy is available in `CITATION.bib` for citation managers that prefer `.bib` files.

## License

Code is released under the MIT License. Public datasets used by the experiments retain their original licenses. Large checkpoints, if mirrored separately, may include additional metadata and terms.
