# Artifact Manifest

## Source Code

- `code/model.py`: GPT-2-family model with DyT, RMSNorm, HardTanh, DiffAttn V1/V2, and gated-attention toggles.
- `code/model_llama.py`: Llama-style architecture checks.
- `code/train.py`: training loop and metadata writer.
- `code/vit_experiment.py`, `code/vit_alpha_sweep.py`: vision checks.

## Configurations

- `configs/scale1_64M.py`
- `configs/scale2_124M.py`
- `configs/scale3_354M.py`
- `configs/scale4_1300M.py`
- `configs/scale5_3780M.py`

## Analysis

- `analysis/sig_tests.py`: paired tests and Bonferroni correction.
- `analysis/predictor_validation.py`: calibration heuristic validation.
- `analysis/saturation_sweep.py`: activation saturation measurement.
- `analysis/lambada_eval.py`: LAMBADA evaluation utility.
- `analysis/blimp_eval.py`: BLIMP evaluation utility.
- `analysis/cross_domain_eval.py`: OpenWebText cross-domain evaluation.

## Results

- `results/full/all_results.json`: aggregate training metrics.
- `results/full/saturation_results.json`: saturation measurements.
- `results/full/lambada_eval/`: LAMBADA result JSONs.
- `results/full/downstream_v5/`: downstream evaluation summaries.
- `results/full/manifests/`: paper-source, citation, coverage, significance, and predictor-validation manifests.

