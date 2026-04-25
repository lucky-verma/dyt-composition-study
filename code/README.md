# When Does Removing LayerNorm Help? Activation Bounding as a Regime-Dependent Implicit Regularizer

Training code and experiment scripts for the normalization-removal study. Built on [nanoGPT](https://github.com/karpathy/nanoGPT) with toggle flags for the mechanisms evaluated in the paper.

## Setup

### Dependencies

```bash
pip install torch numpy tiktoken datasets torchvision
```

- Python 3.10+
- PyTorch 2.0+ (required for `torch.compile` and Flash Attention)
- CUDA 12.x recommended

### Data Preparation

```bash
python prepare_wikitext.py
```

This downloads WikiText-103 (~118M tokens) and creates subsets:
- `data/wikitext/` -- full 118M tokens
- `data/wikitext_1m/` -- 1M token subset (overfitting regime)
- `data/wikitext_10m/` -- 10M token subset
- `data/wikitext_50m/` -- 50M token subset

## Architecture

The key file is `model.py`, which implements a unified GPT with toggle flags:

| Flag | Modification | Reference |
|------|-------------|-----------|
| `--use_dyt=True` | Dynamic Tanh (replaces LayerNorm) | Zhu et al., CVPR 2025 |
| `--use_rmsnorm=True` | RMSNorm (modern baseline) | Llama, Mistral |
| `--use_diff_attn=True` | Differential Attention | Ye et al., ICLR 2025 |
| `--diff_attn_v2=True` | V2-inspired sigmoid-lambda ablation | This paper; legacy flag name |
| `--use_hardtanh=True` | Hard clipping control for activation bounding | This paper |
| `--dyt_alpha_init=X` | DyT alpha initialization | Default: 2.0 |

Flags are independent and composable. Any combination can be enabled.

The `diff_attn_v2` flag name is retained for historical run reproducibility. It
tests sigmoid-bounding of V1's lambda terms inside the V1-style DiffAttn
architecture; it is not a faithful implementation of Microsoft DIFF V2.

## Quick Start

### Single Experiment

```bash
# Vanilla baseline, 64M model, 1M tokens (overfitting regime)
python train.py --dataset=wikitext_1m \
  --n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
  --batch_size=64 --max_iters=5000 --eval_interval=500 \
  --learning_rate=3e-4 --compile=True --bias=False \
  --use_dyt=False --use_diff_attn=False \
  --seed=1337 --out_dir=out/vanilla_1m

# DyT only
python train.py --dataset=wikitext_1m \
  --n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
  --batch_size=64 --max_iters=5000 --eval_interval=500 \
  --learning_rate=3e-4 --compile=True --bias=False \
  --use_dyt=True --use_diff_attn=False --dyt_alpha_init=2.0 \
  --seed=1337 --out_dir=out/dyt_1m

# DiffAttn only
python train.py --dataset=wikitext_1m \
  --n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
  --batch_size=64 --max_iters=5000 --eval_interval=500 \
  --learning_rate=3e-4 --compile=True --bias=False \
  --use_dyt=False --use_diff_attn=True \
  --seed=1337 --out_dir=out/diffattn_1m

# DyT + DiffAttn
python train.py --dataset=wikitext \
  --n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
  --batch_size=64 --max_iters=10000 --eval_interval=500 \
  --learning_rate=3e-4 --compile=True --bias=False \
  --use_dyt=True --use_diff_attn=True \
  --seed=1337 --out_dir=out/dyt_diffattn_118m
```

### Full Experiment Suites

```bash
# Phase diagram: 3 seeds x 3 configs x 4 data scales
bash run_3seed.sh

# Alpha sensitivity sweep
bash run_alpha_sweep.sh

# RMSNorm baseline comparison
bash run_rmsnorm.sh

# 10K convergence comparison
bash run_10k_convergence.sh

# Multi-scale experiments (edit script to select scale)
bash run_scaling.sh

# Scale 5 stress tests (3.78B; requires 80GB+ GPU)
bash run_scale5.sh

# ViT experiments (CIFAR-10/100)
python vit_experiment.py

# ViT alpha sweep
python vit_alpha_sweep.py

# Activation saturation analysis (requires checkpoints)
python activation_analysis.py
```

## Model Scales

| Scale | Layers | Heads | d_model | Params | Batch | GPU Memory |
|-------|--------|-------|---------|--------|-------|------------|
| 1 | 12 | 8 | 512 | 64M | 64 | ~4 GB |
| 2 | 12 | 12 | 768 | 124M | 32 | ~8 GB |
| 3 | 24 | 16 | 1024 | 354M | 16 | ~24 GB |
| 4 | 24 | 32 | 2048 | 1.3B | 4 | ~80 GB |
| 5 | 32 | 32 | 3072 | 3.78B | 1 | ~80-96 GB |

## Hardware Requirements

- **Scale 1-2**: Any GPU with 8+ GB (T4, RTX 3090, etc.)
- **Scale 3**: GPU with 24+ GB (A100 40GB, L4, etc.)
- **Scale 4**: GPU with 80+ GB (H100, A100 80GB)
- **Scale 5**: GPU with 80+ GB; use `batch_size=1` and `gradient_accumulation_steps=64`
- All experiments use bfloat16 mixed precision

## File Overview

| File | Description |
|------|-------------|
| `train.py` | Main training loop (nanoGPT-based) |
| `model.py` | Unified GPT with modification toggles |
| `model_vanilla.py` | Vanilla nanoGPT baseline (no modifications) |
| `configurator.py` | CLI argument parsing |
| `prepare_wikitext.py` | Data preparation (WikiText-103 + subsets) |
| `vit_experiment.py` | ViT experiments on CIFAR-10/100 |
| `vit_alpha_sweep.py` | DyT alpha sensitivity on ViT |
| `activation_analysis.py` | Tanh saturation measurement |
| `analyze_results.py` | Results analysis utilities |
| `extract_train_val_gap.py` | Train/val gap extraction from logs |
| `run_3seed.sh` | 3-seed phase diagram experiments |
| `run_alpha_sweep.sh` | Alpha initialization sweep |
| `run_rmsnorm.sh` | RMSNorm baseline |
| `run_10k_convergence.sh` | 10K-step convergence comparison |
| `run_scaling.sh` | Multi-scale experiments |
| `run_scale5.sh` | Scale 5 stress-test launcher |

## Reproducing Paper Results

1. Prepare data: `python prepare_wikitext.py`
2. Run phase diagram (Table 1): `bash run_3seed.sh`
3. Run scaling (Table 2): Edit and run `bash run_scaling.sh` for Scales 2--4; run `bash run_scale5.sh` for Scale 5
4. Run alpha sweep (Figure 3): `bash run_alpha_sweep.sh`
5. Run 10K convergence (Table 3): `bash run_10k_convergence.sh`
6. Run ViT experiments (Table 4): `python vit_experiment.py && python vit_alpha_sweep.py`
7. Run activation analysis (Figure 4): `python activation_analysis.py` (requires step 2-3 checkpoints)

Total compute for all experiments: approximately 300 GPU-hours.
