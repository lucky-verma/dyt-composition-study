#!/bin/bash
# 10K convergence runs -- vanilla + DyT+DiffAttn at 118M for 10K steps
# Shows convergence behavior and crossover point

cd "$(dirname "$0")"
PYTHON=python
LOG=out/convergence_10k_log.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG; }
mkdir -p out/convergence_10k

BASE="--n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
--batch_size=64 --gradient_accumulation_steps=1 \
--max_iters=10000 --eval_interval=500 --eval_iters=50 \
--learning_rate=3e-4 --compile=True --log_interval=500 \
--wandb_log=False --bias=False --dropout=0.0 --device=cuda --dtype=bfloat16 \
--dataset=wikitext --seed=1337"

log "=== 10K CONVERGENCE STARTED ==="

# Vanilla 10K
log "[1/2] vanilla 10K steps"
CUDA_VISIBLE_DEVICES=0 $PYTHON -u train.py $BASE \
  --use_dyt=False --use_diff_attn=False \
  --out_dir=out/convergence_10k/vanilla_10k 2>&1 | \
  grep -E "step |val loss|Config|parameters" | tee -a $LOG
log "  Done: vanilla 10K"

# DyT+DiffAttn 10K
log "[2/2] DyT+DiffAttn 10K steps"
CUDA_VISIBLE_DEVICES=0 $PYTHON -u train.py $BASE \
  --use_dyt=True --use_diff_attn=True \
  --out_dir=out/convergence_10k/dyt_diffattn_10k 2>&1 | \
  grep -E "step |val loss|Config|parameters" | tee -a $LOG
log "  Done: DyT+DiffAttn 10K"

log "=== 10K CONVERGENCE COMPLETE ==="
