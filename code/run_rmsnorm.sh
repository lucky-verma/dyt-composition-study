#!/bin/bash
# RMSNorm baseline -- answers "does DyT's advantage hold against modern normalization?"
# 4 runs: 1M + 118M x RMSNorm x 2 seeds (compare against existing vanilla/DyT data)

cd "$(dirname "$0")"
PYTHON=python
LOG=out/rmsnorm_log.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG; }
mkdir -p out/rmsnorm

BASE="--n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
--batch_size=64 --gradient_accumulation_steps=1 \
--max_iters=5000 --eval_interval=500 --eval_iters=50 \
--learning_rate=3e-4 --compile=True --log_interval=500 \
--wandb_log=False --bias=False --dropout=0.0 --device=cuda --dtype=bfloat16 \
--use_rmsnorm=True --use_dyt=False --use_diff_attn=False"

TOTAL=4; RUN=0
log "=== RMSNORM BASELINE STARTED ==="

for dataset in wikitext_1m wikitext; do
    for seed in 1337 42; do
        RUN=$((RUN+1))
        OUTDIR="out/rmsnorm/${dataset}_rmsnorm_s${seed}"
        log "[$RUN/$TOTAL] $dataset / RMSNorm / seed=$seed"
        $PYTHON -u train.py --dataset=$dataset $BASE \
            --seed=$seed --out_dir=$OUTDIR 2>&1 | \
            grep -E "step 5000|val loss|Config|parameters" | tee -a $LOG
        log "  Done: $OUTDIR"
    done
done

log "=== RMSNORM BASELINE COMPLETE ==="
