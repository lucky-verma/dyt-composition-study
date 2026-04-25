#!/bin/bash
# Alpha_init sweep at real scale (85M) -- validates toy-model finding
# 10 runs: 2 vanilla baselines + 4 alpha values x 2 seeds each

cd "$(dirname "$0")"
PYTHON=python
LOG=out/alpha_sweep_log.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG; }

BASE="--n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
--batch_size=64 --gradient_accumulation_steps=1 \
--max_iters=5000 --eval_interval=500 --eval_iters=50 \
--learning_rate=3e-4 --compile=True --log_interval=500 \
--wandb_log=False --bias=False --dropout=0.0 \
--device=cuda --dtype=bfloat16 \
--use_dyt=True --use_diff_attn=False --dataset=wikitext_1m"

mkdir -p out/alpha_sweep
TOTAL=10; RUN=0

log "=== ALPHA SWEEP STARTED ==="
# Vanilla baseline (2 seeds)
for seed in 1337 42; do
    RUN=$((RUN+1))
    log "[$RUN/$TOTAL] vanilla / seed=$seed"
    $PYTHON -u train.py --dataset=wikitext_1m \
        --n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
        --batch_size=64 --gradient_accumulation_steps=1 \
        --max_iters=5000 --eval_interval=500 --eval_iters=50 \
        --learning_rate=3e-4 --compile=True --log_interval=500 \
        --wandb_log=False --bias=False --dropout=0.0 \
        --device=cuda --dtype=bfloat16 \
        --use_dyt=False --use_diff_attn=False \
        --seed=$seed --out_dir=out/alpha_sweep/vanilla_s${seed} 2>&1 | \
        grep -E "step 5000|val loss" | tee -a $LOG
    log "  Done: vanilla s$seed"
done

# DyT with alpha_init values (2 seeds each)
for alpha in 0.5 1.0 2.0 3.0; do
    for seed in 1337 42; do
        RUN=$((RUN+1))
        log "[$RUN/$TOTAL] DyT alpha=$alpha / seed=$seed"
        $PYTHON -u train.py $BASE \
            --dyt_alpha_init=$alpha \
            --seed=$seed --out_dir=out/alpha_sweep/dyt_a${alpha}_s${seed} 2>&1 | \
            grep -E "step 5000|val loss" | tee -a $LOG
        log "  Done: alpha=$alpha s$seed"
    done
done

log "=== ALPHA SWEEP COMPLETE ==="
