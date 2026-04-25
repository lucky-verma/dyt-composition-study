#!/bin/bash
# Multi-scale experiments: Scale 2 (124M) and Scale 3 (354M)
# Same configs as Scale 1 but with larger models
# Estimated time: ~6 hours for Scale 2, ~12 hours for Scale 3 on H100

set -e
cd "$(dirname "$0")"
PYTHON=python

SEEDS="1337 42 7"
CONFIGS="vanilla dyt diffattn"
SCALES="wikitext_1m wikitext"

# Scale 2: 124M (GPT-2 architecture)
SCALE2_BASE="--n_layer=12 --n_head=12 --n_embd=768 --block_size=512 \
--batch_size=32 --gradient_accumulation_steps=2 \
--max_iters=5000 --eval_interval=500 --eval_iters=50 \
--learning_rate=3e-4 --compile=True --log_interval=500 \
--wandb_log=False --bias=False --dropout=0.0 --device=cuda --dtype=bfloat16"

# Scale 3: 354M (GPT-2 Medium)
SCALE3_BASE="--n_layer=24 --n_head=16 --n_embd=1024 --block_size=512 \
--batch_size=16 --gradient_accumulation_steps=4 \
--max_iters=5000 --eval_interval=500 --eval_iters=50 \
--learning_rate=3e-4 --compile=True --log_interval=500 \
--wandb_log=False --bias=False --dropout=0.0 --device=cuda --dtype=bfloat16"

# Scale 4: 1.3B (GPT-2 XL-like)
SCALE4_BASE="--n_layer=24 --n_head=32 --n_embd=2048 --block_size=512 \
--batch_size=4 --gradient_accumulation_steps=16 \
--max_iters=5000 --eval_interval=500 --eval_iters=50 \
--learning_rate=1e-4 --compile=True --log_interval=500 \
--wandb_log=False --bias=False --dropout=0.0 --dtype=bfloat16"

run_scale() {
    local SCALE_NAME=$1
    local BASE_ARGS=$2

    echo "=== ${SCALE_NAME} EXPERIMENTS STARTED $(date) ==="
    mkdir -p out/${SCALE_NAME}

    for scale in $SCALES; do
        for config in $CONFIGS; do
            case $config in
                vanilla)  FLAGS="--use_dyt=False --use_diff_attn=False" ;;
                dyt)      FLAGS="--use_dyt=True --use_diff_attn=False" ;;
                diffattn) FLAGS="--use_dyt=False --use_diff_attn=True" ;;
            esac
            for seed in $SEEDS; do
                OUTDIR="out/${SCALE_NAME}/${scale}_${config}_s${seed}"
                echo "  ${scale} / ${config} / seed=${seed}"
                $PYTHON -u train.py --dataset=$scale $BASE_ARGS $FLAGS \
                    --seed=$seed --out_dir=$OUTDIR --device=cuda 2>&1 | \
                    grep -E "step 5000|val loss" | tee -a out/${SCALE_NAME}/summary.log
            done
        done
    done
    echo "=== ${SCALE_NAME} COMPLETE $(date) ==="
}

# Uncomment the scale you want to run:
# run_scale "scale2" "$SCALE2_BASE"
# run_scale "scale3" "$SCALE3_BASE"
# run_scale "scale4" "$SCALE4_BASE"

echo "Edit this script to uncomment the desired scale, then run."
