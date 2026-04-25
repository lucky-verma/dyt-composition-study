#!/bin/bash
# 3-Seed Confidence Interval Experiments (H100)
# 18 runs: 2 scales x 3 configs x 3 seeds
# Estimated time: ~2 hours on H100

set -e
cd "$(dirname "$0")"

PYTHON=python
SEEDS="1337 42 7"
CONFIGS="vanilla dyt diffattn"
SCALES="wikitext_1m wikitext"

# H100-optimized settings (larger batch, bf16, compile=True)
BASE="--n_layer=12 --n_head=8 --n_embd=512 --block_size=512 \
--batch_size=64 --gradient_accumulation_steps=1 \
--max_iters=5000 --eval_interval=500 --eval_iters=50 \
--learning_rate=3e-4 --compile=True \
--log_interval=500 --wandb_log=False \
--bias=False --dropout=0.0 --device=cuda --dtype=bfloat16"

mkdir -p out/3seed

echo "=== 3-SEED EXPERIMENTS STARTED $(date) ==="

TOTAL=18
RUN=0

for scale in $SCALES; do
    for config in $CONFIGS; do
        case $config in
            vanilla)  FLAGS="--use_dyt=False --use_diff_attn=False" ;;
            dyt)      FLAGS="--use_dyt=True --use_diff_attn=False" ;;
            diffattn) FLAGS="--use_dyt=False --use_diff_attn=True" ;;
        esac

        for seed in $SEEDS; do
            RUN=$((RUN + 1))
            OUTDIR="out/3seed/${scale}_${config}_s${seed}"
            echo ""
            echo "=== [$RUN/$TOTAL] ${scale} / ${config} / seed=${seed} === $(date)"

            $PYTHON -u train.py \
                --dataset=$scale \
                $BASE $FLAGS \
                --seed=$seed \
                --out_dir=$OUTDIR 2>&1 | \
                grep -E "step |val loss|Config" | tee -a out/3seed/summary.log

            echo "  Done: $OUTDIR"
        done
    done
done

echo ""
echo "=== ALL DONE $(date) ==="
