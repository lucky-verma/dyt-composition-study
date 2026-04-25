#!/usr/bin/env bash
# Scale 5 stress-test launcher used for the 3.78B GPT-2-family cells.
#
# Runs:
#   vanilla, DyT, DiffAttn V1 on {1M, 118M} x 3 seeds
#   V2-inspired sigmoid-lambda ablation on {1M, 118M} x 3 seeds
#
# Hardware note: this configuration is intended for 80GB+ GPUs. Each run uses
# micro-batch 1 with gradient accumulation 64 to keep eff_batch=64.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-1337 42 7}"
REGIMES="${REGIMES:-1m 118m}"
ARCHS="${ARCHS:-vanilla dyt diffattn_v1 diffattn_v2}"

SCALE5_ARGS="--n_layer=32 --n_head=32 --n_embd=3072 --block_size=512 \
--batch_size=1 --gradient_accumulation_steps=64 \
--learning_rate=1e-4 --max_iters=5000 --eval_interval=500 --eval_iters=50 \
--compile=True --dtype=bfloat16 --bias=False --dropout=0.0 \
--device=cuda --log_interval=100 --wandb_log=False"

dataset_for_regime() {
    case "$1" in
        1m) echo "wikitext_1m" ;;
        118m) echo "wikitext" ;;
        *) echo "unknown regime: $1" >&2; return 1 ;;
    esac
}

flags_for_arch() {
    case "$1" in
        vanilla) echo "--use_dyt=False --use_diff_attn=False --diff_attn_v2=False" ;;
        dyt) echo "--use_dyt=True --use_diff_attn=False --diff_attn_v2=False --dyt_alpha_init=2.0" ;;
        diffattn_v1) echo "--use_dyt=False --use_diff_attn=True --diff_attn_v2=False" ;;
        diffattn_v2) echo "--use_dyt=False --use_diff_attn=True --diff_attn_v2=True" ;;
        *) echo "unknown arch: $1" >&2; return 1 ;;
    esac
}

out_dir_for_run() {
    local regime="$1"
    local dataset="$2"
    local arch="$3"
    local seed="$4"
    case "$arch" in
        vanilla) echo "out/scale5/${dataset}_vanilla_s${seed}" ;;
        dyt) echo "out/scale5/${dataset}_dyt_s${seed}" ;;
        diffattn_v1) echo "out/diff_attn_v1v2_scale5/scale5_${regime}_diffattn_v1_s${seed}" ;;
        diffattn_v2) echo "out/diff_attn_v1v2_scale5/scale5_${regime}_diffattn_v2_s${seed}" ;;
        *) echo "unknown arch: $arch" >&2; return 1 ;;
    esac
}

echo "=== Scale 5 stress tests started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

for regime in $REGIMES; do
    dataset="$(dataset_for_regime "$regime")"
    for arch in $ARCHS; do
        flags="$(flags_for_arch "$arch")"
        for seed in $SEEDS; do
            out_dir="$(out_dir_for_run "$regime" "$dataset" "$arch" "$seed")"
            mkdir -p "$(dirname "$out_dir")"
            echo "  regime=${regime} arch=${arch} seed=${seed}"
            "$PYTHON" -u train.py --dataset="$dataset" $SCALE5_ARGS $flags \
                --seed="$seed" --out_dir="$out_dir" 2>&1 | tee "${out_dir}.log"
        done
    done
done

echo "=== Scale 5 stress tests complete $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
