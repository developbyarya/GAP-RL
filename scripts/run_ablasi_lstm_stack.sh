#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/run_ablasi_lstm_stack.sh
# Deskripsi: Menjalankan training CustomSAC (LSTM) dengan variasi n_stack (1, 2, 8)
#            dan evaluasi dengan 10 random seed untuk tiap model.
# Prefix Eksperimen: ablasi_lstm_stack_[n]
# ==============================================================================

set -euo pipefail

# Menentukan direktori root repository
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Set PYTHONPATH agar modul internal dapat di-import dengan benar
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/gap_rl/algorithms/scripts:$REPO_ROOT/gap_rl/localgrasp:${PYTHONPATH:-}"

# Konfigurasi Environment & Python interpreter
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_NAME="${CONFIG_NAME:-egopoints_ur85_bezier2d_decoupled_lstm}"
TEMPORAL_MODE="${TEMPORAL_MODE:-decoupled}"

# Parameter ablasi n_stack
STACK_PARAMS=(1 2 8)

# 10 random seed untuk evaluasi
SEEDS=(1029 1030 1031 1032 1033 1034 1035 1036 1037 1038)

# Dataset dan mode trajektori untuk evaluasi
EVAL_DATASETS=("ycb_train" "ycb_eval" "graspnet_eval")
GEN_TRAJ_MODES=("bezier2d" "random2d")

echo "=================================================================="
echo " GAP-RL CustomSAC Ablasi LSTM n_stack: (${STACK_PARAMS[*]})"
echo " Config        : $CONFIG_NAME"
echo " Temporal Mode : $TEMPORAL_MODE"
echo " Eval Seeds    : (${SEEDS[*]})"
echo " Eval Datasets : (${EVAL_DATASETS[*]})"
echo " Traj Modes    : (${GEN_TRAJ_MODES[*]})"
echo "=================================================================="

for n in "${STACK_PARAMS[@]}"; do
    EXP_PREFIX="ablasi_lstm_stack_${n}"
    TIMESTAMP="${EXP_PREFIX}_$(date +%Y%m%d_%H%M%S)"
    
    echo ""
    echo "##################################################################"
    echo " [1/2] Memulai Training: n_stack = $n"
    echo " Prefix / Suffix: $EXP_PREFIX"
    echo " Timestamp      : $TIMESTAMP"
    echo "##################################################################"

    # Jalankan training CustomSAC
    $PYTHON_BIN gap_rl/algorithms/scripts/sac_train.py \
        --config-name "$CONFIG_NAME" \
        --use-lstm \
        --temporal-mode "$TEMPORAL_MODE" \
        --n-stack "$n" \
        --exp-suffix "$EXP_PREFIX" \
        --timestamp "$TIMESTAMP"

    # Cari direktori output training yang baru dibuat
    RUN_DIR=$(find . -maxdepth 1 -type d -name "${TIMESTAMP}*" | head -n 1)
    if [ -z "$RUN_DIR" ]; then
        RUN_DIR=$(find gap_rl/algorithms/scripts -maxdepth 1 -type d -name "${TIMESTAMP}*" | head -n 1)
    fi

    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "Error: Direktori run untuk ${TIMESTAMP} tidak ditemukan!"
        exit 1
    fi

    echo "Training selesai. Direktori run: $RUN_DIR"

    # Tentukan model checkpoint yang akan dievaluasi (prioritaskan best_model jika ada)
    if [ -f "$RUN_DIR/best_checkpoint/best_model.zip" ]; then
        MODEL_NAME="best_checkpoint/best_model"
        echo "Menggunakan checkpoint terbaik: $MODEL_NAME"
    elif [ -f "$RUN_DIR/rl_model_2000000_steps.zip" ]; then
        MODEL_NAME="rl_model_2000000_steps"
        echo "Menggunakan checkpoint 2M: $MODEL_NAME"
    else
        # Fallback ke checkpoint zip terakhir yang tersedia
        LATEST_ZIP=$(find "$RUN_DIR" -maxdepth 1 -name "*.zip" | sort -V | tail -n 1)
        if [ -n "$LATEST_ZIP" ]; then
            MODEL_NAME="$(basename "$LATEST_ZIP" .zip)"
            echo "Menggunakan checkpoint fallback: $MODEL_NAME"
        else
            echo "Error: Tidak ada checkpoint .zip yang ditemukan di $RUN_DIR!"
            exit 1
        fi
    fi

    echo ""
    echo "##################################################################"
    echo " [2/2] Memulai Evaluasi dengan 10 Random Seeds"
    echo " Model Checkpoint: $MODEL_NAME"
    echo " Seeds           : ${SEEDS[*]}"
    echo "##################################################################"

    # Jalankan evaluasi LocalGrasp & RL policy
    $PYTHON_BIN scripts/eval_tgaprl.py \
        --run-dir "$RUN_DIR" \
        --model-name "$MODEL_NAME" \
        --eval-datasets "${EVAL_DATASETS[@]}" \
        --gen-traj-modes "${GEN_TRAJ_MODES[@]}" \
        --seeds "${SEEDS[@]}"

    echo "Evaluasi untuk n_stack = $n selesai!"
done

echo ""
echo "=================================================================="
echo " Semua eksperimen ablasi n_stack (1, 2, 8) dan evaluasi telah selesai!"
echo "=================================================================="
