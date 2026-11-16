#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/eval_ablasi_lstm_stack.sh
# Deskripsi: Menjalankan evaluasi terpisah (LocalGrasp + RL) dengan 10 random seed
#            untuk model ablasi n_stack (1, 2, 8) yang sudah selesai ditraining.
# ==============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/gap_rl/algorithms/scripts:$REPO_ROOT/gap_rl/localgrasp:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE="${FORCE:-false}"

# Parameter default stack yang dicari jika tidak ada argumen spesifik
STACK_PARAMS=(1 2 8)

# Dataset dan mode trajektori untuk evaluasi
EVAL_DATASETS=("ycb_train" "ycb_eval" "graspnet_eval")
GEN_TRAJ_MODES=("bezier2d" "random2d")

generate_random_seeds() {
    $PYTHON_BIN -c "
import random
seeds = set()
while len(seeds) < 10:
    seeds.add(random.randint(1000, 999999))
for s in sorted(seeds):
    print(s)
"
}

# Tentukan list run directory yang akan dievaluasi
TARGET_RUNS=()

if [ $# -gt 0 ]; then
    # Jika user memberikan argumen folder run spesifik
    for arg in "$@"; do
        if [ -d "$arg" ]; then
            TARGET_RUNS+=("$(cd "$arg" && pwd)")
        else
            echo "Warning: Direktori '$arg' tidak ditemukan, dilewati."
        fi
    done
else
    # Otomatis mencari folder run terbaru untuk masing-masing n_stack (1, 2, 8)
    echo "Mencari folder run otomatis untuk ablasi n_stack (${STACK_PARAMS[*]})..."
    for n in "${STACK_PARAMS[@]}"; do
        FOUND_DIR=$(find . runs gap_rl/algorithms/scripts -maxdepth 2 -type d -name "ablasi_lstm_stack_${n}_*" 2>/dev/null | sort -V | tail -n 1 || true)
        if [ -n "$FOUND_DIR" ] && [ -d "$FOUND_DIR" ]; then
            TARGET_RUNS+=("$(cd "$FOUND_DIR" && pwd)")
            echo "  -> n_stack=$n: $FOUND_DIR"
        else
            # Coba cari nama folder yang mengandung ablasi_lstm_stack_${n} di bagian manapun
            FOUND_DIR=$(find . runs gap_rl/algorithms/scripts -maxdepth 2 -type d -name "*ablasi_lstm_stack_${n}*" 2>/dev/null | sort -V | tail -n 1 || true)
            if [ -n "$FOUND_DIR" ] && [ -d "$FOUND_DIR" ]; then
                TARGET_RUNS+=("$(cd "$FOUND_DIR" && pwd)")
                echo "  -> n_stack=$n: $FOUND_DIR"
            else
                echo "  -> n_stack=$n: [TIDAK DITEMUKAN]"
            fi
        fi
    done
fi

if [ ${#TARGET_RUNS[@]} -eq 0 ]; then
    echo "Error: Tidak ada direktori run yang ditemukan untuk dievaluasi!"
    echo "Usage:"
    echo "  $0                                      # Otomatis cari folder ablasi_lstm_stack_*"
    echo "  $0 path/to/run_dir1 [path/to/run_dir2]  # Evaluasi folder run tertentu"
    exit 1
fi

echo ""
echo "=================================================================="
echo " GAP-RL Evaluasi Khusus Model Ablasi"
echo " Jumlah Direktori: ${#TARGET_RUNS[@]}"
echo " Eval Datasets   : (${EVAL_DATASETS[*]})"
echo " Traj Modes      : (${GEN_TRAJ_MODES[*]})"
echo " Force Re-eval   : $FORCE"
echo "=================================================================="

for RUN_DIR in "${TARGET_RUNS[@]}"; do
    echo ""
    echo "##################################################################"
    echo " Memproses Evaluasi Direktori: $(basename "$RUN_DIR")"
    echo " Path: $RUN_DIR"
    echo "##################################################################"

    # 1. Tentukan Checkpoint Model
    if [ -f "$RUN_DIR/best_checkpoint/best_model.zip" ]; then
        MODEL_NAME="best_checkpoint/best_model"
        echo "Checkpoint : $MODEL_NAME (Best Model)"
    elif [ -f "$RUN_DIR/rl_model_2000000_steps.zip" ]; then
        MODEL_NAME="rl_model_2000000_steps"
        echo "Checkpoint : $MODEL_NAME (2M steps)"
    else
        LATEST_ZIP=$(find "$RUN_DIR" -maxdepth 1 -name "*.zip" | sort -V | tail -n 1 || true)
        if [ -n "$LATEST_ZIP" ]; then
            MODEL_NAME="$(basename "$LATEST_ZIP" .zip)"
            echo "Checkpoint : $MODEL_NAME (Fallback latest zip)"
        else
            echo "Error: Tidak ada file checkpoint .zip yang ditemukan di $RUN_DIR! Dilewati."
            continue
        fi
    fi

    # 2. Tentukan Seeds Evaluasi
    SEEDS=()
    if [ -n "${CUSTOM_SEEDS:-}" ]; then
        # Jika user mengoper custom seeds melalui env var
        IFS=' ' read -r -a SEEDS <<< "$CUSTOM_SEEDS"
        echo "Seeds      : Menggunakan CUSTOM_SEEDS dari environment (${SEEDS[*]})"
    elif [ -f "$RUN_DIR/eval_seeds.txt" ] && [ -s "$RUN_DIR/eval_seeds.txt" ]; then
        # Gunakan seed yang tersimpan saat training untuk konsistensi
        mapfile -t SEEDS < "$RUN_DIR/eval_seeds.txt"
        echo "Seeds      : Membaca dari $RUN_DIR/eval_seeds.txt (${SEEDS[*]})"
    else
        # Generate 10 seed baru dan simpan
        mapfile -t SEEDS < <(generate_random_seeds)
        printf "%s\n" "${SEEDS[@]}" > "$RUN_DIR/eval_seeds.txt"
        echo "Seeds      : Generate 10 seed baru & disimpan ke eval_seeds.txt (${SEEDS[*]})"
    fi

    # 3. Parameter tambahan
    EXTRA_ARGS=()
    if [ "$FORCE" = "true" ]; then
        EXTRA_ARGS+=("--force")
    fi

    # 4. Jalankan Evaluasi
    echo "Menjalankan evaluasi..."
    $PYTHON_BIN scripts/eval_tgaprl.py \
        --run-dir "$RUN_DIR" \
        --model-name "$MODEL_NAME" \
        --eval-datasets "${EVAL_DATASETS[@]}" \
        --gen-traj-modes "${GEN_TRAJ_MODES[@]}" \
        --seeds "${SEEDS[@]}" \
        "${EXTRA_ARGS[@]}"

    echo "Selesai evaluasi untuk $(basename "$RUN_DIR")!"
done

echo ""
echo "=================================================================="
echo " Semua evaluasi yang diminta telah selesai!"
echo "=================================================================="
