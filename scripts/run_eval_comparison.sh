#!/bin/bash
set -e

# Default paths, can be overridden via environment variables
DIR_TGAP_NO_TRACKING=${DIR_TGAP_NO_TRACKING:-"runs/tgaprl_no_tracking"}
DIR_TGAP_TRACKING=${DIR_TGAP_TRACKING:-"runs/tgaprl_tracking"}
DIR_GAP_BASELINE=${DIR_GAP_BASELINE:-"runs/gaprl_baseline"}

MODEL_STEPS=${MODEL_STEPS:-"rl_model_2000000_steps"}
SEED=${SEED:-0}
EVAL_DATASET="acronym_eval"
TRAJ_MODE="random3d"

echo "Evaluating (a) Current best T-GAP-RL (no tracking)..."
python scripts/eval_tgaprl.py \
    --run-dir "$DIR_TGAP_NO_TRACKING" \
    --model-name "$MODEL_STEPS" \
    --eval-datasets $EVAL_DATASET \
    --gen-traj-modes $TRAJ_MODE \
    --seeds $SEED \
    --cam-mode both

echo "Evaluating (b) New checkpoint with tracking..."
python scripts/eval_tgaprl.py \
    --run-dir "$DIR_TGAP_TRACKING" \
    --model-name "$MODEL_STEPS" \
    --eval-datasets $EVAL_DATASET \
    --gen-traj-modes $TRAJ_MODE \
    --seeds $SEED \
    --cam-mode both

echo "Evaluating (c) Plain vanilla GAP-RL baseline (no LSTM)..."
python scripts/eval_tgaprl.py \
    --run-dir "$DIR_GAP_BASELINE" \
    --model-name "$MODEL_STEPS" \
    --eval-datasets $EVAL_DATASET \
    --gen-traj-modes $TRAJ_MODE \
    --seeds $SEED \
    --cam-mode both

echo "=========================================================="
echo "                  EVALUATION RESULTS                      "
echo "=========================================================="
echo "| Config | Success Rate | Mean Steps | Slipped after lift |"
echo "|--------|--------------|------------|--------------------|"

extract_metrics() {
    local dir=$1
    local name=$2
    local result_file=$(find "$dir" -path "*/simLoG_result_*/$TRAJ_MODE/success_rates_steps.txt" | head -n 1)
    
    if [[ -z "$result_file" || ! -f "$result_file" ]]; then
        echo "| $name | N/A | N/A | N/A |"
        return
    fi
    
    local sr=$(grep "success rates (mean, std)" "$result_file" | awk -F'[(,]' '{print $2}' | xargs)
    local steps=$(grep "success steps (mean, std)" "$result_file" | awk -F'[(,]' '{print $2}' | xargs)
    local slipped=$(grep "slipped_after_lift:" "$result_file" | awk '{print $2}')
    
    echo "| $name | $sr | $steps | $slipped |"
}

extract_metrics "$DIR_TGAP_NO_TRACKING" "T-GAP-RL (No Tracking)"
extract_metrics "$DIR_TGAP_TRACKING" "T-GAP-RL (Tracking)"
extract_metrics "$DIR_GAP_BASELINE" "GAP-RL (Vanilla)"
