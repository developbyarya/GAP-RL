#!/bin/bash
# Usage: ./scripts/run_all_evals.sh runs/YOUR_RUN_DIR

if [ -z "$1" ]; then
    echo "Error: Please provide the run directory as an argument."
    echo "Usage: $0 runs/YOUR_RUN_DIR"
    exit 1
fi

RUN_DIR=$1
MODEL_NAME=${2:-rl_model_2000000_steps}
SEED=${3:-1029}

echo "Starting evaluation for all splits and trajectories..."
echo "Run Directory : $RUN_DIR"
echo "Model Name    : $MODEL_NAME"
echo "Seed          : $SEED"
echo "========================================================="

# The python script inherently supports multiple datasets and trajectory modes
# by using the nargs='+' feature in argparse. We pass all of them in one go.
python scripts/eval_tgaprl.py \
    --run-dir "$RUN_DIR" \
    --model-name "$MODEL_NAME" \
    --eval-datasets ycb_train ycb_eval graspnet_eval \
    --gen-traj-modes bezier2d random2d \
    --seeds "$SEED"

echo "========================================================="
echo "All evaluations completed!"
