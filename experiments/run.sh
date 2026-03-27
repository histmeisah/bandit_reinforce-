#!/bin/bash
# ==============================================================================
# Bandit REINFORCE++ Experiments Runner - Debug 2xGPU
# ==============================================================================
# Available configs:
#   exp1_bandit_ts_dapo      : Bandit(Thompson Sampling) + REINFORCE++ (main)
#   exp2_baseline_no_bandit  : Baseline REINFORCE++ without bandit (ablation)
#   exp3_bandit_ucb_dapo     : Bandit(UCB) + REINFORCE++ (TS vs UCB ablation)
# ==============================================================================
set -e

# Load environment
source activate roll 2>/dev/null || conda activate roll 2>/dev/null || true
module load cuda/12.4.1

# =============================================================================
# CONFIG NAME - Modify this to run different experiments
# =============================================================================
CONFIG_NAME="${1:-exp1_bandit_ts_dapo}"
# =============================================================================

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROLL_DIR="/ibex/user/maw0a/python_project/bandit_reinforce/ROLL"
BASE_OUTPUT="/ibex/user/maw0a/python_project/bandit_reinforce/output"

cd "$SCRIPT_DIR"

# Verify config file exists
if [ ! -f "${CONFIG_NAME}.yaml" ]; then
    echo "Error: Config file '${CONFIG_NAME}.yaml' not found!"
    echo "Available configs:"
    ls -1 *.yaml 2>/dev/null | sed 's/.yaml$//'
    exit 1
fi

# Set ROLL in PYTHONPATH
export PYTHONPATH="$ROLL_DIR:$PYTHONPATH"

# Auto-detect visible GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
else
    export CUDA_VISIBLE_DEVICES=0,1
fi

# Fix vLLM CUDA IPC permission issue in container (ptrace_scope=1)
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1                                                                                                                                                                    
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# Generate unified timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$BASE_OUTPUT/${CONFIG_NAME}_${TIMESTAMP}"

# Set wandb to offline mode
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR/wandb"

# Create output directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/tensorboard"
mkdir -p "$OUTPUT_DIR/wandb"

# Set log file - redirect stdout and stderr to both screen and log file
LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================================"
echo " Bandit REINFORCE++ Experiment"
echo "========================================================"
echo " Config:     ${CONFIG_NAME}"
echo " Timestamp:  $TIMESTAMP"
echo " Log file:   $LOG_FILE"
echo " Output dir: $OUTPUT_DIR"
echo " ROLL dir:   $ROLL_DIR"
echo "========================================================"

# Check tmux session info
if [ -n "$TMUX" ]; then
    TMUX_SESSION=$(tmux display-message -p '#S')
    TMUX_WINDOW=$(tmux display-message -p '#W')
    echo " tmux:       $TMUX_SESSION / $TMUX_WINDOW"
fi
echo ""

# Check GPU availability
echo "========================================================"
echo " GPU Status"
echo "========================================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
else
    echo "nvidia-smi not available"
fi
echo ""

# Check model and data exist
echo "========================================================"
echo " Pre-flight Checks"
echo "========================================================"
MODEL_PATH="/ibex/user/maw0a/python_project/models/Qwen3-4B-Base"
DATA_PATH="/ibex/user/maw0a/python_project/bandit_reinforce/ROLL/data/dapo_math_17k.jsonl"
ENCODER_PATH="/ibex/user/maw0a/python_project/models/Qwen3-Embedding-0.6B"

[ -d "$MODEL_PATH" ]   && echo " [OK] Model:   $MODEL_PATH" || echo " [FAIL] Model not found: $MODEL_PATH"
[ -f "$DATA_PATH" ]    && echo " [OK] Data:    $DATA_PATH ($(wc -l < "$DATA_PATH") rows)" || echo " [FAIL] Data not found: $DATA_PATH"
[ -d "$ENCODER_PATH" ] && echo " [OK] Encoder: $ENCODER_PATH" || echo " [WARN] Encoder not found: $ENCODER_PATH (only needed for bandit experiments)"
echo ""

# Clean up existing Ray clusters
echo "========================================================"
echo " Cleaning up Ray..."
echo "========================================================"
ray stop --force 2>/dev/null || true
pkill -9 -u $USER ray 2>/dev/null || true
rm -rf /tmp/ray/* 2>/dev/null || true
sleep 2
echo " Ray cleanup completed."
echo ""

# Start training
echo "========================================================"
echo " Training started at $(date)"
echo "========================================================"
echo ""

cd "$ROLL_DIR"

# Hydra requires relative config_path (relative to start_agentic_pipeline.py)
REL_CONFIG_PATH=$(python -c "import os; print(os.path.relpath('$SCRIPT_DIR', '$ROLL_DIR/examples'))")
echo " Config path (relative): $REL_CONFIG_PATH"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path "$REL_CONFIG_PATH" \
    --config_name "$CONFIG_NAME"

echo ""
echo "========================================================"
echo " Training completed at $(date)"
echo "========================================================"
echo " Config:     ${CONFIG_NAME}"
echo " Output dir: $OUTPUT_DIR"
echo " Log file:   $LOG_FILE"
echo "========================================================"
