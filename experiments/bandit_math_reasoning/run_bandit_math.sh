#!/bin/bash
# Training script for GSM8K Bandit Reasoning with Diverse Prompts
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate roll

# Set environment variables
export ROLL_PATH="/data1/Chengyang_project/bandit_llm"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Training timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"

# Logging
LOG_DIR="$ROLL_PATH/experiments/bandit_math_reasoning/output/${TIMESTAMP}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "GSM8K Bandit Training with Diverse Prompts" | tee -a "$LOG_FILE"
echo "Started at $(date)" | tee -a "$LOG_FILE"
echo "Timestamp: $TIMESTAMP" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Change to ROLL directory
cd "$ROLL_PATH"

# Run training with BanditAgenticPipeline (IMPORTANT: NOT regular AgenticPipeline!)
python experiments/bandit_math_reasoning/start_bandit_math_reasoning.py \
    --config_path . \
    --config_name config \
    2>&1 | tee -a "$LOG_FILE"

# Training completed
echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "Training completed at $(date)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
