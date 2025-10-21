#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置环境变量
export ROLL_PATH="/data1/Weiyu_project/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置wandb为离线模式
export WANDB_MODE=offline
export WANDB_DIR="$SCRIPT_DIR/output/wandb"

# 创建输出目录（使用绝对路径）
mkdir -p "$SCRIPT_DIR/output/logs"
mkdir -p "$SCRIPT_DIR/output/models"
mkdir -p "$SCRIPT_DIR/output/tensorboard"
mkdir -p "$SCRIPT_DIR/output/render"
mkdir -p "$SCRIPT_DIR/output/wandb"

# 生成时间戳用于日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$SCRIPT_DIR/output/logs/training_${TIMESTAMP}.log"

# 检测是否在tmux中运行
if [ -n "$TMUX" ]; then
    TMUX_SESSION=$(tmux display-message -p '#S')
    TMUX_WINDOW=$(tmux display-message -p '#W')
    TMUX_PANE=$(tmux display-message -p '#P')
    echo "Running in tmux session: $TMUX_SESSION, window: $TMUX_WINDOW, pane: $TMUX_PANE"

    # 在tmux中启用日志记录
    tmux pipe-pane -o "cat >> $LOG_FILE"
    echo "Tmux output will be saved to: $LOG_FILE"
    USING_TMUX=true
else
    USING_TMUX=false
fi

# 设置错误处理 - 确保tmux日志记录能正确停止
cleanup() {
    if [ "$USING_TMUX" = true ]; then
        tmux pipe-pane
        echo "Tmux logging stopped due to error/interruption"
    fi
}
trap cleanup EXIT INT TERM

echo "Starting ROLL Agentic Pipeline (SYNC) with 8 GPUs..."
echo "Config: agent_val_frozen_lake_sync_8gpus.yaml"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTHONPATH: $PYTHONPATH"
echo "Log file: $LOG_FILE"
echo "WANDB mode: offline (saved to $WANDB_DIR)"
echo "Working directory: $SCRIPT_DIR"
echo "Training mode: SYNCHRONOUS (group_size=8, async_generation_ratio=0)"

# 启动训练
cd "$ROLL_PATH"
echo "Training started at $(date)"
echo "Changing to ROLL directory: $ROLL_PATH"

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/test_sync_train \
    --config_name agent_val_frozen_lake_sync_8gpus

echo "Training completed at $(date)"
echo "Log saved to: $LOG_FILE"
echo "Wandb logs saved to: $WANDB_DIR"

# 如果在tmux中，停止日志记录
if [ "$USING_TMUX" = true ]; then
    tmux pipe-pane
    echo "Tmux logging stopped"
fi 