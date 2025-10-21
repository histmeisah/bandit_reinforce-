#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置环境变量（适配Docker环境）
export ROLL_PATH="/data1/Weiyu_project/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置无头模式避免pygame在Docker中的显示问题
export SDL_VIDEODRIVER=dummy
export XDG_RUNTIME_DIR=/tmp/runtime-${USER}
mkdir -p "${XDG_RUNTIME_DIR}"
chmod 700 "${XDG_RUNTIME_DIR}"

# !!! 关键修复：统一时间戳，解决日志分散问题 !!!
# 生成一次时间戳，通过环境变量传递给配置文件，确保所有日志在同一目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
OUTPUT_DIR="$SCRIPT_DIR/output/$TIMESTAMP"

# 设置wandb为离线模式
export WANDB_MODE=offline
export WANDB_DIR="$OUTPUT_DIR/wandb"

# 创建带时间戳的输出目录
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/models"
mkdir -p "$OUTPUT_DIR/tensorboard"
mkdir -p "$OUTPUT_DIR/render"
mkdir -p "$OUTPUT_DIR/wandb"

# 设置日志文件
LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"

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

echo "========================================"
echo "7GPU Jidi Environments Training Pipeline"
echo "========================================"
echo "Config: jidi_environments_training.yaml"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTHONPATH: $PYTHONPATH"
echo "Training Timestamp: $TRAINING_TIMESTAMP"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "WANDB mode: offline (saved to $WANDB_DIR)"
echo "Working directory: $SCRIPT_DIR"
echo ""
echo "=== Training Parameters (Async 7GPU) ==="
echo "Environment Types: CliffWalking, GridWorld, MiniGrid, Sokoban"
echo "max_steps: 1024"
echo "async_generation_ratio: 1"
echo "rollout_batch_size: 1023 (3GPU divisible)"
echo "val_batch_size: 1023"
echo "sequence_length: 8192"
echo "adv_estimator: grpo"
echo "per_device_train_batch_size: 1 (reduced to avoid OOM)"
echo "gradient_accumulation_steps: 128 (3GPU optimized)"
echo "train_env_manager.group_size: 8"
echo "strategy: megatron_train + vLLM"
echo "device_mapping:"
echo "  - actor_train: GPU indices 0-2 (physical GPUs 0-2)"
echo "  - actor_infer: GPU indices 3-6 (physical GPUs 3-6, vLLM)"
echo "  - reference: GPU indices 0-2 (physical GPUs 0-2)"
echo "========================================"

# 验证GPU可用性
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
else
    echo "nvidia-smi not available, skipping GPU check"
fi
echo ""

# 验证Jidi环境可用性
echo "Checking Jidi environments availability..."
cd "$ROLL_PATH"
python -c "
from roll.agentic.env import REGISTERED_ENVS
jidi_envs = ['jidi_cliffwalking', 'jidi_gridworld', 'jidi_minigrid', 'jidi_sokoban']
print('Available Jidi environments:')
for env_name in jidi_envs:
    if env_name in REGISTERED_ENVS:
        print(f'  ✅ {env_name}')
    else:
        print(f'  ❌ {env_name} - NOT FOUND')
        exit(1)
print('All Jidi environments are available!')
"
echo ""

# 启动训练
cd "$ROLL_PATH"
echo "Training started at $(date)"
echo "Changing to ROLL directory: $ROLL_PATH"
echo "Using unified timestamp: $TRAINING_TIMESTAMP"

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/jidi_environments_training \
    --config_name jidi_environments_training

echo "Training completed at $(date)"
echo "All logs unified in directory: $OUTPUT_DIR"
echo "Training log: $LOG_FILE"
echo "Wandb logs: $WANDB_DIR"
echo "Models saved to: $OUTPUT_DIR/models"

# 如果在tmux中，停止日志记录
if [ "$USING_TMUX" = true ]; then
    tmux pipe-pane
    echo "Tmux logging stopped"
fi

# 显示最终日志目录结构
echo ""
echo "=== Final Output Directory Structure ==="
echo "Directory: $OUTPUT_DIR"
find "$OUTPUT_DIR" -type d -name "*" | head -10
echo "========================================"
