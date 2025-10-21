#!/bin/bash
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 设置环境变量（适配Docker环境）
export ROLL_PATH="/data1/Chengyang_project/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

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

# 使用 exec 重定向，确保所有输出都被记录（比 tmux pipe-pane 更可靠）
# 这会将脚本的 stdout 和 stderr 同时输出到屏幕和日志文件
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "NQ Search Training with Replay Buffer"
echo "日志记录已启用"
echo "日志文件: $LOG_FILE"

# 检测是否在tmux中运行（仅用于信息显示）
if [ -n "$TMUX" ]; then
    TMUX_SESSION=$(tmux display-message -p '#S')
    TMUX_WINDOW=$(tmux display-message -p '#W')
    TMUX_PANE=$(tmux display-message -p '#P')
    echo "Running in tmux session: $TMUX_SESSION, window: $TMUX_WINDOW, pane: $TMUX_PANE"
fi
echo "========================================"

# 验证检索服务器是否可用
echo "Checking retrieval server availability..."
if curl -s --max-time 5 http://127.0.0.1:8100/health > /dev/null 2>&1; then
    echo "✓ Retrieval server is available at http://127.0.0.1:8100"
else
    echo "⚠ WARNING: Retrieval server at http://127.0.0.1:8100 is not responding!"
    echo "Please start the retrieval server before training."
    echo "You can check the server status with: curl http://127.0.0.1:8100/health"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Training aborted. Please start the retrieval server first."
        exit 1
    fi
fi
echo ""

# 验证GPU可用性
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
else
    echo "nvidia-smi not available, skipping GPU check"
fi
echo ""

# 验证数据集是否存在
DATASET_PATH="/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet"
if [ -f "$DATASET_PATH" ]; then
    echo "✓ Dataset found at $DATASET_PATH"
else
    echo "⚠ WARNING: Dataset not found at $DATASET_PATH"
    echo "Please ensure the NQ dataset has been converted using the conversion script."
    exit 1
fi
echo ""

# 启动训练
cd "$ROLL_PATH"
echo "Training started at $(date)"
echo "Changing to ROLL directory: $ROLL_PATH"
echo "Using unified timestamp: $TRAINING_TIMESTAMP"
echo "Environment: NQ Search with Retrieval"
echo ""

python examples/start_agentic_pipeline.py \
    --config_path ../../experiments/nq_search_replay \
    --config_name agent_nq_search_trajrb

echo ""
echo "========================================"
echo "Training completed at $(date)"
echo "All logs unified in directory: $OUTPUT_DIR"
echo "Training log: $LOG_FILE"
echo "Wandb logs: $WANDB_DIR"
echo "Models saved to: $OUTPUT_DIR/models"
echo "========================================"

# 显示最终日志目录结构
echo ""
echo "=== Final Output Directory Structure ==="
echo "Directory: $OUTPUT_DIR"
find "$OUTPUT_DIR" -type d -name "*" | head -10
echo "========================================"

# 显示日志文件信息
if [ -f "$LOG_FILE" ]; then
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo "日志文件已保存: $LOG_FILE (大小: $LOG_SIZE)"
else
    echo "警告: 日志文件未找到: $LOG_FILE"
fi

