#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ROLL_PATH="/data1/Weiyu_project/roll_dev/ROLL"
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TRAINING_TIMESTAMP="$TIMESTAMP"
export WANDB_MODE=offline
export WANDB_DIR="$SCRIPT_DIR/output/$TIMESTAMP/wandb"

echo "=== 测试扩展后的并发环境配置 ==="
echo "训练环境工作进程数: 8 (处理64个环境)"
echo "验证环境工作进程数: 4 (处理32个环境)"
echo "Rollout batch size: 64"
echo "Val batch size: 32"
echo ""

# 测试顺序：none -> simple -> areal -> prioritized
CONFIGS=("agent_val_frozen_lake_replay_none" "agent_val_frozen_lake_replay_simple" "agent_val_frozen_lake_replay_areal" "agent_val_frozen_lake_replay_prioritized")

for config in "${CONFIGS[@]}"; do
    echo "=== 测试配置: $config ==="
    
    cd "$ROLL_PATH"
    
    # 运行较少的步数以快速验证
    python examples/start_agentic_pipeline.py \
        --config_path ../../experiments/reproduce_frozen_lake_replay \
        --config_name $config \
        max_steps=5 \
        eval_steps=2 \
        || echo "配置 $config 测试失败"
    
    echo "配置 $config 测试完成"
    echo ""
    
    # 短暂休息，让系统清理资源
    sleep 5
done

echo "=== 所有扩展配置测试完成 ==="
