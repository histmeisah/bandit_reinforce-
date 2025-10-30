#!/bin/bash

# 简单的 WandB 一次性同步脚本
# 不需要文件监控，直接同步指定的 run

set -e

# 配置 - 直接填写绝对路径
WANDB_DIR="E:\code_project\python_code\local_roll_dev\roll_dev\experiments\nq_search_replay\output\offline-run-20251024_131308-osebgtlq"
PROXY_HOST="127.0.0.1"
PROXY_PORT="7898"
WANDB_PROJECT="roll_async_env"
WANDB_API_KEY="5d830c409e2aa7dff34c333a2f79798a877bfc7b"

# Conda环境配置
CONDA_BASHRC="/data1/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="roll_env"

echo "🚀 WandB 一次性同步脚本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📂 WandB路径: $WANDB_DIR"
echo "🌐 代理设置: $PROXY_HOST:$PROXY_PORT"
echo "📊 WandB项目: $WANDB_PROJECT"
echo "🔑 API密钥: ${WANDB_API_KEY:0:10}..."
echo "🐍 Conda环境: $CONDA_ENV"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查WandB目录是否存在
if [ ! -d "$WANDB_DIR" ]; then
    echo "❌ 错误: WandB目录不存在: $WANDB_DIR"
    exit 1
fi

# 检查conda环境文件是否存在
if [ ! -f "$CONDA_BASHRC" ]; then
    echo "❌ 错误: Conda bashrc文件不存在: $CONDA_BASHRC"
    exit 1
fi

echo "✅ 路径检查通过"

# 设置代理环境变量
export HTTP_PROXY="http://$PROXY_HOST:$PROXY_PORT"
export HTTPS_PROXY="http://$PROXY_HOST:$PROXY_PORT"
export http_proxy="http://$PROXY_HOST:$PROXY_PORT"
export https_proxy="http://$PROXY_HOST:$PROXY_PORT"
export NO_PROXY="localhost,127.0.0.1,::1"
export no_proxy="localhost,127.0.0.1,::1"

# 设置 WandB 环境变量
export WANDB_PROJECT="$WANDB_PROJECT"
export WANDB_API_KEY="$WANDB_API_KEY"

echo "🌐 已设置代理: $PROXY_HOST:$PROXY_PORT"
echo "📊 已设置WandB项目: $WANDB_PROJECT"

# 测试网络连接
echo ""
echo "🔍 测试网络连接..."
if curl -s --connect-timeout 10 https://api.wandb.ai > /dev/null; then
    echo "✅ 网络连接正常"
else
    echo "⚠️ 网络连接测试失败，但继续尝试同步..."
fi

# 初始化 conda 环境
echo ""
echo "🐍 初始化 Conda 环境..."
source "$CONDA_BASHRC"
conda activate "$CONDA_ENV"
echo "✅ 已激活环境: $CONDA_ENV"

# 进入 WandB 目录的父目录
cd "$(dirname "$WANDB_DIR")"
echo "📂 当前工作目录: $(pwd)"

# 执行同步
echo ""
echo "🎯 开始同步 WandB 运行..."
echo "📁 同步目录: $(basename "$WANDB_DIR")"

# 使用 wandb sync 命令同步
if wandb sync "$(basename "$WANDB_DIR")"; then
    echo ""
    echo "🎉 同步成功完成!"
    echo "✅ 运行已上传到 WandB 项目: $WANDB_PROJECT"
    echo "🌐 请访问 https://wandb.ai 查看结果"
else
    echo ""
    echo "❌ 同步失败!"
    echo "请检查:"
    echo "  - 网络连接是否正常"
    echo "  - 代理设置是否正确"
    echo "  - WandB API 密钥是否有效"
    echo "  - 目录路径是否正确"
    exit 1
fi

echo ""
echo "📋 如需同步其他运行，请修改脚本中的 WANDB_DIR 路径"