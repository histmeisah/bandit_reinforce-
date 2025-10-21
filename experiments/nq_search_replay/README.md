# NQ Search Training with Replay Buffer

这个实验配置用于训练基于 NQ (Natural Questions) 数据集的检索增强问答系统，使用 Trajectory Replay Buffer 和 Off-Policy 学习。

## 环境要求

### 1. 检索服务器

训练前必须启动检索服务器：

```bash
# 服务器地址: http://127.0.0.1:8100
# 检查服务器状态
curl http://127.0.0.1:8100/health
```

### 2. 数据集

确保已经转换好的 NQ 数据集位于：
```
/data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet
```

如果数据集尚未转换，请运行：
```bash
cd /data1/Chengyang_project/roll_dev/ROLL
python roll/utils/nq_data_process/convert_all_datasets.py
```

### 3. GPU 资源

- 需要 8 张 GPU
- 配置为 4 卡训练 + 4 卡推理
- 建议每张卡至少 24GB 显存

## 配置说明

### 关键参数

**训练配置：**
- `rollout_batch_size`: 64（搜索任务序列较长，减小batch size）
- `sequence_length`: 8192（支持多轮搜索交互）
- `max_steps`: 100
- `learning_rate`: 5e-7（降低学习率，任务更复杂）

**环境配置：**
- `max_actions_per_traj`: 10（最大交互轮数）
- `max_tokens_per_step`: 512（每步生成token数）
- `retrieval_topk`: 3（检索返回top-3文档）
- `max_search_calls`: 5（最多搜索5次）

**Replay Buffer：**
- `capacity`: 50000（轨迹容量）
- `sampling_mode`: "trajectory"（轨迹级采样）
- `sample_method`: "lifo"（后进先出）

**Old Prob 设置：**
- `old_prob_mode`: "turn"（按轮次计算）
- `old_prob_compute`: "trainer"（训练侧计算）

## 使用方法

### 在宿主机运行（推荐）

```bash
cd /data1/Chengyang_project/roll_dev/experiments/nq_search_replay
./run_nq_search.sh
```

### 在 Docker 容器中运行

```bash
docker exec -it roll_vllm /bin/bash
cd /data1/Chengyang_project/roll_dev/experiments/nq_search_replay
./run_nq_search.sh
```

### 在 tmux 中后台运行

```bash
tmux new -s nq_search_training
cd /data1/Chengyang_project/roll_dev/experiments/nq_search_replay
./run_nq_search.sh
# Ctrl+B, D 退出 tmux
```

## 输出结构

训练完成后，所有输出将统一保存在：
```
experiments/nq_search_replay/output/YYYYMMDD_HHMMSS/
├── logs/                    # 训练日志
│   └── training_*.log
├── models/                  # 模型检查点
│   └── nq_search_agentic/
├── wandb/                   # WandB 离线日志
├── tensorboard/             # TensorBoard 日志
└── render/                  # 环境交互渲染
```

## 监控训练

### 查看实时日志

```bash
# 方法1：直接查看日志文件
tail -f experiments/nq_search_replay/output/latest/logs/training_*.log

# 方法2：在 tmux 中查看
tmux attach -t nq_search_training
```

### WandB 可视化

```bash
# 同步离线日志到云端
wandb sync experiments/nq_search_replay/output/YYYYMMDD_HHMMSS/wandb
```

### TensorBoard 可视化

```bash
tensorboard --logdir experiments/nq_search_replay/output/YYYYMMDD_HHMMSS/tensorboard
```

## 问题排查

### 1. 检索服务器连接失败

**症状：** 训练启动时提示 "Retrieval server is not responding"

**解决：**
```bash
# 检查服务器状态
curl http://127.0.0.1:8100/health

# 如果服务器未启动，请先启动检索服务器
```

### 2. 数据集未找到

**症状：** "Dataset not found at /data1/Agentic_LLM-search/datasets/nq_search_converted/train_searchenv.parquet"

**解决：**
```bash
# 运行数据集转换脚本
cd /data1/Chengyang_project/roll_dev/ROLL
python roll/utils/nq_data_process/convert_all_datasets.py
```

### 3. GPU OOM (显存不足)

**症状：** CUDA out of memory 错误

**解决：** 调整配置文件中的以下参数
```yaml
# 减小 batch size
rollout_batch_size: 32  # 从 64 改为 32
per_device_train_batch_size: 1

# 增加梯度累积
gradient_accumulation_steps: 64  # 从 32 改为 64

# 减小序列长度
sequence_length: 4096  # 从 8192 改为 4096
```

## 与 FrozenLake 实验的对比

| 配置项 | FrozenLake | NQ Search | 说明 |
|--------|------------|-----------|------|
| 环境类型 | frozen_lake | nq_search | 搜索任务更复杂 |
| Batch Size | 128 | 64 | 搜索序列更长 |
| 序列长度 | 4096 | 8192 | 支持多轮交互 |
| 学习率 | 1e-6 | 5e-7 | 搜索任务需要更稳定训练 |
| Max Tokens | 128 | 512 | 搜索需要生成更多内容 |
| Action Pattern | `<answer>` | `<think>/<search>/<answer>` | 多种动作类型 |
| Replay Capacity | 100000 | 50000 | 搜索轨迹更长 |

## 参考文档

- [NQ Search Environment Guide](/data1/Chengyang_project/roll_dev/ROLL/docs/nq_search_env_guide.md)
- [ROLL Framework Documentation](/data1/Chengyang_project/roll_dev/ROLL/README.md)
- [Replay Buffer Configuration](/data1/Chengyang_project/roll_dev/ROLL/roll/agentic/replay_buffer/README.md)

