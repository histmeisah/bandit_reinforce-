# ROLL 8GPU Async Training 实验

本目录包含了在单机8卡环境下进行ROLL agentic异步训练的配置和脚本。

## 文件说明

### 配置文件
- `agent_val_frozen_lake_async_8gpus.yaml`: 8GPU异步训练的主配置文件
- `logging_config.yaml`: 详细的日志配置文件

### 脚本文件
- `run_agentic_pipeline_frozen_lake_8gpu.sh`: 启动训练的主脚本
- `monitor_training.sh`: 训练监控脚本

## 主要配置特点

### GPU资源分配
- **Actor Train**: GPU 0-3 (4卡用于策略网络训练)
- **Actor Infer**: GPU 4-7 (4卡用于策略网络推理)
- **Reference**: GPU 0-3 (与训练共享，用于参考模型)

### 异步训练配置
```yaml
async_generation_ratio: 1          # LLM生成完全异步
rollout_batch_size: 256           # 适合8卡的批量大小
sequence_length: 4096             # 序列长度
max_steps: 100                    # 测试用的步数

# 关键: 环境异步配置
train_env_manager:
  group_size: 1                   # 完全环境异步 (每个环境独立执行)
  num_env_groups: 256             # 大量环境组配合异步
```

### 日志和监控
- **Wandb**: 配置为离线模式，避免上传问题
- **终端日志**: 自动保存到 `./output/logs/training_TIMESTAMP.log`
- **调试日志**: 保存到 `./output/logs/roll_debug.log`

## 使用方法

### 1. 启动训练
```bash
./run_agentic_pipeline_frozen_lake_8gpu.sh
```

### 2. 监控训练进度
```bash
# 单次查看
./monitor_training.sh

# 持续监控（每30秒刷新）
watch -n 30 ./monitor_training.sh
```

### 3. 查看实时日志
```bash
# 查看最新的训练日志
tail -f ./output/logs/training_*.log

# 查看调试日志
tail -f ./output/logs/roll_debug.log
```

## 输出目录结构

```
./output/
├── logs/                          # 日志文件
│   ├── training_TIMESTAMP.log     # 训练日志
│   └── roll_debug.log             # 调试日志
├── models/                        # 模型检查点
├── wandb/                         # Wandb离线日志
├── tensorboard/                   # TensorBoard日志（备用）
└── render/                        # 环境渲染文件
```

## 环境要求

- CUDA 11.8+
- Python 3.8+
- 8张GPU（建议V100/A100/H100）
- 足够的磁盘空间用于日志和模型保存

## 故障排除

### 1. GPU内存不足
- 减少 `rollout_batch_size` 和 `val_batch_size`
- 减少 `per_device_train_batch_size`
- 调整 `gpu_memory_utilization`

### 2. 日志文件过大
- 日志文件会自动轮转（最大10MB，保留5个备份）
- 可以手动清理旧的日志文件

### 3. Wandb问题
- 确保设置了 `WANDB_MODE=offline`
- 检查 `./output/wandb` 目录权限

## 异步配置详解

### 双重异步机制
1. **LLM生成异步**: `async_generation_ratio: 1`
2. **环境执行异步**: `group_size: 1`

### group_size 的重要性
- `group_size = 1`: 完全异步，每个环境独立执行（当前配置）
- `group_size > 1`: 部分同步，同组环境需要等待彼此
- 详见: `group_size_comparison.md`

## 性能调优建议

1. **批量大小调整**: 根据GPU内存调整batch size
2. **序列长度**: 根据任务需求调整sequence_length
3. **环境并发**: 调整 `max_env_num_per_worker` 和 `num_env_groups`
4. **异步比例**: 可以调整 `async_generation_ratio` (0-1之间)
5. **环境异步**: 调整 `group_size` (1=完全异步, >1=部分同步)

## 联系信息

如有问题，请查看日志文件或联系开发团队。
