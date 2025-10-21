# StepEnvManager使用指南

## 概述

`StepEnvManager`是专门为步级别（step-level）训练设计的环境管理器。与`TrajEnvManager`不同，它会为轨迹中的每一步生成独立的训练样本。

## 与TrajEnvManager的主要区别

### 1. 数据组织方式

- **TrajEnvManager**: 一个轨迹生成一个训练样本，包含完整的对话历史
- **StepEnvManager**: 一个轨迹生成多个训练样本，每步一个

### 2. Response Mask处理

- **TrajEnvManager**: 需要手动处理step模式的response_mask（如我们刚才的修改）
- **StepEnvManager**: 天然支持step级别的response_mask，每个样本只包含当前步的回复

### 3. 奖励分配

- **TrajEnvManager**: episode奖励放在最后一个token上
- **StepEnvManager**: 每步的奖励放在该步回复的最后一个token上

## 配置修改指南

### 1. 修改环境配置

```yaml
custom_envs:
  FrozenLake:
    env_type: frozen_lake
    # 改为使用StepEnvManager
    env_manager_cls: roll.pipeline.agentic.env_manager.step_env_manager.StepEnvManager
    use_thread_lock: true
    # 其他配置保持不变
    env_config:
      env_instruction: "..."
      action_pattern: ${action_pattern}
      max_steps: ${max_actions_per_traj}
      is_slippery: false
```

### 2. 修改Replay Buffer配置

由于StepEnvManager生成的是步级别的数据，需要使用`StepReplayBuffer`：

```yaml
replay:
  enabled: true
  capacity: 1000000  # 注意：这是步数，不是轨迹数
  min_size: ${rollout_batch_size}
  train_steps_per_env_step: 1
  use_rollout_batch_size: true
  sampling_mode: "step"  # 必须是step模式
  sample_method: lifo
  storage_mode: tokens_only
  lazy_tokenization: false
```

### 3. Old Prob Mode配置

使用StepEnvManager时，`old_prob_mode`应该设置为`step`：

```yaml
old_prob_mode: step       # 与StepEnvManager配合使用
old_prob_compute: trainer # 保持不变
```

## 优缺点分析

### 优点

1. **概念一致性**: 环境管理、数据存储、训练都在步级别
2. **灵活采样**: 可以从不同轨迹的不同步骤组合批次
3. **细粒度控制**: 每步可以有不同的处理逻辑
4. **天然支持step模式**: 不需要额外的mask处理逻辑

### 缺点

1. **内存开销**: 每步都需要存储完整上下文
2. **数据冗余**: 历史信息在多个样本中重复存储
3. **批次组织复杂**: 需要考虑步骤之间的依赖关系

## 适用场景

StepEnvManager适合以下场景：

1. **长轨迹任务**: 轨迹很长，需要细粒度的信用分配
2. **稀疏奖励**: 需要通过中间步骤的学习来引导
3. **复杂决策**: 每步的决策需要不同的处理
4. **部分可观察**: 需要从历史中学习状态表示

对于FrozenLake这样的短轨迹任务，使用修改后的TrajEnvManager可能更简单高效。

## 注意事项

1. **自动检测**: `detect_manager_type_from_config`函数会根据env_manager_cls自动选择合适的ReplayBuffer类型
2. **容量设置**: StepReplayBuffer的capacity是步数，而不是轨迹数，需要相应调大
3. **采样效率**: 步级别采样可能导致相关性较强的样本被一起采样，影响训练稳定性
