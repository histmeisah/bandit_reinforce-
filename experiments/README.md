# Bandit REINFORCE++ Experiments

## 实验列表

| 配置名 | 算法 | 用途 |
|--------|------|------|
| `exp1_bandit_ts_dapo` | Bandit(TS) + REINFORCE++ | 主实验 |
| `exp2_baseline_no_bandit` | REINFORCE++ (无bandit) | 消融对照 |
| `exp3_bandit_ucb_dapo` | Bandit(UCB) + REINFORCE++ | UCB vs TS 对比 |

## 运行方式

```bash
# 默认运行 exp1 (Bandit TS)
bash experiments/run.sh

# 指定实验
bash experiments/run.sh exp1_bandit_ts_dapo
bash experiments/run.sh exp2_baseline_no_bandit
bash experiments/run.sh exp3_bandit_ucb_dapo
```

## 当前配置 (Debug 2 GPU)

- Model: `/mnt/project_modelware/zhaojian/models/pretrain/Qwen3-4B-Base`
- Data: `/mnt/project_modelware/zhaojian/weiyu/bandit_reinforce/ROLL/data/dapo_math_17k.jsonl`
- Encoder: `/mnt/project_modelware/zhaojian/models/qwen3_embed`
- GPUs: 2, batch_size: 16, max_steps: 10
- wandb: offline mode
