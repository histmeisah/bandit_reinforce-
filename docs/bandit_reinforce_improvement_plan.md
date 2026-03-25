# Bandit REINFORCE++ 改进计划

> 日期：2026-03-14
> 分支：freshper-v0314

## 一、当前系统概述

Bandit REINFORCE++ 是一个双层优化系统：
- **外层**：NeuralLinearUCB 上下文 Bandit 根据问题 embedding 自适应选择最优 prompt
- **内层**：REINFORCE++ 训练 LLM 策略
- **编码器**：支持 SentenceTransformer（纯文本）和 Qwen3-VL-Embedding（多模态统一编码）
- **经验回放**：FreshPER 优先级回放与 Bandit 互补

## 二、已完成的工作

| 模块 | 状态 | 说明 |
|------|------|------|
| 算法层 (NeuralLinearUCB, BanditActor, EncoderActor 等) | ✅ 完成 | 8 个文件，从旧 ROLL 移植 |
| MathBanditEnv | ✅ 完成 | 支持多模态观测适配 |
| BanditAgenticPipeline | ✅ 完成 | super().__init__() 前创建 actor |
| 多模态编码器 | ✅ 完成 | EncoderActor 支持 sentence_transformers / qwen3_vl 双后端 |
| 实验配置 | ✅ 完成 | bandit_math_reasoning + bandit_vlm_reasoning |
| 环境注册 + AgenticConfig | ✅ 完成 | roll_math_bandit, bandit_config 字段 |

---

## 三、改进方向（按优先级排序）

### P0-1：GRPO 版本扩展 — Bandit 作为通用插件

**现状**：目前只实现了 REINFORCE++ 版本，Bandit 和具体 RL 算法耦合度低。

**目标**：Bandit prompt 选择机制作为通用插件，可以和 GRPO、PPO 等任意算法组合。

**核心思路**：
- GRPO 天然支持 group contrast（同一问题多个 response 计算相对优势）
- Bandit 为同组的不同 response 选择不同 prompt → 不同 prompt 的 response 在 group 内直接对比
- 这比 REINFORCE++ 的 0/1 reward 提供了更丰富的梯度信号

#### 调研结论

**GRPO 在 ROLL 中的实现机制**：

GRPO 通过配置 `adv_estimator: "grpo"` + `num_return_sequences_in_group: 8` 激活。核心流程：

```
64个prompt × 8个response/prompt = 512个样本
    ↓ reward_postprocess()
自动设置 norm_mean_type="group", norm_std_type="group"
    ↓ reward_norm()
reshape [512] → [64, 8], 每组内: (reward - group_mean) / group_std
    ↓ compute_reinforce_return()
组内归一化后的 reward 作为 advantage
```

**关键发现**：
- GRPO 不需要 Critic 模型，用 group mean 作 baseline
- Advantage 在 group 内做相对排名，不是绝对值
- 代码位置：`roll/utils/functionals.py` 的 `reward_norm()` (L561-597) 和 `compute_advantage()` (L774-849)

**Bandit × GRPO 交互模式（两种方案）**：

| 方案 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| **A：组内同 prompt** | 同一 group 的 8 个 response 都用 bandit 选的同一个 prompt | 和原版 GRPO 一致，variance reduction 不变 | Bandit 学习速度和 REINFORCE++ 一样（每题只试一个 prompt） |
| **B：组内异 prompt** | 同一 group 的 8 个 response 用不同 prompt（bandit 为每个 response 独立选） | Bandit 能在一题内同时对比多个 prompt，学习极快 | 破坏了 GRPO 的 variance reduction 前提（response 来自不同条件） |

**推荐方案 A**：保持 GRPO 语义不变，Bandit 只在 group 粒度选 prompt。不需要新 Pipeline，只需 `adv_estimator: "grpo"` 配置切换。

---

### P0-2：Bandit 架构改进 — Thompson Sampling + 共享 Backbone

**现状**：NeuralLinearUCB 为每个 arm 维护独立网络，arm 之间不共享特征学习。UCB 探索随数据增多快速衰减。

#### 调研结论：Thompson Sampling

##### 论文溯源

Neural Linear + Thompson Sampling **不是一篇独立论文**，而是两项工作的组合：

| 组件 | 论文 | 发表 | 贡献 |
|------|------|------|------|
| Linear Thompson Sampling | Agrawal & Goyal, *"Thompson Sampling for Contextual Bandits with Linear Payoffs"* | **ICML 2013** | 首次为 contextual TS 提供理论保证，regret bound Õ(d^{3/2}√T) |
| Neural Linear 方法 | Riquelme, Tucker & Snoek, *"Deep Bayesian Bandits Showdown"* | **ICLR 2018** | 提出 Neural Linear 并实证验证为最可靠方法 |
| NeuralLinearUCB（当前基线） | Xu et al., *"Neural Contextual Bandits with Deep Representation and Shallow Exploration"* | **ICLR 2022** | 我们当前的 UCB 版本的理论来源 |
| Neural Thompson Sampling | Zhang et al., *"Neural Thompson Sampling"* | **ICLR 2021** | 全梯度 TS，理论更强但计算更贵 |

##### Neural Linear 算法定义（来自 Riquelme et al., ICLR 2018）

```
Neural Linear = 神经网络学习特征表示 φ(x) + 最后一层贝叶斯线性回归 + Thompson Sampling
```

算法步骤：
1. **特征提取**：网络前 L-1 层: x → φ(x) ∈ R^d
2. **Per-arm 后验维护**：对每个 arm a 维护贝叶斯线性回归后验
   ```
   初始化:  A_a = λ·I (d×d 精度矩阵),  b_a = 0 (d 维充分统计量)

   观测到 (φ, reward) 后更新:
     A_a ← A_a + φ · φᵀ           # 精度矩阵更新
     b_a ← b_a + reward · φ       # 充分统计量更新
     μ_a = A_a⁻¹ · b_a            # 后验均值 (对应线性权重的 MAP 估计)
     Σ_a = ν² · A_a⁻¹             # 后验协方差 (ν 控制探索强度)
   ```
3. **Thompson Sampling 选 arm**：
   ```
   对每个 arm a:
     w̃_a ~ N(μ_a, Σ_a)           # 从后验分布采样一组权重
     r̃_a = w̃_aᵀ · φ(x)           # 用采样权重预测 reward
   选择 a* = argmax_a r̃_a          # 选预测最高的 arm
   ```
4. **周期性重训网络**：每隔若干步重训神经网络，更新 φ(x)

##### 为什么选择 Neural Linear + TS 而非其他方法

Riquelme et al. (ICLR 2018) 对比了 9 种深度贝叶斯方法配合 Thompson Sampling：

| 方法 | 思路 | 实验表现 | 问题 |
|------|------|----------|------|
| **Neural Linear** | NN特征 + 最后一层贝叶斯 | **最稳定，综合最优** | — |
| Variational Inference | 全网络变分推断 | 差 | **不确定性估计在线上收敛太慢** |
| MC Dropout | Dropout近似后验 | 不稳定 | 不确定性质量差 |
| Bootstrapped DQN | 多bootstrap网络 | 部分好 | 不一致 |
| BBB (Bayes by Backprop) | 全网络权重分布 | 一般 | 计算贵 |
| Linear Full Posterior | 纯线性贝叶斯 | 线性问题最优 | **非线性问题差** |

核心发现：*"在线决策场景中，许多在监督学习中成功的方法反而表现不佳。缓慢收敛的不确定性估计难以适应在线设置。"*

Neural Linear 胜出原因：**只在最后一层做贝叶斯 → 不确定性估计收敛快**，同时神经网络提供非线性特征表达能力。

##### 与当前 UCB 方法的核心对比

| | NeuralLinearUCB（当前） | Neural Linear + TS（推荐） | NeuralTS（全梯度） |
|---|---|---|---|
| 探索方式 | 确定性 UCB 上界 | **从后验分布随机采样** | 从后验分布随机采样 |
| 选 arm 公式 | `μφ + α√(φᵀA⁻¹φ)` | `w̃ᵀφ, w̃~N(μ,ν²A⁻¹)` | 类似但用全梯度 |
| 特征空间 | 最后隐层 φ(x) | 最后隐层 φ(x) | 全网络梯度 ∇f |
| 精度矩阵 | d×d (128×128) | d×d (128×128) | p×p (~50K×50K) |
| 每步开销 | O(d²) | O(d²) | O(p) ~ O(p²) |
| 渐近最优 | 否 | **是**（匹配 Lai-Robbins 下界） | 是 |
| 探索特性 | 对不确定的 arm 一律加分 | **探索与"该 arm 最优概率"成正比** | 同左 |
| 改动量 | 基线 | **~50 行** | 大幅重写 |

##### 实现方案

当前代码已维护 `A_inv`（即 A_a⁻¹），只需额外维护 `b_a`，然后将选择逻辑从 UCB 改为后验采样：

```python
# 当前 UCB（确定性选择）:
ucb_score = predicted_reward + alpha * sqrt(φᵀ · A_inv · φ)
arm = argmax(ucb_scores)

# 改为 Thompson Sampling（随机选择）:
mu_a = A_inv @ b_a                        # 后验均值
w_sampled = N(mu_a, nu^2 * A_inv)         # 从后验采样权重
r_sampled = w_sampled @ phi               # 采样 reward
arm = argmax(r_sampled)                   # 选最高的
```

#### 调研结论：共享 Backbone + Prompt Embedding

**两种架构选择**：

| 方案 | 架构 | 优势 | 适用场景 |
|------|------|------|----------|
| **A: 共享 Backbone + Per-Arm Head** | `φ(context)` → `w_a^T φ(x)` per arm | 简单，接近当前实现 | arm 数量固定 |
| **B: Context + Prompt 联合编码** | `f(context_emb ⊕ prompt_emb) → reward` | **新 prompt 可 zero-shot 泛化** | arm 需要动态增减 |

**相关工作**：
- **TRIPLE (NeurIPS 2024)**：将 prompt 嵌入向量空间，训练 `g_θ: prompt_emb → reward` 实现 prompt 间信息共享
- **PAK-UCB (arXiv:2410.13287)**：用 CLIP/RoBERTa 编码 prompt 作为 arm 特征，per-arm kernel 回归

**推荐**：先实现方案 A（共享 Backbone + TS），后续再考虑方案 B（Prompt Embedding）。

---

### P1-1：框架性能优化 — 降低 Bandit 在 Rollout 关键路径上的开销

**现状**：每次 `reset()` 调用两次 Ray RPC（encode + select_arm），`step()` 调用一次（update）。均为同步阻塞。

#### 调研结论

**Ray RPC 开销参考**：

| 机制 | 每次调用延迟 |
|------|-------------|
| Ray actor `.remote()` + `ray.get()` | ~0.5-2 ms |
| Ray Compiled Graph | ~0.05 ms（需静态 DAG，不适合我们） |
| gRPC baseline | ~0.1-0.3 ms |

**优化方案（按收益排序）**：

#### 1. 合并 encode + select_arm 为一次 RPC（省 ~1ms/reset）

在 BanditActor 上新增 `encode_and_select()` 方法，将两次 RPC 合并为一次：

```python
# BanditActor 新增方法
def encode_and_select(self, data):
    context_bytes = self.encoder_actor.encode(data)  # 本地调用
    return self.select_arm(context_bytes), context_bytes
```

或者让 EncoderActor 和 BanditActor 合并为一个 actor。

#### 2. Fire-and-forget update（省 ~1ms/step）

`step()` 中的 update 不等待返回：

```python
# 当前（阻塞）:
ray.get(bandit_actor.update.remote(arm, context, reward))

# 改为（非阻塞）:
ref = bandit_actor.update.remote(arm, context, reward)
self._pending_refs.append(ref)  # 后台线程定期 drain
```

**注意**：不能完全不管 ObjectRef，否则会内存泄漏。需要一个后台 drainer 线程定期调用 `ray.wait()` 清理。

#### 3. 并发组分离读写（进一步优化）

```python
@ray.remote(concurrency_groups={"read": 10, "write": 1})
class BanditActor:
    @ray.method(concurrency_group="read")
    async def encode_and_select(self, data): ...  # 10 路并发读

    @ray.method(concurrency_group="write")
    async def update(self, arm, context, reward): ...  # 串行写
```

读（encode_and_select）允许多路并发，写（update）串行化防止竞态。

**优先级**：方案 1 > 方案 2 > 方案 3。前两个改动各省 ~1ms，对大规模 rollout 影响显著。

---

### 暂缓项

| 方向 | 暂缓理由 |
|------|----------|
| Process Reward / Reward Shaping | 意义不大，数学题 outcome reward 已足够 |
| 训练阶段感知（context 加 step） | 有价值但不紧急，GRPO group contrast 能部分缓解 |
| Prompt 进化 / 动态生成 | 工程量大，后续考虑 |
| FreshPER 优先级联动 | 需要先完成 GRPO 扩展再评估 |
| Sliding window / 经验衰减 | 和 Thompson Sampling 结合考虑 |

---

## 四、实施路线图

### Phase 1：Thompson Sampling 替换 UCB（改动最小，收益最大）
- 改 `neural_linear_ucb.py` 的 `select_arm()`，从 UCB 改为后验采样
- 新增 `NeuralLinearTS` 类（或在 NeuralLinearUCB 中加 `use_thompson_sampling` 开关）
- 维护 per-arm 的 `mu_a = A_inv @ b_a` 用于后验均值
- ~50 行代码改动

### Phase 2：RPC 优化
- 合并 encode + select_arm 为一次 RPC
- update 改为 fire-and-forget + 后台 drain
- ~30 行代码改动

### Phase 3：GRPO 兼容
- 验证 `adv_estimator: "grpo"` + `roll_math_bandit` 的配置组合
- 确保 group 内所有 response 使用同一个 bandit 选定的 prompt
- 可能需要在 MathBanditEnv 中用 group_seed 保证同组一致性
- 新增 GRPO 版本的实验配置

### Phase 4：共享 Backbone
- 将 per-arm 独立网络改为共享 backbone + per-arm linear head
- 所有 arm 共享表示学习，提高数据效率
- 后续可扩展为 Prompt Embedding 方案

---

## 五、目标架构演进

```
当前:
  Problem → Encoder → NeuralLinearUCB(独立arm, UCB探索) → Prompt → REINFORCE++

Phase 1-2:
  Problem → Encoder → NeuralLinearTS(独立arm, Thompson Sampling) → Prompt → REINFORCE++
                       (encode+select 合并为1次RPC, update 异步)

Phase 3:
  Problem → Encoder → NeuralLinearTS → Prompt → GRPO / REINFORCE++ / PPO
                                                 (Bandit 作为通用插件)

Phase 4:
  Problem → Encoder(Qwen3-VL) → SharedBackbone + TS → Prompt
                                  (arm 共享表示学习)    ↓
                                                  GRPO / REINFORCE++ / PPO
```

---

## 六、参考文献

- [Deep Bayesian Bandits Showdown (Riquelme et al., ICLR 2018)](https://openreview.net/forum?id=SyYe6k-CW)
- [Neural Thompson Sampling (Zhang et al., ICLR 2021)](https://arxiv.org/abs/2010.00827)
- [Neural Contextual Bandits with Deep Representation and Shallow Exploration (Xu et al., ICLR 2022)](https://arxiv.org/pdf/2012.01780)
- [TRIPLE: Best Arm Identification for Prompt Learning (NeurIPS 2024)](https://arxiv.org/html/2402.09723v3)
- [PAK-UCB: Prompt-Aware Contextual Bandit (arXiv:2410.13287)](https://arxiv.org/html/2410.13287)
- [Thompson Sampling for Contextual Bandits with Linear Payoffs (Agrawal & Goyal)](https://proceedings.mlr.press/v28/agrawal13.pdf)
- [Ray AsyncIO / Concurrency for Actors](https://docs.ray.io/en/latest/ray-core/actors/async_api.html)
- [DSPy: Programming—not Prompting—Language Models](https://dspy.ai/)
- [Qwen3-VL-Embedding (arXiv:2601.04720)](https://arxiv.org/abs/2601.04720)
