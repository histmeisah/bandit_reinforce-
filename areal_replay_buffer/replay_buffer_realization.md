## AReaL Replay Buffer 实现与移植指南

本文件总结 AReaL 中“Replay Buffer”（更准确地说是“序列缓冲 Sequence Buffer”）的核心设计、关键数据结构、并发与调度机制，以及如何将其思想移植到其他训练/推理框架中。所有小节均附带必要的代码片段与调用链说明。

注意：AReaL 的 Replay Buffer 并非传统 RL 中“随机采样/优先级采样”经验回放，而是面向 DFG（数据流图）的“键增量+就绪判定+FIFO 优先”的流水式缓冲，保障不同 RPC（模型函数调用）之间的数据依赖与并发调度。

---

### 1. 顶层结构与数据流

- 底层存储器：`_TensorDictSequenceBuffer`（列表实现、非线程安全，提供批量 put/amend/get/pop）
- 并发包装器：`AsyncIOSequenceBuffer`（`asyncio.Condition` + 槽位状态机 + 就绪掩码）
- 数据载体：`SequenceSample`（支持聚合、拆分、重排、就地增量 `update_`）
- 生产者：`FunctionExecutor.load_data`（从数据 worker 拉取元样本，put 入缓冲）
- 消费者：`ModelFunctionCall.run`（按 RPC 输入键就绪从缓冲取样本 -> 执行 -> 非终点 RPC 增量回写 amend / 终点清理）

数据在 DFG 节点间（RPC）以样本为单位“增量补齐键”，每个样本会被图中的每个 RPC 消费一次；当消费计数归零，自动回收槽位。

---

### 2. 关键数据结构

#### 2.1 `_ReplayEntry` 与 `SequenceSample`

用于保存单条目与其复用计数：

```24:31:realhf/system/buffer.py
@dataclass
class _ReplayEntry:
    reuses_left: int
    receive_time: float
    sample: SequenceSample
```

样本容器支持就地增量（合并新键、对齐元信息）：

```520:533:realhf/api/core/data_api.py
def update_(self, other: "SequenceSample"):
    self.keys = self.keys.union(other.keys)
    self.trailing_shapes.update(other.trailing_shapes)
    self.dtypes.update(other.dtypes)
    assert self.ids == other.ids, (self.ids, other.ids)
    if self.data is not None:
        self.data.update(other.data)
    self.seqlens.update(other.seqlens)
    self.metadata.update(other.metadata)
```

移植建议：
- 在其他框架定义等价的“样本容器”，要求能按键聚合/拆分，并支持就地增量合并；
- 保证 `ids` 是批内唯一键，用于跨 RPC 对齐与增量写回。

---

### 3. 底层存储 `_TensorDictSequenceBuffer`

职责：固定大小槽位数组，保存 `_ReplayEntry` 指针；维护每槽位“拥有的键”布尔掩码；提供批量 put/amend/get/pop。

核心字段与键掩码更新：

```44:74:realhf/system/buffer.py
def __init__(self, keys: List[str], max_size: int, reuses: int):
    self.__storage: List[_ReplayEntry] = [None for _ in range(max_size)]
    self.__has_keys = np.zeros((max_size, len(keys)), dtype=bool)
    self.__keys = keys
    self.__reuses = reuses

def _update_has_keys(self, indices: List[int]):
    for idx in indices:
        self.__has_keys[idx] = [
            k in self.__storage[idx].sample.keys for k in self.__keys
        ]
```

批量接口：

```75:114:realhf/system/buffer.py
def put_batch(self, indices: List[int], xs: List[SequenceSample]):
    for idx, x in zip(indices, xs):
        self.__storage[idx] = _ReplayEntry(
            reuses_left=self.__reuses,
            receive_time=time.time(),
            sample=x,
        )

def amend_batch(self, indices: List[int], xs: List[SequenceSample]):
    for idx, x in zip(indices, xs):
        self.__storage[idx].sample.update_(x)

def get_batch(self, indices: List[int]) -> List[_ReplayEntry]:
    res = []
    for idx in indices:
        r = self.__storage[idx]
        r.reuses_left -= 1
        res.append(r)
    return res

def pop_batch(self, indices: List[int]):
    res = []
    for idx in indices:
        r = self.__storage[idx]
        self.__storage[idx] = None
        self.__has_keys[idx] = False
        res.append(r)
    return res
```

移植要点：
- 用连续数组或向量保存条目指针，避免频繁拷贝；
- 将“键的拥有关系”编码为二维布尔阵列，便于上层并发层计算就绪；
- 为每条目维护 `reuses_left`，初始化为“需要被消费的 RPC 数”。

---

### 4. 并发与调度 `AsyncIOSequenceBuffer`

职责：在协程环境下管理槽位状态、键就绪掩码与消费完成标记；提供阻塞式“按 RPC 取样”API；在 `put/amend` 后通知等待的消费者。

初始化与就绪掩码：

```117:166:realhf/system/buffer.py
class AsyncIOSequenceBuffer:
    def __init__(self, rpcs: List[dfg.MFCDef], max_size: int):
        self._lock = asyncio.Condition(asyncio.Lock())
        self._is_being_put = np.zeros(max_size, dtype=bool)
        self._is_being_amended = np.zeros(max_size, dtype=bool)
        self._is_being_read = np.zeros(max_size, dtype=bool)
        self._is_idle = np.zeros(max_size, dtype=bool)
        self._is_empty = np.ones(max_size, dtype=bool)
        ...
        self._ready_for_rpcs = np.zeros((max_size, len(rpcs)), dtype=bool)
        self._completed_rpc = np.zeros((max_size, len(rpcs)), dtype=bool)
        self._rpc_data_keys = rpc_data_keys = list(
            set().union(*[rpc.input_keys for rpc in rpcs])
        )
        self._rpc_key_mask = np.stack(
            [np.array([k in rpc.input_keys for k in rpc_data_keys], dtype=bool)
             for rpc in rpcs], axis=1)
        self.__buffer = _TensorDictSequenceBuffer(
            keys=rpc_data_keys, max_size=max_size, reuses=len(rpcs)
        )
```

put（写入）与 amend（增量）：

```247:306:realhf/system/buffer.py
async def put_batch(self, samples: List[SequenceSample], birth_times: List[int] | None = None):
    async with self._lock:
        indices = np.where(self._is_empty)[0][:len(samples)]
        self._is_empty[indices] = False
        self._is_being_put[indices] = True
    self.__buffer.put_batch(indices, samples)
    if birth_times is None:
        self._birth_time[indices] = time.monotonic_ns() + np.arange(len(indices))
    else:
        self._birth_time[indices] = birth_times
    async with self._lock:
        self.__buffer._update_has_keys(indices)
        has_keys = self.__buffer._get_has_keys(indices)
        self._ready_for_rpcs[indices] = (has_keys[:, :, None] >= self._rpc_key_mask[None]).all(axis=1)
        self._is_being_put[indices] = False
        self._is_idle[indices] = True
        self._buf_size += len(samples)
        self._lock.notify(len(self._rpc_names))

async def amend_batch(self, indices: List[int], samples: List[SequenceSample]):
    async with self._lock:
        await self._lock.wait_for(lambda: (self._is_idle[indices] | self._is_being_amended[indices]).all())
        self._is_idle[indices] = False
        self._is_being_amended[indices] = True
        self._n_amenders[indices] += 1
    self.__buffer.amend_batch(indices, samples)
    async with self._lock:
        self.__buffer._update_has_keys(indices)
        has_keys = self.__buffer._get_has_keys(indices)
        self._ready_for_rpcs[indices] = (has_keys[:, :, None] >= self._rpc_key_mask[None]).all(axis=1)
        self._n_amenders[indices] -= 1
        self._is_being_amended[indices] = self._n_amenders[indices] > 0
        self._is_idle[indices] = ~self._is_being_amended[indices]
        if self._is_idle[indices].any():
            self._lock.notify(len(self._rpc_names))
```

按 RPC 取样（阻塞等待就绪，FIFO 选最早）：

```337:408:realhf/system/buffer.py
def _can_do_rpc(self, rpc: dfg.MFCDef) -> bool:
    rpc_idx = self._rpc_names.index(rpc.name)
    ready_indices = np.nonzero((self._is_idle | self._is_being_read)
                               & self._ready_for_rpcs[:, rpc_idx]
                               & ~self._completed_rpc[:, rpc_idx])[0]
    return len(ready_indices) >= rpc.n_seqs

async def get_batch_for_rpc(self, rpc: dfg.MFCDef) -> Tuple[List[int], SequenceSample]:
    rpc_idx = self._rpc_names.index(rpc.name)
    async with self._lock:
        while not self._can_do_rpc(rpc):
            await self._lock.wait()
        ready_indices = np.nonzero((self._is_idle | self._is_being_read)
                                   & self._ready_for_rpcs[:, rpc_idx]
                                   & ~self._completed_rpc[:, rpc_idx])[0]
        indices = ready_indices[np.argsort(self._birth_time[ready_indices])[: rpc.n_seqs]]
        self._is_idle[indices] = False
        self._is_being_read[indices] = True
        self._n_readers[indices] += 1
    entries = self.__buffer.get_batch(indices)
    pop_indices = [idx for idx, entry in zip(indices, entries) if entry.reuses_left == 0]
    if len(pop_indices) > 0:
        self.__buffer.pop_batch(pop_indices)
    async with self._lock:
        self._n_readers[indices] -= 1
        self._is_being_read[indices] = self._n_readers[indices] > 0
        self._is_idle[indices] = self._n_readers[indices] == 0
        self._completed_rpc[indices, rpc_idx] = True
        self._is_empty[pop_indices] = True
        self._is_idle[pop_indices] = False
        self._completed_rpc[pop_indices] = False
        self._ready_for_rpcs[pop_indices] = False
        self._buf_size -= len(pop_indices)
        if self._is_idle[indices].any():
            self._lock.notify(len(self._rpc_names))
    return indices, SequenceSample.gather([e.sample for e in entries], keys=rpc.input_keys)
```

移植要点：
- 用“槽位状态 + 计数器”的细粒度状态机替代全局粗锁；
- 就绪判定采用“样本拥有键 ≥ RPC 需求键”的布尔掩码；
- 读取时按最早 `birth_time` 实现 FIFO，避免饥饿；
- 每个 RPC 对同一条目只消费一次，用 `_completed_rpc` 防重复；条目在 `reuses_left==0` 时回收。

---

### 5. 生产者/消费者调用链

生产者：数据加载写入缓冲（每 DP rank 提供一部分 meta 样本，按需 shuffle）：

```121:211:realhf/system/function_executor.py
async def load_data(self, buffer_id: int):
    ...
    resps = await self.stream.call_async(handlers=[f"__data{dp_idx}__" ...], handle_type="fetch", datas=[buffer_id ...])
    ...
    all_data += x.meta_sample.unpack(); all_birth_time += x.birth_times
    ...
    buffer_indices = await buffer.put_batch(all_data, all_birth_time)
    ...
```

消费者：按 RPC 取样 -> 执行 -> 非终点增量写回 / 终点清理：

```500:509:realhf/system/model_function_call.py
async def run(self, buffer_id: int):
    consumed = 0
    while True:
        buf_indices, sample = await self.buffers[buffer_id].get_batch_for_rpc(rpc)
        await self.run_step(buf_indices, sample, buffer_id)
        consumed += sample.bs
        if all(consumed >= c.n_seqs for c in rpc.all_successors()):
            break
```

```474:485:realhf/system/model_function_call.py
if rpc.is_dst:
    async with ctrl.lock:
        ctrl.ids_to_clear = ctrl.ids_to_clear.union(sample.ids)
    await ctrl.train_count.put(1)
else:
    logger.info(f"Amending RPC {rpc.name} output keys: {res.keys}")
    await self.buffers[buffer_id].amend_batch(buf_indices, res.unpack())
```

缓冲创建与容量：

```283:289:realhf/system/master_worker.py
self.__seqbuffers = [
    AsyncIOSequenceBuffer(
        self.__model_rpcs,
        max_size=int(os.getenv("REAL_MASTER_BUFFER_SIZE", str(int(1e7))))
    ) for _ in range(self._n_datasets)
]
```

---

### 6. 与“传统 RL Replay Buffer”的区别

- 采样策略：非随机/优先级抽样；而是基于“键就绪”的阻塞式批量消费；
- 数据形态：同一条样本在不同阶段被增量补齐键（logits/rewards/...），统一留在同一容器上；
- 生命周期：每条样本对每个 RPC 消费一次（`reuses_left = #RPCs`），完成后立即回收；
- 并发模型：多 RPC 协程并发，通过条件变量与状态布尔阵列协调。

---

### 7. 可移植实现步骤（到你的框架）

1) 定义“样本容器”（等价于 `SequenceSample`）
- 要求：键集合、`ids` 唯一、`seqlens/trailing_shapes/dtypes/data/metadata` 对齐；
- 提供 `gather/unpack/reorder/update_`；确保合并时仅追加或覆盖同名键。

2) 实现底层存储器 `_TensorDictSequenceBuffer` 等价物
- 固定容量的槽位数组，元素为 {reuses_left, birth_time, sample}；
- 维护二维 `has_keys[slot, key]`；
- 批量 `put/amend/get/pop`，`get` 时仅改 `reuses_left`，不复制大张量。

3) 实现并发包装器（等价 `AsyncIOSequenceBuffer`）
- 槽位状态：empty/idle/being_put/being_amended/being_read；
- RPC 维度：`ready_for_rpcs[slot, rpc]` 与 `completed_rpc[slot, rpc]`；
- 键就绪判定：`ready = (has_keys >= rpc_key_mask).all(axis=keys)`；
- 取样策略：阻塞等待可用数量，按 `birth_time` 选最早 N 条；
- 回收：当条目 `reuses_left == 0` 时复位 `empty` 并清除 RPC 标记。

4) 嵌入训练执行流
- 生产者：数据加载后 `put_batch`；
- 消费者：按拓扑并发调用 `get_batch_for_rpc` -> 执行 -> 返回新增键则 `amend_batch`；
- 终点 RPC：累积 `ids_to_clear`，在合适时机同步清理下游缓存/GPU 存储。

5) 参数与监控
- 容量通过环境变量或配置注入；>95% 触发告警；
- 日志中打印各阶段吞吐、就绪率与等待时间便于优化。

---

### 8. 关键不变量与并发注意事项

- 槽位不变量：`is_empty + (is_being_put|is_being_amended|is_being_read|is_idle) == 1`；
- 读写者计数：`_n_amenders/_n_readers` 与对应状态一致；
- 任何 `is_empty` 的槽位不应持有 `ready_for_rpcs/completed_rpc` 标记；
- `ids` 必须全局唯一，生产者侧要去重；
- `amend_batch` 只能对已存在槽位进行；
- 取样返回后，必须在 RPC 维度设置 `completed_rpc[indices, rpc] = True`，防止重复消费。

---

### 9. 常见坑与修复建议

- 文案不一致：`BufferFull` 的提示里默认容量“1M”与 `master_worker` 中默认 `1e7` 不一致，移植时注意统一参数与文案；
- 键不匹配：放入/增量后需更新键掩码并重算就绪，防止早读或误读；
- 饥饿与乱序：强制用 `birth_time` 升序选择，避免新数据压制老数据；
- 大张量复制：底层仅存指针，聚合/拆分在 `SequenceSample` 中完成，避免冗余复制；
- 资源追踪：若跨 GPU 重分发，需额外的“存储追踪器”维护数据归属（参考 `RedistribPlanner/GlobalStorageTracker`）。

---

### 10. 最小“伪代码接口”参考

```python
# 用户框架中的抽象接口（伪代码）
class Sample:
    ids: List[str]
    keys: Set[str]
    data: Dict[str, Tensor]
    seqlens: Dict[str, List[List[int]]]
    def update_(self, other: 'Sample'): ...
    @staticmethod
    def gather(samples: List['Sample'], keys: List[str]): ...

class Storage:
    def put_batch(indices: List[int], xs: List[Sample]): ...
    def amend_batch(indices: List[int], xs: List[Sample]): ...
    def get_batch(indices: List[int]) -> List[Entry]: ...  # decref reuses_left
    def pop_batch(indices: List[int]): ...

class AsyncBuffer:
    def __init__(self, rpcs: List[RPC], max_size: int): ...
    async def put_batch(self, xs: List[Sample], birth_times: Optional[List[int]]): ...
    async def amend_batch(self, indices: List[int], xs: List[Sample]): ...
    async def get_batch_for_rpc(self, rpc: RPC) -> Tuple[List[int], Sample]: ...
```

---

### 11. 参考文件与位置

- `realhf/system/buffer.py`：缓冲核心实现（存储器与并发包装器）
- `realhf/api/core/data_api.py`：`SequenceSample` 数据结构与操作
- `realhf/system/function_executor.py`：生产者（数据加载 put）与训练执行调度
- `realhf/system/model_function_call.py`：消费者（按 RPC 取样、执行、增量回写）
- `realhf/system/master_worker.py`：缓冲创建与容量配置




