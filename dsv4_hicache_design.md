# DeepSeek-V4-Flash HiCache 分层 KV 缓存详细设计

版本：v1.0 ｜ 日期：2026-09 ｜ 适用范围：KT 推理引擎（SGLang 内核）× DeepSeek-V4-Flash 混合架构模型

---

## 1. 概述

### 1.1 背景

大模型推理的 prefill 阶段需要将输入序列逐层计算为 KV 缓存，长上下文请求的 prefill 耗时随 token 数线性增长。当多个请求共享相同前缀（system prompt、agent 会话历史、多轮问答上下文）时，该前缀的 KV 缓存完全一致，缓存并复用这些 KV 可以消除重复计算。

SGLang 通过 RadixAttention 利用 GPU 显存空闲空间缓存前缀 KV（L1）；HiCache 将这一思想扩展到主机内存（L2）与分布式存储（L3），形成类似 CPU 多级缓存的三层 KV 缓存体系。DeepSeek-V4-Flash 采用混合注意力架构（全注意力层 + 滑动窗口层 + 压缩域索引器），其 KV 缓存由多个异构子池组成，HiCache 需要针对该多池结构做专门的镜像、传输与生命周期设计——这是本文档的核心内容。

### 1.2 设计目标

1. **容量扩展**：KV 有效容量从 GPU 显存（131k tokens）扩展到 host 内存（2×，262k tokens），突破单副本并发会话数与单会话上下文长度的天花板；
2. **性能收益**：被驱逐到 host 的前缀命中后，以页粒度回迁替代全量重算，TTFT 从数十秒降至亚秒级；
3. **数据无损**：跨"迁出 → 驱逐 → 回迁"完整生命周期的 KV 数据字节级忠实，确定性检索任务（needle/marker）精确命中；
4. **生命周期一致**：SWA 双池与 full 池的分配/释放严格同步，任何调度路径下账本守恒；
5. **可诊断**：内置字节级校验探针与占用位图幂等闸，支持运维定位。

### 1.3 术语表

| 术语 | 含义 |
|---|---|
| L1 / L2 / L3 | GPU 显存池 / host 内存池 / 分布式存储（本文不涉及 L3） |
| full 域 | 全 token 域索引，radix 树与调度器使用的统一地址空间 |
| c4 / c128 | 4-token / 128-token 压缩域的全注意力层 KV 子池 |
| SWA | Sliding Window Attention 滑动窗口层（window=128） |
| indexer | 压缩域索引器（256-token 页粒度的索引 KV） |
| 迁出（backup / write-back） | L1 → L2 的 KV 拷贝 |
| 回迁（restore / load） | L2 → L1 的 KV 拷贝 |
| 影子槽（shadow slot） | swa 槽位与 full 槽位恒等映射的双池统一方案 |
| 页（page/block） | host 池分配粒度，256 个 full token |

---

## 2. 总体架构

### 2.1 分层结构

```
┌─────────────────────────────────────────────────────────────┐
│  请求 (tokens)                                               │
│     │                                                        │
│     ▼                                                        │
│  HiRadixTree 前缀匹配（local match，纯元数据，无数据拷贝）      │
│     │                                                        │
│     ├── L1 命中 → 直接 prefill（GPU radix 复用）               │
│     ├── L2 命中 → load(): host 页粒度回迁 → prefill            │
│     └── 未命中 → 全量 prefill                                 │
│                                                              │
│  请求完成 → write_through: write() → start_writing()          │
│             L1 → L2 迁出（独立 write_stream 异步执行）          │
│                                                              │
│  L1 池满 → LRU 驱逐（evict）：先确保已迁出 L2，再释放 L1 槽位    │
└─────────────────────────────────────────────────────────────┘
  L1: DeepSeekV4TokenToKVPool        （GPU，131k full tokens）
  L2: DeepSeekV4TokenToKVPoolHost    （host，262k full tokens = 2×L1）
```

- **L1 与 L2 实例私有**，L3（Mooncake/3FS/NIXL 等分布式后端）为集群共享扩展位，本设计预留接口但默认关闭（DSV4 当前不支持 L3 后端）。
- HiRadixTree 的每个节点记录一段连续 token 的 KV 存储位置：`node.value`（L1 槽位）与 `node.host_value`（L2 槽位）双址。树侧元数据不感知 DSV4 的压缩域换算，**全部索引交换保持在 full token 域**。

### 2.2 核心工作流

| 阶段 | 动作 | 执行路径 |
|---|---|---|
| local match | 请求 token 序列按页（256）粒度在 HiRadixTree 上匹配，返回"L1 连续前缀 + L2 连续前缀" | 纯树遍历，微秒级 |
| 回迁 restore | L2 命中段按页粒度 H2D 拷贝到新分配的 L1 槽位，随后 prefill 只重算不命中尾部 | cache_controller.load() |
| prefill | 分块（chunked-prefill 2048）计算，产出 KV 写入 L1 | 常规调度 |
| 迁出 write-back | 请求完成（write_through 策略）即入队，异步 D2H 拷贝到 host 槽位 | cache_controller.write() |
| 驱逐 evict | L1 池满时 LRU 选树节点，已迁出者直接释放 L1 槽位 | HiRadixCache evict |

### 2.3 DSV4 适配层总览

HiCache 通用框架（HiRadixTree / cache_controller / 通用 host 池）之上，DSV4 新增三个专属组件：

| 组件 | 文件 | 职责 |
|---|---|---|
| 设备多池容器 | `srt/mem_cache/deepseekv4_memory_pool.py` | 管理 c4/c128/indexer/SWA/状态池五个子池，暴露统一 backup/load 接口 |
| host 镜像池 | `srt/mem_cache/deepseekv4_memory_pool_host.py` | 256-token 块粒度分配，五域镜像，页粒度传输 |
| SWA 影子槽分配器 | `srt/mem_cache/swa_memory_pool.py` | 双池生命周期统一（恒等映射） |
| 状态池 | `srt/mem_cache/compress_state.py` | 压缩器评分状态的 ring 行管理 |

---

## 3. L1 设备端 KV 多池结构

### 3.1 混合架构与子池划分

DeepSeek-V4-Flash 的 43 层注意力分为三类（由 `compression_ratios` 描述）：

- **c4 全注意力层**：KV 以 4-token 压缩槽位存储（每槽 584B = 576B value + 8B scale）；
- **c128 全注意力层**：KV 以 128-token 压缩槽位存储（页 = 2 slot，padded 1728B）；
- **SWA 层**：滑动窗口 128，全分辨率 KV 独立成池（页 256 token）。

配套两个辅助域：

- **c4_indexer**：c4 层的索引器 KV，256-token 页粒度（1 页 = 64 个 c4 slot）；
- **压缩状态池**：压缩器（compressor）的评分状态行（KV+score 双半行），c4 与 c128 各一个池，indexer 另有独立状态池。

`DeepSeekV4TokenToKVPool`（deepseekv4_memory_pool.py L356）聚合全部子池：

```
DeepSeekV4TokenToKVPool
 ├── swa_kv_pool            SWA 全分辨率 KV（swa_size × 256-token 页）
 ├── c4_kv_pool             c4 压缩 KV（c4_size slot，584B/slot）
 ├── c128_kv_pool           c128 压缩 KV（c128_size slot，页 = 2 slot padded）
 ├── c4_indexer_kv_pool     索引器 KV（c4_size/4 页，页粒度独立布局）
 ├── compress_state_pools   c4/c128 压缩状态行池
 └── indexer_compress_state_pools  indexer 状态行池
```

### 3.2 索引域换算

所有子池共用 full token 域地址，换算关系（整数除法）：

| 压缩域 | 槽位/页换算 | 说明 |
|---|---|---|
| c4 slot | `full_index // 4` | 每 slot 覆盖 4 个 full token |
| c128 slot | `full_index // 128` | 每 slot 覆盖 128 个 full token |
| indexer 页 | `full_index // 256` | 1 页 = 64 个 c4 slot |
| SWA 页 | `full_index // 256`（影子槽下 swa_loc ≡ full_loc） | 页 256 token |
| 状态行 | `state_loc = (swa_loc // 256) × ring_size + swa_loc % ring_size` | 见 3.4 |

### 3.3 池容量与实测规模

池容量由 `--max-total-tokens`（full token 域）与 `--swa-full-tokens-ratio` 推导：

- `full = max_total_tokens`，`swa = full × swa_full_tokens_ratio`；
- `c4_size = full // 4`，`c128_size = full // 128`；
- 状态池行数 = `swa_tokens // window_size × ring_size`（window_size=128），运行时按 256-token swa 页索引，实际占用约为容量一半，留有安全余量。

实测配置（`--max-total-tokens 131072 --swa-full-tokens-ratio 1.0`，单 TP rank）：

| 子池 | 容量 | 说明 |
|---|---|---|
| full / swa | 131072 / 131072 | full:swa = 1:1 |
| c4 | 32768 slot | = 131072/4 |
| c128 | 1024 slot | = 131072/128 |
| c4 状态池 | 8192 行 | = 1024 页 × ring 8 |
| c128 状态池 | 131072 行 | = 1024 页 × ring 128 |

### 3.4 压缩状态池（CompressStatePool）

压缩器为每个 token 维护评分状态（KV 与 score 拼接为一条定长行），采用 **ring 行语义**：

- 行定位公式（compress_state.py L147-153）：`state_loc = (swa_loc // swa_page_size) × ring_size + (swa_loc % ring_size)`，即 **每个 swa 页对应 ring_size 行的连续组**，页内槽位按 `swa_loc % ring_size` 在组内环形复用；
- ring 大小（`get_compress_state_ring_size`）：c4 域 = 8，c128 域 = 128（非投机解码）；
- 该"一页 = 一组 ring 行"的布局是 L2 状态镜像传输分组的设计依据（见 5.4）。

### 3.5 SWA 双池问题与影子槽

SWA 池与 full 池若独立记账（各自 free list）并靠 `full_to_swa_index_mapping` 映射表关联，系统即存在三份可漂移状态：full 账本、swa 账本、映射表。任何一条路径只更新其中一份（例如 HiCache 回迁只分配 full 池），生命周期即失衡。DSV4 的影子槽方案从结构上消除该问题，详见第 4 章。

---

## 4. 影子槽设计（Shadow Slot）

### 4.1 设计原则

**swa 槽位不再是独立资源，而是 full 槽位的"影子"**：`swa_loc ≡ full_loc`（恒等函数）。两池共享同一份分配决策，账本、映射、生命周期三者合一。

开关：环境变量 `SGLANG_SWA_SHADOW_SLOT=1`（swa_memory_pool.py L285-317），默认关闭时保持上游独立双池行为。

### 4.2 关键机制

| 机制 | 实现 | 效果 |
|---|---|---|
| 恒等映射 | 初始化时 `full_to_swa_index_mapping[i] = i` | 消除独立映射表状态；注意力内核按裸索引直读，无需哨兵值 |
| 单向分配 | `alloc / alloc_extend / alloc_decode` 只走 full 分配器，随分配标记占用 | swa 不再有自己的 free list |
| 账本镜像 | `available_size() / swa_available_size()` 直接读取 full 分配器 | 任意路径读数恒一致 |
| 释放幂等 | `free_swa()` 为 no-op（随 full 的 free 一起生效） | 不存在二次释放 |
| 树侧同步 | HiRadixCache 维护 `swa_evictable = evictable_size_`（lockstep 假设） | 树计数与池账本严格一致 |
| 容量约束 | 启动断言 `swa 池大小 ≥ full 池大小`（ratio ≥ 1.0） | 保证每个 full 槽位都有影子 |

### 4.3 与回迁路径的配合

HiCache 回迁只分配 full 域槽位（cache_controller.py L690-724：`full_attn_allocator.alloc`）。影子槽下该槽位即同时是 SWA 槽位，因此：

1. 回迁不再需要第二笔 swa 分配（独立双池模式下这正是生命周期失衡的源头）；
2. SWA 层的窗口 KV 随页粒度回迁一并恢复（见 5.4），解码阶段短窗注意力直接可读；
3. 回迁分配绕过了包装分配器的 `alloc()`，因此必须补一次占用位图标记 `_mark_occupied`（见 6.4），保证后续 `free()` 的幂等闸账实相符。

---

## 5. L2 Host 池设计（DeepSeekV4TokenToKVPoolHost）

### 5.1 索引语义与分配粒度

**对外索引一律使用 full token 域**（与 radix 树 `node.host_value` 同域，长度按 radix 页 256 对齐），池内部分别换算到五个压缩域做实际存储。

**分配粒度 = 1 个 256-full-token 块**，恰为各压缩域的公倍数：

```
1 块（256 full tokens）
  = 64 个 c4 slot        （64 × 4 = 256）
  = 2 个 c128 slot       （2 × 128 = 256）
  = 1 个 indexer 页      （256 token）
  = 1 组 SWA 页数据      （影子槽下 swa 页 ≡ full 页）
  = 每状态池 1 组 ring 行（每页 ring_size 行）
```

这保证**任何已分配段在所有压缩域都是 group-complete**——两个 radix 节点永远不会共享同一个 host 块内的某个压缩域槽位，从布局上杜绝跨节点数据串写。

### 5.2 块字节构成

每个 256-token 块的 host 字节数（`bytes_per_block`，所有镜像子池之和）：

```
bytes_per_block = 584B × 64 × c4_layer_num          # c4 域
                + c128_page_bytes × c128_layer_num   # c128 域（页 padded 1728B）
                + indexer_page_bytes × c4_layer_num  # indexer 域
                + swa_page_bytes × 1                 # SWA 域（phase-2 起）
                + Σ ring_size × row_bytes            # 状态池域（每池一组）
```

实测（DSV4-Flash 43 层，MXFP4 KV）：`bytes_per_block = 996864 B ≈ 0.95MB`。
host 池总容量 = blocks × bytes_per_block；实测 `--hicache-ratio 2` → 1024 blocks = 262144 full tokens（单 TP rank 约 1GB 级镜像总量，其中 SWA 镜像约 6.1GB、状态镜像约 11.6GB 量级，随层配置变化）。

### 5.3 内存来源与 NUMA/CXL

- 分配器通过 `get_allocator_from_storage(allocator_type, numa_node)` 接入统一 host 分配机制：默认 `HostTensorAllocator`；指定 `--hicache-numa-node` 时走 `libnuma numa_alloc_onnode`，可将池放置在以 NUMA 节点形式暴露的 **CXL 内存扩展器** 上；
- 分配后执行 `cudaHostRegister` 锁页，保证 kernel io 后端 DMA 传输路径稳定；
- 布局仅支持 `layer_first`（与 GPU 逐层计算天然对齐；`page_first` 变体为 L3 场景预留，DSV4 不启用）。

### 5.4 五域镜像与页粒度索引

host 池为每个子池维护镜像 buffer，**全部按 host 页域索引（`host_index // 256`）组织，而非设备槽位 1:1 映射**。原因：设备槽位在 backup 与 restore 之间会被复用（同一设备页先后承载不同逻辑页的数据），按设备域 1:1 镜像会跨请求串号；而 host 页域索引由 radix 树保证在驱逐/回迁周期内稳定绑定树节点。

各域传输形态：

| 域 | host 布局 | 传输粒度 |
|---|---|---|
| c4 | 线性（584B/slot，匹配 hisparse_transfer 内核的 64-slot 页布局） | 块内 64 slot 连续 |
| c128 | 页布局镜像（padded 1728B/页） | 页粒度（设备池无法走 c4 专用内核） |
| indexer | 页布局镜像 | 页粒度 |
| SWA | 页布局镜像（页 ≡ full 页，影子槽） | 页粒度 |
| 状态池 | 每 host 页一组 ring_size 连续行 | 页组粒度（恰好覆盖 compressor 读写行集合） |

---

## 6. 数据搬运设计（迁出 / 回迁）

### 6.1 控制器流程（cache_controller.py）

**迁出（write_through）**：

```
write(device_indices)                     # L644：host 池 alloc → 入 write_queue
  └─ start_writing()                      # L662：合并队列操作 → move_indices
       └─ write_stream（独立 CUDA 流，异步）
            └─ backup_from_device_all_layer(device_pool, host_idx, dev_idx, io_backend)
       └─ ack_write_queue（start/finish Event，供树侧确认落盘）
```

**回迁**：

```
load(host_indices)                        # L690：full_attn_allocator.alloc（仅 full 域）
  └─ _mark_occupied(device_indices)       # 占用位图补标记（幂等闸）
  └─ 入 load_queue → start_loading()
       └─ load_to_device_per_layer(...)   # 逐层 H2D，见 6.3
```

kernel io 后端下索引张量先 H2D（`move_indices`），使传输内核可在设备端按索引并行寻址。

### 6.2 索引切分（_split_indices）

传输前将 full 域索引一次性切分为五组视图（deepseekv4_memory_pool_host.py L355-374）：

```
dev_c4  = dev_full // 4      host_c4  = host_full // 4
dev_c128= dev_full // 128    host_c128= host_full // 128
dev_pages = dev_full // 256  host_pages = host_full // 256
```

切分结果只与本次传输的索引值绑定，不做跨调用记忆化（设备张量释放后重建可能复用同一地址，按 data_ptr 缓存会跨请求错位）。

### 6.3 迁出与回迁的逐域路径

**迁出 `backup_from_device_all_layer()`**（L441-491）：

| 域 | 传输内核/函数 | 说明 |
|---|---|---|
| c4 | `hisparse_offload_to_host`（JIT） | 专用 64-slot 页布局 D2H 内核 |
| c128 / indexer / SWA | `transfer_kv_all_layer_mla`（sgl_kernel） | 页粒度 all-layer 批量传输 |
| 状态池 | `_backup_states` | 按 host 页分组搬 ring 行 |

**回迁 `load_to_device_per_layer()`**（L571-608）：按 `layer_mapping` 逐层判断压缩比——

- SWA-only 层：`_restore_swa_layer()`（仅 SWA 域页传输）；
- c4 层：`hisparse_load_to_device`（H2D 专用内核）；
- 其余层：页粒度 H2D；
- 状态池：`_restore_states` 按 host 页组回写。

### 6.4 传输原则与守恒保障

1. **页粒度铁律**：设备端 buffer 是分页布局 `[num_pages, bytes_per_page_padded]`，页间有对齐填充；任何按线性 `[size, item]` 的扁平寻址都会越界或错位。c4 专用内核（584B 无填充线性布局）与页布局子池必须各走各的传输函数。
2. **逐层同步钩子（传输-计算流水线）**：回迁 H2D 在独立 `load_stream` 上异步逐层执行，每层传输完成后由控制器记录 producer 事件（`LayerDoneCounter.complete(i)`，cache_controller.py `start_loading`）。设备池全部压缩域 getter——`get_extra_key_buffer`、`get_index_k_with_scale_buffer`、`get_index_k_scale_buffer`、`get_swa_key_buffer`、`get_swa_key_buffer_radix`、`get_attention_compress_states`、`get_indexer_compress_states`——在返回 buffer 前统一执行 `layer_transfer_counter.wait_until(layer_id - start_layer)`，保证注意力/压缩器读到的是本层已落地的数据。该覆盖必须遍及每一个暴露设备 buffer 的入口：任何新增加载域时，其读取 getter 必须同步挂钩，否则异步传输与计算之间存在读到半程数据的竞态窗口。
3. **占用位图幂等闸**：full 分配器维护 occupied 位图，`free()` 对已释放槽位拒绝二次释放并告警；回迁路径绕过包装 `alloc()` 直取底层分配器，故 `load()` 内补 `_mark_occupied`（cache_controller.py L709-716），保证"分配 → 标记 → 释放"账实相符，杜绝槽位双发。
4. **TP 一致性**：多 rank 间对匹配长度、回迁成功长度等关键决策以 `all_reduce(min)` 同步，保证各 rank 树状态一致。

### 6.5 页级匹配边界

radix 匹配与 host 分配均以 256-token 页为粒度：请求末尾不足一页的部分不进树、也不进 L2。回迁命中的是"整页前缀"，尾页由 prefill 重算。例：12312-token 请求回迁命中 12288 token（48 页），仅重算末尾 24 token 的问题段，TTFT 从全量 52.8s 降至 0.49s。

---

## 7. 正确性保障与可观测性

| 机制 | 位置 | 作用 |
|---|---|---|
| 逐层同步钩子 | `deepseekv4_memory_pool.py` 全部压缩域 getter（SWA/状态/索引/c4/c128） | 回迁 H2D 异步流水线与注意力计算的逐层栅栏；getter 返回前 `wait_until(layer_id - start_layer)` 等待本层数据落地 |
| 字节级校验探针 | `SGLANG_P2_VERIFY=1`（deepseekv4_memory_pool_host.py L222-226） | 迁出前 D2H 读回比对 + 所有回迁 H2D 逐字节比对；运维诊断用，生产关闭（回迁 TTFT 0.49s → 3.7s 为其开销） |
| occupied 幂等闸 | swa_memory_pool / full 分配器 | 二次 free 拒绝 + 告警；回迁补标记防泄漏 |
| 树侧 lockstep 守护 | hiradix_cache（`swa_evictable = evictable_size_`） | 双池计数单源，调度器 token 统计公式恒为正 |
| marker/needle 验收 | 测试链 T1/T3/needle | 跨驱逐回迁的确定性检索全对（256/256、3/3） |

---

## 8. 配置参数与容量规划

### 8.1 参数表

| 参数 | 实测值 | 说明 |
|---|---|---|
| `--enable-hierarchical-cache` | 开启 | HiCache 总开关 |
| `--hicache-ratio` | 2 | host 池 = ratio × GPU 池（262k tokens） |
| `--hicache-mem-layout` | layer_first | DSV4 仅支持该布局 |
| `--hicache-io-backend` | kernel | 内核传输（页粒度并行寻址） |
| `--hicache-write-policy` | write_through | 请求完成即迁出（默认） |
| `--hicache-numa-node` | 未指定 | 指定后池落 CXL/NUMA 节点 |
| `--max-total-tokens` | 131072 | full 域 L1 容量 |
| `--swa-full-tokens-ratio` | 1.0 | 影子槽要求 ≥ 1.0 |
| `page_size` | 256 | 断言固定（paged swa 模式） |
| `SGLANG_SWA_SHADOW_SLOT` | 1 | 影子槽开关 |
| `SGLANG_P2_VERIFY` | 0（生产） | 字节级校验探针 |

### 8.2 约束清单

1. `page_size` 必须为 256（host 块粒度与之耦合）；
2. 布局必须 `layer_first`；
3. `swa_full_tokens_ratio ≥ 1.0`（影子槽容量断言）；
4. `enable_hierarchical_cache` 与 `enable_hisparse` 互斥；
5. DSV4 不支持 L3 存储后端（仅 L2 host 内存）；
6. 状态池传输分组与 compressor 读写行强耦合，修改 ring 语义需同步传输分组逻辑。

### 8.3 容量估算公式

```
host_tokens = max_total_tokens × hicache_ratio
host_blocks = host_tokens / 256
host_bytes  = host_blocks × bytes_per_block(per-rank 实测 996864 B)
有效上下文  = GPU 池 + host 池 = max_total_tokens × (1 + hicache_ratio)
```

---

## 9. 源码索引

| 模块 | 文件（`third_party/sglang/python/sglang/srt/`） | 关键符号 |
|---|---|---|
| 设备多池 | `mem_cache/deepseekv4_memory_pool.py` | `DeepSeekV4TokenToKVPool`、`get_compress_state_ring_size`、`register_mapping` |
| host 镜像池 | `mem_cache/deepseekv4_memory_pool_host.py` | `DeepSeekV4TokenToKVPoolHost`、`_split_indices`、`backup_from_device_all_layer`、`load_to_device_per_layer` |
| SWA 分配器/影子槽 | `mem_cache/swa_memory_pool.py` | `SWATokenToKVPoolAllocator`（SGLANG_SWA_SHADOW_SLOT 分支 L285-425） |
| 状态池 | `mem_cache/compress_state.py` | `CompressStatePool`、`translate_from_swa_loc_to_state_loc` |
| 搬运控制器 | `managers/cache_controller.py` | `write`(L644)、`start_writing`(L662)、`load`(L690) |
| 树与调度 | `mem_cache/swa_radix_cache.py`、`mem_cache/hiradix_cache.py` | lockstep 计数、evict 路径 |
| 压缩器接口 | `layers/attention/compressed/compressor.py` | `create_paged_compressor_data`（swa 页/状态行寻址） |
| 容量推导 | `model_executor/memory_profiler.py` | 状态池行数公式（L107-108） |

---

## 10. 设计验证结论（摘要）

测试环境：2× L20 48GB（TP=2），DeepSeek-V4-Flash MXFP4，`--max-total-tokens 131072 --hicache-ratio 2`。详细数据见《DeepSeek-V4-Flash HiCache Agentic 收益测试报告》。

| 验证项 | 结果 |
|---|---|
| 搬运链数据正确性（needle：14k prompt 三深度精确检索，驱逐回迁后问答） | 3/3 命中 |
| 跨驱逐/回迁确定性检索（T1：n=1/4/8/16/32 并发会话 × 8 轮 marker） | 8/8、32/32、64/64、128/128、256/256 全对 |
| 回迁性能（T3：12k 前缀回迁 vs 驱逐后全量重算） | 0.50s vs 49.4s（≈ 99×），回迁轮事实内容完整 |
| 并发回迁（T4：8 会话并发冷恢复，命中率 96.9%） | p50 17.5s vs 全量重算 26.1s（1.49×） |
| 池满热态收益（T1 n=32：HiCache 热态轮 vs base 池满排队） | 68.6s vs 436.1s（6.4×，稳态 52~64s 为 6.8~8.4×） |
| 字节级校验（P2_VERIFY 探针全程） | 迁出/回迁 0 失配 |
| 稳定性（T1 n=32 池满压力 8 轮 + 全场景链） | 零崩溃、零账本异常 |

---

*附：相关测试数据文件（容器 /tmp/）：bench_needle_final.json、bench_multi_final.json、bench_t4_final.json、bench_multi_hicache_p2.json（校验探针轮）。*
