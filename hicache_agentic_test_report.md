# DeepSeek-V4-Flash HiCache 测试报告

日期：2026-09-22 ~ 2026-09-23 ｜ 服务器：10.129.131.90（容器 kt-dsv4-build）｜ 2× NVIDIA L20 48GB（TP=2）

---

## 1. 测试场景

### 1.1 测试环境

| 项 | 值 |
|---|---|
| GPU | 2× NVIDIA L20 48GB，TP=2 |
| 模型 | DeepSeek-V4-Flash（43 层混合架构，sliding_window=128，MLA kv_heads=1 / head_dim=512） |
| 量化/加速 | KT MXFP4，`--kt-num-gpu-experts 30 --kt-cpuinfer 128 --kt-threadpool-count 4 --kt-enable-dynamic-expert-update` |
| KV 池（L1） | `--max-total-tokens 131072 --swa-full-tokens-ratio 1.0`（full/swa 各 131k） |
| 通用参数 | `--chunked-prefill-size 2048 --max-prefill-tokens 2048 --max-running-requests 32 --context-length 16384 --mem-fraction-static 0.85` |
| HiCache 组（L2） | 追加 `--enable-hierarchical-cache --hicache-mem-layout layer_first --hicache-io-backend kernel --hicache-ratio 2`（host 池 = 2×131k = 262k tokens，1024 blocks × 996864 B） |
| base 组 | 同配置去掉 HiCache 参数（对照组） |
| 运行环境 | SGLANG_DSV4_MODE=2604 / SUBMODE=2604B、SGLANG_SWA_SHADOW_SLOT=1，conda env kt-sglang |

### 1.2 场景定义

| 场景 | 设计 | 考察点 |
|---|---|---|
| **T1** | n 并发会话（n=1/4/8/16/32）× 4k 基座 prompt + 每轮追加 512 token × 8 轮，每轮追问历史标记（marker） | 容量边界、多轮会话 TTFT 曲线、确定性检索正确性 |
| **T3** | 单会话 12312-token 请求（12288 filler + 24 token 问题）→ 40×4096 filler 驱逐 → 相同问题重问 | 驱逐回迁路径的 TTFT 收益与内容一致性 |
| **T4** | 18 会话×8192 fill（共 147k > 131k 池，强制驱逐到 host）→ 8 会话×4k 新请求 → 并发回迁原会话 | 池满驱逐路径、多会话并发回连性能 |
| **needle** | 14k prompt 在深度 0/1/2 埋 3 个密钥 → 驱逐回迁 → 问答 | HiCache 搬运链（kernel io + layer_first 布局）数据正确性 |

判据约定：cold=首轮（全量 prefill）；warm=后续轮（前缀命中）；确定性检索（marker/needle 精确命中）作为 KV 数据正确性判据；similarity（difflib 文本相似度）仅作参考指标（差异成因见 4.4）。

---

## 2. 测试数据

### 2.1 T1：并发多轮会话（cold=首轮 mean，warm=第 2-8 轮 mean）

| n | base cold | base warm | HiCache cold | HiCache warm | marker 命中 |
|---|---|---|---|---|---|
| 1 | 18.6s | 4.4s | 18.6s | 3.9s | 8/8 |
| 4 | 50.6s | 8.4s | 51.1s | 9.6s | 32/32 |
| 8 | 87.9s | 13.0s | 88.2s | 16.6s | 64/64 |
| 16 | 161.5s | 24.9s | 162.0s | 25.7s | 128/128 |
| 32 | 308.7s | 436.1s（池满排队） | 309.3s | 68.6s | 256/256 |

n=32（池满边界场景）逐轮 TTFT（32 会话均值）：

| 轮次 | r1（cold） | r2 | r3 | r4 | r5 | r6 | r7 | r8 |
|---|---|---|---|---|---|---|---|---|
| mean TTFT | 309.3s | 136.5s | 64.2s | 55.2s | 52.1s | 56.1s | 60.5s | 55.3s |

### 2.2 T3：单会话 12k 前缀驱逐回迁

| 指标 | base | HiCache |
|---|---|---|
| cold TTFT | 51.3s（cached=0 全量 prefill） | 49.7s（cached=0 全量 prefill） |
| refill TTFT（重问同题） | 49.4s（cached=0，GPU 池已被驱逐，全量重算） | 0.50s（cached=12288，48 页整页回迁 + 24 token 尾部重算） |
| similarity | 1.0（两次全量自比） | 0.696 |
| refill 输出 | 正常 | 事实内容完整（正确概括原文主题），措辞与 cold 轮存在改写（成因见 4.4） |

单会话回迁收益：回迁 0.50s vs 重算 49.4s ≈ 99 倍。

### 2.3 T4：池满驱逐 + 8 会话并发回迁

| 阶段 | base | HiCache | 对比 |
|---|---|---|---|
| fill 18×8192（驱逐源） | 589.6s | 591.3s | — |
| stage1 warm（8 会话共享 4k 前缀首问）p50 | 30.2s | 35.2s | — |
| stage3 restore（并发回迁原前缀）p50 | 26.1s（cached=0 全量重算） | 17.5s（命中率 96.9%） | 1.49× |

### 2.4 needle：搬运链数据正确性（驱逐回迁后精确检索）

| 组 | depth=0 | depth=1 | depth=2 | 总计 |
|---|---|---|---|---|
| base | ✓ | ✓ | ✓ | 3/3 |
| HiCache | ✓ | ✓ | ✓ | 3/3 |

14k prompt 在三档深度埋入 6-8 位密钥，驱逐至 host 后回迁问答，三档全部精确命中。

### 2.5 稳定性与字节级校验

| 观测项 | 结果 |
|---|---|
| 全场景链（needle + T3 + T1 全档 + T4） | 服务零崩溃、零请求失败 |
| 调度器账本（swa/full token 统计） | 全程无异常读数 |
| 字节级校验轮（开启 SGLANG_P2_VERIFY 探针的独立验证轮：迁出 D2H 读回比对 + 全部回迁 H2D 逐字节比对） | 全程 0 失配；探针开销使 refill TTFT 0.49s→3.7s（生产默认关闭） |

---

## 3. 测试结论

1. **回迁收益显著**：单会话 12k 前缀回迁 TTFT 0.50s vs 全量重算 49.4s（≈ 99 倍）；池满压力场景（n=32）HiCache 组热态轮 TTFT 均值 68.6s vs base 组 436.1s（6.4 倍，去除驱逐尾巴后的稳态 52~64s 为 6.8~8.4 倍），冷态首轮两组持平（309.3s vs 308.7s）。
2. **数据正确性全绿**：确定性检索判据全对（T1 marker 256/256、needle 3/3），字节级校验 0 失配——迁出、驱逐、回迁整条搬运链数据无损。
3. **多会话并发回迁有效**：T4 并发回迁 p50 17.5s vs base 26.1s（1.49 倍），回迁命中率 96.9%；收益受并发带宽竞争制约（见 4.3）。
4. **无驱逐场景存在小幅写回开销**：池未满（n≤16，GPU radix 全命中、HiCache 不参与搬运）时，n=4/8 场景 HiCache 组 warm 比 base 慢 14~28%，n=1/16 基本持平（-11%~+3%），来源是 write_through 策略下每请求完成即触发 D2H 迁出。
5. **容量边界结论**：131k GPU 池的并发承载上限约 24 路 4k 会话（base 组 n=32 池满排队 warm 436s）；host 池 2× 扩容后该场景热态 TTFT 稳定在 52~68s 量级，是长会话/多会话产品形态的基础设施收益。

---

## 4. 测试分析

### 4.1 回迁收益构成

T3 场景 TTFT 由"前缀 KV 计算"主导：12312-token 全量 prefill ≈ 50s；命中后仅重算页边界外的 24 token，回迁搬运（48 页 host→GPU）耗时亚秒级（0.50s）。收益本质是**以 PCIe/主机内存带宽的页粒度拷贝替代 GPU 上的全量注意力前向计算**，前缀越长、复用次数越多，收益越大。

### 4.2 T1 曲线形态分析

- n≤16：n×(4k+8×512) < 131k 池，GPU radix 全命中，HiCache 不参与搬运，两组 cold 持平；HiCache 组 warm 差值即写回开销（每请求 finish 的 D2H 迁出），n=4/8 为 14~28%，n=1/16 基本持平（-11%~+3%），属容量型设施的常驻成本。
- n=32：总上下文 128k ≈ 池满。base 组无 L2，驱逐即永久丢失，热态轮退化为全量重算+排队（436s）；HiCache 组驱逐内容落入 host，热态轮由"回迁 + 尾部重算"构成（r2 为驱逐高峰尾巴 136.5s，r3-8 稳态 52~64s，均值 68.6s），收益随驱逐压力放大。

### 4.3 并发回迁瓶颈分析

T4 八会话同时回迁 p50 17.5s，与单会话 0.50s 不呈线性关系：并发回迁共享 host→GPU 传输带宽与 host 池锁，且逐层串行传输（43 层 × 每层页拷贝），并发度提升带宽利用率但引入排队。对比单会话场景，T4 更接近"多 agent 同时冷恢复"的产品形态，优化方向是回迁请求的调度错峰与传输并行度。

### 4.4 similarity < 1 的数值分析

T3 refill 轮与 cold 轮输出措辞存在改写（sim 0.696~0.789），为**良性数值差异**而非数据损坏：

- 页级匹配边界：prompt 末尾 24 token（问题段）不足一整页（256），不进树不回迁，refill 轮仅对这 24 token 做增量 prefill，计算路径与 cold 轮的全量分块 prefill 不同；
- 两条路径的 kernel 归约顺序存在浮点级差异，贪心解码下个别 token 位置 argmax 翻转，引发后续措辞改写；
- refill 输出对原文主题的事实概括完整无误，且全场景确定性检索（256 marker + 3 needle）精确命中，字节级校验 0 失配——数据正确性以确定性判据为准，similarity 仅作参考。

### 4.5 适用场景建议

| 场景 | 建议 |
|---|---|
| 单 agent 长会话（上下文逐步追加） | 收益最大（回迁 99×），建议开启 |
| 多 agent/多会话共享前缀 | 收益明确（并发回迁 1.49×、命中率 96.9%），高并发冷恢复建议错峰 |
| 短会话、池未填满（<20 路 4k） | HiCache 无收益且有写回开销，可不开 |
| 超长文档多轮追问 | 页级匹配下尾页重算量小（<256 token），收益接近全额 |

---

*数据文件（容器 /tmp/）：bench_needle_final.json、bench_multi_final.json、bench_t4_final.json（HiCache 组）；bench_needle_base.json、bench_multi_base.json、bench_t4_base.json（对照组）；字节级校验轮：bench_multi_hicache_p2.json*
