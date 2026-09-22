# 移植报告：NInfer 三元能力 → ninfer-4090

本文件记录这次适配的**判定依据与实测证据**。改动清单与重建步骤见
[../patches/README-改动说明.md](../patches/README-改动说明.md)。

---

## 1. 问题的形状

`ninfer-ada-ternary` 发布的是"工具与方法"：`patches/changed-files/` 是叠在
`Ambolio/ninfer-4090-windows @ 6eb70a07`（v1.0.8-windows 线）上的整文件快照。

目标是 `UDPSendToFailed/ninfer-4090 @ 5c60b7c9`（v1.2.0 线）—— 它能跑 sm_86 与 sm_89、支持 Linux，
但**不是同一条线**。两棵树在受影响文件上的差异（`diff` 行数）：

| 文件 | 三元增量 | v1.2.0 与 v1.0.8 的差异 |
|---|---|---|
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | 10 | **12110** |
| `src/targets/qwen3_6/impl/runtime/dflash_impl.h` | 6 | 423 |
| `src/ops/wrapper/gdn_input_proj.cpp` | 191 | 390 |
| `src/targets/qwen3_6_27b/impl/load/bindings.cpp` | 196 | 403 |
| `src/CMakeLists.txt` | 15 | 280 |

结论：**整文件覆盖不可行**。必须先把"真正的三元增量"从 v1.0.8 基线里分离出来。

## 2. 分离增量的方法

```
git clone --branch v1.0.8-windows --depth 1 <Ambolio/ninfer-4090-windows>  # 拿到 6eb70a07
diff -u <v1.0.8 原文件> <ada-ternary changed-files 同路径文件>              # 这才是三元补丁
```

这样得到 **约 1100 行真实改动 + 12 个新增文件**（相比"38 个文件全量覆盖"小了两个数量级），然后逐处
重新落到 v1.2.0 的对应代码上。每处的落点都以 v1.2.0 自己的上下文为准：例如 `variant.cpp` 里
权重档案枚举叫 `GroupwiseInt / GroupwiseIntW8Endpoints`（Ada 线叫 `Qwen36GroupwiseInt /
`Qwen38GroupwiseInt`）、`bind_weight()` 没有 `placement` 形参、`row_view()` 的每组字节数需要按
格式推导。

## 3. 判定依据：哪些地方"不能照抄"

判定规则只有一条：**目标线上是否存在这条路。** 不存在就不移植，而不是硬塞。

| 原补丁的做法 | v1.2.0 的现实 | 决定 |
|---|---|---|
| 在 `linear.cpp` / `embedding.cpp` / `linear_swiglu.cpp` 里为 `NVFP4`、`FP8_E4M3FN_ROW_BF16S` 加分支 | `QType` 只到 `I32_CTRL=6`，`src/ops/linear/` 下无 `nvfp4/`、`fp8/` | 丢掉这些分支 |
| `Weight` 带 `weight_scale_divisor` / `input_scale_divisor` | 本线 `Weight` 无这两个字段（NVFP4 专用）| 不引入 |
| 改 `program_impl.h` 给 `target_logprobs` 路径传工作区 | 本线**没有** `target_logprobs` / `score_hidden` | 该文件零改动 |
| CMake 放宽 `ninfer_media_acquire` 的条件 | 本线 `NINFER_BUILD_MEDIA_ACQUIRE` ⊇ `NINFER_BUILD_PROMPT_INPUT`，无悬空链接 | 不移植（与三元无关）|
| 旋转内核 `#include "ops/kv_cache/hadamard_d256.cuh"` | 本线删掉了 `ops/kv_cache/` | 补回上游同路径文件，保持三元文件逐字节一致 |
| `pack.py` 调 `row_split_geometry("PTQ1_0_G128", …)` | 本线 `QuantFormat` 按位宽反推每组字节，表达不了 24+2 / 32+0 | 新增 `TernaryFormat` 并注册 |

`QType` 的**取值**刻意保留 Ada 线的 9 / 10（而不是在本线补成 7 / 8）：两条线的日志与诊断要能对照着读，
且 7 / 8 在 Ada 线已被占用，本线将来并入 NVFP4 / FP8 时补空位会与本处静默冲突。

## 4. 实测证据

### 4.1 编译

- **C++**：对打过补丁的树逐翻译单元做 `g++ -std=c++20 -DNINFER_SM89=1 -fsyntax-only`，覆盖 `src/` 与 `apps/` 下全部 `.cpp`。
  失败 6 项，补齐 CMake 等效 include 路径后复测，只有 2 项是真缺系统包（`src/media/decode/decode.cpp` 缺 `libavcodec/avcodec.h`，
  `src/product/media_acquire/acquire.cpp` 缺 `curl/curl.h`）；`apps/serve/main.cpp` 与 `src/serve/responses_http.cpp` 用内置
  `third_party/cpp-httplib` 即可通过（原失败是探针漏传 `-I`），`src/serve/http_server.cpp` 缺的是 CMake 生成的 `ui.h`，
  `frontend.cpp` 缺的 `xgrammar` 由 configure 阶段 FetchContent 拉取。均与本补丁无关，详见 `docs/依赖安装-RockyLinux10.md`。
- **CUDA**：`nvcc 13.3 / sm_89` 编译 3 个三元翻译单元，全部通过；唯一告警是上游既有的
  `launch_pq2_gemv declared but never referenced`（本包注释里已说明该函数为何暂时不接线）。

### 4.2 旋转内核：真机 + 独立 oracle

```bash
NINFER_ROOT=<打过补丁的 ninfer 树> tools/verify/run_rotation_oracle.sh
```

harness 直接 `#include` 引擎同一份 `ternary_rotation_kernels.cuh`，所以编的是**真代码**；oracle 用
显式构造的 1024×1024 归一化 Sylvester-Hadamard 矩阵（已验证对称且正交）逐例比对。

| 用例 | k | tokens | 置换 | 方向 | 结果 |
|---|---|---|---|---|---|
| `plain_t1` | 5120 | 1 | 无 | 正向 | PASS rel_l2 2.2e-3 |
| `plain_t3` | 5120 | 3 | 无 | 正向 | PASS rel_l2 2.2e-3 |
| `perm_t1` | 6144 | 1 | (128,16,3) | 正向 | PASS rel_l2 2.2e-3 |
| `perm_t2` | 6144 | 2 | (128,16,3) | 正向 | PASS rel_l2 2.2e-3 |
| `wide_t1` | 17408 | 1 | 无 | 正向 | PASS rel_l2 2.2e-3 |
| `inv_t2` | 5120 | 2 | 无 | 逆向（词嵌入）| PASS rel_l2 1.5e-3 |

每个用例同时跑 3~5 个负控（符号与旋转次序对调、未归一化、符号行错位、置换方向反了、缺 P），
全部与原值分离（rel_l2 ≥ 0.97）—— 说明这个比对"有牙齿"。

### 4.3 这一节暴露出来的上游缺陷

第一次跑是 **3/6**：T=1 的三个全绿，**T>1 的三个全红**。原因不在内核，而在验证脚本：

- `oracle_rot.py` 用 `x.reshape(k, tokens)`（C 序）读缓冲，而引擎的 `[k, tokens]` 激活是
  **token 主序**（`ne[0]` 连续，元素 `(column, token)` 在 `token*k + column`）。T=1 时两种读法
  **完全重合**，所以只有 T>1 才会暴露。改为 `order="F"` 后 6/6。
- `check_embedding.py` 同一问题，且注释把布局写反了。
- `harness/gemm_test.cu` 少传后期新增的 `out_row_stride`，**根本编译不过**。
- `harness/rot_test.cu` 里 `return prop.name;` 返回局部变量地址。

这正好印证了原包 `docs/04` 里的那条告诫：*"任何 GEMM/变换类改动必须有 T>1 的用例，并在引擎侧验证；
T=1 时 token 主序与行主序完全重合，所以 T=1 全绿 ≠ 正确。"* —— 只是当时的独立 harness 自己没做到。
本包把这四处修掉后，独立 harness 第一次真正具备 T>1 的判别力。

### 4.4 制品几何

`tools/artifact` 与引擎侧 `storage_layouts.cpp::quant_geometry()` 的一致性由
`nifer-ternary check` 与 `tests/test_checks.py` 双重把守：`[248320, 5120]` 下
PTQ1_0_G128 = 278,118,400 B、PQ2_0_G128 = 337,715,200 B，与引擎注释里写死的数字逐字节一致。

## 5. 未能验证的部分（诚实交代）

- **端到端数值（PPL / 困惑度 / 接受率）没有复跑**：需要 Ternary Bonsai 2 27B 权重（约 7 GB GGUF）
  与一份 groupwise-int 模板制品，本环境不具备，且本仓按约定不分发权重。原先的数字
（PPL 6.445、MTP K=2 96.7–130.8 t/s）来自 Ada / Windows 线，**不能直接外推到本线**。
- **完整 CMake 构建未执行**：本机缺 `cmake`、`libcurl` 开发包与 FFmpeg 开发包；`xgrammar` 由 configure 阶段联网拉取，
  不是缺包。依赖补齐命令与实测校验见 `docs/依赖安装-RockyLinux10.md`，装完后需重跑 configure 才能给出完整构建结论。
  因此"逐翻译单元编译通过"是本次能达到的最强编译证据。
- **MMA 路径只做了编译验证**，没有在真机上与 SIMT 路径做数值对照（需要真实三元权重）。
  对照方法已备好：`NINFER_TERNARY_MMA=0/1` 跑同一窗口比数值。
