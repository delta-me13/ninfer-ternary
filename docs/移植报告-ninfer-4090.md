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
- **完整 CMake 构建**：sm_89 与 sm_86 两条都从零 configure、全量编译并链接通过（`exit 0`），
  带 `-DBUILD_TESTING=ON` 的第三条也完成。细节与 cubin 证据见 §4.5。

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

### 4.5 架构支持：三元移植没有收窄任何东西

**先纠正一个前提：`ninfer-4090` v1.2.0 并不支持 `sm_86`~`sm_120`。** 它只支持 `sm_86` 与 `sm_89`，
而且是一道硬闸门（`CMakeLists.txt`，在任何 `option` 之前）：

```cmake
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  message(FATAL_ERROR "NInfer supports CMAKE_CUDA_ARCHITECTURES=86 or 89; got ...")
endif()
```

`sm_86` 来自 `Don-Chad/ninfer-3090` 血缘（3090），`sm_89` 是本线主目标（4090）。传 `120` 会在
configure 阶段被直接拒绝 —— 这不是本补丁引入的，本补丁一行都没碰架构判定。

**NVFP4 / FP8 也不是本补丁丢弃的。** 它们是 v1.0.8-windows（Ada / Windows）那条线的格式：
本线的 `QType` 到 `I32_CTRL = 6` 为止、`src/ops/linear/` 下只有 `bf16 / q4 / q5 / q6 / w8`；
而来源基座 `ninfer-4090-windows @ v1.0.8` 里才是 `NVFP4 = 7`、`FP8_E4M3FN_ROW_BF16S = 8`，
外加 `src/ops/linear/{nvfp4,fp8}/`。这件事写在 v1.2.0 自己的发布说明里：

> **Blackwell SM120 and NVFP4 Purge** —— *Surgically deleted 45+ SM120 NVFP4 kernel files, TMA
> loaders, container parser descriptors, test fixtures, model cards, and conversion scripts,
> keeping the codebase strictly targeted to Ada Lovelace (`sm_89`).*
> —— `RELEASE_NOTES_1.2.0.md`

也就是说：NVFP4 / FP8 是**上游在 v1.2.0 里主动铲掉的**，比本补丁早存在一个版本。本补丁从 Ada 线
搬来的三元增量里确实夹带了一些 `case QType::NVFP4:` / `FP8_E4M3FN_ROW_BF16S` 分支，但那些分支引用
的是 Ada 线的枚举值 —— 在本线这两个符号**根本不存在**，照抄会直接编译不过。去掉它们是编译前提，
不是功能取舍；把它们"加回来"等于把整条 Ada 线的 NVFP4/FP8 内核族反向移植过来，那是另一个工程。

#### 条件编译本来就在，而且两个架构都真编过了

本仓没有另造开关：架构选择走的是上游既有的两处机制 —— `-DCMAKE_CUDA_ARCHITECTURES=86|89` 与由它
派生的 `NINFER_SM86` / `NINFER_SM89` 宏；`tools/verify/build.sh` 只是把它包成 `NINFER_ARCH=86|89`，
并顺手修掉了它自己 `-- <额外参数>` 未被消费的缺陷。

三元内核**没有引入任何架构专属代码**：三元目录下 grep 无 `__CUDA_ARCH__`、无 `NINFER_SM86` /
`NINFER_SM89`（全仓唯一用到这两个宏的是上游既有的 `w8_rowsplit_gemm_splitk.cu`）。
两个架构各自完整构建的结果：

| 构建 | 命令 | 结果 |
|---|---|---|
| sm_89（4090）| `NINFER_ARCH=89 tools/verify/build.sh clean` | **exit 0**；`apps/ninfer` 234 MB、`apps/ninfer-serve` 246 MB |
| sm_86（3090）| `NINFER_ARCH=86 tools/verify/build.sh clean` | **exit 0**；`apps/ninfer` 242 MB |
| sm_89 + 测试 | `NINFER_ARCH=89 … build.sh clean -- -DBUILD_TESTING=ON` | 见 §5 关于上游测试缺陷的说明 |

两个架构里三元内核都是**原生 cubin**（不是 PTX 兜底，也不会在 3090 上退回解释执行）：

```text
ternary_rotation.cu.o      -> ternary_rotation.sm_89.cubin      / ternary_rotation.sm_86.cubin
ternary_rowsplit_gemm.cu.o -> ternary_rowsplit_gemm.sm_89.cubin / ternary_rowsplit_gemm.sm_86.cubin
```

#### 想把 sm_120（5090）加回来要做什么

那是**反向移植**，不是改一个数字：需要把 v1.2.0 铲掉的 45+ 个 SM120/NVFP4 内核、TMA loader、
容器解析描述符从 `ninfer-4090-windows` v1.0.8 线搬回来，让 `QType` 扩到 7 / 8，并把 MMA 调度从
Ada 适配到 Blackwell；然后才谈得上摘掉上面那道 `FATAL_ERROR`。上游把它写成硬失败是有意的：
没有验证过的架构不该静默走进一套只做过 Ada 调优的调度。本补丁把三元取值放在 9 / 10、避开 7 / 8，
正是为了给这件事留出空位而不产生静默冲突。

### 4.6 上游漂移：官方模板制品已经走到容器 v3

在准备端到端验证时发现，`neroued/Qwen3.8-27B-NInfer` 的 `main` 分支上的 `qwen3_8_27b.ninfer`
**不是引擎能读的容器版本**。实测前 8 字节：

| 修订 | 首 8 字节 | 容器 | 目录偏移 | `json_bytes` | 体积 |
|---|---|---|---|---|---|
| `1cbd84e7`（`main`）| `NINFER\x00\x03` | **v3** | 32 | 372,704 | 20,437,521,664 B |
| `dc370fb6295a` | `NINFER\x00\x02` | **v2** | 16 | 185,105 | 20,437,336,576 B |

v3 在前缀之后多了 16 字节摘要，JSON 目录从偏移 32 开始，且顶层键变成
`components / objects / bindings / uses / metadata / provenance / files`（**没有** `identity`）。
而 ninfer-4090 v1.2.0 两边都只认 v1 / v2：

- C++：`src/artifact/reader.cpp` 的 `kMagic` / `kV1Magic`，否则抛 `artifact magic is not NInfer v1 or v2`；
- Python：`tools/artifact/container.py` 的 `MAGIC = b"NINFER\x00\x02"`。

时间线也对得上：HF 提交 `51630a0c` 就是「Publish v3 artifact and updated model card」，在 v1.2.0 之后。
上一个修订 `dc370fb6295a`「Update artifact with DFlash2 companion weights」是 v2，并且正好带 66 个
`dflash2/*` 张量 —— 即 v1.2.0 新增的 validate-only stub 要消费的那批。**做三元模板应该钉这个修订。**

`pack.py` 现在会先读容器前缀并把版本号直接报出来，而不是在偏移 16 上解一个不是 JSON 的东西、
抛一个与真实原因无关的 `JSONDecodeError`。

### 4.7 端到端：两个格式都真的答对了

用真实权重（`/data/Ternary-Bonsai-2-27B-gguf`）转出两个制品，在本机 RTX 4090 上跑通：

| 制品 | 体积 | 对象 | 引擎装载 | `17 * 23` 贪心输出 | decode |
|---|---|---|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.ninfer` | 10,533,732,876 B = 9.810 GiB | 1192 | 772 张量 / 6 资源，权重 6.70 GiB | **391** | 52.3 tok/s |
| `Ternary-Bonsai-2-27B-PTQ1_0.ninfer` | 9,274,212,876 B = 8.637 GiB | 1192 | 772 张量 / 6 资源，权重 5.52 GiB | **391** | 15.8 tok/s |

两个制品各 1192 个对象 = 模板 1190 + 新增的 `text/hadamard_signs` / `text/hadamard_widths`；
借用模板 3.114 GiB（`frontend` 6、`text` 2、`mtp` 12、`vision` 333、`dflash2` 66）。

#### 两个开关的双向对照（这才是关键）

| 运行 | PQ2_0 | PTQ1_0 | 判读 |
|---|---|---|---|
| `NINFER_TERNARY_MMA=1` | `391` | `391` | 张量核路径 |
| `NINFER_TERNARY_MMA=0` | `391` | `391` | SIMT 路径；**与上面逐字节一致** |
| `NINFER_TERNARY_HADAMARD=0` | 乱码 | 乱码 | **负控分离**：关掉折叠基旋转立刻崩坏 |

这三行合起来说明三件事：

1. 两条内核路径（MMA / SIMT）在本模型上给出**完全相同**的贪心序列 —— 此前"MMA 只做过编译验证"
   的空白在这里补上；
2. 关掉旋转就彻底崩坏，证明折叠基旋转是**载荷路径**而不是可选项，也证明 `391` 不是碰巧；
3. 两种三元解码（PQ2_0 的 2-bit 码、PTQ1_0 的 base-3 三元 + 高位平面）都正确 —— 它们走的是不同的解码原子。

PTQ1_0 比 PQ2_0 慢 3.3 倍（15.8 vs 52.3 tok/s）符合预期：PTQ1_0 每组 26 字节装 128 个权重且要做
base-3 的除法与取模，PQ2_0 每组 34 字节、纯移位取码。

复现命令（`--no-thinking --greedy` 让两条路径逐字节可比）：

```bash
NINFER_TERNARY_MMA=1 /data/ninfer-build/apps/ninfer <artifact.ninfer> \
  --prompt "What is 17 * 23? Answer with just the number." \
  --max-new 16 --max-context 512 --no-thinking --greedy --seed 1234
```

## 5. 未能验证的部分（诚实交代）

- **PPL / 困惑度 / MTP 接受率没有测**。端到端的*正确性*已经验过（§4.7：两种格式都答对 `391`，
  MMA 与 SIMT 逐字节一致，关掉旋转即崩坏），但质量分数没跑。原先的数字（PPL 6.445、
  MTP K=2 96.7–130.8 t/s）来自 Ada / Windows 线，**不能直接外推到本线**。
- **引擎自带测试有 2 项失败，且是本补丁引起的**（完整构建本身已通过，见 §4.5）：
  `ninfer_gdn_input_proj_conv_snapshot_test` 与 `ninfer_gdn_input_proj_conv_record_test` 断言
  "工作区查询值 == 执行高水位"，而本补丁让这两个容量查询对**非三元**父权重也按三元规模预留。
  根因是刻意共用了 `GroupwiseInt` 权重档案：三元制品沿用它，于是容量查询在"查询侧只有形状、没有格式"
  的签名下无法区分两者。运行不会出错（多预留是安全的），但该不变量被破坏。正确修法是给折叠三元一个
  独立的 `WeightsProfile`（由 `Package::resolve_weights` 依据制品身份解析），让容量查询重新精确；
  涉及 `export/.../package.h` 枚举、`resolve_weights`、`variant.cpp` 的 14 处 `case` 与 `bindings.cpp` 的 4 处。
  其余 82/84 用例通过。
- **长上下文 / 并发下的数值一致性没有逐段比**。§4.7 只比了 `--max-context 512`、16 token 的贪心窗口；
  MMA 与 SIMT 在更长的 prefill 与 `T>8` 的分块边界上是否仍逐字节一致，本次没有覆盖。
