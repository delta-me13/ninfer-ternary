# nifer-ternary · 把 NInfer 三元能力适配到 ninfer-4090

把 [`ninfer-ada-ternary`](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary.git)（NInfer 三元 /
Ternary Bonsai 2 27B，Ada sm_89 + Windows 线）的成果，**适配到
[`ninfer-4090`](https://github.com/UDPSendToFailed/ninfer-4090)（v1.2.0 线：支持 sm_86 / sm_89，支持 Linux）**。

本仓**不发布任何模型权重**，也不发布由权重派生的 `.ninfer` 制品。这里只有引擎侧源码改动、打包器、
验证脚本与说明。

```
目标树   UDPSendToFailed/ninfer-4090 @ 5c60b7c9（v1.2.0）
改动来源 shensanshu/ninfer-ada-ternary @ ca845a4
来源基座 Ambolio/ninfer-4090-windows @ 6eb70a07（v1.0.8-windows，Ada / Windows）—— 与目标树不是同一条线
```

> **架构**：本线只支持 `sm_86`（3090）与 `sm_89`（4090）—— 上游用 `FATAL_ERROR` 硬拒 `sm_120`。
> NVFP4 / FP8 是上游 v1.2.0 **主动铲掉**的格式，不是本补丁丢弃的；本补丁一个架构判定都没碰，
> 且 sm_86 / sm_89 两条都已完整构建通过。详见[移植报告 §4.5](docs/移植报告-ninfer-4090.md)。

---

## 快速开始

仓根的 `justfile` 把全部操作入口收成一条命令。`just` 列出配方，**`just config` 先打印它解析出来的
路径**（换机器第一件事）。不用 `just` 的，右列就是它实际执行的命令。

| 配方 | 等价命令 | 做什么 |
|---|---|---|
| `just config` | — | 打印 NINFER_ROOT / 构建目录 / 模板 / GGUF / 解释器 |
| `just deps` | `tools/verify/install_deps_rocky10.sh check` | 构建依赖自检（Rocky Linux 10）|
| `just deps-install` | `… install` | 安装构建依赖（需要 root）|
| `just patch-manifest` | `uv run nifer-ternary manifest` | 补丁清单摘要 |
| `just patch-status` | `uv run nifer-ternary status --repo $NINFER_ROOT` | 目标检出相对本补丁的状态 |
| `just patch-dry-run` | `uv run nifer-ternary apply --repo $NINFER_ROOT --dry-run` | 试运行，只报告不写文件 |
| `just patch-apply` | `uv run nifer-ternary apply --repo $NINFER_ROOT` | 落盘改动（分叉文件需 `--force`）|
| `just patch-export` | `uv run nifer-ternary export --repo $NINFER_ROOT` | 用检出内容刷新快照 / 清单 / 聚合 diff |
| `just patch-check` | `uv run nifer-ternary check --repo $NINFER_ROOT` | 落地自检（不需要 torch，也不需要 GPU）|
| `just check` | `ruff check . && ruff format --check . && mypy . && pytest` | 代码门禁 |
| `just build [86\|89]` | `NINFER_ARCH=89 tools/verify/build.sh` | 构建 |
| `just build-tests` | `… -- -DBUILD_TESTING=ON` | 构建测试目标 |
| `just ctest` | `ctest --output-on-failure` | 引擎测试套件 |
| `just oracle` | `tools/verify/run_rotation_oracle.sh` | 旋转内核 vs numpy FP64 |
| `just e2e <artifact>` | `tools/verify/e2e_ternary.sh <artifact>` | 端到端一致性矩阵 |
| `just bench <artifact> [suite]` | `tools/bench/bench.sh <artifact> [suite]` | 标准化跑分（语料/重复/预热钉死）|
| `just pack PQ2_0` | `tools/pack.py build <out>` | 打包三元制品 |
| `just all <artifact>` | — | 依赖 → 门禁 → 构建 → ctest → oracle → e2e |

---

## 仓里有什么

```
justfile                                 ← 全部操作入口（`just` / `just config` / `just --list`）
patches/
  README-改动说明.md                     ← 改动清单、与上游的刻意差异、重建与验证步骤（先读这个）
  manifest.json                          ← 目标提交 + 逐文件上游/改动后 sha256
  0001-ternary-port-on-ninfer-4090.patch ← 统一 diff（仅供审阅）
  changed-files/                         ← 整文件快照，覆盖即可生效（44 个文件：新增 14 / 修改 30）
tools/
  bench/
    bench.sh                             ← 标准化跑分：语料/重复/预热/分块钉死，产出 tidy CSV + 环境清单
    README.md                            ← suite 含义、输出布局、与 CLI 口径的差异
  pack.py                                ← GGUF（Bonsai 2 27B）→ 三元 .ninfer 打包器
                                           （模板必须同时是 groupwise-int **且容器 v2**，见 patches/README §D）
  MAPPING.json                           ← 逐张量映射规格（权威）
  _bootstrap.py                          ← 统一的 ninfer 源码树定位（NINFER_ROOT）
  _ternary_ref.py                        ← 三元解码器的唯一实现（从 pack.py 再导出）
  verify/
    run_rotation_oracle.sh               ← 一键：编译旋转 harness → 真机跑 → numpy oracle 比对
    build.sh / loadtest.sh / gentest.sh  ← Linux 构建与探针（替代原包的 .cmd）
    install_deps_rocky10.sh              ← Rocky Linux 10 依赖安装与自检（check / install）
    e2e_ternary.sh                       ← 端到端一致性：内核路径 × prefill 分块 × MTP，带负控
    check_{payload_order,row_order,signs,assembly,embedding}.py
    oracle_rot.py / gemm_oracle.py / list_objects.py
    harness/{rot_test.cu,gemm_test.cu}   ← 编译引擎同一份真代码的独立 nvcc 测试台
src/nifer_ternary/                       ← 应用 / 状态检查 / 自检 / 快照导出 的 CLI
tests/                                   ← 14 个用例（含负控；只用临时目录与仓内快照，不依赖机器状态）
docs/移植报告-ninfer-4090.md             ← 判定依据 + 实测证据 + 未验证部分
docs/依赖安装-RockyLinux10.md            ← 缺失系统库清单、安装命令、版本校验与备选方案
```

---

## 本次实际产出的制品

用 `/data/Ternary-Bonsai-2-27B-gguf` 与钉在 v2 修订版 `dc370fb6295a` 的模板转出（约 19 GiB）：

```text
/data/Ternary-Bonsai-2-27B-ninfer/
  Ternary-Bonsai-2-27B-PQ2_0.ninfer    10,533,732,876 B = 9.810 GiB   decode 52.3 tok/s
  Ternary-Bonsai-2-27B-PTQ1_0.ninfer    9,274,212,876 B = 8.637 GiB   decode 15.8 tok/s
```

两个都能被引擎装载并答对 `17 * 23`。**本仓不收录这些制品**（由权重派生），重建步骤见
[patches/README-改动说明.md](patches/README-改动说明.md) §D。

---

## 验证状态（本机实测）

| 项 | 结果 |
|---|---|
| **完整构建 sm_89 / sm_86** | **各 exit 0**。三元内核在两个架构下都有原生 cubin，架构支持未被收窄（§4.5）|
| **引擎自带测试 `ctest`** | **84/84 通过**（0 失败；5 项 `real` 用例按设计跳过）。本补丁此前造成的 2 项失败已修复（§4.9）|
| **端到端（真权重 + RTX 4090）** | 两种格式都装载并答对 `17 * 23` → **391**；`MMA=1` 与 `MMA=0` **逐字节一致**；关掉折叠基旋转即崩坏（§4.7）|
| **一致性矩阵（MMA/SIMT × 分块 × 两种格式）** | 10 次正控全部落在同一个 158 token 摘要上；负控（关掉旋转）分离。`e2e_ternary.sh` 两个制品都 PASS（§4.8）|
| **MTP 投机** | 输出与无投机**逐字节一致**；接受率 74–77%（draft 4）、58.5%（draft 8）（§4.8）|
| **KV 量化** | `bf16 / int8 / rk8v4 / rk4v4 / rk4v4-e8 / rk2v4-e8` 六种都能装载跑完 pp512/tg128（§4.8）|
| **长上下文一致性** | 2685 与 11043 token 的 prompt，MMA/SIMT × 分块共 10 次全部同摘要，答案正确；三元 MMA prefill 约为 SIMT 的 **4.2~4.4 倍**（§4.8）|
| **并发服务** | `ninfer-serve --max-concurrency 4`，8 个并发请求 8/8 **200**，输出逐字节一致（§4.8）|
| 折叠基旋转内核（真机 + numpy oracle）| **6/6 PASS**，负控全部分离（正确的 rel_l2 ≈ 2.2e-3，负控 ≥ 0.97）|
| `tools/artifact` 三元几何 vs 引擎 | `[248320,5120]` → PTQ1_0 278,118,400 B / PQ2_0 337,715,200 B，与引擎侧注释逐字节一致 |
| Python 包 | 20 用例通过（正负控只用临时目录与仓内快照）；`ruff check` / `ruff format --check` / `mypy` 全绿 |
| 独立 harness 编译 | `rot_test.cu` / `gemm_test.cu` 均零错误零警告（`gemm_test.cu` 原本编译不过，见下）|
| **标准化跑分（`tools/bench/bench.sh`）** | PQ2_0 `standard`：pp512 **267.2±18.4**、pp2048 **303.4±5.6**、tg128 **50.8±4.3** t/s。CSV 里 `weights_id` 读回 **`folded-ternary`**，`workspace_capacity_bytes` = **180,953,088**（= 修复前后两个二进制算出的同一个 172.57 MiB）|

---

## 适配过程中发现并修掉的上游问题

原包的 `tools/` 是"当时工作树的快照"，与 `patches/` 并不同步。核过之后有四处会直接导致验证失败或结论错误：

1. `harness/gemm_test.cu` **编译不过** —— 调用 `ternary_rowsplit_gemm_kernel` 时少传后期新增的 `out_row_stride`。
2. `verify/oracle_rot.py` 按**行主序**读 token 主序缓冲：T=1 的 3 个用例全绿、**T>1 的 3 个全 FAIL**。改正后 6/6 通过。
3. `verify/check_embedding.py` 同一个问题，且注释写反（"row-major"）。
4. `harness/rot_test.cu` 里 `return prop.name;` 返回局部变量地址（未定义行为）。

另外补上了原包缺失的 `tools/_ternary_ref.py`（原 `check_*.py` 引用了它，包里却没有）。

---

## 本补丁自己引入过的回归（已修复）

改动包不该留下红色的上游测试。这一条是本补丁**自己**造成的，与上游无关，记录在此以免被当成
"上游本来就这样"：

**症状**：`ninfer_gdn_input_proj_conv_snapshot_test` 与 `ninfer_gdn_input_proj_conv_record_test` 报
`workspace query/execution high-water mismatch`。

**根因**：折叠三元的父权重与 Q4/Q5 共用同一套**行几何**（q/k 2048、v 6144），于是两条按形状索引的
ops 容量查询无法区分它们。最初的移植选择"让两条查询都按三元的最坏情况预留"，代价是 groupwise-int
制品也被多留一块激活缓冲 —— 而那两个测试断言 `peak_used() == 查询值`，多留即失败。

**修法**（§4.9 有完整清单）：给折叠三元一档独立的 `WeightsProfile::FoldedTernary`，让容量查询按档
分派；两条 ops 查询恢复上游的精确语义，三元另开两条专属查询。**制品因此必须重打** —— 它的
`identity.weights_id` 要从 `groupwise-int` 改成 `folded-ternary`，否则引擎按 groupwise 规划而三元
执行时还要多要一块旋转缓冲，那是**少留**，不是保守的多留。

**修完 `ctest` 就绿了，但坑还没填完。** 第一版只改了**规划期**的容量查询，`ctest` 回到 84/84，
而 `--spec mtp` 全线崩在 `std::bad_alloc` —— `ctest` 覆盖不到那条路径。真正的原因在**运行期**：
`Variant::gdn_input_projection_record` 会开一块**借来的叶子竞技场**，容量取自
`gdn_record_workspace_bytes()`；而上游那条 record 容量查询对 Q4/Q5 返回 0，于是叶子只有 1 字节，
折叠三元的算子却要在里面分配 51200 字节的旋转缓冲。这套「规划期查询 + 运行期叶子」的双份需求，
只改一半就会以 `std::bad_alloc` 的形式在图构建时炸掉。完整过程见 §4.9。

---

## 环境变量

| 变量 | 用于 |
|---|---|
| `NINFER_ROOT` | 指向 ninfer 源码树（`tools/` 下的脚本都需要）|
| `NINFER_BUILD_ROOT` / `NINFER_ARCH` / `NINFER_JOBS` | `build.sh` |
| `NINFER_CLI` / `NINFER_LOG_DIR` | `loadtest.sh`、`gentest.sh` |
| `NINFER_TERNARY_GGUF` / `NINFER_TERNARY_TEMPLATE` | `pack.py` 的输入 |
| `NINFER_TERNARY_HADAMARD` / `_GDN_PERM` / `_MMA` / `_DUMP_EMBED` / `_TRACE_EMBED` | 引擎运行时开关，见 [patches/README-改动说明.md](patches/README-改动说明.md) §G |

---

## 许可与来源

本仓是 **NInfer（Apache-2.0）** 派生作品的三元适配层：引擎改动来自 `ninfer-ada-ternary`（Apache-2.0），
目标树为 `ninfer-4090`（Apache-2.0）。模型权重不在本仓分发，其权利归原作者（PrismML / Qwen 体系）所有。
