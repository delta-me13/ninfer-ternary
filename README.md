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

```bash
# 1) 看一眼会把哪些文件写到哪
uv run nifer-ternary manifest

# 2) 确认目标检出是干净的
uv run nifer-ternary status --repo /path/to/ninfer-4090

# 3) 落盘改动（--dry-run 先看不动手；已分叉的文件需要显式 --force）
uv run nifer-ternary apply  --repo /path/to/ninfer-4090 --dry-run
uv run nifer-ternary apply  --repo /path/to/ninfer-4090

# 4) 落地自检（不需要 torch，也不需要 GPU）
uv run nifer-ternary check  --repo /path/to/ninfer-4090

# 5) 补齐构建依赖（Rocky Linux 10；其他发行版见同一份文档的对照表）
tools/verify/install_deps_rocky10.sh check     # 只自检
sudo tools/verify/install_deps_rocky10.sh install

# 6) 构建
export NINFER_ROOT=/path/to/ninfer-4090
tools/verify/build.sh clean        # 3090 用 NINFER_ARCH=86
```

---

## 仓里有什么

```
patches/
  README-改动说明.md                     ← 改动清单、与上游的刻意差异、重建与验证步骤（先读这个）
  manifest.json                          ← 目标提交 + 逐文件上游/改动后 sha256
  0001-ternary-port-on-ninfer-4090.patch ← 统一 diff（仅供审阅）
  changed-files/                         ← 整文件快照，覆盖即可生效（42 个文件：新增 14 / 修改 28）
tools/
  pack.py                                ← GGUF（Bonsai 2 27B）→ 三元 .ninfer 打包器
                                           （模板必须同时是 groupwise-int **且容器 v2**，见 patches/README §D）
  MAPPING.json                           ← 逐张量映射规格（权威）
  _bootstrap.py                          ← 统一的 ninfer 源码树定位（NINFER_ROOT）
  _ternary_ref.py                        ← 三元解码器的唯一实现（从 pack.py 再导出）
  verify/
    run_rotation_oracle.sh               ← 一键：编译旋转 harness → 真机跑 → numpy oracle 比对
    build.sh / loadtest.sh / gentest.sh  ← Linux 构建与探针（替代原包的 .cmd）
    install_deps_rocky10.sh              ← Rocky Linux 10 依赖安装与自检（check / install）
    check_{payload_order,row_order,signs,assembly,embedding}.py
    oracle_rot.py / gemm_oracle.py / list_objects.py
    harness/{rot_test.cu,gemm_test.cu}   ← 编译引擎同一份真代码的独立 nvcc 测试台
src/nifer_ternary/                       ← 应用 / 状态检查 / 自检的 CLI
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
| **引擎自带测试 `ctest`** | **82/84 通过**；2 项失败由本补丁引起，根因与修法见 §5 |
| **端到端（真权重 + RTX 4090）** | 两种格式都装载并答对 `17 * 23` → **391**；`MMA=1` 与 `MMA=0` **逐字节一致**；关掉折叠基旋转即崩坏（§4.7）|
| 折叠基旋转内核（真机 + numpy oracle）| **6/6 PASS**，负控全部分离（正确的 rel_l2 ≈ 2.2e-3，负控 ≥ 0.97）|
| `tools/artifact` 三元几何 vs 引擎 | `[248320,5120]` → PTQ1_0 278,118,400 B / PQ2_0 337,715,200 B，与引擎侧注释逐字节一致 |
| Python 包 | 14 用例通过（正负控只用临时目录与仓内快照）；`ruff check` / `ruff format --check` / `mypy` 全绿 |
| 独立 harness 编译 | `rot_test.cu` / `gemm_test.cu` 均零错误零警告（`gemm_test.cu` 原本编译不过，见下）|

---

## 适配过程中发现并修掉的上游问题

原包的 `tools/` 是"当时工作树的快照"，与 `patches/` 并不同步。核过之后有四处会直接导致验证失败或结论错误：

1. `harness/gemm_test.cu` **编译不过** —— 调用 `ternary_rowsplit_gemm_kernel` 时少传后期新增的 `out_row_stride`。
2. `verify/oracle_rot.py` 按**行主序**读 token 主序缓冲：T=1 的 3 个用例全绿、**T>1 的 3 个全 FAIL**。改正后 6/6 通过。
3. `verify/check_embedding.py` 同一个问题，且注释写反（"row-major"）。
4. `harness/rot_test.cu` 里 `return prop.name;` 返回局部变量地址（未定义行为）。

另外补上了原包缺失的 `tools/_ternary_ref.py`（原 `check_*.py` 引用了它，包里却没有）。

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
