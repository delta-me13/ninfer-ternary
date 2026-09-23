# ninfer-ternary

将 [NInfer](https://github.com/UDPSendToFailed/ninfer-4090)（Apache-2.0）的**三元**能力
（Ternary Bonsai 2 27B）移植到 **RTX 3090（sm_86）与 RTX 4090（sm_89）** 上。

引擎源码不在本仓：安装时按固定的提交从上游拉取，应用本仓的三元补丁，编译后打包为 wheel，
临时文件用完即删。安装完成后将得到三个可执行文件：

| 命令 | 用途 |
|---|---|
| `ninfer` | 单卡命令行推理 |
| `ninfer-serve` | OpenAI / Anthropic 兼容的推理服务 |
| `ninfer-convert` | 将 Ternary Bonsai 的 GGUF 转换为 ninfer 制品（`.ninfer`）|

本仓不发布任何模型权重，也不发布由权重派生的 `.ninfer` 制品。

---

## 快速开始

前提：Linux x86_64、NVIDIA 驱动与 CUDA 工具链、`git`、`cmake`、`ninja`、`gcc`、`uv`。
Rocky Linux 10 上可使用仓内脚本一次性安装齐全（见 [依赖安装](docs/依赖安装-RockyLinux10.md)）：

    tools/verify/install_deps_rocky10.sh install

若尚未安装 `uv`：

    curl -LsSf https://astral.sh/uv/install.sh | sh

安装过程需要联网：除拉取上游源码外，CMake 配置阶段还会获取 xgrammar 与服务端 Web UI 两个第三方包。
离线环境请先在一台联网机器上执行 `uv build --wheel`，再将 wheel 复制到目标机器安装（见教程）。

安装推理引擎（将现场编译 CUDA 引擎，本机实测耗时 6 分半）：

    uv tool install git+https://github.com/<你的账号>/ninfer-ternary.git

本地已有本仓时可直接指向目录，效果相同：

    just tool-install          # 等价于 uv tool install --force .

安装完成后立即可用，无需仓库，也无需构建目录：

    ninfer --help
    ninfer-serve --help

如需转换模型，需再添加一个额外项：打包器需要读写张量并依赖 torch，因此转换能力单独提供：

    uv tool install "ninfer-ternary[convert] @ git+https://github.com/<你的账号>/ninfer-ternary.git"
    just tool-install-convert  # 本地目录的等价写法：uv tool install ".[convert]"

    ninfer-convert             # 输出转换器用法

细节、环境变量、离线安装与排错见 [把本仓当工具用](docs/uv-工具安装.md)。

---

## 转换一个模型

`ninfer-convert` 来自上述带 `[convert]` 额外项的安装；仅安装引擎的安装方式会在调用时
明确提示补装，不会让使用者面对 traceback 自行猜测。

转换需要两样输入：

| 输入 | 说明 |
|---|---|
| GGUF | `Ternary-Bonsai-2-27B-PQ2_0.gguf` 或 `-PTQ1_0.gguf`（HuggingFace 上的 Ternary Bonsai 2 27B）|
| 模板 | 一个 **groupwise-int** 的 qwen3.8-27b ninfer 制品。模板提供视觉塔、MTP 头等"借用"张量与对象清单，并非可有可无的参考文件 |

    ninfer-convert \
      --template /data/Ternary-Bonsai-2-27B-ninfer/template/qwen3_8_27b.v2.ninfer \
      --gguf     /data/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PQ2_0.gguf \
      build      /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer

写入磁盘之前会先执行只读自检（几何、解码往返、张量映射），确认无误后再写入：

    ninfer-convert --template <模板> --gguf <GGUF> check

两个路径没有内置默认值，必须显式指定或通过环境变量 `NINFER_TERNARY_TEMPLATE` /
`NINFER_TERNARY_GGUF`。本机实测产物：

| 制品 | 大小 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PQ2_0.ninfer` | 10,533,732,876 B | 2 bit 权码，精度更高、解码更快 |
| `Ternary-Bonsai-2-27B-PTQ1_0.ninfer` | 9,274,212,876 B | 三进制 + 高位平面，更省空间 |

---

## 执行推理

命令行单卡：

    ninfer /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer \
      --prompt "17 * 23 等于多少？" --max-context 4096 --max-new 256

启动服务并发送请求：

    ninfer-serve /data/Ternary-Bonsai-2-27B-ninfer/Ternary-Bonsai-2-27B-PQ2_0.ninfer \
      --host 127.0.0.1 --port 8080 --max-context 8192

    curl http://127.0.0.1:8080/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"你好"}]}'

常用的几组参数（完整列表见 `--help`）：

| 参数 | 作用 |
|---|---|
| `--max-context` / `--prefill-chunk` | 上下文长度与预填充分块（决定显存中工作区的大小）|
| `--kv-dtype bf16\|int8\|rk8v4\|rk4v4\|rk4v4-e8\|rk2v4-e8` | KV 缓存精度，越小越省显存 |
| `--spec mtp --draft-tokens 4` | MTP 投机解码，输出与未启用投机时逐字节一致 |
| `--no-cuda-graph` | 关闭 CUDA Graph（排查问题或显存紧张时使用）|

本机（RTX 4090，PQ2_0）实测：prefill **251–274 t/s**（`pp` 口径）、解码 **42–56 t/s**（CUDA Graph 开启）。
完整基准测试（两个三元制品各 20 条用例）、计量口径与读数离散性见 [基准测试](docs/基准测试.md)；
脚本与 suite 定义见 [tools/bench/README.md](tools/bench/README.md)。

---

## 常见问题

**安装完成后 `ninfer` 提示"找不到引擎可执行文件"。**
该次安装仅安装 Python 侧（`NINFER_TERNARY_SKIP_BUILD=1`），wheel 中不含引擎。
使用 `just tool-install` 重新安装；或将 `NINFER_ENGINE_BIN` 指向已有的构建产物目录。

**没有 CUDA 工具链，仅需要转换器。**
`just tool-install-light`（即 `NINFER_TERNARY_SKIP_BUILD=1 uv tool install --force .`）。
这种安装不含随包的上游制品模块，转换时需使用 `NINFER_ROOT=<ninfer 检出>` 指向一个已有的检出。

**模板从何处获取？**
模板是 qwen3.8-27b 的 **groupwise-int** ninfer 制品，与目标制品同为容器 v2。转换器会先读取其
`identity.weights_id` 进行校验，若非 `groupwise-int` 会直接拒绝并说明原因 —— 若误用其他来源的量化（例如
nvfp4），将在后续以张量名不匹配的形式失败。

**上游仓库在内网无法获取。**
`NINFER_TERNARY_TARGET_REPO=<镜像地址或本地检出> uv tool install --force .`。

**占用多少显存？如何估算？**
工作区容量随 `min(max_context, prefill_chunk)` 线性增长，与制品档案无关；实测与推导见
[权重档案与容量规划](docs/权重档案与容量规划.md)。

---

## 在仓库里开发

仓库根目录的 `justfile` 是全部入口。`just` 列出配方，`just config` 输出其解析得到的路径。

    just check          # ruff / mypy / pytest
    just deps           # 构建依赖自检
    just build          # 增量构建 sm_89（just build 86 编译 sm_86）
    just build-tests && just ctest
    just build-engine   # 拉取上游 -> 打补丁 -> 自检 -> 编译，临时树自动清理
    just oracle         # 旋转内核 vs numpy FP64
    just e2e <制品>     # 端到端一致性矩阵（结果在 out/e2e-<制品名>）
    just bench <制品>   # 标准化基准测试（结果在 out/bench-<制品名>-<时间戳>）
    just evidence <名称> # 把 out/ 的证据冻结进 docs/evidence/（报告定稿时执行）
    just pack PQ2_0     # 打包三元制品
    just clean          # 清理构建目录、字节码缓存与临时根

补丁侧的子命令（`manifest` / `status` / `apply` / `check` / `export`）通过
`uv run python -m ninfer_ternary` 调用，不随 `uv tool install` 安装 —— 它们仅在开发与验证时需要。

### 仓里有什么

    patches/            45 个文件的整文件快照 + 清单摘要（改动清单见 patches/README-改动说明.md）
    tools/pack.py       GGUF -> .ninfer 打包器（ninfer-convert 的本体）
    tools/verify/       oracle、端到端矩阵、依赖安装、落地自检
    tools/bench/        固定语料/重复/预热的标准化基准测试
    build_backend.py    uv tool install 时拉取、打补丁、编译、清理
    docs/               移植报告、权重档案、依赖安装、本工具安装、基准测试与证据快照

主要文档：

| 文档 | 内容 |
|---|---|
| [把本仓当工具用](docs/uv-工具安装.md) | `uv tool install` 全流程、环境变量、离线安装、排错 |
| [移植报告](docs/移植报告-ninfer-4090.md) | 判定依据、实测证据、未验证部分 |
| [权重档案与容量规划](docs/权重档案与容量规划.md) | 制品档案改变了什么、容量查询逐条对照 |
| [基准测试](docs/基准测试.md) | 完整基准测试：计量口径、两个三元制品的读数、读数离散性与 prefill 双峰、困惑度待测 |
| [依赖安装](docs/依赖安装-RockyLinux10.md) | Rocky Linux 10 缺失库清单与安装命令 |
| [改动说明](patches/README-改动说明.md) | 45 个文件的改动清单、与上游的刻意差异 |

---

## 验证状态（本机实测，RTX 4090 / Rocky Linux 10）

| 项 | 结果 |
|---|---|
| 构建 | sm_89 与 sm_86 各 exit 0，三元内核在两个架构下都有原生 cubin |
| 引擎自带测试 | `ctest` 84/84 通过 |
| 端到端 | 两种格式都装载并答对 `17 * 23`；`MMA=1` 与 `MMA=0` 逐字节一致；关闭折叠基旋转即失效 |
| 一致性矩阵 | 内核路径 × 分块 × 两种格式，10 次正控同摘要，负控分离 |
| MTP 投机 | 输出与不启用投机时逐字节一致，接受率 74-77%（draft 4）|
| MTP 实现现状 | 当前版本的 MTP 直接取自上游实现采用的官方 MTP，未做任何三元量化，也未做针对 ninfer 的优化；后续版本可望改进 |
| 长上下文 | 2685 与 11043 token 的 prompt 全部同摘要；三元 MMA prefill 约为 SIMT 的 4.2-4.4 倍 |
| 干净检出可复现 | `git clone` v1.2.0 -> 打补丁 -> `diff -r` 无差异；全量重新编译 726/726 exit 0，`ctest` 84/84 |
| 基准测试 | 两个三元制品各 20 条用例（panel / prefill / decode / kv / mtp / graph，各 5 次重复）：PQ2_0 prefill **251–274 t/s**（`pp` 口径）、decode **42–56 t/s**；CUDA Graph 关闭后 decode 从 48.0 降至 21.4 t/s；PTQ1_0 的 prefill 约为 PQ2_0 的 1/5.5、decode 约为 1/3，以换取节省 1.26 GB 显存 |
| 基准测试中的一处未归因读数 | PQ2_0 的 `pp+tg` prefill 逐次在约 230–320 与约 838 t/s 两峰间交替（20 次中 5 次进入高速峰）。已排除 GPU 频率（同期 SM 2730 MHz 不变）、MMA/SIMT 路径与分块；必要条件为 CUDA Graph 开启 + 单块 512 + 请求内含 decode。引用 prefill 时使用中位数，详见 [基准测试](docs/基准测试.md) §6 |
| **`uv tool install` 一条命令** | 现场拉取 v1.2.0 -> 写入 45 文件 -> 自检 20/20 -> 编译 -> 打包为 223 MiB wheel -> **临时根整个删除**（`/tmp` 不留文件也不留空目录），全程 **6 分 31 秒**；安装完成的 `ninfer` 直接答对 `17 * 23` |

---

## 许可与来源

本仓采用 **Apache-2.0** 许可，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

它是 NInfer（Apache-2.0）派生作品的适配层：三元改动来自
[ninfer-ada-ternary](https://www.modelscope.cn/shensanshu/ninfer-ada-ternary.git)，
目标树是 [ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) v1.2.0
（提交 `5c60b7c9`）。上游仅支持 sm_86 / sm_89，本补丁未收窄或放宽该范围。

模型权重不在本仓分发，其权利归原作者（PrismML / Qwen 体系）所有。
