# 基准测试证据快照

本目录保存**已定稿报告**所引用的原始证据。目录内的文件按生成时的原样冻结，未做编辑（包括其中的中文
标题与注释）；改动其中任何文件都会使报告里的引用失效。

## 快照索引

| 快照 | 对应报告 | 冻结日期（UTC）|
|---|---|---|
| `bench-2026-09-23/` | [基准测试](../基准测试.md) | 2026-09-23 |

## 采集与冻结方式

1. 证据由 `ninfer_bench` 与 [tools/bench/bench.sh](../../tools/bench/bench.sh) 写入工作目录 `out/`，
   用法与输出列的含义见 [基准测试脚本说明](../../tools/bench/README.md)。
2. `out/` 由 `.gitignore` 忽略：它是会随每次运行持续增长的**工作目录**，`just clean` 刻意不删除它。
3. 报告定稿后执行一次 `just evidence bench-<日期>`，把 `out/` 的全部内容复制进本目录（跳过中间产物
   `out/oracle`）。
4. 既有快照只读：新的证据开新目录，不修改旧快照。

## bench-2026-09-23/

### 一、两次完整基准测试

四个跑分目录的结构相同，各含 `manifest.txt`、合并后的 `results.csv` 与每个用例的原始
`all__<case>.csv`：

| 目录 | 制品 | 文档出处 |
|---|---|---|
| `bench-Ternary-Bonsai-2-27B-PQ2_0-20260923T032503Z/` | PQ2_0 | §2 §5.1 |
| `bench-Ternary-Bonsai-2-27B-PTQ1_0-20260923T033934Z/` | PTQ1_0 | §2 §5.2 |
| `bench-Ternary-Bonsai-2-27B-PQ2_0-20260922T161423Z/` | PQ2_0 | §6.6 |
| `bench-Ternary-Bonsai-2-27B-PTQ1_0-20260922T162857Z/` | PTQ1_0 | §6.6 |

`manifest.txt` 记录制品与二进制的 sha256、引擎修订、驱动与固定的参数，是 §2 表的来源；`results.csv`
的全部用例均值与标准差是 §5 各表的来源。

### 二、prefill 双峰的归因实验

`bench-diag-prefill-20260923/` 为本轮实验，`bench-diag-prefill-20260922/` 为早期诊断。所有文件均为
`ninfer_bench -o json` 的逐次数据，逐次速率 = `n_prompt / reps[i].timings.prefill_seconds`：

| 文件 | 实验条件 | 文档出处 |
|---|---|---|
| `bench-diag-prefill-20260923/commands.txt` | 全部命令行与速率换算公式 | §4 §6.4 |
| `a-pptg-r20.json` | `-pg 512,128 -r 20 --warmup 3`，同一进程内的逐次读数 | §6.2 |
| `n-align-r20.json` | 同上，另配 100 ms 采样 | §6.3 |
| `o-align-nwarm-r20.json`、`clocks-align-r20.csv`、`align2-start.txt` | `--warmup 0`，采样与逐次重复对齐 | §6.3 |
| `c-mma0.json`、`d-mma1.json` | `NINFER_TERNARY_MMA=0` / `=1`，`-r 8` | §6.4 |
| `f-nograph.json` | `--no-cuda-graph` | §6.4 |
| `e-chunk128.json`、`i-chunk512.json`、`j-chunk2048.json` | prefill 分块 128 / 512 / 2048 | §6.4 |
| `l-pp1024tg128.json`、`g-pp2048tg128.json` | prompt 1024 / 2048 | §6.4 |
| `k-gen8.json` | `-pg 512,8`，decode 窗口 8 token | §6.4 |
| `b-pp512-r20.json` | `-p 512`，请求中无 decode | §6.4 |
| `m-pp512tg128-noprime.json` | 关闭前缀复用的对照 | §6.4 |
| `h-tg128-r20.json` | `-n 128`，仅 decode | §6.4 |
| `bench-diag-prefill-20260922/d1-kvbf16-r5w1.json`、`d2-kvbf16-r8w3.json`、`d3-panel-r8w3.json` | 高速峰的早期诊断 | §6.2 |

### 三、不在本目录的内容

| 对象 | 体积 | 位置与校验方式 |
|---|---|---|
| 两个 `.ninfer` 制品 | 10.5 GB / 9.3 GB | 不在仓库内（超过 GitHub 单文件上限）；sha256 见 §2 与各 `manifest.txt` |
| 语料 `bench_corpus.ids` | 65536 token | 由 `<ninfer 源树>/bench/fixtures/` 提供；sha256 见 §2 |
| `ninfer_bench` 二进制 | — | 由 `just build-bench` 构建；sha256 见 §2 |
