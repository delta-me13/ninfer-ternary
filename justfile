# ninfer-ternary 的操作入口。
#
# 每条配方只做一件事：把一条已经跑通过的命令序列固定下来。路径全部走变量，换机器改环境变量
# 即可，不需要动这个文件；`just config` 会打印解析结果。
#
#   just                列出全部配方
#   just config         打印解析出来的路径与关键开关
#   just deps           构建依赖自检
#   just check          代码门禁：ruff / mypy / pytest
#   just build          增量构建 sm_89
#   just build 86       增量构建 sm_86
#   just build-tests && just ctest
#   just oracle         旋转内核 oracle
#   just e2e <artifact.ninfer> [prompt]
#   just bench <artifact.ninfer> [suite]
#   just pack PQ2_0     打包三元制品
#   just from-scratch PQ2_0   补丁 -> 编译 -> 测试 -> 打包，全自动

set shell := ["bash", "-euo", "pipefail", "-c"]

# 主干路径。这些是"本机默认值"，全部可以用环境变量覆盖。
build_root    := env_var_or_default("NINFER_BUILD_ROOT", "/data/ninfer-build")
build_root_86 := env_var_or_default("NINFER_BUILD_ROOT_86", "/data/ninfer-build-86")
test_root     := env_var_or_default("NINFER_TEST_BUILD_ROOT", "/data/ninfer-build-test")
bench_root    := env_var_or_default("NINFER_BENCH_BUILD_ROOT", "/data/ninfer-build-bench")
template      := env_var_or_default("NINFER_TERNARY_TEMPLATE", "/data/ninfer-templates/qwen3_8_27b.v2.ninfer")
gguf_dir      := env_var_or_default("NINFER_TERNARY_GGUF_DIR", "/data/Ternary-Bonsai-2-27B-gguf")
artifact_dir  := env_var_or_default("NINFER_TERNARY_ARTIFACT_DIR", "/data/Ternary-Bonsai-2-27B-ninfer")
py            := env_var_or_default("PYTHON", "python3")
py_fallback   := env_var_or_default("NINFER_TERNARY_PYTHON_FALLBACK", "/tmp/venv-torch/bin/python")

# 下面这些要传给被调用的脚本，所以必须 export。
export NINFER_ROOT := env_var_or_default("NINFER_ROOT", "/root/ninfer-4090")
export NINFER_BUILD_ROOT := build_root
export NINFER_CLI := env_var_or_default("NINFER_CLI", build_root + "/apps/ninfer")
export NINFER_BENCH := env_var_or_default("NINFER_BENCH", bench_root + "/bench/ninfer_bench")
export NINFER_JOBS := env_var_or_default("NINFER_JOBS", "16")

# 列出全部配方
default:
    @just --list

# 打印解析后的路径与关键开关（换机器先看这个）
config:
    @printf "%-22s %s\n" \
      "NINFER_ROOT" "{{NINFER_ROOT}}" \
      "NINFER_BUILD_ROOT" "{{build_root}}" \
      "NINFER_BUILD_ROOT_86" "{{build_root_86}}" \
      "NINFER_TEST_BUILD_ROOT" "{{test_root}}" \
      "NINFER_BENCH_BUILD_ROOT" "{{bench_root}}" \
      "NINFER_CLI" "{{NINFER_CLI}}" \
      "NINFER_BENCH" "{{NINFER_BENCH}}" \
      "NINFER_TERNARY_TEMPLATE" "{{template}}" \
      "NINFER_TERNARY_GGUF_DIR" "{{gguf_dir}}" \
      "NINFER_TERNARY_ARTIFACT_DIR" "{{artifact_dir}}" \
      "PYTHON" "{{py}}" \
      "NINFER_TERNARY_PYTHON_FALLBACK" "{{py_fallback}}"

# ---- 依赖 ----------------------------------------------------------------

# 构建依赖自检（Rocky Linux 10）
deps:
    tools/verify/install_deps_rocky10.sh check

# 安装构建依赖（需要 root）
deps-install:
    tools/verify/install_deps_rocky10.sh install

# ---- 代码门禁 ------------------------------------------------------------

# ruff 静态检查
lint:
    uv run ruff check .

# ruff 格式检查
fmt:
    uv run ruff format --check .

# ruff 就地格式化
fmt-fix:
    uv run ruff format .

# mypy 类型检查（只覆盖 src/ 与 tests/，见 pyproject.toml 的 exclude）
typecheck:
    uv run mypy .

# Python 单元测试
pytest *extra:
    uv run pytest -q {{extra}}

# 全部代码门禁
check: lint fmt typecheck pytest

# ---- 构建 ----------------------------------------------------------------

# 增量构建；用法 `just build [86|89] [clean|incremental]`
build arch="89" mode="incremental" *extra:
    #!/usr/bin/env bash
    set -euo pipefail
    root="{{build_root}}"
    if [[ "{{arch}}" == "86" ]]; then root="{{build_root_86}}"; fi
    NINFER_ARCH="{{arch}}" NINFER_BUILD_ROOT="${root}" \
      tools/verify/build.sh "{{mode}}" -- -DNINFER_BUILD_APPS=ON {{extra}}

# 构建测试目标（BUILD_TESTING=ON）
build-tests arch="89" *extra:
    #!/usr/bin/env bash
    set -euo pipefail
    NINFER_ARCH="{{arch}}" NINFER_BUILD_ROOT="{{test_root}}" \
      tools/verify/build.sh incremental -- -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON {{extra}}

# 构建基准目标（NINFER_BUILD_BENCHMARKS=ON）
build-bench *extra:
    #!/usr/bin/env bash
    set -euo pipefail
    NINFER_BUILD_ROOT="{{bench_root}}" \
      tools/verify/build.sh incremental -- -DNINFER_BUILD_BENCHMARKS=ON {{extra}}

# 引擎测试套件；用法 `just ctest -R qwen3_6_27b`
ctest *extra:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{test_root}}"
    ctest --output-on-failure -j "${NINFER_JOBS}" {{extra}}

# ---- 验证 ----------------------------------------------------------------

# 旋转内核 oracle：真机内核 vs numpy FP64
oracle arch="89" out="bm2out":
    NINFER_ARCH={{arch}} PYTHON="{{py}}" tools/verify/run_rotation_oracle.sh "{{out}}"

# 端到端一致性矩阵：内核路径 / 预填充分块 / 投机，判据是贪心 token 序列逐字节一致
e2e artifact prompt="":
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ -n "{{prompt}}" ]]; then
      tools/verify/e2e_ternary.sh "{{artifact}}" "{{prompt}}"
    else
      tools/verify/e2e_ternary.sh "{{artifact}}"
    fi

# 标准化跑分：固定语料 / 重复 / 预热 / 分块，产出 tidy CSV + 环境清单
bench artifact suite="standard":
    tools/bench/bench.sh "{{artifact}}" {{suite}}

# ---- 制品 ----------------------------------------------------------------

# 打包三元制品；kind 取 PQ2_0 或 PTQ1_0
pack kind:
    #!/usr/bin/env bash
    set -euo pipefail
    gguf="{{gguf_dir}}/Ternary-Bonsai-2-27B-{{kind}}.gguf"
    out="{{artifact_dir}}/Ternary-Bonsai-2-27B-{{kind}}.ninfer"
    [[ -f "${gguf}" ]] || { echo "缺少 GGUF: ${gguf}" >&2; exit 1; }
    [[ -f "{{template}}" ]] || { echo "缺少模板: {{template}}" >&2; exit 1; }
    [[ -e "${out}" ]] && { echo "拒绝覆盖已存在的制品: ${out}" >&2; exit 1; }
    # 打包器要真读张量，发行版 python3 没有 numpy 会在装完模板之后才以 traceback 收场。
    py="$(tools/verify/pick_python.sh "{{py}}" "{{py_fallback}}")"
    NINFER_TERNARY_TEMPLATE="{{template}}" NINFER_TERNARY_GGUF="${gguf}" \
      "${py}" tools/pack.py build "${out}"

# 打包前自检：几何 + 解码 + 字节往返证明（只读，不写文件）
pack-check kind:
    #!/usr/bin/env bash
    set -euo pipefail
    gguf="{{gguf_dir}}/Ternary-Bonsai-2-27B-{{kind}}.gguf"
    [[ -f "${gguf}" ]] || { echo "缺少 GGUF: ${gguf}" >&2; exit 1; }
    py="$(tools/verify/pick_python.sh "{{py}}" "{{py_fallback}}")"
    NINFER_TERNARY_TEMPLATE="{{template}}" NINFER_TERNARY_GGUF="${gguf}" \
      "${py}" tools/pack.py check

# 列出制品里的对象与格式
inspect artifact:
    "{{py}}" tools/verify/list_objects.py "{{artifact}}"

# ---- 补丁 ----------------------------------------------------------------

# 打印补丁清单摘要
patch-manifest:
    uv run ninfer-ternary manifest

# 检查 ninfer 检出相对本补丁的状态
patch-status:
    uv run ninfer-ternary status --repo "{{NINFER_ROOT}}"

# 试运行应用补丁（只报告，不写文件）
patch-dry-run:
    uv run ninfer-ternary apply --repo "{{NINFER_ROOT}}" --dry-run

# 应用补丁（覆盖目标检出中的同名文件）
patch-apply:
    uv run ninfer-ternary apply --repo "{{NINFER_ROOT}}"

# 对目标检出执行落地自检
patch-check:
    uv run ninfer-ternary check --repo "{{NINFER_ROOT}}"

# 用检出的当前内容刷新补丁快照 / 清单摘要 / 聚合 diff；额外参数是"新纳入清单"的路径
patch-export *add:
    #!/usr/bin/env bash
    set -euo pipefail
    args=()
    for path in {{add}}; do args+=(--add "${path}"); done
    uv run ninfer-ternary export --repo "{{NINFER_ROOT}}" "${args[@]}"

# ---- 组合 ----------------------------------------------------------------

# 干净检出一条命令走完：补丁 -> 编译 -> 测试 -> 打包三元制品
from-scratch kind:
    #!/usr/bin/env bash
    set -euo pipefail
    just patch-apply
    just patch-check
    just build
    just build-tests
    just ctest
    just pack "{{kind}}"

# 代码门禁 + 旋转 oracle + 端到端矩阵；不含构建，跑之前先 build 与 build-tests
verify artifact:
    #!/usr/bin/env bash
    set -euo pipefail
    just check
    just oracle
    just e2e "{{artifact}}"

# 从依赖自检一路走到端到端矩阵
all artifact:
    #!/usr/bin/env bash
    set -euo pipefail
    just deps
    just check
    just build
    just build-tests
    just ctest
    just oracle
    just e2e "{{artifact}}"
