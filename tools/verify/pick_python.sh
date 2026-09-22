#!/usr/bin/env bash
#
# 挑出一个带 numpy 的解释器。
#
# 打包器要真读张量，而发行版自带的 python3 通常没有 numpy；这里按
# "显式指定 -> 首选 -> 已知回退" 的顺序探测，把可用路径写到 stdout。
#
# 用法：pick_python.sh <首选解释器> [回退解释器...]
#
# 退出码：0 找到；3 一个都没有（stderr 给出补救办法）。

set -euo pipefail

has_numpy() {
  local interpreter="${1}"
  [[ -n "${interpreter}" ]] || return 1
  if [[ "${interpreter}" == */* ]]; then
    [[ -x "${interpreter}" ]] || return 1
  else
    command -v "${interpreter}" >/dev/null 2>&1 || return 1
  fi
  "${interpreter}" -c "import numpy" >/dev/null 2>&1
}

for candidate in "${NINFER_TERNARY_PYTHON:-}" "${@}"; do
  if has_numpy "${candidate}"; then
    printf "%s\n" "${candidate}"
    exit 0
  fi
done

echo "找不到带 numpy 的解释器：用 PYTHON=<解释器> 指定，或 uv add numpy 后重跑" >&2
exit 3
