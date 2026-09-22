#!/usr/bin/env bash
# 一键验证折叠基旋转内核：编译独立 harness -> 在真机上跑 -> 用 numpy oracle 比对。
#
# 用法：run_rotation_oracle.sh [输出目录]
#
# 环境变量：
#   NINFER_ROOT   ninfer 源码树根目录（必填，harness 直接编译引擎同一份真代码）
#   NINFER_ARCH   CUDA 架构，默认 89（本机 4090）
#   PYTHON        带 numpy 的解释器，默认 python3
set -euo pipefail

: "${NINFER_ROOT:?请先设置 NINFER_ROOT=<ninfer 源码树根目录>}"
arch="${NINFER_ARCH:-89}"
python="${PYTHON:-python3}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_dir="${1:-${PWD}/bm2out}"
work="${TMPDIR:-/tmp}/nifer-ternary-harness"

mkdir -p "${out_dir}" "${work}"

echo "== 编译 rot_test（arch=sm_${arch}）=="
nvcc -O2 -std=c++20 -arch="sm_${arch}" -I "${NINFER_ROOT}/src" \
  -o "${work}/rot_test" "${here}/harness/rot_test.cu"

echo "== 在真机上运行旋转内核 =="
NINFER_TERNARY_OUT_DIR="${out_dir}/" "${work}/rot_test"

echo "== numpy oracle 比对 =="
NINFER_ROOT="${NINFER_ROOT}" "${python}" "${here}/oracle_rot.py" "${out_dir}/"
