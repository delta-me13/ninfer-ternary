"""落地自检与 tools/artifact 三元注册。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from nifer_ternary.checks import check_all

PATCHED = Path("/tmp/applytest")
UPSTREAM = Path("/root/ninfer-4090")


def test_checks_fail_on_pristine_upstream() -> None:
    """负控：未打补丁的 ninfer 树上自检必须失败。"""
    if not UPSTREAM.is_dir():
        pytest.skip("本机没有 ninfer-4090 检出")
    findings = check_all(UPSTREAM)
    assert findings, "自检不应为空"
    assert any(not finding.ok for finding in findings)


def test_checks_pass_on_patched_tree() -> None:
    """正控：补丁落盘后自检必须全绿。"""
    if not PATCHED.is_dir():
        pytest.skip("本机没有已打补丁的检出")
    findings = check_all(PATCHED)
    failed = [finding for finding in findings if not finding.ok]
    assert not failed, failed


def _torch_available() -> bool:
    """返回当前解释器是否可导入 torch。"""
    return importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _torch_available(), reason="需要 torch 才能导入 tools/artifact")
def test_artifact_geometry_matches_engine() -> None:
    """tools/artifact 的三元几何必须与引擎侧数值一致。"""
    if not PATCHED.is_dir():
        pytest.skip("本机没有已打补丁的检出")
    import sys

    sys.path.insert(0, str(PATCHED))
    try:
        from tools.artifact import (
            ROW_SPLIT_K128_V1,
            encode_row_split,
            encoded_size,
            row_split_geometry,
        )
    finally:
        sys.path.pop(0)

    expected = {"PTQ1_0_G128": 278_118_400, "PQ2_0_G128": 337_715_200}
    for name, want in expected.items():
        assert name in ROW_SPLIT_K128_V1.formats
        geometry = row_split_geometry(name, (248320, 5120))
        assert geometry.payload_bytes == want
        assert encoded_size("row-split-k128-v1", name, (248320, 5120)) == want
    with pytest.raises(ValueError):
        encode_row_split(None, None, "PQ2_0_G128", (248320, 5120))
