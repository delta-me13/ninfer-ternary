"""命令行入口：查看清单、检查状态、应用补丁、执行落地自检。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .checks import check_all
from .manifest import ManifestError, PatchManifest, changed_files_root, default_manifest_path
from .patchset import FileState, PatchError, apply_patch_set, inspect

_LOGGER = logging.getLogger("nifer_ternary")


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        配置好的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="nifer-ternary",
        description="将 NInfer 三元（Ternary Bonsai 2 27B）移植改动应用到 ninfer-4090 检出。",
    )
    parser.add_argument("--version", action="version", version=f"nifer-ternary {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    parser.add_argument("--manifest", type=Path, default=None, help="补丁清单路径")
    parser.add_argument("--snapshot", type=Path, default=None, help="补丁快照目录")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("manifest", help="打印补丁清单摘要")

    status = sub.add_parser("status", help="检查目标检出相对本补丁的状态")
    status.add_argument("--repo", type=Path, required=True, help="ninfer 源码树根目录")

    apply_cmd = sub.add_parser("apply", help="把补丁快照覆盖到目标检出")
    apply_cmd.add_argument("--repo", type=Path, required=True, help="ninfer 源码树根目录")
    apply_cmd.add_argument("--force", action="store_true", help="即使文件已分叉也覆盖")
    apply_cmd.add_argument("--dry-run", action="store_true", help="只报告，不写文件")

    check = sub.add_parser("check", help="对目标检出执行落地自检")
    check.add_argument("--repo", type=Path, required=True, help="ninfer 源码树根目录")
    return parser


def _load(args: argparse.Namespace) -> PatchManifest:
    """按命令行参数加载补丁清单。

    Args:
        args: 解析后的命令行参数。

    Returns:
        补丁清单。

    Raises:
        ManifestError: 清单缺失或结构不符。
    """
    path = args.manifest if args.manifest is not None else default_manifest_path()
    return PatchManifest.load(path)


def _cmd_manifest(manifest: PatchManifest) -> int:
    """打印补丁清单摘要。

    Args:
        manifest: 补丁清单。

    Returns:
        进程退出码。
    """
    print(f"目标仓库   : {manifest.target_repository}")
    print(f"目标提交   : {manifest.target_commit}")
    print(f"改动来源   : {manifest.source_repository}")
    print(f"来源版本   : {manifest.source_revision}")
    print(f"来源基座   : {manifest.source_baseline}")
    total = len(manifest.files)
    print(f"文件总数   : {total}（新增 {manifest.added_count}，修改 {manifest.modified_count}）")
    for entry in manifest.files:
        print(f"  {entry.status:8s} {entry.path}")
    return 0


def _cmd_status(args: argparse.Namespace, manifest: PatchManifest) -> int:
    """打印目标检出的状态。

    Args:
        args: 解析后的命令行参数。
        manifest: 补丁清单。

    Returns:
        进程退出码：全部分叉或无冲突时为 0，存在分叉时为 1。
    """
    report = inspect(args.repo, manifest)
    for state in (FileState.PATCHED, FileState.DIVERGED, FileState.MISSING, FileState.PRISTINE):
        count = report.count(state)
        if count:
            print(f"{state.value:9s}: {count}")
    for item in report.results:
        if item.state in (FileState.DIVERGED, FileState.MISSING):
            print(f"  {item.state.value:9s} {item.entry.path}")
    if report.is_fully_applied:
        print("结论       : 本补丁已完整应用")
        return 0
    if report.is_pristine:
        print("结论       : 目标树处于上游原状，可以安全应用")
        return 0
    print("结论       : 目标树既非上游原状、也非本补丁结果，请人工确认后再 --force")
    return 1


def _cmd_apply(args: argparse.Namespace, manifest: PatchManifest) -> int:
    """应用补丁快照。

    Args:
        args: 解析后的命令行参数。
        manifest: 补丁清单。

    Returns:
        进程退出码。
    """
    snapshot = args.snapshot if args.snapshot is not None else changed_files_root()
    report = apply_patch_set(args.repo, manifest, snapshot, force=args.force, dry_run=args.dry_run)
    verb = "将写入" if args.dry_run else "已写入"
    print(f"{verb} {len(report.results)} 个文件（快照 {snapshot}）")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """执行落地自检。

    Args:
        args: 解析后的命令行参数。

    Returns:
        进程退出码：全部通过为 0，否则为 1。
    """
    findings = check_all(args.repo)
    failed = 0
    for finding in findings:
        if not finding.ok:
            failed += 1
        print(f"  [{'ok' if finding.ok else 'FAIL'}] {finding.path}: {finding.message}")
    print(f"自检结论   : {len(findings) - failed}/{len(findings)} 通过")
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    """命令行主入口。

    Args:
        argv: 参数列表；默认取 sys.argv[1:]。

    Returns:
        进程退出码。
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "check":
            return _cmd_check(args)
        manifest = _load(args)
        if args.command == "manifest":
            return _cmd_manifest(manifest)
        if args.command == "status":
            return _cmd_status(args, manifest)
        return _cmd_apply(args, manifest)
    except (ManifestError, PatchError) as error:
        _LOGGER.error("%s", error)
        print(f"错误: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
