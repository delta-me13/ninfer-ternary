"""ninfer-ternary：把 NInfer 三元移植改动落到 ninfer-4090 检出上的工具集。"""

from .manifest import FileEntry, ManifestError, PatchManifest
from .patchset import FileState, PatchError, PatchReport, apply_patch_set, inspect

__version__ = "0.2.0"

__all__ = [
    "FileEntry",
    "FileState",
    "ManifestError",
    "PatchError",
    "PatchManifest",
    "PatchReport",
    "__version__",
    "apply_patch_set",
    "inspect",
    "main",
]


def main() -> int:
    """包级入口：转交命令行主函数。

    Returns:
        进程退出码。
    """
    from .cli import main as _main

    return _main()
