from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    if getattr(sys, "frozen", False):
        return
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


def main() -> None:
    _ensure_src_on_path()
    from uav_pipeline.workbench_gui import main as workbench_main

    workbench_main()


if __name__ == "__main__":
    main()
