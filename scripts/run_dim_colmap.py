#!/usr/bin/env python
"""Wrapper that forwards to the reusable CLI in uav_pipeline.cli."""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for path in (SRC_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from uav_pipeline.cli import main  # type: ignore  # noqa: E402


if __name__ == "__main__":
    main()
