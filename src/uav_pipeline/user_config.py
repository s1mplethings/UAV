from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_ENV = "UAV_PIPELINE_CONFIG"
DEFAULT_CONFIG_NAME = ".uav_pipeline_config.json"


def _config_path() -> Path:
    env_path = os.environ.get(CONFIG_ENV)
    if env_path:
        return Path(env_path).expanduser()
    return Path.home() / DEFAULT_CONFIG_NAME


def load_config() -> dict[str, Any]:
    path = _config_path()
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def save_config(cfg: dict[str, Any]) -> None:
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        pass


def load_section(name: str) -> dict[str, Any]:
    cfg = load_config()
    section = cfg.get(name)
    return section if isinstance(section, dict) else {}


def update_section(name: str, data: dict[str, Any]) -> None:
    cfg = load_config()
    cfg[name] = data
    save_config(cfg)


def config_path_str() -> str:
    return str(_config_path())
