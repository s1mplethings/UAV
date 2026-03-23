from __future__ import annotations

import os
from typing import Sequence


_WINDOWS_RETURN_CODE_HINTS: dict[int, str] = {
    0xC000013A: "Windows 报告子进程被外部中断（常见于窗口被关闭、Ctrl+C、系统终止该进程，或外部安全软件/驱动中断）。",
    0xC0000005: "Windows 报告访问冲突（Access Violation）；这通常不是 Python 级报错，而是底层原生库/驱动崩溃。",
    0xC0000409: "Windows 报告栈缓冲区溢出/快速失败；这通常来自底层原生库崩溃。",
    0xC0000142: "Windows 报告 DLL 初始化失败；常见于运行时依赖缺失、版本不兼容或显卡/Qt/OpenCV 相关初始化失败。",
    0xC000007B: "Windows 报告无效映像格式；常见于 32/64 位依赖不匹配或 DLL 冲突。",
}


def describe_return_code(returncode: int) -> str | None:
    if os.name != "nt":
        return None
    normalized = returncode & 0xFFFFFFFF
    hint = _WINDOWS_RETURN_CODE_HINTS.get(normalized)
    if hint is None:
        return None
    return f"{hint} (return code: {returncode}, hex: 0x{normalized:08X})"


def format_command_failure(*, returncode: int, cmd: Sequence[str]) -> str:
    cmd_text = " ".join(str(part) for part in cmd)
    detail = describe_return_code(returncode)
    if detail:
        return f"命令执行失败：{cmd_text}\n{detail}"
    return f"命令执行失败：{cmd_text}\nreturn code: {returncode}"
