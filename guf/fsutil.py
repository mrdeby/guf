"""Tiện ích thao tác file dùng chung."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def atomic_write_json(path: Path, data: Any) -> None:
    """Ghi JSON ra file tạm rồi thay thế, tránh hỏng file khi tắt ngang."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _force_remove(func, path, *_):
    # Windows: file read-only không xoá được -> bỏ cờ read-only rồi thử lại.
    os.chmod(path, stat.S_IWRITE)
    func(path)


def rmtree(path: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force_remove)
    else:
        shutil.rmtree(path, onerror=_force_remove)


def clear_dir(path: Path) -> None:
    """Xoá toàn bộ nội dung bên trong thư mục nhưng giữ lại thư mục."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            rmtree(child)
        else:
            try:
                child.unlink()
            except PermissionError:
                _force_remove(os.unlink, child)


def human_size(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"
