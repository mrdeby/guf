"""Cấu hình ứng dụng và tìm đường dẫn Chrome."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .fsutil import atomic_write_json, read_json

APP_DIR = Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    """Thư mục dữ liệu: biến môi trường GUF_DATA hoặc ./guf_data cạnh mã nguồn."""
    env = os.environ.get("GUF_DATA")
    return Path(env) if env else APP_DIR / "guf_data"


def detect_chrome() -> str | None:
    candidates: list[Path] = []
    if sys.platform.startswith("win"):
        for env in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA"):
            base = os.environ.get(env)
            if base:
                candidates.append(Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe")
    elif sys.platform == "darwin":
        app = "Google Chrome.app/Contents/MacOS/Google Chrome"
        candidates += [Path("/Applications") / app, Path.home() / "Applications" / app]
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
    for c in candidates:
        if c.exists():
            return str(c)
    return None


@dataclass
class Settings:
    chrome_path: str = ""
    max_slots: int = 5
    auto_save_on_close: bool = True
    new_profile_url: str = "https://accounts.google.com/"
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Settings":
        data = read_json(path, default={}) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        s = cls(**known)
        try:
            s.max_slots = max(1, int(s.max_slots))
        except (TypeError, ValueError):
            s.max_slots = 5
        if not s.chrome_path or not Path(s.chrome_path).exists():
            s.chrome_path = detect_chrome() or s.chrome_path
        return s

    def save(self, path: Path) -> None:
        atomic_write_json(path, asdict(self))
