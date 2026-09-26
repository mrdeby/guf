"""Khởi chạy / đóng Chrome và kiểm tra user-data-dir có đang bị Chrome giữ hay không."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable

IS_WIN = sys.platform.startswith("win")

BASE_ARGS = (
    "--profile-directory=Default",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-mode",   # đóng cửa sổ là thoát hẳn -> app biết để lưu Cold State
)


def build_args(chrome_path: str, user_data_dir: Path,
               extra_args: Iterable[str] = (), url: str | None = None) -> list[str]:
    args = [chrome_path, f"--user-data-dir={user_data_dir}", *BASE_ARGS, *extra_args]
    if url:
        args.append(url)
    return args


def launch(chrome_path: str, user_data_dir: Path,
           extra_args: Iterable[str] = (), url: str | None = None) -> subprocess.Popen:
    kwargs: dict = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, close_fds=True)
    if IS_WIN:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(build_args(chrome_path, user_data_dir, extra_args, url), **kwargs)


def request_close(proc: subprocess.Popen) -> None:
    """Yêu cầu Chrome đóng êm (không kill cứng) để cookie được ghi xuống đĩa."""
    if proc.poll() is not None:
        return
    if IS_WIN:
        subprocess.run(["taskkill", "/PID", str(proc.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        proc.terminate()


def lock_held(user_data_dir: Path) -> bool:
    """True nếu có một tiến trình Chrome đang dùng user_data_dir này."""
    user_data_dir = Path(user_data_dir)
    if IS_WIN:
        lock = user_data_dir / "lockfile"
        if not lock.exists():
            return False
        try:
            # Chrome mở lockfile không chia sẻ quyền ghi -> mở được nghĩa là khoá cũ.
            with open(lock, "a"):
                return False
        except OSError:
            return True

    lock = user_data_dir / "SingletonLock"
    if not os.path.islink(lock):
        return False
    try:
        host, _, pid = os.readlink(lock).rpartition("-")
        pid_num = int(pid)
    except (OSError, ValueError):
        return True
    if host != socket.gethostname():
        return True  # khoá từ máy khác: coi như đang dùng cho an toàn
    try:
        os.kill(pid_num, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
