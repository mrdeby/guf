"""Cold State: nén / bung một thư mục user-data-dir của Chrome.

Cold State là file .zip chứa toàn bộ user-data-dir (gồm cả "Local State" chứa khoá
giải mã cookie) nhưng BỎ các thư mục cache có thể tạo lại, nên chỉ còn vài MB.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Iterator

from .fsutil import clear_dir

SLOT_MARKER = ".guf_slot.json"

# Thư mục Chrome tự tạo lại được -> không đưa vào Cold State.
EXCLUDE_DIR_NAMES = frozenset({
    "Cache", "Code Cache", "GPUCache", "DawnCache", "DawnGraphiteCache",
    "DawnWebGPUCache", "GrShaderCache", "GraphiteDawnCache", "ShaderCache",
    "CacheStorage", "ScriptCache", "Crashpad", "Crash Reports",
    "component_crx_cache", "extensions_crx_cache",
    "OptimizationGuidePredictionModels", "optimization_guide_model_store",
    "Safe Browsing", "segmentation_platform", "BrowserMetrics",
    "screen_ai", "SODA", "SODALanguagePacks", "WidevineCdm",
    "MediaFoundationWidevineCdm", "hyphen-data", "ZxcvbnData",
    "OnDeviceHeadSuggestModel", "MEIPreload", "Download Service",
})

# File khoá / tạm của Chrome và file đánh dấu slot của GUF.
EXCLUDE_FILE_NAMES = frozenset({
    "SingletonLock", "SingletonCookie", "SingletonSocket", "lockfile",
    "BrowserMetrics-spare.pma", "CrashpadMetrics-active.pma", SLOT_MARKER,
})


class ColdStateError(Exception):
    pass


def iter_state_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIR_NAMES and not os.path.islink(os.path.join(dirpath, d))
        ]
        for name in filenames:
            if name in EXCLUDE_FILE_NAMES:
                continue
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                continue
            yield Path(full)


def pack(src_dir: Path, dest_zip: Path) -> int:
    """Nén user-data-dir thành Cold State. Chrome phải đã đóng. Trả về dung lượng file."""
    src_dir, dest_zip = Path(src_dir), Path(dest_zip)
    if not (src_dir / "Local State").exists() and not (src_dir / "Default").is_dir():
        raise ColdStateError(f"{src_dir} không chứa dữ liệu Chrome (chưa từng mở Chrome?)")
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest_zip.with_name(dest_zip.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in iter_state_files(src_dir):
                try:
                    zf.write(f, f.relative_to(src_dir).as_posix())
                except FileNotFoundError:
                    continue  # file tạm biến mất trong lúc nén
        os.replace(tmp, dest_zip)   # thay thế nguyên tử: không bao giờ còn nửa file
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest_zip.stat().st_size


def unpack(src_zip: Path, dest_dir: Path) -> None:
    """Xoá sạch dest_dir rồi bung Cold State vào."""
    src_zip, dest_dir = Path(src_zip), Path(dest_dir)
    with zipfile.ZipFile(src_zip) as zf:
        root = dest_dir.resolve()
        for member in zf.infolist():
            target = (dest_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ColdStateError(f"Đường dẫn không hợp lệ trong Cold State: {member.filename}")
        clear_dir(dest_dir)
        zf.extractall(dest_dir)
