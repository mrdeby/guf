"""Slot: các thư mục user-data-dir "nóng" (tối đa N, mặc định 5).

Mỗi slot có file đánh dấu .guf_slot.json cho biết đang chứa profile nào,
có thay đổi chưa lưu Cold State (dirty) và lần dùng cuối.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from . import chrome
from .coldstate import SLOT_MARKER
from .fsutil import atomic_write_json, clear_dir, read_json


@dataclass
class SlotState:
    index: int
    path: Path
    profile_id: str | None = None
    dirty: bool = False
    last_used: float = 0.0


class SlotManager:
    def __init__(self, root: Path, count: int):
        self.root = Path(root)
        self.count = count
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, i: int) -> Path:
        return self.root / f"slot{i + 1}"

    def state(self, i: int) -> SlotState:
        data = read_json(self.path(i) / SLOT_MARKER, default={}) or {}
        return SlotState(
            index=i,
            path=self.path(i),
            profile_id=data.get("profile_id"),
            dirty=bool(data.get("dirty")),
            last_used=float(data.get("last_used") or 0),
        )

    def states(self) -> list[SlotState]:
        return [self.state(i) for i in range(self.count)]

    def set_marker(self, i: int, profile_id: str, dirty: bool = False,
                   last_used: float | None = None) -> None:
        atomic_write_json(self.path(i) / SLOT_MARKER, {
            "profile_id": profile_id,
            "dirty": dirty,
            "last_used": time.time() if last_used is None else last_used,
        })

    def update_marker(self, i: int, **changes) -> None:
        st = self.state(i)
        if not st.profile_id:
            return
        data = {"profile_id": st.profile_id, "dirty": st.dirty, "last_used": st.last_used}
        data.update(changes)
        atomic_write_json(self.path(i) / SLOT_MARKER, data)

    def clear(self, i: int) -> None:
        clear_dir(self.path(i))

    def find(self, profile_id: str) -> int | None:
        for s in self.states():
            if s.profile_id == profile_id:
                return s.index
        return None

    def is_locked(self, i: int) -> bool:
        return chrome.lock_held(self.path(i))

    def pick_slot(self, profile_id: str, unavailable: set[int]) -> int | None:
        """Chọn slot để mở profile: slot đang chứa sẵn > slot trống > slot dùng lâu nhất."""
        states = [s for s in self.states() if s.index not in unavailable]
        if not states:
            return None
        for s in states:
            if s.profile_id == profile_id:
                return s.index
        for s in states:
            if not s.profile_id:
                return s.index
        return min(states, key=lambda s: s.last_used).index
