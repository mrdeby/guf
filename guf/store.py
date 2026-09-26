"""Danh sách profile (profiles.json)."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .fsutil import atomic_write_json, read_json


@dataclass
class Profile:
    id: str
    name: str
    note: str = ""
    created_at: float = 0.0
    saved_at: float | None = None   # lần lưu Cold State gần nhất
    cold_size: int = 0              # dung lượng file Cold State (byte)


class ProfileStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._profiles: dict[str, Profile] = {}
        self.load()

    def load(self) -> None:
        data = read_json(self.path, default={}) or {}
        self._profiles.clear()
        for item in data.get("profiles", []):
            fields = {k: v for k, v in item.items() if k in Profile.__dataclass_fields__}
            try:
                p = Profile(**fields)
            except TypeError:
                continue
            self._profiles[p.id] = p

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {"version": 1, "profiles": [asdict(p) for p in self._profiles.values()]},
        )

    def all(self) -> list[Profile]:
        return sorted(self._profiles.values(), key=lambda p: (p.name.lower(), p.id))

    def get(self, pid: str | None) -> Profile | None:
        return self._profiles.get(pid) if pid else None

    def add(self, name: str) -> Profile:
        pid = uuid.uuid4().hex[:8]
        while pid in self._profiles:
            pid = uuid.uuid4().hex[:8]
        p = Profile(id=pid, name=name, created_at=time.time())
        self._profiles[pid] = p
        self.save()
        return p

    def update(self, pid: str, **fields) -> None:
        p = self._profiles[pid]
        for k, v in fields.items():
            setattr(p, k, v)
        self.save()

    def remove(self, pid: str) -> None:
        self._profiles.pop(pid, None)
        self.save()

    def __len__(self) -> int:
        return len(self._profiles)
