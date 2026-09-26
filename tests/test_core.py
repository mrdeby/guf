import os
import socket
import sys
import zipfile

import pytest

from guf import chrome, coldstate
from guf.slots import SlotManager
from guf.store import ProfileStore


def make_profile_dir(root):
    (root / "Default" / "Cache").mkdir(parents=True)
    (root / "Default" / "Network").mkdir(parents=True)
    (root / "Local State").write_text('{"os_crypt": {}}')
    (root / "Default" / "Network" / "Cookies").write_bytes(b"cookies")
    (root / "Default" / "Cache" / "data_0").write_bytes(b"x" * 1000)
    (root / "Default" / "Service Worker" / "CacheStorage").mkdir(parents=True)
    (root / "Default" / "Service Worker" / "CacheStorage" / "big").write_bytes(b"y")
    (root / "Default" / "Service Worker" / "Database").mkdir()
    (root / "Default" / "Service Worker" / "Database" / "db").write_bytes(b"sw")
    (root / "lockfile").write_text("")
    (root / coldstate.SLOT_MARKER).write_text("{}")


def test_pack_excludes_cache_and_locks(tmp_path):
    src = tmp_path / "slot"
    make_profile_dir(src)
    dest = tmp_path / "cold" / "p.zip"
    size = coldstate.pack(src, dest)
    assert size == dest.stat().st_size
    names = set(zipfile.ZipFile(dest).namelist())
    assert names == {
        "Local State",
        "Default/Network/Cookies",
        "Default/Service Worker/Database/db",
    }
    assert not (dest.parent / "p.zip.tmp").exists()


def test_pack_rejects_empty_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(coldstate.ColdStateError):
        coldstate.pack(tmp_path / "empty", tmp_path / "x.zip")


def test_unpack_replaces_previous_content(tmp_path):
    src = tmp_path / "a"
    make_profile_dir(src)
    coldstate.pack(src, tmp_path / "a.zip")
    slot = tmp_path / "slot"
    (slot / "Default").mkdir(parents=True)
    (slot / "Default" / "other_account").write_text("old")
    coldstate.unpack(tmp_path / "a.zip", slot)
    assert not (slot / "Default" / "other_account").exists()
    assert (slot / "Default" / "Network" / "Cookies").read_bytes() == b"cookies"


def test_unpack_rejects_path_traversal(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    slot = tmp_path / "slot"
    (slot).mkdir()
    (slot / "keep").write_text("k")
    with pytest.raises(coldstate.ColdStateError):
        coldstate.unpack(bad, slot)
    assert (slot / "keep").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_pick_slot_order(tmp_path):
    sm = SlotManager(tmp_path, 3)
    sm.set_marker(0, "a", last_used=100)
    sm.set_marker(1, "b", last_used=50)
    assert sm.pick_slot("b", set()) == 1           # đã có sẵn
    assert sm.pick_slot("c", set()) == 2           # slot trống
    sm.set_marker(2, "c", last_used=200)
    assert sm.pick_slot("d", set()) == 1           # dùng lâu nhất
    assert sm.pick_slot("d", {1}) == 0             # bỏ slot đang mở
    assert sm.pick_slot("d", {0, 1, 2}) is None
    assert sm.find("c") == 2


def test_update_marker(tmp_path):
    sm = SlotManager(tmp_path, 1)
    sm.update_marker(0, dirty=True)                # slot trống: bỏ qua
    assert sm.state(0).profile_id is None
    sm.set_marker(0, "a", dirty=False)
    sm.update_marker(0, dirty=True)
    st = sm.state(0)
    assert st.profile_id == "a" and st.dirty


def test_store_roundtrip(tmp_path):
    s = ProfileStore(tmp_path / "profiles.json")
    p = s.add("acc1@gmail.com")
    s.update(p.id, saved_at=1.0, cold_size=10)
    s2 = ProfileStore(tmp_path / "profiles.json")
    assert s2.get(p.id).name == "acc1@gmail.com"
    assert s2.get(p.id).cold_size == 10
    s2.remove(p.id)
    assert len(ProfileStore(tmp_path / "profiles.json")) == 0


def test_build_args(tmp_path):
    args = chrome.build_args("chrome", tmp_path, ["--x"], "https://a")
    assert args[0] == "chrome"
    assert f"--user-data-dir={tmp_path}" in args
    assert args[-2:] == ["--x", "https://a"]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="SingletonLock chỉ có trên POSIX")
def test_lock_held_posix(tmp_path):
    assert not chrome.lock_held(tmp_path)
    lock = tmp_path / "SingletonLock"
    os.symlink(f"{socket.gethostname()}-{os.getpid()}", lock)
    assert chrome.lock_held(tmp_path)
    lock.unlink()
    os.symlink(f"{socket.gethostname()}-99999999", lock)   # tiến trình không tồn tại
    assert not chrome.lock_held(tmp_path)
