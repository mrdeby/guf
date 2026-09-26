"""Giao diện PyQt6 quản lý profile Google/Chrome."""

from __future__ import annotations

import shlex
import subprocess
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from . import chrome, coldstate
from .config import Settings
from .fsutil import human_size
from .slots import SlotManager, SlotState
from .store import Profile, ProfileStore

COLUMNS = ["Tên", "ID", "Trạng thái", "Cold State", "Lưu lần cuối"]


class _Bridge(QObject):
    """Đưa kết quả từ luồng nền về luồng giao diện."""
    done = pyqtSignal(object, object, object)  # callback, kết quả, lỗi


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt")
        self.setMinimumWidth(560)
        form = QFormLayout(self)

        self.chrome_edit = QLineEdit(settings.chrome_path)
        browse = QPushButton("Chọn...")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.chrome_edit)
        row.addWidget(browse)
        form.addRow("Đường dẫn Chrome:", row)

        self.slots_spin = QSpinBox()
        self.slots_spin.setRange(1, 20)
        self.slots_spin.setValue(settings.max_slots)
        form.addRow("Số profile mở tối đa (slot):", self.slots_spin)

        self.auto_save = QCheckBox("Tự lưu Cold State khi đóng Chrome")
        self.auto_save.setChecked(settings.auto_save_on_close)
        form.addRow("", self.auto_save)

        self.url_edit = QLineEdit(settings.new_profile_url)
        form.addRow("Trang mở khi thêm profile:", self.url_edit)

        self.args_edit = QLineEdit(" ".join(shlex.quote(a) for a in settings.extra_args))
        form.addRow("Tham số Chrome thêm:", self.args_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file chạy Chrome", self.chrome_edit.text())
        if path:
            self.chrome_edit.setText(path)

    def apply_to(self, settings: Settings) -> None:
        settings.chrome_path = self.chrome_edit.text().strip()
        settings.max_slots = self.slots_spin.value()
        settings.auto_save_on_close = self.auto_save.isChecked()
        settings.new_profile_url = self.url_edit.text().strip()
        settings.extra_args = shlex.split(self.args_edit.text(), posix=True)


class MainWindow(QMainWindow):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.data_dir / "settings.json"
        self.settings = Settings.load(self.settings_path)
        self.settings.save(self.settings_path)
        self.store = ProfileStore(self.data_dir / "profiles.json")
        self.cold_dir = self.data_dir / "cold"
        self.cold_dir.mkdir(parents=True, exist_ok=True)
        self.slots = SlotManager(self.data_dir / "slots", self.settings.max_slots)

        self.procs: dict[int, subprocess.Popen] = {}   # slot -> Chrome do app mở
        self.busy: set[int] = set()                    # slot đang nén / bung
        self.pending_save: set[int] = set()            # slot chờ Chrome đóng để lưu
        self._last_running: set[int] = set()

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="guf-io")
        self.bridge = _Bridge()
        self.bridge.done.connect(self._on_job_done)

        self._build_ui()
        self.refresh()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self.timer.start(1000)
        QTimer.singleShot(0, self._sync_dirty_slots)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.setWindowTitle("GUF - Quản lý profile Google")
        self.resize(1100, 700)

        self.btn_add = QPushButton("Thêm profile")
        self.btn_cold = QPushButton("Cold State")
        self.btn_open = QPushButton("Mở profile")
        self.btn_rename = QPushButton("Đổi tên")
        self.btn_delete = QPushButton("Xoá")
        self.btn_settings = QPushButton("Cài đặt")
        self.btn_add.setToolTip("Tạo profile mới và mở Chrome để đăng nhập")
        self.btn_cold.setToolTip("Lưu profile đang chọn thành Cold State (cần đóng Chrome)")
        self.btn_open.setToolTip("Mở profile đang chọn (tự bung Cold State nếu cần)")
        self.btn_add.clicked.connect(self.add_profile)
        self.btn_cold.clicked.connect(self.cold_state_selected)
        self.btn_open.clicked.connect(self.open_selected)
        self.btn_rename.clicked.connect(self.rename_selected)
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_settings.clicked.connect(self.open_settings)

        bar = QHBoxLayout()
        for b in (self.btn_add, self.btn_cold, self.btn_open):
            b.setMinimumHeight(34)
            bar.addWidget(b)
        bar.addSpacing(20)
        for b in (self.btn_rename, self.btn_delete):
            bar.addWidget(b)
        bar.addStretch(1)
        bar.addWidget(self.btn_settings)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Tìm theo tên hoặc ID...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self.refresh)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, len(COLUMNS)):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.doubleClicked.connect(lambda _: self.open_selected())

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.filter_edit)
        lv.addWidget(self.table)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.addWidget(QLabel("Slot (thư mục profile đầy đủ):"))
        self.slot_list = QListWidget()
        rv.addWidget(self.slot_list)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(right)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 1)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(top)
        vsplit.addWidget(self.log_view)
        vsplit.setStretchFactor(0, 4)
        vsplit.setStretchFactor(1, 1)

        central = QWidget()
        main = QVBoxLayout(central)
        main.addLayout(bar)
        main.addWidget(vsplit)
        self.setCentralWidget(central)

    def log(self, msg: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.statusBar().showMessage(msg, 8000)

    def cold_path(self, pid: str) -> Path:
        return self.cold_dir / f"{pid}.zip"

    def running_slots(self) -> set[int]:
        running = set()
        for i in range(self.slots.count):
            proc = self.procs.get(i)
            if (proc is not None and proc.poll() is None) or self.slots.is_locked(i):
                running.add(i)
        return running

    def selected_ids(self) -> list[str]:
        rows = sorted({idx.row() for idx in self.table.selectionModel().selectedRows()})
        ids = []
        for r in rows:
            item = self.table.item(r, 0)
            if item is not None:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _select(self, pid: str) -> None:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == pid:
                self.table.selectRow(r)
                self.table.scrollToItem(item)
                return

    def _status_text(self, p: Profile, slot: SlotState | None, running: set[int]) -> str:
        if slot is not None:
            n = slot.index + 1
            if slot.index in self.busy:
                return f"Đang xử lý... (slot {n})"
            if slot.index in running:
                extra = ", chờ lưu khi đóng" if slot.index in self.pending_save else ""
                return f"Đang mở (slot {n}{extra})"
            if slot.dirty:
                return f"Slot {n} - chưa lưu Cold State"
            return f"Slot {n} (sẵn sàng)"
        if self.cold_path(p.id).exists():
            return "Cold State"
        return "Chưa có Cold State"

    def refresh(self) -> None:
        selected = set(self.selected_ids())
        states = self.slots.states()
        running = self.running_slots()
        self._last_running = running
        slot_of = {s.profile_id: s for s in states if s.profile_id}
        flt = self.filter_edit.text().strip().lower()

        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for p in self.store.all():
            if flt and flt not in p.name.lower() and flt not in p.id:
                continue
            r = self.table.rowCount()
            self.table.insertRow(r)
            cold = self.cold_path(p.id)
            values = [
                p.name,
                p.id,
                self._status_text(p, slot_of.get(p.id), running),
                human_size(cold.stat().st_size) if cold.exists() else "-",
                time.strftime("%d/%m/%Y %H:%M", time.localtime(p.saved_at)) if p.saved_at else "-",
            ]
            for c, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setData(Qt.ItemDataRole.UserRole, p.id)
                self.table.setItem(r, c, item)
            if p.id in selected:
                self.table.selectRow(r)
        self.table.setSortingEnabled(True)

        self.slot_list.clear()
        for s in states:
            p = self.store.get(s.profile_id)
            if not s.profile_id:
                text = "trống"
            else:
                name = p.name if p else f"(đã xoá) {s.profile_id}"
                if s.index in self.busy:
                    flag = "đang xử lý"
                elif s.index in running:
                    flag = "đang mở"
                else:
                    flag = "chưa lưu" if s.dirty else "rảnh"
                text = f"{name} - {flag}"
            self.slot_list.addItem(f"Slot {s.index + 1}: {text}")

        self.statusBar().showMessage(
            f"{len(self.store)} profile - {len(running)}/{self.slots.count} slot đang mở")

    # -------------------------------------------------------- nền (I/O)
    def _run(self, fn: Callable[[], object], on_done: Callable[[object, BaseException | None], None]) -> None:
        def finished(fut: Future) -> None:
            err = fut.exception()
            self.bridge.done.emit(on_done, None if err else fut.result(), err)
        self.executor.submit(fn).add_done_callback(finished)

    def _on_job_done(self, cb, result, err) -> None:
        cb(result, err)

    def _report_error(self, title: str, err: BaseException) -> None:
        self.log(f"LỖI - {title}: {err}")
        traceback.print_exception(type(err), err, err.__traceback__)
        QMessageBox.critical(self, "Lỗi", f"{title}\n\n{err}")

    def _mark_saved(self, pid: str, size: int) -> None:
        if self.store.get(pid):
            self.store.update(pid, saved_at=time.time(), cold_size=size)

    def _check_chrome(self) -> bool:
        path = self.settings.chrome_path
        if path and Path(path).exists():
            return True
        QMessageBox.warning(self, "Chưa có Chrome",
                            "Không tìm thấy Chrome. Anh vào Cài đặt để chọn file chạy Chrome.")
        return False

    # ------------------------------------------------------------ Thêm mới
    def add_profile(self) -> None:
        if not self._check_chrome():
            return
        name, ok = QInputDialog.getText(self, "Thêm profile", "Tên profile (vd: email tài khoản):")
        if not ok:
            return
        name = name.strip() or f"Profile {len(self.store) + 1}"
        p = self.store.add(name)
        self.log(f"Đã tạo profile «{p.name}» ({p.id}). Đăng nhập Google trong cửa sổ Chrome, "
                 f"xong nhấn Cold State để lưu.")
        self.refresh()
        self._select(p.id)
        self.open_profile(p.id, url=self.settings.new_profile_url)

    # ---------------------------------------------------------------- Mở
    def open_selected(self) -> None:
        ids = self.selected_ids()
        if not ids:
            QMessageBox.information(self, "Mở profile", "Anh chọn profile trong danh sách trước.")
            return
        for pid in ids:
            self.open_profile(pid)

    def open_profile(self, pid: str, url: str | None = None) -> None:
        p = self.store.get(pid)
        if p is None or not self._check_chrome():
            return
        running = self.running_slots()
        cur = self.slots.find(pid)
        if cur is not None:
            if cur in self.busy:
                self.log(f"«{p.name}» đang được xử lý, anh đợi chút.")
                return
            if cur in running:
                # Chrome đang chạy với thư mục này: mở thêm cửa sổ trong phiên hiện có.
                chrome.launch(self.settings.chrome_path, self.slots.path(cur), self.settings.extra_args, url)
                self.log(f"«{p.name}» đang mở ở slot {cur + 1}.")
                return
            self.log(f"«{p.name}» đã có sẵn ở slot {cur + 1}, mở luôn.")
            self._launch_in_slot(cur, pid, url)
            return

        idx = self.slots.pick_slot(pid, running | self.busy)
        if idx is None:
            QMessageBox.warning(self, "Hết slot",
                                f"Đã mở tối đa {self.slots.count} profile. Đóng bớt Chrome rồi mở lại.")
            return

        old = self.slots.state(idx)
        evict_id = old.profile_id if (old.profile_id and old.dirty and self.store.get(old.profile_id)) else None
        cold = self.cold_path(pid)
        slot_dir = self.slots.path(idx)
        evict_cold = self.cold_path(evict_id) if evict_id else None

        self.busy.add(idx)
        self.refresh()
        if evict_id:
            self.log(f"Slot {idx + 1}: lưu Cold State «{self.store.get(evict_id).name}» trước khi thay.")
        self.log(f"Slot {idx + 1}: {'bung Cold State' if cold.exists() else 'tạo profile trống'} cho «{p.name}»...")

        def job():
            evicted_size = coldstate.pack(slot_dir, evict_cold) if evict_cold else None
            if cold.exists():
                coldstate.unpack(cold, slot_dir)
            else:
                self.slots.clear(idx)
            self.slots.set_marker(idx, pid, dirty=False)
            return evicted_size

        def done(evicted_size, err):
            self.busy.discard(idx)
            if err:
                self.refresh()
                self._report_error(f"Không chuẩn bị được slot {idx + 1} cho «{p.name}»", err)
                return
            if evict_id:
                self._mark_saved(evict_id, evicted_size)
            self._launch_in_slot(idx, pid, url)

        self._run(job, done)

    def _launch_in_slot(self, idx: int, pid: str, url: str | None) -> None:
        p = self.store.get(pid)
        try:
            proc = chrome.launch(self.settings.chrome_path, self.slots.path(idx), self.settings.extra_args, url)
        except OSError as e:
            self._report_error("Không mở được Chrome", e)
            return
        self.procs[idx] = proc
        self.slots.update_marker(idx, dirty=True, last_used=time.time())
        self.log(f"Đã mở «{p.name if p else pid}» ở slot {idx + 1}.")
        self.refresh()

    # ------------------------------------------------------- Cold State
    def cold_state_selected(self) -> None:
        ids = self.selected_ids()
        if not ids:
            QMessageBox.information(self, "Cold State", "Anh chọn profile cần lưu trước.")
            return
        for pid in ids:
            self._cold_state_one(pid)

    def _cold_state_one(self, pid: str) -> None:
        p = self.store.get(pid)
        if p is None:
            return
        idx = self.slots.find(pid)
        if idx is None:
            self.log(f"«{p.name}» không nằm trong slot nào, Cold State đã là bản mới nhất.")
            return
        if idx in self.busy:
            self.log(f"«{p.name}» đang được xử lý.")
            return
        if idx in self.running_slots():
            if idx in self.pending_save:
                self.log(f"«{p.name}» đã được hẹn lưu khi Chrome đóng.")
                return
            ans = QMessageBox.question(
                self, "Cold State",
                f"Chrome của «{p.name}» đang mở. Phải đóng Chrome thì mới lưu an toàn được.\n\n"
                f"Yes: đóng Chrome ngay rồi lưu.\n"
                f"No: để anh tự đóng, app sẽ tự lưu ngay khi Chrome tắt.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel)
            if ans == QMessageBox.StandardButton.Cancel:
                return
            self.pending_save.add(idx)
            if ans == QMessageBox.StandardButton.Yes:
                proc = self.procs.get(idx)
                if proc is not None:
                    chrome.request_close(proc)
                    self.log(f"Đang đóng Chrome của «{p.name}», sẽ lưu Cold State ngay sau đó.")
                else:
                    self.log(f"Chrome của «{p.name}» không do app mở, anh đóng giúp em, "
                             f"app sẽ tự lưu.")
            else:
                self.log(f"Sẽ lưu Cold State «{p.name}» khi Chrome đóng.")
            self.refresh()
            return
        self.save_slot(idx)

    def save_slot(self, idx: int) -> None:
        st = self.slots.state(idx)
        p = self.store.get(st.profile_id)
        if p is None or idx in self.busy:
            return
        slot_dir, dest = self.slots.path(idx), self.cold_path(p.id)
        self.busy.add(idx)
        self.refresh()
        self.log(f"Đang lưu Cold State «{p.name}» (slot {idx + 1})...")

        def job():
            size = coldstate.pack(slot_dir, dest)
            self.slots.update_marker(idx, dirty=False)
            return size

        def done(size, err):
            self.busy.discard(idx)
            if err:
                self.refresh()
                self._report_error(f"Không lưu được Cold State «{p.name}»", err)
                return
            self._mark_saved(p.id, size)
            self.log(f"Đã lưu Cold State «{p.name}» ({human_size(size)}).")
            self.refresh()

        self._run(job, done)

    def _save_if_idle(self, idx: int) -> None:
        if idx not in self.busy and idx not in self.running_slots():
            self.save_slot(idx)

    def _sync_dirty_slots(self) -> None:
        """Lúc khởi động: slot còn dữ liệu chưa lưu từ lần trước -> lưu Cold State."""
        for s in self.slots.states():
            if s.profile_id and s.dirty and self.store.get(s.profile_id) and not self.slots.is_locked(s.index):
                self.log(f"Slot {s.index + 1} còn dữ liệu chưa lưu từ lần trước.")
                self.save_slot(s.index)

    def _poll(self) -> None:
        changed = False
        for idx, proc in list(self.procs.items()):
            if proc.poll() is None:
                continue
            del self.procs[idx]
            changed = True
            p = self.store.get(self.slots.state(idx).profile_id)
            self.log(f"Chrome ở slot {idx + 1} ({p.name if p else '?'}) đã đóng.")
            if self.settings.auto_save_on_close or idx in self.pending_save:
                self.pending_save.discard(idx)
                # Đợi chút cho các tiến trình con của Chrome nhả file.
                QTimer.singleShot(1500, lambda i=idx: self._save_if_idle(i))
        for idx in list(self.pending_save):
            if idx not in self.procs and not self.slots.is_locked(idx):
                self.pending_save.discard(idx)
                changed = True
                QTimer.singleShot(1500, lambda i=idx: self._save_if_idle(i))
        if changed or self.running_slots() != self._last_running:
            self.refresh()

    # -------------------------------------------------------- Khác
    def rename_selected(self) -> None:
        ids = self.selected_ids()
        if len(ids) != 1:
            QMessageBox.information(self, "Đổi tên", "Anh chọn đúng 1 profile.")
            return
        p = self.store.get(ids[0])
        name, ok = QInputDialog.getText(self, "Đổi tên", "Tên mới:", text=p.name)
        if ok and name.strip():
            self.store.update(p.id, name=name.strip())
            self.refresh()

    def delete_selected(self) -> None:
        ids = self.selected_ids()
        if not ids:
            return
        names = ", ".join(f"«{self.store.get(i).name}»" for i in ids[:5]) + ("..." if len(ids) > 5 else "")
        if QMessageBox.question(
                self, "Xoá profile",
                f"Xoá {len(ids)} profile: {names}?\nCold State và dữ liệu đăng nhập sẽ bị xoá vĩnh viễn.",
        ) != QMessageBox.StandardButton.Yes:
            return
        running = self.running_slots()
        for pid in ids:
            p = self.store.get(pid)
            idx = self.slots.find(pid)
            if idx is not None and (idx in running or idx in self.busy):
                self.log(f"Bỏ qua «{p.name}»: Chrome đang mở hoặc đang xử lý.")
                continue
            try:
                if idx is not None:
                    self.slots.clear(idx)
                self.cold_path(pid).unlink(missing_ok=True)
            except OSError as e:
                self._report_error(f"Không xoá được dữ liệu «{p.name}»", e)
                continue
            self.store.remove(pid)
            self.log(f"Đã xoá «{p.name}».")
        self.refresh()

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        old_slots = self.settings.max_slots
        dlg.apply_to(self.settings)
        self.settings.save(self.settings_path)
        if self.settings.max_slots != old_slots:
            QMessageBox.information(self, "Cài đặt", "Số slot mới sẽ áp dụng khi mở lại ứng dụng.")
        self.log("Đã lưu cài đặt.")

    def closeEvent(self, event) -> None:
        if self.busy:
            QMessageBox.information(self, "Đợi chút", "Đang lưu/bung Cold State, anh đợi xong rồi thoát.")
            event.ignore()
            return
        running = self.running_slots()
        if running and QMessageBox.question(
                self, "Thoát",
                f"Còn {len(running)} Chrome đang mở. Thoát app bây giờ thì Cold State của các profile này "
                f"sẽ được lưu vào lần mở app sau (dữ liệu vẫn nằm trong slot).\n\nThoát?",
        ) != QMessageBox.StandardButton.Yes:
            event.ignore()
            return
        self.timer.stop()
        self.executor.shutdown(wait=True)
        event.accept()
