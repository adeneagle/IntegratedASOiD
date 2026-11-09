"""Pieces every pipeline tab shares: the worker-spawning base class, checklist
helpers, the scroll-with-pinned-actions layout, and the model picker.

The layout and `_Base` follow RealtimeASOID's pipeline_panel: forms scroll,
the action buttons and log stay pinned below them, and every backend call runs
on a worker thread with failures routed to the status bar (plus an optional
per-call handler so a tab can re-enable its buttons).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from PySide6 import QtCore, QtWidgets

from labeling_shell.backend_worker import CallWorker, StreamWorker
from labeling_shell.no_wheel import NoWheelComboBox

from .pipeline_config import MODEL_FILE_FILTER, model_label

CHECKED = QtCore.Qt.CheckState.Checked
UNCHECKED = QtCore.Qt.CheckState.Unchecked
USER_ROLE = QtCore.Qt.ItemDataRole.UserRole


class _Base(QtWidgets.QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._workers: set = set()

    def _client(self):
        c = self.ctx.client()
        if c is None:
            self.ctx.status("No backend connection — it may still be starting "
                            "(Backend menu to retry).")
        return c

    def _spawn_call(self, fn, on_done, on_fail=None):
        w = CallWorker(fn)
        w.done.connect(on_done)
        w.failed.connect(self.ctx.status)
        if on_fail:
            w.failed.connect(on_fail)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()
        return w

    def _spawn_stream(self, make_iter, on_event, on_done=None, on_fail=None):
        w = StreamWorker(make_iter)
        w.event.connect(on_event)
        w.failed.connect(self.ctx.status)
        if on_fail:
            w.failed.connect(on_fail)
        if on_done:
            w.finished_ok.connect(on_done)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()
        return w

    def _warn(self, title: str, msg: str) -> None:
        """A blocking problem: say it in the status bar *and* a dialog, since a
        status message alone vanishes before anyone reads it."""
        self.ctx.status(msg)
        QtWidgets.QMessageBox.warning(self, title, msg)

    @staticmethod
    def _scroll_body(outer: QtWidgets.QVBoxLayout) -> QtWidgets.QVBoxLayout:
        """A scrolling area at the top of `outer`; returns its layout."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        return QtWidgets.QVBoxLayout(inner)

    @staticmethod
    def _wrap(layout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    @staticmethod
    def _hint(text: str = "") -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setWordWrap(True)
        label.setProperty("muted", "true")
        return label

    @staticmethod
    def _log_box(max_height: int = 170) -> QtWidgets.QPlainTextEdit:
        log = QtWidgets.QPlainTextEdit()
        log.setReadOnly(True)
        log.setMaximumHeight(max_height)  # the form is the working surface, not this
        return log


# ----------------------------------------------------------------- checklists
def checked_values(listw: QtWidgets.QListWidget, role=None) -> list:
    """Checked, enabled items — their UserRole data if `role` is given, else text."""
    out = []
    for i in range(listw.count()):
        item = listw.item(i)
        if item.checkState() != CHECKED:
            continue
        if not (item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled):
            continue
        out.append(item.data(role) if role is not None else item.text())
    return out


def fill_checklist(listw: QtWidgets.QListWidget, values: Iterable,
                   checked: Optional[set] = None, *, labels=None,
                   disabled: Optional[set] = None) -> None:
    """Rebuild a checklist. `checked=None` ticks everything.

    Signals are blocked during the rebuild so per-item handlers don't fire once
    per row.
    """
    values = list(values)
    labels = list(labels) if labels is not None else [str(v) for v in values]
    disabled = disabled or set()
    listw.blockSignals(True)
    listw.clear()
    for value, label in zip(values, labels):
        item = QtWidgets.QListWidgetItem(label)
        item.setData(USER_ROLE, value)
        flags = item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        if value in disabled:
            flags &= ~QtCore.Qt.ItemFlag.ItemIsEnabled
        item.setFlags(flags)
        on = (checked is None or value in checked) and value not in disabled
        item.setCheckState(CHECKED if on else UNCHECKED)
        listw.addItem(item)
    listw.blockSignals(False)


def checklist(max_height: int = 120) -> QtWidgets.QListWidget:
    w = QtWidgets.QListWidget()
    w.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
    w.setMaximumHeight(max_height)
    return w


def set_all_checked(listw: QtWidgets.QListWidget, on: bool) -> None:
    for i in range(listw.count()):
        item = listw.item(i)
        if item.flags() & QtCore.Qt.ItemFlag.ItemIsEnabled:
            item.setCheckState(CHECKED if on else UNCHECKED)


# --------------------------------------------------------------- model picker
class ModelPicker(QtWidgets.QWidget):
    """Choose a trained `.joblib` model: the project's, newest first, or browse.

    The list comes from GET /api/training/checkpoints over the project folder,
    which reads each bundle's metadata (iteration, F1, body parts, training
    config) — so a picker entry carries everything the AL panel needs to
    interpret the model, without the user opening the file.
    """

    changed = QtCore.Signal()

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._workers: set = set()
        self.combo = NoWheelComboBox()
        self.combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(18)
        self.combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        refresh = QtWidgets.QPushButton("Rescan")
        refresh.setToolTip("Rescan the project for trained models")
        refresh.clicked.connect(lambda: self.reload())
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.combo, 1)
        row.addWidget(refresh)
        row.addWidget(browse)
        self._placeholder()

    def _placeholder(self) -> None:
        if self.combo.count() == 0:
            self.combo.addItem("(no models yet — train one, or Browse…)", None)

    def entries(self) -> list[dict]:
        return [self.combo.itemData(i) for i in range(self.combo.count())
                if self.combo.itemData(i)]

    def set_entries(self, entries: list[dict], prefer: Optional[str] = None) -> None:
        """Replace the list; keep the current choice if it's still there,
        else `prefer`, else the newest (first)."""
        keep = prefer or self.current_path()
        self.combo.blockSignals(True)
        self.combo.clear()
        for e in entries:
            self.combo.addItem(model_label(e), e)
            self.combo.setItemData(self.combo.count() - 1, e.get("path"),
                                   QtCore.Qt.ItemDataRole.ToolTipRole)
        self._placeholder()
        self.combo.blockSignals(False)
        if not (keep and self.select_path(keep)):
            self.combo.setCurrentIndex(0)
        self.changed.emit()

    def current_entry(self) -> Optional[dict]:
        return self.combo.currentData()

    def current_path(self) -> Optional[str]:
        e = self.current_entry()
        return e.get("path") if e else None

    def select_path(self, path: str) -> bool:
        norm = os.path.normcase(os.path.normpath(path))
        for i in range(self.combo.count()):
            e = self.combo.itemData(i)
            if e and os.path.normcase(os.path.normpath(e.get("path", ""))) == norm:
                self.combo.setCurrentIndex(i)
                return True
        return False

    def add_path(self, path: str, meta: Optional[dict] = None) -> None:
        """Put a model the scan doesn't know about (browsed, or just trained) at
        the top and select it."""
        if self.select_path(path):
            return
        entry = {"path": path, "filename": os.path.basename(path), **(meta or {})}
        self.set_entries([entry, *self.entries()], prefer=path)

    def reload(self, prefer: Optional[str] = None) -> None:
        """Rescan the project's models on a worker thread."""
        pm = self.ctx.project_manager()
        client = self.ctx.client()
        if pm is None or client is None:
            return
        root = str(pm.root)

        def done(res: dict):
            found = res.get("checkpoints") or []
            # Keep anything browsed from outside the project.
            inside = {os.path.normcase(e.get("path", "")) for e in found}
            extra = [e for e in self.entries()
                     if os.path.normcase(e.get("path", "")) not in inside]
            self.set_entries(found + extra, prefer=prefer)

        w = CallWorker(lambda: client.list_checkpoints([root]))
        w.done.connect(done)
        w.failed.connect(self.ctx.status)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()

    def _browse(self) -> None:
        pm = self.ctx.project_manager()
        start = str(pm.root) if pm else ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Model", start, MODEL_FILE_FILTER)
        if path:
            self.add_path(path)
