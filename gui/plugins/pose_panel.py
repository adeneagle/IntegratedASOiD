"""Pose tab — attach pose files, pick bodyparts/animals, extract features.

Qt port of the "pose" tab in LabelingApp/plugins/pipeline.js (~lines 210-767:
pose-file attach/inspect, bodypart/animal selection, feature extraction +
SSE progress, and the iteration-advance bookkeeping).

`ctx` duck-type (provided by MainWindow):
    client() -> BackendClient | None      project_manager() -> PoseProjectManager | None
    video_path() -> str | None            current_video_id -> str | None (attr, not method)
    fps() -> float                        behaviors() -> list[Behavior]
    bouts() -> list[Bout]                 status(msg: str) -> None
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from labeling_shell.backend_worker import CallWorker, StreamWorker
from labeling_shell.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from labeling_shell.theme import mark_accent

from .pipeline_config import build_extract_request

ORIGINS = ["deeplabcut", "sleap", "openmonkeystudio"]


def hash_string(s: str) -> str:
    """FNV-1a, base36 — byte-identical to pipeline.js's hashString()."""
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return _to_base36(h)


def _to_base36(n: int) -> str:
    if n == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(digits[r])
    return "".join(reversed(out))


class _Base(QtWidgets.QWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self._workers: set = set()

    def _client(self):
        c = self.ctx.client()
        if c is None:
            self.ctx.status("Connect a backend first (Backend menu).")
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


class PoseTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.origin = NoWheelComboBox()
        self.origin.addItems(ORIGINS)
        self.multi_animal = QtWidgets.QCheckBox("Multi-animal")
        form.addRow("Pose software", self.origin)
        form.addRow("", self.multi_animal)

        self.pose_path_label = QtWidgets.QLabel("(none)")
        self.pose_path_label.setWordWrap(True)
        attach_btn = QtWidgets.QPushButton("Attach pose file…")
        attach_btn.clicked.connect(self._attach_pose)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.pose_path_label, 1)
        prow.addWidget(attach_btn)
        form.addRow("Pose file", prow)

        self.animals_list = QtWidgets.QListWidget()
        self.animals_list.setMaximumHeight(80)
        form.addRow("Individuals", self.animals_list)

        self.bodyparts_list = QtWidgets.QListWidget()
        self.bodyparts_list.setMaximumHeight(120)
        form.addRow("Body parts", self.bodyparts_list)

        self.framerate = NoWheelDoubleSpinBox()
        self.framerate.setRange(1.0, 1000.0)
        self.framerate.setValue(30.0)
        self.llh = NoWheelDoubleSpinBox()
        self.llh.setRange(0.0, 1.0)
        self.llh.setSingleStep(0.05)
        self.llh.setValue(0.6)
        self.min_duration = NoWheelDoubleSpinBox()
        self.min_duration.setRange(0.01, 10.0)
        self.min_duration.setSingleStep(0.05)
        self.min_duration.setValue(0.1)
        form.addRow("Frame rate", self.framerate)
        form.addRow("Likelihood threshold", self.llh)
        form.addRow("Min behavior duration (s)", self.min_duration)

        layout.addLayout(form)

        iter_row = QtWidgets.QHBoxLayout()
        self.iteration_spin = NoWheelSpinBox()
        self.iteration_spin.setRange(0, 9999)
        new_iter_btn = QtWidgets.QPushButton("New Iteration")
        new_iter_btn.clicked.connect(self._new_iteration)
        iter_row.addWidget(QtWidgets.QLabel("Iteration"))
        iter_row.addWidget(self.iteration_spin)
        iter_row.addWidget(new_iter_btn)
        layout.addLayout(iter_row)

        self.videos_list = QtWidgets.QListWidget()
        self.videos_list.setMaximumHeight(100)
        layout.addWidget(QtWidgets.QLabel("Extract for videos:"))
        layout.addWidget(self.videos_list)

        btns = QtWidgets.QHBoxLayout()
        self.extract_btn = QtWidgets.QPushButton("Extract features")
        mark_accent(self.extract_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._extract)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.extract_btn)
        btns.addWidget(self.stop_btn)
        layout.addLayout(btns)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self._task_id = None
        self._n_frames = None

    # ---------------------------------------------------------------- utils
    def _checked_texts(self, listw: QtWidgets.QListWidget) -> list[str]:
        out = []
        for i in range(listw.count()):
            item = listw.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.text())
        return out

    def _fill_checklist(self, listw: QtWidgets.QListWidget, values: list[str]) -> None:
        listw.clear()
        for v in values:
            item = QtWidgets.QListWidgetItem(v)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            listw.addItem(item)

    def refresh(self) -> None:
        self.videos_list.clear()
        pm = self.ctx.project_manager()
        if pm is None:
            return
        for v in pm.manifest.get("videos", []):
            item = QtWidgets.QListWidgetItem(v.get("name", v.get("id")))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, v)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            self.videos_list.addItem(item)
        pose = self.ctx.pose_path()
        self.pose_path_label.setText(pose or "(none)")
        self.bodyparts_list.clear()
        self.animals_list.clear()
        self._n_frames = None
        if pose:
            self._inspect(pose)

    # -------------------------------------------------------------- attach
    def _attach_pose(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Attach pose file", "",
            "Pose files (*.h5 *.csv *.json);;All files (*)")
        if not path:
            return
        pm = self.ctx.project_manager()
        video_id = getattr(self.ctx, "current_video_id", None)
        if pm is not None and video_id:
            pm.set_video_pose(video_id, path)
        self.pose_path_label.setText(path)
        self._inspect(path)

    def _inspect(self, pose_path: str) -> None:
        client = self._client()
        if not client:
            return
        self._spawn_call(
            lambda: client.inspect_pose(pose_path, self.origin.currentText(),
                                        self.multi_animal.isChecked()),
            self._on_inspected,
        )

    def _on_inspected(self, res: dict) -> None:
        self._fill_checklist(self.bodyparts_list, res.get("bodyparts", []))
        self._fill_checklist(self.animals_list, res.get("animals", []))
        self._n_frames = res.get("n_frames")
        self.ctx.status(f"Pose inspected: {len(res.get('bodyparts', []))} bodyparts, "
                        f"{self._n_frames} frames.")

    # ----------------------------------------------------------- iterations
    def current_iteration_dir(self) -> str:
        n = self.iteration_spin.value()
        return f"iteration-{n}"

    def _new_iteration(self) -> None:
        self.iteration_spin.setValue(self.iteration_spin.value() + 1)
        pm = self.ctx.project_manager()
        if pm is not None:
            pm.set_current_iteration(self.iteration_spin.value())

    def label_signature(self, video_entries: list[dict]) -> str:
        parts = []
        for v in sorted(video_entries, key=lambda e: e.get("id", "")):
            bouts = self.ctx.bouts() if v.get("id") == getattr(
                self.ctx, "current_video_id", None) else []
            n = len(bouts)
            frames = sum(b.end_frame - b.start_frame + 1 for b in bouts)
            spans = ",".join(sorted(f"{b.start_frame}-{b.end_frame}:{b.behavior_name}"
                                    for b in bouts))
            parts.append(f"{v.get('id')}|{n}|{frames}|{hash_string(spans)}")
        return ";".join(parts)

    # -------------------------------------------------------------- extract
    def _extract(self) -> None:
        client = self._client()
        pm = self.ctx.project_manager()
        if client is None:
            QtWidgets.QMessageBox.warning(
                self, "Can't extract features",
                "No backend connection. Connect a backend first (Backend menu).")
            return
        if pm is None:
            msg = ("No project is open. Features are written into the project "
                   "folder, so create or open a project first (File menu).")
            self.ctx.status(msg)
            QtWidgets.QMessageBox.warning(self, "Can't extract features", msg)
            return
        bodyparts = self._checked_texts(self.bodyparts_list)
        if len(bodyparts) < 2:
            msg = ("Select at least two body parts." if self.bodyparts_list.count()
                   else "No body parts loaded. Attach a pose file first.")
            self.ctx.status(msg)
            QtWidgets.QMessageBox.warning(self, "Can't extract features", msg)
            return
        selected_ids = set()
        videos = []
        for i in range(self.videos_list.count()):
            item = self.videos_list.item(i)
            if item.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            entry = item.data(QtCore.Qt.ItemDataRole.UserRole)
            vid = entry.get("id")
            selected_ids.add(vid)
            videos.append({
                "id": vid,
                "pose_path": pm.get_video_pose(vid),
                "label_path": pm.get_video_labels_abs(vid),
                "max_frames": pm.get_video_frame_count(vid),
            })
        if not videos:
            self.ctx.status("No videos selected.")
            QtWidgets.QMessageBox.warning(self, "Can't extract features", "No videos selected.")
            return

        iteration_dir = self.current_iteration_dir()
        animals = self._checked_texts(self.animals_list) or None
        req = build_extract_request(
            project_dir=str(pm.root), videos=videos,
            origin=self.origin.currentText(), bodyparts=bodyparts,
            framerate=self.framerate.value(), min_duration=self.min_duration.value(),
            llh_value=self.llh.value(), animals=animals,
            multi_animal=self.multi_animal.isChecked(), iteration_dir=iteration_dir,
        )

        self.log.appendPlainText(f"[extract] starting on {len(videos)} video(s)…")
        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        task_error = {"msg": None}

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)
            status = ev.get("status")
            if status in ("completed", "error", "cancelled"):
                self.log.appendPlainText(f"[extract] {status}")
            if status == "error":
                task_error["msg"] = ev.get("error") or "Extraction failed."

        def on_done():
            self.extract_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            if task_error["msg"]:
                self.ctx.status("Extraction failed.")
                QtWidgets.QMessageBox.critical(
                    self, "Feature extraction failed", task_error["msg"])
                return
            pm.set_iteration_meta(iteration_dir, {
                "label_signature": self.label_signature(
                    [e for e in pm.manifest.get("videos", []) if e.get("id") in selected_ids]),
                "videos": sorted(selected_ids),
            })
            self.ctx.status("Extraction finished.")

        def on_fail(msg: str):
            self.extract_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log.appendPlainText(f"[extract] failed: {msg}")
            QtWidgets.QMessageBox.critical(self, "Feature extraction failed", msg)

        def start(res: dict):
            self._task_id = res.get("task_id")
            if not self._task_id:
                on_fail(f"Backend did not return a task id: {res}")
                return
            self._spawn_stream(
                lambda: client.extraction_status(self._task_id), on_event, on_done, on_fail)

        self._spawn_call(lambda: client.extract_features(req), start, on_fail)

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_extraction(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))
