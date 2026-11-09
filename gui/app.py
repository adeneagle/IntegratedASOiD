"""IntegratedASOiD desktop app (PySide6) — main window.

The all-Python successor to the Electron shell. Left: frame-accurate video
player. Right: a tabbed panel — "Label" (behaviors + bouts) plus the pipeline
tabs (Pose / Train / Predict / Active Learning), backed by the FastAPI
backend (pose feature extraction + RandomForest active learning).

Keyboard: 1-0 toggle behavior bouts at the current frame, Space play/pause,
Left/Right step, F fullscreen — same shortcuts as the Electron app.

    pixi run gui            (or: pixi run -e gui python gui/app.py [video])
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6 import QtCore, QtWidgets  # noqa: E402

from labeling_shell.backend import Backend, BackendConfig  # noqa: E402
from labeling_shell.backend_worker import CallWorker  # noqa: E402
from labeling_shell.behaviors_panel import BehaviorsPanel  # noqa: E402
from labeling_shell.bouts import KEY_ORDER, Behavior, LabelModel  # noqa: E402
from labeling_shell.bouts_panel import BoutsPanel  # noqa: E402
from labeling_shell.csv_io import (  # noqa: E402
    generate_bouts_csv, generate_training_label_csv, parse_csv_and_import,
)
from labeling_shell.active_learning import hud_info  # noqa: E402
from labeling_shell.theme import apply_theme  # noqa: E402
from labeling_shell.timeline_widget import UncertaintyTimeline  # noqa: E402
from labeling_shell.video_widget import PlayerWidget  # noqa: E402
from plugins.pipeline_panel import PipelinePanel  # noqa: E402
from plugins.pose_overlay import PoseOverlay  # noqa: E402
from plugins.project_manager import PoseProjectManager, sanitize_id  # noqa: E402

EDITABLE = (QtWidgets.QLineEdit, QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox,
            QtWidgets.QPlainTextEdit)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, autostart_backend: bool = True):
        super().__init__()
        self.setWindowTitle("IntegratedASOiD — Labeler")
        self.model = LabelModel()
        self.project: PoseProjectManager | None = None
        self.current_video_id: str | None = None
        self._video_path: str | None = None
        self.backend: Backend | None = None
        self._bg_workers: set = set()
        self._al_state = None
        self.al_threshold = 0.5

        self.player = PlayerWidget()
        self.player.frameChanged.connect(self._on_frame)
        self.player.snapBackRequested.connect(self.snap_back)
        self.behaviors_panel = BehaviorsPanel(self.model)
        self.bouts_panel = BoutsPanel(self.model, self._fps)
        self.behaviors_panel.changed.connect(self._on_model_changed)
        self.bouts_panel.changed.connect(self._on_model_changed)
        self.bouts_panel.seekRequested.connect(self._seek)
        self.pipeline = PipelinePanel(self)  # MainWindow is the pipeline ctx
        self.pipeline.pose.selectionChanged.connect(self._on_pose_selection_changed)

        label_tab = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(label_tab)
        ll.addWidget(self.behaviors_panel, 1)
        ll.addWidget(self.bouts_panel, 2)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(label_tab, "Label")
        self.tabs.addTab(self.pipeline, "Pipeline")

        self.timeline = UncertaintyTimeline()
        self.timeline.seekRequested.connect(self._seek)
        self.overlay_chk = QtWidgets.QCheckBox("Pose overlay")
        self.overlay_chk.toggled.connect(self._on_overlay_toggled)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(self.player, 1)
        lv.addWidget(self.overlay_chk)
        lv.addWidget(self.timeline)

        # On-video HUD (prediction / confidence / label), overlaid on the frame.
        self.hud = QtWidgets.QLabel(self.player.view)
        self.hud.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud.hide()

        # Pose-skeleton overlay, also a child of the video view.
        self.pose_overlay = PoseOverlay(self, self.player.view)
        self.pose_overlay.setGeometry(self.player.view.rect())
        self.player.view.installEventFilter(self)

        self.split = QtWidgets.QSplitter()
        self.split.addWidget(left)
        self.split.addWidget(self.tabs)
        self.split.setStretchFactor(0, 3)
        self.split.setStretchFactor(1, 2)
        self.setCentralWidget(self.split)

        self._settings = QtCore.QSettings("IntegratedASOiD", "Labeler")
        self.split.splitterMoved.connect(self._save_splitter_sizes)

        self._build_menu()
        self.resize(1320, 820)
        self._restore_splitter_sizes()
        QtWidgets.QApplication.instance().installEventFilter(self)

        # The pipeline tabs are useless without a backend and every session
        # began by opening the Backend menu, so just start one. Deferred to the
        # event loop so the window paints first rather than appearing to hang
        # while uvicorn boots; the spawn itself is on a worker thread.
        # Tests pass False — they pump events, and spawning uvicorn is not
        # something a test should do.
        if autostart_backend:
            QtCore.QTimer.singleShot(0, self.connect_local)

    def _save_splitter_sizes(self, *_args) -> None:
        self._settings.setValue("splitterState", self.split.saveState())

    def _restore_splitter_sizes(self) -> None:
        state = self._settings.value("splitterState")
        if state:
            self.split.restoreState(state)

    # ------------------------------------------------------------------ menu
    def _build_menu(self) -> None:
        m = self.menuBar()
        fm = m.addMenu("&File")
        fm.addAction("Open Video…", self.open_video)
        fm.addSeparator()
        fm.addAction("Import Labels (CSV)…", self.import_csv)
        fm.addAction("Export Bout CSV…", self.export_bout_csv)
        fm.addAction("Export Training CSV…", self.export_training_csv)
        fm.addSeparator()
        fm.addAction("New Project…", self.new_project)
        fm.addAction("Open Project…", self.open_project)
        self.videos_menu = fm.addMenu("Project Videos")
        self.videos_menu.aboutToShow.connect(self._populate_videos_menu)
        fm.addSeparator()
        fm.addAction("Export .rlt…", self.export_rlt)
        fm.addAction("Import .rlt…", self.import_rlt)

        bm = m.addMenu("&Backend")
        bm.addAction("Connect Local…", self.connect_local)
        bm.addAction("Connect Remote…", self.connect_remote)
        bm.addAction("Disconnect", self.disconnect_backend)

    # --------------------------------------------------------------- helpers
    def _fps(self) -> float:
        return self.player.reader.fps if self.player.reader else 30.0

    def _rebind_model(self) -> None:
        self.behaviors_panel.model = self.model
        self.bouts_panel.model = self.model

    def _refresh_all(self) -> None:
        self.behaviors_panel.refresh()
        self.bouts_panel.refresh()
        self.pipeline.refresh()

    def _seek(self, frame: int) -> None:
        self.player._show(frame)

    def snap_back(self) -> None:
        """Jump to where labeling left off: the end of the latest bout.

        Ported from the Electron trackbar's Snap Back button — "latest" is by
        end frame, not list order, since bouts are sorted by start and a long
        early bout can outlast a later short one. No bouts means frame 0.
        """
        if self.player.reader is None:
            self._status("Load a video first.")
            return
        target = max((b.end_frame for b in self.model.bouts), default=0)
        self.player.pause()
        self._seek(target)
        self._status(f"Snapped to frame {target}.")

    def _on_frame(self, idx: int) -> None:
        self.timeline.set_current(idx)
        self._update_hud(idx)
        self.pose_overlay.set_current_frame(idx)

    def _on_overlay_toggled(self, checked: bool) -> None:
        if checked and not self._load_overlay():
            self.overlay_chk.blockSignals(True)
            self.overlay_chk.setChecked(False)
            self.overlay_chk.blockSignals(False)
            checked = False
        self.pose_overlay.set_enabled(checked)

    def _load_overlay(self, quiet: bool = False) -> bool:
        pose = self.pipeline.pose
        return self.pose_overlay.ensure_loaded(
            pose.selected_bodyparts(), pose.pose_settings(), quiet=quiet)

    def _on_pose_selection_changed(self) -> None:
        self.pose_overlay.invalidate()
        if self.overlay_chk.isChecked():
            self._load_overlay(quiet=True)

    def _on_model_changed(self) -> None:
        self._refresh_all()
        self._recompute_misclassified()
        self._update_hud(self.player.index)
        self._autosave()

    def _autosave(self) -> None:
        if self.project and self.current_video_id:
            self.project.save_video_labels(self.current_video_id, self.model)

    def _update_title(self) -> None:
        title = "IntegratedASOiD — Labeler"
        if self.project:
            title += f" — {self.project.manifest.get('name') or self.project.root.name}"
        if self._video_path:
            title += f" — {Path(self._video_path).name}"
        self.setWindowTitle(title)

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg, 5000)

    # ---- pipeline context (duck-typed for PipelinePanel) ------------------
    def client(self):
        return self.backend.client if self.backend else None

    def video_path(self):
        return self._video_path

    def fps(self):
        return self._fps()

    def bouts(self):
        return self.model.bouts

    def behaviors(self):
        return self.model.behaviors

    def project_manager(self):
        return self.project

    def pose_path(self):
        return self.pipeline.pose.pose_path_for(self.current_video_id)

    def video_frame_count(self):
        return self.player.reader.num_frames if self.player.reader else None

    def save_labels(self):
        self._autosave()

    def checkpoint_dir(self):
        return str(self.project.checkpoint_dir()) if self.project else None

    def status(self, msg):
        self._status(msg)

    # ---- active-learning ctx ----------------------------------------------
    def al_state(self):
        return self._al_state

    def current_frame(self):
        return self.player.index

    def seek(self, frame):
        self._seek(frame)

    def set_al_threshold(self, t):
        self.al_threshold = t
        self.timeline.set_threshold(t)

    def set_al_results(self, state):
        """Show a video's predictions — or, with None, clear the last video's."""
        self._al_state = state
        if self.player.reader is not None:
            n = self.player.reader.num_frames
        elif state is not None and state.by_index:
            n = max(state.by_index) + 1
        else:
            n = 0
        self.timeline.set_data(state, n)
        if state is None:
            self.timeline.set_misclassified(())
        self._recompute_misclassified()
        self._update_hud(self.player.index)

    def _recompute_misclassified(self) -> None:
        if self._al_state is None:
            return

        def label_at(i):
            b = self.model.bout_at_frame(i)
            return b.behavior_name if b else None

        self.timeline.set_misclassified(self._al_state.misclassified_frames(label_at))

    def _update_hud(self, frame: int) -> None:
        b = self.model.bout_at_frame(frame)
        human = b.behavior_name if b else None
        info = hud_info(self._al_state, frame, human)
        if info["predicted"] is None and not info["label"]:
            self.hud.hide()
            return
        text = f"label: {info['label']}" if info["label"] else "label: —"
        if info["predicted"] is not None:
            text += f"\npred: {info['predicted']}"
            if info["confidence"] is not None:
                text += f"  ({info['confidence']:.2f})"
        color = "#ff5555" if info["mismatch"] else "#ffffff"
        self.hud.setText(text)
        self.hud.setStyleSheet(
            f"background: rgba(0,0,0,160); color: {color}; padding:4px; border-radius:4px;")
        self.hud.adjustSize()
        self.hud.move(10, 10)
        self.hud.show()

    # ------------------------------------------------------------------ video
    def open_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Videos (*.mp4 *.mkv *.avi *.webm *.mov);;All files (*)")
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        self._autosave()
        self.pose_overlay.invalidate()
        self.player.load(path)
        self._video_path = path
        name = Path(path).name
        self.current_video_id = sanitize_id(name)
        if self.project:
            self.model = self.project.load_into_model(self.current_video_id)
            self._rebind_model()
            self.project.manifest["last_video"] = self.current_video_id
            self.project.register_video(
                self.current_video_id, name, path, self.player.reader.num_frames)
        self._refresh_all()
        # Repoints the pose file and restores (or clears) this video's saved
        # predictions, so one video's uncertainty never draws over another's.
        self.pipeline.on_video_switched()
        if self.overlay_chk.isChecked():
            self._load_overlay(quiet=True)
        self._update_title()
        self._status(f"Loaded {name} — {self.player.reader.num_frames} frames")

    # --------------------------------------------------------------- keyboard
    def eventFilter(self, obj, event):
        if obj is self.player.view and event.type() == QtCore.QEvent.Type.Resize:
            self.pose_overlay.setGeometry(self.player.view.rect())
        if event.type() == QtCore.QEvent.Type.KeyPress:
            focus = QtWidgets.QApplication.focusWidget()
            if not isinstance(focus, EDITABLE):
                if self._handle_key(event.text(), event.key()):
                    return True
        return super().eventFilter(obj, event)

    def _handle_key(self, text: str, key: int) -> bool:
        if text in KEY_ORDER:
            self._toggle_behavior(text)
            return True
        if key == QtCore.Qt.Key.Key_Space:
            self.player.toggle_play()
            return True
        if key == QtCore.Qt.Key.Key_Left:
            self.player.step(-1)
            return True
        if key == QtCore.Qt.Key.Key_Right:
            self.player.step(+1)
            return True
        if key == QtCore.Qt.Key.Key_F:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return True
        return False

    def _toggle_behavior(self, key: str) -> None:
        if self.player.reader is None:
            self._status("Load a video first.")
            return
        res = self.model.toggle(key, self.player.index)
        if res.action == "no_behavior":
            return
        if res.action == "started":
            self._status(f"Started {res.behavior.name} @ frame {self.player.index}")
        elif res.action == "ended":
            self._status(f"Ended {res.behavior.name} "
                         f"[{res.bout.start_frame}–{res.bout.end_frame}]")
        elif res.action == "zero_length":
            self._status("Zero-length bout skipped.")
        self._refresh_all()
        self._autosave()

    # ------------------------------------------------------------------- CSV
    def import_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import labels", "", "CSV (*.csv);;All files (*)")
        if not path:
            return
        n = parse_csv_and_import(Path(path).read_text(encoding="utf-8"), self.model, self._fps())
        self._refresh_all()
        self._autosave()
        self._status(f"Imported {n} bouts.")

    def export_bout_csv(self) -> None:
        if not self.model.bouts:
            self._status("No bouts to export.")
            return
        if self.model.find_overlapping_bouts() is not None:
            QtWidgets.QMessageBox.warning(self, "Overlap", "Cannot export: some bouts overlap.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export bout CSV", "bouts.csv", "CSV (*.csv)")
        if path:
            Path(path).write_text(generate_bouts_csv(self.model.bouts, self._fps()),
                                  encoding="utf-8")
            self._status(f"Saved {path}")

    def export_training_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export training CSV", "labels.csv", "CSV (*.csv)")
        if path:
            Path(path).write_text(generate_training_label_csv(self.model.bouts),
                                  encoding="utf-8")
            self._status(f"Saved {path}")

    # --------------------------------------------------------------- project
    def new_project(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose project folder")
        if not d:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "Project name", "Name:")
        if not ok:
            return
        store = PoseProjectManager(d)
        if store.exists():
            QtWidgets.QMessageBox.warning(
                self, "Already a project",
                "That folder already holds a project — use File ▸ Open Project.")
            return
        store.create(name or Path(d).name)
        self.project = store
        # Labels made before the project existed belong in it.
        if self._video_path and self.current_video_id:
            store.register_video(self.current_video_id, Path(self._video_path).name,
                                 self._video_path, self.video_frame_count())
            pose = self.pipeline.pose._pose_by_video.get(self.current_video_id)
            if pose:
                store.set_video_pose(self.current_video_id, pose)
                store.flush_manifest()
            self._autosave()
        self._project_opened()
        self._status(f"Project created at {d}")

    def open_project(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open project folder")
        if not d:
            return
        store = PoseProjectManager(d)
        if not store.exists():
            QtWidgets.QMessageBox.warning(self, "Not a project", "No project.json found.")
            return
        self._autosave()
        self._set_project(store)
        # Pick up where the user left off: the video open when they last worked
        # in this project, else the first one still on disk.
        videos = store.manifest.get("videos") or []
        last = store.manifest.get("last_video")
        candidates = sorted(videos, key=lambda v: v.get("id") != last)
        for v in candidates:
            if v.get("path") and os.path.exists(v["path"]):
                self.load_video(v["path"])
                break
        self._status(f"Opened project {store.manifest.get('name')} — {len(videos)} video(s); "
                     f"switch with File ▸ Project Videos.")

    def _set_project(self, store: PoseProjectManager) -> None:
        """Make `store` the active project with no video loaded from it yet."""
        self.project = store
        self.model = LabelModel()
        self.model.behaviors = [Behavior(**b) for b in store.manifest.get("behaviors", [])]
        self.current_video_id = None
        self._video_path = None
        self._rebind_model()
        self.set_al_results(None)
        self._project_opened()

    def _project_opened(self) -> None:
        self._refresh_all()
        self.pipeline.on_project_opened()
        self._update_title()

    def _populate_videos_menu(self) -> None:
        self.videos_menu.clear()
        videos = (self.project.manifest.get("videos") or []) if self.project else []
        if not videos:
            act = self.videos_menu.addAction("(open a project with videos)")
            act.setEnabled(False)
            return
        for v in videos:
            path = v.get("path")
            label = v.get("name") or v.get("id")
            if v.get("id") == self.current_video_id:
                label = "● " + label
            act = self.videos_menu.addAction(label)
            if path and os.path.exists(path):
                act.triggered.connect(lambda _c=False, p=path: self.load_video(p))
            else:
                act.setText(label + "  (file not found)")
                act.setEnabled(False)

    # ------------------------------------------------------------------- rlt
    def export_rlt(self) -> None:
        if self.project is None:
            self._status("No active project to export.")
            return
        client = self.client()
        if client is None:
            self._status("Connect a backend first (Backend menu).")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Export .rlt")
        box.setText("What should the bundle contain?")
        box.setInformativeText(
            "Labels only: the manifest and labels — small, for sharing annotations.\n"
            "Full package: also pose files, features, predictions and models, so the "
            "project trains and predicts on another machine as-is.")
        light = box.addButton("Labels only", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        full = box.addButton("Full package", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() not in (light, full):
            return
        package = box.clickedButton() is full
        default_name = (self.project.manifest.get("name") or "project") + ".rlt"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export .rlt", default_name, "RLT bundle (*.rlt)")
        if not path:
            return
        self._autosave()
        self._status("Exporting .rlt…")
        self._spawn_bg(
            lambda: self.project.export_rlt(client, path, package=package),
            lambda res: self._status(f"Exported {path}"),
        )

    def import_rlt(self) -> None:
        client = self.client()
        if client is None:
            self._status("Connect a backend first (Backend menu).")
            return
        rlt_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import .rlt", "", "RLT bundle (*.rlt)")
        if not rlt_path:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Unpack into folder")
        if not out_dir:
            return
        self._status("Importing .rlt…")

        def do_import():
            return PoseProjectManager.import_rlt(client, rlt_path, out_dir)

        def on_done(pm: PoseProjectManager) -> None:
            self._set_project(pm)
            self._status(f"Imported project into {out_dir} — videos need re-linking: "
                         f"open each with File ▸ Open Video.")

        self._spawn_bg(do_import, on_done)

    # --------------------------------------------------------------- backend
    def _spawn_bg(self, fn, on_done):
        w = CallWorker(fn)
        w.done.connect(on_done)
        w.failed.connect(self._status)
        w.finished.connect(lambda: self._bg_workers.discard(w))
        self._bg_workers.add(w)
        w.start()

    def connect_local(self) -> None:
        self.disconnect_backend()
        self.backend = Backend(BackendConfig())
        self._status("Starting local backend…")

        def done(ok: bool) -> None:
            if ok:
                self._status("Backend ready.")
                self.pipeline.on_backend_ready()
            else:
                # Don't leave a dead Backend attached: the pipeline tabs check
                # ctx.client() for None to decide whether to prompt.
                self.disconnect_backend()
                self._status(
                    "Backend failed to start — run `pixi install`, or use "
                    "Backend ▸ Connect Local to retry.")

        self._spawn_bg(lambda: self.backend.start(wait=True, timeout=40), done)

    def connect_remote(self) -> None:
        host, ok = QtWidgets.QInputDialog.getText(
            self, "Connect remote backend", "Host (e.g. 192.168.1.20):")
        if not ok or not host.strip():
            return
        self.disconnect_backend()
        self.backend = Backend(BackendConfig.remote(host.strip()))
        self._status(f"Connecting to {host}…")
        def done(ok: bool) -> None:
            self._status("Backend reachable." if ok else "Backend not reachable.")
            if ok:
                self.pipeline.on_backend_ready()

        self._spawn_bg(lambda: self.backend.start(wait=True, timeout=20), done)

    def disconnect_backend(self) -> None:
        if self.backend is not None:
            self.backend.stop()
            self.backend = None
            self._status("Backend disconnected.")

    def closeEvent(self, event) -> None:
        self._autosave()
        if self.project:
            self.project.flush_manifest()
        self.disconnect_backend()
        super().closeEvent(event)


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    if len(sys.argv) > 1:
        win.load_video(sys.argv[1])
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
