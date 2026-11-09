"""Pose tab — attach pose files, pick body parts/animals, extract features.

Qt port of the "pose" tab in LabelingApp/plugins/pipeline.js: pose-file
attach/inspect, body-part/animal selection, feature extraction with SSE
progress and a results summary, and the iteration bookkeeping
(`autoAdvanceIteration`).

Three lifecycles, kept apart on purpose:
    refresh()            — cheap, runs on every label edit: video checklist only.
                           Never calls the backend, never touches the body-part
                           selection (it used to re-inspect the pose file and
                           re-tick every body part on each keypress).
    on_video_switched()  — repoint at the new video's pose file and re-inspect.
    on_project_opened()  — restore the settings that produced this project's
                           features (manifest "preprocessing") and its iteration.

`ctx` duck-type (provided by MainWindow):
    client() -> BackendClient | None      project_manager() -> PoseProjectManager | None
    current_video_id -> str | None (attr) video_frame_count() -> int | None
    fps() -> float                        bouts() -> list[Bout]
    save_labels() -> None                 status(msg: str) -> None
"""

from __future__ import annotations

import datetime
import os

from PySide6 import QtCore, QtWidgets

from labeling_shell.no_wheel import NoWheelComboBox, NoWheelDoubleSpinBox, NoWheelSpinBox
from labeling_shell.theme import mark_accent, mark_ghost

from .common import (
    CHECKED, USER_ROLE, _Base, checked_values, checklist, fill_checklist, set_all_checked,
)
from .pipeline_config import (
    ORIGINS, POSE_FILE_FILTER, bin_width, build_extract_request, feature_width,
    length_mismatch, summarize_extraction,
)

__all__ = ["ORIGINS", "PoseTab", "hash_string"]


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


def bouts_signature(video_id: str, bouts) -> str:
    """One video's slice of the label signature (pipeline.js labelSignature)."""
    frames = sum(b.end_frame - b.start_frame + 1 for b in bouts)
    spans = ",".join(sorted(f"{b.start_frame}-{b.end_frame}:{b.behavior_name}"
                            for b in bouts))
    return f"{video_id}|{len(bouts)}|{frames}|{hash_string(spans)}"


class PoseTab(_Base):
    # Body parts / animals / origin changed: the overlay's cached keypoints and
    # any model trained on the old selection no longer describe what's ticked.
    selectionChanged = QtCore.Signal()
    # The active iteration changed, or features were (re)written into it.
    iterationChanged = QtCore.Signal()

    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        self._task_id = None
        self._n_frames = None          # pose file length, from inspect
        self._inspected = None         # (path, origin, multi_animal) last inspected
        self._saved_selection: set | None = None   # body parts to re-tick after inspect
        self._pose_by_video: dict[str, str] = {}    # fallback when no project is open
        self._unchecked_videos: set = set()
        self._framerate_manual = False
        self._setting_fps = False

        outer = QtWidgets.QVBoxLayout(self)
        body = self._scroll_body(outer)

        # ---- pose file ------------------------------------------------------
        src = QtWidgets.QGroupBox("Pose file (for the current video)")
        sf = QtWidgets.QFormLayout(src)
        self.origin = NoWheelComboBox()
        self.origin.addItems(ORIGINS)
        self.origin.currentTextChanged.connect(self._on_source_changed)
        self.multi_animal = QtWidgets.QCheckBox("Multi-animal")
        self.multi_animal.toggled.connect(self._on_source_changed)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.origin, 1)
        orow.addWidget(self.multi_animal)
        sf.addRow("Pose software", self._wrap(orow))

        self.pose_path_label = QtWidgets.QLabel("(no video loaded)")
        self.pose_path_label.setWordWrap(True)
        self.pose_path_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.attach_btn = QtWidgets.QPushButton("Attach…")
        self.attach_btn.clicked.connect(self._attach_pose)
        prow = QtWidgets.QHBoxLayout()
        prow.addWidget(self.pose_path_label, 1)
        prow.addWidget(self.attach_btn)
        sf.addRow("File", self._wrap(prow))
        self.pose_info = self._hint("Attach the pose file that was estimated on this video.")
        sf.addRow(self.pose_info)
        body.addWidget(src)

        # ---- selection ------------------------------------------------------
        sel = QtWidgets.QGroupBox("Body parts")
        sv = QtWidgets.QVBoxLayout(sel)
        self.bodyparts_list = checklist(150)
        self.bodyparts_list.itemChanged.connect(self._on_bodyparts_changed)
        sv.addWidget(self.bodyparts_list)
        brow = QtWidgets.QHBoxLayout()
        all_btn = QtWidgets.QPushButton("All")
        none_btn = QtWidgets.QPushButton("None")
        for b in (all_btn, none_btn):
            mark_ghost(b)
        all_btn.clicked.connect(lambda: self._set_all_bodyparts(True))
        none_btn.clicked.connect(lambda: self._set_all_bodyparts(False))
        self.feature_width_label = self._hint()
        brow.addWidget(all_btn)
        brow.addWidget(none_btn)
        brow.addWidget(self.feature_width_label, 1)
        sv.addLayout(brow)
        self.animals_box = QtWidgets.QWidget()
        av = QtWidgets.QVBoxLayout(self.animals_box)
        av.setContentsMargins(0, 0, 0, 0)
        av.addWidget(QtWidgets.QLabel("Individuals"))
        self.animals_list = checklist(80)
        self.animals_list.itemChanged.connect(lambda _i: self.selectionChanged.emit())
        av.addWidget(self.animals_list)
        sv.addWidget(self.animals_box)
        self.animals_box.setVisible(False)
        body.addWidget(sel)

        # ---- parameters -----------------------------------------------------
        params = QtWidgets.QGroupBox("Feature extraction")
        pf = QtWidgets.QFormLayout(params)
        self.framerate = NoWheelDoubleSpinBox()
        self.framerate.setRange(1.0, 1000.0)
        self.framerate.setDecimals(3)
        self.framerate.setValue(30.0)
        self.framerate.setToolTip(
            "Read from the video. Only change it if the pose file was estimated "
            "at a different (e.g. subsampled) rate.")
        self.framerate.valueChanged.connect(self._on_framerate_edited)
        self.use_video_fps = QtWidgets.QPushButton("Use video's")
        mark_ghost(self.use_video_fps)
        self.use_video_fps.clicked.connect(self._reset_framerate)
        frow = QtWidgets.QHBoxLayout()
        frow.addWidget(self.framerate, 1)
        frow.addWidget(self.use_video_fps)
        pf.addRow("Frame rate", self._wrap(frow))
        self.llh = NoWheelDoubleSpinBox()
        self.llh.setRange(0.0, 1.0)
        self.llh.setSingleStep(0.05)
        self.llh.setValue(0.6)
        self.llh.setToolTip("Points below this likelihood are replaced by the "
                            "last confident position before features are built.")
        pf.addRow("Likelihood threshold", self.llh)
        self.min_duration = NoWheelDoubleSpinBox()
        self.min_duration.setRange(0.01, 10.0)
        self.min_duration.setSingleStep(0.05)
        self.min_duration.setValue(0.1)
        self.min_duration.setSuffix(" s")
        self.min_duration.setToolTip("The shortest behavior you care about. Sets the "
                                     "feature window, and so the time resolution.")
        self.min_duration.valueChanged.connect(self._update_derived)
        pf.addRow("Min behavior duration", self.min_duration)
        self.bin_label = self._hint()
        pf.addRow("", self.bin_label)
        body.addWidget(params)

        # ---- iteration + videos ----------------------------------------------
        it = QtWidgets.QGroupBox("Iteration")
        itf = QtWidgets.QVBoxLayout(it)
        irow = QtWidgets.QHBoxLayout()
        self.iteration_spin = NoWheelSpinBox()
        self.iteration_spin.setRange(0, 9999)
        self.iteration_spin.setPrefix("iteration-")
        self.iteration_spin.valueChanged.connect(self._on_iteration_edited)
        new_iter_btn = QtWidgets.QPushButton("New iteration")
        new_iter_btn.setToolTip("Start a fresh round: re-extract from the labels as they "
                                "stand now, then retrain. Extract does this for you "
                                "when the labels have changed.")
        new_iter_btn.clicked.connect(self._new_iteration)
        irow.addWidget(self.iteration_spin, 1)
        irow.addWidget(new_iter_btn)
        itf.addLayout(irow)
        self.iteration_info = self._hint()
        itf.addWidget(self.iteration_info)
        itf.addWidget(QtWidgets.QLabel("Extract for videos:"))
        self.videos_list = checklist(110)
        self.videos_list.itemChanged.connect(self._on_video_item_changed)
        itf.addWidget(self.videos_list)
        body.addWidget(it)
        body.addStretch(1)

        # ---- pinned actions -------------------------------------------------
        btns = QtWidgets.QHBoxLayout()
        self.extract_btn = QtWidgets.QPushButton("Extract features")
        mark_accent(self.extract_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.extract_btn.clicked.connect(self._extract)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.extract_btn, 1)
        btns.addWidget(self.stop_btn)
        outer.addLayout(btns)
        self.progress = QtWidgets.QProgressBar()
        outer.addWidget(self.progress)
        self.log = self._log_box()
        outer.addWidget(self.log)

        self._update_derived()

    # ------------------------------------------------------------ accessors
    def current_video_id(self):
        return getattr(self.ctx, "current_video_id", None)

    def pose_path_for(self, video_id) -> str | None:
        if not video_id:
            return None
        pm = self.ctx.project_manager()
        if pm is not None:
            p = pm.get_video_pose(video_id)
            if p:
                return p
        return self._pose_by_video.get(video_id)

    def selected_bodyparts(self) -> list[str]:
        return checked_values(self.bodyparts_list)

    def selected_animals(self) -> list[str] | None:
        if not self.multi_animal.isChecked():
            return None
        return checked_values(self.animals_list) or None

    def pose_settings(self) -> dict:
        """What the other tabs need to read a pose file the way this one does."""
        return {
            "origin": self.origin.currentText(),
            "multi_animal": self.multi_animal.isChecked(),
            "animals": self.selected_animals(),
            "llh_value": self.llh.value(),
        }

    def current_iteration_dir(self) -> str:
        return f"iteration-{self.iteration_spin.value()}"

    def log_line(self, line: str) -> None:
        self.log.appendPlainText(line)

    # -------------------------------------------------------------- lifecycle
    def refresh(self) -> None:
        """Cheap: runs on every label edit. No backend calls."""
        self._render_videos()
        self._update_iteration_info()

    def on_project_opened(self) -> None:
        pm = self.ctx.project_manager()
        self._unchecked_videos.clear()
        self._inspected = None
        if pm is None:
            self.refresh()
            return
        cfg = pm.manifest.get("preprocessing") or {}
        for w in (self.origin, self.multi_animal):
            w.blockSignals(True)
        if cfg.get("origin") in ORIGINS:
            self.origin.setCurrentText(cfg["origin"])
        self.multi_animal.setChecked(bool(cfg.get("multi_animal")))
        for w in (self.origin, self.multi_animal):
            w.blockSignals(False)
        if cfg.get("llh_value") is not None:
            self.llh.setValue(float(cfg["llh_value"]))
        if cfg.get("min_duration") is not None:
            self.min_duration.setValue(float(cfg["min_duration"]))
        self._framerate_manual = bool(cfg.get("framerate_manual"))
        if self._framerate_manual and cfg.get("framerate"):
            self._set_framerate(float(cfg["framerate"]))
        if cfg.get("bodyparts"):
            self._saved_selection = set(cfg["bodyparts"])
        self.iteration_spin.blockSignals(True)
        self.iteration_spin.setValue(pm.get_current_iteration())
        self.iteration_spin.blockSignals(False)
        self.refresh()
        self.iterationChanged.emit()

    def on_video_switched(self) -> None:
        if not self._framerate_manual:
            self._sync_framerate()
        pose = self.pose_path_for(self.current_video_id())
        if self.current_video_id() is None:
            self.pose_path_label.setText("(no video loaded)")
        else:
            self.pose_path_label.setText(pose or "(none attached)")
        self.refresh()
        if pose:
            self._inspect(pose)
        else:
            self._n_frames = None
            self.pose_info.setText("Attach the pose file that was estimated on this video.")

    # ----------------------------------------------------------- frame rate
    def _sync_framerate(self) -> None:
        fps = self.ctx.fps() if self.ctx.video_path() else None
        if fps and fps > 0:
            self._set_framerate(fps)

    def _set_framerate(self, value: float) -> None:
        self._setting_fps = True
        self.framerate.setValue(value)
        self._setting_fps = False
        self._update_derived()

    def _on_framerate_edited(self, _v) -> None:
        if not self._setting_fps:
            self._framerate_manual = True
        self._update_derived()

    def _reset_framerate(self) -> None:
        self._framerate_manual = False
        self._sync_framerate()
        self.ctx.status(f"Frame rate reset to the video's {self.framerate.value():g} fps.")

    def _update_derived(self, *_args) -> None:
        n = len(self.selected_bodyparts())
        self.feature_width_label.setText(
            "select at least 2" if n < 2 else f"{n} parts → {feature_width(n)} features")
        fps, dur = self.framerate.value(), self.min_duration.value()
        bw = bin_width(fps, dur)
        self.bin_label.setText(f"one feature row per {bw} frame{'s' if bw != 1 else ''} "
                               f"({bw / fps * 1000:.0f} ms)")

    # -------------------------------------------------------------- selection
    def _on_bodyparts_changed(self, _item=None) -> None:
        self._saved_selection = set(self.selected_bodyparts())
        self._update_derived()
        self.selectionChanged.emit()

    def _set_all_bodyparts(self, on: bool) -> None:
        self.bodyparts_list.blockSignals(True)
        set_all_checked(self.bodyparts_list, on)
        self.bodyparts_list.blockSignals(False)
        self._on_bodyparts_changed()

    def _on_source_changed(self, *_args) -> None:
        # OpenMonkeyStudio is single-animal 3D by definition.
        if self.origin.currentText() == "openmonkeystudio" and self.multi_animal.isChecked():
            self.multi_animal.blockSignals(True)
            self.multi_animal.setChecked(False)
            self.multi_animal.blockSignals(False)
        self.animals_box.setVisible(self.multi_animal.isChecked())
        self.selectionChanged.emit()
        pose = self.pose_path_for(self.current_video_id())
        if pose:
            self._inspect(pose)

    # ------------------------------------------------------------ attach
    def _attach_pose(self) -> None:
        video_id = self.current_video_id()
        if not video_id:
            self._warn("Attach pose file", "Load a video first — the pose file attaches to it.")
            return
        start = ""
        vp = self.ctx.video_path()
        if vp:
            start = os.path.dirname(vp)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Attach pose file", start, POSE_FILE_FILTER)
        if not path:
            return
        pm = self.ctx.project_manager()
        self._pose_by_video[video_id] = path
        if pm is not None:
            pm.set_video_pose(video_id, path)
            pm.flush_manifest()
        self.pose_path_label.setText(path)
        self._inspected = None
        self._inspect(path)
        self.refresh()
        self.selectionChanged.emit()
        self.ctx.status(f"Pose file attached to {video_id}.")

    def _inspect(self, pose_path: str) -> None:
        key = (pose_path, self.origin.currentText(), self.multi_animal.isChecked())
        client = self.ctx.client()
        if client is None:
            self.pose_info.setText("Waiting for the backend to read this pose file…")
            return
        origin, multi = key[1], key[2]
        video_id = self.current_video_id()

        def done(res: dict):
            if self.current_video_id() != video_id:
                return  # the user moved on to another video meanwhile
            self._inspected = key
            self._on_inspected(res)

        def fail(msg: str):
            self.pose_info.setText(f"Could not read this pose file: {msg}")

        self._spawn_call(lambda: client.inspect_pose(pose_path, origin, multi), done, fail)

    def retry_inspect(self) -> None:
        """Called once the backend comes up, for a pose file that couldn't be
        read while it was still starting."""
        pose = self.pose_path_for(self.current_video_id())
        if pose and self._inspected is None:
            self._inspect(pose)

    def _on_inspected(self, res: dict) -> None:
        bodyparts = res.get("bodyparts", []) or []
        animals = res.get("animals", []) or []
        keep = self._saved_selection
        if keep is not None and not (keep & set(bodyparts)):
            keep = None  # a different rig entirely — start from everything
        fill_checklist(self.bodyparts_list, bodyparts, keep)
        current_animals = set(checked_values(self.animals_list)) or None
        fill_checklist(self.animals_list, animals,
                       current_animals if current_animals and current_animals & set(animals)
                       else None)
        self.animals_box.setVisible(self.multi_animal.isChecked() and bool(animals))
        self._saved_selection = set(self.selected_bodyparts())
        self._n_frames = res.get("n_frames")
        info = (f"{self._n_frames} frames · {len(bodyparts)} body parts"
                + (f" · {len(animals)} animals" if animals else "")
                + (" · 3D" if res.get("is_3d") else "")
                + ("" if res.get("has_likelihood", True) else " · no likelihood column"))
        warn = length_mismatch(self._n_frames, self.ctx.video_frame_count())
        if warn:
            self.pose_info.setText(f"{info}\n{warn}")
            self.pose_info.setStyleSheet("color: #e3a008; font-weight: 600;")
            self.ctx.status(warn.lstrip("⚠ "))
        else:
            self.pose_info.setText(info)
            self.pose_info.setStyleSheet("")
        self._update_derived()
        self.selectionChanged.emit()

    # ------------------------------------------------------------ iterations
    def _on_iteration_edited(self, value: int) -> None:
        pm = self.ctx.project_manager()
        if pm is not None:
            pm.set_current_iteration(value)
        self._update_iteration_info()
        self.iterationChanged.emit()

    def _new_iteration(self) -> None:
        if self.ctx.project_manager() is None:
            self._warn("New iteration", "Open or create a project first (File menu).")
            return
        self.iteration_spin.setValue(self.iteration_spin.value() + 1)
        self.log_line(f"--- {self.current_iteration_dir()}: re-extract features from the "
                      f"current labels, then train ---")

    def _update_iteration_info(self) -> None:
        pm = self.ctx.project_manager()
        if pm is None:
            self.iteration_info.setText("Open or create a project (File menu) — features "
                                        "and models are written into it.")
            return
        meta = pm.get_iteration_meta(self.current_iteration_dir()) or {}
        if meta.get("extracted_at"):
            when = str(meta["extracted_at"]).replace("T", " ")[:16]
            self.iteration_info.setText(
                f"Features extracted {when} from {len(meta.get('videos') or [])} video(s)"
                + (f", {meta['n_rows']} rows" if meta.get("n_rows") else "") + ".")
        else:
            self.iteration_info.setText("No features extracted for this iteration yet.")

    def label_signature(self, video_ids) -> str:
        """Fingerprint of the labels feeding an iteration (pipeline.js labelSignature).

        Changes whenever a bout is added, removed, retimed or relabeled on any of
        the videos. The current video's in-memory bouts are newer than its saved
        copy; every other video's come from the project.
        """
        pm = self.ctx.project_manager()
        current = self.current_video_id()
        parts = []
        for vid in sorted(video_ids):
            if vid == current:
                bouts = list(self.ctx.bouts())
            elif pm is not None:
                bouts = pm.load_into_model(vid).bouts
            else:
                bouts = []
            parts.append(bouts_signature(vid, bouts))
        return ";".join(parts)

    def auto_advance_iteration(self, signature: str) -> bool:
        """Roll onto a new iteration if the labels moved on since this one was
        extracted. Without it the iteration counter is decorative: every
        re-extraction overwrites the previous round's features and models."""
        pm = self.ctx.project_manager()
        if pm is None:
            return False
        meta = pm.get_iteration_meta(self.current_iteration_dir())
        if not meta or not meta.get("label_signature") or meta["label_signature"] == signature:
            return False
        self.iteration_spin.setValue(self.iteration_spin.value() + 1)
        return True

    # ---------------------------------------------------------------- videos
    def _on_video_item_changed(self, item) -> None:
        vid = item.data(USER_ROLE)
        if item.checkState() == CHECKED:
            self._unchecked_videos.discard(vid)
        else:
            self._unchecked_videos.add(vid)

    def _render_videos(self) -> None:
        pm = self.ctx.project_manager()
        if pm is None:
            fill_checklist(self.videos_list, [], None)
            return
        ids, labels, disabled = [], [], set()
        for v in pm.manifest.get("videos", []):
            vid = v.get("id")
            ids.append(vid)
            has_pose = bool(self.pose_path_for(vid))
            labels.append(v.get("name", vid) + ("" if has_pose else "  (no pose file)"))
            if not has_pose:
                disabled.add(vid)
        checked = {i for i in ids if i not in self._unchecked_videos}
        fill_checklist(self.videos_list, ids, checked, labels=labels, disabled=disabled)

    # -------------------------------------------------------------- extract
    def _extract(self) -> None:
        title = "Can't extract features"
        pm = self.ctx.project_manager()
        if pm is None:
            self._warn(title, "No project is open. Features are written into the project "
                              "folder, so create or open a project first (File menu).")
            return
        client = self.ctx.client()
        if client is None:
            self._warn(title, "No backend connection yet. It starts with the app — wait "
                              "a moment, or use Backend ▸ Connect Local to retry.")
            return
        bodyparts = self.selected_bodyparts()
        if len(bodyparts) < 2:
            self._warn(title, "Select at least two body parts — features are built from "
                              "body-part pairs." if self.bodyparts_list.count()
                       else "No body parts loaded. Attach a pose file first.")
            return
        video_ids = checked_values(self.videos_list, USER_ROLE)
        if not video_ids:
            self._warn(title, "No videos with an attached pose file are selected.")
            return

        # Labels must be on disk before the backend reads them.
        save = getattr(self.ctx, "save_labels", None)
        if save:
            save()
        signature = self.label_signature(video_ids)
        advanced = self.auto_advance_iteration(signature)
        iteration_dir = self.current_iteration_dir()

        current = self.current_video_id()
        videos = []
        for vid in video_ids:
            max_frames = pm.get_video_frame_count(vid)
            if vid == current and self.ctx.video_frame_count():
                max_frames = self.ctx.video_frame_count()
            videos.append({
                "id": vid,
                "pose_path": self.pose_path_for(vid),
                "label_path": pm.get_video_labels_abs(vid),
                "max_frames": max_frames,
            })

        settings = self.pose_settings()
        req = build_extract_request(
            project_dir=str(pm.root), videos=videos,
            origin=settings["origin"], bodyparts=bodyparts,
            framerate=self.framerate.value(), min_duration=self.min_duration.value(),
            llh_value=settings["llh_value"], animals=settings["animals"],
            multi_animal=settings["multi_animal"], iteration_dir=iteration_dir,
        )
        # Persist what produced these features, so reopening the project (or
        # handing it to someone) reproduces the exact feature space.
        pm.update_preprocessing_config({
            "origin": req["origin"], "bodyparts": req["bodyparts"],
            "framerate": req["framerate"], "framerate_manual": self._framerate_manual,
            "min_duration": req["min_duration"], "llh_value": req["llh_value"],
            "multi_animal": req["multi_animal"], "animals": req["animals"],
        })
        pm.flush_manifest()

        self.log.clear()
        if advanced:
            self.log_line(f"Labels changed since the last extraction — starting {iteration_dir}.")
            self.ctx.status(f"New labels detected: starting {iteration_dir}.")
        self.log_line(f"Extracting {len(videos)} video(s) into {iteration_dir}…")
        self.extract_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setMaximum(max(1, len(videos)))
        self.progress.setValue(0)
        outcome = {"status": None, "error": None}

        def finish():
            self.extract_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)
            outcome["status"] = ev.get("status")
            if ev.get("status") == "error":
                outcome["error"] = ev.get("error") or "Extraction failed."

        def on_results(results: dict):
            for line in summarize_extraction(results):
                self.log_line(line)
            pm.set_iteration_meta(iteration_dir, {
                "label_signature": signature,
                "extracted_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "videos": [v.get("id") for v in results.get("videos") or []],
                "class_names": results.get("class_names"),
                "bodyparts": results.get("bodyparts"),
                "n_rows": sum(v.get("n_rows") or 0 for v in results.get("videos") or []),
            })
            pm.set_current_iteration(self.iteration_spin.value())
            pm.flush_manifest()
            self._update_iteration_info()
            self.iterationChanged.emit()
            self.ctx.status(f"Features extracted into {iteration_dir}. Next: Train.")

        def on_done():
            finish()
            status = outcome["status"]
            if status == "completed":
                self._spawn_call(lambda: client.extraction_results(self._task_id), on_results)
            elif status == "cancelled":
                self.log_line("Cancelled.")
                self.ctx.status("Extraction cancelled.")
            else:
                msg = outcome["error"] or f"Extraction ended with status {status!r}."
                self.log_line(f"ERROR: {msg}")
                self.ctx.status("Feature extraction failed.")
                QtWidgets.QMessageBox.critical(self, "Feature extraction failed", msg)

        def on_fail(msg: str):
            finish()
            self.log_line(f"ERROR: {msg}")
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
