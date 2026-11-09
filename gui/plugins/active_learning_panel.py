"""Active-learning panel.

Run inference on the current video's pose file, then review: the uncertainty
timeline and the on-video HUD (both owned by the main window) light up, and the
jump buttons walk the frames the model is least sure about — where new labels
help most. Adapted from RealtimeASOID-python's version — the AlState wiring is
pipeline-agnostic, only the inference request body differs (pose file instead
of video crop params).

Results are saved per video into the project's `predictions/` folder and
restored when the video is opened again, so a review session survives a
restart and switching videos never shows one video's predictions over another.

Additional ctx duck-type used here (on top of pipeline_panel's):
    pose_path() -> str | None    set_al_results(state|None)   set_al_threshold(t)
    al_state() -> AlState|None   current_frame() -> int       seek(frame)
    video_frame_count() -> int | None
"""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from labeling_shell.active_learning import AlState
from labeling_shell.no_wheel import NoWheelSpinBox
from labeling_shell.theme import mark_accent

from .common import ModelPicker, _Base
from .pipeline_config import al_label_mapping, build_inference_request


def trim_results(results: dict, n_frames) -> dict:
    """Drop predictions past the end of the video (pose files that track a longer
    recording than the loaded clip). In memory only — the saved file keeps all."""
    if not n_frames:
        return results
    frames = [f for f in results.get("frames") or [] if int(f.get("index", 0)) < n_frames]
    return {**results, "frames": frames}


class ActiveLearningPanel(_Base):
    def __init__(self, ctx, pose_tab=None, parent=None):
        super().__init__(ctx, parent)
        self.pose_tab = pose_tab
        self._task_id = None

        outer = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.model = ModelPicker(ctx)
        self.model.changed.connect(self._update_model_info)
        form.addRow("Model", self.model)
        self.model_info = self._hint()
        form.addRow(self.model_info)
        outer.addLayout(form)

        run_row = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run inference on this video")
        mark_accent(self.run_btn)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.stop_btn)
        outer.addLayout(run_row)
        self.progress = QtWidgets.QProgressBar()
        outer.addWidget(self.progress)

        review = QtWidgets.QGroupBox("Review")
        rf = QtWidgets.QFormLayout(review)
        self.threshold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold.setRange(0, 100)
        self.threshold.setValue(50)
        self.threshold.valueChanged.connect(self._on_threshold)
        self.threshold_label = QtWidgets.QLabel("0.50")
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(self.threshold, 1)
        trow.addWidget(self.threshold_label)
        rf.addRow("Uncertainty ≥", self._wrap(trow))

        self.min_segment = NoWheelSpinBox()
        self.min_segment.setRange(1, 10000)
        self.min_segment.setValue(1)
        self.min_segment.setSuffix(" frames")
        self.min_segment.valueChanged.connect(self._refresh_stats)
        rf.addRow("Min segment", self.min_segment)

        nav = QtWidgets.QHBoxLayout()
        self.prev_btn = QtWidgets.QPushButton("← Prev uncertain")
        self.next_btn = QtWidgets.QPushButton("Next uncertain →")
        self.worst_btn = QtWidgets.QPushButton("Least confident")
        self.prev_btn.clicked.connect(lambda: self._jump(-1))
        self.next_btn.clicked.connect(lambda: self._jump(+1))
        self.worst_btn.clicked.connect(self._jump_worst)
        for b in (self.prev_btn, self.next_btn, self.worst_btn):
            nav.addWidget(b)
        rf.addRow(nav)
        outer.addWidget(review)

        self.stats_group = QtWidgets.QGroupBox("Stats")
        sf = QtWidgets.QFormLayout(self.stats_group)
        self.stat_total = QtWidgets.QLabel("—")
        self.stat_mean = QtWidgets.QLabel("—")
        self.stat_median = QtWidgets.QLabel("—")
        self.stat_above = QtWidgets.QLabel("—")
        self.stat_segments = QtWidgets.QLabel("—")
        self.stat_misclassified = QtWidgets.QLabel("—")
        self.stat_classes = QtWidgets.QLabel("—")
        self.stat_classes.setWordWrap(True)
        sf.addRow("Total frames", self.stat_total)
        sf.addRow("Mean uncertainty", self.stat_mean)
        sf.addRow("Median uncertainty", self.stat_median)
        sf.addRow("Above threshold", self.stat_above)
        sf.addRow("Segments", self.stat_segments)
        sf.addRow("Misclassified", self.stat_misclassified)
        sf.addRow("Classes", self.stat_classes)
        self.stats_group.setVisible(False)
        outer.addWidget(self.stats_group)
        outer.addWidget(self._hint(
            "Relabel in place with the number keys; the timeline's magenta ticks mark "
            "frames where the model disagrees with your label. When you've fixed "
            "enough, re-extract on the Pose tab (a new iteration starts "
            "automatically) and retrain."))
        outer.addStretch(1)
        self._update_model_info()

    # -------------------------------------------------------------- lifecycle
    def refresh(self) -> None:
        self._refresh_stats()

    def on_project_opened(self) -> None:
        self.model.set_entries([])
        self.model.reload()

    def on_video_switched(self) -> None:
        """Show this video's saved predictions, or nothing — never the last video's."""
        pm = self.ctx.project_manager()
        vid = getattr(self.ctx, "current_video_id", None)
        saved = pm.load_predictions(vid) if (pm is not None and vid) else None
        if not saved or not saved.get("frames"):
            self.ctx.set_al_results(None)
            self._refresh_stats()
            return
        self._apply_results(saved, saved.get("model_config"), announce=False)
        ckpt = saved.get("checkpoint_path")
        if ckpt and not self.model.select_path(ckpt) and os.path.exists(ckpt):
            self.model.add_path(ckpt, {"config": saved.get("model_config")})
        self.ctx.status(f"Restored saved predictions for {vid} "
                        f"({os.path.basename(ckpt or '?')}).")

    # ------------------------------------------------------------- model info
    def _update_model_info(self) -> None:
        e = self.model.current_entry()
        if not e:
            self.model_info.setText("Train a model on the Train tab, or Browse… for one.")
            return
        bits = []
        if e.get("class_names"):
            bits.append("classes: " + ", ".join(e["class_names"]))
        bps = e.get("bodyparts")
        if bps:
            bits.append(f"{len(bps)} body parts")
            if self.pose_tab is not None:
                sel = self.pose_tab.selected_bodyparts()
                if sel and list(sel) != list(bps):
                    bits.append("⚠ differs from the Pose tab's selection — inference uses "
                                "the model's own")
        self.model_info.setText(" · ".join(bits) or os.path.basename(e.get("path", "")))

    # ------------------------------------------------------------- handlers
    def _threshold_value(self) -> float:
        return self.threshold.value() / 100.0

    def _on_threshold(self, _v: int) -> None:
        self.threshold_label.setText(f"{self._threshold_value():.2f}")
        self.ctx.set_al_threshold(self._threshold_value())
        self._refresh_stats()

    def _label_at(self, i: int):
        for b in self.ctx.bouts():
            if b.start_frame <= i <= b.end_frame:
                return b.behavior_name
        return None

    def _refresh_stats(self, *_args) -> None:
        al = self.ctx.al_state()
        if al is None:
            self.stats_group.setVisible(False)
            return
        stats = al.stats(self._threshold_value(), self.min_segment.value(),
                         label_at=self._label_at, n_frames=self.ctx.video_frame_count())
        if stats is None:
            self.stats_group.setVisible(False)
            return
        self.stats_group.setVisible(True)
        self.stat_total.setText(str(stats["total_frames"]))
        self.stat_mean.setText(f"{stats['mean_uncertainty']:.3f}")
        self.stat_median.setText(f"{stats['median_uncertainty']:.3f}")
        self.stat_above.setText(
            f"{stats['above_threshold_frames']} ({stats['above_threshold_pct']:.1f}%)")
        minlen = stats["min_segment_length"]
        self.stat_segments.setText(
            f"{stats['n_segments']} (≥{minlen} frame{'' if minlen == 1 else 's'})")
        if "misclassified_count" in stats:
            self.stat_misclassified.setText(
                f"{stats['misclassified_count']} ({stats['misclassified_pct']:.1f}%)")
        else:
            self.stat_misclassified.setText("—")
        self.stat_classes.setText(", ".join(stats["class_names"]))

    def _settings(self) -> dict:
        if self.pose_tab is not None:
            return self.pose_tab.pose_settings()
        return {"origin": "deeplabcut", "multi_animal": False, "animals": None,
                "llh_value": 0.6}

    def _run(self) -> None:
        title = "Can't run inference"
        ckpt = self.model.current_path()
        if not ckpt:
            self._warn(title, "Choose a model first — train one on the Train tab, or Browse….")
            return
        pose_path = self.ctx.pose_path()
        if not pose_path:
            self._warn(title, "This video has no pose file attached. Attach one on the "
                              "Pose tab first.")
            return
        client = self._client()
        if client is None:
            return

        # Scope to the video: a pose file tracking a longer recording would
        # otherwise squeeze hours of predictions into this clip's timeline.
        n = self.ctx.video_frame_count()
        s = self._settings()
        req = build_inference_request(
            checkpoint_path=ckpt, pose_path=pose_path, origin=s["origin"],
            llh_value=s["llh_value"], animals=s["animals"], multi_animal=s["multi_animal"],
            frame_ranges=[[0, n - 1]] if n else None,
        )
        entry = self.model.current_entry() or {}
        config = entry.get("config")
        video_id = getattr(self.ctx, "current_video_id", None)
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        outcome = {"status": None, "error": None}

        def finish():
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)
            outcome["status"] = ev.get("status")
            outcome["error"] = ev.get("error")

        def on_results(results: dict):
            self._save(video_id, results, ckpt, config)
            # The user may have switched videos while it ran; the saved copy is
            # restored when they come back to this one.
            if getattr(self.ctx, "current_video_id", None) == video_id:
                self._apply_results(results, config)

        def on_stream_done():
            finish()
            if outcome["status"] == "completed":
                self._spawn_call(lambda: client.inference_results(self._task_id), on_results)
            elif outcome["status"] == "cancelled":
                self.ctx.status("Inference cancelled.")
            else:
                msg = outcome["error"] or f"Inference ended with status {outcome['status']!r}."
                self.ctx.status("Inference failed.")
                QtWidgets.QMessageBox.critical(self, "Inference failed", msg)

        def on_fail(_msg: str):
            finish()

        def start(res: dict):
            self._task_id = res.get("task_id")
            if not self._task_id:
                self.ctx.status(f"Backend did not return a task id: {res}")
                finish()
                return
            self._spawn_stream(lambda: client.inference_status(self._task_id),
                               on_event, on_stream_done, on_fail)

        self.ctx.status("Running inference…")
        self._spawn_call(lambda: client.run_inference(req), start, on_fail)

    def _save(self, video_id, results: dict, ckpt: str, config) -> None:
        pm = self.ctx.project_manager()
        if pm is None or not video_id:
            return
        try:
            pm.save_predictions(video_id, {**results, "checkpoint_path": ckpt,
                                           "model_config": config})
        except OSError as e:
            self.ctx.status(f"Could not save predictions: {e}")

    def _apply_results(self, results: dict, config, announce: bool = True) -> None:
        state = AlState()
        state.load_results(trim_results(results, self.ctx.video_frame_count()),
                           **al_label_mapping(config))
        self.ctx.set_al_results(state)
        self._refresh_stats()
        if announce:
            self.ctx.status(f"Inference done — {len(state.by_index)} frames, classes "
                            f"{', '.join(state.class_names)}. Use the jump buttons to review.")

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_inference(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))

    # ------------------------------------------------------------ navigation
    def _jump(self, direction: int) -> None:
        al = self.ctx.al_state()
        if not al:
            self.ctx.status("Run inference first.")
            return
        cur, t = self.ctx.current_frame(), self._threshold_value()
        if direction > 0:
            target = al.next_uncertain(cur, t)
        else:
            target = max((i for i in al.uncertain_frames(t) if i < cur), default=None)
        if target is None:
            self.ctx.status(f"No uncertain frames {'after' if direction > 0 else 'before'} "
                            f"this one above {t:.2f}.")
            return
        self.ctx.seek(target)
        self.ctx.status(f"Uncertain frame {target} (uncertainty {al.uncertainty(target):.2f}).")

    def _jump_worst(self) -> None:
        al = self.ctx.al_state()
        if not al or not al.by_index:
            self.ctx.status("Run inference first.")
            return
        target = max(al.by_index, key=lambda i: al.uncertainty(i) or 0.0)
        self.ctx.seek(target)
        self.ctx.status(f"Least confident frame: {target} "
                        f"(uncertainty {al.uncertainty(target):.2f}).")
