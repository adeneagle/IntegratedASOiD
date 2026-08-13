"""Active-learning panel.

Run inference on the current video's pose file, then review: the uncertainty
timeline and the on-video HUD (both owned by the main window) light up, and
"Next uncertain" jumps to the frames the model is least sure about — where
new labels help most. Adapted from RealtimeASOID-python's version — the
AlState wiring is pipeline-agnostic, only the inference request body differs
(pose file instead of video crop params).

Additional ctx duck-type used here (on top of pipeline_panel's):
    pose_path() -> str | None    set_al_results(state)    set_al_threshold(t)
    al_state() -> AlState|None   current_frame() -> int   seek(frame)
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from labeling_shell.active_learning import AlState
from labeling_shell.no_wheel import NoWheelComboBox, NoWheelSpinBox
from labeling_shell.theme import mark_accent

from .pipeline_config import build_inference_request
from .pose_panel import ORIGINS, _Base


class ActiveLearningPanel(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        self._task_id = None
        form = QtWidgets.QFormLayout(self)

        self.ckpt = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._pick_ckpt)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(self.ckpt, 1)
        crow.addWidget(browse)
        form.addRow("Checkpoint", crow)

        self.origin = NoWheelComboBox()
        self.origin.addItems(ORIGINS)
        form.addRow("Pose software", self.origin)

        self.threshold = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.threshold.setRange(0, 100)
        self.threshold.setValue(50)
        self.threshold.valueChanged.connect(self._on_threshold)
        self.threshold_label = QtWidgets.QLabel("0.50")
        trow = QtWidgets.QHBoxLayout()
        trow.addWidget(self.threshold, 1)
        trow.addWidget(self.threshold_label)
        form.addRow("Uncertainty ≥", trow)

        self.min_segment = NoWheelSpinBox()
        self.min_segment.setRange(1, 10000)
        self.min_segment.setValue(1)
        self.min_segment.valueChanged.connect(self._refresh_stats)
        form.addRow("Min segment length", self.min_segment)

        self.progress = QtWidgets.QProgressBar()
        form.addRow(self.progress)

        run_btn = QtWidgets.QPushButton("Run inference (current video)")
        mark_accent(run_btn)
        run_btn.clicked.connect(self._run)
        next_btn = QtWidgets.QPushButton("Next uncertain →")
        next_btn.clicked.connect(self._next_uncertain)
        form.addRow(run_btn)
        form.addRow(next_btn)

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
        form.addRow(self.stats_group)

    def refresh(self) -> None:
        pass

    # ------------------------------------------------------------- handlers
    def _pick_ckpt(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Model", "", "Model bundle (*.joblib);;All files (*)")
        if path:
            self.ckpt.setText(path)

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
        n_frames = None
        if hasattr(self.ctx, "current_frame"):
            player = getattr(self.ctx, "player", None)
            if player is not None and player.reader is not None:
                n_frames = player.reader.num_frames
        stats = al.stats(self._threshold_value(), self.min_segment.value(),
                         label_at=self._label_at, n_frames=n_frames)
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

    def _run(self) -> None:
        client = self._client()
        pose_path = self.ctx.pose_path()
        if not client:
            return
        if not self.ckpt.text():
            self.ctx.status("Choose a model first.")
            return
        if not pose_path:
            self.ctx.status("Attach a pose file for this video first (Pose tab).")
            return
        self.progress.setValue(0)

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)

        def on_stream_done():
            if self._task_id:
                self._spawn_call(
                    lambda: client.inference_results(self._task_id), self._on_results)

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.inference_status(self._task_id), on_event, on_stream_done)

        req = build_inference_request(
            checkpoint_path=self.ckpt.text(), pose_path=pose_path,
            origin=self.origin.currentText(),
        )
        self.ctx.status("Running inference…")
        self._spawn_call(lambda: client.run_inference(req), start)

    def _on_results(self, results: dict) -> None:
        state = AlState()
        state.load_results(results)
        self.ctx.set_al_results(state)
        n = len(state.by_index)
        self.ctx.status(f"Inference done — {n} frames, classes {state.class_names}")
        self._refresh_stats()

    def _next_uncertain(self) -> None:
        al = self.ctx.al_state()
        if not al:
            self.ctx.status("Run inference first.")
            return
        nxt = al.next_uncertain(self.ctx.current_frame(), self._threshold_value())
        if nxt is None:
            self.ctx.status("No more uncertain frames above threshold.")
        else:
            self.ctx.seek(nxt)
            self.ctx.status(f"Jumped to uncertain frame {nxt}")
