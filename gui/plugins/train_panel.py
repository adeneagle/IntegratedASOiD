"""Train tab — RandomForest + active-learning hyperparameters, start/stop, metrics.

Qt port of the "train" tab in LabelingApp/plugins/pipeline.js, laid out like
RealtimeASOID's TrainTab: the form scrolls, actions/metrics/log are pinned,
the positive class is a checklist (0 = multi-class, 1 = binarize, >=2 = union),
and a finished model is recorded in the project manifest.
"""

from __future__ import annotations

import datetime
import os

from PySide6 import QtCore, QtWidgets

from labeling_shell.no_wheel import NoWheelDoubleSpinBox, NoWheelSpinBox
from labeling_shell.theme import mark_accent

from .common import (
    CHECKED, USER_ROLE, _Base, checked_values, checklist, fill_checklist,
)
from .pipeline_config import (
    build_training_config, format_train_event, resolve_binarize, threshold_floor_warning,
)


class TrainTab(_Base):
    # A model finished training: (checkpoint path, manifest entry).
    modelTrained = QtCore.Signal(str, dict)

    def __init__(self, ctx, pose_tab=None, parent=None):
        super().__init__(ctx, parent)
        # The Pose tab owns the active iteration. Wired explicitly by
        # PipelinePanel rather than reached through ctx or the parent chain.
        self.pose_tab = pose_tab
        self._task_id = None
        self._unchecked_videos: set = set()

        outer = QtWidgets.QVBoxLayout(self)
        body = self._scroll_body(outer)

        self.iteration_label = QtWidgets.QLabel()
        self.iteration_label.setWordWrap(True)
        body.addWidget(self.iteration_label)
        body.addWidget(QtWidgets.QLabel("Train on videos:"))
        self.videos_list = checklist(110)
        self.videos_list.itemChanged.connect(self._on_video_item_changed)
        body.addWidget(self.videos_list)

        al = QtWidgets.QGroupBox("Active learning")
        form = QtWidgets.QFormLayout(al)
        self.init_ratio = NoWheelDoubleSpinBox()
        self.init_ratio.setRange(0.001, 1.0)
        self.init_ratio.setDecimals(3)
        self.init_ratio.setSingleStep(0.005)
        self.init_ratio.setValue(0.01)
        self.init_ratio.setToolTip("Fraction of each class the first forest is trained on.")
        self.conf_threshold = NoWheelDoubleSpinBox()
        self.conf_threshold.setRange(0.0, 1.0)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(0.5)
        self.conf_threshold.setToolTip("Samples the forest is less confident than this "
                                       "about are added to the training set next round.")
        self.max_iter = NoWheelSpinBox()
        self.max_iter.setRange(1, 100000)
        self.max_iter.setValue(100)
        self.max_samples_iter = NoWheelSpinBox()
        self.max_samples_iter.setRange(1, 100000)
        self.max_samples_iter.setValue(20)
        self.max_samples_iter.setToolTip("Most low-confidence samples added per class, per round.")
        self.exclude_other = QtWidgets.QCheckBox("Exclude 'other' (unlabeled frames)")
        self.exclude_other.setToolTip(
            "Unlabeled frames are trained on as 'other' by default. Tick this when "
            "you haven't labeled everything you watched — the model then only learns "
            "to tell your behaviors apart.")
        form.addRow("Initial ratio", self.init_ratio)
        form.addRow("Confidence threshold", self.conf_threshold)
        form.addRow("Max rounds", self.max_iter)
        form.addRow("Samples / round", self.max_samples_iter)
        form.addRow("", self.exclude_other)
        self.threshold_hint = self._hint()
        self.threshold_hint.setStyleSheet("color: #e3a008;")
        form.addRow(self.threshold_hint)
        self.conf_threshold.valueChanged.connect(self._update_threshold_hint)
        self.exclude_other.toggled.connect(self._update_threshold_hint)
        body.addWidget(al)

        pos = QtWidgets.QGroupBox("Positive class (binarize)")
        pv = QtWidgets.QVBoxLayout(pos)
        self.binarize_list = checklist(110)
        self.binarize_list.itemChanged.connect(lambda _i: self._update_binarize_hint())
        self.binarize_hint = self._hint()
        pv.addWidget(self.binarize_list)
        pv.addWidget(self.binarize_hint)
        body.addWidget(pos)

        out = QtWidgets.QFormLayout()
        self.model_name = QtWidgets.QLineEdit("rf_model")
        out.addRow("Model name", self.model_name)
        body.addLayout(out)
        body.addStretch(1)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start training")
        mark_accent(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.start_btn, 1)
        btns.addWidget(self.stop_btn)
        outer.addLayout(btns)
        self.progress = QtWidgets.QProgressBar()
        outer.addWidget(self.progress)

        grid = QtWidgets.QGridLayout()
        self.m_round = QtWidgets.QLabel("—")
        self.m_f1 = QtWidgets.QLabel("—")
        self.m_ceiling = QtWidgets.QLabel("—")
        self.m_samples = QtWidgets.QLabel("—")
        self.m_lowconf = QtWidgets.QLabel("—")
        self.m_per_class = QtWidgets.QLabel("—")
        self.m_per_class.setWordWrap(True)
        cells = [("Round", self.m_round), ("Macro F1", self.m_f1),
                 ("Ceiling F1", self.m_ceiling), ("Samples", self.m_samples),
                 ("Low-conf left", self.m_lowconf), ("Per class", self.m_per_class)]
        for i, (name, w) in enumerate(cells):
            grid.addWidget(self._hint(name), i // 3 * 2, i % 3)
            grid.addWidget(w, i // 3 * 2 + 1, i % 3)
        self.metrics_box = QtWidgets.QWidget()
        self.metrics_box.setLayout(grid)
        outer.addWidget(self.metrics_box)
        self.log = self._log_box(130)
        outer.addWidget(self.log)
        self.refresh()

    # -------------------------------------------------------------- refresh
    def iteration_dir(self) -> str:
        if self.pose_tab is not None:
            return self.pose_tab.current_iteration_dir()
        pm = self.ctx.project_manager()
        return pm.iteration_name() if pm else "iteration-0"

    def _features_path(self, video_id: str):
        pm = self.ctx.project_manager()
        return pm.get_video_features_abs(video_id, self.iteration_dir()) if pm else None

    def refresh(self) -> None:
        self._render_videos()
        self._render_positive_classes()

    def on_project_opened(self) -> None:
        self._unchecked_videos.clear()
        pm = self.ctx.project_manager()
        cfg = (pm.manifest.get("training") or {}) if pm else {}
        for key, w in (("init_ratio", self.init_ratio), ("conf_threshold", self.conf_threshold),
                       ("max_iter", self.max_iter), ("max_samples_iter", self.max_samples_iter)):
            if cfg.get(key) is not None:
                w.setValue(cfg[key])
        if cfg.get("exclude_other") is not None:
            self.exclude_other.setChecked(bool(cfg["exclude_other"]))
        if cfg.get("model_name"):
            self.model_name.setText(cfg["model_name"])
        positives = cfg.get("label_union") or ([cfg["binarize"]] if cfg.get("binarize") else [])
        self._render_positive_classes(set(positives))
        self.refresh()

    def _on_video_item_changed(self, item) -> None:
        vid = item.data(USER_ROLE)
        if item.checkState() == CHECKED:
            self._unchecked_videos.discard(vid)
        else:
            self._unchecked_videos.add(vid)

    def _render_videos(self) -> None:
        pm = self.ctx.project_manager()
        it = self.iteration_dir()
        if pm is None:
            self.iteration_label.setText("Open or create a project first (File menu).")
            fill_checklist(self.videos_list, [], None)
            return
        ids, labels, disabled = [], [], set()
        for v in pm.manifest.get("videos", []):
            vid = v.get("id")
            ids.append(vid)
            ready = os.path.exists(self._features_path(vid))
            labels.append(v.get("name", vid) + ("" if ready else f"  (no features in {it})"))
            if not ready:
                disabled.add(vid)
        n_ready = len(ids) - len(disabled)
        self.iteration_label.setText(
            f"Training on <b>{it}</b> — {n_ready} of {len(ids)} video(s) have features."
            + ("" if n_ready else "  Extract features on the Pose tab first."))
        checked = {i for i in ids if i not in self._unchecked_videos}
        fill_checklist(self.videos_list, ids, checked, labels=labels, disabled=disabled)

    def positive_behaviors(self) -> list[str]:
        return checked_values(self.binarize_list)

    def _render_positive_classes(self, checked: set | None = None) -> None:
        """Rebuild the positive-class checklist, keeping whatever is still ticked
        (nothing ticked = multi-class, the default)."""
        keep = set(self.positive_behaviors()) if checked is None else checked
        names = [b.name for b in self.ctx.behaviors()]
        fill_checklist(self.binarize_list, names, keep & set(names))
        self._update_binarize_hint()

    def n_model_classes(self) -> int:
        if self.positive_behaviors():
            return 2  # binary: positive vs everything else
        return len(self.ctx.behaviors()) + (0 if self.exclude_other.isChecked() else 1)

    def _update_threshold_hint(self, *_args) -> None:
        warn = threshold_floor_warning(self.conf_threshold.value(), self.n_model_classes())
        self.threshold_hint.setText(warn or "")
        self.threshold_hint.setVisible(bool(warn))

    def _update_binarize_hint(self) -> None:
        self._update_threshold_hint()
        pos = self.positive_behaviors()
        if not pos:
            self.binarize_hint.setText("Nothing ticked → multi-class over every behavior.")
        elif len(pos) == 1:
            self.binarize_hint.setText(f"Binary: \"{pos[0]}\" vs everything else.")
        else:
            self.binarize_hint.setText(f"Binary: {len(pos)} ticked behaviors merged into "
                                       f"\"positive\" vs everything else.")

    # ---------------------------------------------------------------- config
    def selected_video_ids(self) -> list[str]:
        return checked_values(self.videos_list, USER_ROLE)

    def build_config(self) -> dict:
        pm = self.ctx.project_manager()
        it = self.iteration_dir()
        binarize, label_union = resolve_binarize(self.positive_behaviors())
        return build_training_config(
            project_dir=str(pm.root) if pm else "",
            video_ids=self.selected_video_ids(),
            iteration_dir=it,
            init_ratio=self.init_ratio.value(),
            max_iter=self.max_iter.value(),
            max_samples_iter=self.max_samples_iter.value(),
            conf_threshold=self.conf_threshold.value(),
            exclude_other=self.exclude_other.isChecked(),
            binarize=binarize,
            label_union=label_union,
            # Beside the features it was trained on: iteration-N/models.
            checkpoint_dir=str(pm.get_iteration_models_dir(it)) if pm else None,
            model_name=self.model_name.text().strip() or "rf_model",
        )

    # ------------------------------------------------------------------- run
    def _start(self) -> None:
        title = "Can't start training"
        pm = self.ctx.project_manager()
        if pm is None:
            self._warn(title, "Open or create a project first (File menu).")
            return
        client = self._client()
        if client is None:
            return
        self._render_videos()
        if not self.selected_video_ids():
            self._warn(title, f"No videos with features in {self.iteration_dir()} are "
                              f"selected. Run Extract features on the Pose tab first.")
            return
        config = self.build_config()
        pm.update_training_config(config)
        pm.flush_manifest()

        self.log.clear()
        for w in (self.m_round, self.m_f1, self.m_ceiling, self.m_samples,
                  self.m_lowconf, self.m_per_class):
            w.setText("—")
        self.progress.setMaximum(config["max_iter"])
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log.appendPlainText(f"Training on {len(config['video_ids'])} video(s) from "
                                 f"{config['iteration_dir']}…")
        names = self.ctx.behaviors()
        class_names: list = [b.name for b in names]
        # The router re-sends the terminal event after replaying the log, so the
        # "completed" line can arrive twice; record the model once.
        registered = set()

        def finish():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

        def on_event(ev: dict):
            nonlocal class_names
            line = format_train_event(ev, config["max_iter"])
            if ev.get("status") == "completed" and ev.get("checkpoint_path") in registered:
                line = None
            if line:
                self.log.appendPlainText(line)
            kind, status = ev.get("event"), ev.get("status")
            if kind == "start" and ev.get("classes"):
                class_names = ev["classes"]
            elif kind == "ceiling":
                self.m_ceiling.setText(f"{ev.get('macro_f1', 0):.3f}")
            elif kind == "iteration":
                self.m_round.setText(f"{ev.get('iteration')} / {config['max_iter']}")
                self.m_f1.setText(f"{ev.get('macro_f1', 0):.3f}")
                self.m_samples.setText(str(ev.get("n_train")))
                self.m_lowconf.setText(str(ev.get("n_lowconf_remaining")))
                per = ev.get("f1_per_class") or []
                self.m_per_class.setText(", ".join(
                    f"{class_names[i] if i < len(class_names) else i} {f:.2f}"
                    for i, f in enumerate(per)))
                if ev.get("ceiling_macro_f1") is not None:
                    self.m_ceiling.setText(f"{ev['ceiling_macro_f1']:.3f}")
                self.progress.setValue(min(config["max_iter"], int(ev.get("iteration") or 0)))
            elif status == "completed":
                finish()
                self.progress.setValue(self.progress.maximum())
                path = ev.get("checkpoint_path")
                if path and path not in registered:
                    registered.add(path)
                    self._register_model(config, ev)
            elif status in ("error", "cancelled"):
                finish()
                self.ctx.status("Training stopped." if status == "cancelled"
                                else "Training failed — see the Train tab log.")

        def on_fail(msg: str):
            finish()
            self.log.appendPlainText(f"ERROR: {msg}")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if not self._task_id:
                on_fail(f"Backend did not return a task id: {res}")
                return
            self._spawn_stream(lambda: client.training_status(self._task_id),
                               on_event, finish, on_fail)

        self._spawn_call(lambda: client.start_training(config), start, on_fail)

    def _register_model(self, config: dict, ev: dict) -> None:
        """Record the trained model in the project manifest, with the parts of the
        config needed to interpret it later — binarize/label_union in particular,
        without which Active Learning can't map a behavior name into a binarized
        model's collapsed label space."""
        path = ev["checkpoint_path"]
        entry = {
            "name": os.path.basename(path),
            "path": path,
            "created": datetime.date.today().isoformat(),
            "iteration": config.get("iteration_dir"),
            "macro_f1": ev.get("macro_f1"),
            "ceiling_macro_f1": ev.get("ceiling_macro_f1"),
            "class_names": ev.get("class_names"),
            "source": "trained",
            "config": {k: config.get(k) for k in (
                "video_ids", "init_ratio", "max_iter", "max_samples_iter",
                "conf_threshold", "exclude_other", "binarize", "label_union")},
        }
        pm = self.ctx.project_manager()
        try:
            if pm is not None:
                pm.add_model(entry)
                pm.flush_manifest()
        except Exception as e:  # noqa: BLE001
            # A manifest write failing must not look like a training failure.
            self.log.appendPlainText(f"could not record the model in the project: {e}")
        self.log.appendPlainText(f"saved {path}")
        self.ctx.status(f"Model saved: {entry['name']} — it's now selected on the "
                        f"Active Learning and Predict tabs.")
        self.modelTrained.emit(path, entry)

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_training(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))
