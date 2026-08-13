"""Train tab — RandomForest + active-learning hyperparameters, start/stop, metrics.

Qt port of the "train" tab in LabelingApp/plugins/pipeline.js (~lines 996-1194).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from labeling_shell.no_wheel import NoWheelDoubleSpinBox, NoWheelSpinBox
from labeling_shell.theme import mark_accent

from .pipeline_config import build_training_config
from .pose_panel import _Base


class TrainTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Train on videos:"))
        self.videos_list = QtWidgets.QListWidget()
        self.videos_list.setMaximumHeight(90)
        layout.addWidget(self.videos_list)

        form = QtWidgets.QFormLayout()
        self.init_ratio = NoWheelDoubleSpinBox()
        self.init_ratio.setRange(0.001, 1.0)
        self.init_ratio.setDecimals(3)
        self.init_ratio.setValue(0.01)
        self.conf_threshold = NoWheelDoubleSpinBox()
        self.conf_threshold.setRange(0.0, 1.0)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.setValue(0.5)
        self.max_iter = NoWheelSpinBox()
        self.max_iter.setRange(1, 100000)
        self.max_iter.setValue(100)
        self.max_samples_iter = NoWheelSpinBox()
        self.max_samples_iter.setRange(1, 100000)
        self.max_samples_iter.setValue(20)
        self.exclude_other = QtWidgets.QCheckBox("Exclude 'other'")
        self.model_name = QtWidgets.QLineEdit()
        form.addRow("Init ratio", self.init_ratio)
        form.addRow("Confidence threshold", self.conf_threshold)
        form.addRow("Max iterations", self.max_iter)
        form.addRow("Samples / iteration", self.max_samples_iter)
        form.addRow("", self.exclude_other)
        form.addRow("Model name", self.model_name)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Binarize (positive class only):"))
        self.binarize_list = QtWidgets.QListWidget()
        self.binarize_list.setMaximumHeight(90)
        layout.addWidget(self.binarize_list)

        btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start training")
        mark_accent(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        layout.addLayout(btns)

        self.metrics = QtWidgets.QLabel("—")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self._task_id = None

    def refresh(self) -> None:
        pm = self.ctx.project_manager()
        self.videos_list.clear()
        if pm is not None:
            for v in pm.manifest.get("videos", []):
                item = QtWidgets.QListWidgetItem(v.get("name", v.get("id")))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, v.get("id"))
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked)
                self.videos_list.addItem(item)

        current = self.binarize_list.selectedItems()
        current_names = {i.text() for i in current}
        self.binarize_list.clear()
        for b in self.ctx.behaviors():
            item = QtWidgets.QListWidgetItem(b.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked
                               if not current_names or b.name in current_names
                               else QtCore.Qt.CheckState.Unchecked)
            self.binarize_list.addItem(item)

    def _selected_video_ids(self) -> list[str]:
        out = []
        for i in range(self.videos_list.count()):
            item = self.videos_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                out.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return out

    def _binarize_selection(self) -> str | None:
        checked = [self.binarize_list.item(i).text() for i in range(self.binarize_list.count())
                  if self.binarize_list.item(i).checkState() == QtCore.Qt.CheckState.Checked]
        all_names = [self.binarize_list.item(i).text() for i in range(self.binarize_list.count())]
        if not checked or checked == all_names or len(checked) != 1:
            return None
        return checked[0]

    def build_config(self) -> dict:
        pm = self.ctx.project_manager()
        iteration_dir = getattr(self.ctx, "pose_iteration_dir", lambda: "iteration-0")()
        return build_training_config(
            project_dir=str(pm.root) if pm else "",
            video_ids=self._selected_video_ids(),
            iteration_dir=iteration_dir,
            init_ratio=self.init_ratio.value(),
            max_iter=self.max_iter.value(),
            max_samples_iter=self.max_samples_iter.value(),
            conf_threshold=self.conf_threshold.value(),
            exclude_other=self.exclude_other.isChecked(),
            binarize=self._binarize_selection(),
            checkpoint_dir=str(pm.checkpoint_dir()) if pm else None,
            model_name=self.model_name.text() or None,
        )

    def _start(self) -> None:
        client = self._client()
        if not client:
            return
        if not self._selected_video_ids():
            self.ctx.status("Select at least one video to train on.")
            return
        self.log.appendPlainText("[train] starting…")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        def on_event(ev: dict):
            self.log.appendPlainText(str(ev))
            if "macro_f1" in ev or "ceiling" in ev:
                self.metrics.setText(
                    f"iteration {ev.get('iteration', '?')}  "
                    f"macro F1 {ev.get('macro_f1', '?')}  "
                    f"ceiling F1 {ev.get('ceiling', '?')}")

        def on_done():
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.ctx.status("Training stream ended.")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.training_status(self._task_id), on_event, on_done)

        self._spawn_call(lambda: client.start_training(self.build_config()), start)

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_training(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))
