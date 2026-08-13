"""Predict tab — batch inference over unlabeled pose files.

Qt port of the "predict" tab in LabelingApp/plugins/pipeline.js (~lines 1196-1362).
"""

from __future__ import annotations

from PySide6 import QtWidgets

from labeling_shell.no_wheel import NoWheelComboBox
from labeling_shell.theme import mark_accent

from .pipeline_config import build_batch_request
from .pose_panel import ORIGINS, _Base


class PredictTab(_Base):
    def __init__(self, ctx, parent=None):
        super().__init__(ctx, parent)
        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()

        self.checkpoint_edit = QtWidgets.QLineEdit()
        ckpt_btn = QtWidgets.QPushButton("Browse…")
        ckpt_btn.clicked.connect(self._pick_checkpoint)
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(self.checkpoint_edit, 1)
        crow.addWidget(ckpt_btn)
        form.addRow("Model", crow)

        self.origin = NoWheelComboBox()
        self.origin.addItems(ORIGINS)
        form.addRow("Pose software", self.origin)

        self.out_dir_edit = QtWidgets.QLineEdit()
        out_btn = QtWidgets.QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_out_dir)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.out_dir_edit, 1)
        orow.addWidget(out_btn)
        form.addRow("Output folder", orow)
        layout.addLayout(form)

        layout.addWidget(QtWidgets.QLabel("Pose files:"))
        self.files_list = QtWidgets.QListWidget()
        layout.addWidget(self.files_list, 1)
        file_btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add…")
        add_btn.clicked.connect(self._add_files)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.files_list.clear)
        file_btns.addWidget(add_btn)
        file_btns.addWidget(clear_btn)
        layout.addLayout(file_btns)

        run_btns = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run batch prediction")
        mark_accent(self.run_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)
        run_btns.addWidget(self.run_btn)
        run_btns.addWidget(self.stop_btn)
        layout.addLayout(run_btns)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self._task_id = None

    def refresh(self) -> None:
        pm = self.ctx.project_manager()
        if pm is not None and not self.checkpoint_edit.text():
            models = pm.get_models()
            if models:
                self.checkpoint_edit.setText(models[-1].get("path", ""))

    def _pick_checkpoint(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Model", "", "Model bundle (*.joblib);;All files (*)")
        if path:
            self.checkpoint_edit.setText(path)

    def _pick_out_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder")
        if d:
            self.out_dir_edit.setText(d)

    def _add_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Pose files", "", "Pose files (*.h5 *.csv *.json);;All files (*)")
        for p in paths:
            self.files_list.addItem(p)

    def _run(self) -> None:
        client = self._client()
        if not client:
            return
        pose_paths = [self.files_list.item(i).text() for i in range(self.files_list.count())]
        if not pose_paths or not self.checkpoint_edit.text() or not self.out_dir_edit.text():
            self.ctx.status("Need a model, output folder, and at least one pose file.")
            return

        req = build_batch_request(
            checkpoint_path=self.checkpoint_edit.text(),
            pose_paths=pose_paths,
            output_dir=self.out_dir_edit.text(),
            origin=self.origin.currentText(),
        )
        self.log.appendPlainText(f"[predict] starting on {len(pose_paths)} file(s)…")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)

        def on_event(ev: dict):
            total, prog = ev.get("total") or 0, ev.get("progress") or 0
            if total:
                self.progress.setMaximum(total)
                self.progress.setValue(prog)
            if ev.get("status") in ("completed", "error", "cancelled"):
                self.log.appendPlainText(f"[predict] {ev.get('status')}")

        def on_done():
            self.run_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._spawn_call(lambda: client.inference_results(self._task_id), self._on_results)

        def start(res: dict):
            self._task_id = res.get("task_id")
            if self._task_id:
                self._spawn_stream(
                    lambda: client.inference_status(self._task_id), on_event, on_done)

        self._spawn_call(lambda: client.batch_inference(req), start)

    def _on_results(self, res: dict) -> None:
        for f in res.get("files", []):
            self.log.appendPlainText(
                f"  {f.get('output_path')} — {f.get('n_frames')} frames")
        self.ctx.status(f"Predictions written to {res.get('output_dir')}")

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_inference(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))
