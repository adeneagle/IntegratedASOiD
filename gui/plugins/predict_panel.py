"""Predict tab — batch inference over unlabeled pose files.

Qt port of the "predict" tab in LabelingApp/plugins/pipeline.js. Writes one
`<pose>_predictions.csv` per file in the frame-level one-hot layout the
labeler imports, so a batch result can be loaded back in as labels for review.

Pose files are read with the Pose tab's settings (software, likelihood
threshold, individuals) and the model's own body-part selection, which is the
only one guaranteed to match its feature space.
"""

from __future__ import annotations

import os

from PySide6 import QtWidgets

from labeling_shell.theme import mark_accent, mark_ghost

from .common import ModelPicker, _Base
from .pipeline_config import POSE_FILE_FILTER, build_batch_request


class PredictTab(_Base):
    def __init__(self, ctx, pose_tab=None, parent=None):
        super().__init__(ctx, parent)
        self.pose_tab = pose_tab
        self._task_id = None

        outer = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.model = ModelPicker(ctx)
        form.addRow("Model", self.model)

        self.out_dir_edit = QtWidgets.QLineEdit()
        self.out_dir_edit.setPlaceholderText("default: <project>/predictions")
        out_btn = QtWidgets.QPushButton("Browse…")
        out_btn.clicked.connect(self._pick_out_dir)
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(self.out_dir_edit, 1)
        orow.addWidget(out_btn)
        form.addRow("Output folder", self._wrap(orow))
        outer.addLayout(form)
        outer.addWidget(self._hint(
            "Pose files are read with the Pose tab's settings (software, likelihood "
            "threshold, individuals) and the body parts the model was trained on."))

        outer.addWidget(QtWidgets.QLabel("Pose files:"))
        self.files_list = QtWidgets.QListWidget()
        self.files_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        outer.addWidget(self.files_list, 1)
        file_btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add…")
        add_btn.clicked.connect(self._add_files)
        remove_btn = QtWidgets.QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.files_list.clear)
        for b in (remove_btn, clear_btn):
            mark_ghost(b)
        file_btns.addWidget(add_btn)
        file_btns.addWidget(remove_btn)
        file_btns.addWidget(clear_btn)
        file_btns.addStretch(1)
        outer.addLayout(file_btns)

        run_btns = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run batch prediction")
        mark_accent(self.run_btn)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn.clicked.connect(self._stop)
        run_btns.addWidget(self.run_btn, 1)
        run_btns.addWidget(self.stop_btn)
        outer.addLayout(run_btns)

        self.progress = QtWidgets.QProgressBar()
        outer.addWidget(self.progress)
        self.log = self._log_box()
        outer.addWidget(self.log)

    def refresh(self) -> None:
        pass

    def on_project_opened(self) -> None:
        self.model.set_entries([])
        self.model.reload()

    def _pick_out_dir(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output folder",
                                                       self._default_out_dir() or "")
        if d:
            self.out_dir_edit.setText(d)

    def _default_out_dir(self):
        pm = self.ctx.project_manager()
        return str(pm.predictions_dir()) if pm else None

    def pose_files(self) -> list[str]:
        return [self.files_list.item(i).text() for i in range(self.files_list.count())]

    def _add_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Pose files", "", POSE_FILE_FILTER)
        have = set(self.pose_files())
        for p in paths:
            if p not in have:
                self.files_list.addItem(p)

    def _remove_selected(self) -> None:
        for item in self.files_list.selectedItems():
            self.files_list.takeItem(self.files_list.row(item))

    def _settings(self) -> dict:
        if self.pose_tab is not None:
            return self.pose_tab.pose_settings()
        return {"origin": "deeplabcut", "multi_animal": False, "animals": None,
                "llh_value": 0.6}

    def _run(self) -> None:
        title = "Can't run prediction"
        ckpt = self.model.current_path()
        if not ckpt:
            self._warn(title, "Choose a model first (train one, or Browse…).")
            return
        pose_paths = self.pose_files()
        if not pose_paths:
            self._warn(title, "Add at least one pose file.")
            return
        out_dir = self.out_dir_edit.text().strip() or self._default_out_dir()
        if not out_dir:
            self._warn(title, "Choose an output folder (or open a project).")
            return
        client = self._client()
        if client is None:
            return

        s = self._settings()
        req = build_batch_request(
            checkpoint_path=ckpt, pose_paths=pose_paths, output_dir=out_dir,
            origin=s["origin"], llh_value=s["llh_value"], animals=s["animals"],
            multi_animal=s["multi_animal"],
        )
        self.log.clear()
        self.log.appendPlainText(f"Predicting {len(pose_paths)} file(s) with "
                                 f"{os.path.basename(ckpt)}…")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setMaximum(len(pose_paths))
        self.progress.setValue(0)
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

        def on_done():
            finish()
            if outcome["status"] == "completed":
                self._spawn_call(lambda: client.inference_results(self._task_id),
                                 self._on_results)
            elif outcome["status"] == "cancelled":
                self.log.appendPlainText("Cancelled.")
            else:
                msg = outcome["error"] or f"Prediction ended with status {outcome['status']!r}."
                self.log.appendPlainText(f"ERROR: {msg}")
                QtWidgets.QMessageBox.critical(self, "Prediction failed", msg)

        def on_fail(msg: str):
            finish()
            self.log.appendPlainText(f"ERROR: {msg}")

        def start(res: dict):
            self._task_id = res.get("task_id")
            if not self._task_id:
                on_fail(f"Backend did not return a task id: {res}")
                return
            self._spawn_stream(lambda: client.inference_status(self._task_id),
                               on_event, on_done, on_fail)

        self._spawn_call(lambda: client.batch_inference(req), start, on_fail)

    def _on_results(self, res: dict) -> None:
        for f in res.get("files", []):
            self.log.appendPlainText(
                f"{os.path.basename(f.get('output_path', ''))}  ({f.get('n_frames')} frames)")
            for name, s in (f.get("summary") or {}).items():
                self.log.appendPlainText(
                    f"    {name}: {s.get('frames')} frames, mean uncertainty "
                    f"{s.get('mean_uncertainty')}")
        self.ctx.status(f"Predictions written to {res.get('output_dir')} — File ▸ Import "
                        f"Labels loads one back for review.")

    def _stop(self) -> None:
        client = self._client()
        if client and self._task_id:
            self._spawn_call(lambda: client.stop_inference(self._task_id),
                             lambda _res: self.ctx.status("Stop requested."))

