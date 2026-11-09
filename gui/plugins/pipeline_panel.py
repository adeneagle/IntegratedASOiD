"""Pipeline tab container — Pose / Train / Predict / Active Learning.

Qt port of LabelingApp/plugins/pipeline.js's mountPanels/register wiring. The
main window drives it through four lifecycle calls — the Qt equivalent of the
Electron plugin hooks:

    refresh()            labels/behaviors changed (cheap, no backend calls)
    on_video_switched()  a different video is active   (onVideoSwitched)
    on_project_opened()  a project was opened/created   (onProjectOpened)
    on_backend_ready()   the backend just came up
"""

from __future__ import annotations

from PySide6 import QtWidgets

from .active_learning_panel import ActiveLearningPanel
from .pose_panel import PoseTab
from .predict_panel import PredictTab
from .train_panel import TrainTab


class PipelinePanel(QtWidgets.QTabWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.pose = PoseTab(ctx)
        self.train = TrainTab(ctx, pose_tab=self.pose)
        self.predict = PredictTab(ctx, pose_tab=self.pose)
        self.active_learning = ActiveLearningPanel(ctx, pose_tab=self.pose)
        self.addTab(self.pose, "Pose")
        self.addTab(self.train, "Train")
        self.addTab(self.predict, "Predict")
        self.addTab(self.active_learning, "Active Learning")

        # Features land in (or the user moves to) another iteration: the Train
        # tab's "which videos have features" list is stale.
        self.pose.iterationChanged.connect(self.train.refresh)
        self.pose.selectionChanged.connect(self.active_learning._update_model_info)
        # A fresh model is what you want to review with next.
        self.train.modelTrained.connect(self._on_model_trained)
        self.currentChanged.connect(lambda _i: self.refresh())

    def _on_model_trained(self, path: str, entry: dict) -> None:
        for picker in (self.active_learning.model, self.predict.model):
            picker.add_path(path, entry)
            picker.reload(prefer=path)

    def refresh(self) -> None:
        self.pose.refresh()
        self.train.refresh()
        self.predict.refresh()
        self.active_learning.refresh()

    def on_video_switched(self) -> None:
        self.pose.on_video_switched()
        self.active_learning.on_video_switched()
        self.train.refresh()

    def on_project_opened(self) -> None:
        self.pose.on_project_opened()
        self.train.on_project_opened()
        self.predict.on_project_opened()
        self.active_learning.on_project_opened()

    def on_backend_ready(self) -> None:
        self.pose.retry_inspect()
        self.predict.model.reload()
        self.active_learning.model.reload()
