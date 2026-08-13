"""Pipeline tab container — Pose / Train / Predict / Active Learning.

Qt port of LabelingApp/plugins/pipeline.js's mountPanels/register wiring.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from .pose_panel import PoseTab
from .predict_panel import PredictTab
from .train_panel import TrainTab


class PipelinePanel(QtWidgets.QTabWidget):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        from .active_learning_panel import ActiveLearningPanel  # lazy: avoids import cycle
        self.ctx = ctx
        self.pose = PoseTab(ctx)
        self.train = TrainTab(ctx)
        self.predict = PredictTab(ctx)
        self.active_learning = ActiveLearningPanel(ctx)
        self.addTab(self.pose, "Pose")
        self.addTab(self.train, "Train")
        self.addTab(self.predict, "Predict")
        self.addTab(self.active_learning, "Active Learning")

    def refresh(self) -> None:
        self.pose.refresh()
        self.train.refresh()
        self.predict.refresh()
        self.active_learning.refresh()
