"""Pose-skeleton overlay, painted over the video widget.

Qt port of the "pose-skeleton" overlay in LabelingApp/plugins/pipeline.js
(~lines 769-984: canvas sizing, ensurePoseKeypoints, drawPoseOverlay). The
JS shell exposes a `videoOverlaySlot` canvas positioned above the video;
`labeling_shell.video_widget` deliberately has no such hook yet (see its
module docstring), so this is a thin transparent QWidget child of the
player's VideoView, matching the JS approach without touching shared shell
code.

Points only (no skeleton connectivity/edges — the pose backend's
`inspect`/`keypoints` endpoints return per-bodypart tracks, not a rig
graph), one dot per bodypart, colored by low-confidence-filtered position.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class PoseOverlay(QtWidgets.QWidget):
    def __init__(self, ctx, video_view, parent=None):
        super().__init__(parent or video_view)
        self.ctx = ctx
        self.video_view = video_view
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.enabled = False
        self._workers: set = set()
        self._video_id = None
        self._frames = None       # dict[int, list[[x, y]]] keyed by frame index
        self._n_frames = 0

    # ------------------------------------------------------------- lifecycle
    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self.setVisible(on)
        self.update()

    def reset_for_video(self) -> None:
        self._video_id = None
        self._frames = None
        self.update()

    def ensure_loaded(self, bodyparts: list[str]) -> None:
        """Fetch this video's keypoints once, then cache for repaint-on-seek."""
        if not self.enabled or not bodyparts:
            return
        video_id = getattr(self.ctx, "current_video_id", None)
        if video_id is None or video_id == self._video_id:
            return
        client = self.ctx.client()
        pose_path = self.ctx.pose_path()
        if client is None or not pose_path:
            return

        from labeling_shell.backend_worker import CallWorker

        def fetch():
            return client.pose_keypoints(pose_path, bodyparts)

        def on_done(res: dict):
            self._video_id = video_id
            coords = res.get("keypoints") or res.get("frames") or []
            self._frames = {i: pts for i, pts in enumerate(coords)}
            self._n_frames = len(coords)
            self.update()

        w = CallWorker(fetch)
        w.done.connect(on_done)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()

    # ------------------------------------------------------------------ draw
    def set_current_frame(self, index: int) -> None:
        self._current_frame = index
        self.update()

    def paintEvent(self, event) -> None:
        if not self.enabled or not self._frames:
            return
        pts = self._frames.get(getattr(self, "_current_frame", 0))
        if not pts:
            return
        rect = self.video_view._fit_rect()
        img = self.video_view.current_image
        if img is None or img.width() == 0 or img.height() == 0:
            return
        sx, sy = rect.width() / img.width(), rect.height() / img.height()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtGui.QColor(0, 220, 255, 220))
        painter.setPen(QtGui.QColor(0, 0, 0, 160))
        r = 4
        for pt in pts:
            if pt is None or len(pt) < 2 or pt[0] is None or pt[1] is None:
                continue
            x = rect.x() + pt[0] * sx
            y = rect.y() + pt[1] * sy
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)
