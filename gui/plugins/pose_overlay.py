"""Pose overlay, painted over the video widget.

Qt port of the "pose-skeleton" overlay in LabelingApp/plugins/pipeline.js
(ensurePoseKeypoints, drawPoseOverlay). A thin transparent QWidget child of the
player's VideoView, so the shared shell needs no overlay hook.

Deliberately shows *unfiltered* keypoints — the point is to reveal where
tracking failed: confident points are cyan, points under the Pose tab's
likelihood threshold red. Every pair of confident points is joined, because
the feature extractor uses every pair — an honest picture of what the model
sees, rather than an anatomical skeleton we'd have to ask the user to define.

The backend returns keypoints column-major (one flat array per body part per
axis, see pose_service.keypoints) to keep a long recording's payload small.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from labeling_shell.backend_worker import CallWorker


def keypoints_at(payload: dict | None, frame: int) -> list[tuple]:
    """[(name, x, y, likelihood)] for one frame of a /api/pose/keypoints payload."""
    if not payload:
        return []
    i = frame - int(payload.get("start") or 0)
    out = []
    for p in payload.get("points") or []:
        xs, ys, ls = p.get("x") or [], p.get("y") or [], p.get("likelihood") or []
        if not (0 <= i < len(xs)):
            return []
        x, y = xs[i], ys[i]
        if x is None or y is None:
            continue
        out.append((p.get("name"), float(x), float(y),
                    float(ls[i]) if i < len(ls) and ls[i] is not None else 1.0))
    return out


class PoseOverlay(QtWidgets.QWidget):
    def __init__(self, ctx, video_view, parent=None):
        super().__init__(parent or video_view)
        self.ctx = ctx
        self.video_view = video_view
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.enabled = False
        self.setVisible(False)
        self._workers: set = set()
        self._payload: dict | None = None
        self._loaded_key = None
        self._pending_key = None
        self._current_frame = 0
        self.llh_cutoff = 0.6

    # ------------------------------------------------------------- lifecycle
    def set_enabled(self, on: bool) -> None:
        self.enabled = on
        self.setVisible(on)
        self.update()

    def invalidate(self) -> None:
        """The video, pose file or selection changed: drop cached keypoints."""
        self._payload = None
        self._loaded_key = None
        self.update()

    def ensure_loaded(self, bodyparts: list[str], settings: dict, quiet: bool = False) -> bool:
        """Fetch the current video's keypoints once, then repaint from cache.

        Returns False when there is nothing that could be drawn — and says why,
        unless `quiet` (a video switch with the overlay already on shouldn't nag
        about a video that simply has no pose file).
        """
        say = (lambda _m: None) if quiet else self.ctx.status
        self.llh_cutoff = float(settings.get("llh_value", 0.6))
        video_id = getattr(self.ctx, "current_video_id", None)
        pose_path = self.ctx.pose_path()
        client = self.ctx.client()
        if not pose_path or not bodyparts:
            say("Pose overlay: attach a pose file and select body parts first.")
            return False
        if client is None:
            say("Pose overlay: waiting for the backend.")
            return False
        # Only frames the video has can ever be drawn. A pose file tracking a
        # whole session against a short clip is common, and fetching all of it
        # (a 4-hour SLEAP file is ~90 MB of JSON) can outlast the client timeout.
        n_frames = getattr(self.ctx, "video_frame_count", lambda: None)()
        end = int(n_frames) if n_frames else None
        key = (video_id, pose_path, tuple(bodyparts), settings.get("origin"),
               tuple(settings.get("animals") or ()), bool(settings.get("multi_animal")), end)
        if key in (self._loaded_key, self._pending_key):
            return True
        self._pending_key = key
        say("Loading pose overlay…")

        def fetch():
            return client.pose_keypoints(
                pose_path, list(bodyparts), origin=settings.get("origin", "deeplabcut"),
                animals=settings.get("animals"),
                multi_animal=bool(settings.get("multi_animal")), end=end)

        def on_done(res: dict):
            if self._pending_key != key:
                return  # superseded (video switched / selection changed meanwhile)
            self._pending_key = None
            if getattr(self.ctx, "current_video_id", None) != video_id:
                return
            self._payload = res
            self._loaded_key = key
            say("Pose overlay ready.")
            self.update()

        def on_fail(msg: str):
            if self._pending_key == key:
                self._pending_key = None
            self.ctx.status(f"Pose overlay unavailable: {msg}")

        w = CallWorker(fetch)
        w.done.connect(on_done)
        w.failed.connect(on_fail)
        w.finished.connect(lambda: self._workers.discard(w))
        self._workers.add(w)
        w.start()
        return True

    # ------------------------------------------------------------------ draw
    def set_current_frame(self, index: int) -> None:
        self._current_frame = index
        if self.enabled:
            self.update()

    def paintEvent(self, event) -> None:
        if not self.enabled or not self._payload:
            return
        pts = keypoints_at(self._payload, self._current_frame)
        img = self.video_view.current_image
        if not pts or img is None or img.width() == 0 or img.height() == 0:
            return
        rect = self.video_view._fit_rect()
        sx, sy = rect.width() / img.width(), rect.height() / img.height()
        mapped = [(rect.x() + x * sx, rect.y() + y * sy, llh >= self.llh_cutoff)
                  for _n, x, y, llh in pts]

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor(120, 190, 255, 70), 1))
        for i in range(len(mapped)):
            for j in range(i + 1, len(mapped)):
                if mapped[i][2] and mapped[j][2]:
                    painter.drawLine(QtCore.QPointF(*mapped[i][:2]),
                                     QtCore.QPointF(*mapped[j][:2]))
        r = max(2.5, rect.width() / 250)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 1))
        good, bad = QtGui.QColor(80, 220, 255, 240), QtGui.QColor(255, 80, 80, 240)
        for x, y, ok in mapped:
            painter.setBrush(good if ok else bad)
            painter.drawEllipse(QtCore.QPointF(x, y), r, r)
