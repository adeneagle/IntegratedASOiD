"""Pose/RandomForest project bookkeeping — port of project-manager.js.

Extends the shell's `ProjectStore` (frame-exact bout JSON + one-hot training
CSV, already generic and reused as-is for the Label tab) with everything
project-manager.js added on top for this pipeline: pose-file attachment,
preprocessing/training config, models, and iteration metadata. There is no
Electron IPC layer here, so every `window.api.project*` call in the JS
version collapses into a direct filesystem read/write.

Not a byte-for-byte manifest port: `ProjectStore` already uses a simpler,
frame-exact bout format (JSON) instead of the JS bout CSV, the same
simplification RealtimeASOID's own Python port made. Only `sanitize_id` needs
to match the JS algorithm exactly — it's user-facing (drives label/feature
file names on disk) and independently reimplemented here, not inherited.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from PySide6 import QtCore

from labeling_shell.project import ProjectStore


def sanitize_id(filename: str) -> str:
    """Filesystem-safe video id — byte-identical to project-manager.js's sanitizeId.

    Strip the extension, replace anything outside [A-Za-z0-9_-] with "_",
    collapse runs of "_", then trim leading/trailing "_".
    """
    stem = re.sub(r"\.[^./\\]+$", "", str(filename))
    stem = re.sub(r"[^a-zA-Z0-9_-]", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem.strip("_")


class PoseProjectManager(ProjectStore):
    """A project directory, extended with the pose/RF pipeline's bookkeeping."""

    sanitize_id = staticmethod(sanitize_id)

    def __init__(self, root):
        super().__init__(root)
        self._save_timer: Optional[QtCore.QTimer] = None

    # -------------------------------------------------------------- manifest
    def create(self, name: str) -> dict:
        manifest = super().create(name)
        manifest.update({
            "version": 2,
            "current_iteration": 0,
            "preprocessing": {},
            "training": {},
            "models": [],
            "iterations": {},
        })
        self._write_manifest(manifest)
        return manifest

    def save_manifest(self) -> None:
        """Debounced write (500ms), mirrors project-manager.js's saveManifest."""
        if self._save_timer is None:
            self._save_timer = QtCore.QTimer()
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self.flush_manifest)
        self._save_timer.start(500)

    def flush_manifest(self) -> None:
        if self._save_timer is not None:
            self._save_timer.stop()
        # _manifest is mutated in place by the setters below, then flushed here.
        self._write_manifest(self.manifest)

    # ProjectStore's writers re-read project.json from disk, mutate, and write
    # back — which would clobber (and be clobbered by) the cached _manifest this
    # class keeps. Route them through the cache so there is a single writer.
    def register_video(self, video_id, name, path, frame_count=None) -> None:
        videos = self.manifest.setdefault("videos", [])
        record = {"id": video_id, "name": name, "path": str(path),
                  "frame_count": frame_count}
        existing = next((v for v in videos if v.get("id") == video_id), None)
        if existing:
            existing.update(record)
        else:
            videos.append(record)
        self.flush_manifest()

    def save_behaviors(self, behaviors) -> None:
        self.manifest["behaviors"] = [
            b if isinstance(b, dict) else vars(b) for b in behaviors
        ]
        self.flush_manifest()

    def _load(self) -> dict:
        self._manifest = self.load_manifest()
        return self._manifest

    @property
    def manifest(self) -> dict:
        if not hasattr(self, "_manifest"):
            self._load()
        return self._manifest

    # ---------------------------------------------------------------- videos
    def get_video_entry(self, video_id: str) -> Optional[dict]:
        return next((v for v in self.manifest.get("videos", []) if v.get("id") == video_id), None)

    def set_video_pose(self, video_id: str, pose_path: str) -> Optional[dict]:
        entry = self.get_video_entry(video_id)
        if entry is None:
            return None
        if entry.get("pose") != pose_path:
            entry["pose"] = pose_path
            self.save_manifest()
        return entry

    def get_video_pose(self, video_id: str) -> Optional[str]:
        entry = self.get_video_entry(video_id)
        return entry.get("pose") if entry else None

    def get_video_frame_count(self, video_id: str) -> Optional[int]:
        entry = self.get_video_entry(video_id)
        return entry.get("frame_count") if entry else None

    def list_videos(self) -> list:
        """Copies of the registered video records, in the order they were added."""
        return [dict(v) for v in self.manifest.get("videos", []) or []]

    def relink_video(self, video_id: str, path) -> None:
        """Point an existing entry at a new file, keeping its id (so its labels,
        pose file and features)."""
        entry = self.get_video_entry(video_id)
        if entry is not None:
            entry["path"] = str(path)
            self.flush_manifest()

    def relink_missing_in(self, folder) -> list:
        """Relink every video that can't be found to a same-named file in
        `folder` — one moved video usually means the whole batch moved.
        Returns the ids fixed."""
        fixed = []
        for v in self.list_videos():
            path = v.get("path") or ""
            if path and os.path.exists(path):
                continue
            name = re.split(r"[\\/]", path)[-1] if path else ""
            cand = os.path.join(str(folder), name)
            if name and os.path.isfile(cand):
                self.relink_video(v["id"], cand)
                fixed.append(v["id"])
        return fixed

    def remove_video(self, video_id: str, delete_data: bool = False) -> None:
        """Drop a video from the project. The video file itself is never touched.

        By default its labels and saved predictions stay on disk, so adding the
        same video back restores them; `delete_data` removes them too. Features
        under iteration-N/ are left alone — they're part of that iteration's
        record of what a model was trained on.
        """
        if not str(video_id or "").strip():
            raise ValueError("empty video id")
        m = self.manifest
        m["videos"] = [v for v in m.get("videos", []) if v.get("id") != video_id]
        if m.get("last_video") == video_id:
            m.pop("last_video")
        self.flush_manifest()
        if delete_data:
            for p in (self.labels_dir / f"{video_id}.json", self.label_csv_path(video_id),
                      self.predictions_dir() / f"{video_id}.json"):
                p.unlink(missing_ok=True)

    # -------------------------------------------------------------- behaviors
    def update_behaviors(self, behaviors) -> None:
        self.manifest["behaviors"] = [
            {"name": b.name, "key": b.key} if not isinstance(b, dict) else b
            for b in behaviors
        ]
        self.save_manifest()

    # -------------------------------------------------------------- pipeline
    def update_preprocessing_config(self, params: dict) -> None:
        self.manifest["preprocessing"] = dict(params)
        self.save_manifest()

    def update_training_config(self, params: dict) -> None:
        self.manifest["training"] = dict(params)
        self.save_manifest()

    def add_model(self, model_entry: dict) -> None:
        models = self.manifest.setdefault("models", [])
        existing = next((i for i, m in enumerate(models)
                          if m.get("path") and model_entry.get("path")
                          and m["path"] == model_entry["path"]), None)
        if existing is not None:
            models[existing] = model_entry
        else:
            models.append(model_entry)
        self.save_manifest()

    def get_models(self) -> list:
        return self.manifest.get("models", [])

    def update_model_path(self, old_path: str, new_path: str) -> None:
        for m in self.manifest.get("models", []):
            if m.get("path") == old_path:
                m["path"] = new_path
                self.save_manifest()
                return

    # ------------------------------------------------------------------ dirs
    def models_dir(self) -> Path:
        return self.root / "models"

    def checkpoint_dir(self) -> Path:
        return self.root / "models" / "checkpoints"

    def exports_dir(self) -> Path:
        return self.root / "models" / "exports"

    # ------------------------------------------------------------ iterations
    def get_current_iteration(self) -> int:
        n = self.manifest.get("current_iteration")
        return n if isinstance(n, int) and n >= 0 else 0

    def set_current_iteration(self, n: int) -> None:
        if not isinstance(n, int) or n < 0:
            return
        self.manifest["current_iteration"] = n
        self.save_manifest()

    def iteration_name(self, n: Optional[int] = None) -> str:
        return f"iteration-{n if n is not None else self.get_current_iteration()}"

    def get_iteration_models_dir(self, iter_name: Optional[str] = None) -> Path:
        return self.root / (iter_name or self.iteration_name()) / "models"

    def get_iteration_meta(self, iter_name: str) -> Optional[dict]:
        return self.manifest.get("iterations", {}).get(iter_name)

    def set_iteration_meta(self, iter_name: str, meta: dict) -> None:
        iterations = self.manifest.setdefault("iterations", {})
        iterations[iter_name] = {**iterations.get(iter_name, {}), **meta}
        self.save_manifest()

    def get_video_features_abs(self, video_id: str, iteration_dir: str = "iteration-0") -> str:
        return str(self.root / iteration_dir / "features" / f"{video_id}.npz")

    # ------------------------------------------------------------ label path
    def get_video_labels_abs(self, video_id: str) -> Optional[str]:
        p = self.label_csv_path(video_id)
        return str(p) if p.exists() else None

    # ------------------------------------------------------------ predictions
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    def save_predictions(self, video_id: str, data: dict) -> None:
        self.predictions_dir().mkdir(parents=True, exist_ok=True)
        (self.predictions_dir() / f"{video_id}.json").write_text(
            json.dumps(data), encoding="utf-8")

    def load_predictions(self, video_id: str) -> Optional[dict]:
        p = self.predictions_dir() / f"{video_id}.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    # -------------------------------------------------------------- .rlt I/O
    def export_rlt(self, client, output_path: str, package: bool = False) -> dict:
        self.flush_manifest()
        return client.export_rlt(str(self.root), output_path, package=package)

    @classmethod
    def import_rlt(cls, client, rlt_path: str, output_dir: str) -> "PoseProjectManager":
        client.import_rlt(rlt_path, output_dir)
        return cls(output_dir)

    @staticmethod
    def inspect_rlt(client, rlt_path: str) -> dict:
        return client.inspect_rlt(rlt_path)
