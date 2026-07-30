"""Pose service: inspect pose files and turn pose + labels into training features.

This replaces the TSM app's `preprocess_service` (background subtraction →
animal-centred cropping → PNG frame dumps). The pose-based equivalent is far
cheaper: no video decoding, no images on disk, just
`pose file → likelihood filter → binned spatio-temporal features → .npz`.

Label alignment is the fiddly part and lives here (`dense_targets_from_csv`,
`build_class_names`) — see the module docstring of `asoid_core.indexing` for why
the frame ↔ feature-row mapping has to be exact.
"""

from __future__ import annotations

import csv
import os
import threading
import uuid
from typing import Optional

import numpy as np

from backend.asoid_core import features as feat
from backend.asoid_core import indexing as idx
from backend.asoid_core import pose_io

# ── Task registry ──────────────────────────────────────────────────────────
# Same shape as the other services: a module-level dict of task_id -> state,
# cleared between tests by the autouse fixture in conftest.

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancel"] = True


# ── Label handling ─────────────────────────────────────────────────────────

OTHER = "other"


def read_label_csv(path: str) -> tuple[list[str], dict[int, int]]:
    """Parse a one-hot label CSV into (behavior names, frame -> column index).

    Accepts what the labeling UI writes (`generateTrainingLabelCsv`):
    a `time` column holding the **native frame index**, then one 0/1 column per
    behavior. Rows exist only for frames inside a bout, so the mapping is sparse
    by design — every frame absent from it is unlabeled.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], {}
        names = [h.strip() for h in header[1:] if h.strip()]
        frame_to_col: dict[int, int] = {}
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                frame = int(float(row[0]))
            except ValueError:
                continue
            values = row[1:1 + len(names)]
            best, best_val = None, 0.0
            for i, v in enumerate(values):
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if fv > best_val:
                    best, best_val = i, fv
            if best is not None:
                frame_to_col[frame] = best
    return names, frame_to_col


def build_class_names(label_paths: list[str], keep: list[str] | None = None) -> list[str]:
    """Union of behaviors across every label file, with "other" forced last.

    A-SOID's label codes are positional and the loop below assumes the *last*
    code means "unlabeled", so the ordering here is load-bearing, not cosmetic.
    """
    seen: list[str] = []
    for path in label_paths:
        if not path or not os.path.exists(path):
            continue
        names, _ = read_label_csv(path)
        for n in names:
            if n != OTHER and n not in seen:
                seen.append(n)
    if keep:
        seen = [n for n in seen if n in keep]
    return seen + [OTHER]


def dense_targets_from_csv(path: str, n_frames: int, class_names: list[str]) -> np.ndarray:
    """Expand a sparse one-hot label CSV to one class code per native frame.

    Every frame with no bout becomes the last class ("other"), matching
    `load_preprocess.Preprocess._convert_labels`, which reindexes the label
    frame with "other" last and argmaxes each row.
    """
    other_code = len(class_names) - 1
    targets = np.full(n_frames, other_code, dtype=int)
    if not path or not os.path.exists(path):
        return targets

    names, frame_to_col = read_label_csv(path)
    # Map this file's column order onto the global class ordering.
    col_to_code = {
        i: class_names.index(name)
        for i, name in enumerate(names)
        if name in class_names
    }
    for frame, col in frame_to_col.items():
        if 0 <= frame < n_frames and col in col_to_code:
            targets[frame] = col_to_code[col]
    return targets


# ── Inspection ─────────────────────────────────────────────────────────────

def inspect(path: str, origin: str, multi_animal: bool = False) -> dict:
    """Summarize a pose file so the UI can populate its body-part picker."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pose file not found: {path}")
    return pose_io.inspect_pose(path, origin, multi_animal)


def keypoints(
    path: str,
    origin: str,
    bodyparts: list[str],
    animals: list[str] | None = None,
    multi_animal: bool = False,
    start: int = 0,
    end: int | None = None,
) -> dict:
    """Per-frame keypoints for the video overlay.

    Returned column-major (one flat array per body part per axis) rather than as
    per-frame objects: for a 50k-frame video with 8 body parts that is the
    difference between a ~3 MB payload and a ~30 MB one. Coordinates are rounded
    to 0.1 px — well below what is visible on a canvas.
    """
    names, arr = pose_io.keypoints_for_overlay(
        path, origin, bodyparts, animals=animals, multi_animal=multi_animal
    )
    if arr.size == 0:
        return {"bodyparts": [], "start": 0, "n_frames": 0, "points": []}

    end = arr.shape[0] if end is None else min(end, arr.shape[0])
    start = max(0, start)
    window = arr[start:end]

    points = []
    for i, name in enumerate(names):
        points.append({
            "name": name,
            "x": np.round(window[:, i, 0], 1).tolist(),
            "y": np.round(window[:, i, 1], 1).tolist(),
            "likelihood": np.round(window[:, i, 2], 2).tolist(),
        })
    return {
        "bodyparts": names,
        "start": start,
        "n_frames": int(window.shape[0]),
        "points": points,
    }


# ── Feature extraction ─────────────────────────────────────────────────────

def _features_path(project_dir: str, iteration_dir: str, video_id: str) -> str:
    return os.path.join(project_dir, iteration_dir, "features", f"{video_id}.npz")


def extract_one(
    pose_path: str,
    label_path: str | None,
    origin: str,
    bodyparts: list[str],
    class_names: list[str],
    frames2integ: int,
    llh_value: float = 0.6,
    animals: list[str] | None = None,
    multi_animal: bool = False,
    max_frames: int | None = None,
    limit_to_labeled: bool = False,
) -> dict:
    """Pose + labels for a single recording → aligned (features, targets)."""
    pose_array, is_3d, perc_rect = pose_io.load_and_filter(
        pose_path, origin, bodyparts,
        llh_value=llh_value, animals=animals, multi_animal=multi_animal,
    )
    n_frames_total = pose_array.shape[0]
    bin_width = idx.bin_width_from(frames2integ)

    # Bound extraction to what could actually have been reviewed.
    #
    # `max_frames` is the video's frame count. Pose past the end of the video was
    # never viewable, so it can't have been annotated — yet `dense_targets_from_csv`
    # will happily label all of it "other", and a pose file tracking a whole
    # session against a short review clip then contributes tens of thousands of
    # negatives nobody ever looked at. (This only skews training when "other" is
    # trained on; with "Exclude other" ticked those rows are dropped anyway. It
    # always costs extraction time and disk, though.)
    #
    # The video length is the right boundary, not the last label: footage that
    # was watched and deliberately left unlabeled IS legitimate "other", and
    # truncating at the final label would throw those real negatives away.
    # `limit_to_labeled` is the cruder fallback, off by default.
    n_frames = n_frames_total
    if max_frames and max_frames > 0:
        n_frames = min(n_frames, int(max_frames))
    elif limit_to_labeled and label_path and os.path.exists(label_path):
        _names, frame_to_col = read_label_csv(label_path)
        if frame_to_col:
            n_frames = min(n_frames, max(frame_to_col) + 1 + bin_width)

    pose_array = pose_array[:n_frames]

    features = feat.extract_features([pose_array], frames2integ, is_3d=is_3d)

    targets_frame = dense_targets_from_csv(label_path, n_frames, class_names)
    targets = idx.downsample_targets(targets_frame, bin_width, features.shape[0])

    # Feature extraction and label binning can disagree by a row at the tail
    # (the trailing partial bin); trim both to the shorter.
    n = min(features.shape[0], targets.shape[0])
    return {
        "features": features[:n],
        "targets": targets[:n],
        "is_3d": is_3d,
        "n_frames": n_frames,
        "n_frames_total": n_frames_total,
        "bin_width": bin_width,
        "rejected_fraction": [float(p) for p in perc_rect],
    }


def start_extraction(
    project_dir: str,
    videos: list[dict],
    origin: str,
    bodyparts: list[str],
    framerate: float,
    min_duration: float = 0.1,
    llh_value: float = 0.6,
    animals: list[str] | None = None,
    multi_animal: bool = False,
    iteration_dir: str = "iteration-0",
    keep_classes: list[str] | None = None,
) -> str:
    """Extract features for every video in the project. Returns a task_id.

    Args:
        videos: [{"id", "pose_path", "label_path"}] — pose is required,
            labels optional (a video with none contributes only "other").
    """
    task_id = str(uuid.uuid4())
    frames2integ = idx.frames2integ_from(framerate, min_duration)

    task = {
        "status": "starting",
        "progress": 0,
        "total": len(videos),
        "error": None,
        "cancel": False,
        "results": None,
    }
    _tasks[task_id] = task

    def _run():
        try:
            task["status"] = "running"
            os.makedirs(os.path.join(project_dir, iteration_dir, "features"), exist_ok=True)

            class_names = build_class_names(
                [v.get("label_path") for v in videos], keep=keep_classes
            )
            if len(class_names) <= 1:
                raise ValueError(
                    "No labeled behaviors found. Label some bouts before extracting features."
                )

            summary = []
            for i, video in enumerate(videos):
                if task["cancel"]:
                    task["status"] = "cancelled"
                    return
                out = extract_one(
                    pose_path=video["pose_path"],
                    label_path=video.get("label_path"),
                    origin=origin,
                    bodyparts=bodyparts,
                    class_names=class_names,
                    frames2integ=frames2integ,
                    llh_value=llh_value,
                    animals=animals,
                    multi_animal=multi_animal,
                    max_frames=video.get("max_frames"),
                )
                path = _features_path(project_dir, iteration_dir, video["id"])
                np.savez_compressed(
                    path,
                    features=out["features"],
                    targets=out["targets"],
                    frames2integ=frames2integ,
                    bin_width=out["bin_width"],
                    n_frames=out["n_frames"],
                    is_3d=out["is_3d"],
                    bodyparts=np.array(bodyparts, dtype=object),
                    class_names=np.array(class_names, dtype=object),
                )
                counts = np.bincount(out["targets"], minlength=len(class_names))
                total_rows = int(counts.sum()) or 1
                other_fraction = float(counts[-1]) / total_rows
                summary.append({
                    "id": video["id"],
                    "path": path,
                    "n_rows": int(out["features"].shape[0]),
                    "n_features": int(out["features"].shape[1]),
                    "n_frames": int(out["n_frames"]),
                    "n_frames_total": int(out["n_frames_total"]),
                    # Set when the pose file tracks past the last label, so the
                    # UI can point out that pose and video may not match.
                    "frames_skipped": int(out["n_frames_total"] - out["n_frames"]),
                    "other_fraction": other_fraction,
                    "is_3d": bool(out["is_3d"]),
                    "class_counts": {
                        name: int(counts[c]) for c, name in enumerate(class_names)
                    },
                    "mean_rejected_fraction": (
                        float(np.mean(out["rejected_fraction"]))
                        if out["rejected_fraction"] else 0.0
                    ),
                })
                task["progress"] = i + 1

            task["results"] = {
                "class_names": class_names,
                "frames2integ": frames2integ,
                "bin_width": idx.bin_width_from(frames2integ),
                "bodyparts": bodyparts,
                "videos": summary,
            }
            task["status"] = "completed"
        except Exception as e:
            task["error"] = f"{type(e).__name__}: {e}"
            task["status"] = "error"

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def load_features(project_dir: str, video_ids: list[str], iteration_dir: str = "iteration-0"):
    """Load and stack the extracted features for a set of videos.

    Returns:
        (features, targets, metadata) — metadata carries the class names,
        body parts and `frames2integ` that must be recorded on the checkpoint.
    """
    feats, targs = [], []
    meta = None
    for vid in video_ids:
        path = _features_path(project_dir, iteration_dir, vid)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No extracted features for {vid!r}. Run feature extraction first."
            )
        with np.load(path, allow_pickle=True) as data:
            feats.append(data["features"])
            targs.append(data["targets"])
            if meta is None:
                meta = {
                    "class_names": [str(x) for x in data["class_names"]],
                    "bodyparts": [str(x) for x in data["bodyparts"]],
                    "frames2integ": int(data["frames2integ"]),
                    "bin_width": int(data["bin_width"]),
                    "is_3d": bool(data["is_3d"]),
                }

    if not feats:
        raise ValueError("No videos selected for training.")

    widths = {f.shape[1] for f in feats}
    if len(widths) > 1:
        raise ValueError(
            f"Selected videos have inconsistent feature widths {sorted(widths)} — "
            "they were extracted with different body-part selections. Re-extract them together."
        )
    return np.vstack(feats), np.hstack(targs), meta
