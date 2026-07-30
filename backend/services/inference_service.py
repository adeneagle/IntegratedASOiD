"""Inference service: per-frame predictions + uncertainty from a pose file.

The response shape is **deliberately identical** to the TSM predecessor's:

    {"frames": [{"index", "predicted_class", "uncertainty"}, ...],
     "class_names": [...]}

That is what lets the entire Active Learning panel in `renderer.js` — the
uncertainty heatmap, misclassification ticks, "jump to lowest confidence",
`applyAlResults`, saved-prediction round-tripping — carry over untouched.

Uncertainty is the normalized Shannon entropy of the RandomForest's class
probabilities, the same 0..1 metric the softmax version produced.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Optional

import joblib
import numpy as np

from backend.asoid_core import pose_io
from backend.asoid_core.active_learning import fingerprint_bodyparts
from backend.asoid_core.predict import frameshift_predict_proba, normalized_entropy

# ── Task registry ──────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancel"] = True


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> dict:
    """Load a .joblib checkpoint bundle written by the training service."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    bundle = joblib.load(checkpoint_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            f"{os.path.basename(checkpoint_path)} is not an A-SOID Desktop checkpoint."
        )
    return bundle


def check_bodyparts(bundle: dict, bodyparts: list[str], is_3d: bool) -> None:
    """Reject a body-part selection the model was not trained on.

    Feature columns are built from every *pair* of selected body parts, so a
    different selection shifts the meaning of every column. When the widths
    happen to coincide this produces confident nonsense rather than an error —
    the failure upstream A-SOID patched in commit 1710d15. Fail loudly instead.
    """
    expected = bundle.get("bodypart_fingerprint")
    if not expected:
        return  # older checkpoint without a fingerprint; nothing to check against
    actual = fingerprint_bodyparts(bodyparts, is_3d)
    if actual != expected:
        raise ValueError(
            "This model was trained on a different body-part selection "
            f"({', '.join(bundle.get('bodyparts') or [])}) than the one configured "
            f"({', '.join(bodyparts)}). Re-extract features with the model's "
            "selection, or train a new model."
        )


# ── Inference ──────────────────────────────────────────────────────────────

def _wanted_set(frame_ranges):
    """Inclusive [start, end] native-index pairs → a set of frame indices.

    None means "the whole video" (kept distinct from an empty set, so an empty
    range list can still mean "nothing").
    """
    if not frame_ranges:
        return None
    wanted = set()
    for pair in frame_ranges:
        if not pair or len(pair) < 2:
            continue
        start, end = int(pair[0]), int(pair[1])
        if end < start:
            start, end = end, start
        wanted.update(range(start, end + 1))
    return wanted


def run_inference_pose(
    checkpoint_path: str,
    pose_path: str,
    origin: str,
    bodyparts: list[str] | None = None,
    llh_value: float = 0.6,
    animals: list[str] | None = None,
    multi_animal: bool = False,
    frame_ranges: list | None = None,
    progress_cb=None,
    should_cancel=None,
) -> dict:
    """Predict a behavior + uncertainty for every native frame of a pose file."""
    bundle = load_model(checkpoint_path)
    model = bundle["model"]
    class_names = list(bundle.get("class_names") or [])
    frames2integ = int(bundle.get("frames2integ", 30))
    # Default to the selection recorded on the checkpoint — the only one
    # guaranteed to line up with its feature space.
    bodyparts = bodyparts or list(bundle.get("bodyparts") or [])
    if not bodyparts:
        raise ValueError("No body parts specified and the checkpoint records none.")

    pose_array, is_3d, _perc = pose_io.load_and_filter(
        pose_path, origin, bodyparts,
        llh_value=llh_value, animals=animals, multi_animal=multi_animal,
    )
    check_bodyparts(bundle, bodyparts, is_3d)

    n_frames = pose_array.shape[0]

    def _progress(done, total):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("cancelled")
        if progress_cb is not None:
            progress_cb(done, total)

    labels, proba = frameshift_predict_proba(
        pose_array, model, frames2integ, is_3d=is_3d, progress_cb=_progress
    )

    if proba.shape[1] != len(class_names):
        # Probability columns follow model.classes_, which omits classes absent
        # from the training set; pad back out so predicted_class indexes into
        # class_names the way the renderer assumes.
        full = np.zeros((proba.shape[0], len(class_names)))
        for col, cls in enumerate(model.classes_):
            if 0 <= int(cls) < len(class_names):
                full[:, int(cls)] = proba[:, col]
        proba = full

    uncertainty = normalized_entropy(proba)

    wanted = _wanted_set(frame_ranges)
    frames = []
    for i in range(n_frames):
        if wanted is not None and i not in wanted:
            continue
        frames.append({
            "index": i,
            "predicted_class": int(labels[i]),
            "uncertainty": float(uncertainty[i]),
        })

    return {
        "frames": frames,
        "class_names": class_names,
        "checkpoint_path": checkpoint_path,
        "bin_width": int(bundle.get("bin_width", max(1, frames2integ // 10))),
    }


def start_inference(
    checkpoint_path: str,
    pose_path: str,
    origin: str = "deeplabcut",
    bodyparts: list[str] | None = None,
    llh_value: float = 0.6,
    animals: list[str] | None = None,
    multi_animal: bool = False,
    frame_ranges: list | None = None,
) -> str:
    """Start inference in a background thread. Returns task_id."""
    task_id = str(uuid.uuid4())
    task = {
        "status": "starting",
        "progress": 0,
        "total": 0,
        "error": None,
        "cancel": False,
        "results": None,
    }
    _tasks[task_id] = task

    def _run():
        try:
            task["status"] = "running"

            def progress_cb(done, total):
                task["progress"] = done
                task["total"] = total

            results = run_inference_pose(
                checkpoint_path=checkpoint_path,
                pose_path=pose_path,
                origin=origin,
                bodyparts=bodyparts,
                llh_value=llh_value,
                animals=animals,
                multi_animal=multi_animal,
                frame_ranges=frame_ranges,
                progress_cb=progress_cb,
                should_cancel=lambda: task["cancel"],
            )
            task["results"] = results
            task["progress"] = len(results["frames"])
            task["total"] = len(results["frames"])
            task["status"] = "completed"
        except InterruptedError:
            task["status"] = "cancelled"
        except Exception as e:
            task["error"] = f"{type(e).__name__}: {e}"
            task["status"] = "error"

    threading.Thread(target=_run, daemon=True).start()
    return task_id


# ── Batch prediction (ported from A-SOID's F_predict) ──────────────────────

def start_batch_inference(
    checkpoint_path: str,
    pose_paths: list[str],
    output_dir: str,
    origin: str = "deeplabcut",
    bodyparts: list[str] | None = None,
    llh_value: float = 0.6,
    animals: list[str] | None = None,
    multi_animal: bool = False,
) -> str:
    """Apply a trained model to a set of unlabeled pose files.

    Writes one `<name>_predictions.csv` per input into `output_dir`, in the same
    `time,<behavior>…` one-hot layout the labeling UI reads — so a batch result
    can be imported straight back in as labels for review.
    """
    task_id = str(uuid.uuid4())
    task = {
        "status": "starting",
        "progress": 0,
        "total": len(pose_paths),
        "error": None,
        "cancel": False,
        "results": None,
    }
    _tasks[task_id] = task

    def _run():
        try:
            task["status"] = "running"
            os.makedirs(output_dir, exist_ok=True)
            written = []

            for i, pose_path in enumerate(pose_paths):
                if task["cancel"]:
                    task["status"] = "cancelled"
                    return
                results = run_inference_pose(
                    checkpoint_path=checkpoint_path,
                    pose_path=pose_path,
                    origin=origin,
                    bodyparts=bodyparts,
                    llh_value=llh_value,
                    animals=animals,
                    multi_animal=multi_animal,
                    should_cancel=lambda: task["cancel"],
                )
                stem = os.path.splitext(os.path.basename(pose_path))[0]
                out_path = os.path.join(output_dir, f"{stem}_predictions.csv")
                _write_prediction_csv(out_path, results)
                written.append({
                    "pose_path": pose_path,
                    "output_path": out_path,
                    "n_frames": len(results["frames"]),
                    "summary": _ethogram_summary(results),
                })
                task["progress"] = i + 1

            task["results"] = {"files": written, "output_dir": output_dir}
            task["status"] = "completed"
        except Exception as e:
            task["error"] = f"{type(e).__name__}: {e}"
            task["status"] = "error"

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def _write_prediction_csv(path: str, results: dict) -> None:
    class_names = results["class_names"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("time," + ",".join(class_names) + "\n")
        for frame in results["frames"]:
            row = ["0"] * len(class_names)
            cls = frame["predicted_class"]
            if 0 <= cls < len(class_names):
                row[cls] = "1"
            f.write(f"{frame['index']}," + ",".join(row) + "\n")


def _ethogram_summary(results: dict) -> dict:
    """Frame counts and mean uncertainty per predicted class."""
    class_names = results["class_names"]
    counts = {name: 0 for name in class_names}
    unc_sum = {name: 0.0 for name in class_names}
    for frame in results["frames"]:
        cls = frame["predicted_class"]
        if 0 <= cls < len(class_names):
            name = class_names[cls]
            counts[name] += 1
            unc_sum[name] += frame["uncertainty"]
    return {
        name: {
            "frames": counts[name],
            "mean_uncertainty": round(unc_sum[name] / counts[name], 4) if counts[name] else 0.0,
        }
        for name in class_names
    }
