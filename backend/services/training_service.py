"""Training service: RandomForest + active learning with live metrics streaming.

Keeps the exact task/metrics contract of the TSM predecessor — a module-level
task registry, a JSON-lines metrics file, `read_all_metrics` / `read_latest_metrics`,
cooperative cancellation — so `routers/training.py`'s SSE endpoint and the
renderer's `#trainLog` need no changes.

What did change: the TSM version ran training in a **subprocess** to isolate CUDA
memory and allow a hard `terminate()`. A RandomForest fit takes seconds on CPU,
so this runs in a thread and cancels cooperatively via a flag the active-learning
loop checks between iterations. That removes the temp-dir config/script
marshalling entirely.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from backend.asoid_core.active_learning import RFActiveLearner, fingerprint_bodyparts
from backend.services.pose_service import load_features

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)

# ── Task registry ──────────────────────────────────────────────────────────

_tasks: dict[str, dict] = {}


def get_task(task_id: str) -> Optional[dict]:
    return _tasks.get(task_id)


def cancel_all():
    for task in _tasks.values():
        task["cancel"] = True


class _Cancelled(Exception):
    """Raised inside the training thread when the user hits Stop."""


# ── Training ───────────────────────────────────────────────────────────────

def start_training(config: dict) -> str:
    """Run the active-learning loop in a background thread. Returns task_id.

    Expected config keys:
        project_dir, video_ids, iteration_dir
        init_ratio, max_iter, max_samples_iter, conf_threshold, exclude_other
        checkpoint_dir, model_name
        binarize / label_union  (optional, see `_collapse_labels`)
    """
    task_id = str(uuid.uuid4())

    checkpoint_dir = config.get("checkpoint_dir") or os.path.join(
        config.get("project_dir", tempfile.mkdtemp(prefix="asoid_train_")),
        "models", "checkpoints",
    )
    tmp_dir = tempfile.mkdtemp(prefix="asoid_train_")
    metrics_path = os.path.join(tmp_dir, "metrics.jsonl")
    open(metrics_path, "w", encoding="utf-8").close()

    task = {
        "status": "starting",
        "metrics_path": metrics_path,
        "tmp_dir": tmp_dir,
        "latest_metrics": {},
        "error": None,
        "cancel": False,
        "checkpoint_path": None,
    }
    _tasks[task_id] = task

    def emit(payload: dict):
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _run():
        try:
            task["status"] = "running"

            emit({"status": "starting", "message": "Loading extracted features..."})
            features, targets, meta = load_features(
                config["project_dir"],
                config["video_ids"],
                config.get("iteration_dir", "iteration-0"),
            )
            class_names = meta["class_names"]

            emit({
                "status": "starting",
                "message": (
                    f"{features.shape[0]} samples x {features.shape[1]} features "
                    f"from {len(config['video_ids'])} video(s)"
                ),
            })
            emit({
                "status": "starting",
                "message": f"Body parts ({len(meta['bodyparts'])}): {', '.join(meta['bodyparts'])}",
            })

            binarize = config.get("binarize")
            label_union = config.get("label_union")
            if binarize or label_union:
                targets, class_names = _collapse_labels(
                    targets, class_names, binarize, label_union
                )
                emit({
                    "status": "starting",
                    "message": f"Label space collapsed to: {', '.join(class_names)}",
                })

            counts = np.bincount(targets, minlength=len(class_names))
            emit({
                "status": "starting",
                "message": "Class distribution: " + ", ".join(
                    f"{n}={int(c)}" for n, c in zip(class_names, counts)
                ),
            })

            learner = RFActiveLearner(
                features=features,
                targets=targets,
                class_names=class_names,
                init_ratio=config.get("init_ratio", 0.01),
                max_iter=config.get("max_iter", 100),
                max_samples_iter=config.get("max_samples_iter", 20),
                conf_threshold=config.get("conf_threshold", 0.5),
                exclude_other=config.get("exclude_other", False),
            )

            def progress_cb(payload):
                if task["cancel"]:
                    raise _Cancelled()
                out = dict(payload)
                out.setdefault("status", "training")
                emit(out)
                task["latest_metrics"] = out

            result = learner.run(progress_cb=progress_cb)

            os.makedirs(checkpoint_dir, exist_ok=True)
            best_macro = result.iterations[-1].macro_f1 if result.iterations else 0.0
            name = config.get("model_name") or "rf_model"
            # Self-describing filename, matching the convention the TSM app used.
            filename = f"{name}_f1{best_macro:.3f}_it{len(result.iterations)}.joblib"
            checkpoint_path = os.path.join(checkpoint_dir, filename)

            joblib.dump({
                "model": result.model,
                "class_names": result.class_names,
                "bodyparts": meta["bodyparts"],
                "bodypart_fingerprint": fingerprint_bodyparts(
                    meta["bodyparts"], meta["is_3d"]
                ),
                "frames2integ": meta["frames2integ"],
                "bin_width": meta["bin_width"],
                "is_3d": meta["is_3d"],
                "n_features": int(features.shape[1]),
                "macro_f1": best_macro,
                "ceiling_macro_f1": result.full_macro_f1,
                "config": {
                    k: config.get(k)
                    for k in ("init_ratio", "max_iter", "max_samples_iter",
                              "conf_threshold", "exclude_other", "binarize", "label_union")
                },
            }, checkpoint_path)

            task["checkpoint_path"] = checkpoint_path
            final = {
                "status": "completed",
                "checkpoint_path": checkpoint_path,
                "macro_f1": best_macro,
                "ceiling_macro_f1": result.full_macro_f1,
                "iterations": len(result.iterations),
                "class_names": result.class_names,
                "message": f"Saved {filename}",
            }
            emit(final)
            task["latest_metrics"] = final
            task["status"] = "completed"

        except _Cancelled:
            task["status"] = "cancelled"
            emit({"status": "cancelled", "message": "Training stopped by user."})
        except Exception as e:
            detail = f"{type(e).__name__}: {e}"
            # error before status: the SSE poller breaks as soon as it sees
            # status=="error", so the detail has to already be there.
            task["error"] = detail[-2000:]
            task["status"] = "error"
            emit({"status": "error", "error": task["error"]})

    threading.Thread(target=_run, daemon=True).start()
    return task_id


def _collapse_labels(targets, class_names, binarize, label_union):
    """Collapse the label space to ['other', <positive>].

    Mirrors the TSM app's binarize/label_union semantics so the renderer's
    existing behavior checklist keeps working, and so `mapLabelToModelClass` in
    the Active Learning panel still describes what the model predicts:
    zero behaviors checked = multi-class, one = binarize that behavior,
    two or more = union them into a single positive class first.
    """
    positives = list(label_union or [])
    if binarize and binarize not in positives:
        positives.append(binarize)
    if not positives:
        return targets, class_names

    positive_name = binarize if len(positives) == 1 else "positive"
    codes = {class_names.index(p) for p in positives if p in class_names}
    if not codes:
        return targets, class_names

    collapsed = np.where(np.isin(targets, list(codes)), 1, 0)
    return collapsed.astype(int), ["other", positive_name]


# ── Metrics / lifecycle (unchanged contract) ───────────────────────────────

def read_all_metrics(task_id: str) -> list[dict]:
    """Read every JSON-lines event written so far (for ordered SSE streaming)."""
    task = get_task(task_id)
    if not task:
        return []
    metrics_path = task.get("metrics_path")
    if not metrics_path or not os.path.exists(metrics_path):
        return []
    out = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def read_latest_metrics(task_id: str) -> dict:
    """Read the latest metrics line, with terminal status/error attached."""
    task = get_task(task_id)
    if not task:
        return {}

    metrics_path = task.get("metrics_path")
    if not metrics_path or not os.path.exists(metrics_path):
        return {"status": task["status"], "error": task.get("error")}

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if lines:
            latest = json.loads(lines[-1])
            latest["status"] = task["status"]
            if task["status"] == "error" and task.get("error"):
                latest["error"] = task["error"]
            return latest
    except Exception:
        pass

    return {"status": task["status"], "error": task.get("error")}


def stop_training(task_id: str) -> bool:
    """Ask the training thread to stop at the next iteration boundary."""
    task = get_task(task_id)
    if not task:
        return False
    if task["status"] in ("completed", "error", "cancelled"):
        return False
    task["cancel"] = True
    return True


def list_checkpoints(search_dirs: list[str] | None = None) -> list[dict]:
    """Scan directories for saved models. Returns [{path, filename, size_mb, ...}]."""
    if search_dirs is None:
        search_dirs = [PROJECT_ROOT]

    results = []
    for base_dir in search_dirs:
        if not base_dir or not os.path.isdir(base_dir):
            continue
        for root, _dirs, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".joblib"):
                    continue
                full_path = os.path.join(root, f)
                entry = {
                    "path": full_path,
                    "filename": f,
                    "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 1),
                }
                # Surface the metadata the UI needs to warn about a mismatched
                # body-part selection without the user opening the file.
                try:
                    bundle = joblib.load(full_path)
                    entry.update({
                        "class_names": bundle.get("class_names"),
                        "bodyparts": bundle.get("bodyparts"),
                        "macro_f1": bundle.get("macro_f1"),
                        "frames2integ": bundle.get("frames2integ"),
                    })
                except Exception:
                    pass
                results.append(entry)
    return results
