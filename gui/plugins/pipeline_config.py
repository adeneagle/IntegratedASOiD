"""Pure, Qt-free helpers for the pose/RandomForest pipeline.

One request-body builder per backend endpoint's Pydantic request schema (see
backend/routers/{pose,training,inference}.py), plus the small derivations the
panels display (feature width, bin width) and the formatting of training
events. Kept free of Qt so it's unit-testable without a QApplication —
mirrors RealtimeASOID's plugins/pipeline_config.py, but for pose/RF request
shapes instead of crop-params/TSM ones.
"""

from __future__ import annotations

import os
from typing import Optional

ORIGINS = ["deeplabcut", "sleap", "openmonkeystudio"]
POSE_FILE_FILTER = "Pose files (*.h5 *.csv *.json *.mat *.txt);;All files (*)"
MODEL_FILE_FILTER = "Model bundle (*.joblib);;All files (*)"


# ------------------------------------------------------------- derivations
def feature_width(n_bodyparts: int) -> int:
    """Columns per feature row: nchoose2(n)*2 + n — asoid_core.features' width.

    Shown next to the body-part picker so the quadratic growth is visible
    before someone ticks thirty body parts.
    """
    n = int(n_bodyparts)
    if n < 2:
        return 0
    return (n * (n - 1) // 2) * 2 + n


def frames2integ(framerate: float, min_duration: float) -> int:
    """Mirror of asoid_core.indexing.frames2integ_from (banker's rounding included)."""
    return int(round(framerate * (min_duration / 0.1)))


def bin_width(framerate: float, min_duration: float) -> int:
    """Native frames per feature row — mirror of indexing.bin_width_from."""
    return max(1, frames2integ(framerate, min_duration) // 10)


def resolve_binarize(positives: list[str]) -> tuple[Optional[str], Optional[list[str]]]:
    """The positive-class checklist -> (binarize, label_union) for TrainingConfig.

    0 ticked = multi-class, 1 = binarize that behavior, >=2 = union them into
    one positive class (training_service._collapse_labels names it "positive").
    """
    positives = [p for p in positives if p]
    if not positives:
        return None, None
    if len(positives) == 1:
        return positives[0], None
    return None, list(positives)


def threshold_floor_warning(conf_threshold: float, n_classes: int) -> Optional[str]:
    """Warn when the confidence threshold can never be undercut.

    A forest's top probability over k classes is at least 1/k, so with two
    classes a threshold of 0.5 finds no "confusing" sample and the loop stops
    after its seed round — on a model trained on a handful of samples.
    """
    if n_classes < 2:
        return None
    floor = 1.0 / n_classes
    if conf_threshold > floor + 1e-9:
        return None
    return (f"⚠ With {n_classes} classes the forest is always at least {floor:.2f} sure, "
            f"so a threshold of {conf_threshold:.2f} never finds a confusing sample and "
            f"training stops after the first round. Raise it (e.g. "
            f"{min(0.95, floor + 0.2):.2f}).")


def al_label_mapping(config: Optional[dict]) -> dict:
    """kwargs for AlState.load_results from a checkpoint's training config.

    A binarized model predicts in ['other', <positive>], so a human label has to
    be mapped into that space before a frame can be called misclassified —
    otherwise every labeled non-positive frame lights up as an error.
    """
    config = config or {}
    union = config.get("label_union")
    if union:
        return {"binarize": None, "label_union": {"positive": list(union)}}
    return {"binarize": config.get("binarize") or None, "label_union": None}


def model_label(entry: dict) -> str:
    """One-line description of a checkpoint for a model picker."""
    name = entry.get("filename") or os.path.basename(entry.get("path") or "") or "?"
    parts = [entry.get("iteration") or "", name]
    f1 = entry.get("macro_f1")
    if isinstance(f1, (int, float)):
        parts.append(f"F1 {f1:.3f}")
    if entry.get("trained_at"):
        parts.append(str(entry["trained_at"]).replace("T", " "))
    return "  ·  ".join(p for p in parts if p)


def format_train_event(ev: dict, max_iter: Optional[int] = None) -> Optional[str]:
    """Human log line for one training SSE event (None = nothing worth printing).

    The stream carries the active-learning loop's own events (start / ceiling /
    iteration / converged / done) followed by the service's terminal one
    (completed / error / cancelled) — see training_service.start_training.
    """
    kind = ev.get("event")
    status = ev.get("status")
    if kind == "start":
        # Sample/feature counts and the class distribution were already logged
        # by the service's own "starting" messages; only the split is new.
        return (f"split: {ev.get('n_train')} train / {ev.get('n_heldout')} held out"
                + (" (excluding 'other')" if ev.get("exclude_other") else ""))
    if kind == "ceiling":
        return (f"ceiling (all labeled data): macro F1 {ev.get('macro_f1', 0):.3f} "
                f"on {ev.get('n_train')} samples")
    if kind == "iteration":
        total = f"/{max_iter}" if max_iter else ""
        return (f"iter {ev.get('iteration'):>3}{total}  macro F1 {ev.get('macro_f1', 0):.3f}  "
                f"samples {ev.get('n_train')}  low-confidence left "
                f"{ev.get('n_lowconf_remaining')}")
    if kind == "done":
        return None  # the service's "completed" event follows with the real summary
    if status == "completed":
        f1 = ev.get("macro_f1")
        f1s = f"{f1:.3f}" if isinstance(f1, (int, float)) else "?"
        return f"done: macro F1 {f1s} in {ev.get('iterations')} round(s) — {ev.get('message', '')}"
    if status == "error":
        return f"ERROR: {ev.get('error') or 'unknown error'}"
    if status == "cancelled":
        return ev.get("message") or "cancelled"
    if ev.get("message"):
        return ev["message"]
    return None


def summarize_extraction(results: dict) -> list[str]:
    """Log lines for /api/pose/results — counts plus the warnings that matter.

    The two silent failure modes worth shouting about: a training set that is
    almost all "other" (the forest learns to say "other"), and tracking so poor
    that most points fall under the likelihood cutoff.
    """
    lines = []
    if results.get("class_names"):
        lines.append(f"classes: {', '.join(results['class_names'])}")
    if results.get("bin_width") is not None:
        lines.append(f"bin width: {results['bin_width']} frames "
                     f"(frames2integ {results.get('frames2integ')})")
    for v in results.get("videos") or []:
        counts = " ".join(f"{k}={n}" for k, n in (v.get("class_counts") or {}).items())
        lines.append(f"{v.get('id')}: {v.get('n_rows')} rows × {v.get('n_features')} "
                     f"features | {counts}")
        skipped = v.get("frames_skipped") or 0
        if skipped > 0:
            lines.append(f"  note: pose file has {v.get('n_frames_total')} frames but only "
                         f"{v.get('n_frames')} were used — {skipped} frames past the end "
                         f"of the video were skipped.")
        other = v.get("other_fraction") or 0.0
        if other > 0.9:
            lines.append(f"  warning: {other * 100:.1f}% of samples are \"other\" — the "
                         f"classifier will mostly learn to say \"other\". Label more, or "
                         f"tick \"Exclude 'other'\" on the Train tab.")
        rejected = v.get("mean_rejected_fraction") or 0.0
        if rejected > 0.2:
            lines.append(f"  warning: {rejected * 100:.0f}% of points were below the "
                         f"likelihood threshold — check tracking quality.")
    return lines


def length_mismatch(pose_frames: Optional[int], video_frames: Optional[int],
                    tolerance: float = 0.02) -> Optional[str]:
    """Warning text when a pose file and its video don't cover the same frames.

    Features, labels and predictions are all indexed by frame with no offset,
    so a session-length pose file on a short clip breaks silently: the
    uncertainty timeline squeezes hours into minutes of trackbar.
    """
    if not pose_frames or not video_frames:
        return None
    ratio = pose_frames / video_frames
    if 1 - tolerance <= ratio <= 1 + tolerance:
        return None
    how = (f"{ratio:.1f}× more" if ratio > 1 else f"only {ratio * 100:.0f}%")
    return (f"⚠ The video has {video_frames} frames but this pose file has "
            f"{pose_frames} ({how}). They must describe the same recording, "
            f"aligned from frame 0.")


# ------------------------------------------------------------ request bodies
def build_extract_request(
    *,
    project_dir: str,
    videos: list[dict],           # [{"id", "pose_path", "label_path"?, "max_frames"?}]
    origin: str = "deeplabcut",
    bodyparts: list[str],
    framerate: float = 30.0,
    min_duration: float = 0.1,
    llh_value: float = 0.6,
    animals: Optional[list[str]] = None,
    multi_animal: bool = False,
    iteration_dir: str = "iteration-0",
    keep_classes: Optional[list[str]] = None,
) -> dict:
    """Body for POST /api/pose/extract-features."""
    return {
        "project_dir": project_dir,
        "videos": videos,
        "origin": origin,
        "bodyparts": list(bodyparts),
        "framerate": framerate,
        "min_duration": min_duration,
        "llh_value": llh_value,
        "animals": list(animals) if animals is not None else None,
        "multi_animal": multi_animal,
        "iteration_dir": iteration_dir,
        "keep_classes": list(keep_classes) if keep_classes is not None else None,
    }


def build_training_config(
    *,
    project_dir: str,
    video_ids: list[str],
    iteration_dir: str = "iteration-0",
    init_ratio: float = 0.01,
    max_iter: int = 100,
    max_samples_iter: int = 20,
    conf_threshold: float = 0.5,
    exclude_other: bool = False,
    binarize: Optional[str] = None,
    label_union: Optional[list[str]] = None,
    checkpoint_dir: Optional[str] = None,
    model_name: Optional[str] = None,
) -> dict:
    """Body for POST /api/training/start."""
    return {
        "project_dir": project_dir,
        "video_ids": list(video_ids),
        "iteration_dir": iteration_dir,
        "init_ratio": init_ratio,
        "max_iter": max_iter,
        "max_samples_iter": max_samples_iter,
        "conf_threshold": conf_threshold,
        "exclude_other": exclude_other,
        "binarize": binarize,
        "label_union": list(label_union) if label_union is not None else None,
        "checkpoint_dir": checkpoint_dir,
        "model_name": model_name,
    }


def build_inference_request(
    *,
    checkpoint_path: str,
    pose_path: str,
    origin: str = "deeplabcut",
    bodyparts: Optional[list[str]] = None,
    llh_value: float = 0.6,
    animals: Optional[list[str]] = None,
    multi_animal: bool = False,
    frame_ranges: Optional[list] = None,
) -> dict:
    """Body for POST /api/inference/run (single video, active learning).

    `bodyparts=None` lets the backend use the selection recorded on the
    checkpoint — the only one guaranteed to match its feature space.
    """
    return {
        "checkpoint_path": checkpoint_path,
        "pose_path": pose_path,
        "origin": origin,
        "bodyparts": list(bodyparts) if bodyparts is not None else None,
        "llh_value": llh_value,
        "animals": list(animals) if animals is not None else None,
        "multi_animal": multi_animal,
        "frame_ranges": frame_ranges,
    }


def build_batch_request(
    *,
    checkpoint_path: str,
    pose_paths: list[str],
    output_dir: str,
    origin: str = "deeplabcut",
    bodyparts: Optional[list[str]] = None,
    llh_value: float = 0.6,
    animals: Optional[list[str]] = None,
    multi_animal: bool = False,
) -> dict:
    """Body for POST /api/inference/batch."""
    return {
        "checkpoint_path": checkpoint_path,
        "pose_paths": list(pose_paths),
        "output_dir": output_dir,
        "origin": origin,
        "bodyparts": list(bodyparts) if bodyparts is not None else None,
        "llh_value": llh_value,
        "animals": list(animals) if animals is not None else None,
        "multi_animal": multi_animal,
    }
