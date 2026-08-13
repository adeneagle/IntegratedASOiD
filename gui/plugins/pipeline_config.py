"""Pure, Qt-free request-body builders for the pose/RandomForest pipeline.

One function per backend endpoint's Pydantic request schema (see
backend/routers/{pose,training,inference}.py). Kept free of Qt so it's
unit-testable without a QApplication — mirrors RealtimeASOID-python's
plugins/pipeline_config.py, but for pose/RF request shapes instead of
crop-params/TSM ones.
"""

from __future__ import annotations

from typing import Iterable, Optional


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
    """Body for POST /api/inference/run (single video, active learning)."""
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
