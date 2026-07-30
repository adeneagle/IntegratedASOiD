"""
RLT file format service: HDF5 project files with .rlt extension.

Two flavors, both HDF5:
  - **Lightweight** (default): manifest (project info, behaviors, config), video
    and pose *references*, and the hand-annotated labels only. Small and
    shareable — modeled on SLEAP's base `.slp`. Features and models are
    regenerated after import.
  - **Package** (`package=True`): everything above PLUS the pose files
    themselves, the extracted features, active-learning predictions, and model
    checkpoints. A self-contained, train-anywhere bundle — modeled on SLEAP's
    `.pkg.slp`.

Import auto-detects which groups are present and restores accordingly.

Packages are dramatically smaller than the TSM predecessor's: that format
embedded a PNG per labeled frame, where this one embeds the pose file those
frames were derived from.

HDF5 layout:
    /                               (attr: rlt_version, packaged)
    /                               (attr: manifest — JSON string)
    /labels/{video_id}              (dataset: UTF-8 CSV string)
    /pose/{video_id}                (dataset: uint8 file bytes, gzip)  [package]
      .attrs["filename"]              original pose filename (keeps the extension)
    /features/{iteration}/{video_id} (dataset: uint8 .npz bytes, gzip) [package]
    /predictions/{video_id}         (dataset: UTF-8 JSON string)       [package]
    /models/checkpoints/{name}      (dataset: uint8 .joblib blob)      [package]
"""

import json
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


RLT_VERSION = 5
# v5 replaced the pixel-pipeline groups (/frames, /backgrounds) with /pose and
# /features. Older files describe a project this app cannot act on — they carry
# cropped frame images but no pose data — so they are rejected rather than
# imported into a half-usable state.
MIN_SUPPORTED_VERSION = 5


# ── Export (folder → .rlt) ────────────────────────────────────────────────


def export_rlt(
    project_dir: str,
    output_path: str,
    *,
    package: bool = False,
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Pack a project into an .rlt file.

    Args:
        project_dir: Path to the project folder (must contain project.json).
        output_path: Where to write the .rlt file.
        package:     If True, also embed pose files, features, predictions and
                     models (a self-contained bundle). If False (default),
                     labels + manifest only.
        progress_cb: Optional callback(stage: str, pct: float).

    Returns:
        Summary dict with counts of packed items.
    """
    project_dir = Path(project_dir)
    manifest_path = project_dir / "project.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No project.json in {project_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    summary = {
        "videos": len(manifest.get("videos", [])),
        "labels": 0,
        "pose": 0,
        "features": 0,
        "predictions": 0,
        "checkpoints": 0,
        "packaged": package,
    }

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    with h5py.File(output_path, "w") as hf:
        hf.attrs["rlt_version"] = RLT_VERSION
        hf.attrs["packaged"] = bool(package)
        hf.attrs["manifest"] = json.dumps(manifest, ensure_ascii=False)

        # ── Labels (always) ───────────────────────────────────────────
        _report("labels", 0)
        labels_grp = hf.create_group("labels")
        labels_dir = project_dir / "labels"
        if labels_dir.is_dir():
            csv_files = sorted(labels_dir.glob("*.csv"))
            for i, csv_path in enumerate(csv_files):
                vid_id = csv_path.stem.replace("_bouts", "")
                labels_grp.create_dataset(vid_id, data=csv_path.read_text(encoding="utf-8"))
                summary["labels"] += 1
                _report("labels", (i + 1) / max(len(csv_files), 1) * 100)

        if not package:
            _report("done", 100)
            return summary

        # ── Pose files (package only) ─────────────────────────────────
        # Embedded by value so the bundle trains anywhere. The original filename
        # rides along as an attribute because the loader dispatches on extension
        # (.csv = DeepLabCut, .h5 = SLEAP, .mat/.txt = OpenMonkeyStudio).
        pose_grp = hf.create_group("pose")
        videos = manifest.get("videos", [])
        for vi, video in enumerate(videos):
            pose_path = video.get("pose")
            if not pose_path or not os.path.exists(pose_path):
                continue
            raw = np.frombuffer(Path(pose_path).read_bytes(), dtype=np.uint8)
            ds = pose_grp.create_dataset(
                video["id"], data=raw, compression="gzip", compression_opts=4
            )
            ds.attrs["filename"] = os.path.basename(pose_path)
            summary["pose"] += 1
            _report("pose", (vi + 1) / max(len(videos), 1) * 100)

        # ── Features (package only) ───────────────────────────────────
        # Every iteration-N/features/ directory, so a refined project keeps its
        # history rather than only the latest round.
        feats_grp = hf.create_group("features")
        iter_dirs = sorted(
            d for d in project_dir.glob("iteration-*") if (d / "features").is_dir()
        )
        for di, idir in enumerate(iter_dirs):
            sub = feats_grp.create_group(idir.name)
            for npz in sorted((idir / "features").glob("*.npz")):
                raw = np.frombuffer(npz.read_bytes(), dtype=np.uint8)
                sub.create_dataset(npz.stem, data=raw, compression="gzip", compression_opts=4)
                summary["features"] += 1
            _report("features", (di + 1) / max(len(iter_dirs), 1) * 100)

        # ── Predictions (package only) ────────────────────────────────
        pred_grp = hf.create_group("predictions")
        pred_dir = project_dir / "predictions"
        if pred_dir.is_dir():
            preds = sorted(pred_dir.glob("*.json"))
            for i, pred in enumerate(preds):
                pred_grp.create_dataset(pred.stem, data=pred.read_text(encoding="utf-8"))
                summary["predictions"] += 1
                _report("predictions", (i + 1) / max(len(preds), 1) * 100)

        # ── Models (package only) ─────────────────────────────────────
        models_grp = hf.create_group("models")
        ckpt_grp = models_grp.create_group("checkpoints")
        mdir = project_dir / "models" / "checkpoints"
        if mdir.is_dir():
            files = sorted(p for p in mdir.glob("*.joblib") if p.is_file())
            for i, f in enumerate(files):
                raw = np.frombuffer(f.read_bytes(), dtype=np.uint8)
                ckpt_grp.create_dataset(f.name, data=raw, compression="gzip", compression_opts=4)
                summary["checkpoints"] += 1
                _report("models", (i + 1) / max(len(files), 1) * 100)

        _report("done", 100)

    return summary


# ── Import (.rlt → folder) ────────────────────────────────────────────────


def _check_version(hf) -> int:
    version = int(hf.attrs.get("rlt_version", 1))
    if version < MIN_SUPPORTED_VERSION:
        raise ValueError(
            f"This .rlt is version {version}, from the video/TSM version of the app. "
            f"It contains cropped frame images but no pose data, so it cannot be "
            f"opened here (this app needs version {MIN_SUPPORTED_VERSION} or newer). "
            f"Re-export it from the original app, or start a new project and import "
            f"its label CSVs."
        )
    return version


def import_rlt(
    rlt_path: str,
    output_dir: str,
    *,
    progress_cb: Optional[callable] = None,
) -> dict:
    """
    Unpack an .rlt into a project directory. Restores labels always, plus pose
    files, features, predictions and models if the file is a package.

    Video paths from the manifest are preserved (re-link on the new machine if
    they don't resolve). Pose paths, in contrast, are *rewritten* to point at the
    extracted copies — the bundle carries pose by value, so the imported project
    is immediately trainable without re-linking.

    Returns the manifest dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _report(stage, pct):
        if progress_cb:
            progress_cb(stage, pct)

    with h5py.File(rlt_path, "r") as hf:
        _check_version(hf)
        manifest = json.loads(hf.attrs["manifest"])

        for sub in ("labels", "pose", "predictions",
                    os.path.join("iteration-0", "features"),
                    os.path.join("models", "checkpoints")):
            (output_dir / sub).mkdir(parents=True, exist_ok=True)

        # Labels
        if "labels" in hf:
            labels_dir = output_dir / "labels"
            for vid_id in hf["labels"].keys():
                content = hf["labels"][vid_id][()]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                (labels_dir / f"{vid_id}_bouts.csv").write_text(content, encoding="utf-8")
            _report("labels", 100)

        # Pose (package) — extract and repoint the manifest at the local copy.
        if "pose" in hf:
            pose_dir = output_dir / "pose"
            by_id = {v.get("id"): v for v in manifest.get("videos", [])}
            keys = list(hf["pose"].keys())
            for i, vid_id in enumerate(keys):
                ds = hf["pose"][vid_id]
                filename = ds.attrs.get("filename", f"{vid_id}.csv")
                if isinstance(filename, bytes):
                    filename = filename.decode("utf-8")
                dest = pose_dir / filename
                dest.write_bytes(ds[()].tobytes())
                if vid_id in by_id:
                    by_id[vid_id]["pose"] = str(dest.resolve())
                _report("pose", (i + 1) / max(len(keys), 1) * 100)

        # Features (package)
        if "features" in hf:
            iters = list(hf["features"].keys())
            for ii, iter_name in enumerate(iters):
                dest_dir = output_dir / iter_name / "features"
                dest_dir.mkdir(parents=True, exist_ok=True)
                for vid_id in hf["features"][iter_name].keys():
                    data = hf["features"][iter_name][vid_id][()]
                    (dest_dir / f"{vid_id}.npz").write_bytes(data.tobytes())
                _report("features", (ii + 1) / max(len(iters), 1) * 100)

        # Predictions (package)
        if "predictions" in hf:
            pred_dir = output_dir / "predictions"
            for vid_id in hf["predictions"].keys():
                content = hf["predictions"][vid_id][()]
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
                (pred_dir / f"{vid_id}.json").write_text(content, encoding="utf-8")
            _report("predictions", 100)

        # Models (package)
        if "models" in hf and "checkpoints" in hf["models"]:
            dest_dir = output_dir / "models" / "checkpoints"
            dest_dir.mkdir(parents=True, exist_ok=True)
            for name in hf["models"]["checkpoints"].keys():
                data = hf["models"]["checkpoints"][name][()]
                (dest_dir / name).write_bytes(data.tobytes())
            _report("models", 100)

        (output_dir / "project.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _report("done", 100)

    return manifest


# ── Inspect (read metadata only) ─────────────────────────────────────────


def inspect_rlt(rlt_path: str) -> dict:
    """Read metadata from an .rlt without extracting. Returns counts + flags."""
    with h5py.File(rlt_path, "r") as hf:
        _check_version(hf)
        manifest = json.loads(hf.attrs["manifest"])
        feature_count = 0
        if "features" in hf:
            for iter_name in hf["features"]:
                feature_count += len(hf["features"][iter_name])
        return {
            "version": int(hf.attrs.get("rlt_version", 1)),
            "packaged": bool(hf.attrs.get("packaged", "pose" in hf)),
            "manifest": manifest,
            "videos": len(manifest.get("videos", [])),
            "labels": len(hf["labels"]) if "labels" in hf else 0,
            "pose": len(hf["pose"]) if "pose" in hf else 0,
            "features": feature_count,
            "predictions": len(hf["predictions"]) if "predictions" in hf else 0,
            "checkpoints": (
                len(hf["models"]["checkpoints"])
                if "models" in hf and "checkpoints" in hf["models"] else 0
            ),
        }
