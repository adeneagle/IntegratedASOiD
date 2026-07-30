"""
Shared fixtures for the A-SOID Desktop test suite.

Provides synthetic pose files (DeepLabCut csv + SLEAP h5), sparse label CSVs in
the exact format the labeling UI writes, a sample project on disk, and a FastAPI
test client — all without requiring real data.

The synthetic pose generator produces a genuinely learnable task (two behaviors
that differ in how far the body parts spread from the centroid) so the
active-learning tests can assert on real F1 movement rather than on mocks.
"""

import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


BODYPARTS = ["nose", "leftear", "rightear", "tailbase"]
FPS = 30


# ── Synthetic pose generation ─────────────────────────────────────────────

def make_behavior_series(n_frames: int, block: int = 150) -> np.ndarray:
    """Alternating 0/1 behavior blocks — the ground truth to be recovered."""
    return (np.arange(n_frames) // block) % 2


def make_pose_array(n_frames: int, behavior: np.ndarray, seed: int = 0,
                    noise: float = 1.5) -> np.ndarray:
    """(n_frames, n_bodyparts*2) pose where behavior 1 spreads the parts wider.

    The animal orbits the arena in both states, so the classifier can only
    succeed on the inter-bodypart distances the feature extractor computes — not
    on absolute position.
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_frames)
    cx = 300 + 60 * np.sin(t / 40.0)
    cy = 300 + 60 * np.cos(t / 40.0)
    spread = np.where(behavior == 1, 45.0, 14.0)

    pose = np.zeros((n_frames, len(BODYPARTS) * 2))
    for b in range(len(BODYPARTS)):
        ang = b * (2 * np.pi / len(BODYPARTS))
        pose[:, 2 * b] = cx + spread * np.cos(ang) + rng.randn(n_frames) * noise
        pose[:, 2 * b + 1] = cy + spread * np.sin(ang) + rng.randn(n_frames) * noise
    return pose


@pytest.fixture
def behavior_series():
    return make_behavior_series(3000)


@pytest.fixture
def pose_array(behavior_series):
    return make_pose_array(3000, behavior_series)


@pytest.fixture
def dlc_csv(tmp_path, behavior_series):
    """A single-animal DeepLabCut csv (3-level header: scorer/bodyparts/coords)."""
    n = len(behavior_series)
    pose = make_pose_array(n, behavior_series)
    rng = np.random.RandomState(3)

    cols, data = [], []
    for b, bp in enumerate(BODYPARTS):
        cols += [("scorer", bp, "x"), ("scorer", bp, "y"), ("scorer", bp, "likelihood")]
        data += [pose[:, 2 * b], pose[:, 2 * b + 1],
                 np.clip(0.98 - rng.rand(n) * 0.1, 0, 1)]

    df = pd.DataFrame(
        np.column_stack(data),
        columns=pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"]),
    )
    path = tmp_path / "vid1_DLC.csv"
    df.to_csv(path)
    return str(path)


@pytest.fixture
def dlc_csv_lowconf(tmp_path, behavior_series):
    """A DLC csv where one body part drops below threshold for a stretch.

    Exercises adp_filt's hold-last-position behaviour.
    """
    n = len(behavior_series)
    pose = make_pose_array(n, behavior_series, seed=11)

    cols, data = [], []
    for b, bp in enumerate(BODYPARTS):
        llh = np.full(n, 0.95)
        if bp == "nose":
            llh[100:200] = 0.1  # tracking lost
        cols += [("scorer", bp, "x"), ("scorer", bp, "y"), ("scorer", bp, "likelihood")]
        data += [pose[:, 2 * b], pose[:, 2 * b + 1], llh]

    df = pd.DataFrame(
        np.column_stack(data),
        columns=pd.MultiIndex.from_tuples(cols, names=["scorer", "bodyparts", "coords"]),
    )
    path = tmp_path / "vid_lowconf_DLC.csv"
    df.to_csv(path)
    return str(path)


@pytest.fixture
def sleap_h5(tmp_path, behavior_series):
    """A SLEAP-style analysis .h5 (tracks + node_names), single track."""
    n = len(behavior_series)
    pose = make_pose_array(n, behavior_series, seed=5)
    n_bp = len(BODYPARTS)

    # SLEAP stores tracks as (n_tracks, n_coords, n_nodes, n_frames); the loader
    # transposes it, so build the transpose of what it expects to read back.
    tracks = np.zeros((1, 2, n_bp, n))
    for b in range(n_bp):
        tracks[0, 0, b, :] = pose[:, 2 * b]
        tracks[0, 1, b, :] = pose[:, 2 * b + 1]

    path = tmp_path / "vid1.analysis.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("tracks", data=tracks)
        f.create_dataset("node_names",
                         data=np.array(BODYPARTS, dtype=h5py.special_dtype(vlen=str)))
    return str(path)


# ── Label CSV fixtures ────────────────────────────────────────────────────

def write_sparse_labels(path, behavior, names=("rearing", "grooming")):
    """Write labels exactly as `generateTrainingLabelCsv` does: SPARSE one-hot.

    Rows exist only for in-bout frames. Half the behavior-0 blocks are left
    unlabeled on purpose so the sparse->dense "other" fill is exercised.
    """
    lines = ["time," + ",".join(names)]
    for i, b in enumerate(behavior):
        if b == 1:
            lines.append(f"{i},1,0")
        elif (i // 150) % 4 == 0:
            lines.append(f"{i},0,1")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return list(names)


@pytest.fixture
def label_csv(tmp_path, behavior_series):
    path = tmp_path / "vid1_bouts.csv"
    write_sparse_labels(str(path), behavior_series)
    return str(path)


# ── Sample project fixture (for .rlt / project router tests) ──────────────

@pytest.fixture
def sample_project(tmp_path, dlc_csv, behavior_series):
    """A realistic project directory on disk. Returns (project_dir, manifest).

    Includes regenerable data (features, a model) so tests can assert that a
    lightweight .rlt export deliberately *excludes* them.
    """
    proj = tmp_path / "sample_project"
    for sub in ("labels", "pose", "predictions",
                os.path.join("iteration-0", "features"),
                os.path.join("models", "checkpoints")):
        os.makedirs(proj / sub, exist_ok=True)

    manifest = {
        "name": "sample_project",
        "created": "2026-01-01",
        "version": 2,
        "behaviors": [
            {"name": "rearing", "key": "1"},
            {"name": "grooming", "key": "2"},
        ],
        "videos": [
            {
                "id": "vid_a",
                "path": "/abs/path/to/vid_a.mp4",
                "pose": dlc_csv,
                "labels": "labels/vid_a_bouts.csv",
                "features": "iteration-0/features/vid_a.npz",
            },
            {
                "id": "vid_b",
                "path": "/abs/path/to/vid_b.mp4",
                "pose": None,
                "labels": "labels/vid_b_bouts.csv",
                "features": "iteration-0/features/vid_b.npz",
            },
        ],
        "preprocessing": {"bodyparts": BODYPARTS, "framerate": FPS, "llh_value": 0.6},
        "training": {"init_ratio": 0.05, "max_iter": 10},
        "models": [],
    }
    (proj / "project.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_sparse_labels(str(proj / "labels" / "vid_a_bouts.csv"), behavior_series)
    write_sparse_labels(str(proj / "labels" / "vid_b_bouts.csv"), behavior_series)

    # Regenerable data
    np.savez_compressed(
        proj / "iteration-0" / "features" / "vid_a.npz",
        features=np.zeros((10, 16)), targets=np.zeros(10, dtype=int),
        frames2integ=30, bin_width=3, n_frames=100, is_3d=False,
        bodyparts=np.array(BODYPARTS, dtype=object),
        class_names=np.array(["rearing", "grooming", "other"], dtype=object),
    )
    (proj / "models" / "checkpoints" / "rf_model.joblib").write_bytes(b"\x00" * 512)
    (proj / "predictions" / "vid_a.json").write_text('{"frames": []}', encoding="utf-8")

    return str(proj), manifest


# ── FastAPI test client ──────────────────────────────────────────────────

@pytest.fixture
def client():
    """An httpx AsyncClient bound to the FastAPI app."""
    from httpx import AsyncClient, ASGITransport
    from backend.server import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ── Helpers ──────────────────────────────────────────────────────────────

def wait_for(get_status, timeout=120.0, interval=0.05):
    """Block until a task reaches a terminal state. Returns the final status."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get_status()
        if status in ("completed", "error", "cancelled"):
            return status
        time.sleep(interval)
    raise AssertionError(f"task did not finish within {timeout}s (last: {get_status()})")


# ── Cleanup task registries between tests ─────────────────────────────────

@pytest.fixture(autouse=True)
def clear_task_registries():
    """Clear all service task registries after each test."""
    yield
    from backend.services import pose_service, training_service, inference_service
    pose_service._tasks.clear()
    training_service._tasks.clear()
    inference_service._tasks.clear()
