"""Tests for the .rlt project bundle format (v5: pose + features, no pixels)."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from backend.services.rlt_service import (
    RLT_VERSION,
    export_rlt,
    import_rlt,
    inspect_rlt,
)


# ── Lightweight export ────────────────────────────────────────────────────

def test_lightweight_export_keeps_labels_and_drops_the_rest(sample_project, tmp_path):
    project_dir, _manifest = sample_project
    out = str(tmp_path / "light.rlt")
    summary = export_rlt(project_dir, out, package=False)

    assert summary["packaged"] is False
    assert summary["labels"] == 2
    assert summary["pose"] == 0
    assert summary["features"] == 0
    assert summary["checkpoints"] == 0

    with h5py.File(out, "r") as hf:
        assert hf.attrs["rlt_version"] == RLT_VERSION
        assert "labels" in hf
        assert "pose" not in hf
        assert "features" not in hf
        assert "models" not in hf


def test_lightweight_export_is_small(sample_project, tmp_path):
    project_dir, _ = sample_project
    light = str(tmp_path / "light.rlt")
    package = str(tmp_path / "pkg.rlt")
    export_rlt(project_dir, light, package=False)
    export_rlt(project_dir, package, package=True)
    assert os.path.getsize(light) < os.path.getsize(package)


def test_manifest_is_preserved_verbatim(sample_project, tmp_path):
    project_dir, manifest = sample_project
    out = str(tmp_path / "light.rlt")
    export_rlt(project_dir, out, package=False)
    with h5py.File(out, "r") as hf:
        packed = json.loads(hf.attrs["manifest"])
    assert packed["behaviors"] == manifest["behaviors"]
    assert [v["id"] for v in packed["videos"]] == ["vid_a", "vid_b"]


def test_export_requires_a_project_json(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="project.json"):
        export_rlt(str(empty), str(tmp_path / "x.rlt"))


# ── Package export ────────────────────────────────────────────────────────

def test_package_export_embeds_pose_features_and_models(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "pkg.rlt")
    summary = export_rlt(project_dir, out, package=True)

    assert summary["packaged"] is True
    assert summary["pose"] == 1        # only vid_a has a pose file
    assert summary["features"] == 1
    assert summary["checkpoints"] == 1
    assert summary["predictions"] == 1

    with h5py.File(out, "r") as hf:
        assert "pose" in hf and "vid_a" in hf["pose"]
        assert hf["pose"]["vid_a"].attrs["filename"].endswith(".csv")
        assert "iteration-0" in hf["features"]
        assert "vid_a" in hf["features"]["iteration-0"]
        assert "rf_model.joblib" in hf["models"]["checkpoints"]


def test_package_skips_videos_without_a_pose_file(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "pkg.rlt")
    export_rlt(project_dir, out, package=True)
    with h5py.File(out, "r") as hf:
        assert "vid_b" not in hf["pose"]


# ── Round trip ────────────────────────────────────────────────────────────

def test_lightweight_round_trip_restores_labels(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "light.rlt")
    export_rlt(project_dir, out, package=False)

    dest = tmp_path / "restored"
    manifest = import_rlt(out, str(dest))

    assert (dest / "project.json").exists()
    assert (dest / "labels" / "vid_a_bouts.csv").exists()
    original = Path(project_dir, "labels", "vid_a_bouts.csv").read_text(encoding="utf-8")
    assert (dest / "labels" / "vid_a_bouts.csv").read_text(encoding="utf-8") == original
    assert len(manifest["videos"]) == 2


def test_package_round_trip_restores_pose_and_repoints_the_manifest(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "pkg.rlt")
    export_rlt(project_dir, out, package=True)

    dest = tmp_path / "restored"
    manifest = import_rlt(out, str(dest))

    pose_files = list((dest / "pose").glob("*"))
    assert len(pose_files) == 1

    # The manifest must now point at the extracted copy, not the original path,
    # so the imported project is trainable without re-linking.
    vid_a = next(v for v in manifest["videos"] if v["id"] == "vid_a")
    assert os.path.exists(vid_a["pose"])
    assert Path(vid_a["pose"]).parent == (dest / "pose").resolve()

    # Video paths, by contrast, are left alone for the user to re-link.
    assert vid_a["path"] == "/abs/path/to/vid_a.mp4"


def test_package_round_trip_restores_features_byte_for_byte(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "pkg.rlt")
    export_rlt(project_dir, out, package=True)

    dest = tmp_path / "restored"
    import_rlt(out, str(dest))

    src = Path(project_dir, "iteration-0", "features", "vid_a.npz")
    got = dest / "iteration-0" / "features" / "vid_a.npz"
    assert got.exists()
    assert got.read_bytes() == src.read_bytes()
    # And it must still load as an npz.
    with np.load(got, allow_pickle=True) as data:
        assert data["features"].shape == (10, 16)


def test_package_round_trip_restores_models_and_predictions(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "pkg.rlt")
    export_rlt(project_dir, out, package=True)

    dest = tmp_path / "restored"
    import_rlt(out, str(dest))

    ckpt = dest / "models" / "checkpoints" / "rf_model.joblib"
    assert ckpt.exists()
    assert ckpt.read_bytes() == b"\x00" * 512
    assert (dest / "predictions" / "vid_a.json").exists()


def test_multiple_iterations_are_all_packed(sample_project, tmp_path):
    project_dir, _ = sample_project
    # Add a second iteration.
    it2 = Path(project_dir, "iteration-1", "features")
    it2.mkdir(parents=True)
    np.savez(it2 / "vid_a.npz", features=np.zeros((3, 16)), targets=np.zeros(3, dtype=int))

    out = str(tmp_path / "pkg.rlt")
    summary = export_rlt(project_dir, out, package=True)
    assert summary["features"] == 2

    dest = tmp_path / "restored"
    import_rlt(out, str(dest))
    assert (dest / "iteration-0" / "features" / "vid_a.npz").exists()
    assert (dest / "iteration-1" / "features" / "vid_a.npz").exists()


# ── Inspect ───────────────────────────────────────────────────────────────

def test_inspect_reports_counts_without_extracting(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "pkg.rlt")
    export_rlt(project_dir, out, package=True)

    info = inspect_rlt(out)
    assert info["version"] == RLT_VERSION
    assert info["packaged"] is True
    assert info["videos"] == 2
    assert info["labels"] == 2
    assert info["pose"] == 1
    assert info["features"] == 1
    assert info["checkpoints"] == 1


def test_inspect_flags_a_lightweight_file(sample_project, tmp_path):
    project_dir, _ = sample_project
    out = str(tmp_path / "light.rlt")
    export_rlt(project_dir, out, package=False)
    info = inspect_rlt(out)
    assert info["packaged"] is False
    assert info["pose"] == 0


# ── Version gate ──────────────────────────────────────────────────────────

def _write_legacy_rlt(path, version):
    """A minimal v3-style file from the TSM app (frames, no pose)."""
    with h5py.File(path, "w") as hf:
        hf.attrs["rlt_version"] = version
        hf.attrs["packaged"] = True
        hf.attrs["manifest"] = json.dumps({"name": "old", "videos": []})
        hf.create_group("labels")
        hf.create_group("frames")


@pytest.mark.parametrize("version", [1, 3, 4])
def test_legacy_rlt_is_rejected_with_a_clear_message(tmp_path, version):
    old = str(tmp_path / "old.rlt")
    _write_legacy_rlt(old, version)
    with pytest.raises(ValueError, match="no pose data"):
        import_rlt(old, str(tmp_path / "dest"))
    with pytest.raises(ValueError, match=f"version {version}"):
        inspect_rlt(old)
