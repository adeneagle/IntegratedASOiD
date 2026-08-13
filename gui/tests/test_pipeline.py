"""pipeline_config / project_manager / panel construction (offscreen).

    QT_QPA_PLATFORM=offscreen .venv-gui/Scripts/python.exe gui/tests/test_pipeline.py
    pytest gui/tests/test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PySide6 import QtWidgets  # noqa: E402

from labeling_shell.active_learning import AlState  # noqa: E402
from plugins import pipeline_config as pc  # noqa: E402
from plugins.pose_panel import hash_string  # noqa: E402
from plugins.project_manager import PoseProjectManager, sanitize_id  # noqa: E402


def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ------------------------------------------------------------- sanitize_id
def test_sanitize_id_matches_js_algorithm():
    # Known outputs of project-manager.js's sanitizeId(filename).
    cases = {
        "my video.mp4": "my_video",
        "a__b--c.mkv": "a_b--c",
        "_leading_trailing_.avi": "leading_trailing",
        "weird!!name.mp4": "weird_name",
        "plain.h5": "plain",
    }
    for inp, expect in cases.items():
        assert sanitize_id(inp) == expect


# ------------------------------------------------------------ config builders
def test_build_extract_request_shape():
    req = pc.build_extract_request(
        project_dir="P", videos=[{"id": "v1", "pose_path": "p1"}],
        bodyparts=["nose", "tail"],
    )
    assert req["project_dir"] == "P"
    assert req["bodyparts"] == ["nose", "tail"]
    assert req["origin"] == "deeplabcut"
    assert req["iteration_dir"] == "iteration-0"


def test_build_training_config_shape():
    cfg = pc.build_training_config(project_dir="P", video_ids=["v1", "v2"])
    assert cfg["video_ids"] == ["v1", "v2"]
    assert cfg["init_ratio"] == 0.01
    assert cfg["binarize"] is None


def test_build_inference_and_batch_request_shapes():
    inf = pc.build_inference_request(checkpoint_path="m.joblib", pose_path="p.h5")
    assert inf["checkpoint_path"] == "m.joblib"
    assert inf["frame_ranges"] is None

    batch = pc.build_batch_request(
        checkpoint_path="m.joblib", pose_paths=["a.h5", "b.h5"], output_dir="out")
    assert batch["pose_paths"] == ["a.h5", "b.h5"]
    assert batch["output_dir"] == "out"


def test_hash_string_is_stable_and_nonempty():
    assert hash_string("abc") == hash_string("abc")
    assert hash_string("abc") != hash_string("abd")
    assert hash_string("") != ""


# ------------------------------------------------------------ project manager
def test_pose_project_manager_roundtrip():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        pm = PoseProjectManager(d)
        pm.create("proj")
        pm.register_video("clip1", "clip1.mp4", "/videos/clip1.mp4", frame_count=100)
        pm.set_video_pose("clip1", "/poses/clip1.h5")
        pm.update_preprocessing_config({"origin": "deeplabcut", "llh_value": 0.6})
        pm.update_training_config({"init_ratio": 0.02})
        pm.add_model({"name": "m1", "path": "m1.joblib"})
        pm.set_current_iteration(2)
        pm.set_iteration_meta("iteration-2", {"n_rows": 500})
        pm.flush_manifest()

        pm2 = PoseProjectManager(d)
        m = pm2.load_manifest()
        assert m["version"] == 2
        assert m["current_iteration"] == 2
        assert pm2.get_video_pose("clip1") == "/poses/clip1.h5"
        assert pm2.get_video_frame_count("clip1") == 100
        assert pm2.get_models()[0]["path"] == "m1.joblib"
        assert pm2.get_iteration_meta("iteration-2")["n_rows"] == 500
        assert pm2.checkpoint_dir().name == "checkpoints"
        assert pm2.checkpoint_dir().parent.name == "models"


def test_pose_project_manager_add_model_dedupes_by_path():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        pm = PoseProjectManager(d)
        pm.create("proj")
        pm.add_model({"name": "m1", "path": "same.joblib", "macro_f1": 0.5})
        pm.add_model({"name": "m1b", "path": "same.joblib", "macro_f1": 0.9})
        models = pm.get_models()
        assert len(models) == 1
        assert models[0]["macro_f1"] == 0.9


# ------------------------------------------------------------------ AL stats
def test_uncertain_segments_respects_min_length():
    arr = [-1, 0.9, 0.9, 0.9, -1, 0.9, -1]
    segs = AlState.uncertain_segments(arr, threshold=0.5, min_len=2)
    assert segs == [{"start": 1, "end": 3, "length": 3}]


def test_al_state_stats_summary():
    state = AlState()
    state.load_results({
        "class_names": ["other", "groom"],
        "frames": [
            {"index": 0, "predicted_class": "groom", "uncertainty": 0.9},
            {"index": 1, "predicted_class": "groom", "uncertainty": 0.8},
            {"index": 2, "predicted_class": "other", "uncertainty": 0.1},
        ],
    })
    stats = state.stats(threshold=0.5, min_segment_length=2,
                        label_at=lambda i: "groom" if i == 2 else None, n_frames=3)
    assert stats["total_frames"] == 3
    assert stats["above_threshold_frames"] == 2  # frames 0,1 form one segment >= len 2
    assert stats["n_segments"] == 1
    assert stats["misclassified_count"] == 1  # frame 2 labeled groom, predicted other
    assert stats["class_names"] == ["other", "groom"]


def test_al_state_stats_none_when_empty():
    assert AlState().stats(0.5) is None


# ------------------------------------------------------------------- panels
def test_pipeline_panel_construction_and_refresh():
    _app()
    from plugins.pipeline_panel import PipelinePanel

    class FakeCtx:
        def client(self): return None
        def video_path(self): return None
        def pose_path(self): return None
        def fps(self): return 30.0
        def bouts(self): return []
        def behaviors(self): return []
        def project_manager(self): return None
        def status(self, msg): pass
        def al_state(self): return None
        def current_frame(self): return 0
        def seek(self, frame): pass
        def set_al_threshold(self, t): pass
        def set_al_results(self, state): pass

    panel = PipelinePanel(FakeCtx())
    names = [panel.tabText(i) for i in range(panel.count())]
    assert names == ["Pose", "Train", "Predict", "Active Learning"]
    panel.refresh()  # must not raise with an empty project


def _main() -> int:
    _app()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            failures.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__}: {e}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURE(S)")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())
