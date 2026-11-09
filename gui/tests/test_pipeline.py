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


# ------------------------------------------------------------ pure helpers
def test_feature_and_bin_width_match_the_backend():
    assert pc.feature_width(1) == 0
    assert pc.feature_width(4) == 16          # nchoose2(4)*2 + 4
    assert pc.bin_width(30, 0.1) == 3          # frames2integ 30 -> 3 frames/row
    assert pc.bin_width(25, 0.1) == 2          # round(25) // 10
    assert pc.bin_width(5, 0.1) == 1           # never zero


def test_resolve_binarize_checklist_semantics():
    assert pc.resolve_binarize([]) == (None, None)
    assert pc.resolve_binarize(["groom"]) == ("groom", None)
    assert pc.resolve_binarize(["groom", "rear"]) == (None, ["groom", "rear"])


def test_al_label_mapping_from_model_config():
    assert pc.al_label_mapping(None) == {"binarize": None, "label_union": None}
    assert pc.al_label_mapping({"binarize": "groom"})["binarize"] == "groom"
    union = pc.al_label_mapping({"label_union": ["a", "b"]})
    assert union["label_union"] == {"positive": ["a", "b"]}


def test_binarized_misclassification_uses_the_mapping():
    """A labeled negative predicted as 'other' by a binary model is correct."""
    state = AlState()
    state.load_results({"class_names": ["groom", "other"],
                        "frames": [{"index": 0, "predicted_class": 1, "uncertainty": 0.1},
                                   {"index": 1, "predicted_class": 1, "uncertainty": 0.1}]},
                       **pc.al_label_mapping({"binarize": "groom"}))
    assert not state.is_misclassified(0, "rear")   # rear -> other: agrees
    assert state.is_misclassified(1, "groom")      # groom predicted other: wrong


def test_confidence_falls_back_to_one_minus_uncertainty():
    state = AlState()
    state.load_results({"class_names": ["a", "other"],
                        "frames": [{"index": 3, "predicted_class": 0, "uncertainty": 0.25}]})
    assert abs(state.confidence(3) - 0.75) < 1e-9
    assert state.confidence(4) is None


def test_format_train_event_lines():
    assert "macro F1 0.900" in pc.format_train_event(
        {"event": "iteration", "iteration": 2, "macro_f1": 0.9, "n_train": 40,
         "n_lowconf_remaining": 3}, max_iter=10)
    assert pc.format_train_event({"event": "done", "iterations": 3}) is None
    assert pc.format_train_event({"status": "error", "error": "boom"}) == "ERROR: boom"
    assert pc.format_train_event({"status": "starting", "message": "Loading"}) == "Loading"


def test_summarize_extraction_warns_on_other_and_tracking():
    lines = pc.summarize_extraction({
        "class_names": ["groom", "other"], "bin_width": 3, "frames2integ": 30,
        "videos": [{"id": "v", "n_rows": 10, "n_features": 16,
                    "class_counts": {"groom": 1, "other": 9}, "frames_skipped": 0,
                    "other_fraction": 0.95, "mean_rejected_fraction": 0.5}],
    })
    text = "\n".join(lines)
    assert "Exclude 'other'" in text
    assert "likelihood threshold" in text


def test_length_mismatch_tolerance():
    assert pc.length_mismatch(3000, 3000) is None
    assert pc.length_mismatch(3030, 3000) is None          # within 2%
    assert "more" in pc.length_mismatch(30000, 3000)
    assert "only" in pc.length_mismatch(1500, 3000)
    assert pc.length_mismatch(None, 3000) is None


def test_threshold_floor_warning():
    assert pc.threshold_floor_warning(0.5, 2) is not None   # binary never < 0.5
    assert pc.threshold_floor_warning(0.7, 2) is None
    assert pc.threshold_floor_warning(0.5, 3) is None


def test_keypoints_at_reads_column_major_payload():
    from plugins.pose_overlay import keypoints_at
    payload = {"start": 10, "points": [
        {"name": "nose", "x": [1.0, 2.0], "y": [3.0, 4.0], "likelihood": [0.9, 0.1]},
        {"name": "tail", "x": [5.0, 6.0], "y": [7.0, 8.0], "likelihood": [0.8, 0.8]},
    ]}
    assert keypoints_at(payload, 11) == [("nose", 2.0, 4.0, 0.1), ("tail", 6.0, 8.0, 0.8)]
    assert keypoints_at(payload, 9) == []      # before the window
    assert keypoints_at(payload, 12) == []     # past it
    assert keypoints_at(None, 0) == []


def test_trim_results_drops_frames_past_the_video():
    from plugins.active_learning_panel import trim_results
    res = {"class_names": ["a"], "frames": [{"index": i} for i in range(10)]}
    assert len(trim_results(res, 4)["frames"]) == 4
    assert len(trim_results(res, None)["frames"]) == 10


def test_bouts_signature_changes_on_retime_and_rename():
    from labeling_shell.bouts import Bout
    from plugins.pose_panel import bouts_signature
    base = bouts_signature("v", [Bout("1", "groom", 0, 10)])
    assert base == bouts_signature("v", [Bout("1", "groom", 0, 10)])
    assert base != bouts_signature("v", [Bout("1", "groom", 0, 11)])
    assert base != bouts_signature("v", [Bout("1", "rear", 0, 10)])


# ------------------------------------------------------------------- panels
class FakeCtx:
    def __init__(self, pm=None, behaviors=(), bouts=()):
        self.pm = pm
        self._behaviors = list(behaviors)
        self._bouts = list(bouts)
        self.current_video_id = None
        self.messages = []
        self.results = "unset"

    def client(self): return None
    def video_path(self): return None
    def pose_path(self): return None
    def fps(self): return 30.0
    def bouts(self): return self._bouts
    def behaviors(self): return self._behaviors
    def project_manager(self): return self.pm
    def status(self, msg): self.messages.append(msg)
    def al_state(self): return None
    def current_frame(self): return 0
    def seek(self, frame): pass
    def set_al_threshold(self, t): pass
    def set_al_results(self, state): self.results = state
    def video_frame_count(self): return None
    def save_labels(self): pass


def test_pipeline_panel_construction_and_refresh():
    _app()
    from plugins.pipeline_panel import PipelinePanel

    panel = PipelinePanel(FakeCtx())
    names = [panel.tabText(i) for i in range(panel.count())]
    assert names == ["Pose", "Train", "Predict", "Active Learning"]
    panel.refresh()  # must not raise with an empty project
    panel.on_project_opened()
    panel.on_video_switched()


def test_pose_refresh_keeps_the_body_part_selection():
    """Regression: refresh() (run on every label edit) used to re-inspect the
    pose file and re-tick every body part."""
    _app()
    from PySide6 import QtCore
    from plugins.pose_panel import PoseTab

    tab = PoseTab(FakeCtx())
    tab._on_inspected({"bodyparts": ["nose", "ear", "tail"], "n_frames": 100})
    tab.bodyparts_list.item(2).setCheckState(QtCore.Qt.CheckState.Unchecked)
    assert tab.selected_bodyparts() == ["nose", "ear"]
    tab.refresh()
    assert tab.selected_bodyparts() == ["nose", "ear"]
    # ...and survives re-inspecting the same file (e.g. on a video switch).
    tab._on_inspected({"bodyparts": ["nose", "ear", "tail"], "n_frames": 100})
    assert tab.selected_bodyparts() == ["nose", "ear"]
    assert "2 parts → 4 features" in tab.feature_width_label.text()


def test_pose_restores_project_settings_and_iteration():
    _app()
    from plugins.pose_panel import PoseTab
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        pm = PoseProjectManager(d)
        pm.create("p")
        pm.update_preprocessing_config({"origin": "sleap", "llh_value": 0.8,
                                        "min_duration": 0.2, "bodyparts": ["ear"]})
        pm.set_current_iteration(3)
        pm.flush_manifest()
        tab = PoseTab(FakeCtx(pm))
        tab.on_project_opened()
        assert tab.origin.currentText() == "sleap"
        assert abs(tab.llh.value() - 0.8) < 1e-9
        assert tab.current_iteration_dir() == "iteration-3"
        tab._on_inspected({"bodyparts": ["nose", "ear"], "n_frames": 10})
        assert tab.selected_bodyparts() == ["ear"]


def test_pose_auto_advances_when_labels_change():
    _app()
    from plugins.pose_panel import PoseTab
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        pm = PoseProjectManager(d)
        pm.create("p")
        tab = PoseTab(FakeCtx(pm))
        assert tab.auto_advance_iteration("sig-A") is False      # fresh iteration
        pm.set_iteration_meta("iteration-0", {"label_signature": "sig-A"})
        assert tab.auto_advance_iteration("sig-A") is False      # same labels: redo
        assert tab.auto_advance_iteration("sig-B") is True       # new labels
        assert tab.current_iteration_dir() == "iteration-1"
        assert pm.get_current_iteration() == 1


def test_train_lists_only_videos_with_features_and_keeps_positives():
    _app()
    from labeling_shell.bouts import Behavior
    from PySide6 import QtCore
    from plugins.train_panel import TrainTab
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        pm = PoseProjectManager(d)
        pm.create("p")
        pm.register_video("a", "a.mp4", "/a.mp4", 10)
        pm.register_video("b", "b.mp4", "/b.mp4", 10)
        feats = os.path.join(d, "iteration-0", "features")
        os.makedirs(feats)
        open(os.path.join(feats, "a.npz"), "wb").close()
        ctx = FakeCtx(pm, behaviors=[Behavior("1", "groom", "1"), Behavior("2", "rear", "2")])
        tab = TrainTab(ctx)
        assert tab.selected_video_ids() == ["a"]

        tab.binarize_list.item(0).setCheckState(QtCore.Qt.CheckState.Checked)
        tab.refresh()                                  # a label edit
        assert tab.positive_behaviors() == ["groom"]
        cfg = tab.build_config()
        assert cfg["binarize"] == "groom" and cfg["label_union"] is None
        assert cfg["checkpoint_dir"].replace("\\", "/").endswith("iteration-0/models")
        assert "threshold" in tab.threshold_hint.text()  # binary @0.5 can't refine

        tab.binarize_list.item(1).setCheckState(QtCore.Qt.CheckState.Checked)
        cfg = tab.build_config()
        assert cfg["binarize"] is None and cfg["label_union"] == ["groom", "rear"]


def test_model_picker_keeps_choice_and_adds_paths():
    _app()
    from plugins.common import ModelPicker
    picker = ModelPicker(FakeCtx())
    assert picker.current_path() is None
    picker.set_entries([{"path": "/m/new.joblib"}, {"path": "/m/old.joblib"}])
    assert picker.current_path() == "/m/new.joblib"       # newest first
    picker.select_path("/m/old.joblib")
    picker.set_entries([{"path": "/m/newer.joblib"}, {"path": "/m/new.joblib"},
                        {"path": "/m/old.joblib"}])
    assert picker.current_path() == "/m/old.joblib"       # choice survives a rescan
    picker.add_path("/elsewhere/x.joblib", {"config": {"binarize": "g"}})
    assert picker.current_path() == "/elsewhere/x.joblib"
    assert picker.current_entry()["config"]["binarize"] == "g"


def test_al_panel_clears_or_restores_per_video():
    _app()
    from plugins.active_learning_panel import ActiveLearningPanel
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        pm = PoseProjectManager(d)
        pm.create("p")
        pm.save_predictions("v1", {"class_names": ["a", "other"],
                                   "frames": [{"index": 0, "predicted_class": 0,
                                               "uncertainty": 0.2}],
                                   "checkpoint_path": "/nope.joblib",
                                   "model_config": {"binarize": "a"}})
        ctx = FakeCtx(pm)
        panel = ActiveLearningPanel(ctx)
        ctx.current_video_id = "v2"
        panel.on_video_switched()
        assert ctx.results is None                       # nothing saved: cleared
        ctx.current_video_id = "v1"
        panel.on_video_switched()
        assert ctx.results is not None and ctx.results.binarize == "a"


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
