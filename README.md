# IntegratedASOiD

A desktop application for building **pose-based animal behavior classifiers** — from
hand-labeling video through training to active-learning refinement.

It pairs an Electron labeling UI with a Python (FastAPI + scikit-learn) backend that
turns pose-estimation files into spatio-temporal features, trains a RandomForest via
[A-SOID](https://github.com/YttriLab/A-SOID)'s active-learning loop, and returns
per-frame predictions with uncertainty.

This is a fork of **RealtimeASOID**, which used a TSM convolutional network over
cropped video frames. The labeling interface is inherited; the classifier is
A-SOID's. That trade buys a **CPU-only install, training in seconds instead of
hours, and a far smaller dependency stack** — at the cost of requiring pose
tracking (DeepLabCut / SLEAP / OpenMonkeyStudio) as input.

---

## What it does

A complete pipeline in one app, organized as five tabs:

1. **Label** — Load videos, define behaviors, and mark temporal *bouts* (start/end
   intervals) frame-by-frame, with a frame-accurate player.
2. **Pose** — Attach each video's pose file, pick which animals and body parts to
   use, set the likelihood threshold and minimum behavior duration, and extract
   features.
3. **Train** — RandomForest with active learning: seed on a small fraction of each
   class, then repeatedly add the samples the model is *least sure about* until it
   stops being confused. Per-iteration F1 streams live.
4. **Active Learning** — Run inference, see an uncertainty heatmap and
   misclassification markers on the timeline, jump to the frames most worth
   relabeling, and fix them in place. Optionally overlay the pose skeleton on the
   video to see whether a bad prediction is really a tracking failure.
5. **Predict** — Apply a trained model to unlabeled pose files in batch.

The point of step 3 is that a small, well-chosen subset of labels usually matches or
beats training on everything — and step 4 tells you which frames those are.

---

## Requirements

- **[pixi](https://pixi.sh)** — manages the entire Python side from a single
  lockfile. Install once:
  - Windows (PowerShell): `iwr -useb https://pixi.sh/install.ps1 | iex`
  - macOS/Linux: `curl -fsSL https://pixi.sh/install.sh | bash`
- **Node.js** 18+ (with `npm`) — for the Electron GUI.
- **No GPU.** The classifier is CPU-only scikit-learn; the environment installs and
  runs identically on any machine.

> You don't install Python, scikit-learn, or ffmpeg yourself — `pixi` provisions a
> pinned, project-local environment from `pixi.toml`, and the app discovers it
> automatically.

---

## Installation

```bash
# 1. Python environment (creates .pixi/envs/default from pixi.toml). Run from repo root.
pixi install

# 2. Electron app dependencies
cd LabelingApp
npm install
```

---

## Running

```bash
cd LabelingApp
npm start
```

Electron starts the backend automatically (`uvicorn backend.server:app --port 8741`)
and shuts it down on exit. The header shows **Backend: Connected** once it's up.

Standalone backend: `pixi run backend`. Tests: `pixi run test`.

---

## Input data

Pose files, one per video:

| Software | Format | Notes |
|---|---|---|
| DeepLabCut | `.csv` | single- or multi-animal |
| SLEAP | `.h5` | analysis export; nodes sorted alphabetically |
| OpenMonkeyStudio | `.mat` / `.txt` | 3D, single animal |

Videos are for viewing and labeling only — no pixels reach the classifier.

---

## Project layout

Projects are self-contained directories with a `project.json` manifest:

```
my_experiment/
├── project.json              # videos, pose paths, behaviors, config
├── labels/                   # frame-indexed one-hot label CSVs
├── iteration-N/features/     # extracted features per video
├── predictions/              # saved inference results
└── models/checkpoints/       # trained models (.joblib)
```

Video and pose paths are absolute (files stay where they are); everything else is
relative to the project root. Projects export to a single HDF5 `.rlt` file in two
flavors — lightweight (labels only) or package (everything, including the pose files
themselves, so it trains anywhere).

---

## Iterating

The human-in-the-loop cycle:

1. Label a modest amount of data.
2. Extract features and train.
3. In **Active Learning**, jump to the least-confident frames and fix the labels.
4. Bump **Iteration** in the Pose tab, re-extract, and retrain.

Each iteration keeps its own feature set under `iteration-N/`, so you can compare
rounds rather than overwriting your history.

---

## Relationship to upstream A-SOID

The numeric core (`backend/asoid_core/`) is vendored from
[YttriLab/A-SOID](https://github.com/YttriLab/A-SOID) with Streamlit removed. The 2D
feature extraction and the active-learning loop are arithmetically identical to
upstream, so results are comparable with the reference Streamlit app. Three upstream
issues were corrected during the port — a vestigial scaler, a `np.int` crash in the
3D path, and a body-part stride bug in the same path. See `CLAUDE.md` for details.
