# CLAUDE.md

## Workflow

Push commits to `origin` automatically after creating them — no need to ask
for confirmation first.

## Project Overview

Pose-based animal behavior classifier with active learning. A FastAPI server
extracts spatio-temporal features from pose-estimation files (DeepLabCut /
SLEAP / OpenMonkeyStudio), trains a RandomForest through A-SOID's
active-learning loop, and returns per-frame predictions with uncertainty.

Two UI shells consume that same backend:

- **`gui/` (PySide6, Python)** — the current, actively developed shell, on
  branch `python-rewrite`. Port of RealtimeASOID's completed Python rewrite
  (`RealtimeASOID-python/python-rewrite` branch) to the pose/RF pipeline. No
  Electron, no npm. See [gui/README.md](gui/README.md) for architecture and
  `pixi run gui` to run it.
- **`LabelingApp/` (Electron)** — the original shell, kept on `main` as a
  restore point (mirrors how RealtimeASOID kept its Electron baseline while
  porting). Not being extended further; new work goes into `gui/`.

**This is a fork of RealtimeASOID** with the TSM convolutional network replaced
by [A-SOID](https://github.com/YttriLab/A-SOID)'s random-forest + active-learning
procedure. The labeling UI is inherited essentially intact; the classifier,
feature pipeline and dependency stack are entirely different.

### What changed from RealtimeASOID

| | RealtimeASOID | IntegratedASOiD |
|---|---|---|
| Input | video pixels | pose files (x/y/likelihood per body part) |
| Preprocess | background subtraction + crop + PNG frame dump | likelihood filter + binned pairwise features |
| Model | TSM / EfficientNet (PyTorch Lightning, GPU) | RandomForest (scikit-learn, CPU) |
| Training time | hours | seconds |
| Deps | torch, lightning, timm, albumentations, wandb | scikit-learn, numba, joblib |
| Checkpoint | `.ckpt` / TorchScript `.pt` | `.joblib` bundle |

The Electron **Active Learning panel carries over unchanged** — the backend
returns the identical `{frames: [{index, predicted_class, uncertainty}],
class_names}` payload, so the uncertainty heatmap, misclassification ticks,
jump-to-worst-frame and relabel strip all work as before.

## Project Structure

```
├── LabelingApp/           # Electron app. Four files are ours; the rest is the
│   │                      # behavior-labeler package (see the section below).
│   ├── app.js             # OURS — which application this is
│   ├── app.config.json    # OURS — window title, project folder layout
│   ├── plugins/pipeline.js# OURS — the whole A-SOID pipeline: Pose/Train/Predict
│   ├── project-manager.js # OURS — project manifest (schema is pipeline-specific)
│   ├── index.html         # shell — names no application and no pipeline
│   ├── core/              # shell — 18 classic scripts, loaded in order
│   ├── labeler-plugin.js  # shell — plugin registry: the shell/pipeline seam
│   ├── main.js            # shell — main process; spawns the Python backend
│   ├── preload.js         # shell — IPC bridge (contextBridge)
│   └── tools/             # shell — `npm run check` invariants
├── backend/
│   ├── server.py          # App init + CORS
│   ├── asoid_core/        # Vendored A-SOID numerics (Streamlit-free)
│   │   ├── features.py        # numba feature extraction (2D + 3D)
│   │   ├── pose_io.py         # DLC/SLEAP/OMS loaders + adp_filt
│   │   ├── predict.py         # RF prediction, frame-shifted
│   │   ├── active_learning.py # RFActiveLearner (the loop)
│   │   └── indexing.py        # feature-row <-> native-frame mapping
│   ├── routers/           # pose, training, inference, project
│   └── services/          # pose, training, inference, rlt
└── tests/                 # pytest suite (145 tests)
```

## Running the App

```bash
pixi install
pixi run gui
```

`pixi install` materializes both environments defined in `pixi.toml`: `default`
(backend, scikit-learn/FastAPI) and `gui` (PySide6 frontend) — one lockfile,
no separate venv. The GUI spawns `uvicorn backend.server:app --port 8741`
locally and tears it down on exit; `gui/labeling_shell/backend/process.py::resolve_python()`
resolves the interpreter: **`ASOID_PYTHON`/`PYTHON` override → project-local
pixi env (`.pixi/envs/default/python.exe`) → bare `python`**. `GET /api/health`
reports the interpreter plus numpy/sklearn/numba versions.

**Spawning that interpreter is not the same as activating its environment.**
`process.py::activation_path()` prepends the env's own directories (notably
`<env>\Library\bin`) to the child's PATH before spawning. Skip that on Windows
and conda-forge numpy/scipy cannot resolve the MKL BLAS/LAPACK DLLs — imports
still succeed, and the process then dies with a bare `0xc06d007f`
(DELAYLOAD_DLL_NOT_FOUND) and no traceback the moment something calls LAPACK.
In practice that is `asoid_core.features.angle_between`'s `np.linalg.det`, so
the symptom is the backend vanishing part-way through Extract Features.
`pixi run test` cannot catch this class of bug, because pixi has already
activated the env.

Standalone backend: `pixi run backend`. Tests: `pixi run test` (backend),
`pixi run -e gui python gui/run_all_tests.py` (frontend, headless).

The legacy Electron shell still runs via `cd LabelingApp && npm install && npm start`,
but new work goes into `gui/` — see the shells comparison above.

There is **no GPU dependency** — the environment installs and runs identically
on any machine.

## Workflow Pipeline

1. **Label** — load a video, define behaviors (keys 1-9, 0), mark temporal bouts.
2. **Pose** — attach the video's pose file, pick animals/body parts, set the
   likelihood threshold and minimum behavior duration, extract features into
   `iteration-N/features/<video_id>.npz`.
3. **Train** — RandomForest + active learning: seed on `init_ratio` of each
   class, then repeatedly add the samples the model is least sure about
   (`predict_proba.max(1) < conf_threshold`) until it stops being confused.
   Per-iteration F1 streams into `#trainLog`. Saves a `.joblib` bundle.
4. **Active Learning** — run inference on the pose file, get an uncertainty
   heatmap on the timeline, jump to the least-confident frames, relabel in place.
5. **Predict** — apply the model to unlabeled pose files in batch.

Iterate by bumping the **Iteration** counter in the Pose tab: re-extract from
the refined labels and retrain (A-SOID's `E_create_new_training`).

## Critical invariants

**Frame indexing.** The UI indexes everything by native video frame. A-SOID's
extractor works in *bins* of `bin_width = frames2integ // 10` native frames,
where `frames2integ = round(framerate * min_duration / 0.1)`. All conversion
lives in `asoid_core/indexing.py` — get it wrong and nothing errors, the
heatmap just points at the wrong part of the video. It is unit-tested.

**Body-part selection is part of the model.** Features are built from every
*pair* of selected body parts (`nchoose2(n)*2 + n` columns), and column order
follows the pose file's schema restricted to the selection. A model is only
valid for the exact selection it was trained on. Checkpoints carry a
`bodypart_fingerprint`; `inference_service.check_bodyparts` rejects a mismatch
loudly. (Upstream A-SOID hit this — commit 1710d15.)

**Pose and video must cover the same frames.** Features, labels and predictions
are all indexed by frame number with no offset, so attaching a pose file that
tracks a longer recording than the loaded video (a full session's `.h5` against
a short review clip) breaks silently: inference runs over the whole pose file
and the uncertainty timeline squeezes hours into a few minutes of trackbar.
`inspectPoseFile` warns on any length mismatch, "entire video" scope clamps to
`videoFrameCount()`, and `restorePredictions` trims out-of-range frames from
older saved prediction files (in memory only — the file keeps its full data).
Feature extraction takes `max_frames` per video (the video's frame count,
recorded on the manifest the first time a video is opened) for the same reason.

Note the boundary is the **video length**, not the last label: footage that was
watched and deliberately left unlabeled is legitimate "other", so truncating at
the final label would discard real negatives. `limit_to_labeled` exists as a
cruder fallback but is off by default. And this only skews *training* when
"other" is trained on at all — with **Exclude other** ticked (`exclude_other`),
`RFActiveLearner` drops those rows entirely and the model is unaffected either
way; the bound still saves extraction time and disk.

**Labels are sparse on disk, dense in training.** `generateTrainingLabelCsv`
writes a row only for in-bout frames. `pose_service.dense_targets_from_csv`
expands that to one class per frame with every unlabeled frame set to the *last*
class, `other`. "Other" is trained on but never scored — it means "unlabeled",
so counting it would inflate F1.

## The labeling shell is upstream — don't edit it here

`LabelingApp/` used to be a 6,000-line `renderer.js` maintained as two drifting
copies, one here and one in RealtimeASOID. The shell is now its own repo,
`behavior-labeler`, which both projects consume and which also runs
standalone as a bout labeler with no model behind it.

**Only four files in `LabelingApp/` belong to this project**: `app.js`,
`app.config.json`, `plugins/pipeline.js`, `project-manager.js`. Everything else
is synced in. Editing a synced file here is the mistake to avoid — the next
`npm run shell:sync` overwrites it silently. Fix it in the shell repo, then:

```bash
npm run verify        # sync from the package, then run the checks
```

The synced files are committed rather than gitignored, so an accidental local
edit shows up as a diff in review.

### The rule that keeps this working

**The shell must not name pipeline-specific code.** The video player, behaviors,
bouts, CSV, project manifest, timeline overlays, relabel loop, HUD and layout
make **zero** backend calls — labeling works with the backend down, which is
what makes the standalone build real rather than aspirational. Everything about
pose files, feature extraction, RandomForests or checkpoints reaches them only
through `window.labelerPlugins`:

```js
mountPanels([{ tab: "pose", label: "Pose", after: "label", html: `…` }]);
mountSlot("alPipelineOptions", `…`);

window.labelerPlugins.register({
  id: "asoid-pose",
  hooks:     { onVideoSwitched, onTabShown, onProjectOpened },
  overlays:  [{ id, draw(ctx2d, ctx) }],
  models:    { label: "Models", extensions: ["joblib"] },
  inference: { buildRequest(ctx), validate(ctx) },
  settings:  { save(), load(saved) },
});
```

`ctx` comes from `makeContext()` — video name/path, frame index, frame count,
fps, bouts, behaviors, project manager, `setStatus`. Deliberately a snapshot,
not a handle on the shell's internals. Hooks that throw are caught and logged,
so a broken pipeline can never stop someone labeling.

So: **adding a pipeline feature means adding a hook or capability, not a call
from the shell into the panel.** Wanting `core/` to call something in
`plugins/pipeline.js` is the signal that a hook is missing. `npm run check`
enforces it, along with two other things `node --check` cannot see: that nothing
a script runs at load time is declared later in the load order (hoisting stops
at each `<script>` boundary — this is what `core/boot.js` exists for), and that
`project-manager.js` defines every method `core/` calls.

**What makes the two pipelines interchangeable** is one contract: inference
resolves to `{frames: [{index, predicted_class, uncertainty}], class_names}`
indexed by **native video frame**. That is why the entire Active Learning panel
is shared code rather than a second copy, and why the shell owns the uncertainty
heatmap, misclassification ticks, jump navigation and HUD without knowing what
kind of model ran. Don't break that shape.

## Deviations from upstream A-SOID

Documented inline as `UPSTREAM FIX` comments in `backend/asoid_core/`:

1. **The `StandardScaler` is dropped.** Upstream fits one in
   `feature_extraction()` then discards it — it saves and trains on *unscaled*
   features and predicts via `bsoid_predict_numba_noscale`. It was vestigial;
   a RandomForest is scale-invariant anyway.
2. **`np.int` → `int`** in the 3D path (`extract_features_3D.py:187`), removed
   in numpy ≥ 1.24. The 3D path could not run at all on modern numpy.
3. **3D body-part stride fixed** from `index * 2` to `index * 3`. 3D pose stores
   three coordinates per body part, so scaling an index by 2 made the kernels
   read misaligned slices. Combined with (2), upstream's 3D path cannot have
   produced a working model, so there was no numerical compatibility to preserve.
4. **The active-learning seed takes at least one sample per class**
   (`active_learning.py`). Upstream's `int(n * init_ratio)` is 0 for any class
   with fewer than `1/init_ratio` training rows — under 100 at the default
   0.01, which is typical for hand-labeled behaviors. Those classes were left
   out of the seed entirely: training either failed ("seeds only one class") or
   seeded on nearly all "other", which confidently predicted "other" for every
   behavior sample, found nothing below `conf_threshold`, and "converged" at
   round 0 with F1 ≈ 0. Only the cases where upstream dropped a class differ.

The 2D path is otherwise arithmetically identical to upstream, so F1
trajectories are comparable with the reference Streamlit app.

## Key Patterns

- **Task registries**: services use module-level `_tasks` dicts with
  `status`/`progress`/`total`/`cancel`, cleared between tests by an `autouse`
  fixture. Training cancels cooperatively (a flag the AL loop checks between
  iterations) rather than by killing a subprocess.
- **Metrics streaming**: training appends JSON-lines events; the SSE endpoint
  replays *every* line in order (`read_all_metrics`) so nothing written between
  polls is lost.
- **Behavior names**: normalized (trim + collapse whitespace, case-insensitive)
  at every create/import/rename site; `mergeDuplicateBehaviors()` collapses
  duplicates on load.
- **Binarize**: a checklist of behaviors — 0 ticked = multi-class, 1 = binarize
  that behavior, ≥2 = union into one positive class. Handled by
  `training_service._collapse_labels`, mirroring the TSM app's semantics so
  `mapLabelToModelClass` in the renderer still applies (it finds "other" by
  name, so class order doesn't matter to it). The collapsed classes are
  `[<positive>, "other"]` — **"other" last**, like everywhere else, because
  `RFActiveLearner` treats the last code as "other". A binary run also passes
  `has_other=False`: its "other" is the negative class, not "unlabeled", so it
  must be scored and refined like the positive one (otherwise false positives
  are invisible to F1). "Exclude other" on a binary run drops the *unlabeled*
  rows before collapsing. With two classes the forest is always ≥ 0.5 sure, so
  `conf_threshold` must exceed 0.5 for the loop to refine at all — the Train
  tab warns.
- **MKV / non-MP4 containers**: the frame-accurate WebCodecs path demuxes with
  mp4box (ISO/MP4 only), so other containers are remuxed to MP4 first. The pose
  file is unaffected — only the in-renderer decoder cares.
- **On-video HUD**: `#videoHud` shows the current frame's predicted class, its
  confidence, and the human label directly over the video ("Info overlay"
  checkbox in the player controls). It's plain DOM, not canvas, so text stays
  sharp at any video resolution and needs no letterbox coordinate mapping.
  `currentFrameInfo()` is the single source for both it and the Active Learning
  panel's text readout, so the two can't drift. Confidence is `1 - uncertainty`
  (uncertainty being normalized entropy), and a red border flags a genuine
  prediction/label disagreement. Every `alState` read in `currentFrameInfo()`
  must stay inside its try/catch — `updateAlFrameInfo` runs during module
  evaluation, before `const alState` is initialized, so an unguarded touch is a
  TDZ crash that kills the whole renderer.
- **Pose overlay**: drawn on its own `#poseCanvas` above both playback paths,
  in the pose file's native pixel coordinates on a canvas sized to the video's
  natural resolution, so `object-fit: contain` aligns it for free. Deliberately
  shows *unfiltered* keypoints — the point is to reveal where tracking failed.
  Toggle it with the "Pose overlay" checkbox in the player controls, next to
  "Pause on label end". The redraw loop (`poseOverlayTick`) fires on a frame
  change *or* a canvas-size change, not frame change alone — otherwise
  toggling it on a paused, freshly-loaded video (before its true dimensions are
  known) can leave a wrong-sized, misaligned overlay stuck until the user scrubs.
- **Frame rate is read from the video, not typed in.** `#poseFramerate` mirrors
  the player's `#fpsInput` (auto-detected from the video's own container timing
  in WebCodecs mode, e.g. mp4box sample timescale/duration) via
  `syncPoseFramerateFromVideo()`, called on every video load. It stops
  overwriting the field the moment the user edits it directly
  (`poseState.framerateManual`, persisted across sessions) — pose is estimated
  on the same video, so a manual number is only needed if the pose file used a
  different (typically subsampled) rate.

## Project Directory Structure

```
my_experiment/
├── project.json              # manifest (videos, pose paths, behaviors, config)
├── labels/                   # frame-indexed one-hot label CSVs per video
├── iteration-N/
│   ├── features/             # extracted features (.npz) per video
│   └── models/               # models trained from those features (.joblib)
└── predictions/              # saved AL inference results (.json)
```

**Iterations are the unit of "one pass round the loop"** — the labels as they
stood, the features built from them, and the model trained on those features,
kept together so *which model is newest, and what was it trained on* is
answerable from the layout. `autoAdvanceIteration()` bumps to a new one when
Extract Features runs and `labelSignature()` differs from the signature stored
on the current iteration (bout count, covered frames, and a hash of
span+behavior — so retiming or renaming counts, not just adding). Re-extracting
with unchanged labels is treated as a redo and stays put. The active iteration
lives on the manifest (`current_iteration`), so reopening a project doesn't
start overwriting `iteration-0`.

Checkpoints record their `iteration`, `trained_at` and `video_ids`;
`list_checkpoints` walks recursively (so pre-iteration `models/checkpoints/`
models are still found), reports the iteration, and sorts newest first. "Use
newest" in the Active Learning tab takes the top of that list.

Video and pose paths are **absolute** (files stay where they are); everything
else is relative to the project root.

## API Endpoints

| Route | Method | Purpose |
|-------|--------|---------|
| `/api/health` | GET | Health check (used by Electron startup) |
| `/api/pose/inspect` | POST | Animals, body parts, frame count, 2D/3D |
| `/api/pose/keypoints` | POST | Per-frame keypoints for the video overlay |
| `/api/pose/extract-features` | POST | Start feature extraction |
| `/api/pose/status/{id}` | GET | Extraction progress (SSE) |
| `/api/pose/results/{id}` | GET | Per-video row/class counts |
| `/api/pose/stop/{id}` | POST | Cancel extraction |
| `/api/training/start` | POST | Start the active-learning loop |
| `/api/training/status/{id}` | GET | SSE stream of per-iteration metrics |
| `/api/training/stop/{id}` | POST | Stop training |
| `/api/training/checkpoints` | GET | List `.joblib` models + their metadata |
| `/api/inference/run` | POST | Per-frame inference on a pose file |
| `/api/inference/batch` | POST | Batch predict over several pose files |
| `/api/inference/status/{id}` | GET | Inference progress (SSE) |
| `/api/inference/results/{id}` | GET | Predictions + uncertainty |
| `/api/inference/stop/{id}` | POST | Cancel inference |
| `/api/project/export-rlt` | POST | Pack project into .rlt |
| `/api/project/import-rlt` | POST | Unpack .rlt |
| `/api/project/inspect-rlt` | POST | Read .rlt metadata |

## .rlt File Format

HDF5 project bundles, version **5**. Two flavors: lightweight (manifest +
labels) and package (`package=True`, adds pose files, features, predictions and
models). Packages are far smaller than the TSM predecessor's, which embedded a
PNG per labeled frame.

```
/                                (attrs: rlt_version=5, packaged)
/                                (attr: manifest — JSON string)
/labels/{video_id}               (UTF-8 CSV)
/pose/{video_id}                 (uint8 file bytes, gzip)   [package]
/features/{iteration}/{video_id} (uint8 .npz bytes, gzip)   [package]
/predictions/{video_id}          (UTF-8 JSON)               [package]
/models/checkpoints/{name}       (uint8 .joblib blob)       [package]
```

On import, **pose paths are rewritten** to the extracted copies (so a package is
immediately trainable), while video paths are left for the user to re-link.
Version < 5 files are rejected — they carry cropped frames but no pose data.

## CSV Formats

Import auto-detects two formats. Bout CSV (`start,end,behavior` with MM:SS.mmm
timestamps) and the frame-level one-hot matrix (`time` + one column per
behavior, native frame index). Predictions export in the latter format, so a
batch result re-imports as labels for review.

## Running Tests

```bash
pixi run test                              # all 145
pixi run pytest tests/test_asoid_core.py   # single module
```

Tests use synthetic pose data (two behaviors that differ in how far the body
parts spread from the centroid) — a genuinely learnable task, so the
active-learning tests assert on real F1 movement rather than mocks.

## Dependencies

`pixi.toml` is authoritative (`backend/requirements.txt` is a pip fallback).

- **Python**: FastAPI, Uvicorn, scikit-learn (<1.3, matching upstream A-SOID so
  models load in both apps), numba, joblib, numpy, pandas, scipy, h5py, OpenCV,
  ffmpeg, matplotlib
- **Node**: Electron 31, electron-builder, mp4box, xlsx
- **Test**: pytest, pytest-asyncio, httpx
