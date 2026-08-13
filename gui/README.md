# IntegratedASOiD desktop GUI (Python)

The all-Python successor to the Electron `LabelingApp/` shell: PySide6 (Qt) +
PyAV (ffmpeg/libav) + NumPy. Talks to the existing FastAPI backend
(pose feature extraction + RandomForest active learning) over HTTP/SSE via
`BackendClient`, either spawning it locally from the project's pixi env or
connecting to a remote one. No JavaScript, no Electron, no npm.

Ported from [RealtimeASOID](https://github.com/adeneagle/realtime-behavior-classifier)'s
own completed Python rewrite (`python-rewrite` branch) — same shell/plugin
split (one `MainWindow` owning everything, passing itself as a duck-typed
`ctx` to plain `QWidget` panels), swapped from the TSM/video pipeline to the
pose/RandomForest one.

## Run it

```bash
pixi run gui
```

The backend starts itself on launch (adopting one already on the port, if you
ran `pixi run backend` by hand); the Backend menu is only needed to retry or to
point at a remote one.

Keyboard: `1`-`0` toggle behavior bouts at the current frame, `Space`
play/pause, `Left`/`Right` step, `F` fullscreen. **Snap Back** in the transport
row jumps to the end of the last labeled bout — where you left off.

## Directory layout

```
app.py                        MainWindow, entry point — the pipeline ctx
labeling_shell/                shell, copied from RealtimeASOID-python (generic):
  bouts.py, csv_io.py, project.py    label model, CSV I/O, on-disk ProjectStore
  video_reader.py, video_widget.py   PyAV decoding + Qt video display/transport
  timeline_widget.py, active_learning.py   uncertainty timeline, AlState
  theme.py, theme.qss                Qt dark theme
  backend/{client,manager,process,config}.py   HTTP/SSE seam to backend/
  backend_worker.py                  CallWorker/StreamWorker (Qt threads)
plugins/                       the pose/RF pipeline (app-specific):
  pose_panel.py                 Pose tab — attach/inspect pose, bodyparts,
                                 extract features, iteration bookkeeping
  train_panel.py                Train tab — active-learning hyperparameters
  predict_panel.py              Predict tab — batch inference
  active_learning_panel.py      run inference on the current video, review
  pipeline_config.py            pure request-body builders (Qt-free)
  pipeline_panel.py             QTabWidget container wiring the four above
  project_manager.py            PoseProjectManager — pose/model/iteration
                                 bookkeeping on top of ProjectStore
  pose_overlay.py                pose-skeleton dots painted over the video
packaging/integratedasoid.spec  PyInstaller spec
tests/                          run headless via QT_QPA_PLATFORM=offscreen
tools/index_video.py            synthetic frame-indexed test videos
```

## Workflow

1. **Label** — load a video, define behaviors, mark temporal bouts.
2. **Pose** — attach the video's pose file, pick body parts/animals, set
   likelihood threshold and minimum behavior duration, extract features.
3. **Train** — RandomForest + active learning: seed on `init_ratio` of each
   class, then repeatedly add the samples the model is least sure about
   until it stops being confused. Saves a `.joblib` bundle.
4. **Active Learning** — run inference on the current video's pose file, get
   an uncertainty heatmap on the timeline, jump to the least-confident
   frames, relabel in place.
5. **Predict** — apply a model to unlabeled pose files in batch.

Iterate by bumping the Iteration counter on the Pose tab: re-extract from the
refined labels and retrain.

## Dev environment

The frontend is its own pixi environment (`gui`, defined in `pixi.toml` next
to the backend's `default`) so it never drags in scikit-learn/numba — it
installs and runs even on a machine that only ever talks to a remote backend.
One lockfile, one command:

```bash
pixi install -e gui
```

`pixi run gui` and `pixi run -e gui <anything>` (tests, pyinstaller, …) then
use this environment automatically. `ASOID_PYTHON` (or `PYTHON`) overrides
which interpreter the app spawns the *backend* with; otherwise it resolves
the project-local pixi env (`.pixi/envs/default/python.exe`), then a bare
`python`.

## Tests

```bash
pixi run -e gui python gui/run_all_tests.py
```

Runs every `tests/test_*.py` suite headless (`QT_QPA_PLATFORM=offscreen`,
synthetic videos via `tools/index_video.py`). Live pose extraction/training/
inference (needing the backend's pixi/sklearn env) are not exercised by
these — those paths are validated only at the contract level (stub HTTP
server + config-dict building in `test_pipeline.py` / `test_backend_client.py`).

## Known gaps vs. the Electron app

- `.rlt` single-file project export/import: `PoseProjectManager.export_rlt`/
  `import_rlt`/`inspect_rlt` call the backend's existing `rlt_service`
  endpoints, but there's no menu wiring for them yet.

The splitter width (`QSettings` "splitterState") and the Active Learning stats
grid (`AlState.stats`/`uncertain_segments` → `active_learning_panel`) were both
listed here as gaps but are implemented; RealtimeASOID has since dropped its own
copy of the stats code, so this shell is ahead of it there — don't "re-sync" it
away.
