# Extracting the labeling shell

Goal: `LabelingApp/` becomes its own repo that is **both** a runnable standalone
bout labeler **and** an npm-installable UI core, consumed by ASOiD-Desktop
(pose → RandomForest) and RealtimeASOID (video → TSM).

## Why this is tractable

Measured between the two apps as they stand:

| File | Lines | Differing |
|---|---|---|
| `preload.js` | 146 | **0** |
| `main.js` | 568 | 11 |
| `styles.css` | 1,509 | 23 |
| `project-manager.js` | 558 | 139 |
| `renderer.js` | 5,931 | 1,948 — but **91 of 125 hunks in one region** |

The divergence is concentrated in the pipeline tabs (Pose vs Preprocess, the
Train fields, Predict). The ~3,200-line shell — player, frame indexing,
behaviors, bouts, CSV, project manifest, timeline overlays, relabel loop, HUD,
layout — makes **zero** backend calls. Labeling already works with the backend
down, so a standalone build is viable today.

## What makes it work: one contract

Inference must resolve to

```json
{ "frames": [ { "index": 0, "predicted_class": 2, "uncertainty": 0.31 } ],
  "class_names": ["Rear", "Groom", "other"] }
```

indexed by **native video frame**. Both pipelines already produce exactly this —
it is why the whole Active Learning panel ported between them unchanged — so the
shell owns the uncertainty heatmap, misclassification ticks, jump navigation and
the on-video HUD without knowing what kind of model ran.

## Status

**Done — the seam holds.** `labeler-plugin.js` defines the registry; the shell
reaches the pipeline only through it. No core code names a pose function
(verified: no `poseState` / `resetPoseForVideo` / `getPosePathFor` reference
below the plugin boundary). The A-SOID pipeline registers itself as
`asoid-pose` with hooks, an overlay and an inference request builder. A plugin
that throws is caught and logged rather than taking the shell down.

**Next, in order:**

1. **Move the pipeline code into its own module.** The pose panel functions
   still physically live in `renderer.js`; they just aren't reachable from the
   core any more. Cutting them into `plugins/asoid-pose.js` is now mechanical.
2. **Split the shell into `core/` modules** — `player.js`, `bouts.js`,
   `behaviors.js`, `project.js`, `timeline.js`, `hud.js`, `layout.js`,
   `settings.js`. This is the real work: the shell currently relies on
   module-scope globals (`bouts`, `behaviors`, `alState`), so it means threading
   the context object that `makeContext()` already defines. Worth doing beyond
   tidiness — the TDZ crash fixed in `currentFrameInfo()` (an `alState` read
   during module evaluation, before `const alState` initialises) is exactly the
   failure mode modules remove.
3. **Re-express RealtimeASOID's panels as a plugin** against the same shell.
   Its `preload.js` is already byte-identical and `main.js`/`styles.css` are
   within ~20 lines, so only the panel region needs writing.
4. **`git subtree split`** the `LabelingApp/` path to carve out a repo *with*
   its file history, then have both projects depend on it
   (`npm i github:<owner>/<repo>#v1.0.0`).

Chosen over submodules (collaborators clone without `--recursive` and get an
empty directory) and subtree-in-place (pushing changes back is fiddly).
A monorepo remains the alternative if atomic cross-project changes matter more
than separate identities and releases.

## Plugin contract

```js
window.labelerPlugins.register({
  id: "asoid-pose",
  hooks:    { onVideoSwitched, onTabShown, onProjectOpened },
  overlays: [{ id, draw(ctx2d, ctx) }],
  inference:{ buildRequest(ctx), validate(ctx) },
});
```

`ctx` is a snapshot from `makeContext()` — video name/path, frame index, frame
count, fps, bouts, behaviors, project manager, `setStatus` — deliberately not a
handle on the shell's internals.
