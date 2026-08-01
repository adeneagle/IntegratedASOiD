// plugins/asoid-pose.js — The A-SOID pipeline: pose attachment, feature extraction, training, batch predict.
//
// Registered against the shell through window.labelerPlugins (see
// labeler-plugin.js). Nothing in core/ may name anything defined here.

// ── Panel markup ──────────────────────────────────────────────────────────
// The tabs that configure this pipeline. They live here rather than in
// index.html so the shell's page is the same file for every pipeline — and
// so deleting this plugin removes its UI with it. Mounted first, because the
// pose panel code below queries these elements while it loads.
mountPanels([
  {
    tab: "pose",
    label: "Pose",
    after: "label",
    html: `
<section class="panel-card">
  <h2>Pose &amp; Features</h2>

  <h3>Pose Source</h3>
  <div class="form-row">
    <label><span>Software:</span>
      <select id="poseOrigin">
        <option value="deeplabcut" selected>DeepLabCut</option>
        <option value="sleap">SLEAP</option>
        <option value="openmonkeystudio">OpenMonkeyStudio (3D)</option>
      </select>
    </label>
    <label class="checkbox-label">
      <input id="poseMultiAnimal" type="checkbox" />
      <span>Multi-animal</span>
    </label>
  </div>
  <div class="form-row">
    <button id="poseAttach">Attach Pose File to Video</button>
    <span id="posePath" class="path-display">No pose file for this video</span>
  </div>
  <p class="hint" id="poseInfo">Attach the pose file that matches the video loaded in the player. Each video keeps its own.</p>

  <h3>Individuals</h3>
  <div id="poseAnimalList" class="behavior-checklist"></div>
  <p class="hint" id="poseAnimalHint">Single-animal project — nothing to choose.</p>

  <h3>Body Parts</h3>
  <div class="form-row">
    <button id="poseSelectAllBp" class="ghost">Select all</button>
    <button id="poseSelectNoneBp" class="ghost">Select none</button>
    <span class="muted">Features use every <em>pair</em>, so cost grows quickly: <output id="poseFeatureWidth">—</output></span>
  </div>
  <div id="poseBodypartList" class="behavior-checklist"></div>
  <p class="hint">The selection is baked into the model — a trained model can only be applied to this exact set. Changing it requires re-extracting and retraining.</p>

  <h3>Parameters</h3>
  <div class="form-row">
    <label><span>Frame rate (fps):</span>
      <input id="poseFramerate" type="number" min="1" max="240" step="0.1" value="30" />
    </label>
    <button id="poseUseVideoFps" class="ghost" title="Reset to the video's detected frame rate">Use video FPS</button>
    <label><span>Likelihood threshold:</span>
      <input id="poseLlh" type="number" min="0" max="1" step="0.05" value="0.6" />
    </label>
  </div>
  <p class="hint">Frame rate is read from the loaded video (WebCodecs mode detects it from the file's own container timing — see FPS in the player controls) and kept in sync here automatically. Edit it directly only if the pose file was estimated at a different rate than the video.</p>
  <div class="form-row">
    <label><span>Min behavior duration (s):</span>
      <input id="poseMinDuration" type="number" min="0.05" max="5" step="0.05" value="0.1" />
    </label>
    <span class="muted">Bin: <output id="poseBinWidth">3</output> frames <span id="poseBinMs">(100 ms)</span></span>
  </div>
  <p class="hint">Points below the likelihood threshold hold their last confident position. The bin is the temporal resolution of both training and predictions.</p>

  <h3>Extract Features</h3>
  <div class="form-row">
    <label><span>Iteration:</span>
      <input id="poseIteration" type="number" min="0" step="1" value="0" />
    </label>
    <button id="poseNewIteration" class="ghost">New Iteration</button>
    <span class="muted">Refined labels &rarr; a fresh feature set + model.</span>
  </div>
  <div id="poseVideoChecklist" class="video-checklist"></div>
  <div class="form-row">
    <button id="poseExtract" class="primary">Extract Features</button>
    <button id="poseStopExtract" class="danger" disabled>Stop</button>
  </div>
  <div class="progress-area">
    <progress id="poseProgress" value="0" max="100"></progress>
    <span id="poseStatus">Idle</span>
  </div>
  <pre id="poseLog" class="train-log" style="display:none;"></pre>
</section>
`,
  },
  {
    tab: "train",
    label: "Train",
    after: "pose",
    html: `
<section class="panel-card">
  <h2>Training</h2>

  <h3>Data Sources</h3>
  <div id="trainVideoChecklist" class="video-checklist"></div>
  <p class="hint">Videos with extracted features. Run the Pose tab first if this list is empty.</p>

  <h3>Active Learning</h3>
  <div class="form-row">
    <label><span>Initial ratio:</span>
      <input id="trainInitRatio" type="number" value="0.01" min="0.001" max="1" step="0.005" />
    </label>
    <label><span>Confidence threshold:</span>
      <input id="trainConfThreshold" type="number" value="0.5" min="0.05" max="0.99" step="0.05" />
    </label>
  </div>
  <div class="form-row">
    <label><span>Max iterations:</span>
      <input id="trainMaxIter" type="number" value="100" min="1" max="1000" />
    </label>
    <label><span>Samples per iteration:</span>
      <input id="trainMaxSamplesIter" type="number" value="20" min="1" max="1000" />
    </label>
  </div>
  <div class="form-row">
    <label class="checkbox-label">
      <input id="trainExcludeOther" type="checkbox" />
      <span>Exclude "other" from training</span>
    </label>
  </div>
  <p class="hint">Start from <em>initial ratio</em> of each class, then repeatedly add the samples the model is least sure about (max probability below the threshold) until it stops being confused. "Other" is never scored — it means <em>unlabeled</em>, so counting it would inflate the result.</p>

  <h3>Binarize</h3>
  <div id="trainBinarizeList" class="behavior-checklist"></div>
  <p class="hint">Leave all unticked for multi-class. Tick one behavior to train it against everything else; tick several to merge them into a single positive class first.</p>

  <h3>Run</h3>
  <div class="form-row">
    <label><span>Model name:</span>
      <input id="trainModelName" type="text" value="rf_model" placeholder="rf_model" />
    </label>
  </div>
  <div class="form-row">
    <button id="trainStart" class="primary">Start Training</button>
    <button id="trainStop" class="danger" disabled>Stop</button>
  </div>
  <div class="progress-area">
    <progress id="trainProgress" value="0" max="100"></progress>
    <span id="trainStatus">Idle</span>
  </div>
  <div id="trainMetrics" class="metrics-display" style="display:none;">
    <div>Iteration: <span id="trainIteration">-</span> / <span id="trainTotalIters">-</span></div>
    <div>Macro F1: <span id="trainMacroF1">-</span></div>
    <div>Ceiling (all data): <span id="trainCeilingF1">-</span></div>
    <div>Training samples: <span id="trainNSamples">-</span></div>
    <div>Low-confidence remaining: <span id="trainLowConf">-</span></div>
    <div>Per-class F1: <span id="trainPerClassF1">-</span></div>
  </div>
  <pre id="trainLog" class="train-log" style="display:none;"></pre>
</section>
`,
  },
  {
    tab: "predict",
    label: "Predict",
    after: "active-learning",
    html: `
<section class="panel-card">
  <h2>Batch Predict</h2>
  <p class="hint">Apply a trained model to pose files that were never labeled. Results are written as frame-indexed one-hot CSVs — the same format the Label tab imports, so you can load one back in to review it.</p>

  <h3>Model</h3>
  <div class="form-row">
    <button id="predSelectCheckpoint">Select Model</button>
    <span id="predCheckpointPath" class="path-display">No model</span>
  </div>

  <h3>Pose Files</h3>
  <div class="form-row">
    <button id="predAddFiles">Add Pose Files</button>
    <button id="predClearFiles" class="ghost">Clear</button>
  </div>
  <div id="predFileList" class="data-sources"></div>

  <h3>Output</h3>
  <div class="form-row">
    <button id="predSelectOutDir">Choose Output Folder</button>
    <span id="predOutDir" class="path-display">predictions/ in the project</span>
  </div>

  <div class="form-row">
    <button id="predRun" class="primary">Run Batch Predict</button>
    <button id="predStop" class="danger" disabled>Stop</button>
  </div>
  <div class="progress-area">
    <progress id="predProgress" value="0" max="100"></progress>
    <span id="predStatus">Idle</span>
  </div>
  <pre id="predLog" class="train-log" style="display:none;"></pre>
</section>
`,
  },
]);

// The shell's Active Learning tab leaves a gap for whatever else a pipeline's
// inference needs. This one needs no extra controls — body parts and bin width
// are recorded on the checkpoint — so it only explains what will run.
mountSlot("alPipelineOptions",
  '<p class="hint">Inference runs on this video\'s pose file. Body parts and ' +
  'bin width come from the model itself, so they always match how it was ' +
  'trained.</p>');

// ── Pose Panel ────────────────────────────────────────────────────────────
// Replaces the TSM app's Preprocess panel. Where that one computed a
// background, tuned a crop and dumped PNG frames, this one just points at a
// pose file and picks which body parts feed the feature extractor.

const poseState = {
  origin: "deeplabcut",
  multiAnimal: false,
  bodyparts: [],            // available in the attached pose file
  selected: new Set(),      // body part names to extract features from
  animals: [],
  selectedAnimals: new Set(),
  poseByVideo: {},          // videoName -> pose path (mirrors the manifest)
  nFrames: 0,
  is3d: false,
  taskId: null,
  sseSource: null,
  extractedVideos: new Set(), // videoNames with features on disk this session
  // Overlay
  keypoints: null,          // { bodyparts, points: [{name,x[],y[],likelihood[]}] }
  keypointsVideo: null,     // which video `keypoints` belongs to
  // FPS: mirrors the player's #fpsInput (auto-detected from the video's own
  // container timing in WebCodecs mode) until the user edits this field
  // directly, since the pose file was estimated on this same video.
  framerateManual: false,
};

const poseEl = (id) => document.getElementById(id);

function getPosePathFor(videoName) {
  const pm = window.projectManager;
  if (pm && pm.isActive()) {
    const p = pm.getVideoPose(videoName);
    if (p) return p;
  }
  return poseState.poseByVideo[videoName] || null;
}

function setPosePathFor(videoName, posePath) {
  poseState.poseByVideo[videoName] = posePath;
  const pm = window.projectManager;
  if (pm && pm.isActive()) {
    try { pm.setVideoPose(videoName, posePath); } catch { /* not tracked yet */ }
  }
}

// nchoose2(n) * 2 + n — mirrors asoid_core.features.feature_width, so the UI can
// warn about the quadratic growth before the user commits to 30 body parts.
function poseFeatureWidth(n) {
  if (n < 2) return 0;
  return (n * (n - 1) / 2) * 2 + n;
}

function updatePoseDerivedDisplays() {
  const n = poseState.selected.size;
  const widthEl = poseEl("poseFeatureWidth");
  if (widthEl) {
    widthEl.textContent = n < 2
      ? "select at least 2"
      : `${n} parts → ${poseFeatureWidth(n)} features`;
  }

  const fps = parseFloat(poseEl("poseFramerate").value) || 30;
  const minDur = parseFloat(poseEl("poseMinDuration").value) || 0.1;
  // frames2integ = round(framerate * min_duration / 0.1); bin = frames2integ / 10
  const frames2integ = Math.round(fps * (minDur / 0.1));
  const binWidth = Math.max(1, Math.floor(frames2integ / 10));
  poseEl("poseBinWidth").textContent = binWidth;
  poseEl("poseBinMs").textContent = `(${Math.round((binWidth / fps) * 1000)} ms)`;
}

function renderPoseBodypartList() {
  const container = poseEl("poseBodypartList");
  if (!container) return;
  container.innerHTML = "";

  if (poseState.bodyparts.length === 0) {
    container.innerHTML = '<p class="muted">Attach a pose file to list its body parts.</p>';
    updatePoseDerivedDisplays();
    return;
  }

  for (const bp of poseState.bodyparts) {
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = poseState.selected.has(bp);
    cb.addEventListener("change", () => {
      if (cb.checked) poseState.selected.add(bp);
      else poseState.selected.delete(bp);
      updatePoseDerivedDisplays();
      invalidatePoseOverlay();
    });
    const span = document.createElement("span");
    span.textContent = bp;
    row.append(cb, span);
    container.appendChild(row);
  }
  updatePoseDerivedDisplays();
}

function renderPoseAnimalList() {
  const container = poseEl("poseAnimalList");
  const hint = poseEl("poseAnimalHint");
  if (!container) return;
  container.innerHTML = "";

  if (!poseState.multiAnimal || poseState.animals.length === 0) {
    if (hint) hint.style.display = "";
    return;
  }
  if (hint) hint.style.display = "none";

  for (const an of poseState.animals) {
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = poseState.selectedAnimals.has(an);
    cb.addEventListener("change", () => {
      if (cb.checked) poseState.selectedAnimals.add(an);
      else poseState.selectedAnimals.delete(an);
      invalidatePoseOverlay();
    });
    const span = document.createElement("span");
    span.textContent = an;
    row.append(cb, span);
    container.appendChild(row);
  }
}

// Ask the backend what's in a pose file and repopulate the pickers.
async function inspectPoseFile(posePath) {
  const info = await backendFetch("/api/pose/inspect", {
    method: "POST",
    body: JSON.stringify({
      pose_path: posePath,
      origin: poseState.origin,
      multi_animal: poseState.multiAnimal,
    }),
  });

  poseState.bodyparts = info.bodyparts || [];
  poseState.animals = info.animals || [];
  poseState.nFrames = info.n_frames || 0;
  poseState.is3d = !!info.is_3d;

  // Default to everything selected — deselecting is the rare, deliberate act.
  poseState.selected = new Set(poseState.bodyparts);
  poseState.selectedAnimals = new Set(poseState.animals);

  const infoEl = poseEl("poseInfo");
  infoEl.textContent =
    `${info.n_frames} frames · ${poseState.bodyparts.length} body parts` +
    (poseState.animals.length ? ` · ${poseState.animals.length} animals` : "") +
    (info.is_3d ? " · 3D" : "") +
    (info.has_likelihood ? "" : " · no likelihood column");

  // A pose file that doesn't cover the same frames as the video is the single
  // most damaging mistake here, and it fails *silently*: features, labels and
  // predictions are all indexed by frame number, so a 4-hour pose file attached
  // to a 6-minute clip yields predictions running 40x past the end of the video.
  // The uncertainty timeline then compresses the whole recording into the
  // visible clip and "jump to lowest confidence" seeks off the end.
  const vFrames = videoFrameCount();
  poseState.frameMismatch = null;
  if (vFrames > 0 && info.n_frames > 0) {
    const ratio = info.n_frames / vFrames;
    if (ratio > 1.02 || ratio < 0.98) {
      poseState.frameMismatch = { poseFrames: info.n_frames, videoFrames: vFrames };
      infoEl.textContent +=
        `  ⚠ the video has ${vFrames} frames — this pose file covers ` +
        `${ratio > 1 ? `${ratio.toFixed(1)}x more` : `only ${(ratio * 100).toFixed(0)}%`}. ` +
        `They must describe the same recording, aligned from frame 0.`;
      infoEl.classList.add("pose-warning");
      setStatus(
        `Pose/video length mismatch: pose has ${info.n_frames} frames, video has ${vFrames}.`
      );
    } else {
      infoEl.classList.remove("pose-warning");
    }
  } else {
    infoEl.classList.remove("pose-warning");
  }

  renderPoseAnimalList();
  renderPoseBodypartList();
  return info;
}

poseEl("poseOrigin").addEventListener("change", async () => {
  poseState.origin = poseEl("poseOrigin").value;
  // OpenMonkeyStudio is single-animal 3D by definition.
  if (poseState.origin === "openmonkeystudio") {
    poseState.multiAnimal = false;
    poseEl("poseMultiAnimal").checked = false;
  }
  const p = getPosePathFor(currentVideoName);
  if (p) { try { await inspectPoseFile(p); } catch (e) { setStatus(e.message); } }
});

poseEl("poseMultiAnimal").addEventListener("change", async () => {
  poseState.multiAnimal = poseEl("poseMultiAnimal").checked;
  const p = getPosePathFor(currentVideoName);
  if (p) { try { await inspectPoseFile(p); } catch (e) { setStatus(e.message); } }
});

poseEl("poseAttach").addEventListener("click", async () => {
  if (!window.api) return;
  if (!currentVideoName) { alert("Load a video first — the pose file attaches to it."); return; }
  const res = await window.api.selectFile({
    title: "Select Pose File",
    filters: [
      { name: "Pose files", extensions: ["csv", "h5", "mat", "txt"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (res.canceled) return;

  setPosePathFor(currentVideoName, res.path);
  setPathDisplay("posePath", res.path);
  try {
    await inspectPoseFile(res.path);
    invalidatePoseOverlay();
    setStatus(`Pose attached to ${currentVideoName}`);
  } catch (e) {
    poseEl("poseInfo").textContent = `Could not read that pose file: ${e.message}`;
  }
  renderPoseVideoChecklist();
  renderTrainVideoChecklist();
});

poseEl("poseSelectAllBp").addEventListener("click", () => {
  poseState.selected = new Set(poseState.bodyparts);
  renderPoseBodypartList();
  invalidatePoseOverlay();
});
poseEl("poseSelectNoneBp").addEventListener("click", () => {
  poseState.selected.clear();
  renderPoseBodypartList();
  invalidatePoseOverlay();
});
// Frame rate is read from the video, not typed in: the player's #fpsInput is
// already the app's single source of truth for it (auto-detected from the
// container's own timing in WebCodecs mode, e.g. 29.970 for NTSC-derived
// footage, and user-correctable there via FPS/Lock if detection is ever wrong).
// The pose file is estimated on this same video, so mirror that value here
// rather than asking the user to re-enter a number the app already knows —
// unless they've deliberately typed a different one into this field (rare:
// pose estimated at a subsampled rate), in which case stop overwriting it.
function syncPoseFramerateFromVideo() {
  if (poseState.framerateManual) return;
  const detected = parseFloat(fpsInput.value);
  if (Number.isFinite(detected) && detected > 0) {
    poseEl("poseFramerate").value = detected.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    updatePoseDerivedDisplays();
  }
}

poseEl("poseFramerate").addEventListener("input", () => {
  poseState.framerateManual = true;
  updatePoseDerivedDisplays();
});
poseEl("poseUseVideoFps").addEventListener("click", () => {
  poseState.framerateManual = false;
  syncPoseFramerateFromVideo();
  setStatus(`Frame rate reset to the video's detected ${poseEl("poseFramerate").value} fps.`);
});
fpsInput.addEventListener("input", syncPoseFramerateFromVideo);
poseEl("poseMinDuration").addEventListener("input", updatePoseDerivedDisplays);

// Videos eligible for extraction: those with a pose file attached.
function renderPoseVideoChecklist() {
  const container = poseEl("poseVideoChecklist");
  if (!container) return;
  container.innerHTML = "";

  const names = videoList.map((v) => v.name);
  if (names.length === 0) {
    container.innerHTML = '<p class="muted">No videos in this project.</p>';
    return;
  }

  for (const name of names) {
    const pose = getPosePathFor(name);
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!pose;
    cb.disabled = !pose;
    cb.dataset.video = name;
    const span = document.createElement("span");
    span.textContent = pose ? name : `${name} (no pose file)`;
    if (!pose) span.className = "muted";
    row.append(cb, span);
    container.appendChild(row);
  }
}

function selectedPoseVideos() {
  const container = poseEl("poseVideoChecklist");
  if (!container) return [];
  return [...container.querySelectorAll("input[type=checkbox]")]
    .filter((cb) => cb.checked && !cb.disabled)
    .map((cb) => cb.dataset.video);
}

function poseLog(line) {
  const el = poseEl("poseLog");
  if (!el) return;
  el.style.display = "block";
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

poseEl("poseExtract").addEventListener("click", async () => {
  const pm = window.projectManager;
  if (!pm || !pm.isActive()) { alert("Open a project first."); return; }
  if (poseState.selected.size < 2) {
    alert("Select at least two body parts — features are built from body-part pairs.");
    return;
  }

  const names = selectedPoseVideos();
  if (names.length === 0) { alert("No videos with an attached pose file are selected."); return; }

  // Labels must be on disk before the backend reads them.
  try { await saveProgress(); await pm.flushManifest(); } catch { /* best effort */ }

  const videos = names.map((name) => ({
    id: pm.getVideoIdFromName(name),
    pose_path: getPosePathFor(name),
    label_path: pm.getVideoLabelsAbs(name),
    // Bounds extraction to the footage this video actually shows. Known for any
    // video that has been opened at least once; null otherwise, in which case
    // the whole pose file is used (and the length warning covers the risk).
    max_frames: (name === currentVideoName ? videoFrameCount() : 0)
      || pm.getVideoFrameCount(name) || null,
  }));

  // Roll onto a new iteration if labeling has moved on since this one was built.
  const { iterationDir, advanced, signature } = autoAdvanceIteration(names);

  poseEl("poseLog").textContent = "";
  poseEl("poseExtract").disabled = true;
  poseEl("poseStopExtract").disabled = false;
  poseEl("poseStatus").textContent = "Starting...";
  if (advanced) {
    poseLog(`Labels changed since the last extraction — starting ${iterationDir}.`);
    setStatus(`New labels detected: starting ${iterationDir}.`);
  } else {
    poseLog(`Extracting into ${iterationDir}.`);
  }

  const params = {
    project_dir: pm.getProjectRoot(),
    videos,
    origin: poseState.origin,
    bodyparts: [...poseState.selected],
    framerate: parseFloat(poseEl("poseFramerate").value) || 30,
    min_duration: parseFloat(poseEl("poseMinDuration").value) || 0.1,
    llh_value: parseFloat(poseEl("poseLlh").value) || 0.6,
    animals: poseState.multiAnimal ? [...poseState.selectedAnimals] : null,
    multi_animal: poseState.multiAnimal,
    iteration_dir: iterationDir,
  };

  // Persist the settings that produced these features, so reopening the project
  // (or handing it to someone else) reproduces the exact feature space.
  try {
    pm.updatePreprocessingConfig({
      origin: params.origin,
      bodyparts: params.bodyparts,
      framerate: params.framerate,
      min_duration: params.min_duration,
      llh_value: params.llh_value,
      multi_animal: params.multi_animal,
      animals: params.animals,
    });
  } catch { /* manifest not ready */ }

  try {
    const res = await backendFetch("/api/pose/extract-features", {
      method: "POST",
      body: JSON.stringify(params),
    });
    poseState.taskId = res.task_id;

    poseState.sseSource = backendSSE(
      `/api/pose/status/${res.task_id}`,
      async (data) => {
        const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
        poseEl("poseProgress").value = pct;
        poseEl("poseStatus").textContent = `${data.status}: ${data.progress}/${data.total}`;

        if (data.status === "completed") {
          poseState.sseSource?.close();
          poseEl("poseExtract").disabled = false;
          poseEl("poseStopExtract").disabled = true;
          try {
            const results = await backendFetch(`/api/pose/results/${poseState.taskId}`);
            poseLog(`Classes: ${results.class_names.join(", ")}`);
            poseLog(`Bin width: ${results.bin_width} frames (frames2integ ${results.frames2integ})`);
            for (const v of results.videos) {
              poseState.extractedVideos.add(v.id);
              const counts = Object.entries(v.class_counts)
                .map(([k, n]) => `${k}=${n}`).join(" ");
              poseLog(`${v.id}: ${v.n_rows} rows x ${v.n_features} features | ${counts}`);
              if (v.frames_skipped > 0) {
                poseLog(`  note: pose file has ${v.n_frames_total} frames but labels stop at ` +
                        `${v.n_frames} — skipped ${v.frames_skipped} frames past the last label. ` +
                        `They would have trained as "other" without anyone having reviewed them.`);
              }
              if (v.other_fraction > 0.9) {
                poseLog(`  warning: ${(v.other_fraction * 100).toFixed(1)}% of samples are "other" — ` +
                        `the classifier will mostly learn to say "other". Label more, or tick ` +
                        `"Exclude other" on the Train tab.`);
              }
              if (v.mean_rejected_fraction > 0.2) {
                poseLog(`  warning: ${(v.mean_rejected_fraction * 100).toFixed(0)}% of points ` +
                        `were below the likelihood threshold — check tracking quality.`);
              }
            }
            // Record what the labels looked like for this iteration, so the next
            // extraction can tell whether new labeling has happened since.
            try {
              pm.setIterationMeta(iterationDir, {
                label_signature: signature,
                extracted_at: new Date().toISOString(),
                videos: results.videos.map((v) => v.id),
                class_names: results.class_names,
                n_rows: results.videos.reduce((n, v) => n + v.n_rows, 0),
              });
              pm.setCurrentIteration(parseInt(poseEl("poseIteration").value, 10) || 0);
            } catch { /* manifest not ready */ }
            poseEl("poseStatus").textContent =
              `Done: ${results.videos.length} video(s) in ${iterationDir}`;
            renderTrainVideoChecklist();
          } catch (e) {
            poseEl("poseStatus").textContent = `Error fetching results: ${e.message}`;
          }
        } else if (data.status === "error" || data.status === "cancelled") {
          poseState.sseSource?.close();
          poseEl("poseExtract").disabled = false;
          poseEl("poseStopExtract").disabled = true;
          if (data.error) {
            poseEl("poseStatus").textContent = `Error: ${data.error}`;
            poseLog(data.error);
          }
        }
      },
      () => {
        poseEl("poseExtract").disabled = false;
        poseEl("poseStopExtract").disabled = true;
      }
    );
  } catch (e) {
    poseEl("poseStatus").textContent = `Error: ${e.message}`;
    poseEl("poseExtract").disabled = false;
    poseEl("poseStopExtract").disabled = true;
  }
});

poseEl("poseStopExtract").addEventListener("click", async () => {
  if (!poseState.taskId) return;
  try { await backendFetch(`/api/pose/stop/${poseState.taskId}`, { method: "POST" }); }
  catch { /* already finished */ }
});

// ── Iterations (A-SOID's E_create_new_training) ───────────────────────────
// A-SOID's human-in-the-loop cycle is: train -> find low-confidence frames ->
// relabel them -> retrain on the improved labels, keeping each round's data
// under iteration-N/. Here the relabeling already happens live in the Active
// Learning tab and labels are auto-saved, so "new iteration" is just: bump the
// directory, re-extract from the current labels, retrain.

function currentIterationDir() {
  const n = parseInt(poseEl("poseIteration").value, 10);
  return `iteration-${Number.isFinite(n) && n >= 0 ? n : 0}`;
}

// A cheap fingerprint of the labels feeding an iteration. Changes whenever a
// bout is added, removed, retimed or relabeled on any selected video, which is
// exactly "a new set of frames has been labeled".
function labelSignature(videoNames) {
  const parts = [];
  for (const name of [...videoNames].sort()) {
    // The current video's in-memory bouts are newer than its stored copy.
    const vb = name === currentVideoName ? bouts : (loadProgress(name)?.bouts || []);
    const frames = vb.reduce((n, b) => n + (b.endFrame - b.startFrame + 1), 0);
    const spans = vb
      .map((b) => `${b.startFrame}-${b.endFrame}:${b.behaviorName}`)
      .sort()
      .join(",");
    parts.push(`${name}|${vb.length}|${frames}|${hashString(spans)}`);
  }
  return parts.join(";");
}

// Small non-cryptographic string hash (FNV-1a) — this only needs to change when
// the labels change, not to resist anything.
function hashString(s) {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i += 1) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(36);
}

// Advance to a fresh iteration when the labels have moved on since the current
// one was extracted. Without this the iteration counter is decorative: every
// re-extraction overwrites the previous round's features and models, and there
// is no way to tell which model came from which labeling effort.
function autoAdvanceIteration(videoNames) {
  const pm = window.projectManager;
  if (!pm || !pm.isActive()) return { iterationDir: currentIterationDir(), advanced: false };

  const iterationDir = currentIterationDir();
  const meta = pm.getIterationMeta(iterationDir);
  const signature = labelSignature(videoNames);

  // Nothing extracted here yet → this iteration is still fresh, keep using it.
  if (!meta || !meta.label_signature) {
    return { iterationDir, advanced: false, signature };
  }
  // Same labels as last time → re-extracting is a redo, not a new round.
  if (meta.label_signature === signature) {
    return { iterationDir, advanced: false, signature };
  }

  const next = pm.getCurrentIteration() + 1;
  pm.setCurrentIteration(next);
  poseEl("poseIteration").value = next;
  poseState.extractedVideos.clear();
  return { iterationDir: pm.iterationName(next), advanced: true, signature };
}

poseEl("poseNewIteration").addEventListener("click", async () => {
  const pm = window.projectManager;
  if (!pm || !pm.isActive()) { alert("Open a project first."); return; }

  const next = (parseInt(poseEl("poseIteration").value, 10) || 0) + 1;
  poseEl("poseIteration").value = next;
  // Features from the previous round don't carry over — the whole point is to
  // rebuild them from the labels as they stand now.
  poseState.extractedVideos.clear();
  renderTrainVideoChecklist();
  poseLog(`--- iteration ${next}: re-extract features from the current labels, then train ---`);
  setStatus(`Iteration ${next}. Run Extract Features to rebuild from your refined labels.`);
});

poseEl("poseIteration").addEventListener("change", () => {
  poseState.extractedVideos.clear();
  renderTrainVideoChecklist();
});

// ── Pose skeleton overlay ─────────────────────────────────────────────────
// Drawn on its own canvas above the player so it works in both playback modes
// (the WebCodecs path owns #videoCanvas; the plain <video> path leaves it
// hidden). Points are drawn in the pose file's own pixel coordinates and the
// canvas is sized to the video's natural resolution, so the browser's
// object-fit: contain scaling lines them up with the frame for free.
//
// Deliberately shows RAW, unfiltered keypoints: the whole point is to reveal
// where tracking failed, which the hold-last-position filter would hide.

const poseCanvas = document.getElementById("poseCanvas");
const poseCtx = poseCanvas ? poseCanvas.getContext("2d") : null;
let poseOverlayRaf = null;
let poseOverlayLastFrame = -1;
// Tracks the video/canvas size the overlay was last drawn at (see
// poseOverlayTick) so a size change forces a redraw even on a paused video,
// where currentFrameIndex alone wouldn't change.
let poseOverlayLastSize = "";

function invalidatePoseOverlay() {
  poseState.keypoints = null;
  poseState.keypointsVideo = null;
  poseOverlayLastFrame = -1;
  poseOverlayLastSize = "";
}

async function ensurePoseKeypoints() {
  if (poseState.keypoints && poseState.keypointsVideo === currentVideoName) return true;
  const posePath = getPosePathFor(currentVideoName);
  if (!posePath || poseState.selected.size === 0) return false;

  // Guard against a slow response landing after the user switched videos.
  const requestedFor = currentVideoName;
  try {
    const data = await backendFetch("/api/pose/keypoints", {
      method: "POST",
      body: JSON.stringify({
        pose_path: posePath,
        origin: poseState.origin,
        bodyparts: [...poseState.selected],
        animals: poseState.multiAnimal ? [...poseState.selectedAnimals] : null,
        multi_animal: poseState.multiAnimal,
      }),
    });
    if (currentVideoName !== requestedFor) return false;
    poseState.keypoints = data;
    poseState.keypointsVideo = requestedFor;
    return true;
  } catch (e) {
    setStatus(`Pose overlay unavailable: ${e.message}`);
    return false;
  }
}

function sizePoseCanvas() {
  if (!poseCanvas) return;
  const w = video.videoWidth || videoCanvas.width || 0;
  const h = video.videoHeight || videoCanvas.height || 0;
  if (w && h && (poseCanvas.width !== w || poseCanvas.height !== h)) {
    poseCanvas.width = w;
    poseCanvas.height = h;
  }
}

// Registered as a plugin overlay; the shell clears and sizes the canvas, then
// calls this with the 2d context. Draws nothing when there's no pose to draw.
function drawPoseOverlay() {
  if (!poseCtx || !poseState.keypoints) return;

  const frame = currentFrameIndex;
  const pts = poseState.keypoints.points || [];
  if (pts.length === 0) return;

  const llhCutoff = parseFloat(poseEl("poseLlh").value) || 0.6;
  const coords = [];
  for (const p of pts) {
    if (frame < 0 || frame >= p.x.length) return;
    coords.push({ x: p.x[frame], y: p.y[frame], l: p.likelihood[frame], name: p.name });
  }

  // Connect every pair — the feature extractor uses all pairs, so this is an
  // honest picture of what the model actually sees (rather than an invented
  // anatomical skeleton we'd have to ask the user to define).
  poseCtx.lineWidth = Math.max(1, poseCanvas.width / 800);
  poseCtx.strokeStyle = "rgba(120, 190, 255, 0.25)";
  for (let i = 0; i < coords.length; i++) {
    for (let j = i + 1; j < coords.length; j++) {
      if (coords[i].l < llhCutoff || coords[j].l < llhCutoff) continue;
      poseCtx.beginPath();
      poseCtx.moveTo(coords[i].x, coords[i].y);
      poseCtx.lineTo(coords[j].x, coords[j].y);
      poseCtx.stroke();
    }
  }

  const r = Math.max(2, poseCanvas.width / 250);
  for (const c of coords) {
    // Confident points are cyan, low-likelihood ones red — so a bad prediction
    // caused by bad tracking is visible at a glance while relabeling.
    const good = c.l >= llhCutoff;
    poseCtx.fillStyle = good ? "rgba(80, 220, 255, 0.95)" : "rgba(255, 80, 80, 0.95)";
    poseCtx.beginPath();
    poseCtx.arc(c.x, c.y, r, 0, Math.PI * 2);
    poseCtx.fill();
    poseCtx.lineWidth = Math.max(1, r / 3);
    poseCtx.strokeStyle = "rgba(0, 0, 0, 0.6)";
    poseCtx.stroke();
  }
}

// The shell's overlay loop: sizes and clears the canvas, then hands the 2d
// context to every overlay a pipeline registered. Knows nothing about pose.
function poseOverlayTick() {
  if (!poseEl("showPoseOverlay").checked) { poseOverlayRaf = null; return; }
  // Redraw on a frame change, OR when the video's dimensions have just become
  // known/changed — e.g. the overlay was toggled on before a freshly loaded,
  // paused video finished decoding its first frame. Without the size check,
  // that single early draw could land on a wrong-sized canvas and then never
  // redraw again (currentFrameIndex isn't changing on a paused video), leaving
  // a misaligned overlay until the user scrubs.
  const w = video.videoWidth || videoCanvas.width || 0;
  const h = video.videoHeight || videoCanvas.height || 0;
  const sizeKey = `${w}x${h}`;
  if (currentFrameIndex !== poseOverlayLastFrame || sizeKey !== poseOverlayLastSize) {
    poseOverlayLastFrame = currentFrameIndex;
    poseOverlayLastSize = sizeKey;
    if (poseCtx) {
      sizePoseCanvas();
      poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
      const ctx = makeContext();
      for (const o of window.labelerPlugins.overlays()) {
        try {
          o.draw(poseCtx, ctx);
        } catch (err) {
          console.warn(`[labeler] overlay "${o.id}" draw failed`, err);
        }
      }
    }
  }
  poseOverlayRaf = requestAnimationFrame(poseOverlayTick);
}

poseEl("showPoseOverlay").addEventListener("change", async () => {
  const on = poseEl("showPoseOverlay").checked;
  if (!poseCanvas) return;
  if (!on) {
    poseCanvas.style.display = "none";
    if (poseOverlayRaf) cancelAnimationFrame(poseOverlayRaf);
    poseOverlayRaf = null;
    return;
  }
  const ready = await ensurePoseKeypoints();
  if (!ready) {
    poseEl("showPoseOverlay").checked = false;
    setStatus("Attach a pose file and select body parts first.");
    return;
  }
  poseCanvas.style.display = "block";
  poseOverlayLastFrame = -1;
  poseOverlayLastSize = "";
  if (!poseOverlayRaf) poseOverlayTick();
});

// Called on video switch so one video's pose never draws over another.
function resetPoseForVideo() {
  invalidatePoseOverlay();
  syncPoseFramerateFromVideo();
  const p = getPosePathFor(currentVideoName);
  setPathDisplay("posePath", p || "No pose file for this video");
  if (poseCanvas && poseEl("showPoseOverlay").checked) {
    // Re-fetch for the new video; leave the toggle on if it succeeds.
    ensurePoseKeypoints().then((ok) => {
      if (!ok) {
        poseEl("showPoseOverlay").checked = false;
        poseCanvas.style.display = "none";
      } else {
        poseOverlayLastFrame = -1;
        poseOverlayLastSize = "";
      }
    });
  }
  if (p) {
    inspectPoseFile(p).catch(() => { /* stale/unreadable file */ });
  }
  renderPoseVideoChecklist();
}

// ── Register this pipeline with the shell ─────────────────────────────────
// Everything above is pose-specific: it belongs to the A-SOID pipeline, not to
// the labeling shell. Registering it here means the shell reaches it only
// through the plugin seam — no core code names a pose function — which is what
// makes the shell extractable as a standalone labeler, with RealtimeASOID's
// video pipeline registering in exactly the same way.
//
// The functions themselves still live in this file; moving them into their own
// module is mechanical once the seam holds.

// ── Training Panel ────────────────────────────────────────────────────────

const trainState = {
  taskId: null,
  sseSource: null,
  iterations: 0,
};

function renderTrainVideoChecklist() {
  const container = document.getElementById("trainVideoChecklist");
  if (!container) return;
  container.innerHTML = "";

  const pm = window.projectManager;
  if (!pm || !pm.isActive()) {
    container.innerHTML = '<p class="muted">Open a project to select training videos.</p>';
    return;
  }

  const names = videoList.map((v) => v.name);
  if (names.length === 0) {
    container.innerHTML = '<p class="muted">No videos in this project.</p>';
    return;
  }

  for (const name of names) {
    const id = pm.getVideoIdFromName(name);
    const ready = poseState.extractedVideos.has(id) || !!getPosePathFor(name);
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = ready;
    cb.disabled = !ready;
    cb.dataset.videoId = id;
    const span = document.createElement("span");
    span.textContent = ready ? name : `${name} (no features yet)`;
    if (!ready) span.className = "muted";
    row.append(cb, span);
    container.appendChild(row);
  }
}

function selectedTrainVideoIds() {
  const container = document.getElementById("trainVideoChecklist");
  if (!container) return [];
  return [...container.querySelectorAll("input[type=checkbox]")]
    .filter((cb) => cb.checked && !cb.disabled)
    .map((cb) => cb.dataset.videoId);
}

function trainLog(line) {
  const el = document.getElementById("trainLog");
  if (!el) return;
  el.style.display = "block";
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

function applyTrainMetrics(data) {
  const metrics = document.getElementById("trainMetrics");
  const set = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };

  if (data.event === "ceiling") {
    metrics.style.display = "grid";
    set("trainCeilingF1", data.macro_f1.toFixed(3));
    trainLog(`ceiling (all labeled data): macro F1 ${data.macro_f1.toFixed(3)} ` +
             `on ${data.n_train} samples`);
    return;
  }

  if (data.event === "iteration") {
    metrics.style.display = "grid";
    trainState.iterations = data.iteration;
    set("trainIteration", data.iteration);
    set("trainTotalIters", document.getElementById("trainMaxIter").value);
    set("trainMacroF1", data.macro_f1.toFixed(3));
    set("trainNSamples", data.n_train);
    set("trainLowConf", data.n_lowconf_remaining);
    set("trainPerClassF1", (data.f1_per_class || []).map((f) => f.toFixed(2)).join(", "));
    if (data.ceiling_macro_f1 !== undefined) {
      set("trainCeilingF1", data.ceiling_macro_f1.toFixed(3));
    }
    const pct = Math.min(100, Math.round(
      (data.iteration / (parseInt(document.getElementById("trainMaxIter").value, 10) || 1)) * 100
    ));
    document.getElementById("trainProgress").value = pct;
    trainLog(`iter ${String(data.iteration).padStart(3)}  macro F1 ${data.macro_f1.toFixed(3)}  ` +
             `samples ${data.n_train}  low-confidence left ${data.n_lowconf_remaining}`);
    return;
  }

  if (data.message) trainLog(data.message);
  if (data.error) trainLog(`ERROR: ${data.error}`);
}

document.getElementById("trainStart").addEventListener("click", async () => {
  const pm = window.projectManager;
  if (!pm || !pm.isActive()) { alert("Open a project first."); return; }

  const videoIds = selectedTrainVideoIds();
  if (videoIds.length === 0) {
    alert("No videos with extracted features are selected. Run the Pose tab first.");
    return;
  }

  // binarizeSelected holds behavior NAMES: 0 = multi-class, 1 = binarize that
  // behavior, >=2 = union them into one positive class first.
  const positives = [...binarizeSelected];
  const config = {
    project_dir: pm.getProjectRoot(),
    video_ids: videoIds,
    iteration_dir: currentIterationDir(),
    init_ratio: parseFloat(document.getElementById("trainInitRatio").value) || 0.01,
    max_iter: parseInt(document.getElementById("trainMaxIter").value, 10) || 100,
    max_samples_iter: parseInt(document.getElementById("trainMaxSamplesIter").value, 10) || 20,
    conf_threshold: parseFloat(document.getElementById("trainConfThreshold").value) || 0.5,
    exclude_other: document.getElementById("trainExcludeOther").checked,
    binarize: positives.length === 1 ? positives[0] : null,
    label_union: positives.length > 1 ? positives : null,
    // Save the model beside the features it was trained on.
    checkpoint_dir: pm.getIterationModelsDir(currentIterationDir()),
    model_name: document.getElementById("trainModelName").value || "rf_model",
  };

  try { pm.updateTrainingConfig(config); } catch { /* manifest not ready */ }

  document.getElementById("trainLog").textContent = "";
  document.getElementById("trainStart").disabled = true;
  document.getElementById("trainStop").disabled = false;
  document.getElementById("trainStatus").textContent = "Starting...";
  document.getElementById("trainProgress").value = 0;

  try {
    const res = await backendFetch("/api/training/start", {
      method: "POST",
      body: JSON.stringify(config),
    });
    trainState.taskId = res.task_id;

    trainState.sseSource = backendSSE(
      `/api/training/status/${res.task_id}`,
      (data) => {
        applyTrainMetrics(data);
        if (data.status === "completed") {
          trainState.sseSource?.close();
          document.getElementById("trainStart").disabled = false;
          document.getElementById("trainStop").disabled = true;
          document.getElementById("trainProgress").value = 100;
          document.getElementById("trainStatus").textContent =
            `Done: macro F1 ${(data.macro_f1 ?? 0).toFixed(3)} in ${data.iterations} iterations`;
          if (data.checkpoint_path) {
            // Record the model so the AL tab can find it and validate its
            // body-part selection against the current one.
            try {
              pm.addModel({
                name: data.checkpoint_path.split(/[\\/]/).pop(),
                path: data.checkpoint_path,
                created: new Date().toISOString().slice(0, 10),
                iteration: config.iteration_dir,
                macro_f1: data.macro_f1,
                source: "trained",
                config,
              });
            } catch { /* manifest not ready */ }
            alState.checkpointPath = data.checkpoint_path;
            setAlCheckpointDisplay(data.checkpoint_path, config.iteration_dir);
            trainLog(`checkpoint: ${data.checkpoint_path}`);
          }
        } else if (data.status === "error" || data.status === "cancelled") {
          trainState.sseSource?.close();
          document.getElementById("trainStart").disabled = false;
          document.getElementById("trainStop").disabled = true;
          document.getElementById("trainStatus").textContent =
            data.status === "cancelled" ? "Cancelled" : `Error: ${data.error || "unknown"}`;
        } else {
          document.getElementById("trainStatus").textContent = data.status || "running";
        }
      },
      () => {
        document.getElementById("trainStart").disabled = false;
        document.getElementById("trainStop").disabled = true;
      }
    );
  } catch (e) {
    document.getElementById("trainStatus").textContent = `Error: ${e.message}`;
    document.getElementById("trainStart").disabled = false;
    document.getElementById("trainStop").disabled = true;
  }
});

document.getElementById("trainStop").addEventListener("click", async () => {
  if (!trainState.taskId) return;
  try { await backendFetch(`/api/training/stop/${trainState.taskId}`, { method: "POST" }); }
  catch { /* already finished */ }
});

// ── Batch Predict Panel ───────────────────────────────────────────────────

const predState = {
  checkpointPath: null,
  files: [],
  outDir: null,
  taskId: null,
  sseSource: null,
};

function renderPredFileList() {
  const container = document.getElementById("predFileList");
  if (!container) return;
  container.innerHTML = "";
  if (predState.files.length === 0) {
    container.innerHTML = '<p class="muted">No pose files selected.</p>';
    return;
  }
  for (const f of predState.files) {
    const row = document.createElement("div");
    row.className = "data-source-row";
    const span = document.createElement("span");
    span.textContent = f.split(/[\\/]/).pop();
    span.title = f;
    const btn = document.createElement("button");
    btn.className = "ghost";
    btn.textContent = "×";
    btn.addEventListener("click", () => {
      predState.files = predState.files.filter((x) => x !== f);
      renderPredFileList();
    });
    row.append(span, btn);
    container.appendChild(row);
  }
}

function predLog(line) {
  const el = document.getElementById("predLog");
  if (!el) return;
  el.style.display = "block";
  el.textContent += line + "\n";
  el.scrollTop = el.scrollHeight;
}

document.getElementById("predSelectCheckpoint").addEventListener("click", async () => {
  if (!window.api) return;
  const res = await window.api.selectFile({
    title: "Select Model",
    filters: [{ name: "Models", extensions: ["joblib"] }],
  });
  if (res.canceled) return;
  predState.checkpointPath = res.path;
  setPathDisplay("predCheckpointPath", res.path);
});

document.getElementById("predAddFiles").addEventListener("click", async () => {
  if (!window.api) return;
  const res = await window.api.selectFiles({
    title: "Select Pose Files",
    filters: [
      { name: "Pose files", extensions: ["csv", "h5", "mat", "txt"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (res.canceled) return;
  for (const p of res.paths) if (!predState.files.includes(p)) predState.files.push(p);
  renderPredFileList();
});

document.getElementById("predClearFiles").addEventListener("click", () => {
  predState.files = [];
  renderPredFileList();
});

document.getElementById("predSelectOutDir").addEventListener("click", async () => {
  if (!window.api) return;
  const res = await window.api.selectDirectory();
  if (res.canceled) return;
  predState.outDir = res.path;
  setPathDisplay("predOutDir", res.path);
});

document.getElementById("predRun").addEventListener("click", async () => {
  if (!predState.checkpointPath) { alert("Select a model first."); return; }
  if (predState.files.length === 0) { alert("Add at least one pose file."); return; }

  const pm = window.projectManager;
  const outDir = predState.outDir
    || (pm && pm.isActive() ? pm.getProjectRoot() + "/predictions" : null);
  if (!outDir) { alert("Choose an output folder (or open a project)."); return; }

  document.getElementById("predLog").textContent = "";
  document.getElementById("predRun").disabled = true;
  document.getElementById("predStop").disabled = false;
  document.getElementById("predStatus").textContent = "Starting...";

  try {
    const res = await backendFetch("/api/inference/batch", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_path: predState.checkpointPath,
        pose_paths: predState.files,
        output_dir: outDir,
        origin: poseState.origin,
        // Body parts default to whatever the checkpoint was trained on, which
        // is the only selection guaranteed to match its feature space.
        bodyparts: null,
        llh_value: parseFloat(poseEl("poseLlh").value) || 0.6,
        animals: poseState.multiAnimal ? [...poseState.selectedAnimals] : null,
        multi_animal: poseState.multiAnimal,
      }),
    });
    predState.taskId = res.task_id;

    predState.sseSource = backendSSE(
      `/api/inference/status/${res.task_id}`,
      async (data) => {
        const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
        document.getElementById("predProgress").value = pct;
        document.getElementById("predStatus").textContent =
          `${data.status}: ${data.progress}/${data.total}`;

        if (data.status === "completed") {
          predState.sseSource?.close();
          document.getElementById("predRun").disabled = false;
          document.getElementById("predStop").disabled = true;
          try {
            const out = await backendFetch(`/api/inference/results/${predState.taskId}`);
            for (const f of out.files || []) {
              predLog(`${f.output_path.split(/[\\/]/).pop()}  (${f.n_frames} frames)`);
              for (const [name, s] of Object.entries(f.summary || {})) {
                predLog(`    ${name}: ${s.frames} frames, mean uncertainty ${s.mean_uncertainty}`);
              }
            }
            document.getElementById("predStatus").textContent =
              `Done: ${(out.files || []).length} file(s) -> ${out.output_dir}`;
          } catch (e) {
            document.getElementById("predStatus").textContent = `Error: ${e.message}`;
          }
        } else if (data.status === "error" || data.status === "cancelled") {
          predState.sseSource?.close();
          document.getElementById("predRun").disabled = false;
          document.getElementById("predStop").disabled = true;
          if (data.error) {
            document.getElementById("predStatus").textContent = `Error: ${data.error}`;
            predLog(data.error);
          }
        }
      },
      () => {
        document.getElementById("predRun").disabled = false;
        document.getElementById("predStop").disabled = true;
      }
    );
  } catch (e) {
    document.getElementById("predStatus").textContent = `Error: ${e.message}`;
    document.getElementById("predRun").disabled = false;
    document.getElementById("predStop").disabled = true;
  }
});

document.getElementById("predStop").addEventListener("click", async () => {
  if (!predState.taskId) return;
  try { await backendFetch(`/api/inference/stop/${predState.taskId}`, { method: "POST" }); }
  catch { /* already finished */ }
});

renderPredFileList();
renderPoseBodypartList();
renderPoseAnimalList();

// ── Register this pipeline with the shell ─────────────────────────────────
// Everything above is A-SOID-specific: pose files, feature extraction, the
// random forest, the active-learning sweep. None of it belongs to the labeling
// shell, and nothing in core/ names any of it. The shell reaches this file only
// through the seam below, which is what lets the same shell run RealtimeASOID's
// TSM pipeline — or no pipeline at all, as a standalone labeler.
window.labelerPlugins.register({
  id: "asoid-pose",

  hooks: {
    // A different video is active: repoint the pose file, overlay and panel.
    onVideoSwitched: () => resetPoseForVideo(),
    // A pipeline tab became visible: its video checklist may be stale.
    onTabShown: (ctx) => {
      if (ctx.tab === "pose") renderPoseVideoChecklist();
      if (ctx.tab === "train") renderTrainVideoChecklist();
    },
    // A project was opened or closed. The iteration belongs to the project, not
    // the session, so restore it here — otherwise reopening silently starts
    // writing over iteration-0 again.
    onProjectOpened: (ctx) => {
      renderPoseVideoChecklist();
      renderTrainVideoChecklist();
      const pm = ctx.projectManager;
      if (!pm || !pm.isActive()) return;
      poseEl("poseIteration").value = pm.getCurrentIteration();
      selectNewestCheckpoint({ silent: true });
    },
  },

  // Drawn over the video frame by the shell's overlay loop.
  overlays: [{ id: "pose-skeleton", draw: drawPoseOverlay }],

  // What a trained model looks like on disk, for the shell's file pickers.
  models: { label: "Models", extensions: ["joblib"] },

  // Turn the shell's context into this backend's inference request. The
  // response must come back in the standard predictions shape.
  inference: {
    buildRequest: (ctx) => ({
      pose_path: getPosePathFor(ctx.videoName),
      origin: poseState.origin,
      bodyparts: null,   // null = use the selection recorded on the checkpoint
      llh_value: parseFloat(poseEl("poseLlh").value) || 0.6,
      animals: poseState.multiAnimal ? [...poseState.selectedAnimals] : null,
      multi_animal: poseState.multiAnimal,
    }),
    // What the shell should say when the pipeline can't run yet.
    validate: (ctx) => (getPosePathFor(ctx.videoName)
      ? null
      : "This video has no pose file attached. Attach one in the Pose tab first."),
  },

  // The shell owns one settings blob in localStorage; the pipeline contributes
  // its own keys to it rather than the shell knowing what a body part is.
  // Keys must not collide with the shell's (playerCol, showHud, alThreshold…).
  settings: {
    save: () => ({
      poseOrigin: poseEl("poseOrigin").value,
      poseMultiAnimal: poseEl("poseMultiAnimal").checked,
      poseFramerate: parseFloat(poseEl("poseFramerate").value),
      poseFramerateManual: poseState.framerateManual,
      poseLlh: parseFloat(poseEl("poseLlh").value),
      poseMinDuration: parseFloat(poseEl("poseMinDuration").value),
      poseByVideo: poseState.poseByVideo,
      poseSelectedBodyparts: [...poseState.selected],
      trainInitRatio: parseFloat(poseEl("trainInitRatio").value),
      trainMaxIter: parseInt(poseEl("trainMaxIter").value),
      trainMaxSamplesIter: parseInt(poseEl("trainMaxSamplesIter").value),
      trainConfThreshold: parseFloat(poseEl("trainConfThreshold").value),
      trainExcludeOther: poseEl("trainExcludeOther").checked,
      trainModelName: poseEl("trainModelName").value,
    }),

    load: (s) => {
      const setVal = (id, v) => { if (v !== undefined && v !== null) poseEl(id).value = v; };
      const setChecked = (id, v) => { if (v !== undefined) poseEl(id).checked = v; };

      setVal("poseOrigin", s.poseOrigin);
      setChecked("poseMultiAnimal", s.poseMultiAnimal);
      setVal("poseFramerate", s.poseFramerate);
      // Restore whether the user had detached this from the video's detected
      // rate, so a later video load doesn't silently overwrite their choice.
      poseState.framerateManual = !!s.poseFramerateManual;
      setVal("poseLlh", s.poseLlh);
      setVal("poseMinDuration", s.poseMinDuration);
      if (s.poseOrigin) poseState.origin = s.poseOrigin;
      if (s.poseMultiAnimal !== undefined) poseState.multiAnimal = s.poseMultiAnimal;
      if (s.poseByVideo && typeof s.poseByVideo === "object") {
        poseState.poseByVideo = s.poseByVideo;
      }
      // The body-part list itself is repopulated from the pose file on load;
      // this only restores which of them were ticked.
      if (Array.isArray(s.poseSelectedBodyparts)) {
        poseState.selected = new Set(s.poseSelectedBodyparts);
      }

      setVal("trainInitRatio", s.trainInitRatio);
      setVal("trainMaxIter", s.trainMaxIter);
      setVal("trainMaxSamplesIter", s.trainMaxSamplesIter);
      setVal("trainConfThreshold", s.trainConfThreshold);
      setChecked("trainExcludeOther", s.trainExcludeOther);
      setVal("trainModelName", s.trainModelName);

      updatePoseDerivedDisplays();
    },
  },
});
