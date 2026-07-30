const video = document.getElementById("video");
const videoShell = document.getElementById("videoShell");
const videoCanvas = document.getElementById("videoCanvas");
const trackbar = document.getElementById("trackbar");
const playPauseButton = document.getElementById("playPause");
const snapBackButton = document.getElementById("snapBack");
const fullscreenToggle = document.getElementById("fullscreenToggle");
const playbackSpeedInput = document.getElementById("playbackSpeed");
const pauseOnLabelEndInput = document.getElementById("pauseOnLabelEnd");
const fpsInput = document.getElementById("fpsInput");
const fpsLockInput = document.getElementById("fpsLock");
const useWebCodecsInput = document.getElementById("useWebCodecs");
const modeControl = document.getElementById("modeControl");
const webcodecsStatus = document.getElementById("webcodecsStatus");
const currentTimeEl = document.getElementById("currentTime");
const durationEl = document.getElementById("duration");
const videoFileInput = document.getElementById("videoFile");
const behaviorNameInput = document.getElementById("behaviorName");
const addBehaviorButton = document.getElementById("addBehavior");
const behaviorList = document.getElementById("behaviorList");
const boutsTable = document.getElementById("boutsTable");
const exportCsvButton = document.getElementById("exportCsv");
const importCsvButton = document.getElementById("importCsv");
const exportOneHotButton = document.getElementById("exportOneHot");
const fillBehaviorInput = document.getElementById("fillBehavior");
const generateLabeledVideoButton = document.getElementById("generateLabeledVideo");
const statusEl = document.getElementById("status");

const KEY_ORDER = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];

let behaviors = loadBehaviors();
let bouts = [];
let activeStarts = new Map();
const selectedBehaviorIds = new Set(); // behaviors ticked for merging
const binarizeSelected = new Set();    // behavior NAMES forming the positive class (Train tab)

const insertBoutSorted = (bout) => {
  let i = bouts.length;
  while (i > 0 && bouts[i - 1].startFrame > bout.startFrame) i--;
  bouts.splice(i, 0, bout);
};
let currentVideoName = "";
let frameDurationMs = 1000 / 30;
let lastFrameMediaTime = null;
let currentFrameIndex = 0;
let pendingSeekFrame = null;
let pendingSeekTime = null;
let suppressTimeUpdate = false;
let currentFile = null;

// ── Multi-video sidebar ───────────────────────────────────────────────────
function loadVideoList() {
  try {
    const stored = localStorage.getItem("videoList");
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}
let videoList = loadVideoList();

let webcodecsMode = false;
let webcodecsReady = false;
let webcodecsLoading = false;
let webcodecsSamples = [];
let webcodecsKeyframes = [];
let webcodecsDecoder = null;
let webcodecsConfig = null;
let webcodecsDecodeQueue = [];
let webcodecsDecodedMax = -1;
let webcodecsFrameCache = new Map();
let webcodecsFrameOrder = [];
let webcodecsCacheLimit = 300;
let webcodecsDurationMs = 0;
let webcodecsPlaying = false;
let webcodecsTimer = null;
let webcodecsPendingRender = null;
let mp4boxWorker = null;
let mp4boxChunkId = 0;
let mp4boxChunkResolvers = new Map();
let webcodecsOperationInProgress = false;
// A WebCodecs load mutates shared decoder/sample globals, so only one can run at
// a time. If the user clicks another video mid-load, we stash the latest request
// here and open it once the in-flight load settles (latest pick wins).
let pendingVideoSwitch = null;

const setStatus = (message) => {
  statusEl.textContent = message;
};

const setWebcodecsStatus = (message) => {
  if (!webcodecsStatus) {
    return;
  }
  webcodecsStatus.textContent = message || "";
  webcodecsStatus.title = message || "";
};

const toFileUrl = (filePath) => {
  const normalized = filePath.replace(/\\/g, "/");
  if (/^[A-Za-z]:\//.test(normalized)) {
    return encodeURI(`file:///${normalized}`);
  }
  const prefix = normalized.startsWith("/") ? "" : "/";
  return encodeURI(`file://${prefix}${normalized}`);
};

const canvasContext = videoCanvas.getContext("2d");
const webcodecsApiSupported = () =>
  typeof VideoDecoder !== "undefined" &&
  typeof EncodedVideoChunk !== "undefined" &&
  typeof Worker !== "undefined";

const webcodecsSupported = () => webcodecsApiSupported();

const webcodecsUnavailabilityReason = () => {
  if (typeof VideoDecoder === "undefined") {
    return "WebCodecs is unavailable: VideoDecoder is not supported.";
  }
  if (typeof EncodedVideoChunk === "undefined") {
    return "WebCodecs is unavailable: EncodedVideoChunk is not supported.";
  }
  if (typeof Worker === "undefined") {
    return "WebCodecs is unavailable: Web Worker is not supported.";
  }
  try {
    // Worker will import mp4box for demuxing.
  } catch (error) {
    const message = error && error.message ? error.message : "Unknown error.";
    return `WebCodecs is unavailable: MP4 demuxer failed to load: ${message}`;
  }
  return "";
};

const formatTime = (milliseconds) => {
  const totalMs = Math.max(0, Math.round(milliseconds));
  const mins = Math.floor(totalMs / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(
    2,
    "0"
  )}:${String(ms).padStart(3, "0")}`;
};

const parseTime = (timeString) => {
  // Parse "MM:SS:mmm", "MM:SS", or "HH:MM:SS:mmm" format
  const parts = timeString.trim().split(":");
  if (parts.length < 2 || parts.length > 4) return null;
  let hours = 0, mins, secs, ms = 0;
  if (parts.length === 4) {
    hours = parseInt(parts[0], 10);
    mins = parseInt(parts[1], 10);
    secs = parseInt(parts[2], 10);
    ms = parseInt(parts[3], 10);
  } else if (parts.length === 3) {
    mins = parseInt(parts[0], 10);
    secs = parseInt(parts[1], 10);
    ms = parseInt(parts[2], 10);
  } else {
    mins = parseInt(parts[0], 10);
    secs = parseInt(parts[1], 10);
  }
  if (isNaN(hours) || isNaN(mins) || isNaN(secs) || isNaN(ms)) return null;
  if (hours < 0 || mins < 0 || secs < 0 || ms < 0 || ms >= 1000) return null;
  return hours * 3600000 + mins * 60000 + secs * 1000 + ms;
};

const msToFrame = (ms) => {
  return Math.round(ms / frameDurationMs);
};

const editBoutRow = (boutIndex) => {
  const bout = bouts[boutIndex];
  if (!bout) return;
  
  const rows = boutsTable.querySelectorAll("tr");
  const row = rows[boutIndex];
  if (!row) return;
  
  const cells = row.querySelectorAll("td");
  const startCell = cells[0];
  const endCell = cells[1];
  const actionsCell = cells[3];
  
  // Replace start cell with input
  const startInput = document.createElement("input");
  startInput.type = "text";
  startInput.value = formatTime(getFrameTimeMs(bout.startFrame));
  startInput.style.width = "80px";
  startCell.textContent = "";
  startCell.appendChild(startInput);
  startCell.className = "";
  
  // Replace end cell with input
  const endInput = document.createElement("input");
  endInput.type = "text";
  endInput.value = formatTime(getFrameTimeMs(bout.endFrame));
  endInput.style.width = "80px";
  endCell.textContent = "";
  endCell.appendChild(endInput);
  endCell.className = "";
  
  // Replace actions with Save/Cancel
  actionsCell.innerHTML = "";
  const saveButton = document.createElement("button");
  saveButton.className = "ghost";
  saveButton.textContent = "Save";
  saveButton.addEventListener("click", () => {
    const startMs = parseTime(startInput.value);
    const endMs = parseTime(endInput.value);
    
    if (startMs === null) {
      setStatus("Invalid start time. Use MM:SS:mmm or MM:SS.");
      return;
    }
    if (endMs === null) {
      setStatus("Invalid end time. Use MM:SS:mmm or MM:SS.");
      return;
    }
    
    const newStartFrame = msToFrame(startMs);
    const newEndFrame = msToFrame(endMs);
    
    if (newStartFrame >= newEndFrame) {
      setStatus("Start time must be before end time.");
      return;
    }
    
    bout.startFrame = newStartFrame;
    bout.endFrame = newEndFrame;
    renderBouts();
    saveProgress();
    setStatus(`Updated bout to ${formatTime(getFrameTimeMs(newStartFrame))} - ${formatTime(getFrameTimeMs(newEndFrame))}.`);
  });
  
  const cancelButton = document.createElement("button");
  cancelButton.className = "ghost";
  cancelButton.textContent = "Cancel";
  cancelButton.addEventListener("click", () => {
    renderBouts();
  });
  
  actionsCell.appendChild(saveButton);
  actionsCell.appendChild(cancelButton);
  
  startInput.focus();
  startInput.select();
};

const updateFrameDuration = (mediaTime) => {
  if (fpsLockInput.checked) {
    return;
  }
  if (!Number.isFinite(mediaTime)) {
    return;
  }
  if (lastFrameMediaTime !== null) {
    const deltaMs = (mediaTime - lastFrameMediaTime) * 1000;
    if (deltaMs > 0.5 && deltaMs < 200) {
      frameDurationMs = deltaMs;
    }
  }
  lastFrameMediaTime = mediaTime;
};

const startFrameWatch = () => {
  if (webcodecsMode) {
    return;
  }
  if (typeof video.requestVideoFrameCallback !== "function") {
    return;
  }
  const onFrame = (_now, metadata) => {
    updateFrameDuration(metadata.mediaTime);
    currentFrameIndex = Math.round(
      (metadata.mediaTime * 1000) / frameDurationMs
    );
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
};

const applyLockedFps = () => {
  const fpsValue = Number(fpsInput.value);
  if (!Number.isFinite(fpsValue) || fpsValue <= 0) {
    return;
  }
  frameDurationMs = 1000 / fpsValue;
};

const updateDurationDisplay = () => {
  // Duration is the first point at which the video's frame count is knowable,
  // so record it against the project here (see setVideoFrameCount).
  try {
    const pm = window.projectManager;
    if (pm && pm.isActive() && currentVideoName) {
      pm.setVideoFrameCount(currentVideoName, videoFrameCount());
    }
  } catch { /* project manager or helpers not ready yet */ }

  if (webcodecsMode && webcodecsDurationMs) {
    durationEl.textContent = formatTime(webcodecsDurationMs);
    return;
  }
  if (Number.isFinite(video.duration)) {
    durationEl.textContent = formatTime(video.duration * 1000);
  } else {
    durationEl.textContent = "00:00:000";
  }
};

const updateTrackbar = () => {
  if (webcodecsMode) {
    trackbar.max = Math.floor(webcodecsDurationMs || 0);
    trackbar.value = Math.round(getFrameTimeMs(currentFrameIndex));
    return;
  }
  if (Number.isFinite(video.duration)) {
    trackbar.max = Math.floor(video.duration * 1000);
  } else {
    trackbar.max = 0;
  }
  trackbar.value = getVideoTimeMs();
};

const setWebcodecsMode = (enabled) => {
  console.log("[renderer] setWebcodecsMode called:", enabled, "opInProgress:", webcodecsOperationInProgress, "loading:", webcodecsLoading, "ready:", webcodecsReady, "samples:", webcodecsSamples.length);
  
  // Don't disable WebCodecs during an operation
  if (!enabled && webcodecsOperationInProgress) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) - operation in progress");
    return;
  }
  
  // Don't disable WebCodecs if decoder exists (loading is in progress or complete)
  if (!enabled && webcodecsDecoder) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) - decoder exists");
    return;
  }
  
  // Don't change mode during loading
  if (webcodecsLoading && !enabled) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) during loading");
    return;
  }
  
  // Don't disable WebCodecs if it's ready and working
  if (!enabled && webcodecsReady) {
    console.log("[renderer] Ignoring setWebcodecsMode(false) - WebCodecs is ready");
    return;
  }
  
  webcodecsMode = enabled;
  document.body.classList.toggle("webcodecs", webcodecsMode);
  if (webcodecsMode) {
    video.pause();
    playPauseButton.textContent = "Play";
  } else {
    resetWebcodecsState();
  }
};

const stopWebcodecsPlayback = () => {
  if (webcodecsTimer) {
    clearInterval(webcodecsTimer);
    webcodecsTimer = null;
  }
  webcodecsPlaying = false;
  // Clear the decode queue to prevent stale entries from interfering
  webcodecsDecodeQueue = [];
  playPauseButton.textContent = "Play";
};

const startWebcodecsPlayback = () => {
  if (!webcodecsReady) {
    setStatus("WebCodecs is not ready yet.");
    return;
  }
  const playbackRate = Number(playbackSpeedInput.value) || 1;
  const interval = Math.max(5, frameDurationMs / playbackRate);
  console.log("[renderer] Starting playback, interval:", interval, "ms, rate:", playbackRate);
  stopWebcodecsPlayback();
  webcodecsPlaying = true;
  playPauseButton.textContent = "Pause";
  
  let targetFrame = currentFrameIndex;
  webcodecsTimer = setInterval(() => {
    // Only advance target if decoder is keeping up
    if (!webcodecsDecoder || webcodecsDecoder.state !== "configured") {
      console.warn("[renderer] Decoder not ready, state:", webcodecsDecoder?.state);
      stopWebcodecsPlayback();
      return;
    }
    
    // Check if we've reached the end of available samples
    if (targetFrame + 1 >= webcodecsSamples.length) {
      // Check if we're truly at the end based on time
      const currentTimeMs = targetFrame * frameDurationMs;
      if (currentTimeMs >= webcodecsDurationMs - frameDurationMs) {
        console.log("[renderer] Reached end of video");
        stopWebcodecsPlayback();
        return;
      }
      // More samples should be coming, wait
      return;
    }
    
    // Don't advance too far ahead of what's been decoded
    // This prevents overwhelming the decoder
    const maxAhead = 5;
    if (targetFrame - webcodecsDecodedMax > maxAhead) {
      // Decoder is behind, wait for it to catch up
      return;
    }
    
    targetFrame += 1;
    decodeWebcodecsUpTo(targetFrame);
  }, interval);
};

const renderWebcodecsFrame = (frameIndex) => {
  const bitmap = webcodecsFrameCache.get(frameIndex);
  if (!bitmap) {
    console.warn("[renderer] No bitmap for frame:", frameIndex);
    return;
  }
  canvasContext.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
  canvasContext.drawImage(bitmap, 0, 0, videoCanvas.width, videoCanvas.height);
  currentFrameIndex = frameIndex;
  updateTimeDisplay();
  updateTrackbar();
  saveProgressDebounced();
};

const cacheWebcodecsFrame = (frameIndex, bitmap) => {
  webcodecsFrameCache.set(frameIndex, bitmap);
  webcodecsFrameOrder.push(frameIndex);
  if (webcodecsFrameOrder.length > webcodecsCacheLimit) {
    const evictIndex = webcodecsFrameOrder.shift();
    const evictBitmap = webcodecsFrameCache.get(evictIndex);
    if (evictBitmap) {
      evictBitmap.close();
    }
    webcodecsFrameCache.delete(evictIndex);
  }
};

const handleDecodedFrame = (frame) => {
  const frameIndex = webcodecsDecodeQueue.shift();
  if (frameIndex === undefined) {
    frame.close();
    return;
  }
  createImageBitmap(frame)
    .then((bitmap) => {
      frame.close();
      cacheWebcodecsFrame(frameIndex, bitmap);
      
      if (webcodecsPlaying) {
        renderWebcodecsFrame(frameIndex);
      } else {
        // After caching this frame, check if the CURRENT target is now available
        // This handles the case where user stepped ahead while decode was in progress
        if (webcodecsFrameCache.has(currentFrameIndex)) {
          renderWebcodecsFrame(currentFrameIndex);
        }
      }
    })
    .catch((error) => {
      frame.close();
      console.warn("[renderer] Failed to create bitmap for frame", frameIndex, error);
    });
};

// Close and drop the active decoder, tolerating an already-closed one. A
// VideoDecoder auto-transitions to "closed" on error, and calling close() on a
// closed codec throws InvalidStateError ("Cannot call 'close' on a closed
// codec") — so always check state and swallow. Nulling the reference also lets
// setWebcodecsMode(false) proceed (it bails while a decoder object exists).
const closeWebcodecsDecoder = () => {
  if (webcodecsDecoder) {
    try {
      if (webcodecsDecoder.state !== "closed") webcodecsDecoder.close();
    } catch (e) {
      console.warn("[renderer] decoder close ignored:", e && e.message ? e.message : e);
    }
    webcodecsDecoder = null;
  }
};

const handleDecoderError = (error) => {
  console.error("[renderer] WebCodecs decoder error:", error);
  // The errored decoder is already closed; drop the stale reference first so the
  // HTML5 fallback's setWebcodecsMode(false) isn't blocked by "decoder exists".
  closeWebcodecsDecoder();
  setStatus("WebCodecs decoding failed; switching to HTML5 mode.");
  setWebcodecsStatus(`Decode error: ${error.message || error}`);
  stopWebcodecsPlayback();
  if (currentFile) {
    useWebCodecsInput.checked = false;
    setWebcodecsMode(false);
    loadHtml5File(currentFile);
  }
};

const recreateWebcodecsDecoder = () => {
  if (!webcodecsConfig) {
    return false;
  }
  closeWebcodecsDecoder();
  webcodecsDecoder = new VideoDecoder({
    output: handleDecodedFrame,
    error: handleDecoderError
  });
  webcodecsDecoder.configure(webcodecsConfig);
  return true;
};

const resetWebcodecsState = () => {
  console.log("[renderer] resetWebcodecsState called, had", webcodecsSamples.length, "samples");
  stopWebcodecsPlayback();
  webcodecsReady = false;
  webcodecsSamples = [];
  webcodecsKeyframes = [];
  webcodecsDecodeQueue = [];
  webcodecsDecodedMax = -1;
  webcodecsPendingRender = null;
  webcodecsDurationMs = 0;
  webcodecsConfig = null;
  webcodecsFrameOrder.forEach((index) => {
    const bitmap = webcodecsFrameCache.get(index);
    if (bitmap) {
      bitmap.close();
    }
  });
  webcodecsFrameCache.clear();
  webcodecsFrameOrder = [];
  closeWebcodecsDecoder();
  if (mp4boxWorker) {
    mp4boxWorker.terminate();
    mp4boxWorker = null;
  }
  canvasContext.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
};

const findNearestKeyframe = (targetIndex) => {
  for (let i = webcodecsKeyframes.length - 1; i >= 0; i -= 1) {
    if (webcodecsKeyframes[i] <= targetIndex) {
      return webcodecsKeyframes[i];
    }
  }
  return 0;
};

const decodeWebcodecsUpTo = (targetIndex) => {
  if (!webcodecsDecoder) {
    return;
  }
  
  if (webcodecsFrameCache.has(targetIndex)) {
    renderWebcodecsFrame(targetIndex);
    return;
  }

  // Determine if we need to reset the decoder
  let needsReset = false;
  let startIndex = webcodecsDecodedMax + 1;
  
  if (targetIndex <= webcodecsDecodedMax) {
    // Seeking backwards - need to reset
    needsReset = true;
    startIndex = findNearestKeyframe(targetIndex);
  } else if (targetIndex > webcodecsDecodedMax + 60) {
    // Seeking far forward - reset from keyframe near target instead of decoding everything
    needsReset = true;
    startIndex = findNearestKeyframe(targetIndex);
  }
  
  if (needsReset) {
    webcodecsDecodeQueue = [];
    webcodecsDecodedMax = startIndex - 1;
    webcodecsFrameOrder.forEach((index) => {
      const bitmap = webcodecsFrameCache.get(index);
      if (bitmap) {
        bitmap.close();
      }
    });
    webcodecsFrameCache.clear();
    webcodecsFrameOrder = [];
    recreateWebcodecsDecoder();
    if (!webcodecsDecoder) {
      return;
    }
  }
  
  // Check decoder queue - don't overwhelm it (check AFTER potential reset)
  if (webcodecsDecoder.decodeQueueSize > 10) {
    return;
  }

  webcodecsPendingRender = targetIndex;
  
  for (let i = webcodecsDecodedMax + 1; i <= targetIndex; i += 1) {
    const sample = webcodecsSamples[i];
    if (!sample || !sample.data || sample.data.byteLength === 0) {
      continue;
    }
    
    // Ensure we start with a keyframe after decoder reset
    if (webcodecsDecodedMax === -1 && !sample.is_sync) {
      continue;
    }
    
    try {
      const chunk = new EncodedVideoChunk({
        type: sample.is_sync ? "key" : "delta",
        timestamp: Math.round((sample.cts * 1000000) / sample.timescale),
        duration: Math.round((sample.duration * 1000000) / sample.timescale),
        data: sample.data
      });
      webcodecsDecodeQueue.push(i);
      webcodecsDecoder.decode(chunk);
      webcodecsDecodedMax = i;
    } catch (err) {
      console.error(`[renderer] Decode error at sample ${i}:`, err.message);
      // If we hit a keyframe requirement error, reset decoder from nearest keyframe
      if (err.message && err.message.includes("key frame")) {
        const startIndex = findNearestKeyframe(i);
        webcodecsDecodeQueue = [];
        webcodecsDecodedMax = startIndex - 1;
        recreateWebcodecsDecoder();
      }
      break;
    }
  }
};

const buildDecoderConfig = (track) => {
  const config = {
    codec: track.codec,
    codedWidth: track.video.width,
    codedHeight: track.video.height
  };
  
  // Use the pre-extracted description from the worker (preferred)
  if (track.description) {
    config.description = track.description;
    console.log("[renderer] Using extracted description:", track.description.byteLength, "bytes");
  } else if (track.avcC && track.avcC.buffer) {
    config.description = track.avcC.buffer;
  } else if (track.hvcC && track.hvcC.buffer) {
    config.description = track.hvcC.buffer;
  } else if (track.avcC) {
    config.description = track.avcC;
  } else if (track.hvcC) {
    config.description = track.hvcC;
  }
  
  console.log("[renderer] Decoder config:", config.codec, config.codedWidth, "x", config.codedHeight, "description:", !!config.description);
  return config;
};

const configureWebcodecsDecoder = async (track) => {
  const config = buildDecoderConfig(track);
  
  // Check if description is required but missing (H.264/H.265)
  const needsDescription = config.codec.startsWith("avc") || config.codec.startsWith("hvc") || config.codec.startsWith("hev");
  if (needsDescription && !config.description) {
    console.error("[renderer] Codec", config.codec, "requires description but none was extracted");
    setWebcodecsStatus("Codec config extraction failed");
    return null;
  }
  
  let support = null;
  try {
    support = await VideoDecoder.isConfigSupported(config);
    console.log("[renderer] Config support check:", support?.supported, support?.config?.codec);
  } catch (error) {
    console.error("[renderer] isConfigSupported failed:", error);
    support = null;
  }
  
  if (!support || !support.supported) {
    // Try without description as fallback (won't work for H.264 but might for others)
    if (config.description) {
      console.warn("[renderer] Config not supported with description, trying without...");
      delete config.description;
      try {
        support = await VideoDecoder.isConfigSupported(config);
      } catch (error) {
        support = null;
      }
    }
  }
  
  if (!support || !support.supported) {
    console.error("[renderer] No supported config found for codec:", config.codec);
    return null;
  }

  webcodecsConfig = support.config || config;
  console.log("[renderer] Final config:", webcodecsConfig.codec, "description:", !!webcodecsConfig.description);
  return recreateWebcodecsDecoder() ? webcodecsConfig : null;
};

const createMp4Worker = () => {
  if (mp4boxWorker) {
    mp4boxWorker.terminate();
  }
  mp4boxWorker = new Worker("mp4box-worker.js", { type: "module" });
  mp4boxChunkResolvers = new Map();
  mp4boxChunkId = 0;
  return mp4boxWorker;
};

const initWebcodecsFromFile = async (file) => {
  resetWebcodecsState();
  if (!webcodecsSupported()) {
    setStatus("WebCodecs is not supported on this system.");
    setWebcodecsStatus("WebCodecs unsupported.");
    return false;
  }

  const filePath = file && file.path ? file.path : "";
  if (!filePath || !window.api || typeof window.api.getFileSize !== "function" || typeof window.api.readFileChunk !== "function") {
    setStatus("Failed to read video file.");
    setWebcodecsStatus("Video read failed: no file path or reader.");
    return false;
  }

  let selectedTrack = null;
  let mp4boxModule = null;

  try {
    setWebcodecsStatus("Loading MP4 parser...");
    mp4boxModule = await import("./node_modules/mp4box/dist/mp4box.all.js");
    console.log("[renderer-direct] mp4box module loaded, keys:", Object.keys(mp4boxModule).slice(0, 10).join(", "));
  } catch (importErr) {
    console.error("[renderer-direct] Failed to import mp4box:", importErr);
    setStatus("Failed to load MP4 parser.");
    setWebcodecsStatus("MP4 parser import failed: " + (importErr.message || importErr));
    return false;
  }

  const MP4Box = mp4boxModule.default || mp4boxModule;
  const DataStream = MP4Box.DataStream || mp4boxModule.DataStream;
  if (!MP4Box || typeof MP4Box.createFile !== "function") {
    setStatus("MP4 parser unavailable.");
    setWebcodecsStatus("MP4Box.createFile not found.");
    return false;
  }

  let mp4file = MP4Box.createFile();
  if (!mp4file) {
    setStatus("MP4 parser init failed.");
    return false;
  }
  console.log("[renderer-direct] mp4boxfile created");

  return new Promise(async (resolve) => {
    let resolved = false;
    const finish = (value) => {
      if (resolved) return;
      console.log("[renderer-direct] finish:", value);
      resolved = true;
      resolve(value);
    };

    mp4file.onError = (error) => {
      console.error("[renderer-direct] MP4Box error:", error);
      setStatus("MP4 parse error.");
      setWebcodecsStatus("MP4Box error: " + (error?.message || error));
      finish(false);
    };

    mp4file.onMoovStart = () => {
      console.log("[renderer-direct] moov atom parsing started");
      setWebcodecsStatus("MP4 moov atom found...");
    };

    mp4file.onReady = (info) => {
      console.log("[renderer-direct] onReady, tracks:", info.tracks.length);
      const vTrack = info.tracks.find((t) => t.type === "video");
      if (!vTrack) {
        setStatus("No video track found.");
        finish(false);
        return;
      }
      console.log("[renderer-direct] Video track:", vTrack.id, vTrack.codec, "samples in table:", vTrack.nb_samples);

      let description = null;
      try {
        const trak = mp4file.getTrackById(vTrack.id);
        if (trak?.mdia?.minf?.stbl?.stsd) {
          const entries = trak.mdia.minf.stbl.stsd.entries || [];
          for (const entry of entries) {
            const codecBox = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
            if (codecBox && DataStream) {
              const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
              codecBox.write(stream);
              if (stream.buffer.byteLength > 8) {
                description = new Uint8Array(stream.buffer, 8);
                console.log("[renderer-direct] description:", description.byteLength, "bytes");
              }
              break;
            }
          }
        }
      } catch (err) {
        console.warn("[renderer-direct] description extraction failed:", err);
      }

      selectedTrack = { ...vTrack, description };
      setWebcodecsStatus("Metadata ready, reading samples...");
      console.log("[renderer-direct] Track metadata captured");
    };

    try {
      const totalBytes = await window.api.getFileSize(filePath);
      if (!Number.isFinite(totalBytes) || totalBytes <= 0) {
        throw new Error("File size is " + totalBytes);
      }
      console.log("[renderer-direct] File size:", totalBytes);

      const chunkSize = 512 * 1024;
      let offset = 0;
      let chunkCount = 0;

      while (offset < totalBytes) {
        const length = Math.min(chunkSize, totalBytes - offset);
        const chunk = await window.api.readFileChunk(filePath, offset, length);
        const mp4Buffer = chunk instanceof ArrayBuffer ? chunk : (chunk && chunk.buffer);
        if (!mp4Buffer) throw new Error("Empty chunk at offset " + offset);

        mp4Buffer.fileStart = offset;
        const isLast = offset + length >= totalBytes;

        try {
          const nextOff = mp4file.appendBuffer(mp4Buffer, isLast);
          offset = typeof nextOff === "number" && nextOff > offset ? nextOff : offset + length;
        } catch (appendErr) {
          console.error("[renderer-direct] appendBuffer failed at offset", offset, ":", appendErr);
          throw appendErr;
        }

        chunkCount++;
        if (chunkCount % 50 === 0 || isLast) {
          const pct = Math.floor((offset / totalBytes) * 100);
          if (!resolved) setWebcodecsStatus("Reading MP4... " + pct + "%");
        }

        // Once we have the track metadata (onReady fired), stop feeding mp4box.
        if (selectedTrack) {
          console.log("[renderer-direct] Track metadata obtained after", chunkCount, "chunks");
          break;
        }

        if (chunkCount % 10 === 0) {
          await new Promise((r) => setTimeout(r, 0));
        }
      }

      if (!selectedTrack) {
        setStatus("No video track found in file.");
        finish(false);
        return;
      }

      // Get sample metadata from mp4box, then discard it entirely.
      // We read sample data directly from the file using offset/size metadata.
      const trackId = selectedTrack.id;
      const rawSamples = mp4file.getTrackSamplesInfo(trackId);
      const totalSamples = rawSamples ? rawSamples.length : 0;
      console.log("[renderer-direct] Track has", totalSamples, "samples in metadata");

      // Copy only the metadata fields we need, then free mp4box
      const sampleMeta = [];
      for (let i = 0; i < totalSamples; i++) {
        const s = rawSamples[i];
        sampleMeta.push({
          offset: s.offset, size: s.size, duration: s.duration,
          cts: s.cts, dts: s.dts, timescale: s.timescale,
          is_sync: s.is_sync, number: s.number ?? i,
          track_id: s.track_id ?? trackId
        });
      }
      try { mp4file.stop(); } catch (_) {}

      if (totalSamples === 0) {
        setStatus("Video has no samples.");
        finish(false);
        return;
      }

      // Read compressed frame data directly from the file in batched regions
      let extracted = 0;
      const BATCH = 500;

      for (let batchStart = 0; batchStart < totalSamples; batchStart += BATCH) {
        const batchEnd = Math.min(batchStart + BATCH, totalSamples);
        const regionStart = sampleMeta[batchStart].offset;
        const regionEnd = sampleMeta[batchEnd - 1].offset + sampleMeta[batchEnd - 1].size;
        const regionSize = regionEnd - regionStart;

        let regionView = null;
        if (regionSize > 0 && regionSize < 50 * 1024 * 1024) {
          try {
            const regionBuf = await window.api.readFileChunk(filePath, regionStart, regionSize);
            regionView = new Uint8Array(regionBuf);
          } catch (e) {
            console.warn("[renderer-direct] Batch read failed, offset:", regionStart, "size:", regionSize);
          }
        }

        for (let i = batchStart; i < batchEnd; i++) {
          const si = sampleMeta[i];
          let dataCopy = null;

          if (regionView && si.offset >= regionStart && si.offset + si.size <= regionStart + regionView.byteLength) {
            const lo = si.offset - regionStart;
            dataCopy = new Uint8Array(si.size);
            dataCopy.set(regionView.subarray(lo, lo + si.size));
          } else {
            try {
              const buf = await window.api.readFileChunk(filePath, si.offset, si.size);
              dataCopy = new Uint8Array(buf);
            } catch (_) { continue; }
          }

          if (si.is_sync) webcodecsKeyframes.push(webcodecsSamples.length);
          webcodecsSamples.push({
            data: dataCopy, size: si.size, duration: si.duration,
            cts: si.cts, dts: si.dts, timescale: si.timescale,
            is_sync: si.is_sync, number: si.number,
            track_id: si.track_id, dataType: "Uint8Array"
          });
          extracted++;
        }

        regionView = null;

        // Configure decoder once after first batch
        if (!resolved && extracted >= 30 && webcodecsDecoder === null) {
          console.log("[renderer-direct] Configuring decoder with", extracted, "samples");
          try {
            const config = await configureWebcodecsDecoder(selectedTrack);
            if (config) {
              console.log("[renderer-direct] Decoder configured");
              finish(true);
            } else {
              setStatus("WebCodecs codec not supported.");
              finish(false);
              return;
            }
          } catch (cfgErr) {
            console.error("[renderer-direct] Decoder config error:", cfgErr);
            setStatus("Decoder config failed: " + (cfgErr.message || cfgErr));
            finish(false);
            return;
          }
        }

        const pct = Math.floor((batchEnd / totalSamples) * 100);
        setWebcodecsStatus("Loading frames... " + pct + "% (" + extracted + "/" + totalSamples + ")");
        await new Promise((r) => setTimeout(r, 0));
      }

      console.log("[renderer-direct] All samples loaded:", extracted, "/", totalSamples);
      setWebcodecsStatus("");

      if (!resolved && extracted > 0) {
        try {
          const config = await configureWebcodecsDecoder(selectedTrack);
          if (config) finish(true);
          else { setStatus("WebCodecs not supported."); finish(false); }
        } catch (e) { setStatus("Decoder failed."); finish(false); }
      } else if (!resolved) {
        setStatus("Could not read video samples.");
        finish(false);
      }

    } catch (error) {
      console.error("[renderer-direct] Error:", error);
      setStatus("Failed to read/parse video.");
      setWebcodecsStatus("Error: " + (error.message || error));
      finish(false);
    }
  }).then((success) => {
    console.log("[renderer] Promise resolved, success:", success, "selectedTrack:", !!selectedTrack, "samples:", webcodecsSamples.length);
    
    if (!success || !selectedTrack) {
      console.log("[renderer] Returning false due to !success or !selectedTrack");
      return false;
    }
    if (!webcodecsSamples.length) {
      setStatus("No samples extracted for WebCodecs.");
      setWebcodecsStatus("No samples extracted.");
      return false;
    }

    // Calculate duration from track metadata
    let durationSeconds = selectedTrack.duration / selectedTrack.timescale;
    
    // Fallback: calculate from last sample
    if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
      const lastSample = webcodecsSamples[webcodecsSamples.length - 1];
      if (lastSample) {
        durationSeconds = (lastSample.cts + lastSample.duration) / lastSample.timescale;
      }
    }
    
    // Infer FPS from track timescale and sample duration
    if (!fpsLockInput.checked && durationSeconds > 0) {
      // Use timescale/duration ratio for accurate FPS
      const firstSample = webcodecsSamples[0];
      if (firstSample && firstSample.timescale && firstSample.duration) {
        const inferredFps = firstSample.timescale / firstSample.duration;
        if (inferredFps >= 1 && inferredFps <= 240) {
          fpsInput.value = inferredFps.toFixed(3);
          frameDurationMs = 1000 / inferredFps;
        } else {
          fpsInput.value = "30";
          frameDurationMs = 1000 / 30;
        }
      } else {
        fpsInput.value = "30";
        frameDurationMs = 1000 / 30;
      }
    } else if (fpsLockInput.checked) {
      applyLockedFps();
    }

    webcodecsDurationMs = durationSeconds * 1000;
    videoCanvas.width = selectedTrack.video.width;
    videoCanvas.height = selectedTrack.video.height;
    
    webcodecsReady = true;
    updateDurationDisplay();
    updateTrackbar();
    
    // Decode first frame
    try {
      if (webcodecsSamples.length === 0) {
        setStatus("No video samples available.");
        return false;
      }
      decodeWebcodecsUpTo(0);
      setStatus("WebCodecs ready.");
      setWebcodecsStatus("");
      return true;
    } catch (err) {
      console.error("[renderer] Failed to decode first frame:", err);
      setStatus("WebCodecs decode failed; falling back to HTML5.");
      setWebcodecsStatus(`Decode error: ${err.message || err}`);
      return false;
    }
  }).catch((err) => {
    console.error("[renderer] WebCodecs initialization failed:", err);
    setStatus("WebCodecs failed; falling back to HTML5.");
    setWebcodecsStatus(`Init error: ${err.message || err}`);
    return false;
  });
};

const csvEscape = (value) => {
  const stringValue = String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
};

const getVideoTimeMs = () => Math.round(video.currentTime * 1000);

const getFrameTimeMs = (frameIndex) => {
  // Use frame index * duration for consistent monotonic time display
  // Note: Videos with B-frames have non-monotonic CTS in decode order,
  // so using actual CTS would cause time to jump around when stepping frames
  return frameIndex * frameDurationMs;
};

const updateTimeDisplay = () => {
  currentTimeEl.textContent = formatTime(getFrameTimeMs(currentFrameIndex));
  updateAlFrameInfo();
};

const seekHtml5ToMs = (milliseconds) => {
  if (!video.src) {
    setStatus("Load a video first.");
    return;
  }
  const target = Math.min(
    Math.max(0, milliseconds / 1000),
    Number.isFinite(video.duration) ? video.duration : Infinity
  );
  video.currentTime = target;
  updateTimeDisplay();
  updateTrackbar();
};

const seekToMs = (milliseconds) => {
  if (webcodecsMode) {
    const targetIndex = Math.round(milliseconds / frameDurationMs);
    seekToFrame(targetIndex);
    return;
  }
  seekHtml5ToMs(milliseconds);
};

const seekToFrame = (frameIndex) => {
  const maxIndex = webcodecsSamples.length > 0 ? webcodecsSamples.length - 1 : 0;
  const safeIndex = Math.max(0, Math.min(frameIndex, maxIndex));
  currentFrameIndex = safeIndex;
  if (webcodecsMode) {
    decodeWebcodecsUpTo(safeIndex);
    return;
  }
  pendingSeekFrame = safeIndex;
  pendingSeekTime = (safeIndex * frameDurationMs) / 1000;
  suppressTimeUpdate = true;
  seekHtml5ToMs(safeIndex * frameDurationMs);
};

const renderBehaviors = () => {
  behaviorList.innerHTML = "";

  if (behaviors.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No behaviors yet.";
    behaviorList.appendChild(empty);
    updateMergeButton();
    renderBinarizeChecklist();
    renderAlBehaviorButtons();
    updateAlFrameInfo();
    return;
  }

  behaviors.forEach((behavior) => {
    const row = document.createElement("div");
    row.className = "behavior-row";
    if (activeStarts.has(behavior.id)) {
      row.classList.add("behavior-active");
    }

    const mergeCheck = document.createElement("input");
    mergeCheck.type = "checkbox";
    mergeCheck.className = "merge-check";
    mergeCheck.title = "Select to merge";
    mergeCheck.checked = selectedBehaviorIds.has(behavior.id);
    mergeCheck.addEventListener("change", () => {
      if (mergeCheck.checked) selectedBehaviorIds.add(behavior.id);
      else selectedBehaviorIds.delete(behavior.id);
      updateMergeButton();
    });

    const keyBadge = document.createElement("span");
    keyBadge.className = "key-badge";
    keyBadge.textContent = behavior.key;

    const name = document.createElement("span");
    name.className = "behavior-name";
    name.textContent = behavior.name;
    name.title = "Click to rename";
    name.style.cursor = "pointer";
    name.addEventListener("click", () => {
      const input = document.createElement("input");
      input.type = "text";
      input.value = behavior.name;
      input.className = "behavior-name-edit";
      input.style.width = Math.max(80, name.offsetWidth + 10) + "px";

      const commit = () => {
        const newName = normalizeBehaviorName(input.value);
        if (!newName) { renderBehaviors(); return; }
        const clash = behaviors.find(
          (b) => b !== behavior && behaviorNameKey(b.name) === behaviorNameKey(newName)
        );
        if (clash) {
          setStatus(`"${clash.name}" already exists; rename canceled.`);
          renderBehaviors();
          return;
        }
        if (newName !== behavior.name) {
          behavior.name = newName;
          for (const bout of bouts) {
            if (bout.behaviorId === behavior.id) bout.behaviorName = newName;
          }
          saveBehaviors();
          saveProgress();
          renderBouts();
        }
        renderBehaviors();
      };

      input.addEventListener("blur", commit);
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); input.blur(); }
        else if (e.key === "Escape") { input.value = behavior.name; input.blur(); }
      });

      name.replaceWith(input);
      input.focus();
      input.select();
    });

    const activeIndicator = document.createElement("span");
    activeIndicator.className = "active-indicator";
    activeIndicator.textContent = activeStarts.has(behavior.id)
      ? "Recording"
      : "";

    const removeButton = document.createElement("button");
    removeButton.className = "ghost";
    removeButton.textContent = "Remove";
    removeButton.addEventListener("click", () => {
      behaviors = behaviors.filter((item) => item.id !== behavior.id);
      activeStarts.delete(behavior.id);
      selectedBehaviorIds.delete(behavior.id);
      saveBehaviors();
      renderBehaviors();
      renderBouts();
    });

    row.appendChild(mergeCheck);
    row.appendChild(keyBadge);
    row.appendChild(name);
    row.appendChild(activeIndicator);
    row.appendChild(removeButton);
    behaviorList.appendChild(row);
  });
  updateMergeButton();
  renderBinarizeChecklist();
  renderAlBehaviorButtons();
  updateAlFrameInfo();
};

// Render the Train-tab "Positive class (binarize)" checklist from the current
// behaviors. Tracked by behavior NAME (stable across project reloads, and the
// name is what the training payload needs).
function renderBinarizeChecklist() {
  const container = document.getElementById("trainBinarizeList");
  if (!container) return;
  container.innerHTML = "";

  if (behaviors.length === 0) {
    container.innerHTML = '<p class="muted">No behaviors yet.</p>';
    return;
  }
  // Drop names that no longer exist (renamed/removed/merged behaviors).
  const names = new Set(behaviors.map((b) => b.name));
  for (const n of [...binarizeSelected]) if (!names.has(n)) binarizeSelected.delete(n);

  for (const b of behaviors) {
    const row = document.createElement("label");
    row.className = "checkbox-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = binarizeSelected.has(b.name);
    cb.addEventListener("change", () => {
      if (cb.checked) binarizeSelected.add(b.name);
      else binarizeSelected.delete(b.name);
    });
    const span = document.createElement("span");
    span.textContent = b.name;
    row.append(cb, span);
    container.appendChild(row);
  }
}

const renderBouts = () => {
  boutsTable.innerHTML = "";

  if (bouts.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 4;
    cell.className = "muted";
    cell.textContent = "No bouts recorded.";
    row.appendChild(cell);
    boutsTable.appendChild(row);
    return;
  }

  bouts.forEach((bout, index) => {
    const row = document.createElement("tr");

    const startCell = document.createElement("td");
    startCell.textContent = formatTime(getFrameTimeMs(bout.startFrame));
    startCell.className = "time-link";
    startCell.title = "Jump to start";
    startCell.addEventListener("click", () => {
      seekToFrame(bout.startFrame);
    });

    const endCell = document.createElement("td");
    endCell.textContent = formatTime(getFrameTimeMs(bout.endFrame));
    endCell.className = "time-link";
    endCell.title = "Jump to end";
    endCell.addEventListener("click", () => {
      seekToFrame(bout.endFrame);
    });

    const behaviorCell = document.createElement("td");
    behaviorCell.textContent = bout.behaviorName;
    behaviorCell.className = "time-link";
    behaviorCell.title = "Click to reassign behavior";
    behaviorCell.addEventListener("click", () => {
      const select = document.createElement("select");
      for (const b of behaviors) {
        const opt = document.createElement("option");
        opt.value = b.id;
        opt.textContent = b.name;
        if (b.id === bout.behaviorId) opt.selected = true;
        select.appendChild(opt);
      }
      const commit = () => {
        const chosen = behaviors.find((b) => b.id === select.value);
        if (chosen && chosen.id !== bout.behaviorId) {
          bout.behaviorId = chosen.id;
          bout.behaviorName = chosen.name;
          saveProgress();
        }
        renderBouts();
      };
      select.addEventListener("change", commit);
      select.addEventListener("blur", () => renderBouts());
      behaviorCell.textContent = "";
      behaviorCell.appendChild(select);
      select.focus();
    });

    const actionsCell = document.createElement("td");
    const editButton = document.createElement("button");
    editButton.className = "ghost";
    editButton.textContent = "Edit";
    editButton.addEventListener("click", () => {
      editBoutRow(index);
    });
    const removeButton = document.createElement("button");
    removeButton.className = "ghost";
    removeButton.textContent = "Delete";
    removeButton.addEventListener("click", () => {
      bouts.splice(index, 1);
      renderBouts();
      saveProgress();
    });
    actionsCell.appendChild(editButton);
    actionsCell.appendChild(removeButton);

    row.appendChild(startCell);
    row.appendChild(endCell);
    row.appendChild(behaviorCell);
    row.appendChild(actionsCell);
    boutsTable.appendChild(row);
  });

  // The bout covering the current frame may have just changed, so refresh the
  // label readout (panel + on-video HUD). Without this, closing a bout on a
  // paused frame leaves the overlay showing the pre-edit state until you scrub.
  updateAlFrameInfo();
};

const saveBehaviors = () => {
  localStorage.setItem("behaviors", JSON.stringify(behaviors));
  try {
    if (window.projectManager && window.projectManager.isActive()) {
      window.projectManager.updateBehaviors(behaviors);
    }
  } catch { /* project manager not ready */ }
};

function loadBehaviors() {
  try {
    const stored = localStorage.getItem("behaviors");
    if (!stored) {
      return [];
    }
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn("Failed to load behaviors", error);
    return [];
  }
}

// ── Behavior-name normalization & de-duplication ──────────────────────────
// Trim ends and collapse internal whitespace runs so "face  groom",
// "face groom ", and "face groom" are treated as one behavior.
const normalizeBehaviorName = (s) =>
  String(s == null ? "" : s).trim().replace(/\s+/g, " ");

// Canonical matching key: normalized + case-insensitive.
const behaviorNameKey = (s) => normalizeBehaviorName(s).toLowerCase();

const findBehaviorByName = (name) => {
  const key = behaviorNameKey(name);
  return behaviors.find((b) => behaviorNameKey(b.name) === key);
};

// Merge existing behaviors that share a canonical name: keep the first, remap
// every bout (current video + all stored per-video progress) to it, and clean
// up display names. Returns the number of duplicate behaviors removed.
const mergeDuplicateBehaviors = () => {
  const survivors = new Map(); // canonical key -> survivor behavior
  const idMap = new Map();     // duplicate id -> survivor id
  const kept = [];
  let renamed = false;

  for (const b of behaviors) {
    const key = behaviorNameKey(b.name);
    const survivor = survivors.get(key);
    if (survivor) {
      idMap.set(b.id, survivor.id);
    } else {
      const norm = normalizeBehaviorName(b.name);
      if (norm !== b.name) { b.name = norm; renamed = true; }
      survivors.set(key, b);
      kept.push(b);
    }
  }

  if (idMap.size === 0 && !renamed) return 0;

  behaviors = kept;
  const nameById = new Map(kept.map((b) => [b.id, b.name]));

  const remap = (list) => {
    let changed = false;
    for (const bout of list) {
      const sid = idMap.get(bout.behaviorId);
      if (sid) {
        bout.behaviorId = sid;
        bout.behaviorName = nameById.get(sid) || bout.behaviorName;
        changed = true;
      } else if (nameById.has(bout.behaviorId)) {
        const nn = nameById.get(bout.behaviorId);
        if (bout.behaviorName !== nn) { bout.behaviorName = nn; changed = true; }
      }
    }
    return changed;
  };

  remap(bouts);

  // Rewrite every video's stored progress so merges apply project-wide.
  try {
    const vids = (typeof videoList !== "undefined" && Array.isArray(videoList)) ? videoList : [];
    for (const v of vids) {
      const prog = loadProgress(v.name);
      if (prog && Array.isArray(prog.bouts) && remap(prog.bouts)) {
        localStorage.setItem(getProgressKey(v.name), JSON.stringify(prog));
      }
    }
  } catch { /* ignore */ }

  saveBehaviors();
  return idMap.size;
};

// Reassign every bout from `fromIds` to `target` (id + name), across the current
// video and all stored per-video progress.
const reassignBoutsToBehavior = (fromIds, target) => {
  const fromSet = new Set(fromIds);
  const apply = (list) => {
    let changed = false;
    for (const bout of list) {
      if (fromSet.has(bout.behaviorId)) {
        bout.behaviorId = target.id;
        bout.behaviorName = target.name;
        changed = true;
      }
    }
    return changed;
  };
  apply(bouts);
  try {
    const vids = (typeof videoList !== "undefined" && Array.isArray(videoList)) ? videoList : [];
    for (const v of vids) {
      const prog = loadProgress(v.name);
      if (prog && Array.isArray(prog.bouts) && apply(prog.bouts)) {
        localStorage.setItem(getProgressKey(v.name), JSON.stringify(prog));
      }
    }
  } catch { /* ignore */ }
};

// Destructively merge `ids` into the behavior `targetId`: reassign their bouts
// to the target and remove the others.
const mergeBehaviors = (ids, targetId) => {
  const target = behaviors.find((b) => b.id === targetId);
  if (!target) return;
  const fromIds = ids.filter((id) => id !== targetId && behaviors.some((b) => b.id === id));
  if (fromIds.length === 0) return;

  reassignBoutsToBehavior(fromIds, target);
  const removeSet = new Set(fromIds);
  behaviors = behaviors.filter((b) => !removeSet.has(b.id));
  for (const id of removeSet) { activeStarts.delete(id); selectedBehaviorIds.delete(id); }
  selectedBehaviorIds.delete(targetId);

  saveBehaviors();
  renderBehaviors();
  renderBouts();
  saveProgress();
  setStatus(`Merged ${fromIds.length + 1} behaviors into "${target.name}".`);
};

// Show/hide the "Merge selected" button based on how many behaviors are ticked.
function updateMergeButton() {
  const btn = document.getElementById("mergeBehaviors");
  if (!btn) return;
  const count = behaviors.filter((b) => selectedBehaviorIds.has(b.id)).length;
  btn.classList.toggle("hidden", count < 2);
  btn.textContent = count >= 2 ? `Merge ${count} selected` : "Merge selected";
}

// Modal to pick which selected behavior to keep. Resolves to its id, or null.
function chooseMergeTarget(selected) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";
    const modal = document.createElement("div");
    modal.className = "modal";

    const title = document.createElement("h3");
    title.textContent = "Merge behaviors";
    const body = document.createElement("p");
    body.textContent =
      `Combine ${selected.length} behaviors into one. Their bouts are reassigned ` +
      `to the kept behavior, and the others are removed. This can't be undone.`;

    const label = document.createElement("label");
    label.textContent = "Keep: ";
    const select = document.createElement("select");
    for (const b of selected) {
      const opt = document.createElement("option");
      opt.value = b.id;
      opt.textContent = `${b.name} (key ${b.key})`;
      select.appendChild(opt);
    }
    label.appendChild(select);

    const actions = document.createElement("div");
    actions.className = "modal-actions";
    const mergeBtn = document.createElement("button");
    mergeBtn.className = "primary";
    mergeBtn.textContent = "Merge";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "ghost";
    cancelBtn.textContent = "Cancel";

    const close = (val) => {
      document.removeEventListener("keydown", onKey, true);
      overlay.remove();
      resolve(val);
    };
    const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(null); } };
    mergeBtn.addEventListener("click", () => close(select.value));
    cancelBtn.addEventListener("click", () => close(null));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });
    document.addEventListener("keydown", onKey, true);

    actions.append(mergeBtn, cancelBtn);
    modal.append(title, body, label, actions);
    overlay.append(modal);
    document.body.append(overlay);
    select.focus();
  });
}

document.getElementById("mergeBehaviors").addEventListener("click", async () => {
  const selected = behaviors.filter((b) => selectedBehaviorIds.has(b.id));
  if (selected.length < 2) return;
  const targetId = await chooseMergeTarget(selected);
  if (!targetId) return;
  mergeBehaviors(selected.map((b) => b.id), targetId);
});

const getProgressKey = (videoName) => `progress_${videoName}`;

let saveProgressTimeout = null;
const saveProgressDebounced = () => {
  if (saveProgressTimeout) {
    clearTimeout(saveProgressTimeout);
  }
  saveProgressTimeout = setTimeout(saveProgress, 500);
};

const saveProgress = () => {
  if (!currentVideoName) return;
  const progress = {
    frameIndex: currentFrameIndex,
    bouts: bouts,
    activeStarts: Array.from(activeStarts.entries())
  };
  try {
    localStorage.setItem(getProgressKey(currentVideoName), JSON.stringify(progress));
  } catch (error) {
    console.warn("Failed to save progress", error);
  }
  // Auto-save labels to project directory
  try {
    if (window.projectManager && window.projectManager.isActive() && currentVideoName) {
      const csv = generateTrainingLabelCsv(bouts);
      window.projectManager.saveVideoLabels(currentVideoName, csv).catch(() => {});
    }
  } catch { /* project manager not ready or generators not yet defined */ }
  // Update sidebar bout counts
  try { renderVideoSidebar(); } catch { /* sidebar not ready yet */ }
  // Rebuild misclassification overlay when bouts change
  try {
    if (alState && alState.results) {
      buildMisclassifiedArray();
      renderUncertaintyOverlay();
      updateAlStats();
    }
  } catch { /* alState not yet initialized */ }
};

const loadProgress = (videoName) => {
  try {
    const stored = localStorage.getItem(getProgressKey(videoName));
    if (!stored) return null;
    return JSON.parse(stored);
  } catch (error) {
    console.warn("Failed to load progress", error);
    return null;
  }
};

const restoreProgress = (videoName) => {
  const progress = loadProgress(videoName);
  if (!progress) return;
  
  if (Array.isArray(progress.bouts)) {
    bouts = progress.bouts;
  }
  if (Array.isArray(progress.activeStarts)) {
    activeStarts = new Map(progress.activeStarts);
  }
  renderBehaviors();
  renderBouts();
  
  // Restore frame position after a short delay to ensure video is ready
  if (typeof progress.frameIndex === "number" && progress.frameIndex > 0) {
    setTimeout(() => {
      seekToFrame(progress.frameIndex);
    }, 100);
  }
  
  setStatus(`Restored progress for ${videoName}.`);
};

// ── Video sidebar functions ───────────────────────────────────────────────

const saveVideoList = () => {
  const serializable = videoList.map(v => ({ name: v.name, path: v.path, isWebCodecs: !!v.isWebCodecs }));
  localStorage.setItem("videoList", JSON.stringify(serializable));
};

const getBoutCountForVideo = (videoName) => {
  if (videoName === currentVideoName) return bouts.length;
  const progress = loadProgress(videoName);
  return progress && Array.isArray(progress.bouts) ? progress.bouts.length : 0;
};

// Set the .content grid columns based on whether the video sidebar and the
// player are shown. The player is hidden on the Train tab (it never uses video).
const updateContentLayout = () => {
  const contentEl = document.querySelector(".content");
  if (!contentEl) return;
  // The player (and with it the splitter) is hidden on the Train tab, which has
  // no video. The track list has to match the number of visible children,
  // otherwise the leftovers wrap into implicit rows.
  const playerShown = document.body.dataset.tab !== "train";
  const splitter = document.getElementById("playerSplitter");
  if (splitter) splitter.style.display = playerShown ? "" : "none";
  contentEl.style.gridTemplateColumns = playerShown
    ? "auto var(--player-col) var(--splitter-w) 1fr"
    : "auto 1fr";
};

// ── Player / panel split sizing ───────────────────────────────────────────
// The player's width is a grid `fr` share (--player-col) weighed against the
// side panels' 1fr. Dragging the splitter rewrites that share, and it's
// persisted, so a layout you set up survives a restart. Height needs no control
// — the video shell flex-fills whatever the window leaves it.

const DEFAULT_PLAYER_COL = 2.6;
const MIN_PANE_PX = 260;      // neither pane may be dragged narrower than this
let playerCol = DEFAULT_PLAYER_COL;

function applyPlayerCol() {
  document.documentElement.style.setProperty("--player-col", `${playerCol}fr`);
}

function setPlayerCol(value) {
  if (!Number.isFinite(value)) return;
  playerCol = Math.min(8, Math.max(0.3, value));
  applyPlayerCol();
}

(() => {
  const splitter = document.getElementById("playerSplitter");
  if (!splitter) return;
  const playerEl = document.querySelector(".player");
  const panelsEl = document.querySelector(".side-panels");
  if (!playerEl || !panelsEl) return;
  let dragging = false;

  const onMove = (e) => {
    if (!dragging) return;
    // Convert the pointer position into an fr ratio between the two flexible
    // columns, measuring across the span they jointly occupy.
    const left = playerEl.getBoundingClientRect().left;
    const right = panelsEl.getBoundingClientRect().right;
    const total = right - left;
    if (total <= 0) return;
    const playerW = Math.max(MIN_PANE_PX, Math.min(e.clientX - left, total - MIN_PANE_PX));
    const panelsW = total - playerW;
    if (panelsW > 0) setPlayerCol(playerW / panelsW);
  };

  const stop = () => {
    if (!dragging) return;
    dragging = false;
    splitter.classList.remove("dragging");
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    debouncedSave();
  };

  splitter.addEventListener("pointerdown", (e) => {
    dragging = true;
    splitter.classList.add("dragging");
    // Suppress text selection and hold the resize cursor for the whole drag,
    // even when the pointer runs ahead of the splitter.
    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    splitter.setPointerCapture(e.pointerId);
    e.preventDefault();
  });
  splitter.addEventListener("pointermove", onMove);
  splitter.addEventListener("pointerup", stop);
  splitter.addEventListener("pointercancel", stop);
  splitter.addEventListener("dblclick", () => {
    setPlayerCol(DEFAULT_PLAYER_COL);
    debouncedSave();
    setStatus("Player width reset.");
  });
})();

const renderVideoSidebar = () => {
  const sidebar = document.getElementById("videoSidebar");
  const listEl = document.getElementById("videoListEl");

  // Keep the sidebar (and its "+" Add button) visible even with no videos —
  // otherwise deleting the last video would hide the only way to add more.
  sidebar.classList.remove("hidden");
  updateContentLayout();

  listEl.innerHTML = "";
  if (videoList.length === 0) {
    const li = document.createElement("li");
    li.className = "video-list-empty muted";
    li.textContent = "No videos yet — click + to add.";
    listEl.appendChild(li);
    return;
  }
  videoList.forEach((entry) => {
    const li = document.createElement("li");
    li.className = "video-list-item" + (entry.name === currentVideoName ? " active" : "");
    li.title = entry.path || entry.name;

    const nameSpan = document.createElement("span");
    nameSpan.className = "video-name";
    nameSpan.textContent = entry.name;

    const countSpan = document.createElement("span");
    countSpan.className = "bout-count";
    const count = getBoutCountForVideo(entry.name);
    countSpan.textContent = count > 0 ? `(${count})` : "";

    const removeBtn = document.createElement("button");
    removeBtn.className = "remove-btn";
    removeBtn.textContent = "\u00d7";
    removeBtn.title = "Remove video";
    removeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeVideoFromList(entry.name);
    });

    li.appendChild(nameSpan);
    li.appendChild(countSpan);
    li.appendChild(removeBtn);
    li.addEventListener("click", () => switchToVideo(entry.name));
    listEl.appendChild(li);
  });
};

const addVideoToList = (file) => {
  const name = file.name;
  const existing = videoList.find(v => v.name === name);
  if (existing) {
    // Already tracked — re-link to the newly chosen path if it changed.
    if (file.path && existing.path !== file.path) {
      existing.path = file.path;
      existing.isWebCodecs = useWebCodecsInput.checked;
      saveVideoList();
      renderVideoSidebar();
      try {
        if (window.projectManager && window.projectManager.isActive()) {
          window.projectManager.setVideoPath(name, file.path);
        }
      } catch { /* project manager not ready */ }
    }
    return;
  }
  videoList.push({
    name,
    path: file.path || "",
    isWebCodecs: useWebCodecsInput.checked
  });
  saveVideoList();
  renderVideoSidebar();
  // Add to project manifest
  try {
    if (window.projectManager && window.projectManager.isActive() && file.path) {
      window.projectManager.addVideoToManifest(file.path);
    }
  } catch { /* project manager not ready */ }
};

const removeVideoFromList = (videoName) => {
  videoList = videoList.filter(v => v.name !== videoName);
  saveVideoList();
  localStorage.removeItem(getProgressKey(videoName));
  // Remove from project manifest
  try {
    if (window.projectManager && window.projectManager.isActive()) {
      const id = window.projectManager.getVideoIdFromName(videoName);
      window.projectManager.removeVideoFromManifest(id);
    }
  } catch { /* project manager not ready */ }
  if (videoName === currentVideoName) {
    if (videoList.length > 0) {
      switchToVideo(videoList[0].name);
    } else {
      resetSession();
      currentVideoName = "";
      currentFile = null;
      if (video.src) { URL.revokeObjectURL(video.src); video.removeAttribute("src"); }
      renderVideoSidebar();
    }
  } else {
    renderVideoSidebar();
  }
};

const switchToVideo = (videoName) => {
  if (videoName === currentVideoName) {
    // Re-selecting the video that's currently loading means they changed their
    // mind back to it — drop any queued switch so we don't jump away when it's done.
    pendingVideoSwitch = null;
    return;
  }

  // Never start a second WebCodecs load while one is in flight — they share the
  // decoder/sample state and would corrupt each other (frames from different
  // videos mixed together, or a decoder error). Remember the latest requested
  // video and open it when the current load finishes. The user can keep clicking;
  // whichever they land on last is the one that opens.
  if (webcodecsLoading) {
    pendingVideoSwitch = videoName;
    setStatus(`Still loading ${currentVideoName || "a video"} — will open ${videoName} next.`);
    return;
  }

  // Save current progress before switching
  saveProgress();

  const entry = videoList.find(v => v.name === videoName);
  if (!entry) return;

  // Stop playback
  if (webcodecsMode && webcodecsPlaying) {
    togglePlayPause();
  } else if (!video.paused) {
    video.pause();
    playPauseButton.textContent = "Play";
  }

  // Build file-like object and load
  const fileObj = { name: entry.name, path: entry.path };
  currentFile = fileObj;

  if (entry.isWebCodecs && webcodecsSupported()) {
    useWebCodecsInput.checked = true;
    loadWebcodecsFile(fileObj);
  } else {
    useWebCodecsInput.checked = false;
    setWebcodecsMode(false);
    loadHtml5File(fileObj);
  }

  localStorage.setItem("lastActiveVideo", videoName);
  renderVideoSidebar();
  // Point the Pose panel (and the skeleton overlay) at this video's pose file.
  try { resetPoseForVideo(); } catch { /* pose panel not initialized yet */ }
};

const nextBehaviorKey = () => {
  const used = new Set(behaviors.map((behavior) => behavior.key));
  return KEY_ORDER.find((key) => !used.has(key));
};

const addBehavior = () => {
  const name = normalizeBehaviorName(behaviorNameInput.value);
  if (!name) {
    setStatus("Enter a behavior name before adding.");
    return;
  }

  const existing = findBehaviorByName(name);
  if (existing) {
    setStatus(`"${existing.name}" already exists (key ${existing.key}).`);
    behaviorNameInput.value = "";
    return;
  }

  const key = nextBehaviorKey();
  if (!key) {
    setStatus("All number keys are already assigned.");
    return;
  }

  behaviors.push({
    id: crypto.randomUUID(),
    name,
    key
  });

  behaviorNameInput.value = "";
  saveBehaviors();
  renderBehaviors();
  setStatus(`Assigned "${name}" to key ${key}.`);
};

const resetSession = () => {
  bouts = [];
  activeStarts = new Map();
  renderBehaviors();
  renderBouts();
  // Drop the previous video's predictions/overlays (alState may not be
  // initialized yet during early startup — guard against the TDZ).
  try { clearAlPredictions(); } catch { /* alState not ready */ }
};

const loadHtml5File = (file) => {
  if (video.src) {
    URL.revokeObjectURL(video.src);
  }
  currentVideoName = file.name;
  localStorage.setItem("lastActiveVideo", file.name);
  frameDurationMs = 1000 / (Number(fpsInput.value) || 30);
  lastFrameMediaTime = null;
  currentFrameIndex = 0;
  const filePath = file.path || "";
  const fileUrl = filePath ? toFileUrl(filePath) : "";
  if (fileUrl) {
    video.src = fileUrl;
  } else {
    video.src = URL.createObjectURL(file);
  }
  video.load();
  resetSession();
  restoreProgress(file.name);
  restorePredictions(file.name);
  try { resetPoseForVideo(); } catch { /* pose panel not initialized yet */ }
  setStatus(`Loaded ${file.name}`);
};

// Containers mp4box can't demux (it's ISO/MP4-only). The frame-accurate
// WebCodecs path needs these remuxed to MP4 first (see ensureRemuxedForWebcodecs).
// Backend ops (preprocess/AL/export) read the original via ffmpeg and need no remux.
const REMUX_EXTS = new Set(["mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts", "ogv"]);
const remuxCache = new Map(); // original path -> remuxed MP4 path (per session)

function needsRemux(name) {
  const m = /\.([^.]+)$/.exec(name || "");
  return m ? REMUX_EXTS.has(m[1].toLowerCase()) : false;
}

// Ensure an MP4 the WebCodecs path can decode exists for a non-MP4 source.
// Returns the MP4 path, or null if it can't/shouldn't proceed (user declined,
// no ffmpeg bridge, or remux failed). Stream-copy keeps frames identical so the
// decoded frame indices stay aligned with the backend's view of the original.
async function ensureRemuxedForWebcodecs(file) {
  if (!file || !file.path) return null;

  // Reuse within the session, and across sessions if the file is still on disk.
  const cached = remuxCache.get(file.path);
  if (cached && await fileExists(cached)) return cached;
  if (cached) remuxCache.delete(file.path);

  if (!window.api || typeof window.api.remuxVideo !== "function") {
    setWebcodecsStatus("This container needs conversion to MP4, but ffmpeg is unavailable.");
    return null;
  }

  const proceed = confirm(
    `"${file.name}" isn't an MP4. Frame-accurate (WebCodecs) mode can only ` +
    `demux MP4, so it needs converting.\n\n` +
    `Convert now? A fast stream-copy is used when possible (no re-encode, no ` +
    `quality loss); an MP4 is written next to the original and reused afterward.`
  );
  if (!proceed) return null;

  setWebcodecsStatus("Converting to MP4…");
  let res;
  try {
    res = await window.api.remuxVideo({ inputPath: file.path });
  } catch (e) {
    setWebcodecsStatus("Conversion failed: " + (e && e.message ? e.message : e));
    return null;
  }
  if (!res || !res.success) {
    const msg = (res && (res.error || res.stderr)) || "unknown error";
    setWebcodecsStatus("Conversion failed: " + String(msg).split("\n").pop());
    return null;
  }
  remuxCache.set(file.path, res.outputPath);
  return res.outputPath;
}

const loadWebcodecsFile = async (file) => {
  webcodecsOperationInProgress = true;
  currentVideoName = file.name;
  localStorage.setItem("lastActiveVideo", file.name);
  currentFrameIndex = 0;
  webcodecsDurationMs = 0;
  resetSession();
  webcodecsLoading = true;
  setStatus(`Loading ${file.name} with WebCodecs...`);
  setWebcodecsStatus("Loading...");

  // mp4box (the WebCodecs demuxer) is MP4-only. For MKV and other non-ISO
  // containers, decode a remuxed MP4 instead — but keep currentVideoName/
  // currentFile as the ORIGINAL so labels, progress, and backend ops (which read
  // the source via ffmpeg) stay keyed to it. Stream-copy keeps frame indices aligned.
  let decodeFile = file;
  if (needsRemux(file.name)) {
    const remuxedPath = await ensureRemuxedForWebcodecs(file);
    if (!remuxedPath) {
      // Declined or failed → best-effort HTML5 playback of the original.
      webcodecsLoading = false;
      webcodecsOperationInProgress = false;
      useWebCodecsInput.checked = false;
      setWebcodecsMode(false);
      loadHtml5File(file);
      if (pendingVideoSwitch) {
        const next = pendingVideoSwitch;
        pendingVideoSwitch = null;
        if (next !== currentVideoName) switchToVideo(next);
      }
      return;
    }
    decodeFile = { name: file.name, path: remuxedPath };
  }

  let success = false;
  try {
    success = await initWebcodecsFromFile(decodeFile);
  } catch (err) {
    // A thrown/rejected load must NEVER wedge the loader. Without this catch the
    // rejection would skip the `webcodecsLoading = false` reset below, leaving the
    // flag stuck true — and the file-input change handler's `if (webcodecsLoading)
    // return;` guard would then silently ignore every subsequent video pick, so the
    // UI just sits on "Loading..." forever.
    console.error("[renderer] WebCodecs load threw:", err);
    setWebcodecsStatus("WebCodecs error: " + (err && err.message ? err.message : err));
    success = false;
  } finally {
    webcodecsLoading = false;
  }
  console.log("[renderer] initWebcodecsFromFile returned:", success, "samples:", webcodecsSamples.length);
  if (!success) {
    webcodecsOperationInProgress = false;
    setStatus("Falling back to HTML5 mode.");
    if (!webcodecsStatus.textContent) {
      setWebcodecsStatus("WebCodecs failed.");
    }
    useWebCodecsInput.checked = false;
    setWebcodecsMode(false);
    loadHtml5File(file);
  } else {
    console.log("[renderer] WebCodecs success, enabling mode. Samples:", webcodecsSamples.length);
    setWebcodecsStatus("");
    setWebcodecsMode(true);
    webcodecsOperationInProgress = false;
    restoreProgress(file.name);
    restorePredictions(file.name);
    try { resetPoseForVideo(); } catch { /* pose panel not initialized yet */ }
    video.removeAttribute("src");
    video.load();
  }

  // Honor any video the user queued while this load was running (latest wins).
  // webcodecsLoading is already false, so this re-entrant switch starts a fresh,
  // non-overlapping load rather than racing the one that just finished.
  if (pendingVideoSwitch) {
    const next = pendingVideoSwitch;
    pendingVideoSwitch = null;
    if (next !== currentVideoName) switchToVideo(next);
  }
};

const handleVideoLoaded = () => {
  if (webcodecsMode) {
    return;
  }
  if (fpsLockInput.checked) {
    applyLockedFps();
  }
  video.playbackRate = Number(playbackSpeedInput.value) || 1;
  updateDurationDisplay();
  updateTrackbar();
  updateTimeDisplay();
  playPauseButton.textContent = "Play";
};

const togglePlayPause = () => {
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs is not ready yet.");
      return;
    }
    if (webcodecsPlaying) {
      stopWebcodecsPlayback();
    } else {
      startWebcodecsPlayback();
    }
    return;
  }

  if (!video.src) {
    setStatus("Load a video first.");
    return;
  }

  if (video.paused) {
    video.play();
    suppressTimeUpdate = false;
    playPauseButton.textContent = "Pause";
  } else {
    video.pause();
    playPauseButton.textContent = "Play";
  }
};

const toggleFullscreen = async () => {
  if (!document.fullscreenElement) {
    await document.documentElement.requestFullscreen();
    fullscreenToggle.textContent = "Exit Fullscreen";
  } else {
    await document.exitFullscreen();
    fullscreenToggle.textContent = "Fullscreen";
  }
};

const stepFrame = (direction) => {
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs is not ready yet.");
      return;
    }
    stopWebcodecsPlayback();
    const newIndex = Math.max(
      0,
      Math.min(currentFrameIndex + direction, webcodecsSamples.length - 1)
    );
    if (newIndex === currentFrameIndex) {
      return; // Already at boundary
    }
    
    currentFrameIndex = newIndex;
    updateTimeDisplay();
    updateTrackbar();
    
    // If frame is already cached, render immediately
    if (webcodecsFrameCache.has(newIndex)) {
      renderWebcodecsFrame(newIndex);
    } else {
      // Need to decode
      decodeWebcodecsUpTo(newIndex);
    }
    return;
  }

  if (!video.src) {
    setStatus("Load a video first.");
    return;
  }
  if (!Number.isFinite(video.duration)) {
    setStatus("Video is not ready yet.");
    return;
  }
  video.pause();
  playPauseButton.textContent = "Play";
  currentFrameIndex = Math.max(0, currentFrameIndex + direction);
  seekToFrame(currentFrameIndex);
};

const handleKeyAction = (key) => {
  // Check if video is loaded (either WebCodecs or HTML5)
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs is not ready yet.");
      return;
    }
  } else {
    if (!video.src) {
      setStatus("Load a video first.");
      return;
    }
    if (!Number.isFinite(video.duration)) {
      setStatus("Video time is not available yet.");
      return;
    }
  }

  const behavior = behaviors.find((item) => item.key === key);
  if (!behavior) {
    return;
  }

  if (!activeStarts.has(behavior.id)) {
    activeStarts.set(behavior.id, currentFrameIndex);
    setStatus(
      `Started ${behavior.name} at ${formatTime(
        currentFrameIndex * frameDurationMs
      )}.`
    );
    renderBehaviors();
    saveProgress();
    return;
  }

  const startFrame = activeStarts.get(behavior.id);
  activeStarts.delete(behavior.id);
  if (startFrame === currentFrameIndex) {
    setStatus(`Skipped ${behavior.name}: zero-length bout.`);
    renderBehaviors();
    saveProgress();
    return;
  }
  insertBoutSorted({
    behaviorId: behavior.id,
    behaviorName: behavior.name,
    startFrame,
    endFrame: currentFrameIndex
  });
  setStatus(
    `Ended ${behavior.name} at ${formatTime(
      currentFrameIndex * frameDurationMs
    )}.`
  );
  if (pauseOnLabelEndInput.checked) {
    if (webcodecsMode) {
      stopWebcodecsPlayback();
    } else {
      video.pause();
      playPauseButton.textContent = "Play";
    }
  }
  renderBehaviors();
  renderBouts();
  saveProgress();
};

const findOverlappingBouts = () => {
  for (let i = 0; i < bouts.length; i++) {
    for (let j = i + 1; j < bouts.length; j++) {
      const a = bouts[i];
      const b = bouts[j];
      // Bouts overlap if they share any frames
      if (a.startFrame < b.endFrame && b.startFrame < a.endFrame) {
        return [i, j];
      }
    }
  }
  return null;
};

const generateBoutsCsv = (boutsArr) => {
  const header = ["start", "end", "behavior"].join(",");
  const rows = boutsArr.map((bout) =>
    [
      csvEscape(formatTime(getFrameTimeMs(bout.startFrame))),
      csvEscape(formatTime(getFrameTimeMs(bout.endFrame))),
      csvEscape(bout.behaviorName)
    ].join(",")
  );
  return [header, ...rows].join("\n");
};

// Frame-indexed one-hot matrix — the canonical project label format consumed by
// the trainer (model.py::load_data). The `time` column is the NATIVE frame index
// (same index space as the extracted `{n}.png` files), and there is one 0/1
// column per behavior that has bouts. `load_data` takes argmax over each row to
// get the frame's class, so exactly one column is set per row.
//
// Only frames that fall inside a bout get a row (sparse — matching sparse frame
// extraction). Lead-in / gap frames are intentionally omitted: on disk they act
// only as triplet context for a labeled frame, never as labeled samples, so they
// must not appear in the label map (an all-zero row would otherwise argmax to the
// first behavior and mislabel the frame).
const generateTrainingLabelCsv = (boutsArr) => {
  // Behavior columns: names that actually have bouts, in first-appearance order.
  const behaviorNames = [];
  const seen = new Set();
  for (const bout of boutsArr) {
    if (!seen.has(bout.behaviorName)) {
      seen.add(bout.behaviorName);
      behaviorNames.push(bout.behaviorName);
    }
  }
  if (behaviorNames.length === 0) return "time\n"; // no labels yet

  // frame index → behavior column (bouts are non-overlapping; if any do overlap,
  // last-write-wins, which is fine for an argmax one-hot).
  const frameToCol = new Map();
  for (const bout of boutsArr) {
    const col = behaviorNames.indexOf(bout.behaviorName);
    for (let f = bout.startFrame; f <= bout.endFrame; f++) frameToCol.set(f, col);
  }

  const frames = [...frameToCol.keys()].sort((a, b) => a - b);
  const header = ["time", ...behaviorNames.map(csvEscape)].join(",");
  const rows = frames.map((f) => {
    const cols = new Array(behaviorNames.length).fill(0);
    cols[frameToCol.get(f)] = 1;
    return f + "," + cols.join(",");
  });
  return [header, ...rows].join("\n");
};

const exportCsv = async () => {
  if (bouts.length === 0) {
    setStatus("No bouts to export yet.");
    return;
  }

  const overlap = findOverlappingBouts();
  if (overlap) {
    const [i, j] = overlap;
    const boutA = bouts[i];
    const boutB = bouts[j];
    setStatus(
      `Cannot export: "${boutA.behaviorName}" (${formatTime(getFrameTimeMs(boutA.startFrame))}-${formatTime(getFrameTimeMs(boutA.endFrame))}) overlaps with "${boutB.behaviorName}" (${formatTime(getFrameTimeMs(boutB.startFrame))}-${formatTime(getFrameTimeMs(boutB.endFrame))}).`
    );
    return;
  }

  const content = generateBoutsCsv(bouts);

  if (!window.api || !window.api.saveCsv) {
    setStatus("CSV export is unavailable.");
    return;
  }

  const baseName = currentVideoName
    ? currentVideoName.replace(/\.[^/.]+$/, "")
    : "video";

  const result = await window.api.saveCsv({
    defaultPath: `${baseName}-bouts.csv`,
    content
  });

  if (result.canceled) {
    setStatus("Export canceled.");
  } else {
    setStatus(`CSV saved to ${result.filePath}`);
  }
};

// ── CSV line parser (shared) ────────────────────────────────────────────
const parseCsvLine = (line) => {
  const fields = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"' && line[i + 1] === '"') { current += '"'; i++; }
      else if (ch === '"') { inQuotes = false; }
      else { current += ch; }
    } else {
      if (ch === '"') { inQuotes = true; }
      else if (ch === ',') { fields.push(current.trim()); current = ""; }
      else { current += ch; }
    }
  }
  fields.push(current.trim());
  return fields;
};

// ── Parse CSV content and import into bouts/behaviors ─────────────────
const parseCsvAndImport = (content, sourceName = "CSV") => {
  const lines = content.split(/\r?\n/).filter((l) => l.trim());
  if (lines.length < 2) {
    setStatus("CSV file is empty or has no data rows.");
    return 0;
  }

  const headerFields = parseCsvLine(lines[0]);
  const headerLower = lines[0].toLowerCase();

  // Detect format: bout CSV ("start,end,behavior") vs frame-level label matrix
  const isBoutFormat = /start/.test(headerLower) && /end/.test(headerLower) && /behav/i.test(headerLower);
  const isFrameMatrix = /time/.test(headerLower) || (!isNaN(parseFloat(headerFields[0])) && headerFields.length > 3);

  let imported = 0;

  if (isBoutFormat) {
    // --- Bout format: start, end, behavior ---
    const dataLines = lines.slice(1);
    let skipped = 0;
    for (const line of dataLines) {
      const fields = parseCsvLine(line);
      if (fields.length < 3) { console.warn("[import] Row skipped (need 3 columns, got " + fields.length + "):", line); skipped++; continue; }

      const startMs = parseTime(fields[0]);
      const endMs = parseTime(fields[1]);
      const behaviorName = fields[2];

      if (startMs === null || endMs === null || !behaviorName) { console.warn("[import] Row skipped (parse failed) start='" + fields[0] + "' end='" + fields[1] + "' behavior='" + fields[2] + "' → startMs=" + startMs + " endMs=" + endMs); skipped++; continue; }

      const startFrame = msToFrame(startMs);
      const endFrame = msToFrame(endMs);
      if (startFrame === endFrame) { console.warn("[import] Row skipped (zero-length) start='" + fields[0] + "' end='" + fields[1] + "' → frame " + startFrame); skipped++; continue; }

      let behavior = findBehaviorByName(behaviorName);
      if (!behavior) {
        const key = KEY_ORDER[behaviors.length] || "";
        behavior = { id: Date.now() + "-" + Math.random(), name: normalizeBehaviorName(behaviorName), key };
        behaviors.push(behavior);
      }

      insertBoutSorted({
        behaviorId: behavior.id,
        behaviorName: behavior.name,
        startFrame: Math.min(startFrame, endFrame),
        endFrame: Math.max(startFrame, endFrame)
      });
      imported++;
    }
    const skippedMsg = skipped > 0 ? ` (${skipped} rows skipped)` : "";
    setStatus(`Imported ${imported} bouts from ${sourceName}${skippedMsg}`);

  } else if (isFrameMatrix) {
    // --- Frame-level label matrix: time, behavior1, behavior2, ... ---
    // First column is frame index, remaining columns are 0/1 per behavior.
    const hasTextHeader = isNaN(parseFloat(headerFields[0]));
    const behaviorNames = hasTextHeader
      ? headerFields.slice(1)
      : headerFields.slice(1).map((_, i) => "Behavior " + (i + 1));
    const dataLines = hasTextHeader ? lines.slice(1) : lines;

    console.log("[import] Detected frame-level label matrix with", behaviorNames.length, "behaviors:", behaviorNames.join(", "));

    // Parse all rows into per-behavior frame arrays to find contiguous runs
    const numBehaviors = behaviorNames.length;
    const active = new Array(numBehaviors).fill(false);
    const startFrames = new Array(numBehaviors).fill(0);
    let prevFrame = null;

    const closeBout = (b, endFrame) => {
      let behavior = findBehaviorByName(behaviorNames[b]);
      if (!behavior) {
        const key = KEY_ORDER[behaviors.length] || "";
        behavior = { id: Date.now() + "-" + Math.random(), name: normalizeBehaviorName(behaviorNames[b]), key };
        behaviors.push(behavior);
      }
      insertBoutSorted({
        behaviorId: behavior.id,
        behaviorName: behavior.name,
        startFrame: startFrames[b],
        endFrame
      });
      imported++;
    };

    for (const line of dataLines) {
      const fields = parseCsvLine(line);
      const frameIdx = parseInt(fields[0], 10);
      if (isNaN(frameIdx)) continue;

      // Sparse matrices (only labeled frames present) have no explicit "off"
      // row at a bout's end — a gap in the frame index implies the run ended at
      // the previous present frame. Close any still-open bouts at that boundary.
      if (prevFrame !== null && frameIdx > prevFrame + 1) {
        for (let b = 0; b < numBehaviors; b++) {
          if (active[b]) { active[b] = false; closeBout(b, prevFrame); }
        }
      }
      prevFrame = frameIdx;

      for (let b = 0; b < numBehaviors; b++) {
        const val = parseFloat(fields[b + 1]);
        const on = val >= 0.5;

        if (on && !active[b]) {
          active[b] = true;
          startFrames[b] = frameIdx;
        } else if (!on && active[b]) {
          active[b] = false;
          closeBout(b, frameIdx - 1);
        }
      }
    }

    // Close any still-active bouts at the last present frame
    const lastFields = parseCsvLine(dataLines[dataLines.length - 1]);
    const lastFrame = parseInt(lastFields[0], 10);
    for (let b = 0; b < numBehaviors; b++) {
      if (active[b] && !isNaN(lastFrame)) closeBout(b, lastFrame);
    }

    setStatus(`Imported ${imported} bouts (from frame-level labels) from ${sourceName}`);

  } else {
    setStatus("Unrecognized CSV format. Expected 'start,end,behavior' columns or a frame-level label matrix.");
    return 0;
  }

  if (imported > 0) {
    mergeDuplicateBehaviors();
    saveBehaviors();
    renderBehaviors();
    renderBouts();
    saveProgress();
  }
  return imported;
};

// ── Import CSV via file dialog ────────────────────────────────────────
const importCsv = async () => {
  if (!window.api || !window.api.openCsv) {
    setStatus("Label import is unavailable.");
    return;
  }
  const result = await window.api.openCsv();
  if (result.canceled) return;
  let content;
  if (result.isXlsx) {
    if (!window.api.xlsxToCsv) { setStatus("XLSX support unavailable."); return; }
    content = await window.api.xlsxToCsv(result.filePath);
  } else {
    content = result.content;
  }
  if (!content) { setStatus("Import canceled or file empty."); return; }
  parseCsvAndImport(content, result.filePath || "Labels");
};

const exportOneHot = async () => {
  if (bouts.length === 0) {
    setStatus("No bouts to export.");
    return;
  }

  // Determine frame range: 0 to max endFrame across all bouts
  let maxFrame = 0;
  for (const bout of bouts) {
    if (bout.endFrame > maxFrame) maxFrame = bout.endFrame;
  }

  // Collect unique behavior names in the order they first appear
  const behaviorNames = [];
  const seen = new Set();
  for (const bout of bouts) {
    if (!seen.has(bout.behaviorName)) {
      seen.add(bout.behaviorName);
      behaviorNames.push(bout.behaviorName);
    }
  }

  const fillBehavior = (fillBehaviorInput.value || "").trim();
  if (fillBehavior && !behaviorNames.includes(fillBehavior)) {
    behaviorNames.push(fillBehavior);
  }
  const fillIdx = fillBehavior ? behaviorNames.indexOf(fillBehavior) : -1;

  const minFrame = fillIdx >= 0
    ? bouts.reduce((m, b) => Math.min(m, b.startFrame), Infinity)
    : 0;

  // Build one-hot rows from minFrame..maxFrame (matching fill_missing_frames_onehot range)
  const header = ["time", ...behaviorNames.map(csvEscape)].join(",");
  const rows = [];
  for (let f = minFrame; f <= maxFrame; f++) {
    const cols = new Array(behaviorNames.length).fill(0);
    let hasLabel = false;
    for (const bout of bouts) {
      if (f >= bout.startFrame && f <= bout.endFrame) {
        const idx = behaviorNames.indexOf(bout.behaviorName);
        if (idx >= 0) { cols[idx] = 1; hasLabel = true; }
      }
    }
    if (!hasLabel && fillIdx >= 0) cols[fillIdx] = 1;
    rows.push(f + "," + cols.join(","));
  }
  const content = [header, ...rows].join("\n");

  if (!window.api || !window.api.saveCsv) {
    setStatus("CSV export is unavailable.");
    return;
  }

  const baseName = currentVideoName
    ? currentVideoName.replace(/\.[^/.]+$/, "")
    : "video";

  const result = await window.api.saveCsv({
    defaultPath: `${baseName}-onehot.csv`,
    content
  });

  if (result.canceled) {
    setStatus("Export canceled.");
  } else {
    setStatus(`One-hot CSV saved to ${result.filePath} (${maxFrame + 1} frames, ${behaviorNames.length} behaviors)`);
  }
};

// Burn behavior labels into a copy of the video with ffmpeg drawtext.
// `segments` defaults to the hand-labeled bouts; the Active Learning tab passes
// model predictions instead (A-SOID's `create_labeled_vid`), which is the same
// rendering job over a different source of intervals.
const generateLabeledVideo = async (segments = null) => {
  const source = segments || bouts;
  if (source.length === 0) {
    setStatus(segments ? "No predictions to overlay." : "No bouts to overlay.");
    return;
  }

  const videoPath = currentFile && currentFile.path ? currentFile.path : "";
  if (!videoPath) {
    setStatus("No video loaded. Load a video first.");
    return;
  }

  if (!window.api || !window.api.saveVideoDialog) {
    setStatus("File save is unavailable.");
    return;
  }

  const baseName = currentVideoName
    ? currentVideoName.replace(/\.[^/.]+$/, "")
    : "video";

  const saveResult = await window.api.saveVideoDialog({
    defaultPath: `${baseName}-${segments ? "predicted" : "labeled"}.mp4`
  });

  if (saveResult.canceled || !saveResult.filePath) {
    setStatus("Labeled video canceled.");
    return;
  }

  const outputPath = saveResult.filePath;
  const fps = parseFloat(fpsInput.value) || 30;

  // Build ffmpeg drawtext filters for each interval
  const filters = source.map((bout) => {
    const startSec = (bout.startFrame * frameDurationMs) / 1000;
    const endSec = (bout.endFrame * frameDurationMs) / 1000;
    const escaped = bout.behaviorName.replace(/'/g, "'\\''").replace(/:/g, "\\:");
    return `drawtext=text='${escaped}':fontsize=36:fontcolor=white:borderw=2:bordercolor=black:x=w*0.7:y=h*0.15:enable='between(t,${startSec.toFixed(4)},${endSec.toFixed(4)})'`;
  });

  const filterStr = filters.join(",");

  if (!window.api.runFfmpeg) {
    const cmd = `ffmpeg -i "${videoPath}" -vf "${filterStr}" -c:v libx264 -preset fast -crf 18 -c:a aac "${outputPath}"`;
    setStatus("ffmpeg not available in app. Run this command manually:");
    console.log("[labeled-video] ffmpeg command:\n" + cmd);
    try { await navigator.clipboard.writeText(cmd); } catch (_) {}
    setStatus("ffmpeg command copied to clipboard. Paste into a terminal to generate the labeled video.");
    return;
  }

  setStatus("Generating labeled video...");
  try {
    const result = await window.api.runFfmpeg({
      inputPath: videoPath,
      outputPath,
      filterStr,
      fps
    });
    if (result.success) {
      setStatus(`Labeled video saved to ${outputPath}`);
    } else {
      setStatus("ffmpeg failed: " + (result.error || "unknown error"));
      console.error("[labeled-video] ffmpeg stderr:", result.stderr);
    }
  } catch (err) {
    setStatus("ffmpeg error: " + (err.message || err));
  }
};

video.addEventListener("loadedmetadata", handleVideoLoaded);
video.addEventListener("loadeddata", startFrameWatch);
video.addEventListener("timeupdate", () => {
  if (webcodecsMode) {
    return;
  }
  if (!suppressTimeUpdate) {
    currentFrameIndex = Math.round(
      (video.currentTime * 1000) / frameDurationMs
    );
  }
  updateTimeDisplay();
  updateTrackbar();
});
video.addEventListener("ended", () => {
  if (webcodecsMode) {
    return;
  }
  playPauseButton.textContent = "Play";
});

video.addEventListener("seeked", () => {
  if (webcodecsMode) {
    return;
  }
  if (pendingSeekFrame === null) {
    return;
  }
  const delta = Math.abs(video.currentTime - (pendingSeekTime || 0));
  if (delta > (frameDurationMs / 1000) * 0.5) {
    setStatus("Frame seek limited by codec; landed on nearest frame.");
  }
  suppressTimeUpdate = false;
  pendingSeekFrame = null;
  pendingSeekTime = null;
  updateTimeDisplay();
  updateTrackbar();
});

let trackbarSeekTimeout = null;

// Seek to a frame from the trackbar, pausing any playback so scrubbing is
// authoritative. seekToFrame renders the frame in both modes (canvas for
// WebCodecs, <video> seek for HTML5) — no need to press play.
const trackbarSeekTo = (targetIndex) => {
  if (webcodecsMode) {
    if (webcodecsPlaying) stopWebcodecsPlayback();
  } else if (!video.paused) {
    video.pause();
    playPauseButton.textContent = "Play";
  }
  seekToFrame(targetIndex);
};

trackbar.addEventListener("input", () => {
  const targetIndex = Math.round(Number(trackbar.value) / frameDurationMs);
  currentFrameIndex = targetIndex;
  updateTimeDisplay();

  if (webcodecsMode) {
    // Instant feedback when the frame is already decoded; throttle the
    // heavier decode for uncached frames while the user is dragging.
    if (webcodecsFrameCache.has(targetIndex)) {
      renderWebcodecsFrame(targetIndex);
    }
    if (trackbarSeekTimeout) clearTimeout(trackbarSeekTimeout);
    trackbarSeekTimeout = setTimeout(() => {
      trackbarSeekTimeout = null;
      trackbarSeekTo(targetIndex);
    }, 40);
  } else {
    // Seeking a paused <video> updates the displayed frame immediately.
    trackbarSeekTo(targetIndex);
  }
});

// On release, land authoritatively on the final frame (covers any decode
// dropped by the drag throttle).
trackbar.addEventListener("change", () => {
  if (trackbarSeekTimeout) {
    clearTimeout(trackbarSeekTimeout);
    trackbarSeekTimeout = null;
  }
  const targetIndex = Math.round(Number(trackbar.value) / frameDurationMs);
  currentFrameIndex = targetIndex;
  updateTimeDisplay();
  trackbarSeekTo(targetIndex);
});

playPauseButton.addEventListener("click", togglePlayPause);
fullscreenToggle.addEventListener("click", () => {
  toggleFullscreen().catch((error) => {
    console.warn("Fullscreen failed", error);
  });
});
snapBackButton.addEventListener("click", () => {
  let targetFrame = 0;
  if (bouts.length > 0) {
    // Find the bout with the latest endFrame
    const lastBout = bouts.reduce((latest, bout) => 
      bout.endFrame > latest.endFrame ? bout : latest
    , bouts[0]);
    targetFrame = lastBout.endFrame;
  }
  if (webcodecsMode) {
    if (!webcodecsReady) {
      setStatus("WebCodecs not ready.");
      return;
    }
    stopWebcodecsPlayback();
    currentFrameIndex = targetFrame;
    updateTimeDisplay();
    updateTrackbar();
    seekToFrame(targetFrame);
  } else {
    if (!video.src) {
      setStatus("Load a video first.");
      return;
    }
    video.currentTime = (targetFrame * frameDurationMs) / 1000;
    currentFrameIndex = targetFrame;
    updateTimeDisplay();
    updateTrackbar();
  }
  setStatus(`Snapped to frame ${targetFrame}.`);
});
playbackSpeedInput.addEventListener("input", () => {
  const nextRate = Number(playbackSpeedInput.value);
  if (Number.isFinite(nextRate) && nextRate > 0) {
    if (!webcodecsMode) {
      video.playbackRate = nextRate;
    } else if (webcodecsPlaying) {
      startWebcodecsPlayback();
    }
    setStatus(`Playback speed set to ${nextRate}x.`);
  }
});
fpsInput.addEventListener("input", () => {
  if (fpsLockInput.checked) {
    applyLockedFps();
    updateTimeDisplay();
    updateTrackbar();
    setStatus(`Locked FPS set to ${fpsInput.value}.`);
  }
});
fpsLockInput.addEventListener("change", () => {
  if (fpsLockInput.checked) {
    applyLockedFps();
    setStatus(`FPS locked at ${fpsInput.value}.`);
  } else {
    setStatus("FPS lock disabled.");
  }
  updateTimeDisplay();
  updateTrackbar();
  if (webcodecsMode && webcodecsPlaying) {
    startWebcodecsPlayback();
  }
});
useWebCodecsInput.addEventListener("change", () => {
  console.log("[renderer] useWebCodecsInput changed, checked:", useWebCodecsInput.checked);
  if (useWebCodecsInput.checked) {
    if (!webcodecsSupported()) {
      useWebCodecsInput.checked = false;
      const reason = webcodecsUnavailabilityReason();
      setStatus(
        reason
          ? `WebCodecs not supported; using HTML5 mode. ${reason}`
          : "WebCodecs not supported; using HTML5 mode."
      );
      setWebcodecsStatus(reason);
      return;
    }
    if (currentFile) {
      loadWebcodecsFile(currentFile);
    } else {
      setStatus("Load a video before enabling WebCodecs.");
      setWebcodecsStatus("Load a video first.");
      useWebCodecsInput.checked = false;
    }
  } else {
    console.log("[renderer] WebCodecs checkbox unchecked, disabling mode");
    setWebcodecsMode(false);
    setWebcodecsStatus("");
    if (currentFile) {
      loadHtml5File(currentFile);
    }
  }
});
addBehaviorButton.addEventListener("click", addBehavior);
behaviorNameInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    addBehavior();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.target && ["INPUT", "TEXTAREA"].includes(event.target.tagName)) {
    return;
  }

  if (event.altKey || event.ctrlKey || event.metaKey) {
    return;
  }

  if (/^\d$/.test(event.key)) {
    handleKeyAction(event.key);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    stepFrame(1);
  } else if (event.key === "ArrowLeft") {
    event.preventDefault();
    stepFrame(-1);
  } else if (event.key.toLowerCase() === "f") {
    event.preventDefault();
    toggleFullscreen().catch((error) => {
      console.warn("Fullscreen failed", error);
    });
  } else if (event.key === " ") {
    event.preventDefault();
    togglePlayPause();
  }
});

videoFileInput.addEventListener("change", (event) => {
  console.log("[renderer] videoFileInput changed, webcodecsLoading:", webcodecsLoading, "webcodecsMode:", webcodecsMode);
  const file = event.target.files[0];
  if (!file) {
    return;
  }
  // Clear the input's value so picking the SAME file again still fires `change`
  // (needed to retry after a failed/wedged load, or to reload the same video).
  // `file` is already captured above, so this is safe.
  event.target.value = "";

  // Don't interrupt WebCodecs loading
  if (webcodecsLoading) {
    setStatus(`Still loading ${currentVideoName || "a video"} — wait for it to finish, then add this one.`);
    console.log("[renderer] Ignoring file change during WebCodecs loading");
    return;
  }

  currentFile = file;
  addVideoToList(file);
  console.log("[renderer] File selected:", file.name, "WebCodecs checked:", useWebCodecsInput.checked);
  if (useWebCodecsInput.checked) {
    if (!webcodecsSupported()) {
      useWebCodecsInput.checked = false;
      setStatus("WebCodecs not supported; using HTML5 mode.");
      setWebcodecsMode(false);
      loadHtml5File(file);
      return;
    }
    console.log("[renderer] Loading with WebCodecs");
    loadWebcodecsFile(file).catch((err) => {
      console.error("[renderer] loadWebcodecsFile rejected:", err);
      webcodecsLoading = false;
      webcodecsOperationInProgress = false;
      setWebcodecsStatus("WebCodecs error: " + (err && err.message ? err.message : err));
    });
    return;
  }
  console.log("[renderer] Loading with HTML5");
  setWebcodecsMode(false);
  loadHtml5File(file);
});

video.addEventListener("loadeddata", () => {
  if (webcodecsMode || webcodecsLoading || useWebCodecsInput.checked) {
    return;
  }
  setStatus(`Video ready: ${currentVideoName || "loaded"}.`);
});

// ── Import Project (video + label CSV) ────────────────────────────────
document.getElementById("importProject").addEventListener("click", async () => {
  if (!window.api || !window.api.importProject) {
    setStatus("Import Project is unavailable.");
    return;
  }

  const result = await window.api.importProject();
  if (result.canceled) return;

  const videoPath = result.videoPath;
  const videoName = videoPath.split(/[\\/]/).pop();

  // Resolve label content (CSV text or XLSX binary)
  let labelContent;
  if (result.isXlsx) {
    if (!window.api.xlsxToCsv) { setStatus("XLSX support unavailable."); return; }
    labelContent = await window.api.xlsxToCsv(result.labelPath);
  } else {
    labelContent = result.labelContent;
  }
  if (!labelContent) { setStatus("Label file empty or unreadable."); return; }

  // Build a File-like object for the existing load functions
  const fileObj = { name: videoName, path: videoPath };
  currentFile = fileObj;
  addVideoToList(fileObj);

  // Load the video, then import labels once it's ready
  const importLabelsOnce = () => {
    parseCsvAndImport(labelContent, result.labelPath || "labels");
  };

  if (useWebCodecsInput.checked && webcodecsSupported()) {
    // For WebCodecs: wait for the load to complete, then import labels
    const origRestore = restoreProgress;
    loadWebcodecsFile(fileObj).then(() => {
      // restoreProgress already ran inside loadWebcodecsFile — now layer labels on top
      importLabelsOnce();
    }).catch((err) => {
      console.error("[renderer] importProject WebCodecs load rejected:", err);
      webcodecsLoading = false;
      webcodecsOperationInProgress = false;
      setWebcodecsStatus("WebCodecs error: " + (err && err.message ? err.message : err));
      importLabelsOnce();
    });
  } else {
    // For HTML5: listen for loadeddata once, then import labels
    setWebcodecsMode(false);
    const onReady = () => {
      video.removeEventListener("loadeddata", onReady);
      importLabelsOnce();
    };
    video.addEventListener("loadeddata", onReady);
    loadHtml5File(fileObj);
  }
});

// Modal prompting the user how to handle an unsupported video format.
// Resolves to "choose" | "reencode" | null (dismissed).
function showFormatDialog(videoName) {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "modal-overlay";

    const modal = document.createElement("div");
    modal.className = "modal";

    const title = document.createElement("h3");
    title.textContent = "Unsupported video format";

    const body = document.createElement("p");
    body.textContent =
      `"${videoName}" can't be played directly — its codec or container isn't supported ` +
      `(common with HEVC/H.265 and AVI). Choose a different video that's already MP4 (H.264) ` +
      `or WebM, or re-encode this file to H.264 MP4 with ffmpeg.`;

    const actions = document.createElement("div");
    actions.className = "modal-actions";

    const chooseBtn = document.createElement("button");
    chooseBtn.className = "primary";
    chooseBtn.textContent = "Choose Different Video";

    const reencodeBtn = document.createElement("button");
    reencodeBtn.className = "secondary";
    reencodeBtn.textContent = "Re-encode to MP4";

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "ghost";
    cancelBtn.textContent = "Cancel";

    const close = (result) => {
      document.removeEventListener("keydown", onKey, true);
      overlay.remove();
      resolve(result);
    };
    const onKey = (e) => {
      if (e.key === "Escape") { e.stopPropagation(); close(null); }
    };

    chooseBtn.addEventListener("click", () => close("choose"));
    reencodeBtn.addEventListener("click", () => close("reencode"));
    cancelBtn.addEventListener("click", () => close(null));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(null); });
    // Capture phase so Escape closes the modal before global shortcuts see it.
    document.addEventListener("keydown", onKey, true);

    actions.append(chooseBtn, reencodeBtn, cancelBtn);
    modal.append(title, body, actions);
    overlay.append(modal);
    document.body.append(overlay);
    chooseBtn.focus();
  });
}

const reencodeCurrentVideo = async () => {
  if (!window.api || !window.api.reencodeVideo) {
    setStatus("Re-encode unavailable: ffmpeg integration not present.");
    return;
  }
  setStatus("Re-encoding with ffmpeg...");
  try {
    const result = await window.api.reencodeVideo({ inputPath: currentFile.path });
    if (result.success) {
      setStatus("Re-encoded successfully. Loading " + result.outputPath);
      const newName = result.outputPath.replace(/^.*[\\/]/, "");
      currentVideoName = newName;
      currentFile = { name: newName, path: result.outputPath };
      if (video.src) URL.revokeObjectURL(video.src);
      video.src = toFileUrl(result.outputPath);
      video.load();
      return;
    }
    const isMissing = /ENOENT|not found|not recognized/i.test(result.error || "");
    if (isMissing) {
      setStatus("ffmpeg not found. Install ffmpeg, or run manually: ffmpeg -i input -c:v libx264 -preset fast -crf 18 -c:a aac output.mp4");
    } else {
      setStatus("ffmpeg re-encode failed: " + (result.error || "unknown error"));
      console.error("[reencode] stderr:", result.stderr);
    }
  } catch (e) {
    setStatus("Re-encode error: " + (e.message || e));
  }
};

// Re-link the current (failing) video to a different file. Updates the stored
// path on the video-list entry and the project manifest, keeping the original
// name so its labels/progress stay attached, then reloads from the new path.
const relinkCurrentVideo = async () => {
  if (!window.api || !window.api.selectFile) {
    setStatus("File selection unavailable.");
    return;
  }
  const targetName = currentVideoName;
  const res = await window.api.selectFile({
    title: targetName ? `Locate video for "${targetName}"` : "Select Video",
    filters: [
      { name: "Video Files", extensions: ["mp4", "webm", "avi", "mkv", "mov"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (!res || res.canceled) return;
  const newPath = res.path;

  const entry = videoList.find(v => v.name === targetName);
  if (entry) {
    entry.path = newPath;
    saveVideoList();
    renderVideoSidebar();
    try {
      if (window.projectManager && window.projectManager.isActive()) {
        window.projectManager.setVideoPath(targetName, newPath);
      }
    } catch { /* project manager not ready */ }
    currentFile = { name: targetName, path: newPath };
  } else {
    // No tracked entry — treat as a fresh load.
    currentFile = { name: newPath.split(/[\\/]/).pop(), path: newPath };
    addVideoToList(currentFile);
  }

  setStatus(`Re-linked "${currentFile.name}". Loading…`);
  if (useWebCodecsInput.checked && webcodecsSupported()) {
    loadWebcodecsFile(currentFile);
  } else {
    setWebcodecsMode(false);
    loadHtml5File(currentFile);
  }
};

// If the video fails to load due to an unsupported codec (e.g. HEVC/H.265, AVI),
// prompt the user — never auto-re-encode.
video.addEventListener("error", async () => {
  const name = currentVideoName || "";
  const err = video.error;
  const code = err ? err.code : 0;
  const msg = err && err.message ? err.message : "";

  const isCodecError = code === 3 || code === 4
    || /\.avi$/i.test(name) || /\.av$/i.test(name)
    || (msg && /not supported|decode|format|codec/i.test(msg));

  if (isCodecError && currentFile && currentFile.path) {
    setStatus(`"${name}" is an unsupported format.`);
    const choice = await showFormatDialog(name);
    if (choice === "choose") {
      // Re-link the failing video to a correctly-formatted file (persists path).
      await relinkCurrentVideo();
    } else if (choice === "reencode") {
      await reencodeCurrentVideo();
    } else {
      setStatus("Load a supported video (MP4/H.264 or WebM), or re-encode it.");
    }
    return;
  }

  if (code === 2 || (msg && /network|abort/i.test(msg))) {
    setStatus("Video failed to load (network/abort). Try a local file or a different path.");
  } else if (isCodecError) {
    setStatus("Codec not supported. Re-encode with: ffmpeg -i input -c:v libx264 -preset fast -crf 18 -c:a aac output.mp4");
  } else {
    setStatus(`Video failed to load. Use MP4 (H.264) or WebM. ${msg ? `(${msg})` : ""}`);
  }
});

document.addEventListener("fullscreenchange", () => {
  document.body.classList.toggle("fullscreen", Boolean(document.fullscreenElement));
  fullscreenToggle.textContent = document.fullscreenElement
    ? "Exit Fullscreen"
    : "Fullscreen";
});

exportCsvButton.addEventListener("click", exportCsv);
importCsvButton.addEventListener("click", importCsv);
exportOneHotButton.addEventListener("click", exportOneHot);
generateLabeledVideoButton.addEventListener("click", () => generateLabeledVideo());

renderBehaviors();
renderBouts();
updateDurationDisplay();
if (!webcodecsApiSupported()) {
  useWebCodecsInput.checked = false;
  useWebCodecsInput.disabled = true;
  const reason = webcodecsUnavailabilityReason();
  if (reason) {
    useWebCodecsInput.title = reason;
    if (modeControl) {
      modeControl.title = reason;
    }
    setWebcodecsStatus(reason);
  }
}

if (!window.api) {
  const ua = navigator.userAgent || "";
  const uaElectron = ua.includes("Electron");
  setStatus(
    `Preload API missing. ElectronUA=${uaElectron}. Ensure app is launched with \`npm start\` and use the Electron window (title: LabelingApp (Electron)).`
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// BACKEND INTEGRATION & NEW PANELS
// ═══════════════════════════════════════════════════════════════════════════

// Helper: set path display text + tooltip
function setPathDisplay(id, path) {
  const el = document.getElementById(id);
  el.textContent = path;
  el.title = path;
}

const BACKEND_PORT = (window.api && window.api.backendPort) || 8741;
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`;

async function backendFetch(path, opts = {}) {
  const url = `${BACKEND_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.text().catch(() => res.statusText);
    throw new Error(`Backend ${res.status}: ${err}`);
  }
  return res.json();
}

function backendSSE(path, onMessage, onError) {
  const url = `${BACKEND_URL}${path}`;
  const src = new EventSource(url);
  src.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data));
    } catch (err) {
      console.error("SSE parse error", err);
    }
  };
  src.onerror = (e) => {
    src.close();
    if (onError) onError(e);
  };
  return src;
}

// ── Backend health check ──────────────────────────────────────────────────

const backendStatusEl = document.getElementById("backendStatus");

async function checkBackendHealth() {
  try {
    await backendFetch("/api/health");
    backendStatusEl.textContent = "Backend: Connected";
    backendStatusEl.className = "backend-status connected";
    return true;
  } catch {
    backendStatusEl.textContent = "Backend: Disconnected";
    backendStatusEl.className = "backend-status disconnected";
    return false;
  }
}

// Poll health every 10s
checkBackendHealth();
setInterval(checkBackendHealth, 10000);

// ── Tab switching ─────────────────────────────────────────────────────────

document.querySelectorAll(".tab-bar .tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-bar .tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const tabName = btn.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.toggle("active", p.dataset.tab === tabName);
    });
    document.body.dataset.tab = tabName;
    updateContentLayout();
    if (tabName === "train") renderTrainVideoChecklist();
    if (tabName === "pose") renderPoseVideoChecklist();
  });
});

// ── Pose Panel ────────────────────────────────────────────────────────────
// Replaces the TSM app's Preprocess panel. Where that one computed a
// background, tuned a crop and dumped PNG frames, this one just points at a
// pose file and picks which body parts feed the feature extractor.

function getVideoPath() {
  return currentFile && currentFile.path ? currentFile.path : null;
}

// How many frames the loaded video actually has. Everything the user reviews is
// indexed against this, so it's the ceiling for anything worth predicting.
// Returns 0 when unknown (no video loaded / duration not yet available).
function videoFrameCount() {
  if (webcodecsMode && webcodecsSamples.length) return webcodecsSamples.length;
  if (Number.isFinite(video.duration) && video.duration > 0) {
    const fps = Number(fpsInput.value) || 30;
    return Math.round(video.duration * fps);
  }
  return 0;
}

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

  poseEl("poseLog").textContent = "";
  poseEl("poseExtract").disabled = true;
  poseEl("poseStopExtract").disabled = false;
  poseEl("poseStatus").textContent = "Starting...";

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
    iteration_dir: currentIterationDir(),
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
            poseEl("poseStatus").textContent = `Done: ${results.videos.length} video(s)`;
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

function drawPoseOverlay() {
  if (!poseCtx || !poseState.keypoints) return;
  sizePoseCanvas();
  poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);

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
    drawPoseOverlay();
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
    checkpoint_dir: pm.getCheckpointsDir(),
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
                source: "trained",
                config,
              });
            } catch { /* manifest not ready */ }
            alState.checkpointPath = data.checkpoint_path;
            setPathDisplay("alCheckpointPath", data.checkpoint_path);
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

// ── Active Learning Panel ─────────────────────────────────────────────────

const alState = {
  checkpointPath: null,
  taskId: null,
  sseSource: null,
  results: null, // { frames: [...], class_names: [...] }
  uncertaintyArray: null, // Float32Array of normalized entropy per frame index
  misclassifiedArray: null, // Int8Array: 1 = misclassified, 0 = correct or no label
  frameByIndex: null, // Map<frameIndex, frameResult> for O(1) per-frame lookups
  lowConfCursor: -1, // walk position for "Jump to Lowest Confidence" (worst-first)
};

// ── Relabel strip (AL pane) ───────────────────────────────────────────────
// Surfaces the behavior→key legend as clickable buttons plus this frame's
// predicted class vs. current label, so the correct-in-place loop stays in the
// Active Learning tab. Labeling reuses handleKeyAction (the same path as the
// number keys), so bouts edited here are identical to Label-tab bouts.
function renderAlBehaviorButtons() {
  const container = document.getElementById("alBehaviorButtons");
  if (!container) return;
  container.innerHTML = "";

  if (behaviors.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No behaviors yet — add some in the Label tab.";
    container.appendChild(empty);
    return;
  }

  behaviors.forEach((behavior) => {
    const btn = document.createElement("button");
    btn.className = "al-behavior-btn";
    const recording = activeStarts.has(behavior.id);
    if (recording) btn.classList.add("recording");

    const badge = document.createElement("span");
    badge.className = "key-badge";
    badge.textContent = behavior.key;

    const name = document.createElement("span");
    name.textContent = behavior.name;

    btn.appendChild(badge);
    btn.appendChild(name);
    btn.title = recording
      ? `Recording ${behavior.name} — click to end the bout on this frame`
      : `Start a ${behavior.name} bout on this frame`;
    btn.addEventListener("click", () => handleKeyAction(behavior.key));
    container.appendChild(btn);
  });
}

// What the model says about the current frame vs. what the human labeled.
// Shared by the Active Learning panel's text readout and the on-video HUD so
// the two can never disagree about the same frame.
function currentFrameInfo() {
  let classNames = [];
  let cfg = null;
  let predName = null;
  let uncertainty = null;
  let hasPrediction = false;
  let hasResults = false;

  // Every read of alState stays inside this guard: updateAlFrameInfo runs
  // during module evaluation (via renderBehaviors/renderBouts), which is before
  // `const alState` is initialized, so an unguarded touch is a TDZ crash.
  try {
    if (alState && alState.results && alState.frameByIndex) {
      hasResults = true;
      classNames = alState.results.class_names || [];
      cfg = getAlModelConfig();
      const f = alState.frameByIndex.get(currentFrameIndex);
      if (f) {
        hasPrediction = true;
        predName = classNames[f.predicted_class] ?? `class ${f.predicted_class}`;
        if (typeof f.uncertainty === "number") uncertainty = f.uncertainty;
      }
    }
  } catch { /* alState not initialized yet */ }

  // Current label = the bout covering this frame, if any, translated into the
  // model's label space (a binarized model can't predict the raw behavior name).
  const bout = bouts.find(
    (b) => currentFrameIndex >= b.startFrame && currentFrameIndex <= b.endFrame
  );
  const truth = bout ? mapLabelToModelClass(bout.behaviorName, classNames, cfg) : null;
  // A real disagreement, in the same label space only (null truth = the two
  // aren't comparable, so it is never flagged).
  const mismatch = !!(bout && predName && truth !== null && truth !== predName);

  return { classNames, cfg, predName, uncertainty, hasPrediction, hasResults, bout, truth, mismatch };
}

// Uncertainty is normalized Shannon entropy in [0,1]; confidence is its
// complement, which is what a reader actually wants on screen.
function confidenceFrom(uncertainty) {
  if (typeof uncertainty !== "number") return null;
  return Math.max(0, Math.min(1, 1 - uncertainty));
}

function updateAlFrameInfo() {
  const frameEl = document.getElementById("alCurFrame");
  const predEl = document.getElementById("alCurPredicted");
  const labelEl = document.getElementById("alCurLabel");
  if (!frameEl || !predEl || !labelEl) return;

  frameEl.textContent = currentFrameIndex;

  const info = currentFrameInfo();

  let predText = "—";
  if (info.hasPrediction) {
    predText = info.uncertainty !== null
      ? `${info.predName} (unc ${info.uncertainty.toFixed(2)})`
      : info.predName;
  } else if (info.hasResults) {
    predText = "(no prediction)";
  }
  predEl.textContent = predText;

  // Show "walking -> other" when binarization collapses the label, so it's clear
  // why a differently-named bout is not a disagreement.
  labelEl.textContent = !info.bout
    ? "(none)"
    : (info.truth && info.truth !== info.bout.behaviorName
        ? `${info.bout.behaviorName} → ${info.truth}`
        : info.bout.behaviorName);

  predEl.classList.toggle("al-pred-mismatch", info.mismatch);

  renderVideoHud(info);
}

// ── On-video HUD ──────────────────────────────────────────────────────────
// Draws the current frame's prediction, confidence and label straight onto the
// video, so reviewing a model doesn't mean reading a number in a side panel
// while watching the animal out of the corner of your eye.
function renderVideoHud(info) {
  const hud = document.getElementById("videoHud");
  const toggle = document.getElementById("showHud");
  if (!hud || !toggle) return;

  if (!toggle.checked) {
    hud.classList.remove("visible");
    return;
  }

  const i = info || currentFrameInfo();
  // Nothing to report on this frame — stay entirely out of the way rather than
  // covering the video with em dashes.
  if (!i.hasPrediction && !i.bout) {
    hud.classList.remove("visible");
    return;
  }

  const predRow = hud.querySelector(".hud-pred");
  const labelRow = hud.querySelector(".hud-label");
  const bar = document.getElementById("hudBar");
  const barFill = document.getElementById("hudBarFill");
  const confEl = document.getElementById("hudConf");

  predRow.style.display = i.hasPrediction ? "" : "none";
  document.getElementById("hudPred").textContent = i.predName || "—";

  const confidence = confidenceFrom(i.uncertainty);
  if (i.hasPrediction && confidence !== null) {
    confEl.textContent = `${Math.round(confidence * 100)}%`;
    bar.style.display = "";
    barFill.style.width = `${confidence * 100}%`;
    // Match the timeline heatmap's reading of the same number: green = the
    // model is sure, red = it isn't and this frame is worth a look.
    barFill.style.background = `hsl(${Math.round(confidence * 120)}, 70%, 48%)`;
  } else {
    confEl.textContent = "";
    bar.style.display = "none";
  }

  labelRow.style.display = i.bout ? "" : "none";
  if (i.bout) {
    document.getElementById("hudLabel").textContent =
      i.truth && i.truth !== i.bout.behaviorName
        ? `${i.bout.behaviorName} → ${i.truth}`
        : i.bout.behaviorName;
  }

  hud.classList.toggle("mismatch", i.mismatch);
  hud.classList.add("visible");
}

document.getElementById("showHud").addEventListener("change", () => {
  renderVideoHud();
  debouncedSave();
});

function buildFrameIndex() {
  alState.frameByIndex = (alState.results && alState.results.frames)
    ? new Map(alState.results.frames.map((f) => [f.index, f]))
    : null;
}

// Apply a results object to alState and rebuild all derived views/overlays.
function applyAlResults(results) {
  alState.results = results;
  alState.lowConfCursor = -1; // new predictions → restart the worst-first walk
  buildUncertaintyArray();
  buildMisclassifiedArray();
  buildFrameIndex();
  renderUncertaintyOverlay();
  updateAlStats();
  updateAlFrameInfo();
}

// Drop in-memory predictions + derived overlays. Called on video switch so one
// video's predictions never leak onto another (they're indexed by native frame,
// so a stale array would be silently misaligned).
function clearAlPredictions() {
  if (alState.sseSource) { try { alState.sseSource.close(); } catch { /* already closed */ } }
  alState.taskId = null;
  alState.sseSource = null;
  alState.results = null;
  alState.uncertaintyArray = null;
  alState.misclassifiedArray = null;
  alState.frameByIndex = null;
  alState.lowConfCursor = -1;
  renderUncertaintyOverlay();
  updateAlStats();
  updateAlFrameInfo();
  const statusEl = document.getElementById("alStatus");
  if (statusEl) statusEl.textContent = "Idle";
}

// Persist the current video's predictions to the project (no-op without one).
function persistPredictions() {
  try {
    const pm = window.projectManager;
    if (pm && pm.isActive() && currentVideoName && alState.results) {
      pm.savePredictions(currentVideoName, JSON.stringify(alState.results)).catch(() => {});
    }
  } catch { /* project manager not ready */ }
}

// Load a video's saved predictions (if any) and rebuild overlays. Guarded so a
// late async result can't apply to a video the user has since switched away from.
async function restorePredictions(videoName) {
  try {
    const pm = window.projectManager;
    if (!pm || !pm.isActive()) return;
    const content = await pm.loadPredictions(videoName);
    if (!content) return;
    if (currentVideoName !== videoName) return; // user switched videos meanwhile
    const results = JSON.parse(content);
    if (!results || !Array.isArray(results.frames)) return;

    // Predictions saved before the frame-range clamp (or from a pose file longer
    // than the video) can run far past the last frame. Trim them for display so
    // the uncertainty timeline stays aligned with the trackbar. Done in memory
    // only — the file on disk is left intact, since those extra predictions are
    // still valid for the part of the recording this clip was cut from.
    const vFrames = videoFrameCount();
    let trimmed = 0;
    if (vFrames > 0 && results.frames.length) {
      const kept = results.frames.filter((f) => f.index < vFrames);
      trimmed = results.frames.length - kept.length;
      if (trimmed > 0) results.frames = kept;
    }

    applyAlResults(results);
    const statusEl = document.getElementById("alStatus");
    if (statusEl) {
      statusEl.textContent = trimmed > 0
        ? `Loaded saved predictions: ${results.frames.length} frames ` +
          `(${trimmed} beyond the end of the video ignored)`
        : `Loaded saved predictions: ${results.frames.length} frames`;
    }
    if (trimmed > 0) {
      setStatus(
        `${trimmed} saved predictions fall past the end of this video — ` +
        `the pose file likely covers a longer recording.`
      );
    }
  } catch (e) {
    console.warn("[al] Failed to restore predictions", e);
  }
}

// Threshold slider
const alThresholdSlider = document.getElementById("alThreshold");
const alThresholdOutput = document.getElementById("alThresholdVal");
const alMinSegmentInput = document.getElementById("alMinSegment");
alThresholdSlider.addEventListener("input", () => {
  alThresholdOutput.textContent = parseFloat(alThresholdSlider.value).toFixed(2);
  alState.lowConfCursor = -1; // ranking changed → restart the walk
  renderUncertaintyOverlay();
  updateAlStats();
});

// Minimum consecutive-segment length filter
alMinSegmentInput.addEventListener("input", () => {
  alState.lowConfCursor = -1; // ranking changed → restart the walk
  renderUncertaintyOverlay();
  updateAlStats();
});

const getMinSegmentLength = () => Math.max(1, parseInt(alMinSegmentInput.value, 10) || 1);

// Runs of consecutive frames with uncertainty >= threshold, of length >= minLen.
// Returns [{ start, end, length }] (inclusive frame indices). Frames with no
// prediction (value < 0) break a run.
function getUncertainSegments(arr, threshold, minLen) {
  const segs = [];
  let runStart = -1;
  for (let i = 0; i <= arr.length; i++) {
    const uncertain = i < arr.length && arr[i] >= 0 && arr[i] >= threshold;
    if (uncertain && runStart === -1) {
      runStart = i;
    } else if (!uncertain && runStart !== -1) {
      const length = i - runStart;
      if (length >= minLen) segs.push({ start: runStart, end: i - 1, length });
      runStart = -1;
    }
  }
  return segs;
}

// Per-frame membership mask for qualifying segments.
function getUncertainMask(arr, threshold, minLen) {
  const mask = new Uint8Array(arr.length);
  for (const seg of getUncertainSegments(arr, threshold, minLen)) {
    for (let i = seg.start; i <= seg.end; i++) mask[i] = 1;
  }
  return mask;
}

// Show on timeline toggle
document.getElementById("alShowTimeline").addEventListener("change", () => {
  renderUncertaintyOverlay();
});

// Select checkpoint
document.getElementById("alSelectCheckpoint").addEventListener("click", async () => {
  if (!window.api) return;
  const res = await window.api.selectFile({
    title: "Select Model Checkpoint",
    filters: [{ name: "Models", extensions: ["joblib"] }],
  });
  if (!res.canceled) {
    alState.checkpointPath = res.path;
    setPathDisplay("alCheckpointPath", res.path);
    // Record manually-selected checkpoints in the project manifest too.
    if (window.projectManager && window.projectManager.isActive()) {
      window.projectManager.addModel({
        name: res.path.split(/[\\/]/).pop(),
        path: res.path,
        created: new Date().toISOString().slice(0, 10),
        source: "selected",
      });
    }
  }
});

// ── Inference scope (entire video / frame range / labeled frames) ──────────
const alScopeSelect = document.getElementById("alScope");
const alRangeRow = document.getElementById("alRangeRow");
const alLabeledHint = document.getElementById("alLabeledHint");

function updateAlScopeVisibility() {
  const scope = alScopeSelect.value;
  alRangeRow.style.display = scope === "range" ? "" : "none";
  alLabeledHint.style.display = scope === "labeled" ? "" : "none";
}
alScopeSelect.addEventListener("change", updateAlScopeVisibility);
updateAlScopeVisibility();

document.getElementById("alRangeStartCur").addEventListener("click", () => {
  document.getElementById("alRangeStart").value = currentFrameIndex;
});
document.getElementById("alRangeEndCur").addEventListener("click", () => {
  document.getElementById("alRangeEnd").value = currentFrameIndex;
});

// Resolve the scope selection into frame_ranges for the backend.
// Returns { ranges } where ranges is null (entire video) or a list of
// inclusive [start, end] pairs, or { error } if the selection is invalid.
function resolveAlFrameRanges() {
  const scope = alScopeSelect.value;
  if (scope === "all") {
    // "Entire video" means exactly that — not the entire *pose file*, which may
    // be much longer (a full session's tracking against a short review clip).
    // Bounding it here keeps the uncertainty timeline aligned with the trackbar
    // and stops inference spending minutes on frames that can't be looked at.
    const vFrames = videoFrameCount();
    return { ranges: vFrames > 0 ? [[0, vFrames - 1]] : null };
  }

  if (scope === "range") {
    const start = parseInt(document.getElementById("alRangeStart").value, 10);
    const end = parseInt(document.getElementById("alRangeEnd").value, 10);
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
      return { error: "Enter a valid start and end frame." };
    }
    const lo = Math.max(0, Math.min(start, end));
    const hi = Math.max(start, end);
    if (hi < lo) return { error: "End frame must be ≥ start frame." };
    return { ranges: [[lo, hi]] };
  }

  if (scope === "labeled") {
    if (!bouts.length) {
      return { error: "No labeled frames on this video. Add bouts first, or pick another scope." };
    }
    const ranges = bouts.map((b) => [b.startFrame, b.endFrame]);
    return { ranges };
  }

  return { ranges: null };
}

// Run inference on this video's pose file. Body parts and bin width come from
// the checkpoint itself, so the feature space always matches how it was trained.
document.getElementById("alRunInference").addEventListener("click", async () => {
  if (!alState.checkpointPath) { alert("Select a checkpoint first."); return; }
  const posePath = getPosePathFor(currentVideoName);
  if (!posePath) {
    alert("This video has no pose file attached. Attach one in the Pose tab first.");
    return;
  }

  const scopeResult = resolveAlFrameRanges();
  if (scopeResult.error) { alert(scopeResult.error); return; }
  const frameRanges = scopeResult.ranges;

  document.getElementById("alRunInference").disabled = true;
  document.getElementById("alStopInference").disabled = false;
  document.getElementById("alStatus").textContent = "Starting...";

  try {
    const res = await backendFetch("/api/inference/run", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_path: alState.checkpointPath,
        pose_path: posePath,
        origin: poseState.origin,
        // null = use the checkpoint's own body-part selection.
        bodyparts: null,
        llh_value: parseFloat(document.getElementById("poseLlh").value) || 0.6,
        animals: poseState.multiAnimal ? [...poseState.selectedAnimals] : null,
        multi_animal: poseState.multiAnimal,
        frame_ranges: frameRanges,
      }),
    });
    alState.taskId = res.task_id;

    alState.sseSource = backendSSE(
      `/api/inference/status/${res.task_id}`,
      async (data) => {
        const pct = data.total > 0 ? Math.round((data.progress / data.total) * 100) : 0;
        document.getElementById("alProgress").value = pct;
        document.getElementById("alStatus").textContent = `${data.status}: ${data.progress}/${data.total}`;

        if (data.status === "completed") {
          alState.sseSource?.close();
          document.getElementById("alRunInference").disabled = false;
          document.getElementById("alStopInference").disabled = true;

          // Fetch results
          try {
            const results = await backendFetch(`/api/inference/results/${alState.taskId}`);
            applyAlResults(results);
            persistPredictions();
            document.getElementById("alStatus").textContent = `Done: ${results.frames.length} frames`;
          } catch (e) {
            document.getElementById("alStatus").textContent = `Error fetching results: ${e.message}`;
          }
        } else if (data.status === "error" || data.status === "cancelled") {
          alState.sseSource?.close();
          document.getElementById("alRunInference").disabled = false;
          document.getElementById("alStopInference").disabled = true;
          if (data.error) document.getElementById("alStatus").textContent = `Error: ${data.error}`;
        }
      },
      () => {
        document.getElementById("alRunInference").disabled = false;
        document.getElementById("alStopInference").disabled = true;
      }
    );
  } catch (e) {
    document.getElementById("alStatus").textContent = `Error: ${e.message}`;
    document.getElementById("alRunInference").disabled = false;
    document.getElementById("alStopInference").disabled = true;
  }
});

// Stop inference
document.getElementById("alStopInference").addEventListener("click", async () => {
  if (alState.taskId) {
    try {
      await backendFetch(`/api/inference/stop/${alState.taskId}`, { method: "POST" });
    } catch { /* ignore */ }
  }
});

// Build dense uncertainty array from sparse results
function buildUncertaintyArray() {
  if (!alState.results || !alState.results.frames.length) {
    alState.uncertaintyArray = null;
    return;
  }
  const frames = alState.results.frames;
  const maxIndex = frames[frames.length - 1].index;
  const arr = new Float32Array(maxIndex + 1);
  // Fill with -1 (no data) then overwrite with actual values
  arr.fill(-1);
  for (const f of frames) {
    arr[f.index] = f.uncertainty;
  }
  alState.uncertaintyArray = arr;
}

// ── Model label space (binarized vs multi-class) ──────────────────────────────
// A model trained with a positive class predicts in a COLLAPSED label space:
// load_data sets class_names = ['other', <positive>]. Bouts keep their original
// fine-grained names, so comparing the two raw strings falsely reports correct
// predictions as errors (a 'walking' bout the model rightly calls 'other', or a
// union member 'groom' it rightly calls 'positive'). Map the label first.

// Training config recorded on the manifest entry for the active checkpoint
// (carries `binarize` / `label_union`). Null when the checkpoint isn't tracked.
function getAlModelConfig() {
  try {
    const pm = window.projectManager;
    if (!pm || !pm.isActive() || !pm.getModels || !alState.checkpointPath) return null;
    const entry = pm.getModels().find((m) => m.path === alState.checkpointPath);
    return (entry && entry.config) || null;
  } catch {
    return null;
  }
}

// Translate a bout's behavior name into the class this model would predict for
// it. Returns null when the mapping can't be determined — callers must NOT flag
// those frames, since a label-space gap is indistinguishable from a real error.
function mapLabelToModelClass(behaviorName, classNames, cfg) {
  if (!behaviorName || !classNames || !classNames.length) return null;
  if (classNames.includes(behaviorName)) return behaviorName; // multi-class: direct

  // Binarized models look like ['other', <positive>].
  const positiveName =
    classNames.length === 2 && classNames[0] === "other" ? classNames[1] : null;
  if (!positiveName) return null;

  // Which original behaviors make up the positive class?
  let members = null;
  if (cfg && cfg.label_union && Array.isArray(cfg.label_union[positiveName])) {
    members = cfg.label_union[positiveName];   // >=2 behaviors unioned under a synthetic name
  } else if (cfg && cfg.binarize === positiveName) {
    members = [positiveName];                  // single behavior binarized
  } else if (behaviors.some((b) => b.name === positiveName)) {
    members = [positiveName];                  // no recorded config, but a real behavior
  }
  // Synthetic union name with no recorded membership → unknowable.
  if (!members) return null;
  return members.includes(behaviorName) ? positiveName : "other";
}

// Build dense misclassification array by comparing predictions to labeled bouts
function buildMisclassifiedArray() {
  if (!alState.results || !alState.results.frames.length) {
    alState.misclassifiedArray = null;
    return;
  }
  const frames = alState.results.frames;
  const classNames = alState.results.class_names || [];
  const cfg = getAlModelConfig();
  const maxIndex = frames[frames.length - 1].index;
  const arr = new Int8Array(maxIndex + 1); // 0 = correct / no label, 1 = misclassified

  for (const f of frames) {
    const predictedName = classNames[f.predicted_class] || "";
    // Find the bout that covers this frame index
    const bout = bouts.find(
      (b) => f.index >= b.startFrame && f.index <= b.endFrame
    );
    if (!bout || !predictedName) continue;
    // Compare in the model's label space, not the raw behavior name.
    const truth = mapLabelToModelClass(bout.behaviorName, classNames, cfg);
    if (truth !== null && truth !== predictedName) {
      arr[f.index] = 1;
    }
  }
  alState.misclassifiedArray = arr;
}

// Update stats display
function updateAlStats() {
  if (!alState.results) {
    document.getElementById("alStats").style.display = "none";
    return;
  }
  document.getElementById("alStats").style.display = "grid";
  const frames = alState.results.frames;
  const uncertainties = frames.map((f) => f.uncertainty);
  const threshold = parseFloat(alThresholdSlider.value);

  const sorted = [...uncertainties].sort((a, b) => a - b);
  const mean = uncertainties.reduce((a, b) => a + b, 0) / uncertainties.length;
  const median = sorted[Math.floor(sorted.length / 2)];

  // Filter to qualifying consecutive segments (>= min length).
  const minLen = getMinSegmentLength();
  const segments = alState.uncertaintyArray
    ? getUncertainSegments(alState.uncertaintyArray, threshold, minLen)
    : [];
  const framesInSegments = segments.reduce((sum, s) => sum + s.length, 0);

  document.getElementById("alTotalFrames").textContent = frames.length;
  document.getElementById("alMeanUncertainty").textContent = mean.toFixed(3);
  document.getElementById("alMedianUncertainty").textContent = median.toFixed(3);
  document.getElementById("alAboveThreshold").textContent =
    `${framesInSegments} (${((framesInSegments / frames.length) * 100).toFixed(1)}%)`;
  document.getElementById("alSegments").textContent =
    `${segments.length} (≥${minLen} frame${minLen === 1 ? "" : "s"})`;

  // Misclassified count
  const misArr = alState.misclassifiedArray;
  if (misArr) {
    let misCount = 0;
    for (let i = 0; i < misArr.length; i++) { if (misArr[i] === 1) misCount++; }
    document.getElementById("alMisclassified").textContent = `${misCount} (${((misCount / frames.length) * 100).toFixed(1)}%)`;
  } else {
    document.getElementById("alMisclassified").textContent = "-";
  }

  document.getElementById("alClassNames").textContent = (alState.results.class_names || []).join(", ");
}

// ── Uncertainty timeline canvas ───────────────────────────────────────────

const uncertaintyCanvas = document.getElementById("uncertaintyCanvas");
const ucCtx = uncertaintyCanvas.getContext("2d");

function renderUncertaintyOverlay() {
  const showOnTimeline = document.getElementById("alShowTimeline").checked;
  const arr = alState.uncertaintyArray;

  if (!showOnTimeline || !arr) {
    uncertaintyCanvas.classList.remove("visible");
    return;
  }
  uncertaintyCanvas.classList.add("visible");

  const W = uncertaintyCanvas.clientWidth;
  const H = 20;
  uncertaintyCanvas.width = W;
  uncertaintyCanvas.height = H;

  const N = arr.length;
  if (N === 0 || W === 0) return;

  const threshold = parseFloat(alThresholdSlider.value);
  const minLen = getMinSegmentLength();
  // Only frames inside qualifying segments get the threshold marker.
  const mask = getUncertainMask(arr, threshold, minLen);

  for (let px = 0; px < W; px++) {
    const startFrame = Math.floor((px / W) * N);
    const endFrame = Math.min(Math.floor(((px + 1) / W) * N), N);

    // Get max uncertainty in this pixel's frame range
    let maxU = -1;
    let inSegment = false;
    for (let i = startFrame; i < endFrame; i++) {
      if (arr[i] >= 0 && arr[i] > maxU) maxU = arr[i];
      if (mask[i]) inSegment = true;
    }

    if (maxU < 0) {
      // No data — draw grey
      ucCtx.fillStyle = "#d4d8e3";
    } else {
      // HSL: green (120) to red (0) based on uncertainty
      const hue = (1 - maxU) * 120;
      ucCtx.fillStyle = `hsl(${hue}, 85%, 50%)`;
    }
    ucCtx.fillRect(px, 0, 1, H);

    // Draw threshold marker only for qualifying segments
    if (inSegment) {
      ucCtx.fillStyle = "rgba(0, 0, 0, 0.3)";
      ucCtx.fillRect(px, 0, 1, 3);
    }

    // Draw misclassification marker (magenta tick at bottom)
    const misArr = alState.misclassifiedArray;
    if (misArr) {
      let hasMis = false;
      for (let i = startFrame; i < endFrame && i < misArr.length; i++) {
        if (misArr[i] === 1) { hasMis = true; break; }
      }
      if (hasMis) {
        ucCtx.fillStyle = "rgba(200, 0, 200, 0.85)";
        ucCtx.fillRect(px, H - 4, 1, 4);
      }
    }
  }
}

// Click on uncertainty canvas to seek
uncertaintyCanvas.addEventListener("click", (e) => {
  if (!alState.uncertaintyArray) return;
  const rect = uncertaintyCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const pct = x / rect.width;
  const frameIndex = Math.floor(pct * alState.uncertaintyArray.length);
  seekToFrame(frameIndex);
});

// Redraw on resize
window.addEventListener("resize", () => {
  if (alState.uncertaintyArray) renderUncertaintyOverlay();
});

// Jump to the start of the next qualifying uncertain segment
document.getElementById("alJumpNext").addEventListener("click", () => {
  if (!alState.uncertaintyArray) { alert("Run inference first."); return; }
  const threshold = parseFloat(alThresholdSlider.value);
  const minLen = getMinSegmentLength();
  const segments = getUncertainSegments(alState.uncertaintyArray, threshold, minLen);

  if (segments.length === 0) {
    alert(`No uncertain segments of ${minLen}+ consecutive frames above threshold.`);
    return;
  }

  // Next segment starting after the current frame, else wrap to the first.
  const next = segments.find((s) => s.start > currentFrameIndex) || segments[0];
  seekToFrame(next.start);
  setStatus(`Uncertain segment: frames ${next.start}–${next.end} (${next.length} frames).`);
});

// Rank qualifying uncertain segments worst-first (by their single lowest-
// confidence frame). Each entry: { start, end, length, peakIndex, peakUnc }.
// With min-segment-length = 1 each entry is a single frame, so this covers both
// "lowest-confidence frame" and "lowest-confidence frame set".
function rankLowConfidenceSegments() {
  const arr = alState.uncertaintyArray;
  if (!arr) return [];
  const threshold = parseFloat(alThresholdSlider.value);
  const minLen = getMinSegmentLength();
  const ranked = getUncertainSegments(arr, threshold, minLen).map((s) => {
    let peakIndex = s.start;
    let peakUnc = arr[s.start];
    for (let i = s.start + 1; i <= s.end; i++) {
      if (arr[i] > peakUnc) { peakUnc = arr[i]; peakIndex = i; }
    }
    return { ...s, peakIndex, peakUnc };
  });
  // Worst (highest uncertainty / lowest confidence) first.
  ranked.sort((a, b) => b.peakUnc - a.peakUnc);
  return ranked;
}

// Jump to the next-lowest-confidence frame set, walking the ranking worst-first.
document.getElementById("alJumpLowestConf").addEventListener("click", () => {
  if (!alState.uncertaintyArray) { alert("Run inference first."); return; }
  const ranked = rankLowConfidenceSegments();
  if (ranked.length === 0) {
    const minLen = getMinSegmentLength();
    alert(`No uncertain segments of ${minLen}+ consecutive frames above threshold.`);
    return;
  }

  // Advance the worst-first cursor (wraps at the end).
  alState.lowConfCursor = (alState.lowConfCursor + 1) % ranked.length;
  const item = ranked[alState.lowConfCursor];
  seekToFrame(item.peakIndex);

  // Prefer the model's actual max-probability as "confidence"; fall back to
  // 1 - normalized-entropy if the per-frame prediction vector isn't available.
  const f = alState.frameByIndex ? alState.frameByIndex.get(item.peakIndex) : null;
  const conf = f && Array.isArray(f.predictions) && f.predictions.length
    ? Math.max(...f.predictions)
    : 1 - item.peakUnc;
  const where = item.length > 1
    ? `frames ${item.start}–${item.end}, worst frame ${item.peakIndex}`
    : `frame ${item.peakIndex}`;
  setStatus(
    `Lowest confidence #${alState.lowConfCursor + 1}/${ranked.length}: ${where} ` +
    `(confidence ${conf.toFixed(2)}).`
  );
});

// Jump to next misclassified frame
document.getElementById("alJumpNextMisclassified").addEventListener("click", () => {
  if (!alState.misclassifiedArray) { alert("Run inference first."); return; }
  const arr = alState.misclassifiedArray;
  const start = currentFrameIndex + 1;

  // Search forward
  for (let i = start; i < arr.length; i++) {
    if (arr[i] === 1) {
      seekToFrame(i);
      return;
    }
  }
  // Wrap around
  for (let i = 0; i < start && i < arr.length; i++) {
    if (arr[i] === 1) {
      seekToFrame(i);
      return;
    }
  }
  alert("No misclassified frames found.");
});

// Export model predictions as a frame-indexed one-hot matrix — the SAME format
// as labels (generateTrainingLabelCsv), so they can be re-imported as labels.
const generatePredictionsCsv = () => {
  const frames = alState.results.frames;
  const classNames = alState.results.class_names || [];
  const header = ["time", ...classNames.map(csvEscape)].join(",");
  const rows = frames.map((f) => {
    const cols = new Array(classNames.length).fill(0);
    if (f.predicted_class >= 0 && f.predicted_class < classNames.length) {
      cols[f.predicted_class] = 1;
    }
    return f.index + "," + cols.join(",");
  });
  return [header, ...rows].join("\n");
};

// Collapse per-frame predictions into contiguous same-class intervals, in the
// same {startFrame, endFrame, behaviorName} shape as a bout — so the labeled
// video renderer, which already knows how to burn bouts in, takes them as-is.
// "other" runs are skipped: it means "no behavior", not a behavior to caption.
function predictionSegments() {
  if (!alState.results || !alState.results.frames.length) return [];
  const classNames = alState.results.class_names || [];
  const frames = alState.results.frames;

  const segments = [];
  let run = null;
  for (const f of frames) {
    const name = classNames[f.predicted_class] ?? `class ${f.predicted_class}`;
    // Break the run on a class change OR a gap in frame indices (scoped
    // inference returns only the requested ranges).
    if (run && run.behaviorName === name && f.index === run.endFrame + 1) {
      run.endFrame = f.index;
    } else {
      if (run && run.behaviorName !== "other") segments.push(run);
      run = { startFrame: f.index, endFrame: f.index, behaviorName: name };
    }
  }
  if (run && run.behaviorName !== "other") segments.push(run);
  return segments;
}

document.getElementById("alPredictedVideo").addEventListener("click", async () => {
  if (!alState.results || !alState.results.frames.length) {
    alert("Run inference first.");
    return;
  }
  const segments = predictionSegments();
  if (segments.length === 0) {
    setStatus("Every frame was predicted 'other' — nothing to caption.");
    return;
  }
  setStatus(`Rendering ${segments.length} predicted segments...`);
  await generateLabeledVideo(segments);
});

document.getElementById("alExportPredictions").addEventListener("click", async () => {
  if (!alState.results || !alState.results.frames.length) {
    alert("Run inference first.");
    return;
  }
  if (!window.api || !window.api.saveCsv) {
    setStatus("CSV export is unavailable.");
    return;
  }
  const content = generatePredictionsCsv();
  const baseName = currentVideoName ? currentVideoName.replace(/\.[^/.]+$/, "") : "video";
  const result = await window.api.saveCsv({
    defaultPath: `${baseName}-predictions.csv`,
    content,
  });
  setStatus(result.canceled ? "Export canceled." : `Predictions saved to ${result.filePath}`);
});

// ═══════════════════════════════════════════════════════════════════════════
// Phase 6: Polish — localStorage persistence, auto-populate, error handling
// ═══════════════════════════════════════════════════════════════════════════

// ── Settings persistence ──────────────────────────────────────────────────

const SETTINGS_KEY = "asoidDesktop_settings";

function saveSettings() {
  const settings = {
    // Pose / feature extraction
    poseOrigin: document.getElementById("poseOrigin").value,
    poseMultiAnimal: document.getElementById("poseMultiAnimal").checked,
    poseFramerate: parseFloat(document.getElementById("poseFramerate").value),
    poseFramerateManual: poseState.framerateManual,
    poseLlh: parseFloat(document.getElementById("poseLlh").value),
    poseMinDuration: parseFloat(document.getElementById("poseMinDuration").value),
    poseByVideo: poseState.poseByVideo,
    poseSelectedBodyparts: [...poseState.selected],
    // Training config
    trainInitRatio: parseFloat(document.getElementById("trainInitRatio").value),
    trainMaxIter: parseInt(document.getElementById("trainMaxIter").value),
    trainMaxSamplesIter: parseInt(document.getElementById("trainMaxSamplesIter").value),
    trainConfThreshold: parseFloat(document.getElementById("trainConfThreshold").value),
    trainExcludeOther: document.getElementById("trainExcludeOther").checked,
    trainModelName: document.getElementById("trainModelName").value,
    binarizeSelected: [...binarizeSelected],
    // Active learning
    alThreshold: parseFloat(alThresholdSlider.value),
    alMinSegment: getMinSegmentLength(),
    lastCheckpointPath: alState.checkpointPath,
    // Layout
    playerCol,
    showHud: document.getElementById("showHud").checked,
  };
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch { /* quota exceeded or unavailable */ }
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);

    const setVal = (id, val) => { if (val !== undefined && val !== null) document.getElementById(id).value = val; };
    const setChecked = (id, val) => { if (val !== undefined) document.getElementById(id).checked = val; };

    // Pose / feature extraction
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

    // Training config
    setVal("trainInitRatio", s.trainInitRatio);
    setVal("trainMaxIter", s.trainMaxIter);
    setVal("trainMaxSamplesIter", s.trainMaxSamplesIter);
    setVal("trainConfThreshold", s.trainConfThreshold);
    setChecked("trainExcludeOther", s.trainExcludeOther);
    setVal("trainModelName", s.trainModelName);
    if (Array.isArray(s.binarizeSelected)) {
      binarizeSelected.clear();
      s.binarizeSelected.forEach((n) => binarizeSelected.add(n));
      renderBinarizeChecklist();
    }

    // Active learning
    if (s.alThreshold !== undefined) {
      alThresholdSlider.value = s.alThreshold;
      alThresholdOutput.textContent = s.alThreshold.toFixed(2);
    }
    if (s.alMinSegment !== undefined) alMinSegmentInput.value = s.alMinSegment;
    if (s.lastCheckpointPath) {
      alState.checkpointPath = s.lastCheckpointPath;
      setPathDisplay("alCheckpointPath", s.lastCheckpointPath);
    }

    // Layout
    if (s.playerCol !== undefined) setPlayerCol(s.playerCol);
    setChecked("showHud", s.showHud);

    updatePoseDerivedDisplays();
  } catch { /* corrupted or unavailable */ }
}

// Load on startup
loadSettings();

// Debounced save — saves 2s after last change
let _saveTimer = null;
function debouncedSave() {
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(saveSettings, 2000);
}

// Attach save listeners to all inputs in panels
document.querySelectorAll(".tab-panel input, .tab-panel select").forEach((el) => {
  el.addEventListener("input", debouncedSave);
  el.addEventListener("change", debouncedSave);
});

// ── Enhanced error handling for backend operations ────────────────────────

// Wrap backendFetch to show user-friendly errors
const _origBackendFetch = backendFetch;
// We can't reassign a function declaration, but the existing backendFetch
// already throws meaningful errors. Instead, add a global handler for
// unhandled promise rejections to catch backend errors.
window.addEventListener("unhandledrejection", (e) => {
  const msg = e.reason?.message || String(e.reason);
  if (msg.includes("Backend") || msg.includes("fetch") || msg.includes("NetworkError")) {
    console.error("[Backend Error]", msg);
    // Don't spam alerts — the status indicator shows connection state
  }
});

// ── Backend health check improvements ─────────────────────────────────────
// The existing checkBackendHealth runs every 10s. Add a more aggressive
// initial check sequence: retry every 2s for the first 30s after app load.

let _backendReady = false;
async function initialBackendCheck() {
  for (let i = 0; i < 15; i++) {
    const ok = await checkBackendHealth();
    if (ok) {
      _backendReady = true;
      return;
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  // After 30s of trying, show a more detailed message
  backendStatusEl.textContent = "Backend: Failed to start — check Python/dependencies";
  backendStatusEl.className = "backend-status disconnected";
}
initialBackendCheck();

// ── Video sidebar event listeners ─────────────────────────────────────────

document.getElementById("addVideosBtn").addEventListener("click", async () => {
  if (!window.api || !window.api.selectFiles) {
    setStatus("File selection API unavailable.");
    return;
  }
  const result = await window.api.selectFiles({
    title: "Add Videos",
    filters: [
      { name: "Video Files", extensions: ["mp4", "webm", "avi", "mkv", "mov"] },
      { name: "All Files", extensions: ["*"] }
    ]
  });
  if (result.canceled || !result.paths) return;

  for (const filePath of result.paths) {
    const name = filePath.split(/[\\/]/).pop();
    addVideoToList({ name, path: filePath });
  }

  // If no video is currently loaded, load the first new one
  if (!currentVideoName && videoList.length > 0) {
    switchToVideo(videoList[0].name);
  }
});

document.getElementById("toggleSidebar").addEventListener("click", () => {
  document.getElementById("videoSidebar").classList.toggle("collapsed");
});

// Render sidebar on startup and auto-load last active video
document.body.dataset.tab = "label"; // matches the default active tab
renderVideoSidebar();
updateContentLayout();
// Clean up duplicate behaviors (spacing/case variants) from older saved data.
{
  const merged = mergeDuplicateBehaviors();
  if (merged > 0) {
    renderBehaviors();
    renderBouts();
    setStatus(`Merged ${merged} duplicate behavior${merged === 1 ? "" : "s"}.`);
  }
}
{
  const lastActive = localStorage.getItem("lastActiveVideo");
  if (lastActive && videoList.some(v => v.name === lastActive)) {
    const entry = videoList.find(v => v.name === lastActive);
    if (entry && entry.path) {
      switchToVideo(lastActive);
    }
  }
}

// ── Project management ────────────────────────────────────────────────────

function updateProjectUI() {
  const nameEl = document.getElementById("projectName");
  const saveBtn = document.getElementById("saveProject");
  const closeBtn = document.getElementById("closeProject");
  const exportRltBtn = document.getElementById("exportRlt");
  const exportRltPkgBtn = document.getElementById("exportRltPackage");
  if (window.projectManager && window.projectManager.isActive()) {
    const manifest = window.projectManager.getManifest();
    nameEl.textContent = manifest.name || "Untitled";
    saveBtn.classList.remove("hidden");
    closeBtn.classList.remove("hidden");
    exportRltBtn.classList.remove("hidden");
    exportRltPkgBtn.classList.remove("hidden");
  } else {
    nameEl.textContent = "";
    saveBtn.classList.add("hidden");
    closeBtn.classList.add("hidden");
    exportRltBtn.classList.add("hidden");
    exportRltPkgBtn.classList.add("hidden");
  }
  renderTrainVideoChecklist();
  renderPoseVideoChecklist();
  updateProjectGate();
}

// Show the project gate (blocking overlay) whenever no project is active.
function updateProjectGate() {
  const gate = document.getElementById("projectGate");
  if (!gate) return;
  const active = window.projectManager && window.projectManager.isActive();
  gate.classList.toggle("hidden", Boolean(active));
}

// Gate buttons reuse the existing Project-menu handlers.
document.getElementById("gateNewProject").addEventListener("click", () => {
  document.getElementById("newProject").click();
});
document.getElementById("gateOpenProject").addEventListener("click", () => {
  document.getElementById("openProjectBtn").click();
});
document.getElementById("gateImportRlt").addEventListener("click", () => {
  document.getElementById("importRlt").click();
});

async function restoreProjectVideos(manifest) {
  // Restore behaviors from manifest
  if (Array.isArray(manifest.behaviors) && manifest.behaviors.length > 0) {
    behaviors = manifest.behaviors.map((b, i) => ({
      id: Date.now() + "-" + i,
      name: normalizeBehaviorName(b.name),
      key: b.key || KEY_ORDER[i] || ""
    }));
    saveBehaviors();
    renderBehaviors();
  }

  // Add videos from manifest to videoList
  for (const vEntry of manifest.videos) {
    if (!vEntry.path) continue;
    const name = vEntry.path.split(/[\\/]/).pop();
    addVideoToList({ name, path: vEntry.path });

    // Load labels from project if they exist
    const labelContent = await window.projectManager.loadVideoLabels(name);
    if (labelContent && labelContent.trim()) {
      // Store as progress so restoreProgress picks them up
      // We need to temporarily switch context to parse for this video
      const prevBouts = [...bouts];
      bouts = [];
      parseCsvAndImport(labelContent, "project:" + vEntry.id);
      const importedBouts = [...bouts];
      bouts = prevBouts;

      // Save imported bouts to localStorage for this video
      if (importedBouts.length > 0) {
        const progress = { frameIndex: 0, bouts: importedBouts, activeStarts: [] };
        localStorage.setItem(getProgressKey(name), JSON.stringify(progress));
      }
    }
  }

  // Collapse any duplicate behaviors introduced across the manifest + labels.
  mergeDuplicateBehaviors();
  renderBehaviors();

  // Load first video
  if (videoList.length > 0) {
    switchToVideo(videoList[0].name);
  }

  // Re-link any trained models whose checkpoint path no longer resolves
  await relinkProjectModels(manifest);
}

async function fileExists(path) {
  if (!path || !window.api || typeof window.api.getFileSize !== "function") return false;
  try {
    await window.api.getFileSize(path);
    return true;
  } catch {
    return false;
  }
}

async function relinkProjectModels(manifest) {
  const models = Array.isArray(manifest.models) ? manifest.models : [];
  if (models.length === 0) return;

  const missing = [];
  for (const m of models) {
    if (!(await fileExists(m.path))) missing.push(m);
  }
  if (missing.length === 0) return;

  setStatus(
    `${missing.length} model checkpoint(s) need re-linking (copied separately, or re-train to regenerate).`
  );

  for (const m of missing) {
    const ok = window.confirm(
      `Model "${m.name}" was not found at:\n${m.path}\n\nLocate the checkpoint file now? ` +
      `(Cancel to skip — you can re-train to regenerate it.)`
    );
    if (!ok) continue;
    const res = await window.api.selectFile({
      title: `Locate checkpoint for "${m.name}"`,
      filters: [{ name: "Checkpoints", extensions: ["ckpt", "pt"] }],
    });
    if (res && !res.canceled) {
      window.projectManager.updateModelPath(m.path, res.path);
      // If this is the model the Active Learning panel points at, update it.
      if (alState && alState.checkpointPath === m.path) {
        alState.checkpointPath = res.path;
        setPathDisplay("alCheckpointPath", res.path);
      }
      setStatus(`Re-linked model "${m.name}".`);
    }
  }
}

document.getElementById("newProject").addEventListener("click", async () => {
  if (!window.projectManager) {
    setStatus("Project manager not available.");
    return;
  }
  const result = await window.projectManager.createProject();
  if (!result) return;

  // A new project starts empty — clear the current session so the previously
  // open project's videos/bouts don't carry over into it.
  videoList = [];
  saveVideoList();
  currentVideoName = "";
  currentFile = null;
  if (video.src) { URL.revokeObjectURL(video.src); video.removeAttribute("src"); }
  resetSession();
  renderVideoSidebar();
  // Seed the manifest with the (global) behavior set.
  window.projectManager.updateBehaviors(behaviors);

  updateProjectUI();
  setStatus(`Project "${result.manifest.name}" created (empty).`);
});

document.getElementById("openProjectBtn").addEventListener("click", async () => {
  if (!window.projectManager) return;

  // Close current project if open
  if (window.projectManager.isActive()) {
    await window.projectManager.closeProject();
  }

  const result = await window.projectManager.openProject();
  if (!result) return;

  updateProjectUI();
  await restoreProjectVideos(result.manifest);
  setStatus(`Project "${result.manifest.name}" opened.`);
});

document.getElementById("saveProject").addEventListener("click", async () => {
  if (!window.projectManager || !window.projectManager.isActive()) return;
  saveProgress();
  // Write labels for all videos in the project
  for (const v of videoList) {
    if (v.name === currentVideoName) continue; // already saved by saveProgress
    const stored = loadProgress(v.name);
    if (stored && Array.isArray(stored.bouts) && stored.bouts.length > 0) {
      const csv = generateTrainingLabelCsv(stored.bouts);
      await window.projectManager.saveVideoLabels(v.name, csv);
    }
  }
  await window.projectManager.flushManifest();
  setStatus("Project saved.");
});

document.getElementById("closeProject").addEventListener("click", async () => {
  if (!window.projectManager || !window.projectManager.isActive()) return;
  // Save current state before closing
  saveProgress();
  await window.projectManager.closeProject();
  updateProjectUI();
  setStatus("Project closed.");
});

// ── Project dropdown menu ──
(() => {
  const menu = document.getElementById("projectMenu");
  const trigger = document.getElementById("projectMenuBtn");
  if (!menu || !trigger) return;

  const setOpen = (open) => {
    menu.classList.toggle("open", open);
    trigger.setAttribute("aria-expanded", String(open));
  };

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    setOpen(!menu.classList.contains("open"));
  });
  // Close when clicking outside, on Escape, or after picking an action.
  document.addEventListener("click", (e) => {
    if (!menu.contains(e.target)) setOpen(false);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") setOpen(false);
  });
  menu.querySelectorAll(".menu-panel button").forEach((b) =>
    b.addEventListener("click", () => setOpen(false))
  );
})();

// ── .rlt export / import ──

async function doExportRlt(pkg) {
  if (!window.projectManager || !window.projectManager.isActive()) return;
  // Save all current state first
  saveProgress();
  for (const v of videoList) {
    if (v.name === currentVideoName) continue;
    const stored = loadProgress(v.name);
    if (stored && Array.isArray(stored.bouts) && stored.bouts.length > 0) {
      const csv = generateTrainingLabelCsv(stored.bouts);
      await window.projectManager.saveVideoLabels(v.name, csv);
    }
  }
  await window.projectManager.flushManifest();

  setStatus(pkg ? "Exporting full package (.rlt) — this can take a while..." : "Exporting project to .rlt...");
  try {
    const result = await window.projectManager.exportRlt({ package: pkg });
    if (!result) {
      setStatus("Export canceled.");
      return;
    }
    const s = result.summary;
    if (s.packaged) {
      setStatus(
        `Exported package .rlt: ${s.videos} video(s), ${s.labels} label set(s), ` +
        `${s.pose} pose file(s), ${s.features} feature set(s), ${s.checkpoints} model(s).`
      );
    } else {
      setStatus(`Exported .rlt: ${s.videos} video reference(s), ${s.labels} label set(s).`);
    }
  } catch (err) {
    setStatus("Export failed: " + (err.message || err));
  }
}

document.getElementById("exportRlt").addEventListener("click", () => doExportRlt(false));
document.getElementById("exportRltPackage").addEventListener("click", () => doExportRlt(true));

document.getElementById("importRlt").addEventListener("click", async () => {
  if (!window.projectManager) {
    setStatus("Project manager not available.");
    return;
  }
  // Close any open project first
  if (window.projectManager.isActive()) {
    saveProgress();
    await window.projectManager.closeProject();
  }

  setStatus("Importing .rlt file...");
  try {
    const result = await window.projectManager.importRlt();
    if (!result) {
      setStatus("Import canceled.");
      return;
    }
    updateProjectUI();
    await restoreProjectVideos(result.manifest);
    setStatus(
      `Imported project "${result.manifest.name}". ` +
      "Re-link any video that doesn't load; frames regenerate from the video + config, " +
      "and models can be re-linked (if copied over) or re-trained."
    );
  } catch (err) {
    setStatus("Import failed: " + (err.message || err));
  }
});

// Try to restore last project on startup
(async () => {
  const lastDir = localStorage.getItem("lastProjectDir");
  if (lastDir && window.projectManager) {
    try {
      const result = await window.projectManager.openProjectFromDir(lastDir);
      if (result) {
        updateProjectUI();
        await restoreProjectVideos(result.manifest);
      }
    } catch (err) {
      console.warn("[project] Failed to restore last project:", err);
      localStorage.removeItem("lastProjectDir");
    }
  }
  // Show the gate if nothing restored (no project active).
  updateProjectGate();
})();

// ── Save settings on page unload ──────────────────────────────────────────
window.addEventListener("beforeunload", () => {
  saveSettings();
  // Flush project manifest on unload
  if (window.projectManager && window.projectManager.isActive()) {
    saveProgress();
    window.projectManager.flushManifest();
  }
});
