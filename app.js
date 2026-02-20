// app.js
// Overwrite your entire app.js with this.
//
// Changes vs your previous version:
// 1) Much smoother + “persistent” cursor:
//    - OneEuro filters for cursor X/Y (kept)
//    - Extra smoothing for handScalePx (reduces size jitter)
//    - Grace period when the hand is briefly lost (keeps last cursor + avoids resets)
//    - Hover stabilization (small dwell) to stop hover flicker
//
// 2) Close viewer no longer uses pinch-on-(X).
//    - Close uses a THUMBS-UP gesture (debounced).
//    - Pinch remains ONLY for opening a thumbnail in grid mode.

import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// =====================
// DOM
// =====================
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const video = document.getElementById("webcam");
const stage = document.getElementById("stage");
const overlay = document.getElementById("overlay");
const logs = document.getElementById("logs");

const gridEl = document.getElementById("grid");
const cursorEl = document.getElementById("cursor");

const viewerEl = document.getElementById("viewer");
const viewerImg = document.getElementById("viewerImg");
const closeBtn = document.getElementById("closeBtn"); // now “visual only”, mouse click still works

const overlayCtx = overlay.getContext("2d");

// =====================
// Demo files
// =====================
const FILES = [
  { id: "1", name: "Mountains", url: "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=1600&q=80" },
  { id: "2", name: "City",      url: "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=1600&q=80" },
  { id: "3", name: "Forest",    url: "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?auto=format&fit=crop&w=1600&q=80" }
];

function mountGrid() {
  gridEl.innerHTML = "";
  for (const f of FILES) {
    const card = document.createElement("div");
    card.className = "thumb file";
    card.dataset.id = f.id;
    card.dataset.src = f.url;
    card.innerHTML = `
      <img src="${f.url}" alt="${f.name}">
      <div class="meta">
        <div class="name">${f.name}</div>
        <div class="hint">pinch to open</div>
      </div>
    `;
    gridEl.appendChild(card);
  }
}
mountGrid();

// =====================
// Logging / status
// =====================
function log(msg) {
  const ts = new Date().toISOString().split("T")[1].split(".")[0];
  logs.value = `[${ts}] ${msg}\n` + logs.value;
}
function setStatus(msg) {
  statusEl.textContent = msg;
  log(msg);
}

// =====================
// Utils
// =====================
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

// =====================
// One Euro filter (cursor smoothing)
// =====================
class OneEuroFilter {
  constructor(freq = 60, minCutoff = 1.0, beta = 0.02, dCutoff = 1.0) {
    this.freq = freq;
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
  alpha(cutoff) {
    const te = 1.0 / this.freq;
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / te);
  }
  lowpass(prev, curr, a) {
    return prev + a * (curr - prev);
  }
  filter(x, tMs) {
    if (this.tPrev == null) {
      this.tPrev = tMs;
      this.xPrev = x;
      this.dxPrev = 0;
      return x;
    }
    const dt = Math.max(1e-6, (tMs - this.tPrev) / 1000);
    this.freq = 1.0 / dt;

    const dx = (x - this.xPrev) / dt;
    const ad = this.alpha(this.dCutoff);
    const dxHat = this.lowpass(this.dxPrev, dx, ad);

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = this.alpha(cutoff);

    const xHat = this.lowpass(this.xPrev, x, a);

    this.tPrev = tMs;
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }
  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
}

const xFilter = new OneEuroFilter(60, 1.0, 0.02, 1.0);
const yFilter = new OneEuroFilter(60, 1.0, 0.02, 1.0);

// Extra smoothing for scale (reduces pointer size jitter)
const scaleFilter = new OneEuroFilter(60, 0.9, 0.01, 1.0);

// =====================
// Pinch debounce + hysteresis (OPEN only)
// =====================
let pinchClosed = false;
let pinchOnCount = 0;
let pinchOffCount = 0;

const PINCH_ON = 0.30;
const PINCH_OFF = 0.60;
const PINCH_ON_FRAMES = 2;
const PINCH_OFF_FRAMES = 3;

// =====================
// Thumbs-up debounce (CLOSE)
// =====================
let thumbsUpOnCount = 0;
let thumbsUpOffCount = 0;
let thumbsUpActive = false;

const THUMBS_ON_FRAMES = 4;   // require more frames for reliability
const THUMBS_OFF_FRAMES = 4;

// =====================
// State
// =====================
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

let mode = "grid"; // "grid" | "viewer"

// Cursor persistence / hand-loss handling
let lastSeenMs = 0;
const HAND_LOST_GRACE_MS = 350; // brief occlusions won’t reset everything

// Hover stabilization
let hoveredEl = null;
let hoverCandidate = null;
let hoverCandidateSince = 0;
const HOVER_DWELL_MS = 70; // prevents hover flicker

let selectedEl = null;

// =====================
// Geometry helpers
// =====================
function resizeOverlay() {
  const rect = stage.getBoundingClientRect();
  overlay.width = Math.round(rect.width);
  overlay.height = Math.round(rect.height);
}

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

// rough hand size: wrist ↔ index MCP (normalized)
function getHandScaleNorm(landmarks) {
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const d = distance(wrist, indexMcp);
  return d > 0 ? d : 0.0001;
}

function getHandScalePxRaw(landmarks) {
  const hsNorm = getHandScaleNorm(landmarks);
  const rect = stage.getBoundingClientRect();
  const minDim = Math.min(rect.width, rect.height);
  return hsNorm * minDim;
}

// Mirror X so movement feels natural
function screenCoordsFromLandmark(landmark) {
  const rect = stage.getBoundingClientRect();
  const mirroredX = 1 - landmark.x;
  const x = mirroredX * rect.width;
  const y = landmark.y * rect.height;
  return { x, y };
}

function elementUnderPoint(clientX, clientY) {
  const el = document.elementFromPoint(clientX, clientY);
  if (!el) return null;

  // viewer close button (visual only); keep hover highlight for UX
  if (el === closeBtn || closeBtn.contains(el)) return closeBtn;

  const thumb = el.closest?.(".thumb.file");
  if (thumb) return thumb;

  return null;
}

function applyHovered(next) {
  if (hoveredEl === next) return;

  if (hoveredEl) {
    if (hoveredEl.classList?.contains("thumb")) hoveredEl.classList.remove("hovered");
    if (hoveredEl === closeBtn) closeBtn.classList.remove("hovered");
  }
  hoveredEl = next;
  if (hoveredEl) {
    if (hoveredEl.classList?.contains("thumb")) hoveredEl.classList.add("hovered");
    if (hoveredEl === closeBtn) closeBtn.classList.add("hovered");
  }
}

function stabilizeHover(next, nowMs) {
  if (next !== hoverCandidate) {
    hoverCandidate = next;
    hoverCandidateSince = nowMs;
    return;
  }
  if (hoveredEl !== hoverCandidate && (nowMs - hoverCandidateSince) >= HOVER_DWELL_MS) {
    applyHovered(hoverCandidate);
  }
}

function clearSelected() {
  if (selectedEl && selectedEl.classList?.contains("thumb")) selectedEl.classList.remove("selected");
  selectedEl = null;
}

function openFile(fileEl) {
  viewerImg.src = fileEl.dataset.src;
  viewerEl.classList.add("visible");
  viewerEl.setAttribute("aria-hidden", "false");
  mode = "viewer";
  log(`OPEN file ${fileEl.dataset.id}`);
}

function closeViewer() {
  viewerEl.classList.remove("visible");
  viewerEl.setAttribute("aria-hidden", "true");
  mode = "grid";
  log("CLOSE viewer");
}

// =====================
// Pointer scaling
// =====================
function handScaleToT(hsPx) {
  // Calibrated from your logs:
  // far ~ 38–46px, near ~ 190–230px, very near ~ 290px
  const FAR_PX = 40;
  const NEAR_PX = 300;
  const lin = clamp((hsPx - FAR_PX) / (NEAR_PX - FAR_PX), 0, 1);
  return easeOutCubic(lin);
}

function updateCursorSizeFromT(t) {
  // Cursor diameter range (px)
  const d = lerp(14, 34, t);
  cursorEl.style.width = `${d}px`;
  cursorEl.style.height = `${d}px`;
}

// =====================
// Overlay drawing
// =====================
function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawPoint(x, y, r, fill, stroke) {
  overlayCtx.beginPath();
  overlayCtx.arc(x, y, r, 0, Math.PI * 2);
  overlayCtx.fillStyle = fill;
  overlayCtx.fill();
  overlayCtx.lineWidth = 2;
  overlayCtx.strokeStyle = stroke;
  overlayCtx.stroke();
}

function drawThumbIndexPointsScaled(landmarks, t) {
  clearOverlay();
  const thumb = screenCoordsFromLandmark(landmarks[4]);
  const index = screenCoordsFromLandmark(landmarks[8]);

  const thumbR = lerp(5, 16, t);
  const indexR = lerp(7, 22, t);

  drawPoint(thumb.x, thumb.y, thumbR, "rgba(255,255,255,0.30)", "rgba(255,255,255,0.85)");
  drawPoint(index.x, index.y, indexR, "rgba(120,180,255,0.35)", "rgba(120,180,255,0.95)");
}

// =====================
// Gesture detectors
// =====================
function updatePinchStateDebounced(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];

  const rawDist = distance(thumbTip, indexTip);
  const handScale = getHandScaleNorm(landmarks);
  const normDist = rawDist / handScale;

  if (!pinchClosed) {
    if (normDist < PINCH_ON) pinchOnCount++;
    else pinchOnCount = 0;

    if (pinchOnCount >= PINCH_ON_FRAMES) {
      pinchClosed = true;
      pinchOnCount = 0;
      pinchOffCount = 0;
      log(`Pinch ON (normDist=${normDist.toFixed(2)})`);
    }
  } else {
    if (normDist > PINCH_OFF) pinchOffCount++;
    else pinchOffCount = 0;

    if (pinchOffCount >= PINCH_OFF_FRAMES) {
      pinchClosed = false;
      pinchOffCount = 0;
      pinchOnCount = 0;
      log(`Pinch OFF (normDist=${normDist.toFixed(2)})`);
    }
  }
  return pinchClosed;
}

/**
 * Thumbs-up heuristic (works well in typical webcam framing):
 * - Thumb tip is ABOVE thumb IP and thumb MCP (y smaller = higher)
 * - Thumb tip is above wrist
 * - Other fingertips are NOT above their PIP joints (i.e. folded/neutral), and are close-ish to palm.
 *
 * This is a heuristic, not a classifier; we debounce it for stability.
 */
function isThumbsUpRaw(landmarks) {
  const wrist = landmarks[0];

  const thumbTip = landmarks[4];
  const thumbIp  = landmarks[3];
  const thumbMcp = landmarks[2];

  const indexTip = landmarks[8],  indexPip = landmarks[6],  indexMcp = landmarks[5];
  const midTip   = landmarks[12], midPip   = landmarks[10], midMcp   = landmarks[9];
  const ringTip  = landmarks[16], ringPip  = landmarks[14], ringMcp  = landmarks[13];
  const pinkTip  = landmarks[20], pinkPip  = landmarks[18], pinkMcp  = landmarks[17];

  // Thumb must be clearly "up"
  const thumbUp =
    (thumbTip.y < thumbIp.y) &&
    (thumbIp.y  < thumbMcp.y) &&
    (thumbTip.y < wrist.y);

  if (!thumbUp) return false;

  // Other fingers should be folded/neutral (not extended upward).
  // Two tests:
  // 1) tip is below pip (y greater) => folded/downward
  // 2) tip is close to mcp relative to hand scale => curled
  const hs = getHandScaleNorm(landmarks);

  function folded(tip, pip, mcp) {
    const tipBelowPip = tip.y > pip.y;
    const curled = (distance(tip, mcp) / hs) < 1.10; // tune if needed
    return tipBelowPip || curled;
  }

  const othersFolded =
    folded(indexTip, indexPip, indexMcp) &&
    folded(midTip,   midPip,   midMcp) &&
    folded(ringTip,  ringPip,  ringMcp) &&
    folded(pinkTip,  pinkPip,  pinkMcp);

  return othersFolded;
}

function updateThumbsUpDebounced(landmarks) {
  const raw = isThumbsUpRaw(landmarks);

  if (!thumbsUpActive) {
    if (raw) thumbsUpOnCount++;
    else thumbsUpOnCount = 0;

    if (thumbsUpOnCount >= THUMBS_ON_FRAMES) {
      thumbsUpActive = true;
      thumbsUpOnCount = 0;
      thumbsUpOffCount = 0;
      log("ThumbsUp ON");
    }
  } else {
    if (!raw) thumbsUpOffCount++;
    else thumbsUpOffCount = 0;

    if (thumbsUpOffCount >= THUMBS_OFF_FRAMES) {
      thumbsUpActive = false;
      thumbsUpOffCount = 0;
      thumbsUpOnCount = 0;
      log("ThumbsUp OFF");
    }
  }

  return thumbsUpActive;
}

// =====================
// MediaPipe init/start
// =====================
async function initHandLandmarker() {
  setStatus("Loading MediaPipe hand model…");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task" },
    numHands: 1,
    runningMode: "VIDEO"
  });

  setStatus("Model loaded. Click Start Camera.");
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 1280, height: 720 }
  });
  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => { video.play(); resolve(); };
  });

  resizeOverlay();
  window.addEventListener("resize", resizeOverlay);

  running = true;
  cursorEl.classList.add("visible");
  setStatus("Camera started. Index = cursor. Pinch opens. Thumbs-up closes.");
  log("Ready.");

  requestAnimationFrame(loop);
}

// =====================
// Main loop
// =====================
let wasPinching = false;
let lastScaleLogAt = 0;

// Keep last cursor position if hand briefly disappears
let lastCursorX = 0;
let lastCursorY = 0;
let lastT = 0;

function resetAllFiltersAndStates() {
  pinchClosed = false;
  pinchOnCount = 0;
  pinchOffCount = 0;

  thumbsUpActive = false;
  thumbsUpOnCount = 0;
  thumbsUpOffCount = 0;

  xFilter.reset();
  yFilter.reset();
  scaleFilter.reset();

  wasPinching = false;
}

function loop(nowMs) {
  if (!running || !handLandmarker) return;

  const videoTime = video.currentTime;
  if (videoTime === lastVideoTime) {
    requestAnimationFrame(loop);
    return;
  }
  lastVideoTime = videoTime;

  const results = handLandmarker.detectForVideo(video, nowMs);

  if (results?.landmarks?.length) {
    lastSeenMs = nowMs;

    const landmarks = results.landmarks[0];

    // Cursor position (index tip)
    let { x, y } = screenCoordsFromLandmark(landmarks[8]);

    // Heavy smoothing for jitter reduction
    x = xFilter.filter(x, nowMs);
    y = yFilter.filter(y, nowMs);

    lastCursorX = x;
    lastCursorY = y;

    cursorEl.style.left = `${x}px`;
    cursorEl.style.top = `${y}px`;

    // Smooth scale too
    const hsPxRaw = getHandScalePxRaw(landmarks);
    const hsPx = scaleFilter.filter(hsPxRaw, nowMs);
    const t = handScaleToT(hsPx);
    lastT = t;

    updateCursorSizeFromT(t);
    drawThumbIndexPointsScaled(landmarks, t);

    // Hover
    const stageRect = stage.getBoundingClientRect();
    const clientX = stageRect.left + x;
    const clientY = stageRect.top + y;

    const under = elementUnderPoint(clientX, clientY);

    if (mode === "grid") {
      stabilizeHover(under && under.classList?.contains("thumb") ? under : null, nowMs);
      // don’t highlight close in grid mode
      if (hoveredEl === closeBtn) applyHovered(null);
    } else {
      // viewer mode: hover close only (visual)
      stabilizeHover(under === closeBtn ? closeBtn : null, nowMs);
    }

    // Pinch for OPEN only
    const pinching = updatePinchStateDebounced(landmarks);
    if (pinching && !wasPinching) {
      if (mode === "grid") {
        const target = hoveredEl && hoveredEl.classList?.contains("thumb") ? hoveredEl : null;
        if (target) {
          clearSelected();
          selectedEl = target;
          selectedEl.classList.add("selected");
          openFile(selectedEl);
        } else {
          log("Pinch down: nothing selected");
        }
      }
      // viewer mode: pinch does nothing now
    }
    wasPinching = pinching;

    // Thumbs-up to CLOSE (viewer only)
    const thumbsUp = updateThumbsUpDebounced(landmarks);
    if (mode === "viewer" && thumbsUp) {
      // Close once per thumbs-up activation
      // (thumbsUpActive stays true for a bit; close and then wait until OFF)
      closeViewer();
      clearSelected();
      applyHovered(null);
      // prevent immediate re-close loops; require OFF before re-trigger
      thumbsUpActive = true;
      thumbsUpOnCount = 0;
      thumbsUpOffCount = 0;
    }

    // periodic calibration logging
    if (nowMs - lastScaleLogAt > 1200) {
      lastScaleLogAt = nowMs;
      log(`handScalePx(raw=${hsPxRaw.toFixed(1)} smooth=${hsPx.toFixed(1)}) t=${t.toFixed(2)}`);
    }
  } else {
    // No hand this frame.
    // Keep cursor/pointers for a short grace period to feel “persistent”.
    const since = nowMs - lastSeenMs;

    if (since <= HAND_LOST_GRACE_MS) {
      cursorEl.style.left = `${lastCursorX}px`;
      cursorEl.style.top = `${lastCursorY}px`;
      updateCursorSizeFromT(lastT);
      // keep overlay as-is
    } else {
      // true loss: clear overlay + hover, reset states
      clearOverlay();
      stabilizeHover(null, nowMs);
      applyHovered(null);

      resetAllFiltersAndStates();
    }
  }

  requestAnimationFrame(loop);
}

// =====================
// Wire up
// =====================
startBtn.addEventListener("click", async () => {
  if (!handLandmarker) {
    setStatus("Still loading model…");
    return;
  }
  if (!running) {
    try { await startCamera(); }
    catch (err) { setStatus("Error starting camera: " + (err?.message || err)); }
  }
});

// Mouse close for debugging (optional)
closeBtn.addEventListener("click", () => closeViewer());

if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
  setStatus("getUserMedia not supported in this browser.");
} else {
  initHandLandmarker().catch((err) => setStatus("Failed to init MediaPipe: " + (err?.message || err)));
}
