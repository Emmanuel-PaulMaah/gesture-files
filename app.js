import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

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
const closeBtn = document.getElementById("closeBtn");

const overlayCtx = overlay.getContext("2d");

// =====================
// Demo "files" (online URLs)
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
// One Euro filter
// =====================
class OneEuroFilter {
  constructor(freq = 60, minCutoff = 1.2, beta = 0.05, dCutoff = 1.0) {
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

const xFilter = new OneEuroFilter(60, 1.2, 0.05, 1.0);
const yFilter = new OneEuroFilter(60, 1.2, 0.05, 1.0);

// =====================
// Pinch debounce + hysteresis
// =====================
let pinchClosed = false;
let pinchOnCount = 0;
let pinchOffCount = 0;

const PINCH_ON = 0.30;
const PINCH_OFF = 0.60;
const ON_FRAMES = 2;
const OFF_FRAMES = 3;

// =====================
// State
// =====================
let handLandmarker = null;
let running = false;
let lastVideoTime = -1;

let wasPinching = false;
let isPinching = false;

let hoveredEl = null;
let selectedEl = null;

let mode = "grid"; // "grid" | "viewer"

// =====================
// Geometry + helpers
// =====================
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }

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

// Convert normalized "hand scale" to pixels using stage size
function getHandScalePx(landmarks) {
  const hsNorm = getHandScaleNorm(landmarks);
  const rect = stage.getBoundingClientRect();
  const minDim = Math.min(rect.width, rect.height);
  return hsNorm * minDim;
}

// Mirror X so left/right feels natural
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

  if (el === closeBtn || closeBtn.contains(el)) return closeBtn;

  const thumb = el.closest?.(".thumb.file");
  if (thumb) return thumb;

  return null;
}

function setHovered(next) {
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

function updatePinchStateDebounced(landmarks) {
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];

  const rawDist = distance(thumbTip, indexTip);
  const handScale = getHandScaleNorm(landmarks);
  const normDist = rawDist / handScale;

  if (!pinchClosed) {
    if (normDist < PINCH_ON) pinchOnCount++;
    else pinchOnCount = 0;

    if (pinchOnCount >= ON_FRAMES) {
      pinchClosed = true;
      pinchOnCount = 0;
      pinchOffCount = 0;
      log(`Pinch ON (normDist=${normDist.toFixed(2)})`);
    }
  } else {
    if (normDist > PINCH_OFF) pinchOffCount++;
    else pinchOffCount = 0;

    if (pinchOffCount >= OFF_FRAMES) {
      pinchClosed = false;
      pinchOffCount = 0;
      pinchOnCount = 0;
      log(`Pinch OFF (normDist=${normDist.toFixed(2)})`);
    }
  }

  return pinchClosed;
}

// =====================
// Overlay drawing (distance-scaled pointers)
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

/**
 * Maps hand pixel scale -> t in [0..1] => pointer radius.
 * Tune FAR_PX and NEAR_PX for your camera + typical framing.
 */
function handScaleToT(hsPx) {
  const FAR_PX = 60;   // far hand ~ smaller
  const NEAR_PX = 180; // close hand ~ bigger
  return clamp((hsPx - FAR_PX) / (NEAR_PX - FAR_PX), 0, 1);
}

function drawThumbIndexPointsScaled(landmarks) {
  clearOverlay();

  const thumb = screenCoordsFromLandmark(landmarks[4]);
  const index = screenCoordsFromLandmark(landmarks[8]);

  const hsPx = getHandScalePx(landmarks);
  const t = handScaleToT(hsPx);

  const thumbR = lerp(5, 12, t);
  const indexR = lerp(7, 16, t);

  drawPoint(thumb.x, thumb.y, thumbR, "rgba(255,255,255,0.30)", "rgba(255,255,255,0.85)");
  drawPoint(index.x, index.y, indexR, "rgba(120,180,255,0.35)", "rgba(120,180,255,0.95)");

  return { t, hsPx };
}

function updateCursorSizeFromT(t) {
  // Cursor diameter range 14..28 px
  const d = lerp(14, 28, t);
  cursorEl.style.width = `${d}px`;
  cursorEl.style.height = `${d}px`;
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
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    },
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
  setStatus("Camera started. Use index as cursor; pinch to click.");
  log("Ready.");

  requestAnimationFrame(loop);
}

// =====================
// Main loop
// =====================
let lastScaleLogAt = 0;

function loop() {
  if (!running || !handLandmarker) return;

  const videoTime = video.currentTime;
  if (videoTime === lastVideoTime) {
    requestAnimationFrame(loop);
    return;
  }
  lastVideoTime = videoTime;

  const nowMs = performance.now();
  const results = handLandmarker.detectForVideo(video, nowMs);

  if (results?.landmarks?.length) {
    const landmarks = results.landmarks[0];

    // pinch state
    isPinching = updatePinchStateDebounced(landmarks);

    // cursor position from index fingertip
    let { x, y } = screenCoordsFromLandmark(landmarks[8]);
    x = xFilter.filter(x, nowMs);
    y = yFilter.filter(y, nowMs);

    cursorEl.style.left = `${x}px`;
    cursorEl.style.top = `${y}px`;

    // draw pointers, compute scaling t
    const { t, hsPx } = drawThumbIndexPointsScaled(landmarks);
    updateCursorSizeFromT(t);

    // occasional scale logging to help calibrate FAR/NEAR
    if (nowMs - lastScaleLogAt > 1000) {
      lastScaleLogAt = nowMs;
      log(`handScalePx=${hsPx.toFixed(1)} t=${t.toFixed(2)}`);
    }

    // hover detection uses client coords
    const stageRect = stage.getBoundingClientRect();
    const clientX = stageRect.left + x;
    const clientY = stageRect.top + y;

    const under = elementUnderPoint(clientX, clientY);

    if (mode === "grid") {
      setHovered(under && under.classList?.contains("thumb") ? under : null);
      closeBtn.classList.remove("hovered");
    } else {
      setHovered(under === closeBtn ? closeBtn : null);
    }

    // pinch edge triggers action
    if (isPinching && !wasPinching) {
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
      } else {
        if (hoveredEl === closeBtn) {
          closeViewer();
          clearSelected();
          setHovered(null);
        } else {
          log("Pinch down in viewer: not on (X)");
        }
      }
    }

    wasPinching = isPinching;
  } else {
    // hand lost
    clearOverlay();
    setHovered(null);

    wasPinching = false;
    isPinching = false;

    pinchClosed = false;
    pinchOnCount = 0;
    pinchOffCount = 0;

    xFilter.reset();
    yFilter.reset();
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

// mouse close for debugging
closeBtn.addEventListener("click", () => closeViewer());

if (!("mediaDevices" in navigator && "getUserMedia" in navigator.mediaDevices)) {
  setStatus("getUserMedia not supported in this browser.");
} else {
  initHandLandmarker().catch((err) => setStatus("Failed to init MediaPipe: " + (err?.message || err)));
}
