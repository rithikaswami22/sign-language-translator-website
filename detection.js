const DETECTION_BUILD = "2026-03-17-v15";
console.log(`detection is LOADED (${DETECTION_BUILD})`);

// Alphabet labels for model output
let LABELS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X", "Y", "Z"
];

let model;
let cameraInstance;
let video = document.getElementById("video");
let predictedText = document.getElementById("predicted-letter");
let top3Select = document.getElementById("top3Select");
let useTop3Btn = document.getElementById("useTop3Btn");
let currentLetterBox = null;
let currentWordBox = null;
let modelStatus = document.getElementById("model-status");
let cameraStatus = document.getElementById("camera-status");
let videoFileInput = document.getElementById("videoFile");
let useVideoBtn = document.getElementById("useVideoBtn");
let languageSelect = document.getElementById("language-select");
let captureBtn = document.getElementById("captureBtn");
let isTop3Frozen = false;
let manualOverride = "";
let useImageModel = false;
let imageInputSize = 250;
let latestHandLandmarks = null;
let latestHandIsLeft = false;
let lastLandmarkTime = 0;
const LANDMARK_TTL_MS = 800;
const CROP_MARGIN = 0.35;
const MIN_CROP_BOX_SIZE = 0.12;
let cropToggle = document.getElementById("cropToggle");
let mirrorToggle = document.getElementById("mirrorToggle");
let cameraSelect = document.getElementById("cameraSelect");
let refreshCamerasBtn = document.getElementById("refreshCamerasBtn");

let lastLetter = "";
let currentWord = "";
let videoLoopId = null;
let usingFile = false;
let fallbackStream = null;
let fallbackLoopId = null;
let videoHealthLoopId = null;
let blackFrameCount = 0;
let lastHandTime = 0;
let isProcessing = false;
let lastSendTime = 0;
const VIDEO_FRAME_INTERVAL_MS = 120;
const CAMERA_FRAME_INTERVAL_MS = 150;
let lastPrediction = "";
let stableCount = 0;
let lastAccepted = "";
let lastAcceptedTime = 0;
const STABLE_FRAMES = 3;
const ACCEPT_INTERVAL_MS = 600;
const MIN_CONFIDENCE = 0.5;
let lastTop1 = "";
let lastTopScore = 0;
let smoothedProbs = null;
const EMA_ALPHA = 0.45;
const MIN_DISPLAY_CONFIDENCE = 0.4;
const MIN_TOP_MARGIN = 0.05;
const PREDICTION_HISTORY_SIZE = 7;
const PREDICTION_HISTORY_MIN_VOTES = 3;
let predictionHistory = [];

const LANGUAGE_CODES = {
    English: "en",
    Hindi: "hi",
    Marathi: "mr",
    Tamil: "ta",
    Telugu: "te",
    Gujarati: "gu",
    Bengali: "bn",
    French: "fr",
    German: "de",
    Russian: "ru",
    Arabic: "ar"
};

function firstTensor(input) {
    if (Array.isArray(input)) return input[0];
    return input;
}

class RandomFlipCompat extends tf.layers.Layer {
    static className = "RandomFlip";
    call(inputs) {
        return firstTensor(inputs);
    }
}

class RandomRotationCompat extends tf.layers.Layer {
    static className = "RandomRotation";
    call(inputs) {
        return firstTensor(inputs);
    }
}

class RandomZoomCompat extends tf.layers.Layer {
    static className = "RandomZoom";
    call(inputs) {
        return firstTensor(inputs);
    }
}

class RandomContrastCompat extends tf.layers.Layer {
    static className = "RandomContrast";
    call(inputs) {
        return firstTensor(inputs);
    }
}

class TrueDivideCompat extends tf.layers.Layer {
    static className = "TrueDivide";
    call(inputs) {
        if (!Array.isArray(inputs)) {
            return tf.div(inputs, tf.scalar(127.5));
        }
        const x = inputs[0];
        const y = inputs[1];
        if (typeof y === "undefined") {
            return tf.div(x, tf.scalar(127.5));
        }
        if (typeof y === "number") {
            return tf.div(x, tf.scalar(y));
        }
        return tf.div(x, y);
    }
}

class SubtractCompat extends tf.layers.Layer {
    static className = "Subtract";
    call(inputs) {
        if (!Array.isArray(inputs)) {
            return tf.sub(inputs, tf.scalar(1.0));
        }
        const x = inputs[0];
        const y = inputs[1];
        if (typeof y === "undefined") {
            return tf.sub(x, tf.scalar(1.0));
        }
        if (typeof y === "number") {
            return tf.sub(x, tf.scalar(y));
        }
        return tf.sub(x, y);
    }
}

tf.serialization.registerClass(RandomFlipCompat);
tf.serialization.registerClass(RandomRotationCompat);
tf.serialization.registerClass(RandomZoomCompat);
tf.serialization.registerClass(RandomContrastCompat);
tf.serialization.registerClass(TrueDivideCompat);
tf.serialization.registerClass(SubtractCompat);

// -------------------------
// Load TFJS Model
// -------------------------
const MODEL_CANDIDATES = [
    "./web-model-tfjs-fast/model.json",
    "/web-model-tfjs-fast/model.json",
    "./web-model-tfjs/model.json",
    "/web-model-tfjs/model.json"
];
const MODEL_ASSET_REV = "20260213a";
let lastModelLoadError = "";
const MODEL_LOAD_TIMEOUT_MS = 45000;

function withTimeout(promise, ms, label) {
    let timeoutId = null;
    const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
            reject(new Error(`${label || "Operation"} timed out after ${ms}ms`));
        }, ms);
    });
    return Promise.race([promise, timeoutPromise]).finally(() => {
        if (timeoutId) clearTimeout(timeoutId);
    });
}

async function loadModel() {
    let lastErr = null;

    for (const basePath of MODEL_CANDIDATES) {
        const path = `${basePath}?v=${MODEL_ASSET_REV}`;
        try {
            if (modelStatus) {
                modelStatus.innerText = `Loading model from ${path}...`;
            }
            // Prefer WebGL for better image-model accuracy and speed; fallback to CPU.
            if (tf.getBackend() !== "webgl") {
                try {
                    await tf.setBackend("webgl");
                } catch (err) {
                    await tf.setBackend("cpu");
                }
            }
            await tf.ready();
            model = await withTimeout(tf.loadLayersModel(path), MODEL_LOAD_TIMEOUT_MS, "Model load");
            console.log(`Model loaded from ${path}`);
            await loadLabelsForModel(path);

            if (model.inputs && model.inputs[0] && model.inputs[0].shape && model.inputs[0].shape.length === 4) {
                useImageModel = true;
                const h = model.inputs[0].shape[1];
                const w = model.inputs[0].shape[2];
                if (Number.isFinite(h) && Number.isFinite(w)) {
                    imageInputSize = Math.min(h, w);
                }
            } else {
                useImageModel = false;
            }

            const out = model.outputs && model.outputs[0] && model.outputs[0].shape
                ? model.outputs[0].shape[model.outputs[0].shape.length - 1]
                : null;
            if (out === 36 && LABELS.length !== 36) {
                LABELS = [
                    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                    "A", "B", "C", "D", "E", "F", "G", "H", "I",
                    "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                    "S", "T", "U", "V", "W", "X", "Y", "Z"
                ];
            }
            predictedText.innerText = "--";
            if (modelStatus) {
                modelStatus.innerText = `Model loaded (${useImageModel ? "image" : "landmark"} mode).`;
            }
            return;
        } catch (err) {
            lastErr = err;
            console.error(`Model load failed for ${path}`, err);
        }
    }

    const msg = lastErr && lastErr.message ? lastErr.message : String(lastErr);
    lastModelLoadError = msg;
    if (modelStatus) {
        modelStatus.innerText = `Model load failed: ${msg}`;
    }
    if (cameraStatus) {
        cameraStatus.innerText = `Model load failed: ${msg}`;
    }
}

async function loadLabelsForModel(modelPath) {
    try {
        const cleanPath = modelPath.split("?")[0];
        const labelsPath = `${cleanPath.replace(/model\.json$/, "class_indices.json")}?v=${MODEL_ASSET_REV}`;
        const res = await fetch(labelsPath);
        if (!res.ok) return;
        const mapping = await res.json();
        const entries = Object.entries(mapping).sort((a, b) => a[1] - b[1]);
        LABELS = entries.map(([label]) => label);
    } catch (err) {
        console.warn("Labels load failed, using default labels.", err);
    }
}

// -------------------------
// Preprocess Mediapipe Landmarks -> Tensor
// -------------------------
function preprocessLandmarks(landmarks) {
    const flat = [];

    landmarks.forEach(pt => {
        flat.push(pt.x);
        flat.push(pt.y);
        flat.push(pt.z);
    });

    return tf.tensor([flat]);
}

function normalizeLandmarksForModel(landmarks, isLeftHand) {
    if (!landmarks || !landmarks.length) {
        return landmarks;
    }
    if (!isLeftHand) {
        return landmarks;
    }
    // Landmark model was primarily trained on right-hand orientation.
    // Mirror left-hand x-coordinates into the same canonical space.
    return landmarks.map(pt => ({
        x: 1 - pt.x,
        y: pt.y,
        z: pt.z
    }));
}

function getTopIndices(probs, count = 3) {
    return Array.from(probs.keys())
        .sort((a, b) => probs[b] - probs[a])
        .slice(0, count);
}

function preferAlphabetProbs(probs) {
    if (!probs || !probs.length || !LABELS || LABELS.length !== probs.length) {
        return probs;
    }
    const letterIndices = [];
    for (let i = 0; i < LABELS.length; i += 1) {
        if (/^[A-Z]$/.test(LABELS[i])) {
            letterIndices.push(i);
        }
    }
    if (!letterIndices.length) {
        return probs;
    }
    const adjusted = Array.from(probs, () => 0);
    let sum = 0;
    for (const i of letterIndices) {
        const v = probs[i] || 0;
        adjusted[i] = v;
        sum += v;
    }
    if (sum <= 0) {
        return probs;
    }
    for (const i of letterIndices) {
        adjusted[i] = adjusted[i] / sum;
    }
    return adjusted;
}

function getModePrediction(items) {
    if (!items || !items.length) return "";
    const counts = new Map();
    for (const item of items) {
        if (!item) continue;
        counts.set(item, (counts.get(item) || 0) + 1);
    }
    let best = "";
    let bestCount = 0;
    for (const [label, count] of counts.entries()) {
        if (count > bestCount) {
            best = label;
            bestCount = count;
        }
    }
    if (bestCount < PREDICTION_HISTORY_MIN_VOTES) {
        return "";
    }
    return best;
}

function isFingerCurled(landmarks, tipIdx, pipIdx) {
    const tip = landmarks && landmarks[tipIdx];
    const pip = landmarks && landmarks[pipIdx];
    if (!tip || !pip) return false;
    return tip.y > pip.y;
}

function isTShapeLikely(landmarks) {
    if (!landmarks || landmarks.length < 21) return false;
    const thumbTip = landmarks[4];
    const indexMcp = landmarks[5];
    const middleMcp = landmarks[9];
    const ringMcp = landmarks[13];

    const curledCount = [
        isFingerCurled(landmarks, 8, 6),
        isFingerCurled(landmarks, 12, 10),
        isFingerCurled(landmarks, 16, 14),
        isFingerCurled(landmarks, 20, 18)
    ].filter(Boolean).length;

    const minX = Math.min(indexMcp.x, middleMcp.x) - 0.06;
    const maxX = Math.max(indexMcp.x, middleMcp.x) + 0.06;
    const thumbBetween = thumbTip.x >= minX && thumbTip.x <= maxX;

    const palmMidX = (indexMcp.x + ringMcp.x) / 2;
    const palmMidY = (indexMcp.y + ringMcp.y) / 2;
    const dx = thumbTip.x - palmMidX;
    const dy = thumbTip.y - palmMidY;
    const thumbPalmDist = Math.sqrt((dx * dx) + (dy * dy));
    const thumbNearPalm = thumbPalmDist < 0.2;

    return curledCount >= 3 && thumbBetween && thumbNearPalm;
}

function chooseLetterWithHeuristics(topIndices, smoothed, fallbackLetter, landmarks) {
    if (!landmarks || !topIndices || topIndices.length < 2) {
        return fallbackLetter;
    }
    const candidates = topIndices.map(i => LABELS[i]);
    const tIdx = candidates.indexOf("T");
    if (tIdx === -1) {
        return fallbackLetter;
    }
    const topScore = smoothed[topIndices[0]] || 0;
    const tScore = smoothed[topIndices[tIdx]] || 0;
    const closeToTop = (topScore - tScore) <= 0.12;
    if (closeToTop && isTShapeLikely(landmarks)) {
        return "T";
    }
    return fallbackLetter;
}

function updatePredictionFromProbs(probs, landmarks = null) {
    if (!probs || !probs.length) {
        return;
    }

    const filteredProbs = preferAlphabetProbs(probs);
    if (!smoothedProbs || smoothedProbs.length !== filteredProbs.length) {
        smoothedProbs = Array.from(filteredProbs);
    } else {
        for (let i = 0; i < filteredProbs.length; i += 1) {
            smoothedProbs[i] = (EMA_ALPHA * filteredProbs[i]) + ((1 - EMA_ALPHA) * smoothedProbs[i]);
        }
    }

    const topIndices = getTopIndices(smoothedProbs, 3);
    const top1Index = topIndices[0];
    const top2Index = topIndices[1];
    const letter = LABELS[top1Index];
    const correctedLetter = chooseLetterWithHeuristics(topIndices, smoothedProbs, letter, landmarks);
    const top1Score = smoothedProbs[top1Index] || 0;
    const top2Score = typeof top2Index === "number" ? (smoothedProbs[top2Index] || 0) : 0;
    const margin = top1Score - top2Score;
    const confident = top1Score >= MIN_DISPLAY_CONFIDENCE && margin >= MIN_TOP_MARGIN;

    predictionHistory.push(confident ? correctedLetter : "");
    if (predictionHistory.length > PREDICTION_HISTORY_SIZE) {
        predictionHistory.shift();
    }

    const votedLetter = getModePrediction(predictionHistory);
    const displayLetter = votedLetter || (confident ? correctedLetter : "--");
    lastTopScore = top1Score;
    lastTop1 = displayLetter === "--" ? "" : displayLetter;

    predictedText.innerText = manualOverride || displayLetter;
    if (currentLetterBox) {
        currentLetterBox.innerText = manualOverride || displayLetter;
    }

    if (top3Select && !isTop3Frozen && document.activeElement !== top3Select) {
        const previous = top3Select.value;
        top3Select.innerHTML = "";
        topIndices.forEach(i => {
            const opt = document.createElement("option");
            const label = LABELS[i];
            opt.value = label;
            opt.textContent = `${label} (${(smoothedProbs[i] * 100).toFixed(0)}%)`;
            top3Select.appendChild(opt);
        });
        if (previous && Array.from(top3Select.options).some(o => o.value === previous)) {
            top3Select.value = previous;
        }
    }

    if (!lastTop1) {
        stableCount = 0;
        lastPrediction = "";
        return;
    }

    if (lastTop1 === lastPrediction) {
        stableCount += 1;
    } else {
        lastPrediction = lastTop1;
        stableCount = 1;
    }
}

function clamp01(value) {
    return Math.min(1, Math.max(0, value));
}

async function predictImageFromVideo(landmarks) {
    if (!model) return;
    const prediction = tf.tidy(() => {
        let img = tf.browser.fromPixels(video).toFloat();
        let hasBatch = false;
        if (mirrorToggle && mirrorToggle.checked) {
            if (img.rank === 3) {
                img = img.expandDims(0);
                hasBatch = true;
            }
            img = tf.image.flipLeftRight(img);
        }

        const shouldCrop = !cropToggle || cropToggle.checked;
        if (shouldCrop && landmarks && landmarks.length) {
            const xs = landmarks.map(pt => pt.x);
            const ys = landmarks.map(pt => pt.y);
            let x1 = Math.min(...xs);
            let y1 = Math.min(...ys);
            let x2 = Math.max(...xs);
            let y2 = Math.max(...ys);

            const xMargin = (x2 - x1) * CROP_MARGIN;
            const yMargin = (y2 - y1) * CROP_MARGIN;

            x1 = clamp01(x1 - xMargin);
            y1 = clamp01(y1 - yMargin);
            x2 = clamp01(x2 + xMargin);
            y2 = clamp01(y2 + yMargin);

            // Expand to square to preserve full hand shape (helps M/N/P/Q/R/T).
            const cx = (x1 + x2) / 2;
            const cy = (y1 + y2) / 2;
            const side = Math.max(x2 - x1, y2 - y1);
            const half = side / 2;
            x1 = clamp01(cx - half);
            x2 = clamp01(cx + half);
            y1 = clamp01(cy - half);
            y2 = clamp01(cy + half);

            const bw = x2 - x1;
            const bh = y2 - y1;
            const validCrop = bw >= MIN_CROP_BOX_SIZE && bh >= MIN_CROP_BOX_SIZE;

            if (validCrop) {
                const boxes = [[y1, x1, y2, x2]];
                const boxInd = [0];
                if (!hasBatch) {
                    img = img.expandDims(0);
                    hasBatch = true;
                }
                img = tf.image.cropAndResize(img, boxes, boxInd, [imageInputSize, imageInputSize]);
            } else {
                if (!hasBatch) {
                    img = img.expandDims(0);
                    hasBatch = true;
                }
                img = img.resizeBilinear([imageInputSize, imageInputSize]);
            }
        } else {
            if (!hasBatch) {
                img = img.expandDims(0);
                hasBatch = true;
            }
            img = img.resizeBilinear([imageInputSize, imageInputSize]);
        }

        // MobileNet-based image classifiers are typically trained with [-1, 1] normalization.
        img = img.div(127.5).sub(1);

        return model.predict(img);
    });
    const probs = await prediction.data();
    prediction.dispose();
    updatePredictionFromProbs(probs, landmarks);
}

// -------------------------
// Predict Letter from Landmarks
// -------------------------
function predict(landmarks) {
    if (!model) return;

    const input = preprocessLandmarks(landmarks);
    const prediction = model.predict(input);
    const probs = prediction.dataSync();
    updatePredictionFromProbs(probs, landmarks);

    // Prevent duplicate spam
    // Auto-append disabled; use Capture button instead.

    input.dispose();
    prediction.dispose();
}

// -------------------------
// Mediapipe Hands Setup
// -------------------------
const hands = new Hands({
    locateFile: file => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
    modelComplexity: 1
});

hands.onResults(results => {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const lm = results.multiHandLandmarks[0];
        const handedness = results.multiHandedness && results.multiHandedness[0]
            ? results.multiHandedness[0].label
            : "";
        latestHandIsLeft = handedness === "Left";
        const modelLm = normalizeLandmarksForModel(lm, latestHandIsLeft);
        lastHandTime = Date.now();
        latestHandLandmarks = modelLm;
        lastLandmarkTime = Date.now();
        if (!useImageModel) {
            predict(modelLm);
        }
    } else {
        predictedText.innerText = "--";
        if (currentLetterBox) {
            currentLetterBox.innerText = "--";
        }
        if (top3Select) {
            top3Select.innerHTML = "";
        }
        stableCount = 0;
        smoothedProbs = null;
        predictionHistory = [];
        lastTop1 = "";
        lastTopScore = 0;
        latestHandLandmarks = null;
        latestHandIsLeft = false;
    }
});

// -------------------------
// Camera Start
// -------------------------
function setCameraStatus(msg) {
    if (cameraStatus) cameraStatus.innerText = msg;
}

function setModelStatus(msg) {
    if (modelStatus) modelStatus.innerText = msg;
}

function describeUserMediaError(err) {
    if (!err) return "Unknown camera error.";
    const name = err.name ? String(err.name) : "";
    const message = err.message ? String(err.message) : String(err);
    // Add a bit of actionable context for the common cases.
    if (name === "NotAllowedError" || name === "SecurityError") {
        return `${name}: camera permission denied. Allow camera access in the browser site settings and reload.`;
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        return `${name}: no camera device found. Connect/enable a webcam and try again.`;
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
        return `${name}: camera is already in use by another app (Zoom/Teams/Camera) or blocked by OS. Close other apps and retry.`;
    }
    if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
        return `${name}: camera constraints not supported. Try another device or remove constraints.`;
    }
    return name ? `${name}: ${message}` : message;
}

async function startCameraFallbackStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API unavailable (navigator.mediaDevices.getUserMedia missing).");
    }

    // Prefer front camera on phones; browsers ignore when not applicable.
    const selectedDeviceId = cameraSelect && cameraSelect.value ? String(cameraSelect.value) : "";
    const constraints = {
        audio: false,
        video: {
            facingMode: "user",
            width: { ideal: 500 },
            height: { ideal: 380 },
            ...(selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : {})
        }
    };

    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
        throw new Error(describeUserMediaError(err));
    }
    fallbackStream = stream;
    video.srcObject = stream;
    video.muted = true; // helps autoplay policies
    video.playsInline = true;
    video.autoplay = true;
    await video.play();

    const loop = async () => {
        if (!fallbackStream) return;
        const now = Date.now();
        if (!isProcessing && now - lastSendTime >= CAMERA_FRAME_INTERVAL_MS) {
            isProcessing = true;
            lastSendTime = now;
            try {
                await hands.send({ image: video });
                if (useImageModel) {
                    const useLm = latestHandLandmarks && (Date.now() - lastLandmarkTime) < LANDMARK_TTL_MS
                        ? latestHandLandmarks
                        : null;
                    await predictImageFromVideo(useLm);
                }
            } catch (err) {
                console.error("Fallback camera loop error", err);
            } finally {
                isProcessing = false;
            }
        }
        fallbackLoopId = setTimeout(loop, CAMERA_FRAME_INTERVAL_MS);
    };

    loop();
}

function waitForVideoFrames(timeoutMs = 2500) {
    return new Promise((resolve, reject) => {
        const start = Date.now();
        const tick = () => {
            const hasFrames = (video && video.videoWidth > 0 && video.videoHeight > 0);
            if (hasFrames) return resolve(true);
            if (Date.now() - start >= timeoutMs) {
                return reject(new Error("Camera stream started but no video frames are arriving."));
            }
            setTimeout(tick, 120);
        };
        tick();
    });
}

function attachVideoDebugHandlersOnce() {
    if (!video || video.__debugHandlersAttached) return;
    video.__debugHandlersAttached = true;

    const log = (evt) => {
        try {
            console.log(
                `[video] ${evt.type}`,
                { readyState: video.readyState, w: video.videoWidth, h: video.videoHeight }
            );
        } catch (_) {}
    };

    ["loadedmetadata", "playing", "pause", "waiting", "stalled", "emptied", "error"].forEach((type) => {
        video.addEventListener(type, log);
    });
}

function formatVideoDims() {
    if (!video) return "0×0";
    const w = Number(video.videoWidth) || 0;
    const h = Number(video.videoHeight) || 0;
    return `${w}×${h}`;
}

function updateCameraStatusWithDims(prefix = "Camera: running") {
    const dims = formatVideoDims();
    setCameraStatus(`${prefix} (${dims})`);
}

function startVideoHealthWatcher() {
    if (!video || video.__healthWatcher) return;
    video.__healthWatcher = setInterval(() => {
        if (!video) return;
        const w = Number(video.videoWidth) || 0;
        const h = Number(video.videoHeight) || 0;
        if (w > 0 && h > 0) {
            // Only update if we're not in an explicit error state.
            if (cameraStatus && !String(cameraStatus.innerText || "").startsWith("Camera error")) {
                updateCameraStatusWithDims("Camera: running");
            }
        } else {
            if (cameraStatus && !String(cameraStatus.innerText || "").startsWith("Camera error")) {
                setCameraStatus("Camera: waiting for frames (0×0)");
            }
        }
    }, 700);
}

function stopVideoHealthLoops() {
    if (video && video.__healthWatcher) {
        clearInterval(video.__healthWatcher);
        video.__healthWatcher = null;
    }
    if (videoHealthLoopId) {
        clearInterval(videoHealthLoopId);
        videoHealthLoopId = null;
    }
    blackFrameCount = 0;
}

function startBlackFrameDetector() {
    if (!video || videoHealthLoopId) return;
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });

    videoHealthLoopId = setInterval(() => {
        try {
            const w = Number(video.videoWidth) || 0;
            const h = Number(video.videoHeight) || 0;
            if (!ctx || w < 2 || h < 2) return;

            // Sample a small downscaled frame for speed.
            const sw = 64;
            const sh = 48;
            canvas.width = sw;
            canvas.height = sh;
            ctx.drawImage(video, 0, 0, sw, sh);
            const img = ctx.getImageData(0, 0, sw, sh).data;

            // Compute mean + variance of luma.
            let sum = 0;
            let sumSq = 0;
            const n = sw * sh;
            for (let i = 0; i < img.length; i += 4) {
                const r = img[i];
                const g = img[i + 1];
                const b = img[i + 2];
                const y = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
                sum += y;
                sumSq += y * y;
            }
            const mean = sum / n;
            const variance = (sumSq / n) - (mean * mean);

            const looksBlack = mean < 8 && variance < 12;
            if (looksBlack) {
                blackFrameCount += 1;
            } else {
                blackFrameCount = 0;
            }

            // After a few consecutive black samples, surface a real cause.
            if (blackFrameCount >= 4) {
                const track = video.srcObject && video.srcObject.getVideoTracks
                    ? video.srcObject.getVideoTracks()[0]
                    : null;
                const trackInfo = track
                    ? `track=${track.label || "camera"} state=${track.readyState} muted=${track.muted}`
                    : "no track";
                setCameraStatus(`Camera blocked (black frames). Check Windows camera privacy settings or close other apps. (${trackInfo})`);
            }
        } catch (e) {
            // ignore sampling errors
        }
    }, 900);
}

async function restartCameraWithSimpleConstraints() {
    // Try to restart with very permissive constraints (no width/height/facingMode).
    await stopCamera();
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        fallbackStream = stream;
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        video.autoplay = true;
        await video.play();
        await waitForVideoFrames(2500);
        startBlackFrameDetector();
        updateCameraStatusWithDims("Camera: running");
    } catch (err) {
        setCameraStatus(`Camera error: ${describeUserMediaError(err)}`);
    }
}

async function populateCameraDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices || !cameraSelect) return;
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cams = devices.filter(d => d && d.kind === "videoinput");
        const prev = cameraSelect.value;
        cameraSelect.innerHTML = "";

        if (!cams.length) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "No cameras found";
            cameraSelect.appendChild(opt);
            return;
        }

        cams.forEach((d, idx) => {
            const opt = document.createElement("option");
            opt.value = d.deviceId || "";
            opt.textContent = d.label || `Camera ${idx + 1}`;
            cameraSelect.appendChild(opt);
        });

        if (prev && Array.from(cameraSelect.options).some(o => o.value === prev)) {
            cameraSelect.value = prev;
        }
    } catch (err) {
        // ignore
    }
}

async function startCamera() {
    if (!model) {
        await loadModel();
    }
    if (!model) {
        setCameraStatus(
            lastModelLoadError
                ? `Camera blocked: model not loaded (${lastModelLoadError}).`
                : "Camera blocked: model not loaded."
        );
        return;
    }

    if (cameraInstance) {
        return;
    }

    stopVideoFile();

    // Helpful diagnostics for the most common “nothing happens” cases.
    // Camera access requires a secure context (HTTPS) except for localhost.
    if (typeof window !== "undefined" && window.isSecureContext === false) {
        setCameraStatus("Camera blocked: not a secure context. Use http://localhost:4000 (not file://).");
        return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraStatus("Camera blocked: browser does not support camera access (mediaDevices missing).");
        return;
    }

    // Ensure autoplay is allowed after the click.
    try {
        video.muted = true;
        video.playsInline = true;
        video.autoplay = true;
    } catch (_) {}

    attachVideoDebugHandlersOnce();
    startVideoHealthWatcher();

    // Prefer direct getUserMedia on Windows for reliability.
    // MediaPipe Camera helper is convenient but can behave inconsistently with some webcam/OS setups.
    try {
        await startCameraFallbackStream();
        await waitForVideoFrames(2500);
        await populateCameraDevices(); // labels appear after permission grant
        startBlackFrameDetector();
        updateCameraStatusWithDims("Camera: running");
        return;
    } catch (err) {
        console.warn("Direct getUserMedia start failed; trying MediaPipe Camera helper.", err);
        // continue into MediaPipe Camera flow below
    }

    cameraInstance = new Camera(video, {
        onFrame: async () => {
            const now = Date.now();
            if (isProcessing || now - lastSendTime < CAMERA_FRAME_INTERVAL_MS) {
                return;
            }
            isProcessing = true;
            lastSendTime = now;
            try {
                if (useImageModel) {
                    await hands.send({ image: video });
                    const useLm = latestHandLandmarks && (Date.now() - lastLandmarkTime) < LANDMARK_TTL_MS
                        ? latestHandLandmarks
                        : null;
                    await predictImageFromVideo(useLm);
                    return;
                }
                await hands.send({ image: video });
            } finally {
                isProcessing = false;
            }
        },
        width: 500,
        height: 380
    });

    try {
        await cameraInstance.start();
        // Some environments report “started” but the <video> never receives frames.
        try {
            await waitForVideoFrames(2500);
        } catch (noFramesErr) {
            console.warn("MediaPipe Camera started but no frames; switching to fallback.", noFramesErr);
            try {
                cameraInstance.stop();
            } catch (_) {}
            cameraInstance = null;
            await startCameraFallbackStream();
            await waitForVideoFrames(2500);
        }
        startBlackFrameDetector();
        updateCameraStatusWithDims("Camera: running");

        // If we detect persistent black frames, retry once with simpler constraints.
        setTimeout(() => {
            if (blackFrameCount >= 4) {
                restartCameraWithSimpleConstraints();
            }
        }, 4500);
    } catch (err) {
        console.error("Camera start failed", err);
        // Fall back to raw getUserMedia in case MediaPipe Camera helper fails.
        try {
            await startCameraFallbackStream();
            await waitForVideoFrames(2500);
            startBlackFrameDetector();
            updateCameraStatusWithDims("Camera: running");
        } catch (fallbackErr) {
            console.error("Camera fallback failed", fallbackErr);
            const msg = fallbackErr && fallbackErr.message ? fallbackErr.message : String(fallbackErr);
            setCameraStatus(`Camera error: ${msg}`);
            // Ensure we don't misleadingly appear "running".
            try {
                await stopCamera();
            } catch (_) {}
        }
    }
}

async function stopCamera() {
    if (!cameraInstance) {
        setCameraStatus("Camera: idle");
        return;
    }

    try {
        cameraInstance.stop();
    } catch (err) {
        console.error("Camera stop failed", err);
    }

    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    if (fallbackLoopId) {
        clearTimeout(fallbackLoopId);
        fallbackLoopId = null;
    }
    if (fallbackStream) {
        try {
            fallbackStream.getTracks().forEach(t => t.stop());
        } catch (e) {}
        fallbackStream = null;
    }

    stopVideoHealthLoops();

    cameraInstance = null;
    setCameraStatus("Camera: stopped");
    predictedText.innerText = "--";
    if (currentLetterBox) {
        currentLetterBox.innerText = "--";
    }
}

function stopVideoFile() {
    if (videoLoopId) {
        clearTimeout(videoLoopId);
        videoLoopId = null;
    }
    if (usingFile) {
        video.pause();
        if (video.src) {
            URL.revokeObjectURL(video.src);
        }
        video.removeAttribute("src");
        video.load();
        usingFile = false;
    }
    lastSendTime = 0;
}

async function startVideoFile(file) {
    if (!file) return;
    if (!model) {
        await loadModel();
    }
    if (!model) {
        if (cameraStatus) {
            cameraStatus.innerText = lastModelLoadError
                ? `Video blocked: model not loaded (${lastModelLoadError}).`
                : "Video blocked: model not loaded.";
        }
        return;
    }

    if (cameraInstance) {
        await stopCamera();
    }

    stopVideoFile();
    usingFile = true;

    const objectUrl = URL.createObjectURL(file);
    video.srcObject = null;
    video.src = objectUrl;
    video.muted = true;
    video.playsInline = true;
    video.loop = true;

    try {
        if (video.readyState < 2) {
            await new Promise(resolve => {
                video.onloadeddata = () => resolve();
            });
        }
        await video.play();
        if (cameraStatus) {
            cameraStatus.innerText = "Video: playing";
        }
    } catch (err) {
        console.error("Video play failed", err);
        if (cameraStatus) {
            cameraStatus.innerText = "Video error. Use a valid video file.";
        }
        return;
    }

    const processFrame = async () => {
        if (!usingFile || video.paused || video.ended) {
            if (cameraStatus && usingFile) {
                cameraStatus.innerText = "Video: stopped";
            }
            return;
        }
        const now = Date.now();
        if (video.readyState >= 2 && !isProcessing && now - lastSendTime >= VIDEO_FRAME_INTERVAL_MS) {
            isProcessing = true;
            lastSendTime = now;
            try {
                if (useImageModel) {
                    await hands.send({ image: video });
                    const useLm = latestHandLandmarks && (Date.now() - lastLandmarkTime) < LANDMARK_TTL_MS
                        ? latestHandLandmarks
                        : null;
                    await predictImageFromVideo(useLm);
                } else {
                    await hands.send({ image: video });
                }
            } catch (err) {
                console.error("Hands processing error", err);
            } finally {
                isProcessing = false;
            }
        }
        videoLoopId = setTimeout(processFrame, VIDEO_FRAME_INTERVAL_MS);
    };

    processFrame();
}

// -------------------------
// Button Controls
// -------------------------
document.getElementById("startBtn").addEventListener("click", () => {
    startCamera();
});

document.getElementById("stopBtn").addEventListener("click", () => {
    stopCamera();
});

document.getElementById("clearBtn").addEventListener("click", () => {
    stopCamera();
    stopVideoFile();
    currentWord = "";
    if (currentWordBox) {
        currentWordBox.innerText = "--";
    }
    lastLetter = "";
    lastPrediction = "";
    stableCount = 0;
    lastAccepted = "";
    lastAcceptedTime = 0;
    lastTop1 = "";
    lastTopScore = 0;
    smoothedProbs = null;
    predictionHistory = [];
    manualOverride = "";
    predictedText.innerText = "--";
    if (currentLetterBox) {
        currentLetterBox.innerText = "--";
    }
    if (top3Select) {
        top3Select.innerHTML = "";
    }
    document.getElementById("translated-text").innerText = "--";
});

if (useVideoBtn) {
    useVideoBtn.addEventListener("click", () => {
        const file = videoFileInput && videoFileInput.files && videoFileInput.files[0];
        startVideoFile(file);
    });
}

if (captureBtn) {
    captureBtn.addEventListener("click", () => {
        if (!lastTop1) {
            return;
        }
        if (lastTopScore < MIN_CONFIDENCE) {
            return;
        }
        if (stableCount < STABLE_FRAMES) {
            return;
        }
        const now = Date.now();
        if (lastTop1 !== lastAccepted || now - lastAcceptedTime >= ACCEPT_INTERVAL_MS) {
            currentWord += lastTop1;
            currentWordBox.innerText = currentWord;
            lastAccepted = lastTop1;
            lastAcceptedTime = now;
            translateWord(currentWord);
        }
    });
}

if (useTop3Btn && top3Select) {
    useTop3Btn.addEventListener("click", () => {
        const choice = top3Select.value;
        if (!choice) return;
        const now = Date.now();
        if (choice !== lastAccepted || now - lastAcceptedTime >= ACCEPT_INTERVAL_MS) {
            currentWord += choice;
            if (currentWordBox) {
                currentWordBox.innerText = currentWord;
            }
            lastAccepted = choice;
            lastAcceptedTime = now;
            translateWord(currentWord);
        }
        manualOverride = choice;
        predictedText.innerText = choice;
        if (currentLetterBox) {
            currentLetterBox.innerText = choice;
        }
    });
}

if (top3Select) {
    top3Select.addEventListener("focus", () => {
        isTop3Frozen = true;
    });
    top3Select.addEventListener("blur", () => {
        isTop3Frozen = false;
    });
    top3Select.addEventListener("change", () => {
        isTop3Frozen = true;
    });
}

if (languageSelect) {
    languageSelect.addEventListener("change", () => {
        translateWord(currentWord);
    });
}

// -------------------------
// INIT
// -------------------------
if (modelStatus) {
    modelStatus.innerText = "Model idle. Click Start to load.";
}

// Populate camera list early (labels may be empty until permission is granted).
populateCameraDevices();
if (refreshCamerasBtn) {
    refreshCamerasBtn.addEventListener("click", () => populateCameraDevices());
}

async function translateWord(word) {
    const lang = languageSelect ? languageSelect.value : "English";
    if (!word || word.length === 0) return;

    try {
        if (lang === "English") {
            document.getElementById("translated-text").innerText = word;
            return;
        }

        const langCode = LANGUAGE_CODES[lang] || "en";
        const res = await fetch(
            `https://api.mymemory.translated.net/get?q=${encodeURIComponent(word)}&langpair=en|${langCode}`
        );

        const data = await res.json();
        const translated = data.responseData && data.responseData.translatedText
            ? data.responseData.translatedText
            : word;

        if (!translated || translated.toLowerCase() === word.toLowerCase()) {
            document.getElementById("translated-text").innerText = word;
            return;
        }
        document.getElementById("translated-text").innerText = translated;

    } catch (err) {
        console.error("Translation Error:", err);
    }
}
