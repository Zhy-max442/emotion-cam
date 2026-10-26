/**
 * 浏览器端摄像头 + face-api.js 表情识别
 * 重点高亮 happy(微笑) / sad(伤心)
 */

const MODEL_URL = "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@0.22.2/weights";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const badge = document.getElementById("badge");
const emotionEl = document.getElementById("emotion");
const confEl = document.getElementById("conf");
const barsEl = document.getElementById("bars");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");

const ZH = {
  neutral: "平静",
  happy: "微笑",
  sad: "伤心",
  angry: "生气",
  fearful: "害怕",
  disgusted: "厌恶",
  surprised: "惊讶",
};

const ORDER = ["happy", "sad", "neutral", "surprised", "angry", "fearful", "disgusted"];

let running = false;
let modelsReady = false;
const smooth = {};

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function avgPush(key, value, alpha = 0.35) {
  if (smooth[key] == null) smooth[key] = value;
  else smooth[key] = smooth[key] * (1 - alpha) + value * alpha;
  return smooth[key];
}

function renderBars(expr) {
  barsEl.innerHTML = "";
  const entries = ORDER.map((k) => [k, expr[k] ?? 0]);
  for (const [k, v] of entries) {
    const row = document.createElement("div");
    row.className = "bar-row" + (k === "happy" ? " focus-happy" : k === "sad" ? " focus-sad" : "");
    const pct = Math.round(v * 100);
    row.innerHTML = `
      <span>${ZH[k] || k}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
      <span>${pct}%</span>
    `;
    barsEl.appendChild(row);
  }
}

function pickFocus(expr) {
  const happy = expr.happy || 0;
  const sad = expr.sad || 0;
  if (happy >= sad && happy >= 0.35) {
    return { key: "happy", label: "微笑 😊", conf: happy, cls: "happy" };
  }
  if (sad > happy && sad >= 0.35) {
    return { key: "sad", label: "伤心 😢", conf: sad, cls: "sad" };
  }
  let bestKey = "neutral";
  let best = -1;
  for (const [k, v] of Object.entries(expr)) {
    if (v > best) {
      best = v;
      bestKey = k;
    }
  }
  return {
    key: bestKey,
    label: ZH[bestKey] || bestKey,
    conf: best,
    cls: "other",
  };
}

async function loadModels() {
  setStatus("正在加载人脸模型…");
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
  ]);
  modelsReady = true;
  setStatus("模型已就绪，可以开启摄像头");
}

async function startCamera() {
  if (!modelsReady) {
    startBtn.disabled = true;
    await loadModels();
  }
  setStatus("请求摄像头权限…");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  video.srcObject = stream;
  await video.play();
  running = true;
  startBtn.textContent = "识别中…";
  startBtn.disabled = true;
  badge.textContent = "识别中";
  badge.className = "badge idle";
  setStatus("");
  loop();
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = video.videoWidth || Math.round(rect.width);
  canvas.height = video.videoHeight || Math.round(rect.height);
}

async function loop() {
  if (!running) return;
  if (video.readyState >= 2) {
    resizeCanvas();
    const det = await faceapi
      .detectSingleFace(
        video,
        new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.4 })
      )
      .withFaceLandmarks(true)
      .withFaceExpressions();

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (det) {
      const dims = { width: canvas.width, height: canvas.height };
      const resized = faceapi.resizeResults(det, dims);
      const box = resized.detection.box;
      const exprRaw = resized.expressions;
      const expr = {};
      for (const [k, v] of Object.entries(exprRaw)) {
        expr[k] = avgPush(k, v);
      }
      const focus = pickFocus(expr);

      ctx.strokeStyle = focus.cls === "happy" ? "#3ddc97" : focus.cls === "sad" ? "#6ea8fe" : "#ffd166";
      ctx.lineWidth = 3;
      ctx.strokeRect(box.x, box.y, box.width, box.height);

      badge.textContent = focus.label;
      badge.className = `badge ${focus.cls}`;
      emotionEl.textContent = focus.label;
      confEl.textContent = `${Math.round(focus.conf * 100)}%`;
      renderBars(expr);
    } else {
      badge.textContent = "未检测到人脸";
      badge.className = "badge idle";
      emotionEl.textContent = "—";
      confEl.textContent = "—";
    }
  }
  requestAnimationFrame(loop);
}

startBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (err) {
    console.error(err);
    setStatus("无法打开摄像头：请检查浏览器权限，或换 Chrome / Edge。");
    startBtn.disabled = false;
    startBtn.textContent = "重试开启摄像头";
  }
});

// 预加载模型，减少点击后等待
loadModels().catch((e) => {
  console.error(e);
  setStatus("模型加载失败，请检查网络后刷新页面。");
});
