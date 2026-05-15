const els = {
  mode: document.querySelector("#mode"),
  statusBanner: document.querySelector("#statusBanner"),
  statusIcon: document.querySelector("#statusIcon"),
  cameraBtn: document.querySelector("#cameraBtn"),
  runBtn: document.querySelector("#runBtn"),
  calibrateBtn: document.querySelector("#calibrateBtn"),
  sessionBtn: document.querySelector("#sessionBtn"),
  privacyToggle: document.querySelector("#privacyToggle"),
  careContext: document.querySelector("#careContext"),
  healthcareFields: document.querySelector("#healthcareFields"),
  patientId: document.querySelector("#patientId"),
  observationType: document.querySelector("#observationType"),
  eventNote: document.querySelector("#eventNote"),
  consent: document.querySelector("#consent"),
  feedback: document.querySelector("#feedback"),
  feedbackBtn: document.querySelector("#feedbackBtn"),
  actionCard: document.querySelector(".action-card"),
  video: document.querySelector("#video"),
  canvas: document.querySelector("#captureCanvas"),
  stateTitle: document.querySelector("#stateTitle"),
  qualityPill: document.querySelector("#qualityPill"),
  statusMessage: document.querySelector("#statusMessage"),
  signalReasons: document.querySelector("#signalReasons"),
  qualityList: document.querySelector("#qualityList"),
  faceBadge: document.querySelector("#faceBadge"),
  signalBadge: document.querySelector("#signalBadge"),
  recommendedAction: document.querySelector("#recommendedAction"),
  systemTime: document.querySelector("#systemTime"),
  alertBanner: document.querySelector("#alertBanner"),
  summaryList: document.querySelector("#summaryList"),
  eventList: document.querySelector("#eventList"),
};

const values = {
  fatigue: document.querySelector("#fatigueValue"),
  attention: document.querySelector("#attentionValue"),
  tension: document.querySelector("#tensionValue"),
  readiness: document.querySelector("#readinessValue"),
  fatigueState: document.querySelector("#fatigueState"),
  attentionState: document.querySelector("#attentionState"),
  tensionState: document.querySelector("#tensionState"),
  postureState: document.querySelector("#postureState"),
  fatigueBar: document.querySelector("#fatigueBar"),
  attentionBar: document.querySelector("#attentionBar"),
  tensionBar: document.querySelector("#tensionBar"),
  readinessBar: document.querySelector("#readinessBar"),
};

const charts = {
  fatigue: document.querySelector("#fatigueChart"),
  attention: document.querySelector("#attentionChart"),
  signal: document.querySelector("#tensionChart"),
};

const SEVERITY_ORDER = {
  normal: 0,
  active: 0,
  observed: 0,
  stable_observation: 0,
  resting: 0,
  insufficient_signal: 1,
  watch: 2,
  needs_review: 2,
  elevated: 3,
  waiting_for_consent: 3,
  critical: 4,
  urgent_review: 4,
};

const PULL_OVER_EVENTS = new Set([
  "sustained_eye_closure",
  "microsleep_detected",
  "perclos_critical",
  "head_nod_cluster",
  "rapid_yawning_compound",
  "compound_fatigue_escalation",
  "professional_limit_exceeded",
]);
const PULL_OVER_HOLD_SECONDS = 60;

let stream = null;
let running = true;
let sessionActive = false;
let sending = false;
let frameSocket = null;
let persistentPullOver = null;
const dismissedPullOverKeys = new Set();
const history = { fatigue: [], attention: [], signal: [] };

els.cameraBtn.addEventListener("click", toggleCamera);
els.runBtn.addEventListener("click", () => {
  running = !running;
  els.runBtn.textContent = running ? "Pause analysis" : "Resume analysis";
});
els.calibrateBtn.addEventListener("click", () => postJson("/api/calibrate", {}));
els.sessionBtn.addEventListener("click", startNewSession);
els.mode.addEventListener("change", syncModeControls);
els.privacyToggle.addEventListener("change", () => els.video.classList.toggle("privacy", els.privacyToggle.checked));
els.feedbackBtn.addEventListener("click", saveFeedback);

syncModeControls();
updateSystemTime();
setInterval(sendFrame, 220);
setInterval(updateSystemTime, 1000);
setInterval(updatePersistentPullOverCountdown, 1000);
connectFrameSocket();

async function toggleCamera() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
    els.video.srcObject = null;
    els.cameraBtn.textContent = "Start Camera";
    els.stateTitle.textContent = "Camera stopped";
    els.statusMessage.textContent = "Start the camera to begin local analysis.";
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    els.stateTitle.textContent = "Camera access requires http://127.0.0.1:8000 or localhost";
    return;
  }

  els.cameraBtn.disabled = true;
  els.stateTitle.textContent = "Starting camera";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
    els.cameraBtn.textContent = "Stop Camera";
    els.stateTitle.textContent = "Camera active";
    els.statusMessage.textContent = "Live camera feed is active. Analysis will start automatically.";
  } catch (err) {
    stream = null;
    els.video.srcObject = null;
    els.stateTitle.textContent = cameraErrorMessage(err);
    els.statusMessage.textContent = "Check browser permissions and make sure no other app is using the camera.";
  } finally {
    els.cameraBtn.disabled = false;
  }
}

function cameraErrorMessage(err) {
  if (!err || !err.name) return "Camera could not be started";
  if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
    return "Camera permission was denied";
  }
  if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
    return "No camera was found";
  }
  if (err.name === "NotReadableError" || err.name === "TrackStartError") {
    return "Camera is already in use by another app";
  }
  if (err.name === "OverconstrainedError") {
    return "Camera does not support the requested settings";
  }
  return `Camera error: ${err.name}`;
}

async function startNewSession() {
  els.sessionBtn.disabled = true;
  clearDashboardState();
  try {
    await postJson("/api/session", { mode: els.mode.value, action: "start" });
    sessionActive = true;
    els.sessionBtn.textContent = "Start New Session";
    els.stateTitle.textContent = `${els.mode.value} Status: Normal`;
    els.statusMessage.textContent = "New session started. Waiting for fresh camera analysis.";
  } finally {
    els.sessionBtn.disabled = false;
  }
}

function clearDashboardState() {
  history.fatigue = [];
  history.attention = [];
  history.signal = [];
  drawChart(charts.fatigue, history.fatigue, "#ef4444", "Fatigue");
  drawChart(charts.attention, history.attention, "#f97316", "Attention");
  drawChart(charts.signal, history.signal, "#f59e0b", "Signal Quality");

  values.fatigue.textContent = "0";
  values.attention.textContent = "0";
  values.tension.textContent = "0";
  values.readiness.textContent = "0";
  values.fatigueState.textContent = "idle";
  values.attentionState.textContent = "idle";
  values.tensionState.textContent = "idle";
  values.postureState.textContent = "idle";
  setBar(values.fatigueBar, 0);
  setBar(values.attentionBar, 0);
  setBar(values.tensionBar, 0);
  setBar(values.readinessBar, 0);

  els.statusBanner.className = "status-banner normal";
  els.statusIcon.textContent = "OK";
  els.qualityPill.textContent = "Stable";
  els.qualityPill.className = "pill ok";
  els.signalReasons.textContent = "New session started.";
  els.faceBadge.textContent = "Face pending";
  els.signalBadge.textContent = "Signal pending";
  els.recommendedAction.textContent = "New session started. Waiting for fresh analysis.";
  els.actionCard.className = "action-card normal";
  els.alertBanner.classList.add("hidden");
  els.alertBanner.classList.remove("critical");
  els.alertBanner.textContent = "";
  persistentPullOver = null;
  dismissedPullOverKeys.clear();
  renderDefinitionList(els.summaryList, {});
  renderDefinitionList(els.qualityList, {});
  renderEvents([]);
}

async function saveFeedback() {
  if (!els.feedback.value) return;
  await postJson("/api/feedback", { feedback: els.feedback.value });
  els.feedback.value = "";
}

async function sendFrame() {
  if (!stream || !running || sending || els.video.readyState < 2) return;
  sending = true;
  try {
    const width = 640;
    const ratio = els.video.videoHeight / Math.max(1, els.video.videoWidth);
    els.canvas.width = width;
    els.canvas.height = Math.round(width * ratio);
    const ctx = els.canvas.getContext("2d", { willReadFrequently: false });
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
    const image = els.canvas.toDataURL("image/jpeg", 0.68);
    const payload = {
      image,
      mode: els.mode.value,
      care_context: els.careContext.value,
      healthcare: healthcarePayload(),
    };
    if (frameSocket && frameSocket.readyState === WebSocket.OPEN) {
      frameSocket.send(JSON.stringify(payload));
      return;
    }
    const data = await postJson("/api/frame", payload);
    render(data);
  } catch (err) {
    els.stateTitle.textContent = err.message || "Analysis failed";
    sending = false;
  } finally {
    if (frameSocket && frameSocket.readyState === WebSocket.OPEN) return;
    sending = false;
  }
}

function connectFrameSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  frameSocket = new WebSocket(`${protocol}://${window.location.host}/ws/frame`);
  frameSocket.addEventListener("message", (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.ok) render(payload.data);
      else els.stateTitle.textContent = payload.error || "Analysis failed";
    } catch (err) {
      els.stateTitle.textContent = "Analysis response could not be read";
    } finally {
      sending = false;
    }
  });
  frameSocket.addEventListener("close", () => {
    sending = false;
    setTimeout(connectFrameSocket, 1500);
  });
  frameSocket.addEventListener("error", () => {
    sending = false;
  });
}

function healthcarePayload() {
  return {
    patient_session_id: els.patientId.value,
    observation_type: els.observationType.value,
    note: els.eventNote.value,
    consent_captured: els.consent.checked,
  };
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }
  return res.json();
}

function render(data) {
  updatePersistentPullOver(data);
  const state = displayState(data);
  const status = statusVisual(data);
  els.statusBanner.className = `status-banner ${status.className}`;
  els.statusIcon.textContent = status.icon;
  els.stateTitle.textContent = state;
  els.statusMessage.textContent = statusMessage(data);
  els.qualityPill.textContent = data.quality.signal_ok ? "Stable" : "Degraded";
  els.qualityPill.className = `pill ${data.quality.signal_ok ? "ok" : "warn"}`;
  els.signalReasons.textContent = signalCopy(data.quality);
  els.faceBadge.textContent = data.quality.face_present ? "Face detected" : "Face not detected";
  els.signalBadge.textContent = data.quality.signal_ok ? "Signal stable" : "Signal degraded";
  els.recommendedAction.textContent = recommendedAction(data);
  els.actionCard.className = `action-card ${status.className}`;

  values.fatigue.textContent = Math.round(data.metrics.fatigue);
  values.attention.textContent = Math.round(data.metrics.attention);
  values.tension.textContent = Math.round(data.metrics.tension / 10);
  values.readiness.textContent = Math.round(data.metrics.readiness);
  values.fatigueState.textContent = data.states.fatigue;
  values.attentionState.textContent = data.states.attention;
  values.tensionState.textContent = data.states.tension;
  values.postureState.textContent = `${data.states.posture} / ${data.states.distance}`;
  setBar(values.fatigueBar, data.metrics.fatigue);
  setBar(values.attentionBar, data.metrics.attention);
  setBar(values.readinessBar, data.metrics.readiness);
  setBar(values.tensionBar, data.metrics.tension);
  updateMetricClasses(data);

  pushHistory("fatigue", data.metrics.fatigue);
  pushHistory("attention", data.metrics.attention);
  pushHistory("signal", signalScore(data.quality));
  drawChart(charts.fatigue, history.fatigue, "#ef4444", "Fatigue");
  drawChart(charts.attention, history.attention, "#f97316", "Attention");
  drawChart(charts.signal, history.signal, "#f59e0b", "Signal Quality");

  renderDefinitionList(els.summaryList, data.summary);
  renderDefinitionList(els.qualityList, { ...data.quality, ...attentionDebugValues(data.debug?.attention) });
  renderEvents(activeEvents(data));
  renderAlert(data);
}

function statusVisual(data) {
  if (persistentPullOver) return { className: "critical", icon: "!" };
  const state = String(data.summary?.state || "neutral");
  if (data.calibrating) return { className: "watch", icon: "..." };
  if (["normal", "active", "observed", "stable_observation", "resting"].includes(state)) {
    return { className: state, icon: "✓" };
  }
  if (["watch", "needs_review"].includes(state)) {
    return { className: state, icon: "!" };
  }
  if (["elevated", "insufficient_signal", "waiting_for_consent"].includes(state)) {
    return { className: state, icon: "!" };
  }
  if (["critical", "urgent_review"].includes(state)) {
    return { className: state, icon: "!" };
  }
  return { className: "neutral", icon: "i" };
}

function displayState(data) {
  if (persistentPullOver) return "Driver Status: Critical";
  if (data.calibrating) return "Calibrating";
  if (data.summary && data.summary.state) return `${data.summary.mode} Status: ${titleCase(data.summary.state)}`;
  return "Monitoring";
}

function statusMessage(data) {
  const latest = currentEvent(data);
  if (latest) return latest.message || latest.event;
  if (!data.quality.signal_ok) return signalCopy(data.quality);
  const risk = String(data.summary?.state || "normal");
  if (risk === "watch") return "Early fatigue indicators are present. Keep monitoring; no stop action is active.";
  if (risk === "elevated") return "Fatigue indicators are elevated. Plan a safe rest break if this continues.";
  if (risk === "critical") return "Critical fatigue condition detected. Stop at the nearest safe place.";
  if (data.summary?.mode === "Driver") return "Camera signal is stable. Continue monitoring for fatigue risk.";
  return "Live analysis is running with usable signal quality.";
}

function signalCopy(quality) {
  if (!quality) return "Camera signal is unavailable.";
  const reasons = Array.isArray(quality.reasons) ? quality.reasons : [];
  if (quality.signal_ok) return "Camera signal is reliable.";
  if (!reasons.length) return "Check lighting, face visibility, and blur.";
  return `Check ${reasons.join(", ")}.`;
}

function recommendedAction(data) {
  const latest = currentEvent(data);
  if (latest) return latest.message || "Review the latest event.";
  if (!data.quality.signal_ok) return "Improve lighting and keep your face visible to the camera.";
  const risk = String(data.summary?.state || "normal");
  if (risk === "critical") return "Stop at the nearest safe place.";
  if (risk === "elevated") return "Plan a safe rest break if elevated fatigue signs continue.";
  if (risk === "watch") return "Monitor fatigue. No stop instruction is active right now.";
  return "Continue monitoring. Keep your face visible and posture upright.";
}

function signalScore(quality) {
  if (!quality) return 0;
  let score = quality.signal_ok ? 86 : 48;
  if (quality.face_present) score += 8;
  score += Math.min(6, Number(quality.face_ratio || 0) * 100);
  score -= Math.min(30, (quality.reasons || []).length * 10);
  return Math.max(0, Math.min(100, score));
}

function setBar(node, value) {
  if (!node) return;
  node.style.width = `${Math.max(0, Math.min(100, value))}%`;
}

function updateMetricClasses(data) {
  setMetricClass(values.fatigue.closest(".metric-card"), data.metrics.fatigue, [
    [80, "high"],
    [65, "medium"],
    [50, "moderate"],
    [0, "good"],
  ]);
  setMetricClass(values.attention.closest(".metric-card"), data.metrics.attention, [
    [75, "good"],
    [55, "moderate"],
    [35, "medium"],
    [0, "high"],
  ]);
  setMetricClass(values.readiness.closest(".metric-card"), data.metrics.readiness, [
    [75, "good"],
    [55, "moderate"],
    [35, "medium"],
    [0, "high"],
  ]);
  setMetricClass(values.tension.closest(".metric-card"), data.metrics.tension / 10, [
    [7, "high"],
    [4, "medium"],
    [2, "moderate"],
    [0, "good"],
  ]);
}

function setMetricClass(card, value, thresholds) {
  if (!card) return;
  card.classList.remove("high", "medium", "moderate", "good");
  const match = thresholds.find(([min]) => value >= min);
  card.classList.add(match ? match[1] : "good");
}

function pushHistory(key, value) {
  history[key].push(value);
  if (history[key].length > 180) history[key].shift();
}

function drawChart(canvas, series, color, label) {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(280, Math.floor(rect.width));
  canvas.height = Math.max(112, Math.floor(rect.height || 126));
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#e5e7eb";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height - 34);
  ctx.lineTo(canvas.width, canvas.height - 34);
  ctx.stroke();
  ctx.fillStyle = "#475569";
  ctx.font = "12px system-ui";
  ctx.fillText(label, 12, 22);
  if (series.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  series.forEach((value, index) => {
    const x = (index / (series.length - 1)) * canvas.width;
    const y = canvas.height - 18 - (Math.max(0, Math.min(100, value)) / 100) * (canvas.height - 42);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderDefinitionList(node, data) {
  node.innerHTML = "";
  Object.entries(data || {}).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    const dd = document.createElement("dd");
    dt.textContent = key.replaceAll("_", " ");
    dd.textContent = Array.isArray(value) ? value.join(", ") || "None" : formatValue(value);
    node.append(dt, dd);
  });
}

function attentionDebugValues(debug) {
  if (!debug) return {};
  return {
    attention_state: debug.state,
    offscreen_secs: debug.offscreen_duration,
    gaze_x_delta: debug.gaze_x_delta,
    gaze_y_delta: debug.gaze_y_delta,
    head_yaw_delta_deg: debug.head_yaw_delta_deg,
    attention_signal_ok: debug.attention_signal_ok,
  };
}

function renderEvents(events) {
  els.eventList.innerHTML = "";
  if (!events.length) {
    const empty = document.createElement("div");
    empty.className = "event";
    const title = document.createElement("strong");
    const meta = document.createElement("span");
    title.textContent = "No active events";
    meta.textContent = "The current frame has no actionable driver alert";
    empty.append(title, meta);
    els.eventList.append(empty);
    return;
  }
  events.slice(0, 10).forEach((event, index) => {
    const row = document.createElement("div");
    row.className = `event ${event.severity || "normal"} ${index === 0 ? "featured" : ""}`;
    const title = document.createElement("strong");
    const meta = document.createElement("span");
    title.textContent = event.message || event.event || "Event";
    meta.textContent = `${titleCase(event.severity || "event")} ${event.event || ""}`;
    row.append(title, meta);
    els.eventList.append(row);
  });
}

function renderAlert(data) {
  if (persistentPullOver) {
    renderPersistentPullOverAlert();
    return;
  }
  const latest = currentEvent(data);
  const show = Boolean(latest);
  els.alertBanner.classList.remove("critical");
  els.alertBanner.classList.toggle("hidden", !show);
  els.alertBanner.textContent = show ? latest.message || latest.event : "";
}

function renderPersistentPullOverAlert() {
  if (!persistentPullOver) return;
  const remaining = Math.max(0, Math.ceil(PULL_OVER_HOLD_SECONDS - (Date.now() / 1000 - persistentPullOver.startedAt)));
  els.alertBanner.classList.remove("hidden");
  els.alertBanner.classList.add("critical");
  els.alertBanner.innerHTML = "";

  const copy = document.createElement("span");
  const message = document.createElement("strong");
  message.textContent = persistentPullOver.event.message || "PULL OVER immediately. Stop at the nearest safe place.";
  const hint = document.createElement("small");
  hint.textContent = remaining > 0
    ? `Auto-clear check in ${remaining}s if the condition has cleared.`
    : "Will auto-clear when the critical condition is no longer active.";
  copy.append(message, hint);

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "Cancel";
  button.addEventListener("click", dismissPullOver);

  els.alertBanner.append(copy, button);
}

function currentEvent(data) {
  return activeEvents(data)[0] || null;
}

function activeEvents(data) {
  const containers = [data.mode_state?.driver, data.mode_state?.care, data.mode_state?.healthcare];
  const events = containers.flatMap((container) => Array.isArray(container?.events) ? container.events : []);
  const liveEvents = events
    .filter((event) => !eventContradictsCurrentFrame(event, data))
    .sort((a, b) => (SEVERITY_ORDER[b.severity] || 0) - (SEVERITY_ORDER[a.severity] || 0));
  if (!persistentPullOver) return liveEvents;
  const duplicate = liveEvents.some((event) => pullOverKey(event) === persistentPullOver.key);
  return duplicate ? liveEvents : [persistentPullOver.event, ...liveEvents];
}

function eventContradictsCurrentFrame(event, data) {
  if (!event) return true;
  if (event.event === "face_absent" && data.quality?.face_present) return true;
  if (event.event === "poor_signal" && data.quality?.signal_ok) return true;
  if (event.event === "sustained_eye_closure" && data.states?.attention !== "eyes_closed" && data.states?.fatigue !== "microsleep") {
    return true;
  }
  return false;
}

function updatePersistentPullOver(data) {
  const now = Date.now() / 1000;
  const pullOverEvent = activeLivePullOverEvent(data);
  if (persistentPullOver) {
    persistentPullOver.conditionActive = pullOverConditionStillActive(persistentPullOver.event, data);
    if (now - persistentPullOver.startedAt >= PULL_OVER_HOLD_SECONDS && !persistentPullOver.conditionActive) {
      persistentPullOver = null;
      return;
    }
  }
  if (!pullOverEvent) return;
  const key = pullOverKey(pullOverEvent);
  if (dismissedPullOverKeys.has(key)) return;
  if (persistentPullOver && persistentPullOver.key === key) {
    persistentPullOver.conditionActive = true;
    return;
  }
  persistentPullOver = {
    key,
    event: pullOverEvent,
    startedAt: now,
    conditionActive: true,
  };
}

function activeLivePullOverEvent(data) {
  const containers = [data.mode_state?.driver, data.mode_state?.care, data.mode_state?.healthcare];
  const events = containers.flatMap((container) => Array.isArray(container?.events) ? container.events : []);
  return events
    .filter((event) => isPullOverEvent(event))
    .filter((event) => !eventContradictsCurrentFrame(event, data))
    .sort((a, b) => (SEVERITY_ORDER[b.severity] || 0) - (SEVERITY_ORDER[a.severity] || 0))[0] || null;
}

function isPullOverEvent(event) {
  return event?.severity === "critical" && PULL_OVER_EVENTS.has(event.event);
}

function pullOverKey(event) {
  return `${event?.event || "critical"}:${event?.t_epoch || ""}`;
}

function pullOverConditionStillActive(event, data) {
  if (activeLivePullOverEvent(data)) return true;
  const risk = String(data.summary?.state || "normal");
  if (risk === "critical") return true;
  if (["sustained_eye_closure", "microsleep_detected", "perclos_critical"].includes(event?.event)) {
    return data.states?.fatigue === "microsleep" || data.states?.attention === "eyes_closed";
  }
  return false;
}

function dismissPullOver() {
  if (!persistentPullOver) return;
  dismissedPullOverKeys.add(persistentPullOver.key);
  persistentPullOver = null;
  renderAlert({ mode_state: {}, timeline: [] });
}

function updatePersistentPullOverCountdown() {
  if (!persistentPullOver || els.alertBanner.classList.contains("hidden")) return;
  renderPersistentPullOverAlert();
}

function formatValue(value) {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(1);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (value === null || value === undefined || value === "") return "None";
  return String(value);
}

function formatEventTime(epoch) {
  if (!epoch) return "";
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function titleCase(value) {
  return String(value || "").replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function updateSystemTime() {
  if (!els.systemTime) return;
  els.systemTime.textContent = new Date().toLocaleString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function syncModeControls() {
  const healthcare = els.mode.value === "Healthcare observation";
  els.healthcareFields.classList.toggle("hidden", !healthcare);
  els.careContext.disabled = els.mode.value !== "Care observation";
}
