import csv
import io
import logging
import math
import time
import wave
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from configs.settings import load_config
from core.audit import audit_log, write_healthcare_report
from core.pipeline import WellnessPipeline
from core.session_utils import append_event_feedback, calibration_gate_status, compact_summary, remove_old_session_files
from core.tracking import Tracker
from modules.care_events import CareMonitor
from modules.driver_events import DriverMonitor
from modules.healthcare_events import HealthcareMonitor
from ui.overlays import draw_labels
from ui.record import VideoRecorder, compose_dashboard_frame


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("human_signal_ai")

st.set_page_config(page_title="Human Signal AI - Wellness Monitor", layout="wide")
st.title("Human Signal AI - Wellness & Ergonomics Monitor")
st.caption("Wellness insights only. This is not a medical device and does not diagnose health conditions.")


@st.cache_resource
def get_tracker():
    return Tracker()


def open_camera(preferred, fallbacks, fps, width, height):
    candidates = []
    for src in [preferred, *fallbacks]:
        if src not in candidates:
            candidates.append(src)
    for src in candidates:
        cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = cap.read()
        if ok:
            logger.info("Opened camera source %s", src)
            return cap, src
        cap.release()
        logger.warning("Camera source %s failed", src)
    return None, None


def reset_modules():
    from modules import attention, fatigue, posture, stress

    for mod in (attention, fatigue, posture, stress):
        if hasattr(mod, "reset"):
            mod.reset()


def plot_series(container, ys, title, color, target_fps):
    xs = np.arange(len(ys)) / max(1, target_fps)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3)))
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10), title=title, yaxis=dict(range=[0, 100]))
    container.plotly_chart(fig, use_container_width=True)


def alert_labels(attention, fatigue, tension):
    labels = []
    if attention["state"] in {"looking_away", "head_turned", "face_absent"}:
        labels.append("AWAY")
    if fatigue["state"] in {"drowsy", "microsleep", "fatigue_signs"}:
        labels.append("FATIGUE SIGNS")
    if tension["state"] == "elevated_tension":
        labels.append("FACIAL TENSION")
    return labels


def driver_alert_labels(driver_state):
    risk = driver_state.get("risk", "normal")
    if risk == "critical":
        return ["PULL OVER SAFELY"]
    if risk == "elevated":
        return ["FATIGUE RISK"]
    if risk == "watch":
        return ["WATCH FATIGUE"]
    if risk == "insufficient_signal":
        return ["CHECK CAMERA"]
    return []


def care_alert_labels(care_state):
    state = care_state.get("state", "observed")
    if state == "urgent_review":
        return ["URGENT REVIEW"]
    if state == "needs_review":
        return ["CARE REVIEW"]
    if state == "insufficient_signal":
        return ["CHECK CAMERA"]
    if state == "resting":
        return ["RESTING"]
    return []


def healthcare_alert_labels(healthcare_state):
    state = healthcare_state.get("state", "stable_observation")
    if state == "urgent_review":
        return ["URGENT REVIEW"]
    if state == "needs_review":
        return ["CLINICIAN REVIEW"]
    if state == "insufficient_signal":
        return ["CHECK SIGNAL"]
    if state == "resting":
        return ["RESTING"]
    return []


def format_duration(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def make_alert_tone(duration=0.35, freq=880, sample_rate=16000):
    n_samples = int(duration * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(n_samples):
            amp = int(16000 * math.sin(2 * math.pi * freq * (i / sample_rate)))
            wav.writeframesraw(amp.to_bytes(2, byteorder="little", signed=True))
    return buf.getvalue()


def draw_alert_banner(img_bgr, labels):
    if not labels:
        return img_bgr
    h, w = img_bgr.shape[:2]
    text = "  |  ".join(labels)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 24, 14
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    x0, y0 = max(0, (w - box_w) // 2), 36
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)
    cv2.putText(img_bgr, text, (x0 + pad_x, y0 + pad_y + th), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img_bgr


def draw_keypoint_overlay(img_bgr, face_landmarks):
    if face_landmarks is None or len(face_landmarks) < 478:
        return img_bgr
    pts = face_landmarks[:, :2].astype(np.int32)
    idxs = {"left_eye": 33, "right_eye": 263, "nose_tip": 1, "chin": 152, "forehead": 10, "mouth_left": 61, "mouth_right": 291}
    overlay = img_bgr.copy()
    for a, b in [("left_eye", "right_eye"), ("nose_tip", "chin"), ("forehead", "nose_tip"), ("mouth_left", "mouth_right")]:
        cv2.line(overlay, tuple(pts[idxs[a]]), tuple(pts[idxs[b]]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img_bgr, 0.55, 0, img_bgr)
    return img_bgr


def privacy_filter(img_bgr, enabled):
    if not enabled:
        return img_bgr
    blurred = cv2.GaussianBlur(img_bgr, (35, 35), 0)
    cv2.putText(blurred, "Privacy preview", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return blurred


def write_dataset_row(path, label, result):
    feats = result["features"]
    quality = result["quality"]
    sm = result["smoothed"]
    row = {
        "t_epoch": time.time(),
        "label": label,
        "fatigue": f"{sm['fatigue']:.2f}",
        "attention": f"{sm['attention']:.2f}",
        "facial_tension": f"{sm['tension']:.2f}",
        "fatigue_state": result["fatigue"]["state"],
        "attention_state": result["attention"]["state"],
        "tension_state": result["tension"]["state"],
        "posture": result["posture"]["state"],
        "distance": result["distance"]["state"],
        "brightness": f"{quality['brightness']:.2f}",
        "sharpness": f"{quality['sharpness']:.2f}",
        "face_present": quality["face_present"],
        "signal_ok": quality["signal_ok"],
        "ear_left": feats.get("ear_left"),
        "ear_right": feats.get("ear_right"),
        "mar": feats.get("mar"),
        "gaze_proxy": feats.get("gaze"),
        "head_yaw": feats.get("head_yaw"),
        "neck_angle": feats.get("neck_angle"),
    }
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def write_driver_events(path, events, summary):
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_epoch",
        "event",
        "severity",
        "message",
        "fatigue_state",
        "attention_state",
        "signal_ok",
        "fatigue_score",
        "attention_score",
        "perclos",
        "trip_secs",
        "since_break_secs",
        "usable_signal_pct",
    ]
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for event in events:
            row = {k: event.get(k) for k in fieldnames if k in event}
            row["trip_secs"] = f"{summary['trip_secs']:.1f}"
            row["since_break_secs"] = f"{summary['since_break_secs']:.1f}"
            row["usable_signal_pct"] = f"{summary['usable_signal_pct']:.1f}"
            writer.writerow(row)


def write_care_events(path, events, summary):
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_epoch",
        "event",
        "state",
        "message",
        "context",
        "posture_state",
        "attention_state",
        "fatigue_state",
        "signal_ok",
        "visible",
        "session_secs",
        "usable_signal_pct",
        "visible_pct",
    ]
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for event in events:
            row = {k: event.get(k) for k in fieldnames if k in event}
            row["session_secs"] = f"{summary['session_secs']:.1f}"
            row["usable_signal_pct"] = f"{summary['usable_signal_pct']:.1f}"
            row["visible_pct"] = f"{summary['visible_pct']:.1f}"
            writer.writerow(row)


def write_healthcare_events(path, events, summary):
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t_epoch",
        "event",
        "state",
        "message",
        "observation_type",
        "patient_session_id",
        "fatigue_state",
        "attention_state",
        "posture_state",
        "signal_ok",
        "visible",
        "note",
        "session_secs",
        "usable_signal_pct",
        "visible_pct",
    ]
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        for event in events:
            row = {k: event.get(k) for k in fieldnames if k in event}
            row["session_secs"] = f"{summary['session_secs']:.1f}"
            row["usable_signal_pct"] = f"{summary['usable_signal_pct']:.1f}"
            row["visible_pct"] = f"{summary['visible_pct']:.1f}"
            writer.writerow(row)


try:
    cfg_model = load_config()
    cfg = cfg_model.as_legacy_dict()
except Exception as exc:
    st.error(f"Configuration error: {exc}")
    logger.exception("Configuration failed")
    st.stop()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = WellnessPipeline(cfg)
if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None
if "motion_ema" not in st.session_state:
    st.session_state.motion_ema = 0.0
if "calibrating" not in st.session_state:
    st.session_state.calibrating = False
if "cal_start" not in st.session_state:
    st.session_state.cal_start = 0.0
if "recording" not in st.session_state:
    st.session_state.recording = False
if "recorder" not in st.session_state:
    st.session_state.recorder = None
if "driver_monitor" not in st.session_state:
    st.session_state.driver_monitor = DriverMonitor()
if "driver_trip_active" not in st.session_state:
    st.session_state.driver_trip_active = False
if "driver_events_path" not in st.session_state:
    st.session_state.driver_events_path = None
if "last_driver_sound" not in st.session_state:
    st.session_state.last_driver_sound = 0.0
if "care_monitor" not in st.session_state:
    st.session_state.care_monitor = CareMonitor()
if "care_session_active" not in st.session_state:
    st.session_state.care_session_active = False
if "care_events_path" not in st.session_state:
    st.session_state.care_events_path = None
if "healthcare_monitor" not in st.session_state:
    st.session_state.healthcare_monitor = HealthcareMonitor()
if "healthcare_session_active" not in st.session_state:
    st.session_state.healthcare_session_active = False
if "healthcare_events_path" not in st.session_state:
    st.session_state.healthcare_events_path = None
if "healthcare_report_path" not in st.session_state:
    st.session_state.healthcare_report_path = None
if "event_timeline" not in st.session_state:
    st.session_state.event_timeline = []
if "feedback_path" not in st.session_state:
    st.session_state.feedback_path = Path("recordings") / f"event_feedback_{time.strftime('%Y%m%d_%H%M%S')}.csv"
if "latest_calibration_gate" not in st.session_state:
    st.session_state.latest_calibration_gate = {"ok": False, "checks": {}}
if "generic_event_last_t" not in st.session_state:
    st.session_state.generic_event_last_t = {}

with st.sidebar:
    st.header("Run")
    run_monitor = st.toggle("Run monitor", value=True)
    app_mode = st.selectbox("Mode", ["Driver", "Desk ergonomics", "Care observation", "Healthcare observation"], index=0)
    cam_index = st.number_input("Camera index", value=int(cfg["video"]["source"]), step=1)
    target_fps = st.slider("Target FPS", 5, 60, int(cfg["video"]["fps"]))
    show_overlay = st.toggle("Keypoint overlay", value=True)
    privacy_preview = st.toggle("Privacy preview blur", value=False)
    metadata_only = st.toggle("Metadata-only exports", value=True)
    driver_sound = st.toggle("Driver alert sound", value=False, disabled=app_mode != "Driver")
    care_context = "Chair"
    if app_mode == "Care observation":
        care_context = st.selectbox("Care context", ["Chair", "Bed", "Room", "Desk"], index=0)
    healthcare_observation_type = "General observation"
    healthcare_patient_id = ""
    healthcare_location = ""
    healthcare_observer = ""
    healthcare_reason = ""
    healthcare_note = ""
    healthcare_consent = False
    healthcare_recording_consent = False
    if app_mode == "Healthcare observation":
        healthcare_observation_type = st.selectbox(
            "Observation type",
            ["General observation", "Post-procedure recovery", "Sleep/rest observation", "Rehab/session monitoring", "Waiting room / triage support"],
            index=0,
        )
        healthcare_patient_id = st.text_input("Patient/session ID", value="")
        healthcare_location = st.text_input("Location/room", value="")
        healthcare_observer = st.text_input("Observer/staff initials", value="")
        healthcare_reason = st.text_input("Observation reason", value="")
        healthcare_note = st.text_input("Event note/marker", value="")
        healthcare_consent = st.checkbox("Observation consent captured")
        healthcare_recording_consent = st.checkbox("Raw video recording consent captured")
    st.markdown("---")
    calibration_seconds = float(cfg.get("calibration", {}).get("seconds", 10.0))
    if st.button(f"Calibrate {int(calibration_seconds)}s"):
        if st.session_state.latest_calibration_gate.get("ok"):
            reset_modules()
            st.session_state.pipeline = WellnessPipeline(cfg)
            st.session_state.prev_gray = None
            st.session_state.motion_ema = 0.0
            st.session_state.calibrating = True
            st.session_state.cal_start = time.time()
            audit_log("calibration_started", {"mode": app_mode})
        else:
            st.warning("Calibration gate is not ready. Fix camera setup first.")
    st.caption("For calibration: sit upright, face the camera, keep a neutral expression, and use steady lighting.")

    if app_mode == "Driver":
        st.markdown("---")
        st.subheader("Driver trip")
        if not st.session_state.driver_trip_active:
            if st.button("Start trip"):
                now = time.time()
                st.session_state.driver_monitor.reset(now)
                st.session_state.driver_trip_active = True
                event_dir = Path(cfg["recording"]["output_dir"])
                event_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.driver_events_path = event_dir / f"driver_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            if st.button("Mark break"):
                st.session_state.driver_monitor.mark_break(time.time())
            if st.button("End trip"):
                st.session_state.driver_trip_active = False
        if st.session_state.driver_events_path:
            st.caption(f"Event export: {st.session_state.driver_events_path}")

    if app_mode == "Care observation":
        st.markdown("---")
        st.subheader("Care session")
        st.caption("Metadata-only by default. This mode supports caregiver review, not diagnosis.")
        if not st.session_state.care_session_active:
            if st.button("Start care session"):
                now = time.time()
                st.session_state.care_monitor.reset(now)
                st.session_state.care_session_active = True
                event_dir = Path(cfg["recording"]["output_dir"])
                event_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.care_events_path = event_dir / f"care_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            if st.button("End care session"):
                st.session_state.care_session_active = False
        if st.session_state.care_events_path:
            st.caption(f"Event export: {st.session_state.care_events_path}")

    if app_mode == "Healthcare observation":
        st.markdown("---")
        st.subheader("Healthcare session")
        st.caption("Structured observation support only. Not diagnostic.")
        can_start_healthcare = healthcare_consent and bool(healthcare_patient_id.strip())
        if not st.session_state.healthcare_session_active:
            if st.button("Start healthcare session", disabled=not can_start_healthcare):
                now = time.time()
                st.session_state.healthcare_monitor.reset(now)
                st.session_state.healthcare_session_active = True
                event_dir = Path(cfg["recording"]["output_dir"])
                event_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.healthcare_events_path = event_dir / f"healthcare_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                st.session_state.healthcare_report_path = None
                audit_log("healthcare_session_started", {"patient_session_id": healthcare_patient_id, "observation_type": healthcare_observation_type})
        else:
            if st.button("End healthcare session"):
                st.session_state.healthcare_session_active = False
                audit_log("healthcare_session_ended", {"patient_session_id": healthcare_patient_id})
        if not can_start_healthcare and not st.session_state.healthcare_session_active:
            st.caption("Patient/session ID and consent are required to start.")
        if st.session_state.healthcare_events_path:
            st.caption(f"Event export: {st.session_state.healthcare_events_path}")

    st.markdown("---")
    record_dir = Path(cfg["recording"]["output_dir"])
    if not st.session_state.recording:
        if metadata_only:
            st.caption("Metadata-only mode is on; raw video recording is disabled.")
        if st.button("Record 30s", disabled=metadata_only):
            record_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            st.session_state.rec_path = record_dir / f"session_{ts}.mp4"
            st.session_state.csv_path = record_dir / f"session_{ts}.csv"
            rec_fps = max(cfg["recording"]["fps_min"], min(cfg["recording"]["fps_max"], cfg["windows"]["update_hz"] * 2))
            st.session_state.rec_fps = rec_fps
            st.session_state.recorder = VideoRecorder(str(st.session_state.rec_path), fps=rec_fps, size=(1280, 900))
            st.session_state.rec_end_time = time.time() + 30.0
            st.session_state.last_write_t = 0.0
            with st.session_state.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(["t_epoch", "fatigue", "attention", "facial_tension", "posture", "distance", "signal_ok"])
            audit_log("recording_started", {"path": str(st.session_state.rec_path), "mode": app_mode, "recording_consent": healthcare_recording_consent if app_mode == "Healthcare observation" else None})
            st.session_state.recording = True
    else:
        st.write(f"Recording: {max(0, int(st.session_state.rec_end_time - time.time()))}s left")
        if st.button("Stop recording"):
            st.session_state.rec_end_time = time.time()
            audit_log("recording_stopped", {"mode": app_mode})

    st.markdown("---")
    st.subheader("Privacy & retention")
    retention_days = st.number_input("Retention days", min_value=1, max_value=365, value=30, step=1)
    if st.button("Delete expired local exports"):
        removed = remove_old_session_files([cfg["recording"]["output_dir"], cfg["dataset"]["output_dir"]], int(retention_days))
        audit_log("retention_cleanup", {"retention_days": int(retention_days), "removed_count": len(removed)})
        st.success(f"Deleted {len(removed)} expired file(s).")

    st.markdown("---")
    st.subheader("Event feedback")
    feedback_choice = st.selectbox("Latest event feedback", ["None", "Acknowledge", "False alarm", "Expected/resting", "Needs follow-up"])
    if st.button("Save feedback"):
        if st.session_state.event_timeline and feedback_choice != "None":
            latest_event = st.session_state.event_timeline[-1]
            feedback_value = feedback_choice.lower().replace("/", "_").replace(" ", "_")
            append_event_feedback(st.session_state.feedback_path, {"feedback": feedback_value, **latest_event})
            audit_log("event_feedback", {"feedback": feedback_value, "event": latest_event.get("event")})
            st.success("Feedback saved.")
        else:
            st.warning("No event is available for feedback yet.")

    st.markdown("---")
    consent = st.checkbox("I have consent to capture this dataset row stream")
    capture_dataset = st.toggle("Dataset capture", value=False, disabled=not consent)
    dataset_label = st.text_input("Dataset label", value="unlabeled")

if not run_monitor:
    st.info("Monitor is stopped.")
    st.stop()

cap, active_source = open_camera(
    int(cam_index),
    cfg["video"].get("fallback_sources", [0, 1, 2]),
    target_fps,
    cfg["video"]["width"],
    cfg["video"]["height"],
)
if cap is None:
    st.error("No camera source could be opened. Try a different camera index or check camera permissions.")
    st.stop()

st.caption(f"Using camera source: {active_source}")
tracker = get_tracker()

sec_window = 60
buf_len = sec_window * max(1, target_fps)
fatigue_buf, attention_buf, tension_buf = (deque(maxlen=buf_len) for _ in range(3))

video_placeholder = st.empty()
calibration_placeholder = st.empty()
driver_alert_placeholder = st.empty()
c1, c2, c3 = st.columns(3)
chart1, chart2, chart3 = c1.empty(), c2.empty(), c3.empty()
quality_placeholder = st.empty()
status_placeholder = st.empty()
driver_summary_placeholder = st.empty()
driver_events_placeholder = st.empty()
care_alert_placeholder = st.empty()
care_summary_placeholder = st.empty()
care_events_placeholder = st.empty()
healthcare_alert_placeholder = st.empty()
healthcare_summary_placeholder = st.empty()
healthcare_events_placeholder = st.empty()
healthcare_report_placeholder = st.empty()
timeline_placeholder = st.empty()
feedback_placeholder = st.empty()
summary_dashboard_placeholder = st.empty()
sound_placeholder = st.empty()

update_hz = int(cfg["windows"]["update_hz"])
chart_every = max(1, int(target_fps / update_hz))
frame_count = 0
t_last = time.time()

dataset_path = Path(cfg["dataset"]["output_dir"]) / f"dataset_{time.strftime('%Y%m%d')}.csv"
if capture_dataset:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

try:
    while run_monitor:
        ok, frame = cap.read()
        if not ok:
            logger.error("Camera read failed for source %s", active_source)
            st.error("Camera read failed. Try another camera index.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = tracker.process(frame_rgb)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if st.session_state.prev_gray is not None:
            diff = cv2.absdiff(gray, st.session_state.prev_gray)
            motion_energy = float(diff.mean()) / 255.0
            st.session_state.motion_ema = 0.2 * motion_energy + 0.8 * st.session_state.motion_ema
        else:
            st.session_state.motion_ema = 0.0
        st.session_state.prev_gray = gray

        calibrating = st.session_state.calibrating and (time.time() - st.session_state.cal_start <= float(cfg.get("calibration", {}).get("seconds", 10.0)))
        if st.session_state.calibrating and not calibrating:
            st.session_state.calibrating = False

        result = st.session_state.pipeline.process(
            frame_rgb,
            det,
            calibrating=calibrating,
            motion_energy=st.session_state.motion_ema,
        )
        sm = result["smoothed"]
        st.session_state.latest_calibration_gate = calibration_gate_status(result["quality"], cfg)
        fatigue_buf.append(sm["fatigue"])
        attention_buf.append(sm["attention"])
        tension_buf.append(sm["tension"])

        driver_state = {"risk": "normal", "events": [], "summary": {}}
        care_state = {"state": "observed", "events": [], "summary": {}}
        healthcare_state = {"state": "stable_observation", "events": [], "summary": {}}
        if app_mode == "Driver":
            if not st.session_state.driver_trip_active:
                st.session_state.driver_monitor.reset(time.time())
                st.session_state.driver_trip_active = True
                event_dir = Path(cfg["recording"]["output_dir"])
                event_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.driver_events_path = event_dir / f"driver_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            driver_state = st.session_state.driver_monitor.update(result, time.time())
            if st.session_state.driver_events_path:
                write_driver_events(st.session_state.driver_events_path, driver_state["events"], driver_state["summary"])
            for event in driver_state["events"]:
                st.session_state.event_timeline.append({"mode": app_mode, **event})
        elif app_mode == "Care observation":
            if not st.session_state.care_session_active:
                st.session_state.care_monitor.reset(time.time())
                st.session_state.care_session_active = True
                event_dir = Path(cfg["recording"]["output_dir"])
                event_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.care_events_path = event_dir / f"care_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            care_state = st.session_state.care_monitor.update(
                result,
                time.time(),
                context=care_context,
                motion_energy=st.session_state.motion_ema,
            )
            if st.session_state.care_events_path:
                write_care_events(st.session_state.care_events_path, care_state["events"], care_state["summary"])
            for event in care_state["events"]:
                st.session_state.event_timeline.append({"mode": app_mode, **event})
        elif app_mode == "Healthcare observation":
            calibration_ok = (
                result["quality"].get("face_present")
                and result["quality"].get("brightness", 0.0) >= cfg["quality"]["min_brightness"]
                and result["quality"].get("sharpness", 0.0) >= cfg["quality"].get("min_sharpness", 35.0)
            )
            if not st.session_state.healthcare_session_active and healthcare_consent and healthcare_patient_id.strip():
                st.session_state.healthcare_monitor.reset(time.time())
                st.session_state.healthcare_session_active = True
                event_dir = Path(cfg["recording"]["output_dir"])
                event_dir.mkdir(parents=True, exist_ok=True)
                st.session_state.healthcare_events_path = event_dir / f"healthcare_events_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                audit_log("healthcare_session_started", {"patient_session_id": healthcare_patient_id, "observation_type": healthcare_observation_type, "auto_started": True})
            if st.session_state.healthcare_session_active:
                result["motion_energy"] = st.session_state.motion_ema
                healthcare_state = st.session_state.healthcare_monitor.update(
                    result,
                    time.time(),
                    observation_type=healthcare_observation_type,
                    patient_session_id=healthcare_patient_id,
                    note=healthcare_note,
                    calibration_ok=bool(calibration_ok),
                )
                if st.session_state.healthcare_events_path:
                    write_healthcare_events(st.session_state.healthcare_events_path, healthcare_state["events"], healthcare_state["summary"])
                for event in healthcare_state["events"]:
                    st.session_state.event_timeline.append({"mode": app_mode, **event})
        elif app_mode == "Desk ergonomics":
            generic_events = []
            now_event = time.time()
            if not result["quality"].get("signal_ok") and now_event - st.session_state.generic_event_last_t.get("poor_signal", 0.0) >= 30.0:
                generic_events.append({
                    "t_epoch": now_event,
                    "event": "poor_signal",
                    "state": "insufficient_signal",
                    "severity": "insufficient_signal",
                    "message": "Camera signal is unreliable.",
                })
                st.session_state.generic_event_last_t["poor_signal"] = now_event
            if result["posture"]["state"] == "slouching" and now_event - st.session_state.generic_event_last_t.get("posture_decline", 0.0) >= 60.0:
                generic_events.append({
                    "t_epoch": now_event,
                    "event": "posture_decline",
                    "state": "ergonomic_warning",
                    "severity": "watch",
                    "message": "Slouched posture observed.",
                })
                st.session_state.generic_event_last_t["posture_decline"] = now_event
            for event in generic_events:
                st.session_state.event_timeline.append({"mode": app_mode, **event})

        labeled = draw_labels(frame, result["posture"], result["distance"])
        if show_overlay:
            labeled = draw_keypoint_overlay(labeled, result["features"].get("face"))
        if app_mode == "Driver":
            labels = driver_alert_labels(driver_state)
        elif app_mode == "Care observation":
            labels = care_alert_labels(care_state)
        elif app_mode == "Healthcare observation":
            labels = healthcare_alert_labels(healthcare_state)
        else:
            labels = alert_labels(result["attention"], result["fatigue"], result["tension"])
        labeled = draw_alert_banner(labeled, labels)
        preview_frame = privacy_filter(labeled, privacy_preview)
        video_placeholder.image(preview_frame[:, :, ::-1], channels="RGB", use_container_width=True)

        if app_mode == "Driver" and driver_state["risk"] in {"elevated", "critical", "insufficient_signal"}:
            messages = [event["message"] for event in driver_state["events"]]
            msg = messages[-1] if messages else {
                "elevated": "Fatigue risk is elevated. Plan a safe break.",
                "critical": "Critical fatigue signal. Pull over safely and rest.",
                "insufficient_signal": "Camera signal is unreliable.",
            }[driver_state["risk"]]
            if driver_state["risk"] == "critical":
                driver_alert_placeholder.error(msg)
            elif driver_state["risk"] == "elevated":
                driver_alert_placeholder.warning(msg)
            else:
                driver_alert_placeholder.warning(msg)
            if driver_sound and driver_state["risk"] == "critical" and time.time() - st.session_state.last_driver_sound > 8.0:
                sound_placeholder.audio(make_alert_tone(), format="audio/wav", autoplay=True)
                st.session_state.last_driver_sound = time.time()
        elif app_mode == "Driver":
            driver_alert_placeholder.success("Driver risk: normal")

        if app_mode == "Care observation":
            state = care_state["state"]
            messages = [event["message"] for event in care_state["events"]]
            msg = messages[-1] if messages else {
                "observed": "Care state: observed",
                "resting": "Care state: resting",
                "needs_review": "Caregiver review recommended.",
                "urgent_review": "Urgent caregiver review recommended.",
                "insufficient_signal": "Observation signal is unreliable.",
            }[state]
            if state == "urgent_review":
                care_alert_placeholder.error(msg)
            elif state in {"needs_review", "insufficient_signal"}:
                care_alert_placeholder.warning(msg)
            else:
                care_alert_placeholder.success(msg)

        if app_mode == "Healthcare observation":
            state = healthcare_state["state"]
            messages = [event["message"] for event in healthcare_state["events"]]
            msg = messages[-1] if messages else {
                "stable_observation": "Healthcare state: stable observation",
                "resting": "Healthcare state: resting",
                "needs_review": "Clinician review recommended.",
                "urgent_review": "Urgent clinician review recommended.",
                "insufficient_signal": "Observation signal is unreliable.",
            }[state]
            if state == "urgent_review":
                healthcare_alert_placeholder.error(msg)
            elif state in {"needs_review", "insufficient_signal"}:
                healthcare_alert_placeholder.warning(msg)
            else:
                healthcare_alert_placeholder.success(msg)

        if capture_dataset and consent:
            write_dataset_row(dataset_path, dataset_label, result)

        if st.session_state.recording:
            rec_source = privacy_filter(labeled, metadata_only or privacy_preview)
            rec_frame = compose_dashboard_frame(rec_source, list(fatigue_buf), list(attention_buf), list(tension_buf))
            now = time.time()
            frame_interval = 1.0 / st.session_state.rec_fps
            if st.session_state.last_write_t == 0.0:
                st.session_state.last_write_t = now
            while st.session_state.last_write_t + frame_interval <= now:
                if st.session_state.recorder and st.session_state.recorder.is_open:
                    st.session_state.recorder.write(rec_frame)
                st.session_state.last_write_t += frame_interval
            with st.session_state.csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    now,
                    f"{sm['fatigue']:.2f}",
                    f"{sm['attention']:.2f}",
                    f"{sm['tension']:.2f}",
                    result["posture"]["state"],
                    result["distance"]["state"],
                    result["quality"]["signal_ok"],
                ])
            if now >= st.session_state.rec_end_time:
                st.session_state.recording = False
                st.session_state.last_write_t = 0.0
                if st.session_state.recorder:
                    st.session_state.recorder.release()
                st.success(f"Saved video: {st.session_state.rec_path}")
                st.info(f"Saved metrics: {st.session_state.csv_path}")

        frame_count += 1
        if frame_count % chart_every == 0:
            plot_series(chart1, list(fatigue_buf), "Fatigue", "#1f77b4", target_fps)
            plot_series(chart2, list(attention_buf), "Attention proxy", "#2ca02c", target_fps)
            plot_series(chart3, list(tension_buf), "Facial tension proxy", "#d62728", target_fps)

            q = result["quality"]
            if q["signal_ok"]:
                quality_placeholder.success("Signal quality OK")
            else:
                quality_placeholder.warning("Insufficient signal: " + ", ".join(q["reasons"]))

            gate = st.session_state.latest_calibration_gate
            gate_msg = " | ".join(f"{name}: {'OK' if ok else 'fix'}" for name, ok in gate["checks"].items())
            if gate["ok"]:
                calibration_placeholder.success("Calibration gate ready: " + gate_msg)
            else:
                calibration_placeholder.warning("Calibration gate not ready: " + gate_msg)

            status_placeholder.info(
                f"{'Driver risk: ' + driver_state['risk'] + ' | ' if app_mode == 'Driver' else 'Care state: ' + care_state['state'] + ' | ' if app_mode == 'Care observation' else 'Healthcare state: ' + healthcare_state['state'] + ' | ' if app_mode == 'Healthcare observation' else 'Readiness: ' + format(result['fused']['readiness'], '0.0f') + ' | '}"
                f"Fatigue: {sm['fatigue']:0.0f} ({result['fatigue']['state']}, conf {result['fatigue']['conf']:.2f}) | "
                f"Attention: {sm['attention']:0.0f} ({result['attention']['state']}, conf {result['attention']['conf']:.2f}) | "
                f"Facial tension: {sm['tension']:0.0f} ({result['tension']['state']}, conf {result['tension']['conf']:.2f}) | "
                f"Posture: {result['posture']['state']} | Distance: {result['distance']['state']} | "
                f"Brightness: {q['brightness']:.0f} | Sharpness: {q['sharpness']:.0f} | Drops: {q['frame_drops']}"
            )

            if app_mode == "Driver":
                summary = driver_state["summary"]
                cols = driver_summary_placeholder.columns(6)
                cols[0].metric("Trip", format_duration(summary.get("trip_secs", 0.0)))
                cols[1].metric("Since break", format_duration(summary.get("since_break_secs", 0.0)))
                cols[2].metric("Usable signal", f"{summary.get('usable_signal_pct', 0.0):.0f}%")
                cols[3].metric("Microsleeps", int(summary.get("microsleeps", 0)))
                cols[4].metric("Away time", format_duration(summary.get("away_secs", 0.0)))
                cols[5].metric("Critical alerts", int(summary.get("critical_alerts", 0)))
                if driver_state["events"]:
                    latest = driver_state["events"][-5:]
                    driver_events_placeholder.dataframe(
                        [
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(e["t_epoch"])),
                                "event": e["event"],
                                "severity": e["severity"],
                                "message": e["message"],
                            }
                            for e in latest
                        ],
                        use_container_width=True,
                    )
            elif app_mode == "Care observation":
                summary = care_state["summary"]
                cols = care_summary_placeholder.columns(6)
                cols[0].metric("Session", format_duration(summary.get("session_secs", 0.0)))
                cols[1].metric("Visible", f"{summary.get('visible_pct', 0.0):.0f}%")
                cols[2].metric("Usable signal", f"{summary.get('usable_signal_pct', 0.0):.0f}%")
                cols[3].metric("Longest absence", format_duration(summary.get("longest_absence", 0.0)))
                cols[4].metric("Review events", int(summary.get("needs_review_events", 0)))
                cols[5].metric("Urgent events", int(summary.get("urgent_review_events", 0)))
                if care_state["events"]:
                    latest = care_state["events"][-5:]
                    care_events_placeholder.dataframe(
                        [
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(e["t_epoch"])),
                                "event": e["event"],
                                "state": e["state"],
                                "message": e["message"],
                            }
                            for e in latest
                        ],
                        use_container_width=True,
                    )
            elif app_mode == "Healthcare observation":
                summary = healthcare_state["summary"]
                cols = healthcare_summary_placeholder.columns(6)
                cols[0].metric("Session", format_duration(summary.get("session_secs", 0.0)))
                cols[1].metric("Visible", f"{summary.get('visible_pct', 0.0):.0f}%")
                cols[2].metric("Usable signal", f"{summary.get('usable_signal_pct', 0.0):.0f}%")
                cols[3].metric("Longest eye closure", format_duration(summary.get("longest_eye_closure", 0.0)))
                cols[4].metric("Review events", int(summary.get("needs_review_events", 0)))
                cols[5].metric("Urgent events", int(summary.get("urgent_review_events", 0)))
                if healthcare_state["events"]:
                    latest = healthcare_state["events"][-5:]
                    healthcare_events_placeholder.dataframe(
                        [
                            {
                                "time": time.strftime("%H:%M:%S", time.localtime(e["t_epoch"])),
                                "event": e["event"],
                                "state": e["state"],
                                "message": e["message"],
                            }
                            for e in latest
                        ],
                        use_container_width=True,
                    )
                if st.session_state.healthcare_session_active and st.button("Export healthcare report"):
                    metadata = {
                        "patient_session_id": healthcare_patient_id,
                        "location": healthcare_location,
                        "observer": healthcare_observer,
                        "observation_type": healthcare_observation_type,
                        "observation_reason": healthcare_reason,
                        "consent_captured": healthcare_consent,
                        "recording_consent_captured": healthcare_recording_consent,
                    }
                    report_dir = Path(cfg["recording"]["output_dir"])
                    report_path = report_dir / f"healthcare_report_{time.strftime('%Y%m%d_%H%M%S')}.md"
                    st.session_state.healthcare_report_path = write_healthcare_report(
                        report_path,
                        metadata,
                        summary,
                        st.session_state.healthcare_events_path,
                    )
                if st.session_state.healthcare_report_path:
                    healthcare_report_placeholder.success(f"Report exported: {st.session_state.healthcare_report_path}")

            generic_summary = {
                "mode": app_mode,
                "state": "insufficient_signal" if not q["signal_ok"] else "active",
                "duration_secs": len(fatigue_buf) / max(1, target_fps),
                "usable_signal_pct": 100.0 if q["signal_ok"] else 0.0,
                "review_events": 0,
                "visible_pct": 100.0 if q.get("face_present") else 0.0,
            }
            shared_summary = compact_summary(app_mode, driver_state, care_state, healthcare_state, generic_summary)
            s_cols = summary_dashboard_placeholder.columns(5)
            s_cols[0].metric("Mode", shared_summary["mode"])
            s_cols[1].metric("State", shared_summary["state"])
            s_cols[2].metric("Duration", format_duration(shared_summary.get("duration_secs", 0.0)))
            s_cols[3].metric("Usable signal", f"{shared_summary.get('usable_signal_pct', 0.0):.0f}%")
            s_cols[4].metric("Review events", int(shared_summary.get("review_events", 0)))

            if st.session_state.event_timeline:
                st.session_state.event_timeline = st.session_state.event_timeline[-200:]
                latest_events = st.session_state.event_timeline[-12:]
                timeline_placeholder.dataframe(
                    [
                        {
                            "time": time.strftime("%H:%M:%S", time.localtime(e["t_epoch"])),
                            "mode": e.get("mode"),
                            "event": e.get("event"),
                            "level": e.get("severity") or e.get("state"),
                            "message": e.get("message"),
                        }
                        for e in reversed(latest_events)
                    ],
                    use_container_width=True,
                )
                feedback_placeholder.caption(f"Feedback target: latest event in the timeline. Saved to {st.session_state.feedback_path}")

        target_dt = 1.0 / update_hz
        dt = time.time() - t_last
        if dt < target_dt:
            time.sleep(target_dt - dt)
        t_last = time.time()
except Exception:
    logger.exception("Runtime failure")
    st.error("The monitor stopped because of an unexpected runtime error. See logs/app.log.")
finally:
    if st.session_state.get("recorder"):
        st.session_state.recorder.release()
    cap.release()
    logger.info("Monitor stopped")
