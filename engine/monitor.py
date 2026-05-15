from __future__ import annotations

import time
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np

from configs.settings import load_config
from core.pipeline import WellnessPipeline
from core.session_utils import append_event_feedback, calibration_gate_status, compact_summary
from engine.frame_decoder import decode_data_url_image
from engine.schemas import EngineStatus, FeedbackRequest, FrameRequest, FrameResponse, SessionRequest
from engine.tracker import create_tracker
from modules.care_events import CareMonitor
from modules.driver_events import DriverMonitor
from modules.healthcare_events import HealthcareMonitor


MODES = ["Driver", "Desk ergonomics", "Care observation", "Healthcare observation"]


class MonitorEngine:
    def __init__(self) -> None:
        self.cfg = load_config().as_legacy_dict()
        self.tracker = create_tracker(self.cfg)
        self.pipeline = WellnessPipeline(self.cfg)
        self.driver_monitor = DriverMonitor()
        self.driver_monitor.thresholds = self.cfg.get("thresholds", {})
        self.driver_monitor.driver_cfg = self.cfg.get("driver", {})
        self.care_monitor = CareMonitor()
        self.healthcare_monitor = HealthcareMonitor()
        self.prev_gray = None
        self.motion_ema = 0.0
        self.calibrating_until = 0.0
        self.driver_active = False
        self.care_active = False
        self.healthcare_active = False
        self.timeline: list[dict[str, Any]] = []
        self.latest_result: dict[str, Any] | None = None
        self.feedback_path = Path("recordings") / f"event_feedback_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.lock = Lock()

    @property
    def tracker_backend(self) -> str:
        return self.tracker.name

    def status(self) -> EngineStatus:
        return EngineStatus(
            tracker_backend=self.tracker_backend,
            modes=MODES,
            sessions={
                "driver": self.driver_active,
                "care": self.care_active,
                "healthcare": self.healthcare_active,
            },
        )

    def set_session(self, req: SessionRequest) -> EngineStatus:
        now = time.time()
        with self.lock:
            active = req.action == "start"
            if active:
                self._reset_runtime_state(now)
            if req.mode == "Driver":
                if active:
                    self.driver_monitor.reset(now)
                self.driver_active = active
            elif req.mode == "Care observation":
                if active:
                    self.care_monitor.reset(now)
                self.care_active = active
            elif req.mode == "Healthcare observation":
                if active:
                    self.healthcare_monitor.reset(now)
                self.healthcare_active = active
            elif req.mode == "Desk ergonomics":
                pass
            else:
                raise ValueError("Unknown mode")
            return self.status()

    def _reset_runtime_state(self, now: float) -> None:
        reset_scoring_modules()
        self.pipeline = WellnessPipeline(self.cfg)
        self.driver_monitor.reset(now)
        self.care_monitor.reset(now)
        self.healthcare_monitor.reset(now)
        self.driver_active = False
        self.care_active = False
        self.healthcare_active = False
        self.prev_gray = None
        self.motion_ema = 0.0
        self.calibrating_until = 0.0
        self.timeline.clear()
        self.latest_result = None
        self.feedback_path = Path("recordings") / f"event_feedback_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    def calibrate(self) -> dict[str, Any]:
        with self.lock:
            reset_scoring_modules()
            self.pipeline = WellnessPipeline(self.cfg)
            self.prev_gray = None
            self.motion_ema = 0.0
            seconds = float(self.cfg.get("calibration", {}).get("seconds", 10.0))
            self.calibrating_until = time.time() + seconds
        return {"ok": True, "seconds": seconds}

    def save_feedback(self, req: FeedbackRequest) -> dict[str, Any]:
        with self.lock:
            if not self.timeline:
                raise ValueError("No event is available for feedback")
            append_event_feedback(self.feedback_path, {"feedback": req.feedback, **self.timeline[-1]})
        return {"ok": True, "path": str(self.feedback_path)}

    def process_image_request(self, req: FrameRequest) -> FrameResponse:
        frame_bgr = decode_data_url_image(req.image)
        return self.process_frame(frame_bgr, req)

    def process_frame(self, frame_bgr, req: FrameRequest) -> FrameResponse:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        now = time.time()

        with self.lock:
            det = self.tracker.process(frame_rgb)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is not None:
                diff = cv2.absdiff(gray, self.prev_gray)
                motion_energy = float(diff.mean()) / 255.0
                self.motion_ema = 0.2 * motion_energy + 0.8 * self.motion_ema
            self.prev_gray = gray

            result = self.pipeline.process(
                frame_rgb,
                det,
                calibrating=now <= self.calibrating_until,
                motion_energy=self.motion_ema,
            )
            mode_state = self._update_mode(req.mode, result, req, now)
            gate = calibration_gate_status(result["quality"], self.cfg)
            generic_summary = {
                "mode": req.mode,
                "state": "insufficient_signal" if not result["quality"].get("signal_ok") else "active",
                "duration_secs": 0.0,
                "usable_signal_pct": 100.0 if result["quality"].get("signal_ok") else 0.0,
                "review_events": 0,
                "visible_pct": 100.0 if result["quality"].get("face_present") else 0.0,
            }
            shared_summary = compact_summary(
                req.mode,
                mode_state.get("driver", {}),
                mode_state.get("care", {}),
                mode_state.get("healthcare", {}),
                generic_summary,
            )
            payload = FrameResponse(
                time=now,
                calibrating=now <= self.calibrating_until,
                calibration_gate=gate,
                metrics={
                    "fatigue": round(result["smoothed"]["fatigue"], 2),
                    "attention": round(result["smoothed"]["attention"], 2),
                    "tension": round(result["smoothed"]["tension"], 2),
                    "readiness": round(result["fused"]["readiness"], 2),
                },
                states={
                    "fatigue": result["fatigue"]["state"],
                    "attention": result["attention"]["state"],
                    "tension": result["tension"]["state"],
                    "posture": result["posture"]["state"],
                    "distance": result["distance"]["state"],
                },
                quality=self._public_quality(result["quality"]),
                debug={
                    "attention": {
                        "state": result["attention"].get("state"),
                        "offscreen_duration": json_ready(result["attention"].get("offscreen_duration")),
                        "gaze_x": json_ready(result["attention"].get("gaze_x")),
                        "gaze_y": json_ready(result["attention"].get("gaze_y")),
                        "gaze_x_delta": json_ready(result["attention"].get("gaze_x_delta")),
                        "gaze_y_delta": json_ready(result["attention"].get("gaze_y_delta")),
                        "gaze_confidence": json_ready(result["attention"].get("gaze_confidence")),
                        "head_yaw_delta_deg": json_ready(result["attention"].get("head_yaw_delta_deg")),
                        "attention_signal_ok": json_ready(result["attention"].get("attention_signal_ok")),
                    }
                },
                mode_state=mode_state,
                summary=shared_summary,
                timeline=self.timeline[-25:],
                tracker_backend=self.tracker_backend,
            )
            self.latest_result = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            return payload

    def _update_mode(self, mode: str, result: dict[str, Any], req: FrameRequest, now: float) -> dict[str, Any]:
        if mode == "Driver":
            if not self.driver_active:
                self.driver_monitor.reset(now)
                self.driver_active = True
            driver = self.driver_monitor.update(result, now)
            self._add_events(mode, driver.get("events", []))
            return {"driver": driver}
        if mode == "Care observation":
            if not self.care_active:
                self.care_monitor.reset(now)
                self.care_active = True
            care = self.care_monitor.update(result, now, context=req.care_context, motion_energy=self.motion_ema)
            self._add_events(mode, care.get("events", []))
            return {"care": care}
        if mode == "Healthcare observation":
            healthcare_meta = req.healthcare
            patient_id = healthcare_meta.patient_session_id.strip()
            if healthcare_meta.consent_captured and patient_id and not self.healthcare_active:
                self.healthcare_monitor.reset(now)
                self.healthcare_active = True
            if not self.healthcare_active:
                return {"healthcare": {"state": "waiting_for_consent", "events": [], "summary": {}}}
            result["motion_energy"] = self.motion_ema
            healthcare = self.healthcare_monitor.update(
                result,
                now,
                observation_type=healthcare_meta.observation_type,
                patient_session_id=patient_id,
                note=healthcare_meta.note,
                calibration_ok=bool(result["quality"].get("signal_ok")),
            )
            self._add_events(mode, healthcare.get("events", []))
            return {"healthcare": healthcare}
        return {"desk": {"state": "active", "events": [], "summary": {}}}

    def _add_events(self, mode: str, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.timeline.append({"mode": mode, **json_ready(event)})
        self.timeline = self.timeline[-200:]

    def _public_quality(self, quality: dict[str, Any]) -> dict[str, Any]:
        keys = ["signal_ok", "reasons", "brightness", "sharpness", "face_present", "face_ratio", "frame_drops"]
        return {key: json_ready(quality.get(key)) for key in keys}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 4)
    return value


def reset_scoring_modules() -> None:
    from modules import attention, fatigue, posture, stress

    for module in (attention, fatigue, posture, stress):
        reset = getattr(module, "reset", None)
        if reset:
            reset()
