from dataclasses import dataclass, field
from typing import Dict, List, Optional


STATE_ORDER = {
    "stable_observation": 0,
    "resting": 1,
    "insufficient_signal": 2,
    "needs_review": 3,
    "urgent_review": 4,
}


@dataclass
class HealthcareEvent:
    t_epoch: float
    event: str
    state: str
    message: str
    observation_type: str
    patient_session_id: str
    fatigue_state: str
    attention_state: str
    posture_state: str
    signal_ok: bool
    visible: bool
    note: str = ""


@dataclass
class HealthcareMonitor:
    session_start: Optional[float] = None
    last_update: Optional[float] = None
    last_visible: Optional[float] = None
    signal_samples: int = 0
    usable_signal_samples: int = 0
    visible_samples: int = 0
    longest_eye_closure: float = 0.0
    longest_absence: float = 0.0
    event_counts: Dict[str, int] = field(default_factory=dict)
    last_event_t: Dict[str, float] = field(default_factory=dict)
    latest_state: str = "stable_observation"

    def reset(self, now: float) -> None:
        self.session_start = now
        self.last_update = now
        self.last_visible = now
        self.signal_samples = 0
        self.usable_signal_samples = 0
        self.visible_samples = 0
        self.longest_eye_closure = 0.0
        self.longest_absence = 0.0
        self.event_counts.clear()
        self.last_event_t.clear()
        self.latest_state = "stable_observation"

    def _max_state(self, current: str, candidate: str) -> str:
        return candidate if STATE_ORDER[candidate] > STATE_ORDER[current] else current

    def _count(self, event: str) -> None:
        self.event_counts[event] = self.event_counts.get(event, 0) + 1

    def _emit_once(self, now: float, event: str, cooldown: float) -> bool:
        last = self.last_event_t.get(event)
        if last is not None and now - last < cooldown:
            return False
        self.last_event_t[event] = now
        self._count(event)
        return True

    def update(
        self,
        result: Dict,
        now: float,
        observation_type: str = "General observation",
        patient_session_id: str = "",
        note: str = "",
        calibration_ok: bool = True,
    ) -> Dict:
        if self.session_start is None:
            self.reset(now)

        self.last_update = now
        quality = result["quality"]
        fatigue = result["fatigue"]
        attention = result["attention"]
        posture = result["posture"]
        visible = bool(quality.get("face_present")) and attention["state"] != "face_absent"
        signal_ok = bool(quality.get("signal_ok"))

        self.signal_samples += 1
        if signal_ok:
            self.usable_signal_samples += 1
        if visible:
            self.visible_samples += 1
            self.last_visible = now

        absence_secs = 0.0 if self.last_visible is None else now - self.last_visible
        self.longest_absence = max(self.longest_absence, absence_secs)
        if fatigue.get("microsleep"):
            self.longest_eye_closure = max(self.longest_eye_closure, 1.5)
        self.longest_eye_closure = max(self.longest_eye_closure, float(fatigue.get("blink_duration", 0.0) or 0.0))

        events: List[HealthcareEvent] = []

        def add(event: str, state: str, message: str, cooldown: float = 45.0) -> None:
            if self._emit_once(now, event, cooldown):
                events.append(HealthcareEvent(
                    t_epoch=now,
                    event=event,
                    state=state,
                    message=message,
                    observation_type=observation_type,
                    patient_session_id=patient_session_id,
                    fatigue_state=fatigue["state"],
                    attention_state=attention["state"],
                    posture_state=posture["state"],
                    signal_ok=signal_ok,
                    visible=visible,
                    note=note,
                ))

        state = "stable_observation"
        if fatigue["state"] in {"microsleep", "drowsy"} or attention["state"] == "eyes_closed":
            state = "resting"

        if not calibration_ok:
            state = self._max_state(state, "insufficient_signal")
            add("calibration_failed", "insufficient_signal", "Calibration quality is insufficient for reliable observation.", cooldown=60.0)
        if not signal_ok:
            state = self._max_state(state, "insufficient_signal")
            add("poor_signal", "insufficient_signal", "Observation signal is unreliable. Check lighting, blur, camera position, or face visibility.", cooldown=60.0)

        if not visible and absence_secs >= 60.0:
            state = self._max_state(state, "needs_review")
            add("patient_out_of_frame", "needs_review", "Patient is out of frame or face is not visible.", cooldown=60.0)
        if not visible and absence_secs >= 240.0:
            state = self._max_state(state, "urgent_review")
            add("observation_interrupted", "urgent_review", "Observation has been interrupted for an extended period.", cooldown=120.0)

        if fatigue.get("microsleep") or fatigue["state"] == "drowsy":
            state = self._max_state(state, "needs_review")
            add("prolonged_eye_closure", "needs_review", "Prolonged eye closure or high eye-closure pattern observed.", cooldown=60.0)
        if fatigue["state"] == "fatigue_signs":
            state = self._max_state(state, "needs_review")
            add("repeated_fatigue_signs", "needs_review", "Repeated fatigue signs observed.", cooldown=90.0)
        if attention["state"] in {"face_absent", "eyes_closed"} and fatigue["state"] in {"drowsy", "microsleep"}:
            state = self._max_state(state, "needs_review")
            add("reduced_responsiveness_proxy", "needs_review", "Reduced responsiveness proxy observed. Clinician review recommended.", cooldown=90.0)
        if posture["state"] == "slouching":
            state = self._max_state(state, "needs_review")
            add("posture_decline", "needs_review", "Posture decline observed during the session.", cooldown=180.0)
        if result.get("motion_energy", 0.0) >= 0.08:
            state = self._max_state(state, "needs_review")
            add("restlessness", "needs_review", "Restlessness or elevated movement observed.", cooldown=90.0)

        self.latest_state = state
        return {
            "state": state,
            "events": [ev.__dict__ for ev in events],
            "summary": self.summary(now),
            "absence_secs": absence_secs,
        }

    def summary(self, now: float) -> Dict:
        session_secs = 0.0 if self.session_start is None else now - self.session_start
        usable_pct = 0.0 if self.signal_samples == 0 else 100.0 * self.usable_signal_samples / self.signal_samples
        visible_pct = 0.0 if self.signal_samples == 0 else 100.0 * self.visible_samples / self.signal_samples
        needs_review = sum(
            self.event_counts.get(k, 0)
            for k in {
                "patient_out_of_frame",
                "prolonged_eye_closure",
                "repeated_fatigue_signs",
                "reduced_responsiveness_proxy",
                "posture_decline",
                "restlessness",
            }
        )
        urgent = self.event_counts.get("observation_interrupted", 0)
        return {
            "session_secs": session_secs,
            "usable_signal_pct": usable_pct,
            "visible_pct": visible_pct,
            "longest_eye_closure": self.longest_eye_closure,
            "longest_absence": self.longest_absence,
            "needs_review_events": needs_review,
            "urgent_review_events": urgent,
            "poor_signal_events": self.event_counts.get("poor_signal", 0),
            "state": self.latest_state,
        }
