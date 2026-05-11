from dataclasses import dataclass, field
from typing import Dict, List, Optional


STATE_ORDER = {
    "observed": 0,
    "resting": 1,
    "insufficient_signal": 2,
    "needs_review": 3,
    "urgent_review": 4,
}


@dataclass
class CareEvent:
    t_epoch: float
    event: str
    state: str
    message: str
    context: str
    posture_state: str
    attention_state: str
    fatigue_state: str
    signal_ok: bool
    visible: bool


@dataclass
class CareMonitor:
    session_start: Optional[float] = None
    last_update: Optional[float] = None
    last_visible: Optional[float] = None
    last_motion: Optional[float] = None
    signal_samples: int = 0
    usable_signal_samples: int = 0
    visible_samples: int = 0
    invisible_total: float = 0.0
    inactive_total: float = 0.0
    longest_absence: float = 0.0
    event_counts: Dict[str, int] = field(default_factory=dict)
    last_event_t: Dict[str, float] = field(default_factory=dict)
    latest_state: str = "observed"

    def reset(self, now: float) -> None:
        self.session_start = now
        self.last_update = now
        self.last_visible = now
        self.last_motion = now
        self.signal_samples = 0
        self.usable_signal_samples = 0
        self.visible_samples = 0
        self.invisible_total = 0.0
        self.inactive_total = 0.0
        self.longest_absence = 0.0
        self.event_counts.clear()
        self.last_event_t.clear()
        self.latest_state = "observed"

    def _count(self, event: str) -> None:
        self.event_counts[event] = self.event_counts.get(event, 0) + 1

    def _emit_once(self, now: float, event: str, cooldown: float) -> bool:
        last = self.last_event_t.get(event)
        if last is not None and now - last < cooldown:
            return False
        self.last_event_t[event] = now
        self._count(event)
        return True

    def update(self, result: Dict, now: float, context: str = "Chair", motion_energy: float = 0.0) -> Dict:
        if self.session_start is None:
            self.reset(now)

        dt = 0.0 if self.last_update is None else max(0.0, min(5.0, now - self.last_update))
        self.last_update = now

        quality = result["quality"]
        attention = result["attention"]
        fatigue = result["fatigue"]
        posture = result["posture"]
        visible = bool(quality.get("face_present")) and attention["state"] != "face_absent"
        signal_ok = bool(quality.get("signal_ok"))

        self.signal_samples += 1
        if signal_ok:
            self.usable_signal_samples += 1
        if visible:
            self.visible_samples += 1
            self.last_visible = now
        else:
            self.invisible_total += dt

        if motion_energy >= 0.015 or visible:
            self.last_motion = now
        else:
            self.inactive_total += dt

        absence_secs = 0.0 if self.last_visible is None else now - self.last_visible
        inactive_secs = 0.0 if self.last_motion is None else now - self.last_motion
        self.longest_absence = max(self.longest_absence, absence_secs)

        events: List[CareEvent] = []

        def add(event: str, state: str, message: str, cooldown: float = 30.0) -> None:
            if self._emit_once(now, event, cooldown):
                events.append(CareEvent(
                    t_epoch=now,
                    event=event,
                    state=state,
                    message=message,
                    context=context,
                    posture_state=posture["state"],
                    attention_state=attention["state"],
                    fatigue_state=fatigue["state"],
                    signal_ok=signal_ok,
                    visible=visible,
                ))

        state = "observed"
        if fatigue["state"] in {"microsleep", "drowsy"} or attention["state"] == "eyes_closed":
            state = "resting"

        if not signal_ok:
            state = self._max_state(state, "insufficient_signal")
            add("camera_or_signal_issue", "insufficient_signal", "Observation signal is unreliable. Check camera placement, lighting, and visibility.", cooldown=45.0)

        absence_review = 45.0 if context in {"Chair", "Bed"} else 120.0
        absence_urgent = 180.0 if context in {"Chair", "Bed"} else 300.0
        if not visible and absence_secs >= absence_review:
            state = self._max_state(state, "needs_review")
            add("resident_absent", "needs_review", "Resident is not visible in the expected observation area.", cooldown=60.0)
        if not visible and absence_secs >= absence_urgent:
            state = self._max_state(state, "urgent_review")
            add("resident_absent_extended", "urgent_review", "Resident has been absent from view for an extended period.", cooldown=90.0)

        inactivity_review = 20 * 60.0 if context == "Bed" else 10 * 60.0
        if visible and inactive_secs >= inactivity_review:
            state = self._max_state(state, "needs_review")
            add("prolonged_inactivity", "needs_review", "Prolonged inactivity observed. Review if this is expected.", cooldown=120.0)

        possible_slump = (
            context in {"Chair", "Bed"}
            and posture["state"] == "slouching"
            and attention["state"] in {"eyes_closed", "face_absent", "head_turned"}
        )
        if possible_slump:
            state = self._max_state(state, "needs_review")
            add("possible_slump", "needs_review", "Possible slump or posture decline. Caregiver review recommended.", cooldown=60.0)

        if motion_energy >= 0.08 and context in {"Bed", "Chair"}:
            state = self._max_state(state, "needs_review")
            add("restlessness", "needs_review", "Elevated movement/restlessness observed.", cooldown=60.0)

        if posture["state"] == "slouching" and visible:
            add("posture_decline", "needs_review", "Sustained slouched posture observed.", cooldown=180.0)

        self.latest_state = state
        return {
            "state": state,
            "events": [ev.__dict__ for ev in events],
            "summary": self.summary(now),
            "absence_secs": absence_secs,
            "inactive_secs": inactive_secs,
        }

    def _max_state(self, current: str, candidate: str) -> str:
        return candidate if STATE_ORDER[candidate] > STATE_ORDER[current] else current

    def summary(self, now: float) -> Dict:
        session_secs = 0.0 if self.session_start is None else now - self.session_start
        usable_pct = 0.0 if self.signal_samples == 0 else 100.0 * self.usable_signal_samples / self.signal_samples
        visible_pct = 0.0 if self.signal_samples == 0 else 100.0 * self.visible_samples / self.signal_samples
        urgent = self.event_counts.get("resident_absent_extended", 0)
        review = sum(
            self.event_counts.get(k, 0)
            for k in {"resident_absent", "prolonged_inactivity", "possible_slump", "restlessness", "posture_decline"}
        )
        return {
            "session_secs": session_secs,
            "usable_signal_pct": usable_pct,
            "visible_pct": visible_pct,
            "longest_absence": self.longest_absence,
            "invisible_total": self.invisible_total,
            "inactive_total": self.inactive_total,
            "needs_review_events": review,
            "urgent_review_events": urgent,
            "possible_slumps": self.event_counts.get("possible_slump", 0),
            "state": self.latest_state,
        }
