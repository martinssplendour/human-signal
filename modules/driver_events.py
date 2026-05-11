from dataclasses import dataclass, field
from typing import Dict, List, Optional


RISK_ORDER = {
    "normal": 0,
    "insufficient_signal": 1,
    "watch": 2,
    "elevated": 3,
    "critical": 4,
}


@dataclass
class DriverEvent:
    t_epoch: float
    event: str
    severity: str
    message: str
    fatigue_state: str
    attention_state: str
    signal_ok: bool
    fatigue_score: float
    attention_score: float
    perclos: float


@dataclass
class DriverMonitor:
    trip_start: Optional[float] = None
    last_break: Optional[float] = None
    last_update: Optional[float] = None
    signal_samples: int = 0
    usable_signal_samples: int = 0
    event_counts: Dict[str, int] = field(default_factory=dict)
    active_event_since: Dict[str, float] = field(default_factory=dict)
    last_event_t: Dict[str, float] = field(default_factory=dict)
    looking_away_total: float = 0.0
    high_perclos_total: float = 0.0
    longest_eye_closure: float = 0.0
    latest_risk: str = "normal"

    def reset(self, now: float) -> None:
        self.trip_start = now
        self.last_break = now
        self.last_update = now
        self.signal_samples = 0
        self.usable_signal_samples = 0
        self.event_counts.clear()
        self.active_event_since.clear()
        self.last_event_t.clear()
        self.looking_away_total = 0.0
        self.high_perclos_total = 0.0
        self.longest_eye_closure = 0.0
        self.latest_risk = "normal"

    def mark_break(self, now: float) -> None:
        self.last_break = now
        self.event_counts.clear()
        self.active_event_since.clear()
        self.latest_risk = "normal"

    def _count(self, event: str) -> None:
        self.event_counts[event] = self.event_counts.get(event, 0) + 1

    def _emit_once(self, now: float, event: str, cooldown: float = 8.0) -> bool:
        last = self.last_event_t.get(event)
        if last is not None and now - last < cooldown:
            return False
        self.last_event_t[event] = now
        self._count(event)
        return True

    def update(self, result: Dict, now: float) -> Dict:
        if self.trip_start is None:
            self.reset(now)

        dt = 0.0 if self.last_update is None else max(0.0, min(5.0, now - self.last_update))
        self.last_update = now
        self.signal_samples += 1

        fatigue = result["fatigue"]
        attention = result["attention"]
        quality = result["quality"]
        sm = result["smoothed"]
        signal_ok = bool(quality.get("signal_ok"))
        if signal_ok:
            self.usable_signal_samples += 1

        if attention["state"] in {"looking_away", "head_turned", "face_absent"}:
            self.looking_away_total += dt
        if fatigue.get("perclos", 0.0) >= 0.25:
            self.high_perclos_total += dt
        if fatigue.get("microsleep"):
            self.longest_eye_closure = max(self.longest_eye_closure, fatigue.get("blink_duration", 0.0), 1.5)

        events: List[DriverEvent] = []

        def add(event: str, severity: str, message: str, cooldown: float = 8.0) -> None:
            if self._emit_once(now, event, cooldown):
                events.append(DriverEvent(
                    t_epoch=now,
                    event=event,
                    severity=severity,
                    message=message,
                    fatigue_state=fatigue["state"],
                    attention_state=attention["state"],
                    signal_ok=signal_ok,
                    fatigue_score=float(sm["fatigue"]),
                    attention_score=float(sm["attention"]),
                    perclos=float(fatigue.get("perclos", 0.0)),
                ))

        if not signal_ok:
            add("poor_signal", "insufficient_signal", "Camera signal is unreliable. Check face visibility, lighting, and blur.", cooldown=12.0)

        if fatigue["state"] == "microsleep":
            add("microsleep_detected", "critical", "Possible microsleep detected. Pull over safely and rest.", cooldown=5.0)
        elif fatigue["state"] == "drowsy":
            add("sustained_eye_closure", "elevated", "Sustained eye-closure pattern detected. Take a break soon.", cooldown=10.0)
        elif fatigue["state"] == "fatigue_signs":
            add("fatigue_signs", "watch", "Fatigue signs are increasing. Consider a rest break.", cooldown=20.0)

        if fatigue.get("yawns", 0) >= 2:
            add("repeated_yawning", "watch", "Repeated yawning detected.", cooldown=30.0)
        if fatigue.get("head_nods", 0) >= 2:
            add("head_nod_cluster", "elevated", "Repeated head nods detected. Take a break soon.", cooldown=20.0)
        if attention["state"] in {"looking_away", "head_turned"} and attention.get("offscreen_duration", 0.0) >= 2.0:
            add("looking_away_sustained", "elevated", "Eyes or head are away from the road for too long.", cooldown=8.0)
        if attention["state"] == "face_absent":
            add("face_absent", "insufficient_signal", "Driver face is not visible to the camera.", cooldown=12.0)

        trip_secs = now - (self.trip_start or now)
        break_secs = now - (self.last_break or now)
        if break_secs >= 2 * 60 * 60:
            add("break_recommended", "watch", "Two hours since last marked break. Plan a safe rest stop.", cooldown=300.0)

        risk = "normal"
        for ev in events:
            if RISK_ORDER[ev.severity] > RISK_ORDER[risk]:
                risk = ev.severity

        recent_critical = any(
            event in {"microsleep_detected", "head_nod_cluster"} and now - t <= 60.0
            for event, t in self.last_event_t.items()
        )
        if recent_critical:
            risk = "critical"
        elif sm["fatigue"] >= 70.0 or attention["state"] in {"looking_away", "head_turned"}:
            risk = "elevated" if RISK_ORDER[risk] < RISK_ORDER["elevated"] else risk
        elif sm["fatigue"] >= 45.0 or fatigue["state"] == "fatigue_signs":
            risk = "watch" if RISK_ORDER[risk] < RISK_ORDER["watch"] else risk
        if not signal_ok and RISK_ORDER[risk] < RISK_ORDER["insufficient_signal"]:
            risk = "insufficient_signal"

        self.latest_risk = risk
        return {"risk": risk, "events": [ev.__dict__ for ev in events], "summary": self.summary(now)}

    def summary(self, now: float) -> Dict:
        trip_secs = 0.0 if self.trip_start is None else now - self.trip_start
        break_secs = 0.0 if self.last_break is None else now - self.last_break
        usable_pct = 0.0 if self.signal_samples == 0 else 100.0 * self.usable_signal_samples / self.signal_samples
        return {
            "trip_secs": trip_secs,
            "since_break_secs": break_secs,
            "usable_signal_pct": usable_pct,
            "microsleeps": self.event_counts.get("microsleep_detected", 0),
            "critical_alerts": sum(v for k, v in self.event_counts.items() if k in {"microsleep_detected", "head_nod_cluster"}),
            "yawn_alerts": self.event_counts.get("repeated_yawning", 0),
            "away_secs": self.looking_away_total,
            "high_perclos_secs": self.high_perclos_total,
            "longest_eye_closure": self.longest_eye_closure,
            "risk": self.latest_risk,
        }
