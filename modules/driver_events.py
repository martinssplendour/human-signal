import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


RISK_ORDER = {
    "normal": 0,
    "insufficient_signal": 1,
    "watch": 2,
    "elevated": 3,
    "critical": 4,
}

DEFAULT_DRIVER_CFG = {
    "yawn_watch_count": 3,
    "yawn_elevated_count": 5,
    "yawn_window_secs": 300.0,
    "rapid_yawn_count": 3,
    "rapid_yawn_window_secs": 120.0,
    "eye_closed_elevated_secs": 1.5,
    "eye_closed_critical_secs": 2.0,
    "perclos_watch": 0.10,
    "perclos_elevated": 0.15,
    "perclos_critical": 0.25,
    "blink_watch_ms": 200.0,
    "blink_elevated_ms": 300.0,
    "head_nod_critical_count": 3,
    "head_nod_window_secs": 120.0,
    "hydration_reminder_secs": 5400.0,
    "break_reminder_secs": 7200.0,
    "professional_limit_secs": 16200.0,
    "fatigue_score_elevated": 80.0,
    "compound_elevated_count": 2,
    "compound_window_secs": 600.0,
    "long_trip_secs": 10800.0,
    "long_trip_threshold_multiplier": 0.8,
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
class _Candidate:
    event: str
    severity: str
    message: str
    group: str
    cooldown: float


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
    yawn_total_seen: int = 0
    head_nod_total_seen: int = 0
    yawn_times: deque = field(default_factory=lambda: deque(maxlen=30))
    head_nod_times: deque = field(default_factory=lambda: deque(maxlen=30))
    elevated_event_times: deque = field(default_factory=lambda: deque(maxlen=20))

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
        self.yawn_total_seen = 0
        self.head_nod_total_seen = 0
        self.yawn_times.clear()
        self.head_nod_times.clear()
        self.elevated_event_times.clear()

    def mark_break(self, now: float) -> None:
        self.last_break = now
        self.event_counts.clear()
        self.active_event_since.clear()
        self.latest_risk = "normal"
        self.yawn_total_seen = 0
        self.head_nod_total_seen = 0
        self.yawn_times.clear()
        self.head_nod_times.clear()
        self.elevated_event_times.clear()

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
        raw = result.get("raw_scores", sm)
        signal_ok = bool(quality.get("signal_ok"))
        if signal_ok:
            self.usable_signal_samples += 1

        if attention["state"] == "face_absent":
            self.looking_away_total += dt
        if fatigue.get("perclos", 0.0) >= self._cfg_float("perclos_critical"):
            self.high_perclos_total += dt
        if fatigue.get("microsleep"):
            self.longest_eye_closure = max(
                self.longest_eye_closure,
                fatigue.get("closed_duration", 0.0),
                fatigue.get("blink_duration", 0.0),
                1.5,
            )
        self._collect_counter_events(now, fatigue)

        final_candidates = self._suppressed_candidates(self._build_candidates(result, now), now)
        emitted_candidates = []
        events = []
        for candidate in final_candidates:
            if self._emit_once(now, candidate.event, candidate.cooldown):
                emitted_candidates.append(candidate)
                events.append(self._to_event(candidate, now, fatigue, attention, signal_ok, raw))

        for candidate in emitted_candidates:
            if candidate.severity == "elevated":
                self.elevated_event_times.append((now, candidate.group))

        compound = self._compound_candidate(now)
        if compound and self._emit_once(now, compound.event, compound.cooldown):
            events.append(self._to_event(compound, now, fatigue, attention, signal_ok, raw))

        risk = "normal"
        for ev in events:
            if RISK_ORDER[ev.severity] > RISK_ORDER[risk]:
                risk = ev.severity

        if not events:
            risk = self._background_risk(result, now)
        if not signal_ok and RISK_ORDER[risk] < RISK_ORDER["insufficient_signal"]:
            risk = "insufficient_signal"

        self.latest_risk = risk
        return {"risk": risk, "events": [ev.__dict__ for ev in events], "summary": self.summary(now)}

    def _build_candidates(self, result: Dict, now: float) -> List[_Candidate]:
        fatigue = result["fatigue"]
        attention = result["attention"]
        quality = result["quality"]
        raw = result.get("raw_scores", result["smoothed"])
        candidates: List[_Candidate] = []

        def add(event: str, severity: str, message: str, group: str, cooldown: float = 60.0) -> None:
            candidates.append(_Candidate(event, severity, message, group, cooldown))

        if not quality.get("signal_ok"):
            add("poor_signal", "insufficient_signal", "Camera signal is unreliable. Check face visibility, lighting, and blur.", "signal", 12.0)

        closed_duration = float(fatigue.get("closed_duration", 0.0))
        blink_ms = 1000.0 * float(fatigue.get("blink_duration", 0.0))
        perclos = float(fatigue.get("perclos", 0.0))
        yawns_window = self._count_recent(self.yawn_times, now, self._cfg_float("yawn_window_secs"))
        rapid_yawns = self._count_recent(self.yawn_times, now, self._cfg_float("rapid_yawn_window_secs"))
        nods_window = self._count_recent(self.head_nod_times, now, self._cfg_float("head_nod_window_secs"))

        if closed_duration >= self._threshold("eye_closed_critical_secs"):
            add("sustained_eye_closure", "critical", "PULL OVER immediately - eyes closed for 2 seconds. Do not continue.", "eye", 10.0)
        elif fatigue.get("microsleep"):
            add("microsleep_detected", "critical", "PULL OVER immediately - possible microsleep detected. Do not continue.", "eye", 10.0)
        elif closed_duration >= self._threshold("eye_closed_elevated_secs"):
            add("eye_closure_elevated", "elevated", "Eyes have been closed for 1.5 seconds. Prepare to stop safely.", "eye", 30.0)

        if perclos >= self._threshold("perclos_critical"):
            add("perclos_critical", "critical", "PULL OVER immediately - sustained eye closure is at a critical level.", "eye", 60.0)
        elif perclos >= self._threshold("perclos_elevated"):
            add("perclos_elevated", "elevated", "Eye-closure pattern is elevated. Plan to stop soon.", "eye", 60.0)
        elif perclos >= self._threshold("perclos_watch"):
            add("perclos_watch", "watch", "Early drowsiness signs detected. Stay alert and consider a break.", "eye", 120.0)

        fatigue_score = float(raw.get("fatigue", 0.0))
        if blink_ms >= self._threshold("blink_elevated_ms") and (
            perclos >= self._threshold("perclos_watch") or fatigue_score >= 65.0
        ):
            add("slow_blink_elevated", "elevated", "Slow blink pattern detected. Plan to stop soon.", "eye", 90.0)
        elif blink_ms >= self._threshold("blink_watch_ms"):
            add("slow_blink_watch", "watch", "Long blink detected. Continue monitoring for repeated fatigue signs.", "eye", 120.0)

        if nods_window >= self._threshold_count("head_nod_critical_count"):
            add("head_nod_cluster", "critical", "PULL OVER immediately - repeated head nods detected.", "head_nod", 120.0)

        rapid_yawn_critical = (
            rapid_yawns >= self._threshold_count("rapid_yawn_count")
            and (perclos >= self._threshold("perclos_elevated") or closed_duration >= self._threshold("eye_closed_elevated_secs") or nods_window >= 2)
        )
        if rapid_yawn_critical:
            add("rapid_yawning_compound", "critical", "PULL OVER immediately - rapid yawning combined with fatigue signs.", "yawn", 120.0)
        elif yawns_window >= self._threshold_count("yawn_elevated_count"):
            add("repeated_yawning_5m", "elevated", "Five yawns detected in 5 minutes. Plan to stop in the next 10-15 minutes.", "yawn", 300.0)
        elif rapid_yawns >= self._threshold_count("rapid_yawn_count"):
            add("rapid_yawning", "elevated", "Rapid yawning detected. Plan to stop in the next 10-15 minutes.", "yawn", 120.0)
        elif yawns_window >= self._threshold_count("yawn_watch_count"):
            add("repeated_yawning_watch", "watch", "Three yawns detected in 5 minutes. Monitor fatigue.", "yawn", 180.0)

        if attention["state"] == "face_absent":
            add("face_absent", "insufficient_signal", "Driver face is not visible to the camera.", "signal", 12.0)

        trip_secs = now - (self.trip_start if self.trip_start is not None else now)
        break_secs = now - (self.last_break if self.last_break is not None else now)
        if trip_secs >= self._cfg_float("professional_limit_secs"):
            add("professional_limit_exceeded", "critical", "Professional driving break limit reached. Stop at the nearest safe rest area.", "trip", 300.0)
        elif break_secs >= self._cfg_float("break_reminder_secs"):
            add("break_recommended", "watch", "Two hours since last marked break. Plan a safe rest stop.", "trip", 300.0)
        elif break_secs >= self._cfg_float("hydration_reminder_secs"):
            add("hydration_reminder", "watch", "You've been driving 90 minutes - drink water. Dehydration can increase fatigue.", "wellness", 300.0)

        if float(raw.get("fatigue", 0.0)) >= self._threshold("fatigue_score_elevated"):
            add("fatigue_score_elevated", "elevated", "Fatigue score is elevated. Plan to stop soon.", "score", 90.0)

        return candidates

    def _suppressed_candidates(self, candidates: List[_Candidate], now: float) -> List[_Candidate]:
        best_by_group: Dict[str, _Candidate] = {}
        for candidate in candidates:
            existing = best_by_group.get(candidate.group)
            if existing is None or RISK_ORDER[candidate.severity] > RISK_ORDER[existing.severity]:
                best_by_group[candidate.group] = candidate
        return sorted(best_by_group.values(), key=lambda item: RISK_ORDER[item.severity], reverse=True)

    def _compound_candidate(self, now: float) -> Optional[_Candidate]:
        groups = {
            group
            for event_t, group in self.elevated_event_times
            if now - event_t <= self._cfg_float("compound_window_secs")
        }
        if len(groups) >= self._cfg_int("compound_elevated_count"):
            return _Candidate(
                "compound_fatigue_escalation",
                "critical",
                "Multiple elevated fatigue warnings occurred within 10 minutes. Stop at the nearest safe place.",
                "compound",
                300.0,
            )
        return None

    def _background_risk(self, result: Dict, now: float) -> str:
        fatigue = result["fatigue"]
        raw = result.get("raw_scores", result["smoothed"])
        yawns_window = self._count_recent(self.yawn_times, now, self._cfg_float("yawn_window_secs"))
        if (
            yawns_window >= self._threshold_count("yawn_elevated_count")
            or float(raw.get("fatigue", 0.0)) >= 80.0
        ):
            return "elevated"
        if yawns_window >= self._threshold_count("yawn_watch_count") or float(raw.get("fatigue", 0.0)) >= 65.0 or fatigue.get("perclos", 0.0) >= self._threshold("perclos_watch"):
            return "watch"
        return "normal"

    def _to_event(self, candidate: _Candidate, now: float, fatigue: Dict, attention: Dict, signal_ok: bool, raw: Dict) -> DriverEvent:
        return DriverEvent(
            t_epoch=now,
            event=candidate.event,
            severity=candidate.severity,
            message=candidate.message,
            fatigue_state=fatigue["state"],
            attention_state=attention["state"],
            signal_ok=signal_ok,
            fatigue_score=float(raw["fatigue"]),
            attention_score=float(raw["attention"]),
            perclos=float(fatigue.get("perclos", 0.0)),
        )

    def _collect_counter_events(self, now: float, fatigue: Dict) -> None:
        yawn_count = int(fatigue.get("yawns", 0))
        while self.yawn_total_seen < yawn_count:
            self.yawn_times.append(now)
            self.yawn_total_seen += 1
        head_nod_count = int(fatigue.get("head_nods", 0))
        while self.head_nod_total_seen < head_nod_count:
            self.head_nod_times.append(now)
            self.head_nod_total_seen += 1

    @staticmethod
    def _count_recent(events: deque, now: float, window: float) -> int:
        return sum(1 for event_t in events if now - event_t <= window)

    def _driver_thresholds(self) -> Dict:
        return getattr(self, "thresholds", {})

    def _driver_cfg(self) -> Dict:
        cfg = dict(DEFAULT_DRIVER_CFG)
        cfg.update(getattr(self, "driver_cfg", {}) or {})
        return cfg

    def _cfg_float(self, key: str) -> float:
        return float(self._driver_cfg()[key])

    def _cfg_int(self, key: str) -> int:
        return int(self._driver_cfg()[key])

    def _threshold(self, key: str) -> float:
        value = self._cfg_float(key)
        if self.trip_start is not None and self.last_update is not None and self.last_update - self.trip_start >= self._cfg_float("long_trip_secs"):
            value *= self._cfg_float("long_trip_threshold_multiplier")
        return value

    def _threshold_count(self, key: str) -> int:
        value = self._cfg_int(key)
        if self.trip_start is not None and self.last_update is not None and self.last_update - self.trip_start >= self._cfg_float("long_trip_secs"):
            value = max(1, math.ceil(value * self._cfg_float("long_trip_threshold_multiplier")))
        return value

    def summary(self, now: float) -> Dict:
        trip_secs = 0.0 if self.trip_start is None else now - self.trip_start
        break_secs = 0.0 if self.last_break is None else now - self.last_break
        usable_pct = 0.0 if self.signal_samples == 0 else 100.0 * self.usable_signal_samples / self.signal_samples
        return {
            "trip_secs": trip_secs,
            "since_break_secs": break_secs,
            "usable_signal_pct": usable_pct,
            "microsleeps": self.event_counts.get("microsleep_detected", 0),
            "critical_alerts": sum(v for k, v in self.event_counts.items() if k in {"microsleep_detected", "head_nod_cluster", "sustained_eye_closure", "compound_fatigue_escalation"}),
            "yawn_alerts": self.event_counts.get("repeated_yawning_5m", 0) + self.event_counts.get("rapid_yawning", 0),
            "yawns_5m": self._count_recent(self.yawn_times, now, self._cfg_float("yawn_window_secs")),
            "head_nods_2m": self._count_recent(self.head_nod_times, now, self._cfg_float("head_nod_window_secs")),
            "away_secs": self.looking_away_total,
            "high_perclos_secs": self.high_perclos_total,
            "longest_eye_closure": self.longest_eye_closure,
            "risk": self.latest_risk,
        }
