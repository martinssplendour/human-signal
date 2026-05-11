import time


DEFAULT_EAR_CLOSED = 0.20
BLINK_MIN_SECS = 0.06
BLINK_MAX_SECS = 0.80

_state = {
    "closed_started": None,
    "blink_times": [],
    "blink_durations": [],
    "eye_samples": [],
    "yawn_start": None,
    "yawn_count": 0,
    "open_ear_base": None,
    "last_neck": None,
    "head_nod_count": 0,
}


def reset():
    _state.update({
        "closed_started": None,
        "blink_times": [],
        "blink_durations": [],
        "eye_samples": [],
        "yawn_start": None,
        "yawn_count": 0,
        "open_ear_base": None,
        "last_neck": None,
        "head_nod_count": 0,
    })


def _ear(feats):
    le, re = feats.get("ear_left"), feats.get("ear_right")
    if le is None:
        return re
    if re is None:
        return le
    return (le + re) / 2.0


def fatigue_score(feats, quality, cfg, calibrating=False, posture_state=None, motion_energy=0.0):
    now = time.time()
    ear = _ear(feats)
    mar = feats.get("mar")
    neck = feats.get("neck_angle")
    window = float(cfg["windows"].get("fatigue_seconds", 60))

    if calibrating:
        if ear is not None and ear > DEFAULT_EAR_CLOSED:
            base = _state["open_ear_base"]
            _state["open_ear_base"] = ear if base is None else 0.9 * base + 0.1 * ear
        return {
            "score": 0.0,
            "state": "calibrating",
            "blink_rate": 0.0,
            "perclos": 0.0,
            "blink_duration": 0.0,
            "microsleep": False,
            "yawns": _state["yawn_count"],
            "head_nods": _state["head_nod_count"],
            "conf": quality["conf_base"] * 0.8,
        }

    if ear is None or not quality.get("face_present", False):
        return {
            "score": 0.0,
            "state": "insufficient_signal",
            "blink_rate": 0.0,
            "perclos": 0.0,
            "blink_duration": 0.0,
            "microsleep": False,
            "yawns": _state["yawn_count"],
            "head_nods": _state["head_nod_count"],
            "conf": quality["conf_base"] * 0.4,
        }

    closed_threshold = DEFAULT_EAR_CLOSED
    if _state["open_ear_base"] is not None:
        closed_threshold = max(0.12, _state["open_ear_base"] * cfg["thresholds"].get("fatigue_ear_closed_ratio", 0.7))
    closed = ear < closed_threshold

    _state["eye_samples"].append((now, closed))
    _state["eye_samples"] = [(t, c) for t, c in _state["eye_samples"] if now - t <= window]

    if closed and _state["closed_started"] is None:
        _state["closed_started"] = now
    elif not closed and _state["closed_started"] is not None:
        duration = now - _state["closed_started"]
        if BLINK_MIN_SECS <= duration <= BLINK_MAX_SECS:
            _state["blink_times"].append(now)
            _state["blink_durations"].append(duration)
        _state["closed_started"] = None

    _state["blink_times"] = [t for t in _state["blink_times"] if now - t <= 60.0]
    _state["blink_durations"] = _state["blink_durations"][-120:]
    blink_rate = len(_state["blink_times"])
    perclos = 0.0
    if _state["eye_samples"]:
        perclos = sum(1 for _, c in _state["eye_samples"] if c) / len(_state["eye_samples"])

    closed_duration = (now - _state["closed_started"]) if _state["closed_started"] is not None else 0.0
    microsleep = closed_duration >= cfg["thresholds"].get("microsleep_secs", 1.5)

    yawn_thr = cfg["thresholds"].get("yawn_mar", 0.6)
    yawn_min = cfg["thresholds"].get("yawn_min_secs", 0.5)
    if mar is not None and mar >= yawn_thr:
        if _state["yawn_start"] is None:
            _state["yawn_start"] = now
        elif now - _state["yawn_start"] >= yawn_min:
            _state["yawn_count"] += 1
            _state["yawn_start"] = None
    else:
        _state["yawn_start"] = None

    if neck is not None and _state["last_neck"] is not None:
        if neck - _state["last_neck"] > 12.0:
            _state["head_nod_count"] += 1
    _state["last_neck"] = neck

    avg_blink_duration = sum(_state["blink_durations"]) / len(_state["blink_durations"]) if _state["blink_durations"] else 0.0
    score = 100.0 * min(1.0, perclos / max(0.01, cfg["thresholds"].get("perclos_drowsy", 0.25)))
    if blink_rate > cfg["thresholds"].get("blink_rate_high", 25):
        score = max(score, 60.0)
    if microsleep:
        score = max(score, 95.0)
    if _state["yawn_count"] > 0:
        score = max(score, min(90.0, 45.0 + 10.0 * _state["yawn_count"]))
    if posture_state == "slouching":
        score = max(score, 35.0)

    state = "alert"
    if microsleep:
        state = "microsleep"
    elif perclos >= cfg["thresholds"].get("perclos_drowsy", 0.25):
        state = "drowsy"
    elif blink_rate > cfg["thresholds"].get("blink_rate_high", 25) or _state["yawn_count"] > 0:
        state = "fatigue_signs"

    return {
        "score": float(max(0.0, min(100.0, score))),
        "state": state,
        "blink_rate": blink_rate,
        "perclos": float(perclos),
        "blink_duration": float(avg_blink_duration),
        "microsleep": microsleep,
        "yawns": _state["yawn_count"],
        "head_nods": _state["head_nod_count"],
        "ear_closed_threshold": float(closed_threshold),
        "conf": quality["conf_base"] * (1.0 if quality.get("signal_ok") else 0.55),
    }
