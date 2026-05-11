import time


_state = {
    "baseline_gaze": None,
    "baseline_yaw": None,
    "candidate": None,
    "candidate_since": 0.0,
    "state": "calibrating",
    "offscreen_since": None,
}


def reset():
    _state.update({
        "baseline_gaze": None,
        "baseline_yaw": None,
        "candidate": None,
        "candidate_since": 0.0,
        "state": "calibrating",
        "offscreen_since": None,
    })


def _commit_state(proposed, now, dwell=0.5):
    if proposed == _state["state"]:
        _state["candidate"] = None
        _state["candidate_since"] = 0.0
        return proposed
    if _state["candidate"] != proposed:
        _state["candidate"] = proposed
        _state["candidate_since"] = now
        return _state["state"]
    if now - _state["candidate_since"] >= dwell:
        _state["state"] = proposed
        _state["candidate"] = None
        _state["candidate_since"] = 0.0
    return _state["state"]


def attention_score(feats, quality, cfg, calibrating=False):
    now = time.time()
    face = feats.get("face")
    ear_left, ear_right = feats.get("ear_left"), feats.get("ear_right")
    ear_values = [v for v in (ear_left, ear_right) if v is not None]
    ear = sum(ear_values) / len(ear_values) if ear_values else None
    gaze = feats.get("gaze", 0.0)
    yaw = feats.get("head_yaw")

    if face is None:
        state = _commit_state("face_absent", now, dwell=0.25)
        return {"score": 0.0, "state": state, "offscreen_duration": 0.0, "conf": quality["conf_base"] * 0.3}

    if calibrating:
        _state["baseline_gaze"] = gaze if _state["baseline_gaze"] is None else 0.9 * _state["baseline_gaze"] + 0.1 * gaze
        if yaw is not None:
            _state["baseline_yaw"] = yaw if _state["baseline_yaw"] is None else 0.9 * _state["baseline_yaw"] + 0.1 * yaw
        _state["state"] = "calibrating"
        return {"score": 100.0, "state": "calibrating", "offscreen_duration": 0.0, "conf": quality["conf_base"] * 0.8}

    eye_closed = ear is not None and ear < 0.18
    baseline_yaw = _state["baseline_yaw"] if _state["baseline_yaw"] is not None else (yaw or 0.0)
    yaw_delta = abs((yaw or 0.0) - baseline_yaw)
    head_turned = yaw_delta >= cfg["thresholds"].get("head_turn_yaw", 0.18)
    gaze_away = gaze <= cfg["thresholds"].get("gaze_away", 0.32)

    if eye_closed:
        proposed = "eyes_closed"
        score = 20.0
    elif head_turned:
        proposed = "head_turned"
        score = 35.0
    elif gaze_away:
        proposed = "looking_away"
        score = 45.0
    else:
        proposed = "looking_forward"
        score = max(65.0, min(100.0, 100.0 * gaze))

    state = _commit_state(proposed, now)
    off = state in {"looking_away", "head_turned", "face_absent"}
    if off and _state["offscreen_since"] is None:
        _state["offscreen_since"] = now
    if not off:
        _state["offscreen_since"] = None
    off_dur = (now - _state["offscreen_since"]) if _state["offscreen_since"] else 0.0
    if off_dur > cfg["thresholds"]["gaze_offscreen_secs"]:
        score *= 0.6

    return {
        "score": float(max(0.0, min(100.0, score))),
        "state": state,
        "offscreen_duration": off_dur,
        "head_yaw_delta": float(yaw_delta),
        "gaze_proxy": float(gaze),
        "conf": quality["conf_base"] * (1.0 if quality.get("signal_ok") else 0.55),
    }
