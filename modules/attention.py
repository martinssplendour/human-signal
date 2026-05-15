import time


_state = {
    "baseline_gaze": None,
    "baseline_gaze_x": None,
    "baseline_gaze_y": None,
    "baseline_yaw": None,
    "baseline_yaw_deg": None,
    "candidate": None,
    "candidate_since": 0.0,
    "state": "calibrating",
    "offscreen_since": None,
}


def reset():
    _state.update({
        "baseline_gaze": None,
        "baseline_gaze_x": None,
        "baseline_gaze_y": None,
        "baseline_yaw": None,
        "baseline_yaw_deg": None,
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
    eye_closed_score = feats.get("eye_closed_score")
    gaze = feats.get("gaze", 0.0)
    gaze_x = feats.get("gaze_x")
    gaze_y = feats.get("gaze_y")
    gaze_confidence = float(feats.get("gaze_confidence") or 0.0)
    yaw = feats.get("head_yaw")
    yaw_deg = feats.get("head_yaw_deg")

    if face is None:
        state = _commit_state("face_absent", now, dwell=0.25)
        return {"score": 0.0, "state": state, "offscreen_duration": 0.0, "conf": quality["conf_base"] * 0.3}

    if calibrating:
        _state["baseline_gaze"] = gaze if _state["baseline_gaze"] is None else 0.9 * _state["baseline_gaze"] + 0.1 * gaze
        if gaze_x is not None:
            _state["baseline_gaze_x"] = gaze_x if _state["baseline_gaze_x"] is None else 0.9 * _state["baseline_gaze_x"] + 0.1 * gaze_x
        if gaze_y is not None:
            _state["baseline_gaze_y"] = gaze_y if _state["baseline_gaze_y"] is None else 0.9 * _state["baseline_gaze_y"] + 0.1 * gaze_y
        if yaw is not None:
            _state["baseline_yaw"] = yaw if _state["baseline_yaw"] is None else 0.9 * _state["baseline_yaw"] + 0.1 * yaw
        if yaw_deg is not None:
            _state["baseline_yaw_deg"] = yaw_deg if _state["baseline_yaw_deg"] is None else 0.9 * _state["baseline_yaw_deg"] + 0.1 * yaw_deg
        _state["state"] = "calibrating"
        return {"score": 100.0, "state": "calibrating", "offscreen_duration": 0.0, "conf": quality["conf_base"] * 0.8}

    eye_closed = (ear is not None and ear < 0.18) or (eye_closed_score is not None and eye_closed_score >= 0.55)
    baseline_yaw = _state["baseline_yaw"] if _state["baseline_yaw"] is not None else (yaw or 0.0)
    yaw_delta = abs((yaw or 0.0) - baseline_yaw)
    baseline_yaw_deg = _state["baseline_yaw_deg"] if _state["baseline_yaw_deg"] is not None else (yaw_deg or 0.0)
    yaw_delta_deg = abs((yaw_deg or 0.0) - baseline_yaw_deg) if yaw_deg is not None else None
    attention_signal_ok = bool(quality.get("signal_ok")) and gaze_confidence >= 0.8
    head_turned = attention_signal_ok and (
        (yaw_delta_deg is not None and yaw_delta_deg >= cfg["thresholds"].get("head_turn_yaw_deg", 30.0))
        or yaw_delta >= cfg["thresholds"].get("head_turn_yaw", 0.18)
    )
    base_gaze_x = _state["baseline_gaze_x"] if _state["baseline_gaze_x"] is not None else 0.0
    base_gaze_y = _state["baseline_gaze_y"] if _state["baseline_gaze_y"] is not None else 0.0
    gaze_x_delta = abs((gaze_x or 0.0) - base_gaze_x) if gaze_x is not None else 0.0
    gaze_y_delta = abs((gaze_y or 0.0) - base_gaze_y) if gaze_y is not None else 0.0
    gaze_away = attention_signal_ok and (
        gaze_x_delta >= cfg["thresholds"].get("gaze_offset_x", 0.18)
        or gaze_y_delta >= cfg["thresholds"].get("gaze_offset_y", 0.22)
        or gaze <= cfg["thresholds"].get("gaze_away", 0.45)
    )

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
    if quality.get("signal_ok") and state == "looking_forward" and score >= 65.0:
        alpha = cfg.get("calibration", {}).get("baseline_ema_alpha", 0.005)
        if gaze_x is not None:
            _state["baseline_gaze_x"] = gaze_x if _state["baseline_gaze_x"] is None else (1 - alpha) * _state["baseline_gaze_x"] + alpha * gaze_x
        if gaze_y is not None:
            _state["baseline_gaze_y"] = gaze_y if _state["baseline_gaze_y"] is None else (1 - alpha) * _state["baseline_gaze_y"] + alpha * gaze_y
        if yaw_deg is not None:
            _state["baseline_yaw_deg"] = yaw_deg if _state["baseline_yaw_deg"] is None else (1 - alpha) * _state["baseline_yaw_deg"] + alpha * yaw_deg

    return {
        "score": float(max(0.0, min(100.0, score))),
        "state": state,
        "offscreen_duration": off_dur,
        "head_yaw_delta": float(yaw_delta),
        "head_yaw_delta_deg": float(yaw_delta_deg) if yaw_delta_deg is not None else None,
        "gaze_proxy": float(gaze),
        "gaze_x": float(gaze_x) if gaze_x is not None else None,
        "gaze_y": float(gaze_y) if gaze_y is not None else None,
        "gaze_x_delta": float(gaze_x_delta),
        "gaze_y_delta": float(gaze_y_delta),
        "gaze_confidence": float(gaze_confidence),
        "attention_signal_ok": attention_signal_ok,
        "eye_closed_score": float(eye_closed_score) if eye_closed_score is not None else None,
        "conf": quality["conf_base"] * (1.0 if quality.get("signal_ok") else 0.55),
    }
