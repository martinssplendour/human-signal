from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FEATURE_COLUMNS = [
    "ear_left",
    "ear_right",
    "mar",
    "gaze",
    "gaze_x",
    "gaze_y",
    "head_yaw",
    "head_yaw_deg",
    "head_pitch_deg",
    "neck_angle",
    "brightness",
    "sharpness",
]


STATE_SCORE = {
    "fatigue": {
        "alert": 10.0,
        "fatigue_signs": 55.0,
        "drowsy": 80.0,
        "microsleep": 98.0,
        "insufficient_signal": 0.0,
        "calibrating": 0.0,
    },
    "attention": {
        "looking_forward": 95.0,
        "looking_away": 45.0,
        "head_turned": 35.0,
        "eyes_closed": 20.0,
        "face_absent": 0.0,
        "calibrating": 100.0,
    },
    "tension": {
        "neutral": 5.0,
        "moderate_tension": 50.0,
        "elevated_tension": 85.0,
        "insufficient_signal": 0.0,
        "calibrating": 0.0,
    },
}


class SignalClassifierBundle:
    def __init__(self, models=None, feature_columns=None) -> None:
        self.models = models or {}
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SignalClassifierBundle":
        model_path = cfg.get("models", {}).get("classifier_bundle", "")
        if not model_path:
            return cls()
        path = Path(model_path)
        if not path.exists():
            return cls()
        try:
            import joblib

            payload = joblib.load(path)
            return cls(models=payload.get("models", {}), feature_columns=payload.get("feature_columns", DEFAULT_FEATURE_COLUMNS))
        except Exception:
            return cls()

    @property
    def enabled(self) -> bool:
        return bool(self.models)

    def apply(self, feats: dict[str, Any], quality: dict[str, Any], outputs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if not self.enabled or not quality.get("face_present", False):
            return outputs
        x = self._vector(feats, quality)
        for signal_name, output_key in [("fatigue", "fatigue"), ("attention", "attention"), ("tension", "tension")]:
            model = self.models.get(signal_name)
            if model is None or output_key not in outputs:
                continue
            state, conf = self._predict(model, x)
            if state is None:
                continue
            score = STATE_SCORE.get(signal_name, {}).get(state)
            if score is None:
                continue
            updated = dict(outputs[output_key])
            updated["state"] = state
            updated["score"] = score
            updated["conf"] = max(float(updated.get("conf", 0.0)), conf)
            updated["ml_state"] = state
            updated["ml_conf"] = conf
            outputs[output_key] = updated
        return outputs

    def _vector(self, feats: dict[str, Any], quality: dict[str, Any]):
        row = {**feats, **quality}
        values = []
        for col in self.feature_columns:
            value = row.get(col)
            if value is None or isinstance(value, (dict, list, tuple)):
                value = 0.0
            values.append(float(value))
        return np.asarray(values, dtype=np.float32).reshape(1, -1)

    def _predict(self, model, x):
        try:
            state = str(model.predict(x)[0])
            conf = 0.5
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)[0]
                conf = float(np.max(proba))
            return state, conf
        except Exception:
            return None, 0.0
