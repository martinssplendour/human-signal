import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List


def calibration_gate_status(quality: Dict, cfg: Dict) -> Dict:
    checks = {
        "face visible": bool(quality.get("face_present")),
        "lighting OK": quality.get("brightness", 0.0) >= cfg["quality"]["min_brightness"],
        "sharpness OK": quality.get("sharpness", 0.0) >= cfg["quality"].get("min_sharpness", 35.0),
        "distance OK": quality.get("face_ratio", 0.0) >= 0.01,
    }
    return {"ok": all(checks.values()), "checks": checks}


def append_event_feedback(path: str | Path, feedback: Dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"t_epoch": time.time(), **feedback}
    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def remove_old_session_files(paths: Iterable[str | Path], retention_days: int, now: float | None = None) -> List[str]:
    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    removed = []
    for root in paths:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.glob("*"):
            if not path.is_file():
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(str(path))
    return removed


def compact_summary(mode: str, driver_state: Dict, care_state: Dict, healthcare_state: Dict, generic: Dict) -> Dict:
    if mode == "Driver":
        summary = driver_state.get("summary", {})
        return {
            "mode": mode,
            "state": driver_state.get("risk", "normal"),
            "duration_secs": summary.get("trip_secs", 0.0),
            "usable_signal_pct": summary.get("usable_signal_pct", 0.0),
            "review_events": summary.get("critical_alerts", 0),
            "visible_pct": None,
        }
    if mode == "Care observation":
        summary = care_state.get("summary", {})
        return {
            "mode": mode,
            "state": care_state.get("state", "observed"),
            "duration_secs": summary.get("session_secs", 0.0),
            "usable_signal_pct": summary.get("usable_signal_pct", 0.0),
            "review_events": summary.get("needs_review_events", 0) + summary.get("urgent_review_events", 0),
            "visible_pct": summary.get("visible_pct", 0.0),
        }
    if mode == "Healthcare observation":
        summary = healthcare_state.get("summary", {})
        return {
            "mode": mode,
            "state": healthcare_state.get("state", "stable_observation"),
            "duration_secs": summary.get("session_secs", 0.0),
            "usable_signal_pct": summary.get("usable_signal_pct", 0.0),
            "review_events": summary.get("needs_review_events", 0) + summary.get("urgent_review_events", 0),
            "visible_pct": summary.get("visible_pct", 0.0),
        }
    return generic
