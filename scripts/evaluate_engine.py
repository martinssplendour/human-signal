import argparse
import csv
import json
from pathlib import Path

import cv2

from engine import MonitorEngine
from engine.schemas import FrameRequest, HealthcareContext


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def iter_samples(paths, stride):
    for root in paths:
        path = Path(root)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                yield from iter_samples([child], stride)
        elif path.suffix.lower() in IMAGE_EXTS:
            frame = cv2.imread(str(path))
            if frame is not None:
                yield str(path), label_for(path), frame
        elif path.suffix.lower() in VIDEO_EXTS:
            cap = cv2.VideoCapture(str(path))
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    yield f"{path}#{idx}", label_for(path), frame
                idx += 1
            cap.release()


def label_for(path):
    parent = path.parent.name
    return parent if parent and parent != "." else ""


def run_eval(args):
    engine = MonitorEngine()
    req = FrameRequest(
        image="",
        mode=args.mode,
        care_context=args.care_context,
        healthcare=HealthcareContext(consent_captured=True, patient_session_id="eval"),
    )
    rows = []
    for sample_id, label, frame in iter_samples(args.inputs, args.stride):
        result = engine.process_frame(frame, req)
        data = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        rows.append({
            "sample": sample_id,
            "label": label,
            "mode": args.mode,
            "tracker_backend": data["tracker_backend"],
            "fatigue": data["metrics"]["fatigue"],
            "attention": data["metrics"]["attention"],
            "tension": data["metrics"]["tension"],
            "readiness": data["metrics"]["readiness"],
            "fatigue_state": data["states"]["fatigue"],
            "attention_state": data["states"]["attention"],
            "posture_state": data["states"]["posture"],
            "signal_ok": data["quality"]["signal_ok"],
            "face_present": data["quality"]["face_present"],
            "summary_state": data["summary"].get("state"),
            "events": len(data["timeline"]),
        })
    write_rows(rows, args.output)
    print(json.dumps(summarize(rows), indent=2))


def write_rows(rows, output):
    if not output:
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["sample"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    total = len(rows)
    by_label = {}
    for row in rows:
        label = row["label"] or "unlabeled"
        bucket = by_label.setdefault(label, {"count": 0, "attention_states": {}, "fatigue_states": {}})
        bucket["count"] += 1
        bucket["attention_states"][row["attention_state"]] = bucket["attention_states"].get(row["attention_state"], 0) + 1
        bucket["fatigue_states"][row["fatigue_state"]] = bucket["fatigue_states"].get(row["fatigue_state"], 0) + 1
    return {"samples": total, "labels": by_label}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Human Signal engine evaluation over image folders or videos.")
    parser.add_argument("inputs", nargs="+", help="Image, video, or directory paths.")
    parser.add_argument("--mode", default="Driver", help="Engine mode to evaluate.")
    parser.add_argument("--care-context", default="Chair")
    parser.add_argument("--stride", type=int, default=15, help="Video frame stride.")
    parser.add_argument("--output", default="datasets/eval_results.csv")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
