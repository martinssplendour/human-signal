import cv2
import numpy as np


def estimate_quality(frame_rgb, det, feats, cfg):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    face = det.get("face")
    pose = det.get("pose")
    face_conf = det.get("face_confidence")
    pose_conf = det.get("pose_confidence")
    face_present = face is not None
    pose_present = pose is not None
    face_landmark_conf = float(np.nanmean(face_conf)) if face_conf is not None else 0.0
    pose_landmark_conf = float(np.nanmean(pose_conf)) if pose_conf is not None else 0.0
    face_ratio = 0.0

    if face is not None:
        xs, ys = face[:, 0], face[:, 1]
        fw, fh = xs.max() - xs.min(), ys.max() - ys.min()
        w, h = det.get("size", (1, 1))
        face_ratio = float((fw * fh) / max(1.0, w * h))

    quality_cfg = cfg["quality"]
    bright_ok = brightness >= quality_cfg["min_brightness"]
    sharp_ok = sharpness >= quality_cfg.get("min_sharpness", 35.0)
    face_ok = face_present and face_landmark_conf >= quality_cfg.get("min_face_landmark_confidence", 0.85)

    reasons = []
    if not face_present:
        reasons.append("face not detected")
    if not bright_ok:
        reasons.append("low lighting")
    if not sharp_ok:
        reasons.append("blur or camera motion")
    if face_present and face_ratio < 0.01:
        reasons.append("face too small")

    conf_base = 1.0
    if not face_ok:
        conf_base *= 0.45
    if not bright_ok:
        conf_base *= 0.55
    if not sharp_ok:
        conf_base *= 0.70

    return {
        "brightness": brightness,
        "sharpness": sharpness,
        "face_present": face_present,
        "pose_present": pose_present,
        "face_landmark_conf": face_landmark_conf,
        "pose_landmark_conf": pose_landmark_conf,
        "face_ratio": face_ratio,
        "signal_ok": face_ok and bright_ok and sharp_ok,
        "reasons": reasons,
        "conf_base": float(max(0.0, min(1.0, conf_base))),
    }
