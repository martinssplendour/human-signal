from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.tracking import Tracker as LegacyTracker


class TrackerBackend:
    name = "base"

    def process(self, frame_rgb):
        raise NotImplementedError


class LegacyMediaPipeTracker(TrackerBackend):
    name = "mediapipe_legacy_solutions"

    def __init__(self) -> None:
        self._tracker = LegacyTracker()

    def process(self, frame_rgb):
        det = self._tracker.process(frame_rgb)
        det["tracker_backend"] = self.name
        det["blendshapes"] = {}
        det["face_transform"] = None
        return det


class MediaPipeTasksTracker(TrackerBackend):
    name = "mediapipe_tasks"

    def __init__(self, face_model_path: str | Path, pose_model_path: str | Path) -> None:
        face_path = Path(face_model_path)
        pose_path = Path(pose_model_path)
        if not face_path.exists():
            raise FileNotFoundError(f"Face Landmarker model not found: {face_path}")
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose Landmarker model not found: {pose_path}")

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        face_options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(face_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(pose_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._face = vision.FaceLandmarker.create_from_options(face_options)
        self._pose = vision.PoseLandmarker.create_from_options(pose_options)

    def process(self, frame_rgb):
        h, w, _ = frame_rgb.shape
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        face_res = self._face.detect(image)
        pose_res = self._pose.detect(image)

        face_landmarks = None
        face_confidence = None
        if face_res.face_landmarks:
            face_landmarks = np.array(
                [(lm.x * w, lm.y * h, lm.z) for lm in face_res.face_landmarks[0]],
                dtype=np.float32,
            )
            face_confidence = np.ones((face_landmarks.shape[0],), dtype=np.float32)

        pose_landmarks = None
        pose_confidence = None
        if pose_res.pose_landmarks:
            pose_landmarks = np.array(
                [(lm.x * w, lm.y * h, lm.z) for lm in pose_res.pose_landmarks[0]],
                dtype=np.float32,
            )
            pose_confidence = np.array(
                [getattr(lm, "visibility", 1.0) for lm in pose_res.pose_landmarks[0]],
                dtype=np.float32,
            )

        blendshapes = {}
        if face_res.face_blendshapes:
            blendshapes = {
                category.category_name: float(category.score)
                for category in face_res.face_blendshapes[0]
            }

        face_transform = None
        if getattr(face_res, "facial_transformation_matrixes", None):
            face_transform = np.array(face_res.facial_transformation_matrixes[0], dtype=np.float32)

        return {
            "face": face_landmarks,
            "pose": pose_landmarks,
            "size": (w, h),
            "face_confidence": face_confidence,
            "pose_confidence": pose_confidence,
            "blendshapes": blendshapes,
            "face_transform": face_transform,
            "tracker_backend": self.name,
        }


def create_tracker(cfg: dict[str, Any]) -> TrackerBackend:
    models = cfg.get("models", {})
    face_model = models.get("face_landmarker")
    pose_model = models.get("pose_landmarker")
    prefer_tasks = bool(models.get("prefer_tasks", True))
    if prefer_tasks and face_model and pose_model:
        try:
            return MediaPipeTasksTracker(face_model, pose_model)
        except Exception:
            pass
    return LegacyMediaPipeTracker()
