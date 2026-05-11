from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field, root_validator, validator


class VideoConfig(BaseModel):
    source: int = 0
    fps: int = Field(default=30, ge=1, le=120)
    width: int = Field(default=1280, ge=320, le=3840)
    height: int = Field(default=720, ge=240, le=2160)
    fallback_sources: list[int] = Field(default_factory=lambda: [0, 1, 2])


class WindowConfig(BaseModel):
    fatigue_seconds: int = Field(default=60, ge=10, le=600)
    attention_seconds: int = Field(default=30, ge=5, le=300)
    stress_seconds: int = Field(default=30, ge=5, le=300)
    update_hz: int = Field(default=5, ge=1, le=30)


class ThresholdConfig(BaseModel):
    perclos_drowsy: float = Field(default=0.25, ge=0.0, le=1.0)
    blink_rate_high: int = Field(default=25, ge=1, le=120)
    gaze_offscreen_secs: float = Field(default=3.0, ge=0.1, le=30.0)
    stress_tension_high: float = Field(default=0.7, ge=0.0, le=1.0)
    neck_angle_slouch: float = Field(default=22.0, ge=0.0, le=90.0)
    yawn_mar: float = Field(default=0.6, ge=0.1, le=2.0)
    yawn_min_secs: float = Field(default=0.5, ge=0.1, le=10.0)
    distance_face_ratio_close: float = Field(default=0.32, ge=0.01, le=1.0)
    distance_face_ratio_far: float = Field(default=0.12, ge=0.01, le=1.0)
    fatigue_ear_closed_ratio: float = Field(default=0.70, ge=0.40, le=0.95)
    microsleep_secs: float = Field(default=1.5, ge=0.3, le=10.0)
    head_turn_yaw: float = Field(default=0.18, ge=0.05, le=0.8)
    gaze_away: float = Field(default=0.32, ge=0.05, le=1.0)

    @validator("distance_face_ratio_close")
    def close_must_exceed_far(cls, value: float, values: Dict[str, Any]) -> float:
        far = values.get("distance_face_ratio_far")
        if far is not None and value <= far:
            raise ValueError("distance_face_ratio_close must be greater than distance_face_ratio_far")
        return value

    @root_validator(skip_on_failure=True)
    def distance_thresholds_must_be_ordered(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["distance_face_ratio_close"] <= values["distance_face_ratio_far"]:
            raise ValueError("distance_face_ratio_close must be greater than distance_face_ratio_far")
        return values


class SmoothingConfig(BaseModel):
    ema_alpha: float = Field(default=0.25, ge=0.01, le=1.0)


class QualityConfig(BaseModel):
    min_brightness: float = Field(default=60.0, ge=0.0, le=255.0)
    max_motion_px: float = Field(default=5.0, ge=0.0, le=100.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_sharpness: float = Field(default=35.0, ge=0.0, le=1000.0)
    min_face_landmark_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    min_pose_landmark_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    max_frame_gap_secs: float = Field(default=0.5, ge=0.05, le=5.0)


class RecordingConfig(BaseModel):
    output_dir: str = "recordings"
    fps_min: int = Field(default=8, ge=1, le=60)
    fps_max: int = Field(default=24, ge=1, le=60)


class DatasetConfig(BaseModel):
    output_dir: str = "datasets"


class FusionWeights(BaseModel):
    fatigue: float = Field(default=0.4, ge=0.0, le=1.0)
    attention: float = Field(default=0.35, ge=0.0, le=1.0)
    stress: float = Field(default=0.25, ge=0.0, le=1.0)


class FusionConfig(BaseModel):
    weights: FusionWeights = Field(default_factory=FusionWeights)


class AppConfig(BaseModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    windows: WindowConfig = Field(default_factory=WindowConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    def as_legacy_dict(self) -> Dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    cfg_path = Path(path)
    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    return AppConfig(**raw)
