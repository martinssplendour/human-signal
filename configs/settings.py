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
    gaze_offscreen_secs: float = Field(default=5.0, ge=0.1, le=30.0)
    stress_tension_high: float = Field(default=0.7, ge=0.0, le=1.0)
    neck_angle_slouch: float = Field(default=22.0, ge=0.0, le=90.0)
    yawn_mar: float = Field(default=0.6, ge=0.1, le=2.0)
    yawn_min_secs: float = Field(default=0.5, ge=0.1, le=10.0)
    distance_face_ratio_close: float = Field(default=0.32, ge=0.01, le=1.0)
    distance_face_ratio_far: float = Field(default=0.12, ge=0.01, le=1.0)
    fatigue_ear_closed_ratio: float = Field(default=0.70, ge=0.40, le=0.95)
    microsleep_secs: float = Field(default=1.5, ge=0.3, le=10.0)
    head_turn_yaw: float = Field(default=0.18, ge=0.05, le=0.8)
    head_turn_yaw_deg: float = Field(default=30.0, ge=5.0, le=80.0)
    gaze_away: float = Field(default=0.32, ge=0.05, le=1.0)
    gaze_offset_x: float = Field(default=0.28, ge=0.02, le=0.6)
    gaze_offset_y: float = Field(default=0.32, ge=0.02, le=0.8)

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


class CalibrationConfig(BaseModel):
    seconds: float = Field(default=10.0, ge=1.0, le=30.0)
    baseline_ema_alpha: float = Field(default=0.005, ge=0.0001, le=0.1)
    baseline_update_max_score: float = Field(default=25.0, ge=0.0, le=100.0)


class DriverAlertConfig(BaseModel):
    yawn_watch_count: int = Field(default=3, ge=1, le=20)
    yawn_elevated_count: int = Field(default=5, ge=1, le=30)
    yawn_window_secs: float = Field(default=300.0, ge=30.0, le=1800.0)
    rapid_yawn_count: int = Field(default=3, ge=1, le=20)
    rapid_yawn_window_secs: float = Field(default=120.0, ge=30.0, le=600.0)
    eye_closed_elevated_secs: float = Field(default=1.5, ge=0.3, le=10.0)
    eye_closed_critical_secs: float = Field(default=2.0, ge=0.5, le=15.0)
    perclos_watch: float = Field(default=0.10, ge=0.0, le=1.0)
    perclos_elevated: float = Field(default=0.15, ge=0.0, le=1.0)
    perclos_critical: float = Field(default=0.25, ge=0.0, le=1.0)
    blink_watch_ms: float = Field(default=200.0, ge=50.0, le=2000.0)
    blink_elevated_ms: float = Field(default=300.0, ge=50.0, le=3000.0)
    head_nod_critical_count: int = Field(default=3, ge=1, le=20)
    head_nod_window_secs: float = Field(default=120.0, ge=30.0, le=600.0)
    hydration_reminder_secs: float = Field(default=5400.0, ge=300.0, le=21600.0)
    break_reminder_secs: float = Field(default=7200.0, ge=900.0, le=28800.0)
    professional_limit_secs: float = Field(default=16200.0, ge=1800.0, le=43200.0)
    fatigue_score_elevated: float = Field(default=80.0, ge=0.0, le=100.0)
    compound_elevated_count: int = Field(default=2, ge=2, le=10)
    compound_window_secs: float = Field(default=600.0, ge=60.0, le=3600.0)
    long_trip_secs: float = Field(default=10800.0, ge=900.0, le=43200.0)
    long_trip_threshold_multiplier: float = Field(default=0.8, ge=0.5, le=1.0)


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


class ModelConfig(BaseModel):
    prefer_tasks: bool = True
    face_landmarker: str = ""
    pose_landmarker: str = ""
    classifier_bundle: str = ""


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
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    driver: DriverAlertConfig = Field(default_factory=DriverAlertConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)

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
