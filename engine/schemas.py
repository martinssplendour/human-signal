from typing import Any

from pydantic import BaseModel, Field


class HealthcareContext(BaseModel):
    patient_session_id: str = ""
    observation_type: str = "General observation"
    note: str = ""
    consent_captured: bool = False


class FrameRequest(BaseModel):
    image: str
    mode: str = "Driver"
    care_context: str = "Chair"
    healthcare: HealthcareContext = Field(default_factory=HealthcareContext)


class SessionRequest(BaseModel):
    mode: str
    action: str


class FeedbackRequest(BaseModel):
    feedback: str


class EngineStatus(BaseModel):
    ok: bool = True
    tracker_backend: str
    modes: list[str]
    sessions: dict[str, bool]


class FrameResponse(BaseModel):
    time: float
    calibrating: bool
    calibration_gate: dict[str, Any]
    metrics: dict[str, float]
    states: dict[str, str]
    quality: dict[str, Any]
    debug: dict[str, Any] = Field(default_factory=dict)
    mode_state: dict[str, Any]
    summary: dict[str, Any]
    timeline: list[dict[str, Any]]
    tracker_backend: str
