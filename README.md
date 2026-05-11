# Human Signal

Human Signal is a local, privacy-first wellness and observation dashboard for fatigue, attention proxy, posture, signal quality, and session review. It uses a webcam with MediaPipe face/pose landmarks, engineered signals, and Streamlit dashboards.

This project is for wellness, safety-support, and observation workflows only. It is not a medical device and does not diagnose health conditions.

## Features

- Live webcam monitoring with MediaPipe FaceMesh and Pose.
- Fatigue proxy tracking with EAR, rolling PERCLOS, blink duration, yawns, microsleep, and head-nod events.
- Attention proxy states: face absent, eyes closed, head turned, looking away, looking forward.
- Posture and camera-distance monitoring.
- Signal quality checks for lighting, blur, face visibility, face size, landmark confidence, and frame drops.
- Calibration quality gate before baseline capture.
- Unified event timeline and session summary dashboard.
- Privacy preview blur, metadata-only mode, retention cleanup, and local audit logs.
- Event feedback capture for false alarms, acknowledgement, expected/resting, and follow-up.
- CSV/Markdown exports for sessions, events, and reports.

## Modes

### Driver

Driver mode provides a fatigue-risk workflow for long driving sessions:

- risk states: `normal`, `watch`, `elevated`, `critical`, `insufficient_signal`
- events: microsleep, sustained eye closure, repeated yawning, head-nod cluster, looking away, face absent, poor signal, break recommendation
- trip timer, break timer, critical alert count, usable signal percentage
- optional critical alert sound

### Care Observation

Care observation mode supports aged-care style review:

- contexts: chair, bed, room, desk
- states: `observed`, `resting`, `insufficient_signal`, `needs_review`, `urgent_review`
- events: resident absent, extended absence, prolonged inactivity, possible slump, restlessness, posture decline, camera/signal issue

### Healthcare Observation

Healthcare observation mode provides structured, non-diagnostic review support:

- observation types: general observation, post-procedure recovery, sleep/rest observation, rehab/session monitoring, waiting room/triage support
- optional patient/session metadata and notes
- events: calibration failed, poor signal, patient out of frame, observation interrupted, prolonged eye closure, repeated fatigue signs, reduced responsiveness proxy, posture decline, restlessness
- local audit log and Markdown report export

### Desk Ergonomics

Desk mode supports everyday posture and wellness monitoring:

- posture trend
- fatigue and attention proxy charts
- camera setup guidance
- signal-quality feedback

## Project Structure

```text
app.py                  Streamlit application
configs/                YAML config and Pydantic settings loader
core/                   pipeline, tracking, features, quality, audit, session helpers
modules/                scoring and mode-specific event engines
ui/                     overlays and recording/dashboard composition
tests/                  unit tests for config, quality, scoring, and event rules
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Tests

```bash
python -m unittest discover -v
```

## Local Data

The app writes runtime outputs locally:

- `recordings/` for event CSVs, optional recordings, and reports
- `datasets/` for consented feature datasets
- `logs/` for application and audit logs

These folders are ignored by git by default.

## Safety and Privacy Notes

- Metadata-only mode is enabled by default for safer local use.
- Raw video recording should only be enabled with explicit consent.
- Scores and events are proxies derived from webcam signals and can be wrong under poor lighting, occlusion, unusual camera angles, or motion.
- Do not use this software as the sole basis for driving, workplace, care, or clinical decisions.
