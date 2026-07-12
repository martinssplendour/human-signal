# Human Signal

## Quick Start

Run the local API and browser dashboard from the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

If `uvicorn` is not on your PATH, use:

```bash
python -m uvicorn backend_api:app --reload --host 127.0.0.1 --port 8000
```

In the dashboard, click **Start Camera**, allow browser camera access, choose a mode, and optionally click **Calibrate (10s)** or **Start New Session**.

## What It Is

Human Signal is a local, privacy-first wellness and observation dashboard for fatigue, attention proxy, posture, signal quality, and session review. It uses a browser webcam frontend, a local FastAPI backend, MediaPipe face/pose landmarks, and engineered signal rules.

This project is for wellness, safety-support, and observation workflows only. It is not a medical device and does not diagnose health conditions.

## Requirements

- Python 3.10+ recommended.
- A webcam available to the browser.
- A modern browser such as Chrome, Edge, or Firefox.
- Local camera permission for `http://127.0.0.1:8000`.
- No cloud service or external API key is required for the default local workflow.

## Features

- Live browser webcam monitoring with MediaPipe FaceMesh and Pose.
- Fatigue proxy tracking with EAR, rolling PERCLOS, blink duration, yawns, microsleep, and head-nod events.
- Attention proxy states: face absent, eyes closed, head turned, looking away, looking forward.
- Posture and camera-distance monitoring.
- Signal quality checks for lighting, blur, face visibility, face size, landmark confidence, and frame drops.
- Calibration quality gate before baseline capture.
- Unified event timeline and session summary dashboard.
- Privacy blur for the browser preview.
- Local event feedback capture for false alarms, acknowledgement, expected/resting, and follow-up.
- Helper utilities for feedback CSVs, audit CSVs, healthcare Markdown reports, and retention cleanup.

## First Run

1. Start the server with the Quick Start command.
2. Open `http://127.0.0.1:8000`.
3. Click **Start Camera** and approve the browser permission prompt.
4. Select a monitoring mode.
5. Click **Calibrate (10s)** if you want a fresh baseline.
6. Click **Start New Session** to reset runtime counters and begin a clean session.

The frontend sends webcam frames to the local Python API for tracking and signal scoring. It uses the WebSocket endpoint `/ws/frame` first and falls back to the HTTP `/api/frame` endpoint if the socket is unavailable.

## Modes

### Driver

Driver mode provides a fatigue-risk workflow for long driving sessions:

- risk states: `normal`, `watch`, `elevated`, `critical`, `insufficient_signal`
- events: microsleep, sustained eye closure, repeated yawning, head-nod cluster, face absent, poor signal, break recommendation
- trip timer, break timer, critical alert count, usable signal percentage
- optional critical alert sound in the browser

### Desk Ergonomics

Desk mode supports everyday posture and wellness monitoring:

- posture trend
- fatigue and attention proxy charts
- camera setup guidance
- signal-quality feedback

### Care Observation

Care observation mode supports aged-care style review:

- contexts: chair, bed, room, desk
- states: `observed`, `resting`, `insufficient_signal`, `needs_review`, `urgent_review`
- normal observations: sleep/rest observed, tired/fatigue signs observed
- review events: not in view/check for fall, resident absent, extended absence, context-aware prolonged inactivity, repeated-movement restlessness, possible fall-risk signal, possible slump, posture decline, camera/signal issue
- records: visibility, signal quality, sleep time in the current 24-hour window, fatigue state, posture state, inactivity duration, restlessness events, motion burst count, fall-risk review count, absence duration, tiredness count, review-event counts

### Healthcare Observation

Healthcare observation mode provides structured, non-diagnostic review support:

- observation types: general observation, post-procedure recovery, sleep/rest observation, rehab/session monitoring, waiting room/triage support
- patient/session metadata and notes
- consent gate before monitoring starts
- events: calibration failed, poor signal, patient out of frame, observation interrupted, prolonged eye closure, repeated fatigue signs, reduced responsiveness proxy, posture decline, restlessness
- helper support for local audit logs and Markdown report generation

For healthcare mode, fill **Patient/session ID** and check **Consent captured**. Without both values, the mode remains in `waiting_for_consent`.

## Project Structure

```text
backend_api.py             FastAPI routes, static frontend serving, frame WebSocket
configs/                   YAML config and Pydantic settings loader
core/                      pipeline, tracking features, quality, audit, session helpers
engine/                    runtime orchestration, schemas, tracker selection
frontend/                  browser camera capture and dashboard
ml/                        optional local classifier support
modules/                   scoring and mode-specific event engines
scripts/evaluate_engine.py offline image/video evaluation harness
tests/                     unit tests for config, quality, scoring, and event rules
training/                  local classifier training script
```

## Configuration

The app loads `configs/default.yaml` at startup. Edit that file to tune thresholds, calibration, quality checks, output directories, and optional model paths.

By default the engine prefers MediaPipe Tasks if local model files are configured:

```yaml
models:
  prefer_tasks: true
  face_landmarker: "models/face_landmarker.task"
  pose_landmarker: "models/pose_landmarker.task"
```

If those files are absent, the app falls back to the compatible legacy MediaPipe FaceMesh/Pose tracker so local startup still works.

Optional trained classifiers can be enabled with:

```yaml
models:
  classifier_bundle: "models/classifiers.joblib"
```

If no classifier bundle is configured, the engine keeps using the rule-based scorers.

## API Surface

The local backend exposes:

- `GET /` - serves the browser dashboard.
- `GET /api/status` - returns tracker backend, available modes, and session state.
- `POST /api/session` - starts or stops a mode session.
- `POST /api/calibrate` - starts the configured calibration window.
- `POST /api/feedback` - saves feedback for the latest event to a local CSV.
- `POST /api/frame` - analyzes a base64 image frame.
- `WS /ws/frame` - analyzes frames over WebSocket.

## Local Data

The current dashboard does not save raw webcam video by default. Runtime state and the event timeline are kept in memory while the server is running.

Local files can be written by specific workflows:

- `recordings/event_feedback_*.csv` when **Save Feedback** is clicked after an event.
- `logs/audit.csv` when audit helper functions are called.
- healthcare Markdown reports when `core.audit.write_healthcare_report` is used.
- `datasets/eval_results.csv` or another path when the evaluator is run with `--output`.
- `models/classifiers.joblib` or another path when local classifiers are trained.

`recordings/`, `datasets/`, and `logs/` are intended for local runtime data and are ignored by git.

## Evaluate Engine

Run the engine over image files, video files, or directories:

```bash
python scripts/evaluate_engine.py datasets/eval --mode Driver --output datasets/eval_results.csv
```

Folder names are treated as labels in the CSV output. For videos, use `--stride` to control how often frames are sampled:

```bash
python scripts/evaluate_engine.py videos/sample.mp4 --mode "Care observation" --care-context Chair --stride 15 --output datasets/eval_results.csv
```

## Train Local Classifiers

The trainer reads `dataset_*.csv` files from the provided directories:

```bash
python training/train_classifiers.py datasets --output models/classifiers.joblib
```

Each target needs at least two labeled classes before a classifier can be trained. After training, configure `models.classifier_bundle` in `configs/default.yaml`.

## Tests

Run the unit test suite with:

```bash
python -m unittest discover -v
```

The suite covers config validation, quality checks, fatigue/attention scoring, driver events, care observation events, healthcare events, and compact session summaries.

## Troubleshooting

### Camera Permission Fails

- Open the app at `http://127.0.0.1:8000`, not from the filesystem.
- Allow camera access in the browser permission prompt.
- Close other apps that may be using the webcam.
- Try another browser if the camera device is not listed.

### Server Starts But Dashboard Does Not Load

- Confirm the server is running on `127.0.0.1:8000`.
- Check that you started it from the project root.
- If port `8000` is busy, start on another port:

```bash
uvicorn backend_api:app --reload --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001`.

### MediaPipe Or Protobuf Warning Appears

Some MediaPipe/protobuf combinations print a warning similar to:

```text
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
```

If the server continues running or the tests finish with `OK`, this warning is non-fatal.

### Poor Signal Or Calibration Fails

- Improve lighting.
- Keep the face visible and centered.
- Avoid strong backlighting.
- Keep the camera stable.
- Re-run **Calibrate (10s)** after changing camera position.

## Safety and Privacy Notes

- Processing runs locally in the default setup.
- The browser preview can be blurred with **Privacy Blur**.
- Raw webcam video is not persisted by the default backend workflow.
- Saved feedback, reports, audit logs, datasets, and model files stay on the local machine unless you move or publish them.
- Scores and events are heuristic proxies and can be wrong under poor lighting, occlusion, unusual camera angles, or motion.
- Do not use this software as the sole basis for driving, workplace, care, or clinical decisions.
