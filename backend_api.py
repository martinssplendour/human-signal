from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine import MonitorEngine
from engine.schemas import FeedbackRequest, FrameRequest, SessionRequest


FRONTEND_DIR = "frontend"

engine = MonitorEngine()
app = FastAPI(title="Human Signal API")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def index():
    return FileResponse("frontend/index.html")


@app.get("/api/status")
def status():
    return engine.status()


@app.post("/api/session")
def session(req: SessionRequest):
    try:
        return engine.set_session(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/calibrate")
def calibrate():
    return engine.calibrate()


@app.post("/api/feedback")
def feedback(req: FeedbackRequest):
    try:
        return engine.save_feedback(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/frame")
def process_frame(req: FrameRequest):
    try:
        return engine.process_image_request(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/frame")
async def process_frame_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                req = FrameRequest(**data)
                result = engine.process_image_request(req)
                payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
                await websocket.send_json({"ok": True, "data": payload})
            except Exception as exc:
                await websocket.send_json({"ok": False, "error": str(exc)})
    except WebSocketDisconnect:
        return
