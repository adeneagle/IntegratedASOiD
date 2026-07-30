"""Training router: start/stop training, stream metrics, list checkpoints.

The endpoint paths and the SSE streaming logic are unchanged from the TSM
version — only the request schema differs, because a RandomForest active-learning
run is configured by four numbers instead of a deep-learning hyperparameter set.
"""

import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.training_service import (
    start_training,
    read_latest_metrics,
    read_all_metrics,
    stop_training,
    list_checkpoints,
    get_task,
)

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    # Data
    project_dir: str
    video_ids: List[str]
    iteration_dir: str = "iteration-0"

    # Active learning
    init_ratio: float = 0.01
    max_iter: int = 100
    max_samples_iter: int = 20
    conf_threshold: float = 0.5
    exclude_other: bool = False

    # Optional label-space collapse (mirrors the TSM app's binarize semantics,
    # so the renderer's behavior checklist keeps working unchanged).
    binarize: Optional[str] = None
    label_union: Optional[List[str]] = None

    # Output
    checkpoint_dir: Optional[str] = None
    model_name: Optional[str] = None


class CheckpointSearchRequest(BaseModel):
    search_dirs: Optional[List[str]] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/start")
async def api_start_training(req: TrainingConfig):
    """Start the active-learning training loop. Returns task_id."""
    if not req.video_ids:
        raise HTTPException(status_code=422, detail="Select at least one video to train on.")
    try:
        task_id = start_training(req.model_dump())
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_training_status(task_id: str):
    """SSE stream of training metrics."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        # Stream every event line in order (startup info messages + per-iteration
        # metrics), not just the latest, so nothing written between polls is lost.
        sent = 0
        while True:
            lines = read_all_metrics(task_id)
            while sent < len(lines):
                yield f"data: {json.dumps(lines[sent])}\n\n"
                sent += 1
            # Terminal states (error/cancelled) live on the task, not the file —
            # emit a final event carrying status + error detail, then stop.
            status = get_task(task_id).get("status")
            if status in ("completed", "error", "cancelled"):
                yield f"data: {json.dumps(read_latest_metrics(task_id))}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/stop/{task_id}")
async def api_stop_training(task_id: str):
    """Ask a running training thread to stop at the next iteration boundary."""
    success = stop_training(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already stopped")
    return {"status": "cancelled"}


@router.get("/checkpoints")
async def api_list_checkpoints(search_dirs: Optional[str] = None):
    """List available .joblib models, with their class names and body parts."""
    dirs = search_dirs.split(",") if search_dirs else None
    return {"checkpoints": list_checkpoints(dirs)}
