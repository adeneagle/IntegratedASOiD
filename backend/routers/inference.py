"""Inference router: run per-frame inference, stream progress, return results.

Endpoint paths, the SSE status stream and the results payload are unchanged from
the TSM version — the renderer's Active Learning panel consumes them as-is.
Only the run request changed: a pose file replaces video + background + crop
params.
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.inference_service import (
    start_inference,
    start_batch_inference,
    get_task,
)

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    checkpoint_path: str
    pose_path: str
    origin: str = "deeplabcut"
    # Defaults to the selection recorded on the checkpoint when omitted.
    bodyparts: Optional[list[str]] = None
    llh_value: float = 0.6
    animals: Optional[list[str]] = None
    multi_animal: bool = False
    # Optional scoping: inclusive [start, end] native-index pairs.
    # Omit/None to run on the entire recording.
    frame_ranges: Optional[list] = None


class BatchInferenceRequest(BaseModel):
    checkpoint_path: str
    pose_paths: list[str]
    output_dir: str
    origin: str = "deeplabcut"
    bodyparts: Optional[list[str]] = None
    llh_value: float = 0.6
    animals: Optional[list[str]] = None
    multi_animal: bool = False


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/run")
async def api_run_inference(req: InferenceRequest):
    """Start inference in background. Returns task_id."""
    try:
        task_id = start_inference(
            checkpoint_path=req.checkpoint_path,
            pose_path=req.pose_path,
            origin=req.origin,
            bodyparts=req.bodyparts,
            llh_value=req.llh_value,
            animals=req.animals,
            multi_animal=req.multi_animal,
            frame_ranges=req.frame_ranges,
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def api_run_batch_inference(req: BatchInferenceRequest):
    """Apply a model to several unlabeled pose files. Returns task_id."""
    if not req.pose_paths:
        raise HTTPException(status_code=422, detail="No pose files selected.")
    try:
        task_id = start_batch_inference(
            checkpoint_path=req.checkpoint_path,
            pose_paths=req.pose_paths,
            output_dir=req.output_dir,
            origin=req.origin,
            bodyparts=req.bodyparts,
            llh_value=req.llh_value,
            animals=req.animals,
            multi_animal=req.multi_animal,
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_inference_status(task_id: str):
    """SSE stream of inference progress."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_stream():
        while True:
            data = json.dumps({
                "status": task["status"],
                "progress": task["progress"],
                "total": task["total"],
                "error": task["error"],
            })
            yield f"data: {data}\n\n"
            if task["status"] in ("completed", "error", "cancelled"):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/results/{task_id}")
async def api_inference_results(task_id: str):
    """Return inference results (per-frame predictions + uncertainty)."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Inference not complete (status: {task['status']})"
        )

    results = task.get("results")
    if results is None:
        raise HTTPException(status_code=500, detail="No results available")

    return results


@router.post("/stop/{task_id}")
async def api_stop_inference(task_id: str):
    """Cancel a running inference task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task["cancel"] = True
    return {"status": "cancelling"}
