"""Pose router: inspect pose files, extract training features, serve keypoints.

Replaces the TSM app's `preprocessing` router. Same task/SSE conventions.
"""

import asyncio
import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.pose_service import (
    get_task,
    inspect,
    keypoints,
    start_extraction,
)

router = APIRouter()


# ── Request schemas ────────────────────────────────────────────────────────

class InspectRequest(BaseModel):
    pose_path: str
    origin: str = "deeplabcut"
    multi_animal: bool = False


class KeypointsRequest(BaseModel):
    pose_path: str
    origin: str = "deeplabcut"
    bodyparts: List[str]
    animals: Optional[List[str]] = None
    multi_animal: bool = False
    start: int = 0
    end: Optional[int] = None


class VideoSource(BaseModel):
    id: str
    pose_path: str
    label_path: Optional[str] = None


class ExtractRequest(BaseModel):
    project_dir: str
    videos: List[VideoSource]
    origin: str = "deeplabcut"
    bodyparts: List[str]
    framerate: float = 30.0
    min_duration: float = 0.1
    llh_value: float = 0.6
    animals: Optional[List[str]] = None
    multi_animal: bool = False
    iteration_dir: str = "iteration-0"
    keep_classes: Optional[List[str]] = None


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/inspect")
async def api_inspect_pose(req: InspectRequest):
    """Animals, body parts, frame count and dimensionality of a pose file."""
    try:
        return inspect(req.pose_path, req.origin, req.multi_animal)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@router.post("/keypoints")
async def api_keypoints(req: KeypointsRequest):
    """Per-frame keypoints for the video overlay (unfiltered, column-major)."""
    try:
        return keypoints(
            req.pose_path, req.origin, req.bodyparts,
            animals=req.animals, multi_animal=req.multi_animal,
            start=req.start, end=req.end,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")


@router.post("/extract-features")
async def api_extract_features(req: ExtractRequest):
    """Start feature extraction in the background. Returns task_id."""
    if not req.bodyparts:
        raise HTTPException(status_code=422, detail="Select at least two body parts.")
    if len(req.bodyparts) < 2:
        raise HTTPException(
            status_code=422,
            detail="Features are built from body-part pairs — select at least two.",
        )
    if not req.videos:
        raise HTTPException(status_code=422, detail="No videos selected.")
    try:
        task_id = start_extraction(
            project_dir=req.project_dir,
            videos=[v.model_dump() for v in req.videos],
            origin=req.origin,
            bodyparts=req.bodyparts,
            framerate=req.framerate,
            min_duration=req.min_duration,
            llh_value=req.llh_value,
            animals=req.animals,
            multi_animal=req.multi_animal,
            iteration_dir=req.iteration_dir,
            keep_classes=req.keep_classes,
        )
        return {"task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def api_extract_status(task_id: str):
    """SSE stream of feature-extraction progress."""
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
async def api_extract_results(task_id: str):
    """Per-video row/class counts from a completed extraction."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Extraction not complete (status: {task['status']})",
        )
    return task.get("results") or {}


@router.post("/stop/{task_id}")
async def api_stop_extraction(task_id: str):
    """Cancel a running extraction."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    task["cancel"] = True
    return {"status": "cancelling"}
