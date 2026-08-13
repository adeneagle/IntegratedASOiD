"""HTTP client for the FastAPI backend.

One method per endpoint the Electron renderer used (see LabelingApp/core/*.js
and plugins/pipeline.js), matching the same request/response contract so the
backend needs no changes. Long-running jobs (extraction/train/inference)
return a task_id; progress is consumed by iterating the corresponding
*_status generator, which parses the backend's Server-Sent Events into plain
dicts.
"""

from __future__ import annotations

import json
import time
from typing import Iterator, Optional

import httpx


class BackendError(RuntimeError):
    """A backend request failed (transport error or non-2xx response)."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _extract_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return resp.text or f"HTTP {resp.status_code}"
    if isinstance(data, dict) and "detail" in data:
        d = data["detail"]
        return d if isinstance(d, str) else json.dumps(d)
    return json.dumps(data)


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BackendClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- low-level
    def _request(self, method: str, path: str, **kw) -> dict:
        try:
            resp = self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise BackendError(f"{method} {path} failed: {e}") from e
        if resp.status_code >= 400:
            detail = _extract_detail(resp)
            raise BackendError(
                f"{method} {path} -> {resp.status_code}: {detail}",
                status_code=resp.status_code, detail=detail,
            )
        return resp.json() if resp.content else {}

    def _post(self, path: str, payload: Optional[dict] = None) -> dict:
        return self._request("POST", path, json=payload or {})

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._request("GET", path, params=params)

    def _stream(self, method: str, path: str, payload: Optional[dict] = None) -> Iterator[dict]:
        """Yield parsed dicts from an SSE endpoint until the server closes it."""
        try:
            with self._client.stream(method, path, json=payload) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    detail = _extract_detail(resp)
                    raise BackendError(
                        f"{method} {path} -> {resp.status_code}: {detail}",
                        status_code=resp.status_code, detail=detail,
                    )
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        raw = line[5:].strip()
                        if raw:
                            yield json.loads(raw)
        except httpx.HTTPError as e:
            raise BackendError(f"{method} {path} stream failed: {e}") from e

    # --------------------------------------------------------------- health
    def health(self) -> dict:
        return self._get("/api/health")

    def wait_healthy(self, timeout: float = 30.0, interval: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.health().get("status") == "ok":
                    return True
            except BackendError:
                pass
            time.sleep(interval)
        return False

    # ------------------------------------------------------------------ pose
    def inspect_pose(self, pose_path: str, origin: str = "deeplabcut",
                     multi_animal: bool = False) -> dict:
        return self._post("/api/pose/inspect", {
            "pose_path": pose_path, "origin": origin, "multi_animal": multi_animal,
        })

    def pose_keypoints(self, pose_path: str, bodyparts: list, origin: str = "deeplabcut",
                       animals: Optional[list] = None, multi_animal: bool = False,
                       start: int = 0, end: Optional[int] = None) -> dict:
        return self._post("/api/pose/keypoints", {
            "pose_path": pose_path, "origin": origin, "bodyparts": bodyparts,
            "animals": animals, "multi_animal": multi_animal, "start": start, "end": end,
        })

    def extract_features(self, request: dict) -> dict:
        """`request` is the flat dict from pipeline_config.build_extract_request()."""
        return self._post("/api/pose/extract-features", request)

    def extraction_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/pose/status/{task_id}")

    def extraction_results(self, task_id: str) -> dict:
        return self._get(f"/api/pose/results/{task_id}")

    def stop_extraction(self, task_id: str) -> dict:
        return self._post(f"/api/pose/stop/{task_id}")

    # ------------------------------------------------------------- training
    def start_training(self, config: dict) -> dict:
        """`config` is the flat TrainingConfig dict from build_training_config()."""
        return self._post("/api/training/start", config)

    def training_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/training/status/{task_id}")

    def stop_training(self, task_id: str) -> dict:
        return self._post(f"/api/training/stop/{task_id}")

    def list_checkpoints(self, search_dirs: Optional[list] = None) -> dict:
        params = {"search_dirs": ",".join(search_dirs)} if search_dirs else None
        return self._get("/api/training/checkpoints", params=params)

    # ------------------------------------------------------------ inference
    def run_inference(self, request: dict) -> dict:
        """`request` is the flat dict from pipeline_config.build_inference_request()."""
        return self._post("/api/inference/run", request)

    def batch_inference(self, request: dict) -> dict:
        """`request` is the flat dict from pipeline_config.build_batch_request()."""
        return self._post("/api/inference/batch", request)

    def inference_status(self, task_id: str) -> Iterator[dict]:
        return self._stream("GET", f"/api/inference/status/{task_id}")

    def inference_results(self, task_id: str) -> dict:
        return self._get(f"/api/inference/results/{task_id}")

    def stop_inference(self, task_id: str) -> dict:
        return self._post(f"/api/inference/stop/{task_id}")

    # -------------------------------------------------------------- project
    def export_rlt(self, project_dir: str, output_path: str, package: bool = False) -> dict:
        return self._post("/api/project/export-rlt", {
            "project_dir": project_dir, "output_path": output_path, "package": package,
        })

    def import_rlt(self, rlt_path: str, output_dir: str) -> dict:
        return self._post("/api/project/import-rlt", {
            "rlt_path": rlt_path, "output_dir": output_dir,
        })

    def inspect_rlt(self, rlt_path: str) -> dict:
        return self._post("/api/project/inspect-rlt", {"rlt_path": rlt_path})
