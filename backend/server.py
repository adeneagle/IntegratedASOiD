"""
FastAPI backend for IntegratedASOiD.

Provides endpoints for pose ingestion + feature extraction, RandomForest
training with active learning, and per-frame inference. Designed to be spawned
by the Electron app.

Usage:
    python -m uvicorn backend.server:app --port 8741
    (run from the repository root)
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the project root is importable so `backend.asoid_core` resolves when
# uvicorn is launched from elsewhere.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.routers import pose, training, inference, project  # noqa: E402


def runtime_info() -> dict:
    """Report the interpreter and numeric stack the backend is running on.

    The classifier is CPU-only scikit-learn, so unlike the TSM predecessor there
    is no GPU to probe — what matters for a support report is which interpreter
    Electron picked and whether numba (the feature-extraction JIT) imported.
    """
    info = {"python": sys.executable}
    for mod in ("numpy", "sklearn", "numba", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[f"{mod}_error"] = f"{type(e).__name__}: {e}"
    return info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    print(f"[backend] project root: {PROJECT_ROOT}")
    ri = runtime_info()
    print(f"[backend] python: {ri.get('python')}")
    print(f"[backend] numpy {ri.get('numpy')} · scikit-learn {ri.get('sklearn')} · "
          f"numba {ri.get('numba')}")
    yield
    # Cleanup: cancel any running tasks
    from backend.services.pose_service import cancel_all as pose_cancel
    from backend.services.training_service import cancel_all as training_cancel
    from backend.services.inference_service import cancel_all as inference_cancel
    pose_cancel()
    training_cancel()
    inference_cancel()


app = FastAPI(title="IntegratedASOiD Backend", lifespan=lifespan)

# Allow requests from Electron renderer (any localhost port)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(pose.router, prefix="/api/pose", tags=["pose"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(project.router, prefix="/api/project", tags=["project"])


@app.get("/api/health")
async def health():
    return {"status": "ok", "runtime": runtime_info()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8741, reload=False)
