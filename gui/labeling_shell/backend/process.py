"""Local backend subprocess management.

Ports the Electron main.js backend lifecycle (`resolveBackendPython`,
`startBackend`, `stopBackend`) to Python: pick an interpreter, spawn
`uvicorn backend.server:app`, and drain its output so it neither deadlocks on a
full pipe nor vanishes silently.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

from .config import BackendConfig

LogCallback = Callable[[str], None]


def _pixi_env_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".pixi" / "envs" / "default" / "python.exe"
    return root / ".pixi" / "envs" / "default" / "bin" / "python"


def _conda_prefix(python_path: str) -> Path | None:
    """The conda/pixi environment prefix an interpreter belongs to, if any."""
    exe = Path(python_path)
    if not exe.is_absolute() or not exe.exists():
        return None
    prefix = exe.parent if os.name == "nt" else exe.parent.parent
    return prefix if (prefix / "conda-meta").is_dir() else None


def activation_path(python_path: str, base_path: str | None = None) -> str | None:
    """`base_path` with the interpreter's conda/pixi env directories prepended.

    Spawning `<env>/python.exe` directly is **not** the same as activating that
    environment, and on Windows the difference is fatal rather than cosmetic:
    conda-forge's numpy/scipy reach BLAS/LAPACK through DLLs in
    `<env>\\Library\\bin`, resolved off PATH at call time. Without those entries
    every import still succeeds — the failure lands only when the first LAPACK
    routine actually runs, and it lands as a *hard* process crash
    (`0xc06d007f`, DELAYLOAD_DLL_NOT_FOUND) with no Python traceback.

    For this app that means feature extraction: `asoid_core.features.angle_between`
    calls `np.linalg.det` on every body-part pair, so the backend dies part-way
    through Extract Features and the GUI sees nothing but the connection
    dropping. `pixi run test` never caught it because pixi activates first.

    Returns None when the interpreter is not a conda-style env — there is then
    nothing to prepend and PATH should be left alone.
    """
    prefix = _conda_prefix(python_path)
    if prefix is None:
        return None
    if os.name == "nt":
        # Same set, in the same order, that `pixi run` / `conda activate` prepend.
        entries = [
            prefix,
            prefix / "Library" / "mingw-w64" / "bin",
            prefix / "Library" / "usr" / "bin",
            prefix / "Library" / "bin",
            prefix / "Scripts",
            prefix / "bin",
        ]
    else:
        entries = [prefix / "bin"]
    base = os.environ.get("PATH", "") if base_path is None else base_path
    return os.pathsep.join([str(e) for e in entries] + ([base] if base else []))


def resolve_python(config: BackendConfig) -> str:
    """Choose the interpreter to run the backend.

    Resolution order (mirrors the Electron app):
      1. ASOID_PYTHON / PYTHON env override.
      2. config.python explicit override.
      3. Project-local pixi env at <repo>/.pixi/envs/default.
      4. Bare python / python3 on PATH.
    """
    override = os.environ.get("ASOID_PYTHON") or os.environ.get("PYTHON")
    if override and override.strip():
        return override.strip()
    if config.python:
        return config.python
    pixi = _pixi_env_python(config.repo_root)
    if pixi.exists():
        return str(pixi)
    return "python" if os.name == "nt" else "python3"


class BackendProcess:
    """A spawned local uvicorn process running the FastAPI backend."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._proc: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, on_log: LogCallback | None = None) -> None:
        if self.running:
            return
        python = resolve_python(self.config)
        cmd = [
            python, "-m", "uvicorn", self.config.module,
            "--port", str(self.config.port), "--host", self.config.host,
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        # Activate the interpreter's env for the child. See `activation_path`:
        # without this the backend hard-crashes inside feature extraction.
        activated = activation_path(python, env.get("PATH"))
        if activated is not None:
            env["PATH"] = activated
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(self.config.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._log_thread = threading.Thread(
            target=self._drain, args=(on_log,), daemon=True
        )
        self._log_thread.start()

    def _drain(self, on_log: LogCallback | None) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.rstrip("\n")
            if on_log is not None:
                on_log(line)
            else:
                print(f"[backend] {line}")

    def stop(self, timeout: float = 5.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
