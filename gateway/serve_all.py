"""Start every Glimms service plus the gateway inside one container.

Used as the command of the root ``Dockerfile``.  One uvicorn process is
spawned per service (bound to the loopback interface by default so the
services are only reachable through the gateway) and one gateway process is
bound to ``0.0.0.0:$PORT`` (default 8080).  If any child process exits, the
supervisor shuts the whole tree down and exits non-zero, letting the
container runtime's restart policy decide what happens next.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (label, uvicorn app import path, port) — ports match docker-compose.yml.
SERVICES: list[tuple[str, str, int]] = [
    ("object-detection", "services.object_detection.app.main:app", 8001),
    ("attribute-extractor", "services.attribute_extractor.app.main:app", 8002),
    ("embedding-engine", "services.embedding_engine.app.main:app", 8003),
    ("permutation-engine", "services.permutation_engine.app.main:app", 8004),
    ("llm-reasoning", "services.llm_reasoning.app.main:app", 8005),
    ("mockup-compositor", "services.mockup_compositor.app.main:app", 8006),
    ("quality-guard", "services.quality_guard.app.main:app", 8007),
    ("context-inference", "services.context_inference.app.main:app", 8008),
]

SERVICE_BIND = os.getenv("SERVICE_BIND", "127.0.0.1")
GATEWAY_PORT = int(os.getenv("PORT", "8080"))


def log(message: str) -> None:
    print(f"[serve_all] {message}", flush=True)


def spawn(label: str, app_path: str, port: int, host: str) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        app_path,
        "--host",
        host,
        "--port",
        str(port),
    ]
    env = {**os.environ, "PORT": str(port)}
    log(f"starting {label} on {host}:{port}")
    return subprocess.Popen(command, cwd=REPO_ROOT, env=env)


def terminate_all(processes: list[subprocess.Popen], timeout: float = 10.0) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - kill() failure
            log(f"child pid {process.pid} did not exit")


def main() -> int:
    processes: list[subprocess.Popen] = [
        spawn(label, app_path, port, SERVICE_BIND)
        for label, app_path, port in SERVICES
    ]
    gateway = spawn("gateway", "gateway.app:app", GATEWAY_PORT, "0.0.0.0")
    processes.append(gateway)

    log(f"gateway listening on 0.0.0.0:{GATEWAY_PORT}; "
        f"route /<service-name>/... for each pipeline service")

    stopping = {"flag": False}

    def request_stop(signum: int, _frame: object) -> None:
        if not stopping["flag"]:
            stopping["flag"] = True
            log(f"received signal {signum}; shutting down children")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    exit_code = 0
    try:
        while True:
            for process in processes:
                return_code = process.poll()
                if return_code is not None:
                    log(f"child pid {process.pid} exited with code {return_code}")
                    exit_code = 1
                    stopping["flag"] = True
                    break
            if stopping["flag"]:
                break
            time.sleep(0.5)
    finally:
        terminate_all(processes)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
