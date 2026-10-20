#!/usr/bin/env python3
"""Loopback-only Ollama lifetime owned by one existing Slurm GPU allocation."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import Request, urlopen


def disk_bytes(root):
    # Include incomplete pulls and state; do not follow symlinks or double count.
    seen, total = set(), 0
    for directory, _, names in os.walk(root, followlinks=False):
        for name in names:
            entry = (Path(directory) / name).lstat()
            identity = (entry.st_dev, entry.st_ino)
            if identity not in seen:
                total += entry.st_blocks * 512
                seen.add(identity)
    return total


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
    # A parent may have exited before its runner; clean its private session too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--mode", choices=("provision", "check", "serve", "worker"), default="check")
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--port", type=int, default=11437)
    args = parser.parse_args()
    if (args.mode == "worker") != (args.worker_config is not None):
        parser.error("Worker mode requires --worker-config; other modes do not accept credentials")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Start through the approved gpu:1 Slurm job; GPU allocation is required")
    if not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged loopback port")
    project = args.project.expanduser().resolve()
    if not project.is_dir() or project.stat().st_uid != os.getuid() or project.stat().st_mode & 0o077:
        parser.error("Project must exist, be owned by the current user and have permissions 0700")
    pin = json.loads(Path(__file__).with_name("release.json").read_text())
    runtime = project / "runtime" / pin["version"]
    marker = runtime / "verified-release.json"
    from install_runtime import same_runtime_artifact
    if not marker.is_file() or not same_runtime_artifact(json.loads(marker.read_text()), pin):
        parser.error("Install the pinned verified runtime first")
    executable = runtime / "bin" / "ollama"
    checker = Path(__file__).parent.parent / "check_local_model_services.py"
    if args.mode in {"provision", "check"} and not checker.is_file():
        parser.error("Copy the standalone conformance checker beside minpaku_pilot/")
    if args.mode == "worker":
        from worker import load_config
        # Validate ownership, mode, project scope and staging origin before startup.
        load_config(args.worker_config, project)
    # Fail if occupied: never submit to an existing listener with unknown settings.
    with socket.socket() as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        candidate.bind(("127.0.0.1", args.port))
    state = project / "state-home"
    state.mkdir(mode=0o700, exist_ok=True)
    models = project / "models"
    models.mkdir(mode=0o700, exist_ok=True)
    logs = project / "logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    base = "http://127.0.0.1:" + str(args.port)
    # A clean environment prevents forwarding Cloud keys or research env settings.
    environment = {
        "HOME": str(state), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "LD_LIBRARY_PATH": str(runtime / "lib" / "ollama") + ":/usr/lib/wsl/lib",
        "OLLAMA_HOST": base, "OLLAMA_MODELS": str(models),
        "OLLAMA_NO_CLOUD": "1", "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_MAX_QUEUE": "2",
        "OLLAMA_CONTEXT_LENGTH": "4096", "OLLAMA_KEEP_ALIVE": "0",
    }
    budget = pin["budget_bytes"]
    # Keep a 1 GB safety margin for in-flight writes between measurements.
    cutoff = budget - 1_000_000_000
    child = server = None
    job_id = os.environ["SLURM_JOB_ID"]
    if not job_id.isdigit():
        parser.error("Unexpected Slurm job ID")
    deadline = time.monotonic() + 7100  # Slurm also enforces the two-hour ceiling.

    def guard():
        if time.monotonic() >= deadline:
            raise RuntimeError("Pilot deadline reached")
        if disk_bytes(project) >= cutoff:
            raise RuntimeError("Pilot storage safety cutoff reached; stopped without deleting evidence")
        if server is not None and server.poll() is not None:
            raise RuntimeError("Ollama exited; inspect its private job log")

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        with urlopen(Request(base + "/api/" + path, data=data,
                             headers={"Content-Type": "application/json"}),
                     timeout=90 if payload is not None else 3) as response:
            return json.load(response)

    def interrupted(signum, frame):
        raise InterruptedError("Slurm job interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    log_path = logs / ("ollama-" + job_id + ".log")
    try:
        guard()
        with log_path.open("x") as log:
            server = subprocess.Popen([str(executable), "serve"], env=environment,
                                      stdout=log, stderr=log, start_new_session=True)
            ready = False
            for _ in range(60):
                guard()
                try:
                    if request("version").get("version") != pin["version"]:
                        raise RuntimeError("Runtime version does not match approved release")
                    ready = True
                    break
                except OSError:
                    time.sleep(1)
            if not ready:
                raise RuntimeError("Loopback model server did not become ready")
            if args.mode == "provision":
                for model in pin["models"]:
                    child = subprocess.Popen([str(executable), "pull", model], env=environment,
                                             stdout=log, stderr=log, start_new_session=True)
                    while child.poll() is None:
                        guard()
                        time.sleep(0.5)
                    if child.returncode:
                        raise RuntimeError("Approved model pull failed; cached partial data retained")
            inventory = request("tags").get("models", [])
            chosen = []
            for model in pin["models"]:
                found = next((item for item in inventory if item.get("name") == model), None)
                if not found or not found.get("digest"):
                    raise RuntimeError("Approved model is missing: " + model)
                chosen.append({key: found.get(key) for key in ("name", "digest", "size", "details")})
            manifest = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                        "slurm_job_id": job_id, "runtime": pin, "models": chosen,
                        "disk_bytes": disk_bytes(project), "network_scope": "loopback-only"}
            # A successful response alone could be CPU fallback. Observe a loaded
            # synthetic model and require actual VRAM use before claiming GPU success.
            request("chat", {"model": pin["models"][0], "stream": False,
                "think": False, "keep_alive": "30s",
                "options": {"num_ctx": 4096, "num_predict": 1},
                "messages": [{"role": "user", "content": "Say OK."}]})
            active = request("ps").get("models", [])
            gpu_model = next((model for model in active
                              if model.get("name") == pin["models"][0]), None)
            if not gpu_model or not gpu_model.get("size_vram", 0) > 0:
                raise RuntimeError("GPU execution was not observed; CPU fallback is not accepted")
            manifest["observed_gpu_model"] = {key: gpu_model.get(key)
                                               for key in ("name", "digest", "size", "size_vram")}
            request("chat", {"model": pin["models"][0], "messages": [],
                              "stream": False, "keep_alive": 0})
            (logs / ("inventory-" + job_id + ".json")).write_text(json.dumps(manifest, indent=2) + "\n")
            if args.mode == "worker":
                import sys
                child = subprocess.Popen([sys.executable, str(Path(__file__).with_name("worker.py")),
                    "--project", str(project), "--config", str(args.worker_config),
                    "--ollama-base", base], env=environment, start_new_session=True)
                print("Allocated outbound staging worker started in Slurm job " + job_id, flush=True)
                while child.poll() is None:
                    guard()
                    time.sleep(0.5)
                if child.returncode:
                    raise RuntimeError("Outbound worker exited; inspect its safe status output")
            elif args.mode != "serve":
                import sys
                child = subprocess.Popen([sys.executable, str(checker), "--base-url", base,
                    "--chat-model", pin["models"][0], "--vision-model", pin["models"][1],
                    "--embedding-model", pin["models"][2], "--output",
                    str(logs / ("conformance-" + job_id + ".json"))],
                    env=environment, start_new_session=True)
                while child.poll() is None:
                    guard()
                    time.sleep(0.5)
                if child.returncode:
                    raise RuntimeError("Model conformance failed; review the independent service results")
                print("Synthetic checks passed; document evaluation and hosted transport remain separate")
            else:
                print("Loopback pilot ready within Slurm job " + job_id, flush=True)
                while True:
                    guard()
                    time.sleep(0.5)
    finally:
        stop_group(child)
        stop_group(server)


if __name__ == "__main__":
    main()
