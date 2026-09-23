#!/usr/bin/env python3
"""nvidia_gpu stats daemon.

Polls `nvidia-smi` on an interval and serves a small JSON document over
HTTP so a remote Home Assistant instance (or anything else) can poll it
without needing GPU drivers or the NVIDIA stack.

Stdlib only. No third-party deps.

Endpoints:
  GET /            -> JSON snapshot (latest poll)
  GET /health      -> {"status": "ok"}
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

HOST = "0.0.0.0"
PORT = int(__import__("os").environ.get("NVIDIA_GPU_STATS_PORT", "8790"))
POLL_INTERVAL_S = float(__import__("os").environ.get("NVIDIA_GPU_STATS_POLL", "3"))

# nvidia-smi fields we need, with the csv query string.
QUERY = (
    "name,driver_version,uuid,"
    "utilization.gpu,utilization.memory,"
    "memory.used,memory.total,"
    "power.draw,power.limit,"
    "temperature.gpu,fan.speed"
)

log = logging.getLogger("nvidia_gpu_stats")


def _to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if not s or s in ("[Not Supported]", "[N/A]", "N/A", "N/A [Not Supported]", "[N/A ]"):
        return None
    m = re.search(r"-?\d+(\.\d+)?", s)
    if not m:
        return None
    return float(m.group(0))


def _to_int(s: str) -> Optional[int]:
    f = _to_float(s)
    return None if f is None else int(f)


def collect() -> Optional[dict]:
    """Run nvidia-smi once and return a snapshot dict (or None on failure)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu", QUERY, "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError as exc:
        log.warning("nvidia-smi failed: %s", exc)
        return None

    lines = [ln for ln in out.splitlines() if ln.strip()]
    if not lines:
        return None

    parts = [p.strip() for p in lines[0].split(",")]
    # Expected order matches QUERY: 11 fields.
    if len(parts) < 11:
        log.warning("unexpected nvidia-smi column count: %d", len(parts))
        return None

    name, driver_version, uuid = parts[0], parts[1], parts[2]
    gpu_util = _to_float(parts[3])
    mem_util = _to_float(parts[4])
    mem_used_mib = _to_float(parts[5])
    mem_total_mib = _to_float(parts[6])
    power_draw_w = _to_float(parts[7])
    power_limit_w = _to_float(parts[8])
    temp_c = _to_int(parts[9])
    fan_pct = _to_float(parts[10])

    def gib(v: Optional[float]) -> Optional[float]:
        return None if v is None else round(v / 1024.0, 3)

    mem_used_gib = gib(mem_used_mib)
    mem_total_gib = gib(mem_total_mib)
    power_usage_pct = (
        round(100.0 * power_draw_w / power_limit_w, 2)
        if (power_draw_w is not None and power_limit_w)
        else None
    )

    return {
        "name": name,
        "driver_version": driver_version,
        "uuid": uuid,
        "gpu_utilization_pct": gpu_util,
        "memory_used_pct": mem_util,
        "memory_used_gib": mem_used_gib,
        "memory_total_gib": mem_total_gib,
        "power_draw_w": power_draw_w,
        "power_limit_w": power_limit_w,
        "power_usage_pct": power_usage_pct,
        "temperature_c": temp_c,
        "fan_speed_pct": fan_pct,
        "poll_interval_s": POLL_INTERVAL_S,
        "collected_at": time.time(),
    }


class State:
    """Thread-safe holder for the latest snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.data: Optional[dict] = None
        self.error: Optional[str] = None
        self.last_ok: Optional[float] = None

    def set(self, data: Optional[dict], err: Optional[str] = None) -> None:
        with self._lock:
            if data is not None:
                self.data = data
                self.error = None
                self.last_ok = time.time()
            else:
                self.error = err or "collection failed"

    def get(self) -> dict:
        with self._lock:
            return {"data": self.data, "error": self.error, "last_ok": self.last_ok}


STATE = State()


def poll_loop() -> None:
    while True:
        started = time.time()
        data = collect()
        STATE.set(data, err=None if data is not None else "nvidia-smi failed")
        if data is None:
            log.debug("poll failed")
        elapsed = time.time() - started
        # sleep the remainder of the interval (min 0.2s)
        time.sleep(max(0.2, POLL_INTERVAL_S - elapsed))


class Handler(BaseHTTPRequestHandler):
    server_version = "nvidia-gpu-stats/1.0"

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("", "/"):
            snap = STATE.get()
            if not snap["data"]:
                self._send(
                    503,
                    json.dumps({"error": snap["error"] or "not ready"}).encode(),
                )
                return
            self._send(200, json.dumps(snap["data"], indent=2).encode())
        elif path == "/health":
            self._send(200, json.dumps({"status": "ok"}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def log_message(self, fmt: str, *args) -> None:  # quiet access log
        log.debug("http: " + fmt, *args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Prime the first snapshot before we start serving so the first request
    # doesn't 503.
    first = collect()
    STATE.set(first, err=None if first is not None else "initial collection failed")
    threading.Thread(target=poll_loop, name="poll", daemon=True).start()

    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    log.info("nvidia_gpu_stats listening on http://%s:%d/", HOST, PORT)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
