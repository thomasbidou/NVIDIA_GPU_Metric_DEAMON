#!/usr/bin/env python3
"""nvidia_gpu stats daemon.

Polls `nvidia-smi` + system metrics (CPU %, CPU temp, RAM) on an interval
and serves a small JSON document over HTTP so a remote Home Assistant
instance (or anything else) can poll it without needing GPU drivers or the
NVIDIA stack.

Stdlib only. No third-party deps.

Endpoints:
  GET /            -> JSON snapshot (latest poll)
  GET /health      -> {"status": "ok"}

Response shape (GPU fields at top level, system metrics under "cpu"):
  {
    "name": "...", "gpu_utilization_pct": ..., "memory_used_pct": ..., ...
    "cpu": {
      "name": "...",
      "usage_pct": ...,
      "temperature_c": ...,
      "ram_used_gib": ...,
      "ram_total_gib": ...,
      "ram_used_pct": ...
    }
  }
"""
from __future__ import annotations

import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

HOST = "0.0.0.0"
PORT = int(os.environ.get("NVIDIA_GPU_STATS_PORT", "8790"))
POLL_INTERVAL_S = float(os.environ.get("NVIDIA_GPU_STATS_POLL", "3"))

# CPU usage: sample /proc/stat over this window (seconds).
CPU_SAMPLE_S = 0.5
# CPU temp: first hwmon that matches this name, else the first hwmon with
# a temp1_input labelled "Tctl" (typical AMD) — else None.
CPU_TEMP_CHIP = "k10temp"

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


def _read_int(path: str) -> Optional[int]:
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _read_str(path: str) -> Optional[str]:
    try:
        with open(path) as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _cpu_usage_pct() -> Optional[float]:
    """Sample /proc/stat twice and compute busy% over the window."""
    def sample():
        try:
            with open("/proc/stat") as fh:
                line = fh.readline().split()
        except OSError:
            return None
        # user nice system idle iowait irq softirq steal ...
        nums = list(map(int, line[1:]))
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        return idle, total

    s1 = sample()
    if s1 is None:
        return None
    time.sleep(CPU_SAMPLE_S)
    s2 = sample()
    if s2 is None:
        return None
    (idle1, total1), (idle2, total2) = s1, s2
    d_total = total2 - total1
    if d_total <= 0:
        return None
    d_busy = d_total - (idle2 - idle1)
    return round(100.0 * d_busy / d_total, 1)


def _cpu_temperature_c() -> Optional[int]:
    """Find the CPU core temperature from /sys/class/hwmon."""
    import glob

    base = "/sys/class/hwmon"
    try:
        chips = sorted(os.listdir(base))
    except OSError:
        return None

    def temp_of(chip: str) -> Optional[int]:
        cdir = os.path.join(base, chip)
        # Prefer a Tctl label (AMD), else temp1_input.
        for f in sorted(glob.glob(os.path.join(cdir, "temp*_input"))):
            label = _read_str(f.replace("_input", "_label")) or ""
            if "Tctl" in label:
                v = _read_int(f)
                if v is not None:
                    return v // 1000
        # fallback: temp1_input
        v = _read_int(os.path.join(cdir, "temp1_input"))
        return None if v is None else v // 1000

    for chip in chips:
        if chip == CPU_TEMP_CHIP:
            t = temp_of(chip)
            if t is not None:
                return t
    # no named chip matched — return first plausible CPU-like chip
    for chip in chips:
        name = (_read_str(os.path.join(base, chip, "name")) or "").lower()
        if name in ("k10temp", "cpu_thermal", "coretemp", "zenpower", "cpu"):
            t = temp_of(chip)
            if t is not None:
                return t
    return None


def _cpu_name() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def collect_system() -> dict:
    """Collect CPU %, CPU temp, and RAM metrics."""
    def gib_kb(v: Optional[int]) -> Optional[float]:
        return None if v is None else round(v / 1024.0 / 1024.0, 3)

    mem = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                val = rest.strip().split()
                if val:
                    mem[key] = int(val[0])
    except OSError:
        pass

    ram_total_kb = mem.get("MemTotal")
    ram_available_kb = mem.get("MemAvailable")
    ram_used_kb = (ram_total_kb - ram_available_kb) if (ram_total_kb and ram_available_kb is not None) else None
    ram_used_gib = gib_kb(ram_used_kb)
    ram_total_gib = gib_kb(ram_total_kb)
    ram_used_pct = (
        round(100.0 * ram_used_kb / ram_total_kb, 1)
        if (ram_used_kb is not None and ram_total_kb)
        else None
    )

    return {
        "name": _cpu_name() or "CPU",
        "usage_pct": _cpu_usage_pct(),
        "temperature_c": _cpu_temperature_c(),
        "ram_used_gib": ram_used_gib,
        "ram_total_gib": ram_total_gib,
        "ram_used_pct": ram_used_pct,
    }


def _gpu_from_parts(parts: list[str]) -> dict:
    """Build one GPU dict from 11 nvidia-smi csv fields."""
    name, driver_version, uuid = parts[0], parts[1], parts[2]
    gpu_util = _to_float(parts[3])
    mem_controller_util = _to_float(parts[4])  # nvidia-smi memory-controller utilization (bandwidth), NOT "how full"
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
    # memory_used_pct = "how full is the GPU memory" = used/total.
    # Do NOT use nvidia-smi's utilization.memory for this: that is the
    # memory-controller (bandwidth) utilization, which reads 0% on an idle
    # GPU even when many GiB are allocated — confusing and wrong for a gauge.
    mem_used_pct = (
        round(100.0 * mem_used_mib / mem_total_mib, 1)
        if (mem_used_mib is not None and mem_total_mib)
        else None
    )
    power_usage_pct = (
        round(100.0 * power_draw_w / power_limit_w, 2)
        if (power_draw_w is not None and power_limit_w)
        else None
    )
    return {
        "name": name,
        "uuid": uuid,
        "driver_version": driver_version,
        "gpu_utilization_pct": gpu_util,
        "memory_used_pct": mem_used_pct,
        "memory_used_gib": mem_used_gib,
        "memory_total_gib": mem_total_gib,
        "memory_controller_util_pct": mem_controller_util,
        "power_draw_w": power_draw_w,
        "power_limit_w": power_limit_w,
        "power_usage_pct": power_usage_pct,
        "temperature_c": temp_c,
        "fan_speed_pct": fan_pct,
    }


def collect() -> Optional[dict]:
    """Run nvidia-smi + system metrics and return a snapshot dict.

    New shape (v2):
      { "box": ..., "gpus": [ {...}, ... ], "cpu": {...}, ... }
    For backward compat with older integrations we ALSO keep the first
    GPU's fields at the top level (v1 shape) and a convenience "name".
    """
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
    gpus: list[dict] = []
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 11:
            log.warning("unexpected nvidia-smi column count: %d", len(parts))
            continue
        gpus.append(_gpu_from_parts(parts))
    if not gpus:
        return None

    first = gpus[0]
    return {
        # v2: explicit list (may contain multiple GPUs)
        "gpus": gpus,
        # v1 compat: first GPU promoted to top level
        **first,
        # system
        "box": socket.gethostname(),
        "cpu": collect_system(),
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
