# nvidia-gpu-stats

A tiny, dependency-free HTTP daemon that exposes **NVIDIA GPU metrics** on a
box where `nvidia-smi` is available. It polls `nvidia-smi` on an interval and
serves the latest snapshot as JSON over HTTP, so a remote machine (for example
a [Home Assistant](https://github.com/thomasbidou/NVIDIA_GPU_Metric) box on the
same LAN) can read GPU stats without needing the NVIDIA driver stack itself.

Stdlib only — **no pip, no apt, no third‑party packages**.

## What it reports

| Field | Meaning | Unit |
|---|---|---|
| `name` | GPU product name | – |
| `driver_version` | NVIDIA driver version | – |
| `uuid` | GPU UUID | – |
| `gpu_utilization_pct` | GPU core utilization | % |
| `memory_used_pct` | VRAM utilization | % |
| `memory_used_gib` | VRAM used | GiB |
| `memory_total_gib` | VRAM total | GiB |
| `power_draw_w` | Current power draw | W |
| `power_limit_w` | Configured power limit | W |
| `power_usage_pct` | `power_draw_w / power_limit_w * 100` | % |
| `temperature_c` | GPU temperature | °C |
| `fan_speed_pct` | Fan speed | % |
| `poll_interval_s` | Seconds between polls | s |
| `collected_at` | Unix timestamp of the last poll | – |

## Quick start (run it directly)

```bash
# Requires: nvidia-smi on PATH
python3 server.py
# → nvidia_gpu_stats listening on http://0.0.0.0:8790/
```

Then:

```bash
curl http://127.0.0.1:8790/          # latest snapshot (JSON)
curl http://127.0.0.1:8790/health    # {"status":"ok"}
```

## Configuration

Settings come from environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `NVIDIA_GPU_STATS_PORT` | `8790` | TCP port to listen on |
| `NVIDIA_GPU_STATS_POLL` | `3` | Seconds between `nvidia-smi` polls |

Example:

```bash
NVIDIA_GPU_STATS_PORT=9000 NVIDIA_GPU_STATS_POLL=5 python3 server.py
```

## Run it as a systemd service

A ready‑made unit file is included (`nvidia-gpu-stats.service`).

```bash
sudo cp nvidia-gpu-stats.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-gpu-stats
sudo systemctl status nvidia-gpu-stats
```

The unit runs as `User=thomas`, restarts on failure, and reads the same
environment variables if you add `Environment=` lines or a drop‑in.

> Adjust `WorkingDirectory`, `ExecStart`, and `User=` in the unit to match your
> layout before installing.

## API

### `GET /`

Returns the most recent snapshot as pretty‑printed JSON. Returns `503` with an
`error` field if no snapshot has been collected yet (e.g. `nvidia-smi` failed
on startup).

### `GET /health`

Returns `{"status": "ok"}` — a cheap liveness probe.

All endpoints send `Access-Control-Allow-Origin: *` so browsers and other
clients can read them.

## Design notes

- **Single file, stdlib only** (`http.server`, `threading`, `subprocess`,
  `re`, `json`). Easy to audit, easy to deploy anywhere Python 3.8+ runs.
- A background thread owns the poll loop; the HTTP server serves the last
  good snapshot, so a transient `nvidia-smi` hiccup never 500s a reader.
- The daemon binds `0.0.0.0` by design (it is meant to be polled over the
  LAN). **Put a firewall / VLAN boundary in front of it** if the machine is
  reachable from untrusted networks — there is no auth by design.

## License

MIT.
