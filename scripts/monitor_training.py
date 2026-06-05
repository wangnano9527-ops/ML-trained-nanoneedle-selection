from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import tomllib
from typing import Any
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)/(?P<total_epochs>\d+)\s+"
    r"(?P<phase>[A-Za-z_]+):\s+(?P<percent>\d+)%"
    r".*\|\s*(?P<step>\d+)/(?P<total_steps>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)<(?P<remaining>[^,\]]+)"
)
VAL_RE = re.compile(
    r"val:\s+(?P<percent>\d+)%"
    r".*\|\s*(?P<step>\d+)/(?P<total_steps>\d+)\s+"
    r"\[(?P<elapsed>[^<\]]+)<(?P<remaining>[^,\]]+)"
)


@dataclass(frozen=True)
class MonitorConfig:
    config_path: Path
    run_dir: Path
    log_dir: Path
    host: str
    port: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local dashboard for Needle Select training.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.toml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/unet_baseline"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs/training_logs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MonitorConfig(
        config_path=resolve_project_path(args.config),
        run_dir=resolve_project_path(args.run_dir),
        log_dir=resolve_project_path(args.log_dir),
        host=args.host,
        port=args.port,
    )
    server = TrainingMonitorServer((config.host, config.port), TrainingMonitorHandler, config)
    url = f"http://{config.host}:{config.port}"
    print(f"Training monitor: {url}")
    print("Press Ctrl+C to stop the monitor. Training will keep running.")
    server.serve_forever()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class TrainingMonitorServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config: MonitorConfig):
        super().__init__(server_address, handler)
        self.config = config


class TrainingMonitorHandler(BaseHTTPRequestHandler):
    server: TrainingMonitorServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_text(HTML_PAGE, "text/html; charset=utf-8")
            return
        if route == "/api/status":
            self.send_json(build_status(self.server.config))
            return
        self.send_error(404, "Not found")

    def do_HEAD(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if route == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_error(404, "Not found")

    def send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_status(config: MonitorConfig) -> dict[str, Any]:
    out_log, err_log = find_latest_logs(config.log_dir)
    metrics = parse_metrics(out_log)
    err_lines = read_tail_lines(err_log, max_bytes=350_000)
    out_lines = read_tail_lines(out_log, max_bytes=120_000)
    progress = parse_progress(err_lines, metrics, config.config_path)
    processes = inspect_training_processes()
    gpu = inspect_gpu()
    checkpoints = inspect_checkpoints(config.run_dir)
    history = inspect_history(config.run_dir / "history.json")
    finished_marker = any("Best validation dice:" in line for line in out_lines)

    log_mtime = max(path_mtime(out_log), path_mtime(err_log), 0.0)
    log_age_seconds = round(time.time() - log_mtime, 1) if log_mtime else None
    running = bool(processes)
    if running and log_age_seconds is not None and log_age_seconds > 300:
        state = "stale"
    elif running:
        state = "running"
    elif finished_marker or history:
        state = "finished"
    else:
        state = "stopped"

    return {
        "state": state,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(PROJECT_ROOT),
        "config_path": str(config.config_path),
        "run_dir": str(config.run_dir),
        "log_dir": str(config.log_dir),
        "logs": {
            "out": str(out_log) if out_log else None,
            "err": str(err_log) if err_log else None,
            "last_update_seconds_ago": log_age_seconds,
        },
        "progress": progress,
        "metrics": metrics,
        "best": best_metric(metrics),
        "processes": processes,
        "gpu": gpu,
        "checkpoints": checkpoints,
        "history_exists": bool(history),
        "recent_output": out_lines[-12:],
        "recent_progress": err_lines[-35:],
    }


def find_latest_logs(log_dir: Path) -> tuple[Path | None, Path | None]:
    err_logs = sorted(log_dir.glob("train-*.err.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if err_logs:
        err_log = err_logs[0]
        out_log = err_log.with_name(err_log.name.replace(".err.log", ".out.log"))
        return out_log if out_log.exists() else None, err_log
    out_logs = sorted(log_dir.glob("train-*.out.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    if out_logs:
        out_log = out_logs[0]
        err_log = out_log.with_name(out_log.name.replace(".out.log", ".err.log"))
        return out_log, err_log if err_log.exists() else None
    return None, None


def read_tail_lines(path: Path | None, *, max_bytes: int) -> list[str]:
    if path is None or not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    text = data.decode("utf-8", errors="replace").replace("\r", "\n")
    return [line for line in text.splitlines() if line.strip()]


def parse_metrics(out_log: Path | None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in read_tail_lines(out_log, max_bytes=2_000_000):
        stripped = line.strip()
        if not stripped.startswith("{") or "epoch" not in stripped:
            continue
        try:
            value = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict) and "epoch" in value:
            result.append(value)
    return result


def parse_progress(lines: list[str], metrics: list[dict[str, Any]], config_path: Path) -> dict[str, Any]:
    total_epochs = load_total_epochs(config_path)
    latest_metric_epoch = int(metrics[-1]["epoch"]) if metrics else 0
    last_epoch_context: tuple[int, int] | None = None
    current: dict[str, Any] | None = None

    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            values = epoch_match.groupdict()
            epoch = int(values["epoch"])
            total_epochs = int(values["total_epochs"])
            last_epoch_context = (epoch, total_epochs)
            current = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "phase": values["phase"],
                "percent": int(values["percent"]),
                "step": int(values["step"]),
                "total_steps": int(values["total_steps"]),
                "elapsed": values["elapsed"].strip(),
                "remaining": values["remaining"].strip(),
            }
            continue
        val_match = VAL_RE.search(line)
        if val_match:
            values = val_match.groupdict()
            epoch, total = last_epoch_context or (latest_metric_epoch + 1, total_epochs)
            total_epochs = total
            current = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "phase": "val",
                "percent": int(values["percent"]),
                "step": int(values["step"]),
                "total_steps": int(values["total_steps"]),
                "elapsed": values["elapsed"].strip(),
                "remaining": values["remaining"].strip(),
            }

    if current is None:
        current = {
            "epoch": latest_metric_epoch,
            "total_epochs": total_epochs,
            "phase": "idle",
            "percent": 100 if latest_metric_epoch >= total_epochs and total_epochs else 0,
            "step": None,
            "total_steps": None,
            "elapsed": None,
            "remaining": None,
        }

    current["completed_epochs"] = latest_metric_epoch
    current["global_percent"] = compute_global_percent(current, latest_metric_epoch)
    return current


def load_total_epochs(config_path: Path) -> int:
    try:
        with config_path.open("rb") as handle:
            config = tomllib.load(handle)
        return int(config.get("training", {}).get("epochs", 80))
    except Exception:
        return 80


def compute_global_percent(progress: dict[str, Any], completed_epochs: int) -> float:
    total_epochs = int(progress.get("total_epochs") or 0)
    if total_epochs <= 0:
        return 0.0
    epoch = int(progress.get("epoch") or completed_epochs or 0)
    phase = str(progress.get("phase") or "idle")
    percent = float(progress.get("percent") or 0.0) / 100.0
    if phase == "val":
        epoch_fraction = 0.95 + 0.05 * percent
        base_epoch = max(epoch - 1, completed_epochs)
    elif phase not in {"idle", "done"} and epoch > completed_epochs:
        epoch_fraction = 0.95 * percent
        base_epoch = epoch - 1
    else:
        epoch_fraction = 0.0
        base_epoch = completed_epochs
    return round(100.0 * min(total_epochs, base_epoch + epoch_fraction) / total_epochs, 2)


def inspect_training_processes() -> list[dict[str, Any]]:
    if platform.system().lower() == "windows":
        command = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
                "Where-Object { $_.CommandLine -like '*train_unet.py*' -or "
                "$_.CommandLine -like '*run_training_pipeline.py*' } | "
                "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
            ),
        ]
    else:
        command = ["ps", "-eo", "pid,args"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return []

    if platform.system().lower() == "windows":
        output = output.strip()
        if not output:
            return []
        try:
            parsed = json.loads(output)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            parsed = [parsed]
        return [
            {"pid": item.get("ProcessId"), "command": item.get("CommandLine", "")}
            for item in parsed
            if isinstance(item, dict)
        ]

    processes = []
    for line in output.splitlines():
        if "train_unet.py" in line or "run_training_pipeline.py" in line:
            pid, _, command = line.strip().partition(" ")
            processes.append({"pid": pid, "command": command.strip()})
    return processes


def inspect_gpu() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:
        return None
    first = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        used = float(parts[1])
        total = float(parts[2])
        util = float(parts[3])
    except ValueError:
        return None
    return {
        "name": parts[0],
        "memory_used_mb": used,
        "memory_total_mb": total,
        "memory_percent": round(100.0 * used / total, 1) if total else 0.0,
        "utilization_percent": util,
    }


def inspect_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    if not run_dir.exists():
        return []
    checkpoints = []
    for path in sorted(run_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True):
        stat = path.stat()
        checkpoints.append(
            {
                "name": path.name,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
            }
        )
    return checkpoints


def inspect_history(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def best_metric(metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not metrics:
        return None
    return max(metrics, key=lambda row: float(row.get("dice", -1.0)))


def path_mtime(path: Path | None) -> float:
    if path is None or not path.exists():
        return 0.0
    return path.stat().st_mtime


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Needle Select Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d2430;
      --muted: #637083;
      --line: #d9dee7;
      --green: #16794c;
      --blue: #2463a6;
      --amber: #a26207;
      --red: #a33131;
      --ink: #263447;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Roboto, Arial, sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
    }
    .subtle { color: var(--muted); font-size: 13px; }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      font-size: 13px;
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--muted);
    }
    .running .dot { background: var(--green); }
    .finished .dot { background: var(--blue); }
    .stale .dot { background: var(--amber); }
    .stopped .dot { background: var(--red); }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 18px 22px 28px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-7 { grid-column: span 7; }
    .span-8 { grid-column: span 8; }
    .span-2 { grid-column: span 2; }
    .span-12 { grid-column: span 12; }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }
    .value {
      font-size: 24px;
      font-weight: 680;
      line-height: 1.15;
      word-break: break-word;
    }
    .small-value {
      font-size: 15px;
      font-weight: 600;
      line-height: 1.35;
      word-break: break-word;
    }
    .bar {
      width: 100%;
      height: 12px;
      background: #e7ebf0;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 12px;
    }
    .bar-fill {
      height: 100%;
      width: 0%;
      background: var(--blue);
      transition: width 260ms ease;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .stat {
      border-top: 1px solid var(--line);
      padding-top: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      white-space: nowrap;
    }
    th { color: var(--muted); font-weight: 600; }
    canvas {
      display: block;
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }
    pre {
      margin: 0;
      max-height: 280px;
      overflow: auto;
      background: #111827;
      color: #d7dde8;
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .paths {
      display: grid;
      grid-template-columns: 110px minmax(0, 1fr);
      gap: 6px 10px;
      font-size: 12px;
    }
    .path { color: var(--ink); word-break: break-all; }
    @media (max-width: 980px) {
      .span-2, .span-3, .span-4, .span-5, .span-7, .span-8 { grid-column: span 12; }
      header { align-items: flex-start; flex-direction: column; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Needle Select Training Monitor</h1>
      <div class="subtle" id="updated">Loading...</div>
    </div>
    <div class="status stopped" id="state"><span class="dot"></span><span>unknown</span></div>
  </header>
  <main>
    <section class="grid">
      <div class="panel span-4">
        <div class="label">Current Progress</div>
        <div class="value" id="progress-title">--</div>
        <div class="bar"><div class="bar-fill" id="global-bar"></div></div>
        <div class="stats">
          <div class="stat"><div class="label">Global</div><div class="small-value" id="global-percent">--</div></div>
          <div class="stat"><div class="label">Epoch</div><div class="small-value" id="epoch-percent">--</div></div>
          <div class="stat"><div class="label">Step</div><div class="small-value" id="step-count">--</div></div>
          <div class="stat"><div class="label">Remaining</div><div class="small-value" id="remaining">--</div></div>
        </div>
      </div>
      <div class="panel span-3">
        <div class="label">Best Dice</div>
        <div class="value" id="best-dice">--</div>
        <div class="subtle" id="best-detail">--</div>
      </div>
      <div class="panel span-3">
        <div class="label">GPU</div>
        <div class="small-value" id="gpu-name">--</div>
        <div class="bar"><div class="bar-fill" id="gpu-memory"></div></div>
        <div class="subtle" id="gpu-detail">--</div>
      </div>
      <div class="panel span-2">
        <div class="label">Processes</div>
        <div class="value" id="process-count">--</div>
        <div class="subtle" id="process-detail">--</div>
      </div>
      <div class="panel span-7">
        <div class="label">Metrics</div>
        <canvas id="chart" width="900" height="300"></canvas>
      </div>
      <div class="panel span-5">
        <div class="label">Latest Epochs</div>
        <table>
          <thead>
            <tr><th>Epoch</th><th>Train</th><th>Val</th><th>Dice</th><th>Precision</th><th>Recall</th></tr>
          </thead>
          <tbody id="metrics-table"></tbody>
        </table>
      </div>
      <div class="panel span-4">
        <div class="label">Checkpoints</div>
        <table>
          <thead><tr><th>Name</th><th>MB</th><th>Modified</th></tr></thead>
          <tbody id="checkpoints"></tbody>
        </table>
      </div>
      <div class="panel span-8">
        <div class="label">Paths</div>
        <div class="paths" id="paths"></div>
      </div>
      <div class="panel span-12">
        <div class="label">Recent Progress Log</div>
        <pre id="progress-log"></pre>
      </div>
      <div class="panel span-12">
        <div class="label">Recent Epoch Output</div>
        <pre id="output-log"></pre>
      </div>
    </section>
  </main>
  <script>
    const fmt = (value, digits = 4) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
      return Number(value).toFixed(digits);
    };

    function setText(id, text) {
      document.getElementById(id).textContent = text;
    }

    function setBar(id, percent) {
      document.getElementById(id).style.width = `${Math.max(0, Math.min(100, percent || 0))}%`;
    }

    async function refresh() {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        const data = await response.json();
        render(data);
      } catch (error) {
        setText("updated", `Monitor fetch failed: ${error}`);
      }
    }

    function render(data) {
      const state = document.getElementById("state");
      state.className = `status ${data.state}`;
      state.querySelector("span:last-child").textContent = data.state;
      setText("updated", `Updated ${data.generated_at} | log age ${data.logs.last_update_seconds_ago ?? "--"}s`);

      const p = data.progress || {};
      setText("progress-title", `Epoch ${p.epoch || "--"}/${p.total_epochs || "--"} ${p.phase || ""}`);
      setText("global-percent", `${fmt(p.global_percent, 2)}%`);
      setText("epoch-percent", `${p.percent ?? "--"}%`);
      setText("step-count", p.step ? `${p.step}/${p.total_steps}` : "--");
      setText("remaining", p.remaining || "--");
      setBar("global-bar", p.global_percent || 0);

      if (data.best) {
        setText("best-dice", fmt(data.best.dice, 4));
        setText("best-detail", `epoch ${data.best.epoch} | val ${fmt(data.best.val_loss, 4)}`);
      } else {
        setText("best-dice", "--");
        setText("best-detail", "waiting for first validation");
      }

      if (data.gpu) {
        setText("gpu-name", data.gpu.name);
        setText("gpu-detail", `${data.gpu.memory_used_mb}/${data.gpu.memory_total_mb} MB | util ${data.gpu.utilization_percent}%`);
        setBar("gpu-memory", data.gpu.memory_percent);
      } else {
        setText("gpu-name", "No NVIDIA data");
        setText("gpu-detail", "--");
        setBar("gpu-memory", 0);
      }

      setText("process-count", String((data.processes || []).length));
      setText("process-detail", (data.processes || []).map(p => `PID ${p.pid}`).join(", ") || "none");
      renderMetricsTable(data.metrics || []);
      renderCheckpoints(data.checkpoints || []);
      renderPaths(data);
      setText("progress-log", (data.recent_progress || []).join("\n"));
      setText("output-log", (data.recent_output || []).join("\n"));
      drawChart(data.metrics || []);
    }

    function renderMetricsTable(metrics) {
      const body = document.getElementById("metrics-table");
      body.innerHTML = "";
      metrics.slice(-8).reverse().forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${row.epoch}</td><td>${fmt(row.train_loss)}</td><td>${fmt(row.val_loss)}</td><td>${fmt(row.dice)}</td><td>${fmt(row.precision)}</td><td>${fmt(row.recall)}</td>`;
        body.appendChild(tr);
      });
    }

    function renderCheckpoints(items) {
      const body = document.getElementById("checkpoints");
      body.innerHTML = "";
      items.slice(0, 8).forEach(item => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${item.name}</td><td>${item.size_mb}</td><td>${item.modified}</td>`;
        body.appendChild(tr);
      });
    }

    function renderPaths(data) {
      const paths = [
        ["run_dir", data.run_dir],
        ["out_log", data.logs.out],
        ["err_log", data.logs.err],
        ["config", data.config_path],
      ];
      const root = document.getElementById("paths");
      root.innerHTML = "";
      paths.forEach(([label, value]) => {
        const left = document.createElement("div");
        left.className = "label";
        left.textContent = label;
        const right = document.createElement("div");
        right.className = "path";
        right.textContent = value || "--";
        root.appendChild(left);
        root.appendChild(right);
      });
    }

    function drawChart(metrics) {
      const canvas = document.getElementById("chart");
      const ctx = canvas.getContext("2d");
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#fbfcfd";
      ctx.fillRect(0, 0, width, height);
      const pad = { left: 52, right: 18, top: 22, bottom: 38 };
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      ctx.strokeStyle = "#d9dee7";
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = pad.top + (plotH * i / 4);
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(width - pad.right, y);
        ctx.stroke();
        ctx.fillStyle = "#637083";
        ctx.font = "12px Segoe UI";
        ctx.fillText((1 - i / 4).toFixed(2), 8, y + 4);
      }
      ctx.strokeStyle = "#263447";
      ctx.beginPath();
      ctx.moveTo(pad.left, pad.top);
      ctx.lineTo(pad.left, height - pad.bottom);
      ctx.lineTo(width - pad.right, height - pad.bottom);
      ctx.stroke();
      if (!metrics.length) {
        ctx.fillStyle = "#637083";
        ctx.font = "14px Segoe UI";
        ctx.fillText("Waiting for validation metrics", pad.left + 16, pad.top + 28);
        return;
      }
      const maxEpoch = Math.max(...metrics.map(m => Number(m.epoch || 0)), 1);
      const xFor = epoch => pad.left + ((epoch - 1) / Math.max(maxEpoch - 1, 1)) * plotW;
      const yFor = value => pad.top + (1 - Math.max(0, Math.min(1, value || 0))) * plotH;
      drawLine(ctx, metrics, "dice", "#2463a6", xFor, yFor);
      drawLine(ctx, metrics, "precision", "#16794c", xFor, yFor);
      drawLine(ctx, metrics, "recall", "#a26207", xFor, yFor);
      ctx.font = "12px Segoe UI";
      ctx.fillStyle = "#2463a6"; ctx.fillText("dice", pad.left + 8, height - 12);
      ctx.fillStyle = "#16794c"; ctx.fillText("precision", pad.left + 58, height - 12);
      ctx.fillStyle = "#a26207"; ctx.fillText("recall", pad.left + 138, height - 12);
    }

    function drawLine(ctx, metrics, key, color, xFor, yFor) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      metrics.forEach((m, index) => {
        const x = xFor(Number(m.epoch));
        const y = yFor(Number(m[key]));
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.fillStyle = color;
      metrics.forEach(m => {
        const x = xFor(Number(m.epoch));
        const y = yFor(Number(m[key]));
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    refresh();
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
