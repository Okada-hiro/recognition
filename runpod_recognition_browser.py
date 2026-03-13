#!/usr/bin/env python3
"""
Read-only browser for the recognition directory.

Default:
  - serves the directory containing this file
  - listens on port 8000

Useful for browsing outputs such as:
  - test_annotated.mp4
  - logs/events.jsonl
  - snapshots/*.jpg
"""

from __future__ import annotations

import html
import mimetypes
import os
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import json

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, Response

from recognition.config import AppConfig
from recognition.pipeline import ReceptionMonitor

ROOT_DIR = Path(os.getenv("RECOGNITION_BROWSE_ROOT", Path(__file__).resolve().parent)).resolve()
PORT = int(os.getenv("PORT", "8000"))
TITLE = "Recognition Browser"
LIVE_PERSON_MODEL = os.getenv("RECOGNITION_LIVE_PERSON_MODEL", "yolo11n.pt")
LIVE_DEVICE = os.getenv("RECOGNITION_LIVE_DEVICE", "auto")
LIVE_DATABASE_DIR = Path(os.getenv("RECOGNITION_LIVE_DATABASE_DIR", ROOT_DIR.parent / "data_base")).resolve()
LIVE_JPEG_QUALITY = int(os.getenv("RECOGNITION_LIVE_JPEG_QUALITY", "85"))

app = FastAPI(title=TITLE, version="1.0.0")
_live_monitor_lock = threading.Lock()
_live_monitor: ReceptionMonitor | None = None
_live_frame_index = 0


def _resolve_relative_path(relative_path: str) -> Path:
    clean = relative_path.strip().lstrip("/").replace("\\", "/")
    target = (ROOT_DIR / clean).resolve()
    try:
        target.relative_to(ROOT_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed.") from exc
    return target


def _format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size_bytes} B"


def _render_entry(path: Path) -> str:
    rel = path.relative_to(ROOT_DIR).as_posix()
    href = f"/browse/{quote(rel)}" if path.is_dir() else f"/files/{quote(rel)}"
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    kind = "dir" if path.is_dir() else "file"
    size = "-" if path.is_dir() else _format_size(stat.st_size)
    label = html.escape(path.name + ("/" if path.is_dir() else ""))
    return (
        f"<tr>"
        f"<td>{kind}</td>"
        f"<td><a href=\"{href}\">{label}</a></td>"
        f"<td>{size}</td>"
        f"<td>{modified}</td>"
        f"</tr>"
    )


def _render_preview(target: Path) -> str:
    rel = target.relative_to(ROOT_DIR).as_posix()
    file_url = f"/files/{quote(rel)}"
    suffix = target.suffix.lower()

    if suffix in {".mp4", ".mov", ".webm"}:
        return (
            f"<h2>Preview</h2>"
            f"<video controls style=\"max-width:100%;height:auto;\">"
            f"<source src=\"{file_url}\">"
            f"</video>"
        )

    if suffix in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return f"<h2>Preview</h2><img src=\"{file_url}\" style=\"max-width:100%;height:auto;\" alt=\"preview\">"

    if suffix in {".json", ".jsonl", ".txt", ".log", ".md", ".py", ".sh"}:
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
        preview = html.escape(text[:100_000])
        return f"<h2>Preview</h2><pre>{preview}</pre>"

    return ""


def _render_directory_page(current: Path) -> str:
    rel = "." if current == ROOT_DIR else current.relative_to(ROOT_DIR).as_posix()
    parent_link = ""
    if current != ROOT_DIR:
        parent = current.parent
        parent_rel = "" if parent == ROOT_DIR else parent.relative_to(ROOT_DIR).as_posix()
        parent_link = f'<p><a href="/browse/{quote(parent_rel)}">.. parent</a></p>'

    rows = [_render_entry(path) for path in sorted(current.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))]
    rows_html = "".join(rows) if rows else '<tr><td colspan="4">empty directory</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{TITLE}</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: #fffaf0;
      --ink: #1c1a18;
      --line: #d7c8a8;
      --link: #0d5c63;
      --muted: #6b665d;
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, var(--bg), #efe5d1);
      color: var(--ink);
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(60, 40, 10, 0.08);
    }}
    a {{ color: var(--link); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: 0.95rem; }}
    code, pre {{
      font-family: "SFMono-Regular", Menlo, monospace;
      background: #f5ecd8;
      border-radius: 8px;
    }}
    pre {{
      padding: 12px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>{TITLE}</h1>
      <p>Root: <code>{html.escape(str(ROOT_DIR))}</code></p>
      <p>Current: <code>{html.escape(rel)}</code></p>
      <p><a href="/live">open live recognition</a></p>
      {parent_link}
      <table>
        <thead>
          <tr><th>Type</th><th>Name</th><th>Size</th><th>Modified</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
  </main>
</body>
</html>"""


def _render_file_page(target: Path) -> str:
    rel = target.relative_to(ROOT_DIR).as_posix()
    parent = target.parent
    parent_rel = "" if parent == ROOT_DIR else parent.relative_to(ROOT_DIR).as_posix()
    preview = _render_preview(target)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(target.name)} - {TITLE}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: #f6f2e8;
      color: #1c1a18;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    .panel {{
      background: #fffaf0;
      border: 1px solid #d7c8a8;
      border-radius: 16px;
      padding: 20px;
    }}
    a {{ color: #0d5c63; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code, pre {{
      font-family: "SFMono-Regular", Menlo, monospace;
      background: #f5ecd8;
      border-radius: 8px;
    }}
    pre {{
      padding: 12px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>{html.escape(target.name)}</h1>
      <p><a href="/browse/{quote(parent_rel)}">back to directory</a></p>
      <p>Path: <code>{html.escape(rel)}</code></p>
      <p><a href="/files/{quote(rel)}">direct file link</a></p>
      {preview}
    </div>
  </main>
</body>
</html>"""


def _render_live_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Recognition</title>
  <style>
    :root {
      --bg: #f6f2e8;
      --panel: #fffaf0;
      --ink: #1c1a18;
      --line: #d7c8a8;
      --link: #0d5c63;
      --accent: #9c6644;
    }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, var(--bg), #efe5d1);
      color: var(--ink);
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(60, 40, 10, 0.08);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
      align-items: start;
    }
    video, img, canvas {
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #000;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 16px 0;
    }
    button, input {
      font: inherit;
    }
    button {
      border: 1px solid var(--line);
      background: var(--accent);
      color: white;
      border-radius: 999px;
      padding: 10px 16px;
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.5;
      cursor: default;
    }
    .status {
      color: #6b665d;
      min-height: 1.5em;
    }
    .note {
      font-size: 0.95rem;
      color: #6b665d;
    }
    code {
      font-family: "SFMono-Regular", Menlo, monospace;
      background: #f5ecd8;
      border-radius: 8px;
      padding: 2px 6px;
    }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>Live Recognition</h1>
      <p><a href="/">back to browser root</a></p>
      <p class="note">
        Safari will access the Mac camera locally, then send snapshots to RunPod for person detection, face detection,
        and face recognition. This is a simple first step, so frames are sampled at a fixed interval.
      </p>
      <div class="controls">
        <button id="startBtn">Start Camera</button>
        <button id="stopBtn" disabled>Stop</button>
        <label>Interval (ms) <input id="intervalInput" type="number" min="300" step="100" value="1000"></label>
        <label>Width <input id="widthInput" type="number" min="320" step="80" value="960"></label>
      </div>
      <div class="status" id="status">Idle.</div>
      <div class="note" id="eventStatus">No approach/leave events yet.</div>
      <div class="row">
        <div>
          <h2>Camera</h2>
          <video id="camera" autoplay playsinline muted></video>
        </div>
        <div>
          <h2>Processed</h2>
          <img id="processed" alt="processed frame">
        </div>
      </div>
      <canvas id="captureCanvas" hidden></canvas>
    </div>
  </main>
  <script>
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");
    const statusEl = document.getElementById("status");
    const cameraEl = document.getElementById("camera");
    const processedEl = document.getElementById("processed");
    const eventStatusEl = document.getElementById("eventStatus");
    const canvasEl = document.getElementById("captureCanvas");
    const intervalInput = document.getElementById("intervalInput");
    const widthInput = document.getElementById("widthInput");

    let mediaStream = null;
    let timerId = null;
    let inFlight = false;

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function setEventStatus(message) {
      eventStatusEl.textContent = message;
    }

    async function startCamera() {
      if (mediaStream) {
        return;
      }
      const width = Number(widthInput.value) || 960;
      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: width } },
        audio: false,
      });
      cameraEl.srcObject = mediaStream;
      await cameraEl.play();
      startBtn.disabled = true;
      stopBtn.disabled = false;
      setStatus("Camera started. Sending frames to RunPod...");
      scheduleLoop();
    }

    function stopCamera() {
      if (timerId) {
        clearInterval(timerId);
        timerId = null;
      }
      if (mediaStream) {
        mediaStream.getTracks().forEach((track) => track.stop());
        mediaStream = null;
      }
      cameraEl.srcObject = null;
      startBtn.disabled = false;
      stopBtn.disabled = true;
      setStatus("Stopped.");
    }

    function scheduleLoop() {
      if (timerId) {
        clearInterval(timerId);
      }
      const intervalMs = Math.max(300, Number(intervalInput.value) || 1000);
      timerId = setInterval(sendFrame, intervalMs);
      sendFrame();
    }

    async function sendFrame() {
      if (!mediaStream || inFlight || cameraEl.videoWidth === 0 || cameraEl.videoHeight === 0) {
        return;
      }
      inFlight = true;
      try {
        canvasEl.width = cameraEl.videoWidth;
        canvasEl.height = cameraEl.videoHeight;
        const ctx = canvasEl.getContext("2d");
        ctx.drawImage(cameraEl, 0, 0, canvasEl.width, canvasEl.height);
        const blob = await new Promise((resolve) => canvasEl.toBlob(resolve, "image/jpeg", 0.85));
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");
        const response = await fetch("/api/live-frame", {
          method: "POST",
          body: formData,
        });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `HTTP ${response.status}`);
        }
        const imageBlob = await response.blob();
        const objectUrl = URL.createObjectURL(imageBlob);
        const prevUrl = processedEl.dataset.url;
        processedEl.src = objectUrl;
        processedEl.dataset.url = objectUrl;
        if (prevUrl) {
          URL.revokeObjectURL(prevUrl);
        }
        const frameIndex = response.headers.get("x-frame-index");
        const trackEventsRaw = response.headers.get("x-track-events");
        let eventMessage = "No approach/leave events.";
        if (trackEventsRaw) {
          const trackEvents = JSON.parse(trackEventsRaw);
          if (trackEvents.length > 0) {
            eventMessage = trackEvents
              .map((event) => `${event.event_type} track=${event.track_id}${event.person_id ? ` ${event.person_id}` : ""}`)
              .join(" | ");
          }
        }
        setStatus(`Processed frame ${frameIndex}.`);
        setEventStatus(eventMessage);
      } catch (error) {
        setStatus(`Error: ${error.message}`);
        setEventStatus("Approach/leave events unavailable.");
      } finally {
        inFlight = false;
      }
    }

    startBtn.addEventListener("click", async () => {
      try {
        await startCamera();
      } catch (error) {
        setStatus(`Camera start failed: ${error.message}`);
      }
    });
    stopBtn.addEventListener("click", stopCamera);
    intervalInput.addEventListener("change", () => {
      if (mediaStream) {
        scheduleLoop();
      }
    });
  </script>
</body>
</html>"""


def _get_live_monitor() -> ReceptionMonitor:
    global _live_monitor
    if _live_monitor is None:
        config = AppConfig(
            person_model=LIVE_PERSON_MODEL,
            device=LIVE_DEVICE,
            database_dir=LIVE_DATABASE_DIR,
            save_snapshots=False,
        )
        _live_monitor = ReceptionMonitor(config)
    return _live_monitor


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "root": str(ROOT_DIR), "port": PORT}


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_render_directory_page(ROOT_DIR))


@app.get("/live", response_class=HTMLResponse)
async def live_page() -> HTMLResponse:
    return HTMLResponse(_render_live_page())


@app.get("/browse/{relative_path:path}", response_class=HTMLResponse)
async def browse(relative_path: str) -> HTMLResponse:
    target = _resolve_relative_path(relative_path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found.")
    if target.is_dir():
        return HTMLResponse(_render_directory_page(target))
    return HTMLResponse(_render_file_page(target))


@app.get("/files/{relative_path:path}")
async def serve_file(relative_path: str):
    target = _resolve_relative_path(relative_path)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    media_type, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=target.name)


@app.post("/api/live-frame")
async def live_frame(frame: UploadFile = File(...), save_snapshot: str = Form(default="false")) -> Response:
    global _live_frame_index

    data = await frame.read()
    np_buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded frame.")

    with _live_monitor_lock:
        monitor = _get_live_monitor()
        frame_index = _live_frame_index
        _live_frame_index += 1
        annotated, event = monitor.process_frame(image, frame_index)
        if save_snapshot.lower() == "true":
            monitor.storage.save_snapshot(frame_index, annotated)

    ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), LIVE_JPEG_QUALITY])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode annotated frame.")

    headers = {
        "x-frame-index": str(frame_index),
        "x-match-count": str(len(event.matches)),
        "x-face-count": str(len(event.faces)),
        "x-person-count": str(len(event.persons)),
        "x-track-events": json.dumps(
            [
                {"track_id": item.track_id, "event_type": item.event_type, "person_id": item.person_id}
                for item in event.track_events
            ],
            ensure_ascii=False,
        ),
    }
    return Response(content=encoded.tobytes(), media_type="image/jpeg", headers=headers)


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots() -> str:
    return "User-agent: *\nDisallow: /\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
