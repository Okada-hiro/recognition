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

import asyncio
import html
import io
import json
import mimetypes
import os
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path
from urllib import request
from urllib.parse import quote

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
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
VOICE_TALK_DIR = (ROOT_DIR.parent / "lab_voice_talk").resolve()
VOICE_TALK_NOTIFY_BASE = os.getenv("RECOGNITION_VOICE_TALK_NOTIFY_BASE", "http://127.0.0.1:8002").rstrip("/")
VOICE_TALK_HTTP_BASE = os.getenv("RECOGNITION_VOICE_TALK_HTTP_BASE", VOICE_TALK_NOTIFY_BASE).rstrip("/")
VOICE_TALK_WS_URL = os.getenv("RECOGNITION_VOICE_TALK_WS_URL", "ws://127.0.0.1:8002/ws")

app = FastAPI(title=TITLE, version="1.0.0")
_live_monitor_lock = threading.Lock()
_live_monitor: ReceptionMonitor | None = None
_live_frame_index = 0
_voice_tts_lock = threading.Lock()
_voice_tts_module = None
_voice_tts_error: str | None = None


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
      <audio id="voiceAudio" hidden></audio>
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
    const voiceAudioEl = document.getElementById("voiceAudio");
    const intervalInput = document.getElementById("intervalInput");
    const widthInput = document.getElementById("widthInput");

    let mediaStream = null;
    let timerId = null;
    let inFlight = false;
    let voiceQueue = Promise.resolve();

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function setEventStatus(message) {
      eventStatusEl.textContent = message;
    }

    async function playTrackEvents(trackEvents) {
      for (const event of trackEvents) {
        voiceQueue = voiceQueue.then(() => speakTrackEvent(event)).catch(() => {});
      }
      return voiceQueue;
    }

    async function speakTrackEvent(event) {
      const params = new URLSearchParams({
        event_type: event.event_type,
        track_id: String(event.track_id),
      });
      if (event.person_id) {
        params.set("person_id", event.person_id);
      }
      const response = await fetch(`/api/live-utterance?${params.toString()}`);
      if (!response.ok) {
        return;
      }
      const audioBlob = await response.blob();
      const objectUrl = URL.createObjectURL(audioBlob);
      const prevUrl = voiceAudioEl.dataset.url;
      voiceAudioEl.src = objectUrl;
      voiceAudioEl.dataset.url = objectUrl;
      if (prevUrl) {
        URL.revokeObjectURL(prevUrl);
      }
      await voiceAudioEl.play();
      await new Promise((resolve) => {
        const onDone = () => {
          voiceAudioEl.removeEventListener("ended", onDone);
          voiceAudioEl.removeEventListener("error", onDone);
          resolve();
        };
        voiceAudioEl.addEventListener("ended", onDone);
        voiceAudioEl.addEventListener("error", onDone);
      });
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
            playTrackEvents(trackEvents);
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


def _render_reception_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reception Assistant</title>
  <style>
    :root {
      --bg: #f6f2e8;
      --panel: #fffaf0;
      --ink: #1c1a18;
      --line: #d7c8a8;
    }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, var(--bg), #efe5d1);
      color: var(--ink);
    }
    main {
      max-width: 1480px;
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
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 1200px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
    iframe {
      width: 100%;
      min-height: 920px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }
    a {
      color: #0d5c63;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .note {
      color: #6b665d;
    }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>Reception Assistant</h1>
      <p><a href="/">back to browser root</a></p>
      <p class="note">
        Single-page receptionist view. Left: camera recognition. Right: microphone conversation UI.
      </p>
      <div class="grid">
        <section>
          <h2>Vision</h2>
          <iframe src="/live" allow="camera"></iframe>
        </section>
        <section>
          <h2>Voice</h2>
          <iframe src="/voice-ui" allow="microphone"></iframe>
        </section>
      </div>
    </div>
  </main>
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


def _get_voice_tts_module():
    global _voice_tts_module, _voice_tts_error
    if _voice_tts_module is not None:
        return _voice_tts_module
    if _voice_tts_error is not None:
        raise RuntimeError(_voice_tts_error)

    with _voice_tts_lock:
        if _voice_tts_module is not None:
            return _voice_tts_module
        if _voice_tts_error is not None:
            raise RuntimeError(_voice_tts_error)
        try:
            if str(VOICE_TALK_DIR) not in sys.path:
                sys.path.insert(0, str(VOICE_TALK_DIR))
            import parallel_faster_text_to_speech as tts_module
        except Exception as exc:
            _voice_tts_error = f"voice_tts_import_failed: {exc}"
            raise RuntimeError(_voice_tts_error) from exc
        _voice_tts_module = tts_module
        return _voice_tts_module


def _build_utterance_text(event_type: str, person_id: str | None) -> str:
    if event_type == "approached":
        if person_id:
            return f"{person_id}さん、こんにちは。受付です。"
        return "こんにちは。受付です。"
    if event_type == "left":
        if person_id:
            return f"{person_id}さん、ありがとうございました。"
        return "ありがとうございました。"
    return "こんにちは。"


def _pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buffer.getvalue()


def _proxy_voice_talk(path: str = "/", method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    endpoint = f"{VOICE_TALK_HTTP_BASE}{path}"
    req = request.Request(endpoint, data=body, method=method)
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with request.urlopen(req, timeout=10) as response:
        content_type = response.headers.get_content_type()
        return response.read(), content_type


def _rewrite_voice_ui(html_text: str) -> str:
    html_text = html_text.replace("fetch('/enable-registration'", "fetch('/voice-enable-registration'")
    html_text = html_text.replace('fetch("/enable-registration"', 'fetch("/voice-enable-registration"')
    html_text = html_text.replace("window.location.host + '/ws'", "window.location.host + '/voice-ws'")
    html_text = html_text.replace('window.location.host + "/ws"', 'window.location.host + "/voice-ws"')
    return html_text


def _notify_voice_talk(track_events) -> None:
    if not VOICE_TALK_NOTIFY_BASE:
        return
    for track_event in track_events:
        if track_event.event_type not in {"approached", "left"}:
            continue
        endpoint = f"{VOICE_TALK_NOTIFY_BASE}/recognition/{'approach' if track_event.event_type == 'approached' else 'leave'}"
        payload = json.dumps({"person_id": track_event.person_id}, ensure_ascii=False).encode("utf-8")
        req = request.Request(endpoint, data=payload, headers={"content-type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=1.5):
                pass
        except Exception:
            continue


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "root": str(ROOT_DIR), "port": PORT}


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_render_directory_page(ROOT_DIR))


@app.get("/live", response_class=HTMLResponse)
async def live_page() -> HTMLResponse:
    return HTMLResponse(_render_live_page())


@app.get("/reception", response_class=HTMLResponse)
async def reception_page() -> HTMLResponse:
    return HTMLResponse(_render_reception_page())


@app.get("/voice-ui", response_class=HTMLResponse)
async def voice_ui() -> HTMLResponse:
    try:
        body, _content_type = await asyncio.to_thread(_proxy_voice_talk, "/")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"voice_ui_proxy_failed: {exc}") from exc
    html_text = body.decode("utf-8", errors="replace")
    return HTMLResponse(_rewrite_voice_ui(html_text))


@app.post("/voice-enable-registration")
async def voice_enable_registration() -> Response:
    try:
        body, content_type = await asyncio.to_thread(_proxy_voice_talk, "/enable-registration", "POST")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"voice_registration_proxy_failed: {exc}") from exc
    return Response(content=body, media_type=content_type)


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

    if event.track_events:
        await asyncio.to_thread(_notify_voice_talk, event.track_events)

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


@app.get("/api/live-utterance")
async def live_utterance(event_type: str, track_id: int, person_id: str | None = None) -> Response:
    del track_id
    try:
        tts_module = _get_voice_tts_module()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    text = _build_utterance_text(event_type, person_id)
    try:
        pcm_bytes = await asyncio.to_thread(tts_module.synthesize_speech_to_memory, text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"voice_tts_failed: {exc}") from exc
    if not pcm_bytes:
        raise HTTPException(status_code=500, detail="voice_tts_empty")

    wav_bytes = _pcm16_to_wav_bytes(pcm_bytes)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.websocket("/voice-ws")
async def voice_ws_proxy(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        import websockets
    except Exception:
        await websocket.close(code=1011)
        return

    try:
        async with websockets.connect(VOICE_TALK_WS_URL, max_size=None) as upstream:
            async def client_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        break
                    if message.get("bytes") is not None:
                        await upstream.send(message["bytes"])
                    elif message.get("text") is not None:
                        await upstream.send(message["text"])

            async def upstream_to_client() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            client_task = asyncio.create_task(client_to_upstream())
            upstream_task = asyncio.create_task(upstream_to_client())
            done, pending = await asyncio.wait({client_task, upstream_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, WebSocketDisconnect):
                    raise exc
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close(code=1011)


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots() -> str:
    return "User-agent: *\nDisallow: /\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
