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
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse


ROOT_DIR = Path(os.getenv("RECOGNITION_BROWSE_ROOT", Path(__file__).resolve().parent)).resolve()
PORT = int(os.getenv("PORT", "8000"))
TITLE = "Recognition Browser"

app = FastAPI(title=TITLE, version="1.0.0")


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


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, "root": str(ROOT_DIR), "port": PORT}


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    return HTMLResponse(_render_directory_page(ROOT_DIR))


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


@app.get("/robots.txt", response_class=PlainTextResponse)
async def robots() -> str:
    return "User-agent: *\nDisallow: /\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
