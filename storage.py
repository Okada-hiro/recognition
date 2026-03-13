from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from recognition.config import AppConfig
from recognition.models import EventRecord


class EventStorage:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.config.logs_dir / "events.jsonl"

    def save_event(self, event: EventRecord) -> None:
        with self.events_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

    def save_snapshot(self, frame_index: int, frame: np.ndarray) -> Path:
        snapshot_path = self.config.snapshots_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(snapshot_path), frame)
        return snapshot_path

