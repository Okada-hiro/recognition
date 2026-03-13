from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from recognition.config import AppConfig
from recognition.face_recognition import InsightFaceAnalyzer


class FaceDatabase:
    def __init__(self, config: AppConfig, embedder: InsightFaceAnalyzer) -> None:
        self.config = config
        self.embedder = embedder
        self.embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}

    def build(self) -> None:
        self.embeddings = {}
        self.config.database_dir.mkdir(parents=True, exist_ok=True)

        for person_dir in sorted(path for path in self.config.database_dir.iterdir() if path.is_dir()):
            person_embeddings: list[tuple[str, np.ndarray]] = []
            for image_path in sorted(person_dir.iterdir()):
                if image_path.suffix.lower() not in self.config.image_extensions:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                person_embeddings.append((str(image_path), self.embedder.embed_face(image)))
            if person_embeddings:
                self.embeddings[person_dir.name] = person_embeddings
