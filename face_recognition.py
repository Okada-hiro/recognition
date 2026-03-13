from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recognition.models import FaceMatch


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


class DeepFaceEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            from deepface import DeepFace
        except Exception as exc:
            raise RuntimeError(
                "deepface is not installed. Install it to enable ArcFace-based identification."
            ) from exc

        self.model_name = model_name
        self._deepface = DeepFace

    def embed_face(self, face_image_bgr: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        response: list[dict[str, Any]] = self._deepface.represent(
            img_path=rgb_image,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
        )
        if not response:
            raise ValueError("DeepFace did not return an embedding.")
        return np.asarray(response[0]["embedding"], dtype=np.float32)


class FaceMatcher:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def match(
        self,
        query_embedding: np.ndarray,
        embeddings: dict[str, list[tuple[str, np.ndarray]]],
    ) -> FaceMatch | None:
        best: FaceMatch | None = None
        for person_id, items in embeddings.items():
            for image_path, embedding in items:
                distance = cosine_distance(query_embedding, embedding)
                score = 1.0 - distance
                if best is None or distance < best.distance:
                    best = FaceMatch(
                        person_id=person_id,
                        score=score,
                        distance=distance,
                        matched_image=image_path,
                    )
        if best is None or best.distance > self.threshold:
            return None
        return best
