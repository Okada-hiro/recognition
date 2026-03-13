from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from recognition.config import AppConfig
from recognition.models import BoundingBox, FaceDetection, FaceMatch


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


class InsightFaceAnalyzer:
    def __init__(self, config: AppConfig) -> None:
        try:
            package_root = Path(__file__).resolve().parents[1] / "insightface" / "python-package"
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            import onnxruntime
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                "insightface could not be imported. Install insightface and onnxruntime to enable face analysis."
            ) from exc

        provider_names = set(onnxruntime.get_available_providers())
        use_cuda = config.device != "cpu" and "CUDAExecutionProvider" in provider_names
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        ctx_id = 0 if use_cuda else -1
        self._app = FaceAnalysis(
            name=config.insightface_model_name,
            root=str(config.insightface_root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self._app.prepare(
            ctx_id=ctx_id,
            det_thresh=config.face_confidence,
            det_size=(config.insightface_det_size, config.insightface_det_size),
        )

    def analyze(self, frame_bgr: np.ndarray) -> list[Any]:
        return self._app.get(frame_bgr)

    def detect_faces(self, frame_bgr: np.ndarray) -> list[tuple[FaceDetection, np.ndarray]]:
        detections: list[tuple[FaceDetection, np.ndarray]] = []
        for face in self.analyze(frame_bgr):
            bbox = face.bbox.astype(int).tolist()
            landmarks: dict[str, list[float]] = {}
            if face.kps is not None:
                names = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
                for idx, point in enumerate(face.kps.tolist()):
                    key = names[idx] if idx < len(names) else f"point_{idx}"
                    landmarks[key] = [float(point[0]), float(point[1])]
            embedding = face.normed_embedding if face.normed_embedding is not None else face.embedding
            if embedding is None:
                continue
            detections.append(
                (
                    FaceDetection(
                        bbox=BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]), x2=int(bbox[2]), y2=int(bbox[3])),
                        confidence=float(face.det_score or 0.0),
                        landmarks=landmarks,
                    ),
                    np.asarray(embedding, dtype=np.float32),
                )
            )
        return detections

    def embed_face(self, face_image_bgr: np.ndarray) -> np.ndarray:
        faces = self.detect_faces(face_image_bgr)
        if not faces:
            raise ValueError("InsightFace did not detect a face in the image.")
        return faces[0][1]


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
