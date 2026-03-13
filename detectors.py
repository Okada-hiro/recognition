from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from recognition.models import BoundingBox, FaceDetection


class YoloPersonDetector:
    def __init__(self, model_path: str, confidence: float, device: str = "auto") -> None:
        self.confidence = confidence
        try:
            from ultralytics import YOLO
            import torch
        except Exception as exc:
            raise RuntimeError(
                "ultralytics could not be imported. Install dependencies before running this pipeline."
            ) from exc

        self.model_path = model_path
        if device == "auto":
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[tuple[BoundingBox, float]]:
        results = self.model.predict(
            frame,
            classes=[0],
            conf=self.confidence,
            verbose=False,
            device=self.device,
        )
        detections: list[tuple[BoundingBox, float]] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    (
                        BoundingBox(
                            x1=int(xyxy[0]),
                            y1=int(xyxy[1]),
                            x2=int(xyxy[2]),
                            y2=int(xyxy[3]),
                        ),
                        conf,
                    )
                )
        return detections


class RetinaFaceDetector:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        try:
            import sys

            retinaface_root = Path(__file__).resolve().parents[1] / "retinaface"
            if str(retinaface_root) not in sys.path:
                sys.path.insert(0, str(retinaface_root))
            from retinaface import RetinaFace
        except Exception as exc:
            raise RuntimeError(
                "retinaface could not be imported. Install TensorFlow and retinaface dependencies before running."
            ) from exc

        self._retinaface = RetinaFace

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_faces: dict[str, Any] = self._retinaface.detect_faces(rgb_frame, threshold=self.confidence)
        detections: list[FaceDetection] = []
        for face in raw_faces.values():
            bbox = face["facial_area"]
            detections.append(
                FaceDetection(
                    bbox=BoundingBox(x1=int(bbox[0]), y1=int(bbox[1]), x2=int(bbox[2]), y2=int(bbox[3])),
                    confidence=float(face["score"]),
                    landmarks={k: [float(v[0]), float(v[1])] for k, v in face["landmarks"].items()},
                )
            )
        return detections
