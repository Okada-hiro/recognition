from __future__ import annotations

import numpy as np

from recognition.models import BoundingBox


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
