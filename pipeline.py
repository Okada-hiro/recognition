from __future__ import annotations

import cv2
import numpy as np

from recognition.config import AppConfig
from recognition.database import FaceDatabase
from recognition.detectors import RetinaFaceDetector, YoloPersonDetector
from recognition.face_recognition import DeepFaceEmbedder, FaceMatcher
from recognition.models import BoundingBox, EventRecord, FaceDetection, FaceMatch
from recognition.storage import EventStorage
from recognition.tracker import PersonTracker, iou


class ReceptionMonitor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.person_detector = YoloPersonDetector(config.person_model, config.person_confidence, config.device)
        self.face_detector = RetinaFaceDetector(config.face_confidence)
        self.embedder = DeepFaceEmbedder(config.face_model_name)
        self.database = FaceDatabase(config, self.embedder)
        self.database.build()
        self.matcher = FaceMatcher(config.face_match_threshold)
        self.storage = EventStorage(config)
        self.tracker = PersonTracker(config.approach_area_ratio, config.approach_min_frames)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, EventRecord]:
        person_candidates = self.person_detector.detect(frame)
        persons = self.tracker.update(person_candidates)
        faces = self._attach_faces_to_persons(frame, persons)
        matches: list[FaceMatch] = []
        notes: list[str] = []

        for face in faces:
            crop = self._crop(frame, face.bbox)
            if crop is None:
                continue
            try:
                embedding = self.embedder.embed_face(crop)
            except Exception as exc:
                notes.append(f"face_embedding_failed: {exc}")
                continue
            match = self.matcher.match(embedding, self.database.embeddings)
            if match is not None:
                match.source_track_id = face.person_track_id
                matches.append(match)

        annotated = self._annotate(frame.copy(), persons, faces, matches)
        event = EventRecord.create(frame_index=frame_index, persons=persons, faces=faces, matches=matches, notes=notes)
        self.storage.save_event(event)
        if self.config.save_snapshots and (matches or any(person.approaching for person in persons)):
            self.storage.save_snapshot(frame_index, annotated)
        return annotated, event

    def _attach_faces_to_persons(self, frame: np.ndarray, persons: list) -> list[FaceDetection]:
        faces = self.face_detector.detect(frame)
        for face in faces:
            best_track_id: int | None = None
            best_iou = 0.0
            for person in persons:
                overlap = iou(face.bbox, person.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = person.track_id
            face.person_track_id = best_track_id
        return faces

    @staticmethod
    def _crop(frame: np.ndarray, bbox: BoundingBox) -> np.ndarray | None:
        crop = frame[max(0, bbox.y1) : max(0, bbox.y2), max(0, bbox.x1) : max(0, bbox.x2)]
        if crop.size == 0:
            return None
        return crop

    @staticmethod
    def _annotate(frame: np.ndarray, persons: list, faces: list, matches: list[FaceMatch]) -> np.ndarray:
        match_by_track = {match.source_track_id: match for match in matches if match.source_track_id is not None}
        for person in persons:
            color = (0, 200, 255) if person.approaching else (255, 128, 0)
            cv2.rectangle(frame, (person.bbox.x1, person.bbox.y1), (person.bbox.x2, person.bbox.y2), color, 2)
            label = f"person#{person.track_id}"
            if person.approaching:
                label += " approaching"
            match = match_by_track.get(person.track_id)
            if match is not None:
                label += f" {match.person_id}"
            cv2.putText(frame, label, (person.bbox.x1, max(20, person.bbox.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for face in faces:
            cv2.rectangle(frame, (face.bbox.x1, face.bbox.y1), (face.bbox.x2, face.bbox.y2), (0, 255, 0), 2)
            label = f"face {face.confidence:.2f}"
            cv2.putText(frame, label, (face.bbox.x1, max(20, face.bbox.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        y = 24
        for match in matches:
            cv2.putText(
                frame,
                f"match track={match.source_track_id} {match.person_id} score={match.score:.3f}",
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 220, 50),
                2,
            )
            y += 24
        return frame
