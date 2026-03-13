from __future__ import annotations

import cv2
import numpy as np

from recognition.config import AppConfig
from recognition.database import FaceDatabase
from recognition.detectors import YoloPersonDetector
from recognition.face_recognition import FaceMatcher, InsightFaceAnalyzer
from recognition.models import EventRecord, FaceDetection, FaceMatch, TrackEvent
from recognition.storage import EventStorage
from recognition.tracker import PersonTracker, iou


class ReceptionMonitor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.person_detector = YoloPersonDetector(config.person_model, config.person_confidence, config.device)
        self.face_analyzer = InsightFaceAnalyzer(config)
        self.embedder = self.face_analyzer
        self.database = FaceDatabase(config, self.embedder)
        self.database.build()
        self.matcher = FaceMatcher(config.face_match_threshold)
        self.storage = EventStorage(config)
        self.tracker = PersonTracker(
            config.approach_area_ratio,
            config.approach_min_frames,
            config.track_max_missing_frames,
        )
        self.track_identities: dict[int, str] = {}

    def process_frame(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, EventRecord]:
        person_candidates = self.person_detector.detect(frame)
        persons, track_events = self.tracker.update(person_candidates)
        face_items = self._attach_faces_to_persons(frame, persons)
        faces = [face for face, _embedding in face_items]
        matches: list[FaceMatch] = []
        notes: list[str] = []

        for face, embedding in face_items:
            try:
                embedding = np.asarray(embedding, dtype=np.float32)
            except Exception as exc:
                notes.append(f"face_embedding_failed: {exc}")
                continue
            match = self.matcher.match(embedding, self.database.embeddings)
            if match is not None:
                match.source_track_id = face.person_track_id
                if match.source_track_id is not None:
                    self.track_identities[match.source_track_id] = match.person_id
                matches.append(match)

        resolved_track_events = self._resolve_track_events(track_events)
        for track_event in resolved_track_events:
            if track_event.event_type == "approached":
                notes.append(
                    f"track_approached: track={track_event.track_id}"
                    + (f" person_id={track_event.person_id}" if track_event.person_id else "")
                )
            elif track_event.event_type == "left":
                notes.append(
                    f"track_left: track={track_event.track_id}"
                    + (f" person_id={track_event.person_id}" if track_event.person_id else "")
                )

        annotated = self._annotate(frame.copy(), persons, faces, matches, resolved_track_events)
        event = EventRecord.create(
            frame_index=frame_index,
            persons=persons,
            faces=faces,
            matches=matches,
            track_events=resolved_track_events,
            notes=notes,
        )
        self.storage.save_event(event)
        if self.config.save_snapshots and (matches or resolved_track_events or any(person.approaching for person in persons)):
            self.storage.save_snapshot(frame_index, annotated)
        return annotated, event

    def _resolve_track_events(self, track_events: list[TrackEvent]) -> list[TrackEvent]:
        resolved_events: list[TrackEvent] = []
        for track_event in track_events:
            person_id = self.track_identities.get(track_event.track_id)
            resolved_events.append(
                TrackEvent(
                    track_id=track_event.track_id,
                    event_type=track_event.event_type,
                    person_id=person_id,
                )
            )
            if track_event.event_type == "left":
                self.track_identities.pop(track_event.track_id, None)
        return resolved_events

    def _attach_faces_to_persons(self, frame: np.ndarray, persons: list) -> list[tuple[FaceDetection, np.ndarray]]:
        face_items = self.face_analyzer.detect_faces(frame)
        for face, _embedding in face_items:
            best_track_id: int | None = None
            best_iou = 0.0
            for person in persons:
                overlap = iou(face.bbox, person.bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = person.track_id
            face.person_track_id = best_track_id
        return face_items

    @staticmethod
    def _annotate(
        frame: np.ndarray,
        persons: list,
        faces: list,
        matches: list[FaceMatch],
        track_events: list[TrackEvent],
    ) -> np.ndarray:
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
        for track_event in track_events:
            label = f"{track_event.event_type} track={track_event.track_id}"
            if track_event.person_id:
                label += f" {track_event.person_id}"
            color = (20, 180, 20) if track_event.event_type == "approached" else (30, 30, 220)
            cv2.putText(frame, label, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24
        return frame
