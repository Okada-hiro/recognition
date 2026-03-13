from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(slots=True)
class PersonDetection:
    track_id: int
    confidence: float
    bbox: BoundingBox
    approaching: bool = False


@dataclass(slots=True)
class FaceDetection:
    bbox: BoundingBox
    confidence: float
    landmarks: dict[str, list[float]] = field(default_factory=dict)
    person_track_id: int | None = None


@dataclass(slots=True)
class FaceMatch:
    person_id: str
    score: float
    distance: float
    matched_image: str | None = None
    source_track_id: int | None = None


@dataclass(slots=True)
class TrackEvent:
    track_id: int
    event_type: str
    person_id: str | None = None


@dataclass(slots=True)
class EventRecord:
    timestamp: str
    frame_index: int
    persons: list[PersonDetection]
    faces: list[FaceDetection]
    matches: list[FaceMatch]
    track_events: list[TrackEvent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        frame_index: int,
        persons: list[PersonDetection],
        faces: list[FaceDetection],
        matches: list[FaceMatch],
        track_events: list[TrackEvent] | None = None,
        notes: list[str] | None = None,
    ) -> "EventRecord":
        return cls(
            timestamp=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            frame_index=frame_index,
            persons=persons,
            faces=faces,
            matches=matches,
            track_events=track_events or [],
            notes=notes or [],
        )
