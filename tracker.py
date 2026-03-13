from __future__ import annotations

from dataclasses import dataclass, field

from recognition.models import BoundingBox, PersonDetection, TrackEvent


def _intersection(a: BoundingBox, b: BoundingBox) -> int:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    inter = _intersection(a, b)
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: BoundingBox
    frames_seen: int = 1
    initial_area: int = 0
    last_area: int = 0
    missing_frames: int = 0
    approach_announced: bool = False


@dataclass(slots=True)
class PersonTracker:
    approach_area_ratio: float
    approach_min_frames: int
    max_missing_frames: int
    next_track_id: int = 1
    tracks: dict[int, TrackState] = field(default_factory=dict)

    def update(self, detections: list[tuple[BoundingBox, float]]) -> tuple[list[PersonDetection], list[TrackEvent]]:
        assigned: set[int] = set()
        results: list[PersonDetection] = []
        events: list[TrackEvent] = []

        for bbox, confidence in detections:
            track = self._match_existing_track(bbox, assigned)
            if track is None:
                track = TrackState(
                    track_id=self.next_track_id,
                    bbox=bbox,
                    initial_area=max(1, bbox.area),
                    last_area=max(1, bbox.area),
                )
                self.tracks[track.track_id] = track
                self.next_track_id += 1
            else:
                new_area = max(1, bbox.area)
                track.bbox = bbox
                track.last_area = new_area
                track.frames_seen += 1
                track.missing_frames = 0

            baseline_area = max(1, track.initial_area)
            approaching = (
                track.frames_seen >= self.approach_min_frames
                and (track.last_area / baseline_area) >= self.approach_area_ratio
            )
            if approaching and not track.approach_announced:
                track.approach_announced = True
                events.append(TrackEvent(track_id=track.track_id, event_type="approached"))

            assigned.add(track.track_id)
            results.append(
                PersonDetection(
                    track_id=track.track_id,
                    confidence=confidence,
                    bbox=bbox,
                    approaching=approaching,
                )
            )

        stale_track_ids: list[int] = []
        for track_id, track in self.tracks.items():
            if track_id in assigned:
                continue
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                if track.approach_announced:
                    events.append(TrackEvent(track_id=track.track_id, event_type="left"))
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del self.tracks[track_id]

        return results, events

    def _match_existing_track(self, bbox: BoundingBox, assigned: set[int]) -> TrackState | None:
        best_track: TrackState | None = None
        best_iou = 0.0
        for track_id, track in self.tracks.items():
            if track_id in assigned:
                continue
            score = iou(track.bbox, bbox)
            if score > best_iou:
                best_iou = score
                best_track = track
        return best_track if best_iou >= 0.2 else None
