from __future__ import annotations

from dataclasses import dataclass, field

from recognition.models import BoundingBox, PersonDetection


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
    last_area: int = 0
    growth_streak: int = 0


@dataclass(slots=True)
class PersonTracker:
    approach_area_ratio: float
    approach_min_frames: int
    next_track_id: int = 1
    tracks: dict[int, TrackState] = field(default_factory=dict)

    def update(self, detections: list[tuple[BoundingBox, float]]) -> list[PersonDetection]:
        assigned: set[int] = set()
        results: list[PersonDetection] = []

        for bbox, confidence in detections:
            track = self._match_existing_track(bbox, assigned)
            if track is None:
                track = TrackState(track_id=self.next_track_id, bbox=bbox, last_area=bbox.area)
                self.tracks[track.track_id] = track
                self.next_track_id += 1
            else:
                new_area = max(1, bbox.area)
                old_area = max(1, track.last_area)
                if new_area / old_area >= self.approach_area_ratio:
                    track.growth_streak += 1
                else:
                    track.growth_streak = 0
                track.bbox = bbox
                track.last_area = new_area
                track.frames_seen += 1

            assigned.add(track.track_id)
            results.append(
                PersonDetection(
                    track_id=track.track_id,
                    confidence=confidence,
                    bbox=bbox,
                    approaching=track.growth_streak >= self.approach_min_frames,
                )
            )

        self.tracks = {track_id: track for track_id, track in self.tracks.items() if track_id in assigned}
        return results

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

