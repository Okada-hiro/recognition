from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class AppConfig:
    database_dir: Path = PROJECT_ROOT / "data_base"
    logs_dir: Path = PROJECT_ROOT / "recognition" / "logs"
    snapshots_dir: Path = PROJECT_ROOT / "recognition" / "snapshots"
    person_model: str = "yolo11n.pt"
    device: str = "auto"
    insightface_model_name: str = "buffalo_l"
    insightface_det_size: int = 640
    insightface_root: Path = PROJECT_ROOT / "models" / "insightface"
    camera_index: int = 0
    person_confidence: float = 0.45
    face_confidence: float = 0.55
    face_match_threshold: float = 0.35
    approach_area_ratio: float = 1.08
    approach_min_frames: int = 2
    track_max_missing_frames: int = 6
    save_snapshots: bool = True
    image_extensions: tuple[str, ...] = field(default_factory=lambda: (".jpg", ".jpeg", ".png"))
