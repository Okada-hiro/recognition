from __future__ import annotations

import argparse
from pathlib import Path

from recognition.config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reception monitor POC")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--input-video", default=None, help="Path to an input video file.")
    parser.add_argument("--output-video", default=None, help="Path to save the annotated video.")
    parser.add_argument("--person-model", default="yolo11n.pt", help="YOLO weights path or model name.")
    parser.add_argument("--device", default="auto", help="Inference device. Use auto, cpu, or CUDA device like 0.")
    parser.add_argument("--database-dir", default=None, help="Directory that stores employee face images.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save annotated frames when events happen.")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable cv2.imshow. Useful for Colab, RunPod, and batch video processing.",
    )
    return parser


def _create_video_writer(cv2, output_path: Path, capture) -> "cv2.VideoWriter":
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}.")
    return writer


def main() -> int:
    import cv2

    from recognition.pipeline import ReceptionMonitor

    args = build_parser().parse_args()
    if args.output_video and not args.input_video:
        raise ValueError("--output-video requires --input-video.")

    config = AppConfig(
        camera_index=args.camera_index,
        person_model=args.person_model,
        device=args.device,
        save_snapshots=args.save_snapshots,
        database_dir=Path(args.database_dir).resolve() if args.database_dir else AppConfig().database_dir,
    )
    monitor = ReceptionMonitor(config)

    if args.input_video:
        input_video = Path(args.input_video).resolve()
        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open input video {input_video}.")
        default_output = input_video.with_name(f"{input_video.stem}_annotated.mp4")
        output_path = Path(args.output_video).resolve() if args.output_video else default_output
        writer = _create_video_writer(cv2, output_path, capture)
        display = not args.no_display
    else:
        capture = cv2.VideoCapture(config.camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {config.camera_index}.")
        writer = None
        display = not args.no_display

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, _ = monitor.process_frame(frame, frame_index)
            if writer is not None:
                writer.write(annotated)
            if display:
                cv2.imshow("reception-monitor", annotated)
            frame_index += 1
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
