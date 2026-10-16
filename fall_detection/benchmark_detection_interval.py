"""Compare skipped-frame and per-frame pose detection on the same videos."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from services.inference_service import InferenceService


# Put videos you are allowed to process in the local ``test/`` directory.
# The directory is intentionally ignored by Git because recordings may contain
# personal data and are usually too large for a source repository.
VIDEOS = sorted(Path(__file__).resolve().parent.joinpath("test").glob("*.mp4"))


def run(pipeline, video: Path, interval: int) -> dict:
    pipeline.detection_interval = interval
    started = time.perf_counter()
    result = pipeline.process_video(str(video))
    elapsed = time.perf_counter() - started
    detection_frames = sum(1 for item in result.frame_results if item.is_detect_frame)
    detection_frames_with_person = sum(
        1 for item in result.frame_results
        if item.is_detect_frame and item.raw_detections
    )
    all_frames_with_visual_person = sum(
        1 for item in result.frame_results if item.visual_persons
    )
    return {
        "video": video.name,
        "interval": interval,
        "total_frames": result.total_frames,
        "source_fps": round(result.total_frames / result.total_time, 3)
        if result.total_time else 0,
        "processing_seconds": round(elapsed, 3),
        "avg_latency_ms": round(result.avg_latency_ms, 3),
        "events": len(result.fall_events),
        "event_frames": [
            {
                "start": event.start_frame,
                "trigger": event.trigger_frame,
                "end": event.end_frame,
            }
            for event in result.fall_events
        ],
        "yolo_detection_frame_ratio": round(
            detection_frames / result.total_frames, 4
        ) if result.total_frames else 0,
        "yolo_positive_detection_frame_ratio": round(
            detection_frames_with_person / result.total_frames, 4
        ) if result.total_frames else 0,
        "visual_overlay_frame_ratio": round(
            all_frames_with_visual_person / result.total_frames, 4
        ) if result.total_frames else 0,
    }


def main() -> int:
    videos = [video for video in VIDEOS if video.exists()]
    if not videos:
        print("No test videos found", file=sys.stderr)
        return 2
    service = InferenceService()
    pipeline = service._build_pipeline(realtime=False)
    pipeline.save_video = False
    pipeline.output_dir = "eval_report/interval_benchmark"
    Path(pipeline.output_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for interval in (2, 1):
        for video in videos:
            print(f"running interval={interval}: {video.name}", flush=True)
            rows.append(run(pipeline, video, interval))
    output = Path("eval_report/detection_interval_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
