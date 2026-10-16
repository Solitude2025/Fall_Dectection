"""Replay recorded camera observations through the production live profile.

Screen recordings require --crop x y width height and --frame-marker x y w h.
The marker is the application's Frame counter, in source-screen coordinates.
Repeated screen frames never enter inference or the 0.5 second alarm timer.
Reports stay outside upload history; no events are sent to Spring Boot.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from services.inference_service import InferenceService
from services.realtime_camera import prepare_detection_frame


def region(frame, rect):
    if rect is None:
        return frame
    x, y, w, h = rect
    return frame[y:y+h, x:x+w]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--crop', type=int, nargs=4)
    parser.add_argument('--frame-marker', type=int, nargs=4)
    parser.add_argument('--marker-threshold', type=float, default=2)
    parser.add_argument('--input-width', type=int)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--sample-fps', type=float, default=30)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pipeline = InferenceService()._build_pipeline(realtime=True)
    if args.input_width:
        pipeline.detector.input_size = (args.input_width, round(args.input_width * 9/16 / 32) * 32)
    if args.baseline:
        pipeline.transition_max_gap_seconds = .6
        if hasattr(pipeline, 'preserve_short_gaps_frames'):
            pipeline.preserve_short_gaps_frames = 0
        if hasattr(pipeline, 'preserve_short_gaps_seconds'):
            pipeline.preserve_short_gaps_seconds = 0
        pipeline.model_support_grace_seconds = 0
    pipeline.alarm_state.use_wall_clock = True
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open {args.source}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pipeline.fps = fps
    pipeline.alarm_state.set_fps(fps)
    base = datetime(2026, 10, 16, tzinfo=timezone.utc)
    writer = None
    rows, screen_rows, durations, cache, unique_frames = [], [], [], [], []
    last_marker = None
    last_sample_time = -1e3
    last_vis = None
    index = count = 0
    started = time.perf_counter()
    while True:
        ok, full = cap.read()
        if not ok:
            break
        seconds = index / fps
        marker = region(full, args.frame_marker) if args.frame_marker else None
        diff = float(cv2.absdiff(marker, last_marker).mean()) if marker is not None and last_marker is not None else None
        fresh = (diff is None or diff > args.marker_threshold) if marker is not None else seconds - last_sample_time >= 1/args.sample_fps - 1e-6
        if marker is not None:
            last_marker = marker.copy()
        frame = region(full, args.crop)
        screen_rows.append(dict(source_frame=index, seconds=round(seconds, 4), new_observation=bool(fresh), marker_difference=diff))
        if fresh:
            t0 = time.perf_counter()
            prepared = prepare_detection_frame(frame, pipeline.detector)
            result = pipeline.process_frame(prepared, index, timestamp=base + timedelta(seconds=seconds))
            durations.append((time.perf_counter() - t0)*1000)
            persons = {}
            for tid, det in result.visual_persons.items():
                persons[str(tid)] = dict(
                    bbox=det.bbox.tolist(), keypoints=det.keypoints.tolist(), confidence=float(det.confidence),
                    completeness=float((det.keypoints[:, 2] > .3).mean()),
                    core_confidence=float(min(det.keypoints[[5,6,11,12],2])),
                    sequence_length=len(pipeline._kp_buffers.get(tid, [])),
                    score=result.raw_fall_probs.get(tid), alarm=result.alarms.get(tid, False),
                    candidate=result.model_alarms.get(tid, False), decision=result.decision_details.get(tid))
            rows.append(dict(observation=count, source_frame=index, seconds=round(seconds, 4), persons=persons,
                             alarm=any(result.alarms.values()), event_updates=result.event_updates))
            cache.append(dict(source_frame=index, seconds=seconds, shape=list(prepared.shape[:2]),
                              detections=[dict(bbox=d.bbox.tolist(), keypoints=d.keypoints.tolist(), confidence=float(d.confidence)) for d in result.raw_detections]))
            last_vis = pipeline._draw_result(frame, result)
            cv2.putText(last_vis, f'Replay t={seconds:.2f}s | observed frame {count}', (8,frame.shape[0]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,255), 2)
            unique_frames.append(last_vis.copy())
            count += 1
            last_sample_time = seconds
        if not args.no_video:
            if writer is None:
                writer = cv2.VideoWriter(str(out/'replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1],frame.shape[0]))
            writer.write(last_vis)
        index += 1
    cap.release()
    if writer is not None:
        writer.release()
    for start in range(0,len(unique_frames),18):
        page=np.zeros((3*200,6*356,3),dtype=np.uint8)
        for j,vis in enumerate(unique_frames[start:start+18]):
            thumb=cv2.resize(vis,(356,200))
            page[(j//6)*200:(j//6+1)*200,(j%6)*356:(j%6+1)*356]=thumb
        cv2.imwrite(str(out/f'observations_{start:03}.jpg'),page)
    summary=dict(source=str(Path(args.source).resolve()), source_fps=fps, source_frames=index,
                 observations=count, input_size=list(pipeline.detector.input_size), baseline=args.baseline,
                 event_count=len(pipeline._fall_events),
                 alarm_times=[round(e.trigger_frame / fps,4) for e in pipeline._fall_events],
                 events=[e.to_dict() for e in pipeline._fall_events],
                 mean_processing_ms=round(float(np.mean(durations[8:])),2),
                 p95_processing_ms=round(float(np.percentile(durations[8:],95)),2),
                 wall_seconds=round(time.perf_counter()-started,2),
                 limitations='Screen crop contains previous model overlays and display enhancement; not original camera pixels.' if args.crop else None)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'observations.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'source_frames.json').write_text(json.dumps(screen_rows,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'detections.json').write_text(json.dumps(cache,ensure_ascii=False),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
