"""Measure warmed-up camera capture and real-time inference, without a web UI."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def capture_probe():
    from services.realtime_camera import configure_capture
    rows = []
    cv2.setNumThreads(2)
    for width, height, exposure in [(1280, 720, -4), (1280, 720, -5), (640, 480, -5)]:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            if not cap.isOpened():
                raise RuntimeError("Camera is not available; stop live detection first")
            configure_capture(cap, dict(camera_width=width, camera_height=height,
                                       camera_fps=30, camera_auto_exposure=.25, camera_exposure=exposure))
            for _ in range(20):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Camera read failed")
            count = 0
            brightness = []
            started = time.perf_counter()
            while time.perf_counter() - started < 2:
                ok, frame = cap.read()
                if not ok:
                    break
                count += 1
                brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            elapsed = time.perf_counter() - started
            row = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                       backend=cap.getBackendName(), exposure=cap.get(cv2.CAP_PROP_EXPOSURE),
                       capture_fps=round(count / elapsed, 2),
                       brightness=round(statistics.mean(brightness), 2))
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            cap.release()
    return rows


def pipeline_probe():
    import torch
    import numpy as np
    from services.inference_service import InferenceService

    # Profile a real moving person, so TCN and skeleton drawing are included.
    video = Path(__file__).resolve().parents[2] / "测试视频" / "Desktop 2026.10.15 - 18.58.26.10.mp4"
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    frames = []
    for _ in range(60):
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (1280, 720)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames: {video}")
    service = InferenceService()
    pipeline = service._build_pipeline(realtime=True)
    pipeline.detection_interval = 1
    rows = []
    for limited in (False, True):
        if limited:
            cv2.setNumThreads(2)
            torch.set_num_threads(2)
        pipeline.reset()
        timings = []
        for index, frame in enumerate(frames):
            t0 = time.perf_counter()
            preview = pipeline.detector.preprocess_for_lowlight(cv2.resize(frame, (512, 288)))
            t1 = time.perf_counter()
            result = pipeline.process_frame(preview, index)
            t2 = time.perf_counter()
            rendered = pipeline._draw_result(frame, result)
            t3 = time.perf_counter()
            cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 88])
            t4 = time.perf_counter()
            if index >= 15:
                timings.append(np.diff([t0,t1,t2,t3,t4]) * 1000)
        timings = np.array(timings)
        row = dict(opencv_threads=cv2.getNumThreads(), torch_threads=torch.get_num_threads(),
                   mean_ms=dict(zip(['enhance','inference','draw','jpeg'], timings.mean(axis=0).round(2).tolist())),
                   total_mean_ms=round(float(timings.sum(axis=1).mean()), 2),
                   total_p95_ms=round(float(np.percentile(timings.sum(axis=1), 95)), 2),
                   processing_fps=round(1000/float(timings.sum(axis=1).mean()), 2))
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', action='store_true')
    args = parser.parse_args()
    result = pipeline_probe() if args.pipeline else capture_probe()
    output = Path('eval_report') / ('realtime_pipeline_probe.json' if args.pipeline else 'camera_capture_probe.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
