"""Local camera inference service.

This process owns the laptop camera and the existing Python model.  Spring
Boot does not need to know anything about PyTorch: it receives the small JSON
event messages emitted here and stores them in MySQL.

Run from the project root:

    python -m services.inference_service --camera-index 0

    The service exposes a small HTTP API on port 8001. Spring Boot on port
    8080 forwards the camera preview and owns the browser-facing APIs.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, Generator, Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import torch
import uvicorn
import yaml

from models.fall_detector import create_fall_detector
from models.rule_refinement import RuleConfig
from pipeline.detector import YOLOPoseDetector, YOLOPoseDetectorSim
from pipeline.inference import FallDetectionPipeline
from services.video_analysis import VideoAnalysisJobs, add_video_routes
from services.realtime_camera import configure_capture, RollingFPS, LatestPreview, PreviewBrightness, prepare_detection_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InferenceService:
    def __init__(
        self,
        checkpoint: str = "checkpoints/best.pth",
        config_path: str = "configs/config.yaml",
        camera_index: int = 0,
        backend_url: str = "http://localhost:8080",
    ):
        self.checkpoint = str((PROJECT_ROOT / checkpoint).resolve()) if not os.path.isabs(checkpoint) else checkpoint
        self.config_path = str((PROJECT_ROOT / config_path).resolve()) if not os.path.isabs(config_path) else config_path
        self.camera_index = camera_index
        self.backend_url = backend_url.rstrip("/")
        self.pipeline: Optional[FallDetectionPipeline] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        # The camera producer and the inference worker exchange only the most
        # recent frame.  A bounded hand-off is important for a live camera:
        # when inference is slower than the webcam, processing every queued
        # frame makes the preview increasingly delayed and jerky.
        self._frame_condition = threading.Condition(self._lock)
        self._latest_frame = None
        self._frame_seq = 0
        self._jpeg_condition = threading.Condition(self._lock)
        self._jpeg_seq = 0
        self.video_jobs = VideoAnalysisJobs(self)
        self._latest_jpeg: Optional[bytes] = None
        self._latest_status: Dict[str, Any] = {
            "running": False,
            "camera_index": camera_index,
            "camera_id": "laptop_camera",
            "alarm": False,
            "frame_idx": 0,
            "last_error": None,
        }
        self._recent_events = deque(maxlen=100)
        self._event_queue: queue.Queue[dict] = queue.Queue()
        self._sent_event_ids: set[str] = set()
        self._sender_thread = threading.Thread(target=self._event_sender, daemon=True)
        self._sender_thread.start()

    def _read_config(self) -> dict:
        with open(self.config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _build_pipeline(self, realtime: bool = False) -> FallDetectionPipeline:
        config = self._read_config()
        model_config = config.get("model", {})
        pipeline_config = config.get("pipeline", {})
        detector_config = config.get("detector", {})
        rule_cfg = model_config.get("rule", {})
        rule_config = RuleConfig(
            torso_angle_threshold=rule_cfg.get("torso_angle_threshold", 0.5),
            torso_angle_duration=rule_cfg.get("torso_angle_duration", 8),
            velocity_peak_threshold=rule_cfg.get("velocity_peak_threshold", 0.03),
            stillness_threshold=rule_cfg.get("stillness_threshold", 0.005),
            stillness_duration=rule_cfg.get("stillness_duration", 15),
            vote_window=rule_cfg.get("vote_window", 10),
            vote_threshold=rule_cfg.get("vote_threshold", 0.6),
            recovery_duration=rule_cfg.get("recovery_duration", 5),
            tcn_prob_threshold=rule_cfg.get("tcn_prob_threshold", 0.3),
            aspect_ratio_change_threshold=rule_cfg.get("aspect_ratio_change_threshold", 0.2),
            fall_memory_frames=rule_cfg.get("fall_memory_frames", 60),
            rules_min_pass=rule_cfg.get("rules_min_pass", 1),
        )

        # Prefer the application setting, but use the checkpoint tensor
        # shapes as the source of truth when the two disagree.  Older
        # checkpoints in this project can contain a ``model.version`` value
        # that no longer matches the current config file.
        model_version = pipeline_config.get("model_version", "standard")
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"找不到模型权重：{self.checkpoint}")
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        state = {k: v for k, v in state.items() if k.split(".")[-1] not in {"total_ops", "total_params"}}
        model_version = self._infer_model_version(state, model_version)
        model = create_fall_detector(version=model_version)
        model.load_state_dict(state, strict=True)
        model.eval()

        configured_input_size = tuple(detector_config.get("input_size", [640, 384]))
        configured_device = detector_config.get("device", "cpu")
        if not realtime:
            configured_input_size = tuple(pipeline_config.get("video_input_size", configured_input_size))
            configured_device = pipeline_config.get("video_device", configured_device)
            torch.set_num_threads(max(1, int(pipeline_config.get("video_cpu_threads", 2))))
        if realtime:
            # The live preview favours a bounded per-frame cost.  The full
            # resolution detector and pose recovery remain enabled for file
            # analysis below.
            configured_input_size = tuple(
                pipeline_config.get("camera_input_size", configured_input_size)
            )
            configured_device = pipeline_config.get("camera_device", configured_device)
        if str(configured_device).lower() in {"cuda", "gpu", "cuda:0"} and not torch.cuda.is_available():
            print("[InferenceService] CUDA unavailable; falling back to CPU detector")
            configured_device = "cpu"
        try:
            detector = YOLOPoseDetector(
                model_path=(detector_config.get("model", "yolov8n-pose.pt") if realtime
                            else pipeline_config.get("video_model", detector_config.get("model", "yolov8n-pose.pt"))),
                device=configured_device,
                input_size=configured_input_size,
                conf_threshold=detector_config.get("conf_threshold", 0.15),
                iou_threshold=detector_config.get("iou_threshold", 0.7),
                kp_conf_threshold=detector_config.get("kp_conf_threshold", 0.3),
                max_det=detector_config.get("max_det", 5),
            )
            detector.load_model()
            if not realtime:
                # Ultralytics may change the CPU thread count while loading.
                # Reapply the bounded pool for the small CPU temporal model.
                torch.set_num_threads(max(1, int(pipeline_config.get("video_cpu_threads", 2))))
        except Exception as exc:
            raise RuntimeError(f"人体检测模型加载失败：{exc}") from exc

        pipeline = FallDetectionPipeline(
            detector=detector,
            fall_detector=model,
            sequence_length=pipeline_config.get("sequence_length", 32),
            detection_interval=(
                pipeline_config.get("camera_detection_interval", 2)
                if realtime else pipeline_config.get("video_detection_interval", 1)
            ),
            # The live worker enhances once and shares that frame with the
            # preview. File analysis still preprocesses inside process_frame.
            auto_lowlight=not realtime,
            save_video=False,
            camera_id=pipeline_config.get("camera_id", "laptop_camera"),
            alarm_trigger_seconds=pipeline_config.get("alarm_trigger_seconds", 0.5),
            alarm_hold_seconds=pipeline_config.get("alarm_hold_seconds", 30.0),
            alarm_cooldown_frames=pipeline_config.get("alarm_cooldown_frames", 30),
            rule_config=rule_config,
            pose_recovery=(
                pipeline_config.get("camera_pose_recovery", False)
                if realtime else pipeline_config.get("pose_recovery", False)
            ),
            high_res_rescue=(not realtime),
            cpu_threads=max(1, int(pipeline_config.get("camera_cpu_threads" if realtime else "video_cpu_threads", 2))),
            rescue_input_size=tuple(pipeline_config.get("rescue_input_size", [1280, 736])),
            require_observed_transition=pipeline_config.get("require_observed_transition", False),
            # Offline uploads favour recall: fallen people often expose fewer
            # keypoints than an upright person.  The live camera keeps the
            # stricter gate so a noisy partial pose cannot cause flicker.
            min_kp_completeness=(
                pipeline_config.get("camera_min_kp_completeness", 0.5)
                if realtime else pipeline_config.get("video_min_kp_completeness", 0.35)
            ),
            transition_max_gap_seconds=(
                pipeline_config.get("camera_transition_max_gap_seconds", 1.5)
                if realtime else pipeline_config.get("transition_max_gap_seconds", 0.6)
            ),
            preserve_short_gaps_seconds=(
                pipeline_config.get("camera_preserve_short_gaps_seconds", 1.2)
                if realtime else 0.0
            ),
            model_support_grace_seconds=(
                pipeline_config.get("camera_model_support_grace_seconds", 1.2)
                if realtime else 0.0
            ),
            max_candidate_gap_seconds=(0.35 if realtime else 0.0),
        )
        return pipeline

    @staticmethod
    def _infer_model_version(state: dict, configured_version: str) -> str:
        """Infer a compatible factory preset from checkpoint tensor shapes."""
        input_proj = state.get("tcn.input_proj.weight")
        first_layer = state.get("tcn.tcn_layers.0.causal_conv.conv.weight")
        layer_keys = [key for key in state if key.startswith("tcn.tcn_layers.") and key.endswith(".bn.weight")]
        if input_proj is None or first_layer is None:
            return configured_version

        input_dim = int(input_proj.shape[1])
        hidden_dim = int(input_proj.shape[0])
        kernel_size = int(first_layer.shape[-1])
        num_layers = len(layer_keys)
        presets = {
            (48, 64, 3, 5): "standard",
            (48, 32, 3, 4): "light",
            (64, 128, 5, 6): "large",
        }
        inferred = presets.get((input_dim, hidden_dim, kernel_size, num_layers))
        if inferred and inferred != configured_version:
            print(
                f"[InferenceService] checkpoint architecture is {inferred}; "
                f"overriding configured model_version={configured_version}"
            )
        return inferred or configured_version

    def start(self) -> dict:
        with self._lock:
            if self.video_jobs.busy:
                raise RuntimeError("视频正在分析，请完成后再启动设备监测")
            if self._thread and self._thread.is_alive():
                return self.status()
            # Camera mode uses the bounded real-time profile.  Offline video
            # jobs call _build_pipeline() without this flag and keep all pose
            # recovery passes for maximum detection quality.
            self.pipeline = self._build_pipeline(realtime=True)
            camera_config = self._read_config().get('pipeline', {})
            threads = max(1, int(camera_config.get('camera_cpu_threads', 2)))
            cv2.setNumThreads(threads)
            torch.set_num_threads(threads)
            self._stop.clear()
            self._latest_jpeg = None
            self._latest_frame = None
            self._frame_seq = 0
            self._jpeg_seq = 0
            self._latest_status.update({
                "running": False,
                "last_error": None,
                "alarm": False,
                "frame_idx": 0,
                "camera_index": self.camera_index,
                "camera_id": self.pipeline.camera_id,
            })
            self._thread = threading.Thread(target=self._camera_loop, daemon=True)
            self._thread.start()

        # Wait briefly so the HTTP caller receives the actual camera result,
        # instead of an optimistic response before OpenCV opens the device.
        # The first CUDA inference can take several seconds while the driver
        # initialises.  Do not report a false startup failure to the browser
        # during that one-time warm-up.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            current = self.status()
            if current.get("running") or current.get("last_error"):
                return current
            time.sleep(0.05)
        return self.status()

    def stop(self) -> dict:
        self._stop.set()
        with self._frame_condition:
            self._frame_condition.notify_all()
            self._jpeg_condition.notify_all()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._latest_status["running"] = False
        return self.status()

    def status(self) -> dict:
        with self._lock:
            status = dict(self._latest_status)
            status["recent_event_count"] = len(self._recent_events)
            return status

    def _camera_loop(self) -> None:
        camera_config = self._read_config().get('pipeline', {})
        # PyTorch's CPU parallelism setting must also be applied in the actual
        # worker thread. This small TCN is slower with a 24-thread pool.
        torch.set_num_threads(max(1, int(camera_config.get('camera_cpu_threads', 2))))
        # DirectShow honours MJPEG/exposure on this Windows UVC device.
        backends = ((cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
                    if camera_config.get('camera_backend', 'dshow') == 'dshow'
                    else (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY))
        cap = None
        for backend in backends:
            candidate = cv2.VideoCapture(self.camera_index, backend)
            if candidate.isOpened():
                cap = candidate
                break
            candidate.release()
        if cap is None:
            with self._lock:
                self._latest_status.update({"running": False, "last_error": f"cannot open camera {self.camera_index}"})
            return

        negotiated = configure_capture(cap, camera_config)
        negotiated['camera_backend'] = cap.getBackendName()
        capture_done = threading.Event()
        previews = LatestPreview()
        capture_rate, inference_rate, output_rate = RollingFPS(), RollingFPS(), RollingFPS()
        brightness = PreviewBrightness(camera_config.get('camera_preview_brightness', 100))
        jpeg_quality = max(50, min(100, int(camera_config.get('camera_jpeg_quality', 90))))

        def capture_latest() -> None:
            """Continuously capture, retaining only the newest frame."""
            try:
                while not self._stop.is_set():
                    ok, captured = cap.read()
                    if not ok:
                        with self._frame_condition:
                            self._latest_status["last_error"] = "camera frame read failed"
                            self._frame_condition.notify_all()
                        break
                    with self._frame_condition:
                        self._latest_frame = captured
                        self._latest_captured_at = datetime.now(timezone.utc)
                        self._latest_capture_clock = time.perf_counter()
                        self._frame_seq += 1
                        self._latest_status["captured_frames"] = self._frame_seq
                        self._latest_status['captured_fps'] = capture_rate.tick()
                        self._frame_condition.notify_all()
            finally:
                capture_done.set()
                with self._frame_condition:
                    self._frame_condition.notify_all()

        def encode_previews() -> None:
            """Encode each inference result on its own source image.

            This overlaps CPU rendering/JPEG with CUDA inference on the next
            frame. Never draw an older pose onto a newer camera frame.
            """
            try:
                while not self._stop.is_set():
                    item = previews.get()
                    if item is None:
                        break
                    frame, result, captured_clock = item
                    started = time.perf_counter()
                    shown = brightness.apply(frame)
                    annotated = self.pipeline._draw_result(shown, result)
                    ok, encoded = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    if not ok:
                        raise RuntimeError('camera JPEG encoding failed')
                    with self._jpeg_condition:
                        self._latest_jpeg = encoded.tobytes()
                        self._jpeg_seq += 1
                        self._latest_status.update({
                            'output_fps': output_rate.tick(),
                            'output_frames': self._jpeg_seq,
                            'output_source_frame': result.frame_idx,
                            'render_ms': round((time.perf_counter() - started) * 1000, 2),
                            'frame_age_ms': round((time.perf_counter() - captured_clock) * 1000, 2),
                            'preview_dropped_frames': previews.dropped,
                            **brightness.metrics,
                        })
                        self._jpeg_condition.notify_all()
            except Exception as exc:
                with self._frame_condition:
                    self._latest_status['last_error'] = str(exc)
                    self._stop.set()
                    self._frame_condition.notify_all()

        frame_idx = 0
        if self.pipeline is not None:
            self.pipeline.alarm_state.use_wall_clock = True
        with self._lock:
            self._latest_status["running"] = True
            self._latest_status.update({
                "output_fps": 0.0,
                "captured_fps": 0.0,
                'inference_fps': 0.0,
                'inference_frames': 0,
                'output_frames': 0,
                "dropped_frames": 0,
                'detection_interval': self.pipeline.detection_interval,
                'same_frame_annotations': self.pipeline.detection_interval == 1,
                **negotiated,
            })
        capture_thread = threading.Thread(target=capture_latest, daemon=True)
        preview_thread = threading.Thread(target=encode_previews, daemon=True)
        capture_thread.start()
        preview_thread.start()
        last_sequence = 0
        last_captured_count = 0
        try:
            while not self._stop.is_set():
                with self._frame_condition:
                    self._frame_condition.wait_for(
                        lambda: self._stop.is_set()
                        or self._frame_seq > last_sequence
                        or capture_done.is_set(),
                        timeout=1.0,
                    )
                    if self._stop.is_set():
                        break
                    if self._frame_seq <= last_sequence:
                        if capture_done.is_set():
                            break
                        continue
                    # Copy the current frame while holding the lock.  The
                    # producer immediately replaces the reference afterwards,
                    # so the inference worker never processes a mutable frame.
                    last_sequence = self._frame_seq
                    frame = self._latest_frame.copy()
                    captured_count = self._frame_seq
                    captured_at = self._latest_captured_at
                    captured_clock = self._latest_capture_clock
                dropped = max(0, captured_count - last_captured_count - 1)
                last_captured_count = captured_count
                started = time.perf_counter()
                with self._lock:
                    pipeline = self.pipeline
                if pipeline is None:
                    break
                # Enhance at detector resolution, keeping the full 720p image
                # for display. Normalized keypoints map back to that SAME
                # image. This preserves low-light model preprocessing without
                # paying for full-resolution CLAHE on the inference thread.
                detection_frame = prepare_detection_frame(frame, pipeline.detector)
                enhanced_at = time.perf_counter()
                result = pipeline.process_frame(detection_frame, frame_idx, timestamp=captured_at)
                with self._lock:
                    self._latest_status.update({
                        'running': True,
                        'alarm': any(result.alarms.values()),
                        'frame_idx': captured_count - 1,
                        'inference_frames': frame_idx + 1,
                        'inference_fps': inference_rate.tick(),
                        'latency_ms': round((time.perf_counter() - started) * 1000, 2),
                        'dropped_frames': int(self._latest_status.get('dropped_frames', 0) + dropped),
                        'detected_persons': len(result.raw_detections),
                        'enhance_ms': round((enhanced_at - started) * 1000, 2),
                        'pipeline_ms': round((time.perf_counter() - enhanced_at) * 1000, 2),
                        'cpu_threads': torch.get_num_threads(),
                    })
                previews.put((frame, replace(result, frame_idx=captured_count - 1), captured_clock))
                for update in result.event_updates:
                    self._recent_events.appendleft(update)
                    self._event_queue.put(update)
                frame_idx += 1
        except Exception as exc:  # keep service status inspectable after an error
            with self._lock:
                self._latest_status.update({"running": False, "last_error": str(exc)})
        finally:
            self._stop.set()
            previews.close()
            with self._frame_condition:
                self._frame_condition.notify_all()
            capture_thread.join(timeout=1.0)
            preview_thread.join(timeout=2.0)
            cap.release()
            with self._lock:
                self._latest_status["running"] = False
                self._jpeg_condition.notify_all()

    def _event_sender(self) -> None:
        while True:
            event = self._event_queue.get()
            try:
                response = requests.post(
                    f"{self.backend_url}/api/internal/events",
                    json=event,
                    timeout=1.0,
                )
                if response.status_code >= 300:
                    print(f"[InferenceService] backend rejected event: {response.status_code}")
                else:
                    self._sent_event_ids.add(event.get("event_id", ""))
            except requests.RequestException as exc:
                # The event remains visible in this process and can be
                # inspected from /api/events until Spring Boot is available.
                print(f"[InferenceService] backend unavailable: {exc}")
            finally:
                self._event_queue.task_done()

    def clear_event(self, event_id: str, reason: str = "manual") -> Optional[dict]:
        with self._lock:
            if self.pipeline is None:
                return None
            event = self.pipeline.clear_event(event_id, reason=reason)
        if event:
            self._recent_events.appendleft(event)
            self._event_queue.put(event)
        return event

    def events(self) -> list[dict]:
        return list(self._recent_events)

    def mjpeg(self) -> Generator[bytes, None, None]:
        last_sequence = 0
        while True:
            with self._jpeg_condition:
                self._jpeg_condition.wait_for(
                    lambda: self._jpeg_seq > last_sequence or self._stop.is_set(),
                    timeout=1.0,
                )
                if self._jpeg_seq <= last_sequence and self._stop.is_set():
                    return
                if self._jpeg_seq <= last_sequence:
                    continue
                last_sequence = self._jpeg_seq
                jpeg = self._latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"


def build_app(service: InferenceService) -> FastAPI:
    app = FastAPI(title="Fall Detection Inference Service", version="1.0.0")
    add_video_routes(app, service.video_jobs)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "UP", "service": "python-inference", "camera": service.status(),
                "analysis_version": "2026-10-16-observed-v4"}

    @app.get("/api/status")
    def status() -> dict:
        return service.status()

    @app.post("/api/camera/start")
    def start_camera() -> dict:
        try:
            return service.start()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/api/camera/stop")
    def stop_camera() -> dict:
        return service.stop()

    @app.get("/api/events")
    def recent_events() -> list[dict]:
        return service.events()

    @app.post("/api/internal/events/{event_id}/clear")
    def clear_event(event_id: str, reason: str = "manual") -> JSONResponse:
        event = service.clear_event(event_id, reason=reason)
        if event is None:
            raise HTTPException(status_code=404, detail="event not found")
        return JSONResponse(event)

    @app.get("/api/stream.mjpeg")
    def stream() -> StreamingResponse:
        return StreamingResponse(service.mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="笔记本摄像头跌倒检测服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--backend-url", default="http://localhost:8080")
    parser.add_argument("--auto-start", action="store_true", help="服务启动后立即打开摄像头")
    args = parser.parse_args()

    service = InferenceService(
        checkpoint=args.checkpoint,
        config_path=args.config,
        camera_index=args.camera_index,
        backend_url=args.backend_url,
    )
    if args.auto_start:
        service.start()
    app = build_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

