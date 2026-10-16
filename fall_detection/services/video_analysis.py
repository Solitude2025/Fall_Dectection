"""Bounded, asynchronous video analysis. Uploaded videos never own the camera."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import subprocess
import shutil
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from services.video_diagnostics import VideoDiagnostics


MAX_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 300
MAX_PIXELS = 1920 * 1080
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoAnalysisJobs:
    def __init__(self, service, storage: Path | None = None):
        self.service = service
        self.storage = storage or Path(__file__).resolve().parents[1] / "app_data" / "video_jobs"
        self.storage.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._busy = False
        self._jobs: dict[str, dict] = {}
        for report in self.storage.glob("*/report.json"):
            try:
                job = json.loads(report.read_text(encoding="utf-8"))
                if job["status"] not in {"completed", "failed"}:
                    job.update(status="failed", error="服务重启中断了分析，请重新上传")
                self._jobs[job["job_id"]] = job
            except (ValueError, KeyError, OSError):
                continue

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def reserve(self, filename: str) -> dict:
        name = filename.replace("\\", "/").rsplit("/", 1)[-1]
        if not name or Path(name).suffix.lower() not in EXTENSIONS:
            raise HTTPException(400, "请选择 MP4、AVI、MOV、MKV 或 WebM 视频")
        # Camera start and video reservation share this lock, so they cannot
        # both begin while the other's model is still loading.
        with self.service._lock, self._lock:
            if self._busy or (self.service._thread and self.service._thread.is_alive()):
                raise HTTPException(409, "请先停止设备监测，并等待当前视频分析完成")
            self._busy = True
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id, "filename": name[:200], "status": "uploading",
                "progress": 0, "frames_processed": 0, "event_count": 0,
                "events": [], "created_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
            self._jobs[job_id] = job
            (self.storage / job_id).mkdir()
            self._save(job)
            return {k: v for k, v in job.items() if k != "timeline"}

    def _save(self, job: dict) -> None:
        (self.storage / job["job_id"] / "report.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "视频分析任务不存在")
            # Per-frame traces are persisted locally for diagnosis. Polling
            # only needs the small summary, not thousands of frame records.
            return {k: v for k, v in job.items() if k != "timeline"}

    def list(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)
            return [{k: v for k, v in job.items() if k not in {"events", "timeline"}} for job in jobs[:20]]

    def delete(self, job_id: str) -> dict:
        """Delete one finished history record and its generated replay."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "视频分析任务不存在")
            if job.get("status") not in {"completed", "failed"}:
                raise HTTPException(409, "分析尚未结束，暂时不能删除此记录")
            directory = self.storage / job_id
            self._jobs.pop(job_id, None)
        # Do filesystem work outside the lock; the in-memory record has
        # already disappeared from the list shown to the browser.
        shutil.rmtree(directory, ignore_errors=True)
        return {"deleted": 1, "job_id": job_id}

    def delete_all(self) -> dict:
        """Delete all finished records without interrupting a running job."""
        with self._lock:
            removable = [
                (job_id, job) for job_id, job in self._jobs.items()
                if job.get("status") in {"completed", "failed"}
            ]
            for job_id, _ in removable:
                self._jobs.pop(job_id, None)
            active_count = len(self._jobs)
        for job_id, _ in removable:
            shutil.rmtree(self.storage / job_id, ignore_errors=True)
        return {"deleted": len(removable), "skipped_active": active_count}

    def _update(self, job_id: str, **values) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def fail(self, job_id: str, message: str, release: bool = True) -> None:
        with self._lock:
            self._jobs[job_id].update(status="failed", error=message)
            self._save(self._jobs[job_id])
            if release:
                self._busy = False

    def enqueue(self, job_id: str) -> None:
        self._update(job_id, status="queued")
        threading.Thread(target=self._run, args=(job_id,), daemon=True).start()

    def _run(self, job_id: str) -> None:
        directory = self.storage / job_id
        cap = None
        writer = None
        try:
            cap = cv2.VideoCapture(str(directory / "input.video"))
            if not cap.isOpened():
                raise ValueError("视频无法解码，请转换为常见的 MP4 视频后重试")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or not 1 <= fps <= 240 or count <= 0:
                raise ValueError("视频帧率或时长无效，无法进行可靠的时序分析")
            if width <= 0 or height <= 0 or width * height > MAX_PIXELS or max(width, height) > 1920:
                raise ValueError("视频分辨率不能超过 1080p，请压缩后重试")
            if count / fps > MAX_SECONDS:
                raise ValueError("视频时长不能超过 5 分钟")
            self._update(job_id, status="processing", fps=round(fps, 3),
                         duration_seconds=round(count / fps, 3), total_frames=count)
            pipeline = self.service._build_pipeline()
            analysis_started = time.perf_counter()
            pipeline.camera_id = f"video_{job_id}"
            pipeline.alarm_state.camera_id = pipeline.camera_id
            pipeline.fps = fps
            pipeline.alarm_state.set_fps(fps)
            diagnostics = VideoDiagnostics(fps, pipeline.min_kp_completeness)
            scale = min(1.0, 960 / max(width, height))
            size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
            writer = cv2.VideoWriter(str(directory / "annotated.avi"),
                                     cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
            if not writer.isOpened():
                raise RuntimeError("无法创建分析回放文件")
            origin = datetime.now(timezone.utc)
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index >= int(fps * MAX_SECONDS) + 1:
                    raise ValueError("实际视频时长超过 5 分钟")
                # Use media time, not processing speed, for file analysis.
                timestamp = origin + timedelta(seconds=index / fps)
                # Preserve full-frame normalized features used to train TCN.
                # PoseRecovery performs local inference and restores coordinates
                # before the tracker/model; never crop only the model's canvas.
                result = pipeline.process_frame(frame, index, timestamp=timestamp)
                diagnostics.update(result)
                rendered = pipeline._draw_result(frame, result)
                writer.write(cv2.resize(rendered, size))
                index += 1
                if index % 10 == 0:
                    self._update(job_id, frames_processed=index, progress=min(95, int(index / count * 95)))
            if index == 0 or index < count - max(2, int(fps)):
                raise ValueError("视频未完整解码，请检查文件是否损坏")
            writer.release()
            writer = None
            cap.release()
            cap = None
            self._update(job_id, status="encoding", progress=96, frames_processed=index)
            import imageio_ffmpeg
            encoded = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(directory / "annotated.avi"),
                 "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(directory / "result.mp4")],
                capture_output=True, timeout=180,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if encoded.returncode:
                raise RuntimeError("分析完成，但回放编码失败：" + encoded.stderr.decode(errors="replace")[-300:])
            # Reports keep the existing event contract. They do not emit live
            # alarm events or get mixed into the device's MySQL statistics.
            events = [event.to_dict() | {"video_time_seconds": round(event.timestamp, 3),
                       "trigger_time_seconds": round(event.trigger_frame / fps, 3)}
                      for event in pipeline.alarm_state.events]
            self._update(job_id, status="completed", progress=100, events=events,
                         event_count=len(events), has_fall=bool(events),
                         analysis_version="2026-10-16-observed-v4",
                         decision_policy="observed_transition" if pipeline.require_observed_transition else "legacy_rules",
                         processing_seconds=round(time.perf_counter() - analysis_started, 2),
                         **diagnostics.report(),
                         result_url=f"/api/videos/{job_id}/result")
            with self._lock:
                self._save(self._jobs[job_id])
        except Exception as exc:
            self.fail(job_id, str(exc), release=False)
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            # Only remove intermediate files owned by this task.
            (directory / "input.video").unlink(missing_ok=True)
            (directory / "annotated.avi").unlink(missing_ok=True)
            with self._lock:
                self._busy = False


def add_video_routes(app: FastAPI, jobs: VideoAnalysisJobs) -> None:
    @app.post("/api/videos", status_code=202)
    async def upload_video(request: Request, filename: str):
        job = jobs.reserve(filename)
        job_id = job["job_id"]
        source = jobs.storage / job_id / "input.video"
        try:
            size = 0
            with source.open("wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise HTTPException(413, "视频不能超过 100 MB")
                    output.write(chunk)
            if size == 0:
                raise HTTPException(400, "视频文件为空")
            jobs.enqueue(job_id)
            return jobs.get(job_id)
        except BaseException as exc:
            jobs.fail(job_id, str(getattr(exc, "detail", exc)))
            source.unlink(missing_ok=True)
            raise

    @app.get("/api/videos")
    def list_videos():
        return jobs.list()

    @app.delete("/api/videos")
    def delete_all_videos():
        return jobs.delete_all()

    @app.delete("/api/videos/{job_id}")
    def delete_video(job_id: str):
        return jobs.delete(job_id)

    @app.get("/api/videos/{job_id}")
    def video_status(job_id: str):
        return jobs.get(job_id)

    @app.get("/api/videos/{job_id}/result")
    def video_result(job_id: str):
        if jobs.get(job_id)["status"] != "completed":
            raise HTTPException(409, "分析结果尚未生成")
        return FileResponse(jobs.storage / job_id / "result.mp4", media_type="video/mp4")
