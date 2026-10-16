import json
import threading
import time
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.video_analysis import VideoAnalysisJobs, add_video_routes


def client_for(tmp_path):
    service = SimpleNamespace(_lock=threading.RLock(), _thread=None)
    jobs = VideoAnalysisJobs(service, tmp_path)
    app = FastAPI()
    add_video_routes(app, jobs)
    return TestClient(app), jobs, service


def test_upload_limits_and_camera_exclusion(tmp_path, monkeypatch):
    client, jobs, service = client_for(tmp_path)
    assert client.post('/api/videos?filename=x.txt', content=b'video').status_code == 400
    assert not jobs.busy
    monkeypatch.setattr('services.video_analysis.MAX_BYTES', 4)
    assert client.post('/api/videos?filename=x.mp4', content=b'12345').status_code == 413
    assert not jobs.busy
    assert not list(tmp_path.glob('*/input.video'))
    service._thread = SimpleNamespace(is_alive=lambda: True)
    assert client.post('/api/videos?filename=x.mp4', content=b'1234').status_code == 409


def test_invalid_video_reports_failure_and_never_returns_result(tmp_path):
    client, jobs, _ = client_for(tmp_path)
    response = client.post('/api/videos?filename=bad.mp4', content=b'not-a-video')
    assert response.status_code == 202
    job_id = response.json()['job_id']
    deadline = time.monotonic() + 5
    while jobs.busy and time.monotonic() < deadline:
        time.sleep(.05)
    report = client.get(f'/api/videos/{job_id}').json()
    assert report['status'] == 'failed'
    assert report['error']
    assert client.get(f'/api/videos/{job_id}/result').status_code == 409
    assert not (tmp_path / job_id / 'input.video').exists()
    assert json.loads((tmp_path / job_id / 'report.json').read_text(encoding='utf-8'))['status'] == 'failed'


def test_playback_supports_byte_ranges_and_reports_survive_restart(tmp_path):
    client, jobs, service = client_for(tmp_path)
    job = jobs.reserve('demo.mp4')
    job_id = job['job_id']
    jobs._update(job_id, status='completed', progress=100, timeline=[{'time_seconds': 0}])
    jobs._save(jobs._jobs[job_id])
    (tmp_path / job_id / 'result.mp4').write_bytes(b'0123456789')
    response = client.get(f'/api/videos/{job_id}/result', headers={'Range': 'bytes=2-5'})
    assert response.status_code == 206
    assert response.headers['content-range'] == 'bytes 2-5/10'
    assert response.content == b'2345'
    restarted = VideoAnalysisJobs(service, tmp_path)
    assert restarted.get(job_id)['status'] == 'completed'
    assert restarted.list()[0]['filename'] == 'demo.mp4'
    assert 'timeline' not in restarted.get(job_id)
    assert 'timeline' not in restarted.list()[0]
    assert json.loads((tmp_path / job_id / 'report.json').read_text('utf8'))['timeline']


def test_worker_keeps_alarm_after_later_detector_miss(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from pipeline.inference import FrameResult

    _, jobs, service = client_for(tmp_path)
    job = jobs.reserve('fall_then_missing.mp4')
    directory = tmp_path / job['job_id']
    # Keep a real decoder and writer in the test; the model and final encoder
    # are replaced so no weights or external services are needed.
    video = directory / 'input.video'
    source = directory / 'source.avi'
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
    for _ in range(10):
        writer.write(np.full((64, 64, 3), 100, dtype=np.uint8))
    writer.release()
    source.rename(video)
    frames = []
    event = SimpleNamespace(timestamp=.1, trigger_frame=5, to_dict=lambda: {'confidence': .8})

    def process(frame, index, timestamp=None):
        frames.append(index)
        return FrameResult(index, [], {}, {0: index >= 5})

    pipeline = SimpleNamespace(
        camera_id='', fps=10, min_kp_completeness=.5, require_observed_transition=True,
        alarm_state=SimpleNamespace(camera_id='', set_fps=lambda fps: None, events=[event]),
        process_frame=process, _draw_result=lambda frame, result: frame,
    )
    service._build_pipeline = lambda: pipeline
    monkeypatch.setattr('services.video_analysis.subprocess.run', lambda *a, **kw: SimpleNamespace(returncode=0))
    jobs._run(job['job_id'])
    report = jobs.get(job['job_id'])
    assert report['status'] == 'completed', report.get('error')
    assert report['event_count'] == 1
    assert frames == list(range(10))  # no second process/reset on the miss

