import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipeline.inference import FallDetectionPipeline
from services.realtime_camera import configure_capture, LatestPreview, PreviewBrightness, RollingFPS


def test_camera_format_is_set_after_size_and_frame_rate():
    # Reproduce this UVC driver's mode reset when width/height are changed.
    props = {}
    def set_property(prop, value):
        props[prop] = value
        if prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            props[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*'YUY2')
        return True
    cap = SimpleNamespace(set=set_property, get=lambda key: props.get(key, 0))
    mode = configure_capture(cap, {})
    assert mode['camera_format'] == 'MJPG'
    assert (mode['camera_width'], mode['camera_height']) == (1280, 720)
    assert props[cv2.CAP_PROP_FPS] == 30


def test_preview_discards_frame_and_its_pose_together():
    mailbox = LatestPreview()
    mailbox.put(('old frame', 'old pose'))
    mailbox.put(('new frame', 'new pose'))
    assert mailbox.get() == ('new frame', 'new pose')
    assert mailbox.dropped == 1
    receiver = threading.Thread(target=mailbox.get)
    receiver.start()
    mailbox.close()
    receiver.join(timeout=1)
    assert not receiver.is_alive()
    assert mailbox.get() is None


def test_rolling_rate_ignores_startup_time_and_does_not_invent_frames():
    rate = RollingFPS()
    assert rate.tick(100) == 0
    for i in range(1, 91):
        result = rate.tick(100 + i / 30)
    assert result == pytest.approx(30, abs=.01)


def test_auto_exposure_is_restored_after_format_without_setting_manual_value():
    props, calls = {}, []
    def set_property(prop, value):
        calls.append((prop, value))
        props[prop] = value
        return True
    cap = SimpleNamespace(set=set_property, get=lambda key: props.get(key, -1))
    mode = configure_capture(cap, {'camera_exposure': -5})
    assert calls[-1] == (cv2.CAP_PROP_AUTO_EXPOSURE, .75)
    assert not any(prop == cv2.CAP_PROP_EXPOSURE for prop, _ in calls)
    assert mode['camera_exposure_mode_requested'] == 'auto'
    assert mode['camera_auto_exposure_accepted']


def test_manual_exposure_remains_an_explicit_override():
    props = {}
    cap = SimpleNamespace(set=lambda key, value: props.update({key: value}) or True,
                          get=lambda key: props.get(key, -1))
    configure_capture(cap, {'camera_auto_exposure': .25, 'camera_exposure': -4})
    assert props[cv2.CAP_PROP_EXPOSURE] == -4


def test_preview_brightness_preserves_black_and_does_not_change_model_input():
    dark = np.full((100, 100, 3), 20, dtype=np.uint8)
    corrected = PreviewBrightness().apply(dark)
    # Do not turn underexposed dark pixels into an opaque grey fog.
    assert 20 < corrected.mean() < 40
    assert dark.mean() == 20
    bright = np.full((100, 100, 3), 150, dtype=np.uint8)
    assert np.array_equal(PreviewBrightness().apply(bright), bright)
    levels = np.arange(256, dtype=np.uint8)[None, :, None].repeat(3, axis=2)
    curve = PreviewBrightness(target=150).apply(levels)[0, :, 0]
    assert curve[0] == 0
    assert curve[12] <= 18
    assert np.all(np.diff(curve.astype(int)) >= 0)


def test_preview_stops_lifting_as_soon_as_auto_exposure_recovers():
    preview = PreviewBrightness()
    preview.apply(np.full((100, 100, 3), 20, dtype=np.uint8))
    bright = np.full((100, 100, 3), 150, dtype=np.uint8)
    assert np.array_equal(preview.apply(bright), bright)
    assert preview.metrics['preview_gamma'] == 1
    black = np.zeros_like(bright)
    assert not preview.apply(black).any()


def test_per_frame_mode_does_not_draw_previous_pose_after_a_miss():
    from pipeline.detector import PersonDetection
    person = PersonDetection(np.array([.2, .1, .6, .9]), np.full((17, 3), .9), .9)
    detector = SimpleNamespace(detect=lambda *args: [person])
    pipeline = FallDetectionPipeline(detector=detector, fall_detector=SimpleNamespace(eval=lambda: None),
                                     auto_lowlight=False, detection_interval=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert pipeline.process_frame(frame, 0).visual_persons
    detector.detect = lambda *args: []
    missing = pipeline.process_frame(frame, 1)
    assert not missing.visual_persons
    assert not missing.persons
    assert not missing.raw_detections
