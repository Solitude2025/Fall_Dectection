from types import SimpleNamespace

import numpy as np
import torch

from pipeline.detector import PersonDetection
from pipeline.detector import YOLOPoseDetector
from pipeline.inference import FallDetectionPipeline
from pipeline.tracker import ByteTrackWrapper
from services.video_diagnostics import VideoDiagnostics


def person():
    return PersonDetection(np.array([.2, .1, .6, .9]), np.full((17, 3), .9), .9)


def test_missed_observation_is_not_reused_but_id_can_be_reassociated():
    tracker = ByteTrackWrapper()
    assert list(tracker.update([person()], 2, True)) == [0]
    assert list(tracker.update([], 2, False)) == [0]
    assert tracker.update([], 2, True) == {}
    assert tracker.update([], 2, False) == {}
    assert list(tracker.update([person()], 2, True)) == [0]


def test_empty_detection_is_not_confused_with_skipped_frame():
    tracker = ByteTrackWrapper()
    assert list(tracker.update([person()], 2)) == [0]
    assert list(tracker.update([], 2)) == [0]
    assert tracker.update([], 2) == {}


def test_pipeline_discards_sequence_after_detector_gap():
    detector = SimpleNamespace(detect=lambda *args: [person()])
    model = SimpleNamespace(eval=lambda: None,
                            motion_encoder=SimpleNamespace(compute_sequence=lambda kp, bb: np.zeros((len(kp), 64), dtype=np.float32)),
                            tcn=lambda features: torch.full((1, features.shape[1], 1), .8))
    pipeline = FallDetectionPipeline(detector=detector, fall_detector=model, auto_lowlight=False)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for index in range(16):
        pipeline.process_frame(frame, index)
    assert pipeline._kp_buffers[0]
    detector.detect = lambda *args: []
    missing = pipeline.process_frame(frame, 16)
    assert not missing.persons
    assert not missing.raw_fall_probs
    assert 0 not in pipeline._kp_buffers
    assert 0 not in pipeline.rule_engines
    detector.detect = lambda *args: [person()]
    recovered = pipeline.process_frame(frame, 18)
    assert len(recovered.persons) == 1
    assert len(pipeline._kp_buffers[0]) == 1
    assert recovered.model_alarms[0] is False


def test_missing_pose_warning_does_not_create_an_alarm():
    diagnostics = VideoDiagnostics(10, .5)
    for index in range(10):
        result = SimpleNamespace(frame_idx=index, persons=[person()] if index < 4 else [],
                                 raw_fall_probs={}, model_alarms={}, alarms={})
        diagnostics.update(result)
    report = diagnostics.report()
    assert report['quality']['longest_pose_gap_seconds'] == .6
    assert report['quality']['observed_frame_ratio'] == .4
    assert report['warnings']
    assert not any(r['alarm'] for r in report['timeline'])


def test_detector_passes_height_then_width_and_surfaces_inference_errors():
    import pytest
    received = {}
    detector = YOLOPoseDetector(input_size=(640, 384))
    def model(image, **options):
        received.update(options)
        return []
    detector._model = model
    assert detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8)) == []
    assert received['imgsz'] == (384, 640)
    def broken(*args, **kwargs):
        raise ValueError('invalid inference')
    detector._model = broken
    with pytest.raises(RuntimeError, match='invalid inference'):
        detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_pose_association_keeps_id_when_box_shrinks_during_fall():
    tracker = ByteTrackWrapper(pose_association=True)
    first = person()
    tracker.update([first], 1, True)
    changed = person()
    changed.bbox = np.array([.2, .5, .7, .7])
    changed.keypoints[:, :2] += .01
    assert list(tracker.update([changed], 1, True)) == [0]
    unrelated = person()
    unrelated.bbox = np.array([.01, .01, .1, .1])
    unrelated.keypoints[:, :2] = .05
    assert list(tracker.update([unrelated], 1, True)) == [1]


def test_box_with_incomplete_core_is_not_reported_as_reliable_core_pose():
    from pipeline.inference import FrameResult
    detection = person()
    detection.keypoints[[5, 6, 11, 12], 2] = .1
    diagnostics = VideoDiagnostics(30, .35)
    diagnostics.update(FrameResult(0, [detection], {}, {}))
    report = diagnostics.report()
    assert report['quality']['observed_frame_ratio'] == 1
    assert report['quality']['core_pose_frame_ratio'] == 0
    assert report['timeline'][0]['core_pose_observed'] is False


def test_overlay_labels_only_the_person_with_an_alarm(monkeypatch):
    import cv2
    from pipeline.inference import FrameResult
    labels = []
    original = cv2.putText
    def capture(image, text, *args, **kwargs):
        labels.append(text)
        return original(image, text, *args, **kwargs)
    monkeypatch.setattr(cv2, 'putText', capture)
    detection = person()
    pipeline = object.__new__(FallDetectionPipeline)
    result = FrameResult(0, [detection], {1: .8, 2: .1}, {1: True, 2: False},
                         visual_persons={1: detection, 2: detection})
    pipeline._draw_result(np.zeros((200, 300, 3), dtype=np.uint8), result)
    assert 'ALARM 0.80' in labels
    assert 'Monitoring 0.10' in labels
    assert 'ALARM 0.10' not in labels
