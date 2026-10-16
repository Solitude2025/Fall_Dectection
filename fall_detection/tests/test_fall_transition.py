import numpy as np

from pipeline.detector import PersonDetection
from pipeline.fall_transition import FallTransition
from pipeline.pose_recovery import PoseRecovery
from pipeline.alarm_state import AlarmStateManager


def pose(shoulder_y=.2, height=.6, horizontal=False):
    keypoints = np.zeros((17, 3), dtype=float)
    keypoints[:, 0] = .5
    keypoints[:, 1] = np.linspace(shoulder_y - .1 * height, shoulder_y + .9 * height, 17)
    keypoints[:, 2] = .9
    keypoints[[5, 6], 1] = shoulder_y
    keypoints[[11, 12], 1] = shoulder_y + height * .35
    if horizontal:
        keypoints[[11, 12], 0] += .2
        keypoints[[11, 12], 1] = shoulder_y
    return PersonDetection(np.array([.2, .1, .8, .9]), keypoints, .9)


def test_frontal_descent_remains_candidate_without_torso_tilt():
    engine = FallTransition()
    for index in range(10):
        assert not engine.update(pose(), (1000, 1000), index / 10).candidate
    for index in range(10, 18):
        decision = engine.update(pose(.55, .25), (1000, 1000), index / 10)
        assert decision.candidate
    assert decision.reason == 'observed_descent_then_low_pose'


def test_resting_walking_and_small_sit_down_do_not_form_transition():
    for mode in ('resting', 'walking', 'sitting'):
        engine = FallTransition()
        for index in range(30):
            detection = pose(.55, .2, True) if mode == 'resting' else pose()
            if mode == 'sitting' and index >= 15:
                detection = pose(.28, .45)
            assert not engine.update(detection, (1000, 1000), index / 10).candidate


def test_missing_pose_or_recovery_stops_continuous_candidate():
    engine = FallTransition()
    alarms = AlarmStateManager(fps=10)
    for index in range(10):
        engine.update(pose(), (1000, 1000), index / 10)
    assert engine.update(pose(.55, .25), (1000, 1000), 1).candidate
    assert not alarms.update(1, True, .8, 10)
    engine.missing()
    alarms.mark_missing(1)
    # Prior real observations can still explain a new pose after a brief
    # gap. They must never count the unobserved interval toward 0.5 s.
    assert engine.update(pose(.55, .25), (1000, 1000), 1.2).candidate
    for index in range(12, 16):
        assert not alarms.update(1, True, .8, index)
    assert not alarms.events
    assert not engine.update(pose(.55, .25), (1000, 1000), 2).candidate


def test_rotated_crop_coordinates_return_to_original_image():
    for rotation in (0, 1, 3):
        original = pose()
        detection = pose()
        xy = original.keypoints[:, :2]
        corners = np.array([[.2, .1], [.8, .9]])
        def rotate(points):
            if rotation == 1:
                return np.stack([points[:, 1], 1 - points[:, 0]], axis=1)
            if rotation == 3:
                return np.stack([1 - points[:, 1], points[:, 0]], axis=1)
            return points
        detection.keypoints[:, :2] = rotate(xy)
        transformed = rotate(corners)
        detection.bbox = np.r_[transformed.min(axis=0), transformed.max(axis=0)]
        restored = PoseRecovery.restore(detection, rotation, (200, 100, 600, 700), (1000, 1000))
        np.testing.assert_allclose(restored.keypoints[:, :2], (xy * [400, 600] + [200, 100]) / 1000)
        np.testing.assert_allclose(restored.bbox, [.28, .16, .52, .64])


def test_partial_core_poses_do_not_bridge_a_long_unobserved_transition():
    engine = FallTransition()
    for index in range(10):
        engine.update(pose(), (1000, 1000), index / 10)
    assert engine.update(pose(.55, .25), (1000, 1000), 1).candidate
    partial = pose(.55, .25)
    partial.keypoints[[5, 6, 11, 12], 2] = .1
    for index in range(11, 19):
        assert not engine.update(partial, (1000, 1000), index / 10).candidate
    assert not engine.update(pose(.55, .25), (1000, 1000), 1.9).candidate


def test_one_small_pose_outlier_does_not_break_an_almost_qualified_candidate():
    engine = FallTransition()
    for index in range(10):
        engine.update(pose(), (1000, 1000), index / 10)
    for index in range(10, 15):
        assert engine.update(pose(.55, .25), (1000, 1000), index / 10).candidate
    # A single distant-person skeleton outlier occurs just before the 0.5 s
    # product timer would qualify. It must not turn a real observed transition
    # into a false negative.
    assert engine.update(pose(), (1000, 1000), 1.5).candidate
    assert engine.update(pose(.55, .25), (1000, 1000), 1.6).candidate
    assert not engine.update(pose(), (1000, 1000), 1.75).candidate
