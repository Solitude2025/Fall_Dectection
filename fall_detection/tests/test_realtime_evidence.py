"""Behaviour tests for gaps and model support in the live alarm decision."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pipeline.detector import PersonDetection
from pipeline.inference import FallDetectionPipeline
from models.motion_encoder import MotionFeatureEncoder


def pose(low=False, partial=False):
    height, shoulder = (.25, .55) if low else (.6, .2)
    kp = np.full((17, 3), .9)
    kp[:, 0] = .5
    kp[:, 1] = np.linspace(shoulder-.1*height, shoulder+.9*height, 17)
    kp[[5,6],1] = shoulder
    kp[[11,12],1] = shoulder+.35*height
    if partial:
        kp[3:,2] = .1
    return PersonDetection(np.array([.2,.1,.8,.9]), kp, .9)


def live_pipeline():
    model = SimpleNamespace(eval=lambda: None, motion_encoder=MotionFeatureEncoder(feature_dim=64),
                            score=.8)
    model.tcn = lambda features: torch.full((1,features.shape[1],1),model.score)
    pipeline = FallDetectionPipeline(
        detector=SimpleNamespace(), fall_detector=model, detection_interval=1,
        auto_lowlight=False, require_observed_transition=True,
        transition_max_gap_seconds=1.5, preserve_short_gaps_seconds=1.2,
        model_support_grace_seconds=1.2, max_candidate_gap_seconds=.35)
    pipeline.alarm_state.use_wall_clock = True
    return pipeline, model


def run(pipeline, seconds, detection):
    return pipeline.process_frame(
        np.zeros((100,100,3),np.uint8), round(seconds*100),
        timestamp=datetime(2026,10,16,tzinfo=timezone.utc)+timedelta(seconds=seconds),
        detections_override=[] if detection is None else [detection])


def warmup(pipeline):
    for i in range(10):
        run(pipeline,i/10,pose())


@pytest.mark.parametrize('missing', [None, pose(partial=True)])
def test_short_gaps_retain_history_but_never_qualify_alarm(missing):
    pipeline, _ = live_pipeline()
    warmup(pipeline)
    assert not run(pipeline,1.0,pose(True)).alarms[0]
    old_length=len(pipeline._kp_buffers[0])
    gap=run(pipeline,1.1,missing)
    assert not any(gap.alarms.values())
    assert len(pipeline._kp_buffers[0])==old_length
    assert not run(pipeline,1.3,pose(True)).alarms[0]
    for seconds in (1.4,1.5,1.6,1.7):
        assert not run(pipeline,seconds,pose(True)).alarms[0]
    assert run(pipeline,1.8,pose(True)).alarms[0]
    assert len(pipeline._fall_events)==1


@pytest.mark.parametrize('missing', [None, pose(partial=True)])
def test_long_missing_or_partial_pose_expires_history(missing):
    pipeline, _ = live_pipeline()
    warmup(pipeline)
    for i in range(10,27):
        run(pipeline,i/10,missing)
    assert not pipeline._kp_buffers.get(0)
    assert 0 not in pipeline.transitions
    for i in range(27,37):
        assert not any(run(pipeline,i/10,pose(True)).alarms.values())
    assert not pipeline._fall_events


def test_model_support_is_bound_to_actual_transition():
    pipeline, model = live_pipeline()
    warmup(pipeline)  # High scores during standing must not enable a descent.
    model.score=.1
    for i in range(10,19):
        assert not run(pipeline,i/10,pose(True)).model_alarms[0]
    assert not pipeline._fall_events


def test_transition_supported_by_model_peak_can_survive_score_dip():
    pipeline, model = live_pipeline()
    warmup(pipeline)
    run(pipeline,1,pose(True))
    model.score=.1
    for i in range(11,15):
        assert not run(pipeline,i/10,pose(True)).alarms[0]
    assert run(pipeline,1.5,pose(True)).alarms[0]
    # Finite support cannot enable another fall later without model evidence.
    run(pipeline,2.3,pose())
    assert not run(pipeline,2.4,pose(True)).model_alarms[0]


def test_worker_pause_does_not_count_as_continuous_observation():
    pipeline, _ = live_pipeline()
    warmup(pipeline)
    run(pipeline,1,pose(True))
    assert not run(pipeline,1.7,pose(True)).alarms[0]
    assert not pipeline._fall_events


def test_initially_low_person_never_inherits_other_person_history():
    pipeline, _ = live_pipeline()
    warmup(pipeline)
    other=pose(True)
    other.keypoints[:,0]=.05
    other.bbox=np.array([.01,.4,.1,.9])
    for i in range(10,22):
        assert not any(run(pipeline,i/10,other).alarms.values())
    assert not pipeline._fall_events


def test_gap_velocity_is_scaled_by_elapsed_time_without_fake_poses():
    encoder=MotionFeatureEncoder(feature_dim=64)
    kp=np.stack([pose().keypoints for _ in range(8)])
    bb=np.stack([pose().bbox for _ in range(8)])
    kp[4:,:,1]+=.2
    bb[4:,[1,3]]+=.2
    plain=encoder.compute_sequence(kp,bb)
    scales=np.ones(8,np.float32)
    scales[4]=.2
    corrected=encoder.compute_sequence(kp,bb,step_scales=scales)
    assert corrected[4,3]==pytest.approx(plain[4,3]*.2)
    assert len(corrected)==len(kp)
    np.testing.assert_allclose(encoder.compute_sequence(kp,bb,step_scales=np.ones(8,np.float32)),plain)


def test_seated_vertical_torso_cannot_be_a_standing_reference():
    from pipeline.fall_transition import FallTransition
    transition=FallTransition(require_upright_reference=True)
    seated=pose()
    seated.keypoints[[11,12],:2]=[.4,.5]
    seated.keypoints[[13,14],:2]=[.65,.5]
    seated.keypoints[[15,16],:2]=[.65,.8]
    for i in range(10):
        assert not transition.update(seated,(100,100),i/10).candidate
    for i in range(10,20):
        assert not transition.update(pose(True),(100,100),i/10).candidate
