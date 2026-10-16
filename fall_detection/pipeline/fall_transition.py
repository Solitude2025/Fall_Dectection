"""Observed upright-to-low transitions, independent of left/right direction.

This is a physical evidence branch, not a calibrated probability model. The
alarm timer resets on missing poses; missing time never counts as a fall.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class TransitionDecision:
    candidate: bool = False
    reason: str = "no_transition"
    drop: float = 0.0
    height_ratio: float = 1.0


class FallTransition:
    def __init__(self, history_seconds=1.5, jitter_grace_seconds=.12,
                 min_candidate_before_jitter=.35,
                 candidate_window_seconds=3.0,
                 max_observation_gap_seconds=.6,
                 require_upright_reference=False):
        self.history_seconds = history_seconds
        # Small, distant people often produce one outlier skeleton while the
        # person is already on the floor.  A very short grace period keeps that
        # observation noise from breaking the product's 0.5 s timer.  Missing
        # poses still go through AlarmStateManager as a hard reset.
        self.jitter_grace_seconds = jitter_grace_seconds
        self.min_candidate_before_jitter = min_candidate_before_jitter
        self.candidate_window_seconds = candidate_window_seconds
        # Live inference can be slower than the camera.  A short detector
        # gap is allowed to separate two real observations of the same track;
        # no synthetic pose is inserted and the alarm timer still resets on
        # every missing frame.
        self.max_observation_gap_seconds = max(.0, float(max_observation_gap_seconds))
        self.require_upright_reference = bool(require_upright_reference)
        self.history = deque()
        self.reference = None
        self.transition_time = None
        self.last_candidate_time = None
        self.last_seen_time = None

    @staticmethod
    def measure(detection, shape):
        h, w = shape
        k = detection.keypoints
        if min(k[[5, 6, 11, 12], 2]) < .3:
            return None
        shoulder = (k[5, :2] + k[6, :2]) / 2 * [w, h]
        hip = (k[11, :2] + k[12, :2]) / 2 * [w, h]
        visible = k[:, 2] > .3
        points = k[visible, :2] * [w, h]
        if len(points) < 8:
            return None
        height = float(np.ptp(points[:, 1]))
        vector = shoulder - hip
        angle = float(np.arctan2(abs(vector[0]), abs(vector[1]) + 1e-6))
        # A seated person can have a vertical torso too. Live mode requires
        # an observed extended leg before using a pose as standing evidence.
        standing_leg = False
        for indices in ([11, 13, 15], [12, 14, 16]):
            if min(k[indices, 2]) < .3:
                continue
            leg = k[indices, :2] * [w, h]
            upper, lower = leg[0]-leg[1], leg[2]-leg[1]
            cosine = float(np.dot(upper, lower)/(np.linalg.norm(upper)*np.linalg.norm(lower)+1e-6))
            if (cosine < -.75 and leg[0,1] < leg[1,1] < leg[2,1]
                    and leg[2,1]-leg[0,1] > .25*height):
                standing_leg = True
        return dict(shoulder=shoulder, hip=hip, height=max(height, 1), angle=angle,
                    upright_reference=bool(standing_leg and angle < .65 and shoulder[1] < hip[1]))

    def missing(self):
        # Keep actual pre-gap evidence for the bounded history window, but
        # the caller resets the alarm's consecutive timer on every gap.
        return TransitionDecision(reason="pose_missing")

    def update(self, detection, shape, time_seconds):
        if (self.last_seen_time is not None
                and time_seconds - self.last_seen_time > self.max_observation_gap_seconds):
            self.reference = None
            self.transition_time = None
            self.last_candidate_time = None
            self.history.clear()
        current = self.measure(detection, shape)
        if current is None:
            self.missing()
            return TransitionDecision(reason="core_keypoints_missing")
        # A box with missing shoulders/hips is not a reliable observation.
        # Do not let repeated partial poses extend an old transition's life.
        self.last_seen_time = time_seconds
        while self.history and time_seconds - self.history[0][0] > self.history_seconds:
            self.history.popleft()
        decision = TransitionDecision()
        # Compare only to actually observed earlier non-horizontal poses, never a
        # synthetic or extrapolated reference. At least two observations
        # spanning 0.1 s are needed before a new transition can begin.
        if self.reference is None:
            upright = [(t, m) for t, m in self.history
                       if m['angle'] < 1.3 and time_seconds - t >= .1
                       and (not self.require_upright_reference or m['upright_reference'])]
            if len(upright) >= 2:
                baseline = max(upright, key=lambda item: item[1]['height'])[1]
                drop = float((current['shoulder'][1] - baseline['shoulder'][1]) / baseline['height'])
                ratio = current['height'] / baseline['height']
                changed = ratio < .65 or current['angle'] > .8
                rotated_and_compressed = ratio < .6 and current['angle'] - baseline['angle'] > .5
                if (drop > .25 and changed) or rotated_and_compressed:
                    self.reference = baseline
                    self.transition_time = time_seconds
                    self.last_candidate_time = None
        if self.reference is not None:
            baseline = self.reference
            decision.drop = float((current['shoulder'][1] - baseline['shoulder'][1]) / baseline['height'])
            decision.height_ratio = current['height'] / baseline['height']
            low = ((decision.drop > .2 and (decision.height_ratio < .85 or current['angle'] > .8))
                   or (decision.height_ratio < .7 and current['angle'] > .65)
                   or (current['angle'] > .8 and current['angle'] - baseline['angle'] > .35))
            if low and time_seconds - self.transition_time <= self.candidate_window_seconds:
                decision.candidate = True
                decision.reason = "observed_descent_then_low_pose"
                self.last_candidate_time = time_seconds
            elif (
                self.transition_time is not None
                and self.last_candidate_time is not None
                and time_seconds - self.transition_time >= self.min_candidate_before_jitter
                and time_seconds - self.last_candidate_time <= self.jitter_grace_seconds
            ):
                # Do not move last_candidate_time here. The grace covers one
                # short outlier only, rather than keeping a person who has
                # stood up in the candidate state indefinitely.
                decision.candidate = True
                decision.reason = "observed_descent_then_low_pose_jitter_tolerated"
            else:
                self.reference = None
                self.transition_time = None
                self.last_candidate_time = None
                decision.reason = "recovered_or_expired"
        self.history.append((time_seconds, current))
        return decision
