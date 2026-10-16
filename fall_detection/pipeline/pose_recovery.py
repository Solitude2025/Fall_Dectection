"""Bounded extra pose inference for small, occluded or rotated people.

All returned coordinates refer to the original image. Recovery never invents
keypoints, carries an old observation forward, or decides whether to alarm.
"""
from dataclasses import dataclass, field

import numpy as np

from .detector import PersonDetection


@dataclass
class _Reference:
    bbox: np.ndarray
    torso_lengths: list = field(default_factory=list)
    missed: int = 0
    anchor: np.ndarray = None
    recent_angles: list = field(default_factory=list)


class PoseRecovery:
    def __init__(self, detector, max_regions=2, max_missing_seconds=.6):
        self.detector = detector
        self.max_regions = max_regions
        self.max_missing_seconds = max_missing_seconds
        self.references = {}
        self.extra_calls = 0

    def reset(self):
        self.references.clear()
        self.extra_calls = 0

    @staticmethod
    def _iou(a, b):
        overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
        intersection = float(np.prod(overlap))
        return intersection / max(float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2])) - intersection, 1e-9)

    @staticmethod
    def _torso(detection, shape):
        h, w = shape
        k = detection.keypoints
        if min(k[[5, 6, 11, 12], 2]) < .3:
            return None
        vector = (k[5, :2] + k[6, :2] - k[11, :2] - k[12, :2]) / 2
        return float(np.linalg.norm(vector * [w, h]))

    def observe(self, tracked, shape):
        for tid, detection in tracked.items():
            reference = self.references.setdefault(tid, _Reference(detection.bbox.copy()))
            reference.bbox = detection.bbox.copy()
            reference.missed = 0
            length = self._torso(detection, shape)
            if length and detection.confidence >= .6:
                reference.torso_lengths.append(length)
                reference.torso_lengths = reference.torso_lengths[-30:]
            k = detection.keypoints
            v = (k[5, :2] + k[6, :2] - k[11, :2] - k[12, :2]) / 2
            angle = float(np.arctan2(abs(v[0]) * shape[1], abs(v[1]) * shape[0] + 1e-6))
            reference.recent_angles.append(angle)
            reference.recent_angles = reference.recent_angles[-8:]
            if reference.anchor is None or (angle < .35 and detection.confidence >= .6):
                reference.anchor = detection.bbox.copy()

    @staticmethod
    def restore(detection, rotation, bounds, shape):
        """Invert np.rot90, then map a crop back to full-image coordinates."""
        x1, y1, x2, y2 = bounds
        h, w = shape
        keypoints = detection.keypoints.copy()
        xy = keypoints[:, :2].copy()
        box = detection.bbox
        corners = np.array([[box[0], box[1]], [box[2], box[3]]])
        def invert(points):
            if rotation == 1:
                return np.stack([1 - points[:, 1], points[:, 0]], axis=1)
            if rotation == 3:
                return np.stack([points[:, 1], 1 - points[:, 0]], axis=1)
            return points
        scale = np.array([x2 - x1, y2 - y1])
        keypoints[:, :2] = (invert(xy) * scale + [x1, y1]) / [w, h]
        corners = (invert(corners) * scale + [x1, y1]) / [w, h]
        restored_box = np.r_[corners.min(axis=0), corners.max(axis=0)]
        return PersonDetection(restored_box, keypoints, detection.confidence)

    def recover(self, image, detections, fps, interval):
        h, w = image.shape[:2]
        output = list(detections)
        regions = 0
        expired = []
        for tid, reference in self.references.items():
            if reference.missed * interval / fps > self.max_missing_seconds:
                expired.append(tid)
        for tid in expired:
            del self.references[tid]
        for reference in self.references.values():
            reference.missed += 1
            if reference.missed * interval / fps > self.max_missing_seconds:
                continue
            bbox = reference.bbox
            matches = [d for d in output if self._iou(d.bbox, bbox) > .1]
            if (matches and max(d.confidence for d in matches) >= .6
                    and not any(a > .35 for a in reference.recent_angles)):
                continue
            if regions >= self.max_regions:
                break
            regions += 1
            anchor = reference.anchor if reference.anchor is not None else bbox
            region = np.r_[np.minimum(anchor[:2], bbox[:2]), np.maximum(anchor[2:], bbox[2:])]
            center = (region[:2] + region[2:]) / 2
            extent = np.maximum(region[2:] - region[:2], [.05, .06]) * [2.8, 2.2]
            lower = np.maximum(0, (center - extent / 2) * [w, h]).astype(int)
            upper = np.minimum([w, h], (center + extent / 2) * [w, h]).astype(int)
            x1, y1 = lower
            x2, y2 = upper
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            candidates = list(matches)
            for rotation in (0, 1, 3):
                crop = np.ascontiguousarray(np.rot90(image[y1:y2, x1:x2], rotation))
                self.extra_calls += 1
                for local in self.detector.detect(crop):
                    restored = self.restore(local, rotation, (x1, y1, x2, y2), (h, w))
                    if self._iou(restored.bbox, bbox) > .05:
                        candidates.append(restored)
            def quality(detection):
                length = self._torso(detection, (h, w))
                completeness = float((detection.keypoints[:, 2] > .3).mean())
                if not length or completeness < .5:
                    return -1
                consistency = 1.0
                if reference.torso_lengths:
                    ratio = length / max(float(np.median(reference.torso_lengths)), 1)
                    # A detector sometimes hallucinates upright legs under a
                    # horizontal body. Penalise abrupt torso-size changes.
                    consistency = float(np.exp(-abs(np.log(max(ratio, .01)))))
                return float(detection.confidence) * completeness * consistency ** 2
            if not candidates or max(quality(d) for d in candidates) < .15:
                # A final higher-resolution pass is restricted to a known
                # person's small region, so full-frame cost stays bounded.
                for rotation in (0, 1, 3):
                    crop = np.ascontiguousarray(np.rot90(image[y1:y2, x1:x2], rotation))
                    self.extra_calls += 1
                    for local in self.detector.detect(crop, input_size=(960, 960)):
                        restored = self.restore(local, rotation, (x1, y1, x2, y2), (h, w))
                        if self._iou(restored.bbox, bbox) > .05:
                            candidates.append(restored)
            if candidates:
                best = max(candidates, key=quality)
                if quality(best) > 0:
                    output = [d for d in output if not any(d is m for m in matches)]
                    output.append(best)
        return output
