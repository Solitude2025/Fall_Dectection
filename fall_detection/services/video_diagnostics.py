"""Observation quality and stage traces; never override the alarm decision."""
from collections import defaultdict


class VideoDiagnostics:
    def __init__(self, fps: float, min_completeness: float):
        self.fps = fps
        self.min_completeness = min_completeness
        self.timeline = []
        self._seen = False
        self._gap_start = None
        self._longest_gap = None
        self._streaks = defaultdict(int)
        self._longest_streak = 0
        self._core_observed = 0

    def update(self, result):
        reliable = any(float((p.keypoints[:, 2] > .3).mean()) >= self.min_completeness
                       for p in result.persons)
        core_reliable = any(float((p.keypoints[:, 2] > .3).mean()) >= self.min_completeness
                            and min(p.keypoints[[5, 6, 11, 12], 2]) >= .3
                            for p in result.persons)
        self._core_observed += int(core_reliable)
        index = result.frame_idx
        if reliable:
            self._seen = True
            self._gap_start = None
        elif self._seen:
            if self._gap_start is None:
                self._gap_start = index
            gap = (self._gap_start, index)
            if self._longest_gap is None or gap[1] - gap[0] > self._longest_gap[1] - self._longest_gap[0]:
                self._longest_gap = gap
        ids = set(self._streaks) | set(result.model_alarms)
        for track_id in ids:
            self._streaks[track_id] = self._streaks[track_id] + 1 if result.model_alarms.get(track_id) else 0
            self._longest_streak = max(self._longest_streak, self._streaks[track_id])
        raw = max(result.raw_fall_probs.values(), default=None)
        candidate = any(result.model_alarms.values())
        alarm = any(result.alarms.values())
        self.timeline.append({
            "time_seconds": round(index / self.fps, 4),
            "person_observed": reliable,
            "core_pose_observed": core_reliable,
            "raw_probability": round(raw, 5) if raw is not None else None,
            "rule_candidate": candidate,
            "alarm": alarm,
            "decisions": getattr(result, "decision_details", {}),
            "observations": {
                str(tid): {"bbox": p.bbox.round(5).tolist(),
                           "confidence": round(float(p.confidence), 4),
                           "keypoints": p.keypoints.round(5).tolist(),
                           "visible_keypoints": int((p.keypoints[:, 2] > .3).sum())}
                for tid, p in getattr(result, "visual_persons", {}).items()
            },
            "stage": "alarm" if alarm else "pose_missing" if not reliable else
                     "warming_up" if raw is None else "candidate" if candidate else "rule_rejected",
        })

    def report(self):
        total = len(self.timeline)
        observed = sum(r["person_observed"] for r in self.timeline)
        gap_seconds = 0 if self._longest_gap is None else (self._longest_gap[1] - self._longest_gap[0] + 1) / self.fps
        warnings = []
        if not observed:
            warnings.append("未持续识别到完整人体，无法可靠判断，请人工查看视频。")
        elif gap_seconds >= .25:
            start, end = self._longest_gap
            warnings.append(f"{start / self.fps:.2f}～{(end + 1) / self.fps:.2f} 秒未可靠识别到人体姿态，可能漏检，请人工复核。建议上传原视频，减少黑边和遮挡。")
        return {
            "quality": {
                "observed_frame_ratio": round(observed / max(total, 1), 4),
                "core_pose_frame_ratio": round(self._core_observed / max(total, 1), 4),
                "longest_pose_gap_seconds": round(gap_seconds, 3),
                "max_raw_probability": max((r["raw_probability"] for r in self.timeline if r["raw_probability"] is not None), default=None),
                "longest_candidate_seconds": round(self._longest_streak / self.fps, 3),
            },
            "warnings": warnings,
            "timeline": self.timeline,
        }
