"""
ByteTrack 人体跟踪封装
------------------------
轻量级多目标跟踪，保持人体 ID 连续性。

作用：
1. 维持同一人体的时序 ID，确保关键点序列属于同一个人
2. 支持隔帧检测优化：无新人进入时不重新检测，仅更新跟踪
3. 减少 YOLO 推理次数（可隔2帧运行），提升整体速度
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from .detector import PersonDetection


@dataclass
class TrackedPerson:
    """跟踪中的人体"""
    track_id: int
    detection: PersonDetection
    age: int = 0                     # 跟踪持续的帧数
    lost_count: int = 0              # 连续丢失帧数
    history: List[PersonDetection] = field(default_factory=list)
    
    def update(self, detection: PersonDetection):
        """更新跟踪状态"""
        self.detection = detection
        self.age += 1
        self.lost_count = 0
        self.history.append(detection)
        if len(self.history) > 60:
            self.history.pop(0)
    
    def mark_lost(self):
        """标记为丢失"""
        self.lost_count += 1


class ByteTrackWrapper:
    """
    ByteTrack 轻量封装
    
    ByteTrack 优势：
    - 基于 IoU 匹配，计算极轻量
    - 支持低分框关联，减少 ID Switch
    - 适合端侧部署
    """
    
    def __init__(
        self,
        track_thresh: float = 0.5,         # 高分框阈值
        match_thresh: float = 0.3,          # 首次匹配 IoU 阈值（原 0.8 过严，框抖动即失配 → ID 碎片化）
        second_match_thresh: float = 0.2,   # 二次匹配 IoU 阈值
        track_buffer: int = 30,             # 丢失后保留帧数
        frame_rate: int = 30,               # 帧率
        max_age: int = 60,                  # 最大跟踪寿命
        pose_association: bool = False,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        self.max_age = max_age
        self.pose_association = pose_association
        
        # 内部状态
        self._next_id = 0
        self._tracks: Dict[int, TrackedPerson] = {}
        self._frame_count = 0
    
    def reset(self):
        """重置跟踪器状态"""
        self._next_id = 0
        self._tracks.clear()
        self._frame_count = 0
    
    def update(
        self,
        detections: List[PersonDetection],
        detection_skip: int = 1,
        is_detection_frame: Optional[bool] = None,
    ) -> Dict[int, PersonDetection]:
        """
        更新跟踪状态
        
        Args:
            detections: 当前帧检测结果
            detection_skip: 检测间隔（如为2则隔帧检测）
        
        Returns:
            Dict[track_id, PersonDetection]: 当前帧的跟踪结果
        """
        # The caller knows whether YOLO actually ran. An empty result from
        # YOLO means "missed", not an intentionally skipped observation.
        if is_detection_frame is None:
            is_detection_frame = self._frame_count % max(1, detection_skip) == 0
        self._frame_count += 1
        if not is_detection_frame:
            return self._predict_only()
        
        # 分离高低分框
        high_score_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_score_dets = [d for d in detections if d.confidence < self.track_thresh]
        
        # 获取活跃跟踪
        active_tracks = self._get_active_tracks()
        
        # Step 1: 高分框与所有活跃跟踪匹配
        matched, unmatched_dets, unmatched_trks = self._match_detections(
            high_score_dets, active_tracks, self.match_thresh
        )
        
        # Step 2: 更新已匹配的跟踪
        # 注意：matched 的第一个元素是 active_tracks 列表索引，不是 track_id
        for trk_idx, det_idx in matched:
            tid = active_tracks[trk_idx].track_id
            self._tracks[tid].update(high_score_dets[det_idx])
        
        # Step 3: 低分框与剩余未匹配跟踪匹配
        remaining_dets = [high_score_dets[i] for i in unmatched_dets] + low_score_dets
        remaining_trks = [active_tracks[i] for i in unmatched_trks]
        matched2 = []  # 初始化，防止 UnboundLocalError
        
        if len(remaining_dets) > 0 and len(remaining_trks) > 0:
            matched2, _, _ = self._match_detections(
                remaining_dets, remaining_trks, self.second_match_thresh
            )
            
            # Map back to original track_id
            for trk_remap, det_idx in matched2:
                tid = remaining_trks[trk_remap].track_id
                self._tracks[tid].update(remaining_dets[det_idx])
        
        # Step 4: 未匹配的高分框 → 新跟踪
        # matched 的 det_idx 基于 high_score_dets；matched2 的 det_idx 基于
        # remaining_dets（前 len(unmatched_dets) 个对应 unmatched_dets 中的高分框）
        matched_high = set(det_idx for _, det_idx in matched)
        for _, det_idx in matched2:
            if det_idx < len(unmatched_dets):
                matched_high.add(unmatched_dets[det_idx])
        
        new_ids = []
        for i, det in enumerate(high_score_dets):
            if i not in matched_high:
                track_id = self._next_id
                self._next_id += 1
                new_ids.append(track_id)
                self._tracks[track_id] = TrackedPerson(
                    track_id=track_id,
                    detection=det,
                    age=0,
                    history=[det],
                )
        
        # Step 5: 标记丢失的跟踪（含二次匹配与新建 track，避免误标 lost）
        matched_trk_ids = set(active_tracks[t].track_id for t, _ in matched)
        matched_trk_ids |= set(remaining_trks[t].track_id for t, _ in matched2)
        matched_trk_ids |= set(new_ids)
        for track_id, track in self._tracks.items():
            if track_id not in matched_trk_ids:
                track.mark_lost()
        
        # Step 6: 清理过期跟踪
        self._cleanup()
        
        # 返回当前帧结果
        return self._get_current_tracks()
    
    def _predict_only(self) -> Dict[int, PersonDetection]:
        """仅预测（无新检测时的跟踪维持）"""
        # 保持现有跟踪
        return {
            tid: t.detection
            for tid, t in self._tracks.items()
            if t.lost_count == 0
        }
    
    def _get_active_tracks(self) -> List['TrackedPerson']:
        """获取活跃的跟踪目标"""
        return [
            t for t in self._tracks.values()
            if t.lost_count < self.track_buffer
        ]
    
    def _match_detections(
        self,
        detections: List[PersonDetection],
        tracks: List['TrackedPerson'],
        iou_threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        IoU 匹配
        
        Returns:
            matched: [(track_index, det_index), ...]
            unmatched_dets: [det_index, ...]
            unmatched_tracks: [track_index, ...]
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # 计算 IoU 矩阵
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracks):
                iou_matrix[i, j] = self._compute_iou(
                    det.bbox, trk.detection.bbox
                )
                # A falling person's box can shrink/rotate abruptly while
                # shoulders and hips still describe the same nearby person.
                # Use agreement of several visible joints only as a fallback
                # for an otherwise unmatched pair, never a stale observation.
                if self.pose_association and iou_matrix[i, j] < iou_threshold:
                    before = trk.detection.keypoints
                    after = det.keypoints
                    valid = (before[5:, 2] > .3) & (after[5:, 2] > .3)
                    if valid.sum() >= 6:
                        scale = np.linalg.norm(trk.detection.bbox[2:] - trk.detection.bbox[:2])
                        distances = np.linalg.norm(before[5:, :2][valid] - after[5:, :2][valid], axis=1)
                        if scale > 0 and np.median(distances) / scale < .25:
                            iou_matrix[i, j] = max(iou_matrix[i, j], iou_threshold + 1e-6)
        
        # 贪婪匹配（简化版匈牙利算法）
        matched = []
        used_dets = set()
        used_trks = set()
        
        # 按 IoU 降序排列
        indices = np.argsort(-iou_matrix, axis=None)
        for idx in indices:
            i = idx // len(tracks)
            j = idx % len(tracks)
            if iou_matrix[i, j] >= iou_threshold and i not in used_dets and j not in used_trks:
                matched.append((j, i))
                used_dets.add(i)
                used_trks.add(j)
        
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
        unmatched_trks = [j for j in range(len(tracks)) if j not in used_trks]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """计算两个边界框的 IoU"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter_area = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        return inter_area / (area1 + area2 - inter_area + 1e-6)
    
    def _get_current_tracks(self) -> Dict[int, PersonDetection]:
        """获取当前帧的有效跟踪结果"""
        return {
            tid: t.detection
            for tid, t in self._tracks.items()
            # Lost tracks remain available for association, but their old
            # skeleton is not a new observation for the temporal model.
            if t.lost_count == 0
        }

    def get_visual_tracks(self, max_lost: int = 2) -> Dict[int, PersonDetection]:
        """Return recent tracks for drawing on every video frame.

        A detector may intentionally run every Nth frame, and an occasional
        detector miss should not make the overlay blink.  This method is only
        for visualization: the temporal fall logic continues to use
        ``_get_current_tracks`` and therefore does not treat stale poses as
        new observations.
        """
        return {
            tid: track.detection
            for tid, track in self._tracks.items()
            if track.lost_count <= max(0, max_lost)
        }
    
    def _cleanup(self):
        """清理过期跟踪"""
        to_remove = [
            tid for tid, t in self._tracks.items()
            if t.lost_count > self.track_buffer * 2
        ]
        for tid in to_remove:
            del self._tracks[tid]
    
    def get_track_history(
        self,
        track_id: int,
        max_len: int = 32,
    ) -> List[PersonDetection]:
        """获取指定跟踪 ID 的历史检测"""
        if track_id not in self._tracks:
            return []
        return self._tracks[track_id].history[-max_len:]
    
    @property
    def active_count(self) -> int:
        """当前活跃跟踪数"""
        return sum(1 for t in self._tracks.values() if t.lost_count < self.track_buffer)
