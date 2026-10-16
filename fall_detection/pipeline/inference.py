"""
完整推理管线
--------------
整合 检测 → 跟踪 → 运动编码 → TCN → 规则校正 的完整端到端推理流程。

支持模式：
1. 视频文件推理（离线评测）
2. 摄像头实时推理（在线部署）
3. 批量推理（数据集评测）

核心优化：
- 隔帧检测：YOLO 可以隔 2 帧运行，TCN 每帧运行
- 分辨率自适应：1080P → 320/640 推理
- 人体 ROI 裁剪：仅对检测到的人体区域进行后续处理
"""

import os
import cv2
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict
from dataclasses import dataclass, field
import time
import json
from datetime import datetime

from .detector import (
    YOLOPoseDetector, YOLOPoseDetectorSim,
    PersonDetection,
)
from .tracker import ByteTrackWrapper
from models.motion_encoder import MotionFeatureEncoder
from models.light_tcn import LightTCN
from models.rule_refinement import RuleRefinementOnline, RuleConfig
from models.fall_detector import FallDetector
from .alarm_state import AlarmEventRecord, AlarmStateManager
from .pose_recovery import PoseRecovery
from .fall_transition import FallTransition


# Keep the public name used by older evaluation scripts while using the new
# application event contract internally.
FallEvent = AlarmEventRecord


@dataclass
class FrameResult:
    """单帧推理结果"""
    frame_idx: int
    persons: List[PersonDetection]
    fall_probs: Dict[int, float]    # track_id → probability
    alarms: Dict[int, bool]          # track_id → alarm
    features: Optional[Dict[int, np.ndarray]] = None
    is_detect_frame: bool = True     # 当前帧是否执行了 YOLO 检测
    raw_detections: List[PersonDetection] = field(default_factory=list)  # YOLO 原始检测结果
    event_updates: List[dict] = field(default_factory=list)  # 新建/解除事件消息
    model_alarms: Dict[int, bool] = field(default_factory=dict)  # 去抖前模型信号
    raw_fall_probs: Dict[int, float] = field(default_factory=dict)  # TCN 分数，用于诊断
    decision_details: Dict[int, dict] = field(default_factory=dict)
    # Visualization-only tracks. These may reuse the latest pose for a few
    # frames when YOLO is skipped or briefly misses; they never enter the
    # alarm/TCN calculation.
    visual_persons: Dict[int, PersonDetection] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """完整推理结果"""
    frame_results: List[FrameResult]
    fall_events: List[FallEvent]
    total_frames: int
    total_time: float
    fps: float
    avg_latency_ms: float


class FallDetectionPipeline:
    """
    完整跌倒检测推理管线
    
    ┌─────────────┐
    │  视频输入    │
    │ (RGB/红外)   │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 预处理      │
    │ Resize,CLAHE │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ YOLOv8n-Pose │  (隔帧运行)
    │ 检测+关键点  │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ ByteTrack   │
    │ 人体跟踪     │
    └──────┬──────┘
           ▼
    ┌─────────────────────┐
    │ Motion Feature       │
    │ Encoder              │
    │ (关键点→运动特征)     │
    └──────┬──────────────┘
           ▼
    ┌─────────────┐
    │ Light-TCN   │  (每帧运行)
    │ 时序建模     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ Rule         │
    │ Refinement   │
    │ 规则校正     │
    └──────┬──────┘
           ▼
    ┌─────────────┐
    │ 报警/记录    │
    └─────────────┘
    """
    
    def __init__(
        self,
        detector: Optional[YOLOPoseDetector] = None,
        fall_detector: Optional[FallDetector] = None,
        # 管线参数
        sequence_length: int = 32,
        detection_interval: int = 2,       # YOLO 检测间隔（隔N帧）
        input_size: Tuple[int, int] = (640, 384),
        conf_threshold: float = 0.25,
        # 红外模式
        ir_mode: bool = False,
        # 自动低光增强（亮度自适应，正常光照无影响）
        auto_lowlight: bool = True,
        # 关键点完整性门控（低于该比例的帧不喂 TCN，防半入境误判）
        min_kp_completeness: float = 0.5,
        # 概率时序平滑窗口（对最近 N 帧 TCN 概率取均值，1=不平滑）
        prob_smooth: int = 3,
        # 报警参数
        alarm_cooldown_frames: int = 30,
        camera_id: str = "laptop_camera",
        alarm_trigger_seconds: float = 0.5,
        alarm_hold_seconds: float = 30.0,
        fps: float = 30.0,
        # 输出参数
        save_video: bool = False,
        output_dir: str = "output",
        # 规则配置（从 config.yaml 传入）
        rule_config: Optional[RuleConfig] = None,
        pose_recovery: bool = False,
        require_observed_transition: bool = False,
        high_res_rescue: bool = False,
        rescue_input_size: Tuple[int, int] = (1280, 736),
        cpu_threads: Optional[int] = None,
        transition_max_gap_seconds: float = .6,
        preserve_short_gaps_seconds: float = 0.0,
        model_support_grace_seconds: float = 0.0,
        max_candidate_gap_seconds: float = 0.0,
    ):
        """
        Args:
            detector: YOLO Pose 检测器
            fall_detector: 跌倒检测模型
            sequence_length: TCN 序列长度
            detection_interval: YOLO 检测间隔
            input_size: 模型输入尺寸
            conf_threshold: 检测置信度阈值
            ir_mode: 是否红外模式
            auto_lowlight: 是否自动低光增强（默认开，亮帧不触发）
            alarm_cooldown_frames: 兼容旧配置，应用层报警保持时间由 alarm_hold_seconds 控制
            save_video: 是否保存结果视频
            output_dir: 输出目录
        """
        # 检测器
        if detector is None:
            try:
                detector = YOLOPoseDetector(
                    input_size=input_size,
                    conf_threshold=conf_threshold,
                )
                detector.load_model()
            except:
                detector = YOLOPoseDetectorSim()
        self.detector = detector
        self.pose_recovery = PoseRecovery(detector) if pose_recovery else None
        self.require_observed_transition = require_observed_transition
        # Offline files can afford a second, larger YOLO pass when the normal
        # pass misses a person or returns a very incomplete pose.  Live camera
        # mode leaves this disabled to preserve its frame rate.
        self.high_res_rescue = bool(high_res_rescue)
        self.rescue_input_size = tuple(rescue_input_size)
        self.cpu_threads = cpu_threads
        self.transitions: Dict[int, FallTransition] = {}
        self.transition_max_gap_seconds = max(0.0, float(transition_max_gap_seconds))
        self.preserve_short_gaps_seconds = max(0.0, float(preserve_short_gaps_seconds))
        self.max_candidate_gap_seconds = max(0.0, float(max_candidate_gap_seconds))
        self._last_observed_at: Dict[int, float] = {}
        self._time_buffers: Dict[int, list] = defaultdict(list)
        self.model_support_grace_seconds = max(0.0, float(model_support_grace_seconds))
        self._model_support_until: Dict[int, float] = {}

        # 跟踪器
        self.tracker = ByteTrackWrapper(track_buffer=30, pose_association=pose_recovery)

        # 跌倒检测模型
        if fall_detector is None:
            fall_detector = FallDetector()
        self.fall_detector = fall_detector
        self.fall_detector.eval()
        
        # 在线规则校正（每人体一个实例）
        self.rule_engines: Dict[int, RuleRefinementOnline] = {}
        
        # 规则配置（从 config.yaml 传入）
        self.rule_config = RuleConfig()
        if rule_config is not None:
            self.rule_config = rule_config
        
        # 参数
        self.sequence_length = sequence_length
        self.detection_interval = detection_interval
        self.input_size = input_size
        self.ir_mode = ir_mode
        self.auto_lowlight = auto_lowlight
        self.min_kp_completeness = min_kp_completeness
        self.prob_smooth = max(1, prob_smooth)
        self.alarm_cooldown = alarm_cooldown_frames
        self.fps = max(float(fps), 1.0)
        self.camera_id = camera_id
        self.alarm_state = AlarmStateManager(
            camera_id=camera_id,
            fps=self.fps,
            trigger_seconds=alarm_trigger_seconds,
            hold_seconds=alarm_hold_seconds,
        )
        
        # 特征缓冲区（按人体 ID 维护）
        self._feature_buffers: Dict[int, list] = defaultdict(list)
        self._prob_buffers: Dict[int, list] = defaultdict(list)
        self._kp_buffers: Dict[int, list] = defaultdict(list)
        self._bbox_buffers: Dict[int, list] = defaultdict(list)
        
        # 输出
        self.save_video = save_video
        self.output_dir = output_dir
        if save_video:
            os.makedirs(output_dir, exist_ok=True)
        
        # 跌倒事件记录
        self._fall_events: List[FallEvent] = self.alarm_state.events
        self._fall_cooldowns: Dict[int, int] = {}
    
    def reset(self):
        """重置管线状态"""
        self.tracker.reset()
        self.rule_engines.clear()
        self.transitions.clear()
        self._last_observed_at.clear()
        self._time_buffers.clear()
        self._model_support_until.clear()
        if self.pose_recovery:
            self.pose_recovery.reset()
        self._feature_buffers.clear()
        self._prob_buffers.clear()
        self._kp_buffers.clear()
        self._bbox_buffers.clear()
        self._fall_events.clear()
        self._fall_cooldowns.clear()
        self.alarm_state.reset()
        self._fall_events = self.alarm_state.events
    
    def _clear_temporal_evidence(self, track_id):
        for buffers in (self._feature_buffers, self._prob_buffers, self._kp_buffers,
                        self._bbox_buffers, self._time_buffers, self.rule_engines,
                        self._model_support_until):
            buffers.pop(track_id, None)

    def _unreliable_observation(self, track_id, observation_time):
        """Missing and partial poses share the same bounded retention policy."""
        self.alarm_state.mark_missing(track_id)
        last = self._last_observed_at.get(track_id)
        gap = observation_time - last if last is not None else float('inf')
        if gap > self.preserve_short_gaps_seconds:
            self._clear_temporal_evidence(track_id)
        if gap > self.transition_max_gap_seconds:
            self.transitions.pop(track_id, None)
            self._last_observed_at.pop(track_id, None)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: Optional[datetime] = None,
        detections_override: Optional[List[PersonDetection]] = None,
    ) -> FrameResult:
        """
        处理单帧图像
        
        Args:
            frame: (H, W, 3) BGR/RGB 图像
            frame_idx: 帧索引
        
        Returns:
            FrameResult: 包含检测、概率、报警信息
        """
        h, w = frame.shape[:2]
        observation_time = timestamp.timestamp() if timestamp is not None else frame_idx / self.fps

        # Advance timed alarms even if a person is temporarily not detected.
        self.alarm_state.tick(frame_idx, now=timestamp)
        
        # 红外预处理
        if self.ir_mode:
            frame = self.detector.preprocess_for_ir(frame)
        elif self.auto_lowlight:
            # 亮度自适应低光增强（亮帧原样返回，无额外开销）
            frame = self.detector.preprocess_for_lowlight(frame)
        
        # Step 1: YOLO Pose 检测（可能跳过）
        if detections_override is not None:
            raw_detections = list(detections_override)
            detections = raw_detections
            is_detect_frame = True
        elif frame_idx % self.detection_interval == 0:
            raw_detections = self.detector.detect(frame, (h, w))
            if self.high_res_rescue:
                def pose_quality(items):
                    return max(
                        (float((d.keypoints[:, 2] > .3).mean()) * float(d.confidence)
                         for d in items),
                        default=0.0,
                    )
                # Small/fallen people are often absent at 640 px even though
                # the same frame is recoverable at the source resolution.
                # Only pay for the larger pass when the first result is empty
                # or its best pose is too weak to feed the temporal model.
                if not raw_detections or pose_quality(raw_detections) < .30:
                    rescued = self.detector.detect(
                        frame, (h, w), input_size=self.rescue_input_size
                    )
                    if pose_quality(rescued) > pose_quality(raw_detections):
                        raw_detections = rescued
            detections = raw_detections
            is_detect_frame = True
        else:
            raw_detections = []
            detections = []
            is_detect_frame = False

        if self.pose_recovery and is_detect_frame:
            raw_detections = self.pose_recovery.recover(
                frame, raw_detections, self.fps, self.detection_interval
            )
            detections = raw_detections
        
        # Step 2: ByteTrack 跟踪
        # Ultralytics' first prediction initialises its backend and can reset
        # torch's global CPU pool after load_model has returned.
        if self.cpu_threads and torch.get_num_threads() != self.cpu_threads:
            torch.set_num_threads(self.cpu_threads)
        tracked = self.tracker.update(
            detections, self.detection_interval, is_detection_frame=is_detect_frame
        )
        visual_tracked = (tracked if self.detection_interval == 1
                          else self.tracker.get_visual_tracks(max_lost=2))
        if self.pose_recovery and is_detect_frame:
            self.pose_recovery.observe(tracked, (h, w))
        
        # Step 3-6: 对每个跟踪的人体处理
        fall_probs = {}
        alarms = {}
        model_alarms = {}
        raw_fall_probs = {}
        frame_features = {}
        decision_details = {}
        
        for track_id, detection in tracked.items():
            previous_time = self._last_observed_at.get(track_id)
            observation_gap = observation_time - previous_time if previous_time is not None else 0.0
            # A blocked camera/inference worker may skip time without an
            # explicit empty detection. Such time cannot qualify the timer.
            if (self.max_candidate_gap_seconds > 0
                    and observation_gap > self.max_candidate_gap_seconds):
                self.alarm_state.mark_missing(track_id)
            if observation_gap > max(self.preserve_short_gaps_seconds, self.max_candidate_gap_seconds):
                if self.preserve_short_gaps_seconds > 0:
                    self._clear_temporal_evidence(track_id)
            # 关键点完整性门控：半入境/严重遮挡时骨架残缺，
            # 运动特征会畸变（此前实测人入画时 TCN 概率误冲 1.00），
            # 这类帧不追加进序列缓冲，避免污染 TCN 输入
            kp_conf = detection.keypoints[:, 2]
            incomplete_core = (self.preserve_short_gaps_seconds > 0
                               and min(kp_conf[[5, 6, 11, 12]]) < .3)
            if float((kp_conf > 0.3).mean()) < self.min_kp_completeness or incomplete_core:
                # A missing/partial skeleton must break a pending 0.5 s
                # candidate; otherwise a later frame could incorrectly count
                # the gap as continuous fall behaviour.
                alarms[track_id] = self.alarm_state.update(
                    track_id=track_id,
                    fall_signal=False,
                    confidence=0.0,
                    frame_idx=frame_idx,
                    now=timestamp,
                )
                model_alarms[track_id] = False
                # In live mode a blurred falling person can briefly produce
                # only a few joints. Keep the last *real* sequence so the
                # next complete observation can finish the same transition;
                # the partial frame itself is never appended to the TCN.
                # Offline analysis keeps the strict reset behaviour.
                self._unreliable_observation(track_id, observation_time)
                if track_id in self.transitions:
                    self.transitions[track_id].missing()
                decision_details[track_id] = {"reason": "insufficient_keypoints", "transition_candidate": False}
                continue

            transition = self.transitions.setdefault(
                track_id,
                FallTransition(max_observation_gap_seconds=self.transition_max_gap_seconds,
                               require_upright_reference=self.preserve_short_gaps_seconds > 0),
            )
            transition_decision = transition.update(detection, (h, w), observation_time)
            decision_details[track_id] = {
                "transition_candidate": transition_decision.candidate,
                "reason": transition_decision.reason,
                "shoulder_drop": round(transition_decision.drop, 4),
                "pose_height_ratio": round(transition_decision.height_ratio, 4),
            }

            # 获取或创建规则引擎
            if track_id not in self.rule_engines:
                self.rule_engines[track_id] = RuleRefinementOnline(
                    self.rule_config,
                    history_len=self.sequence_length,
                )
            
            # 更新缓冲区
            self._kp_buffers[track_id].append(detection.keypoints)
            self._bbox_buffers[track_id].append(detection.bbox)
            self._time_buffers[track_id].append(observation_time)
            self._last_observed_at[track_id] = observation_time
            
            # 保持缓冲区大小
            if len(self._kp_buffers[track_id]) > self.sequence_length:
                self._kp_buffers[track_id].pop(0)
                self._bbox_buffers[track_id].pop(0)
                self._time_buffers[track_id].pop(0)
            
            # 需要足够的帧才能运行 TCN
            if len(self._kp_buffers[track_id]) >= max(8, self.sequence_length // 4):
                # 提取运动特征
                kp_seq = np.stack(self._kp_buffers[track_id][-self.sequence_length:])
                bb_seq = np.stack(self._bbox_buffers[track_id][-self.sequence_length:])
                
                if self.preserve_short_gaps_seconds > 0:
                    # A retained pre-gap snapshot is genuine history, but its
                    # displacement must not look like one-frame velocity.
                    intervals = np.diff(self._time_buffers[track_id])
                    typical = float(np.median(intervals))
                    scales = np.ones(len(kp_seq), dtype=np.float32)
                    scales[1:] = np.minimum(1.0, typical / np.maximum(intervals, 1e-6))
                    scales[0] = scales[1]
                    motion_feat = self.fall_detector.motion_encoder.compute_sequence(
                        kp_seq, bb_seq, step_scales=scales
                    )
                else:
                    motion_feat = self.fall_detector.motion_encoder.compute_sequence(kp_seq, bb_seq)
                # (T, C)
                
                # TCN 推理
                feat_tensor = torch.from_numpy(motion_feat).float().unsqueeze(0)  # (1, T, C)
                with torch.no_grad():
                    tcn_prob = self.fall_detector.tcn(feat_tensor)  # (1, T, 1)
                tcn_prob = tcn_prob.squeeze().numpy()  # (T,)
                
                last_prob = float(tcn_prob[-self.prob_smooth:].mean())  # 时序平滑：抑制单帧概率抖动
                raw_fall_probs[track_id] = last_prob
                last_feat = motion_feat[-1]
                
                # Rule Refinement 在线处理
                refined_prob, is_alarm = self.rule_engines[track_id].update(
                    last_prob, last_feat
                )
                # The online rule engine keeps a confirmed alarm alive for a
                # while.  Feed the instantaneous candidate to the product
                # alarm debouncer so "continuous for 0.5 s" remains literal.
                candidate_alarm = self.rule_engines[track_id].is_candidate_active()
                if self.require_observed_transition:
                    # A transient motion peak is followed by a sustained low
                    # pose. The model must also support the observation; an
                    # initially resting person has no descent and is rejected.
                    # A live pose model and the temporal classifier do not
                    # finish on the same frame: blur and a low-resolution
                    # pose can make the TCN score dip immediately after the
                    # observed descent.  Once a real transition has received
                    # temporal support, retain that support briefly while the
                    # same real low pose remains visible.  This is hysteresis,
                    # not a lower threshold and it never invents a pose.
                    if not transition_decision.candidate:
                        self._model_support_until.pop(track_id, None)
                    if transition_decision.candidate and last_prob >= self.rule_config.tcn_prob_threshold:
                        self._model_support_until[track_id] = (
                            observation_time + self.model_support_grace_seconds
                        )
                    model_supported = (
                        last_prob >= self.rule_config.tcn_prob_threshold
                        or observation_time <= self._model_support_until.get(track_id, -1.0)
                    )
                    candidate_alarm = transition_decision.candidate and model_supported
                    # Do not present the old rules' weighted score as the
                    # probability used by a different decision policy.
                    refined_prob = last_prob
                decision_details[track_id]["model_supported"] = bool(
                    last_prob >= self.rule_config.tcn_prob_threshold
                    or observation_time <= self._model_support_until.get(track_id, -1.0)
                )

                fall_probs[track_id] = refined_prob
                model_alarms[track_id] = candidate_alarm
                alarms[track_id] = self.alarm_state.update(
                    track_id=track_id,
                    fall_signal=candidate_alarm,
                    confidence=refined_prob,
                    frame_idx=frame_idx,
                    now=timestamp,
                )
                frame_features[track_id] = last_feat
            else:
                fall_probs[track_id] = 0.0
                alarms[track_id] = self.alarm_state.update(
                    track_id=track_id,
                    fall_signal=False,
                    confidence=0.0,
                    frame_idx=frame_idx,
                    now=timestamp,
                )
                model_alarms[track_id] = False
        
        # 更新冷却计数器
        for track_id in list(self._fall_cooldowns.keys()):
            self._fall_cooldowns[track_id] -= 1
            if self._fall_cooldowns[track_id] <= 0:
                del self._fall_cooldowns[track_id]

        # A pending 0.5 s candidate must be continuous.  Do not let a track
        # that temporarily disappears from the detector count the gap.
        current_track_ids = set(tracked.keys())
        for track_id in list(self._last_observed_at):
            if track_id not in current_track_ids:
                self._unreliable_observation(track_id, observation_time)
                if track_id in self.transitions:
                    self.transitions[track_id].missing()

        # A person can disappear for a few detector frames while the
        # application alarm is still active.  Keep the global alarm visible
        # to the UI until the state machine clears it.
        for track_id in self.alarm_state.active_track_ids:
            alarms.setdefault(track_id, True)
            model_alarms.setdefault(track_id, False)

        # The event queue contains only create/clear changes, so the backend
        # can persist them without polling every frame.
        event_updates = self.alarm_state.consume_updates()

        return FrameResult(
            frame_idx=frame_idx,
            persons=list(tracked.values()),
            fall_probs=fall_probs,
            alarms=alarms,
            features=frame_features if frame_features else None,
            is_detect_frame=is_detect_frame,
            raw_detections=raw_detections,
            event_updates=event_updates,
            model_alarms=model_alarms,
            raw_fall_probs=raw_fall_probs,
            decision_details=decision_details,
            visual_persons=visual_tracked,
        )

    def clear_event(self, event_id: str, reason: str = "manual") -> Optional[dict]:
        """人工解除一个报警，供 Python API 或 Spring Boot 调用。"""

        return self.alarm_state.clear_event(event_id, reason=reason)
    
    def process_video(
        self,
        video_path: str,
        progress_callback=None,
    ) -> PipelineResult:
        """
        处理完整视频
        
        Args:
            video_path: 视频文件路径
            progress_callback: 进度回调 (current_frame, total_frames) -> None
        
        Returns:
            PipelineResult: 包含所有帧结果和跌倒事件
        """
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        self.fps = fps
        self.alarm_state.set_fps(fps)
        
        frame_results = []
        start_time = time.time()
        frame_idx = 0
        
        # 视频写入器
        writer = None
        if self.save_video:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            basename = os.path.basename(video_path)
            out_path = os.path.join(self.output_dir, basename)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            print(f"[Pipeline] 输出视频: {out_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.process_frame(frame, frame_idx)
            frame_results.append(result)
            
            # 保存视频
            if writer is not None:
                annotated = self._draw_result(frame, result)
                writer.write(annotated)
            
            frame_idx += 1
            
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)
        
        cap.release()
        if writer is not None:
            writer.release()
        
        total_time = time.time() - start_time
        avg_fps = total_frames / total_time if total_time > 0 else 0
        avg_latency = (total_time / total_frames * 1000) if total_frames > 0 else 0
        
        result = PipelineResult(
            frame_results=frame_results,
            fall_events=list(self._fall_events),  # 拷贝快照，避免下次 reset 清空
            total_frames=total_frames,
            total_time=total_time,
            fps=avg_fps,
            avg_latency_ms=avg_latency,
        )
        
        # 保存结果
        self._save_results(result, video_path)
        
        return result
    
    def process_webcam(
        self,
        camera_id: int = 0,
    ) -> Generator[FrameResult, None, None]:
        """
        实时摄像头推理（生成器模式）
        
        Yields:
            FrameResult: 每帧推理结果
        """
        self.reset()
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"无法打开摄像头: {camera_id}")
        
        camera_fps = cap.get(cv2.CAP_PROP_FPS)
        if camera_fps and camera_fps > 0:
            self.fps = camera_fps
            self.alarm_state.set_fps(camera_fps)
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = self.process_frame(frame, frame_idx)
                frame_idx += 1
                yield result
                
                # 按 'q' 退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def evaluate_dataset(
        self,
        video_paths: List[str],
        annotations: Optional[Dict] = None,
    ) -> Dict:
        """
        批量数据集评测
        
        Returns:
            dict: 包含 mAP, recall, precision, F1 等指标
        """
        all_events = []
        all_results = []
        
        for video_path in video_paths:
            print(f"[Pipeline] 评测: {video_path}")
            result = self.process_video(video_path)
            all_results.append(result)
            all_events.extend(result.fall_events)
        
        # 汇总指标
        total_fall_events = len(all_events)
        total_frames = sum(r.total_frames for r in all_results)
        total_time = sum(r.total_time for r in all_results)
        
        metrics = {
            "total_videos": len(video_paths),
            "total_frames": total_frames,
            "total_time_seconds": total_time,
            "total_fall_events": total_fall_events,
            "avg_fps": total_frames / total_time if total_time > 0 else 0,
            "avg_latency_ms": (total_time / total_frames * 1000) if total_frames > 0 else 0,
        }
        
        if annotations:
            # TODO: 计算准确率/召回率
            pass
        
        return metrics
    
    def _draw_result(
        self,
        frame: np.ndarray,
        result: FrameResult,
    ) -> np.ndarray:
        """在图像上绘制检测结果：骨架、边框、Fall 状态

        逐帧检测时使用当前帧的真实人体结果；隔帧模式可显示最近的跟踪结果。
        """
        h, w = frame.shape[:2]
        vis = frame.copy()

        # ── 判断跌倒状态 ──
        is_any_alarm = any(result.alarms.values())

        # ── 左上角：Fall 状态面板（每帧都画） ──
        panel_x, panel_y = 15, 15
        panel_w, panel_h = 260, 70
        overlay = vis.copy()
        if is_any_alarm:
            cv2.rectangle(overlay, (panel_x, panel_y),
                         (panel_x + panel_w, panel_y + panel_h), (0, 0, 60), -1)
        else:
            cv2.rectangle(overlay, (panel_x, panel_y),
                         (panel_x + panel_w, panel_y + panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)
        border_color = (0, 0, 255) if is_any_alarm else (100, 100, 100)
        cv2.rectangle(vis, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h), border_color, 2)
        status_text = "Alarm: YES" if is_any_alarm else "Alarm: NO"
        status_color = (0, 0, 255) if is_any_alarm else (0, 255, 0)
        cv2.putText(vis, status_text, (panel_x + 12, panel_y + 38),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, status_color, 2, cv2.LINE_AA)
        if result.fall_probs:
            max_prob = max(result.fall_probs.values())
            cv2.putText(vis, f"Model score: {max_prob:.3f}", (panel_x + 12, panel_y + 62),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # ── 每帧画最近的跟踪框/骨架 ──
        # YOLO intentionally runs every Nth frame. Drawing only raw YOLO
        # detections made the overlay blink on skipped frames. The tracker
        # keeps the latest pose for display; stale tracks are limited to a
        # couple of frames and are not used by the alarm logic.
        draw_persons = result.visual_persons or {
            index: detection for index, detection in enumerate(result.raw_detections)
        }
        if draw_persons:
            from utils.visualization import SKELETON_CONNECTIONS, SKELETON_COLORS

            for track_id, det in draw_persons.items():
                person_alarm = bool(result.alarms.get(track_id, False))
                color_bgr = (0, 0, 255) if person_alarm else (0, 255, 0)

                # 边界框
                bbox = det.bbox * np.array([w, h, w, h])
                bbox = bbox.astype(int)
                cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color_bgr, 2)

                # 骨架连线
                kp_pixel = det.keypoints[:, :2] * np.array([w, h])
                kp_pixel = kp_pixel.astype(int)
                kp_conf = det.keypoints[:, 2]
                for conn_idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
                    if kp_conf[i] > 0.3 and kp_conf[j] > 0.3:
                        line_clr = SKELETON_COLORS[conn_idx % len(SKELETON_COLORS)]
                        cv2.line(vis, tuple(kp_pixel[i]), tuple(kp_pixel[j]),
                                line_clr, 2, cv2.LINE_AA)

                # 关键点圆点
                for i in range(17):
                    if kp_conf[i] > 0.3:
                        px, py = kp_pixel[i]
                        cv2.circle(vis, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(vis, (px, py), 5, color_bgr, 2, cv2.LINE_AA)

                # 人体框上方显示概率
                prob_val = result.fall_probs.get(track_id, 0.0)
                label = f"{'ALARM' if person_alarm else 'Monitoring'} {prob_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (bbox[0], bbox[1] - th - 8),
                             (bbox[0] + tw + 6, bbox[1]), color_bgr, -1)
                cv2.putText(vis, label, (bbox[0] + 3, bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # ── 右下角：帧号 ──
        frame_text = f"Frame: {result.frame_idx}"
        (fw, fh), _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (w - fw - 25, h - fh - 20),
                     (w - 10, h - 8), (0, 0, 0), -1)
        cv2.putText(vis, frame_text, (w - fw - 20, h - 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return vis
    
    def _save_results(
        self,
        result: PipelineResult,
        video_path: str,
    ):
        """保存推理结果到 JSON"""
        basename = os.path.splitext(os.path.basename(video_path))[0]
        result_path = os.path.join(self.output_dir, f"{basename}_results.json")
        
        data = {
            "video": video_path,
            "total_frames": result.total_frames,
            "total_time": result.total_time,
            "fps": result.fps,
            "avg_latency_ms": result.avg_latency_ms,
            "fall_events": [e.to_dict() for e in result.fall_events],
            "frame_alarms": [
                {
                    "frame": r.frame_idx,
                    "alarms": {str(k): v for k, v in r.alarms.items()},
                    "probs": {str(k): v for k, v in r.fall_probs.items()},
                    "event_updates": r.event_updates,
                }
                for r in result.frame_results
                if any(r.alarms.values())
            ],
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[Pipeline] 结果保存到: {result_path}")
    
    def get_statistics(self) -> Dict:
        """获取管线统计信息"""
        # 参数量
        detector_params = sum(p.numel() for p in self.fall_detector.motion_encoder.parameters())
        tcn_params = sum(p.numel() for p in self.fall_detector.tcn.parameters())
        
        return {
            "model_params": {
                "motion_encoder": detector_params,
                "tcn": tcn_params,
                "total": detector_params + tcn_params,
            },
            "model_size_mb": self.fall_detector.get_model_size_mb(),
            "detection_interval": self.detection_interval,
            "sequence_length": self.sequence_length,
            "input_size": self.input_size,
            "ir_mode": self.ir_mode,
        }
