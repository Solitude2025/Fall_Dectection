"""
Rule Refinement (时空一致性校正)
---------------------------------
基于物理约束和时序一致性的规则后处理模块。

这是本方案的核心创新模块之二：
- 对 TCN 输出的概率进行物理规则校验
- 几乎零计算量，但能大幅降低误报（坐下、弯腰、躺卧等）
- 工业界最常用的降低误报方法

规则包括：
1. 角度持续时间判断：躯干角度 < 阈值，且持续 N 帧
2. 速度峰值判断：人体中心速度超过阈值
3. 人体静止持续时间：跌倒后人体应静止一段时间
4. 连续多帧投票：滑动窗口投票机制
5. 姿态恢复检测：排除快速恢复的"假跌倒"
6. 边界框形态约束：跌倒后 bbox 宽高比应变化

设计理念：
    TCN 输出 "可能是跌倒" → Rule Refinement 二次确认 → 最终报警
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class RuleConfig:
    """规则校正的超参数配置"""
    
    # 躯干角度阈值
    torso_angle_threshold: float = 0.5  # 弧度，约 30°
    torso_angle_duration: int = 8        # 需要持续的帧数
    
    # 速度峰值阈值
    velocity_peak_threshold: float = 0.03  # 归一化速度（跟踪器修复后真实运动速度恢复：
                                           # 跌倒 0.09-0.16，缓慢坐地 0.055，0.08 可分离）
    
    # 静止检测
    stillness_threshold: float = 0.005     # 人体中心移动阈值
    stillness_duration: int = 15           # 跌倒后需静止帧数
    stillness_check_window: int = 30       # 检查窗口大小
    
    # 多帧投票
    vote_window: int = 10                  # 滑动窗口大小
    vote_threshold: float = 0.6            # 投票阈值
    
    # 姿态恢复
    recovery_duration: int = 5             # 判定为"恢复"的最短时间
    recovery_angle_threshold: float = 1.0  # 恢复后的角度阈值（约60°）
    
    # TCN 概率阈值
    tcn_prob_threshold: float = 0.5        # TCN 原始概率阈值
    
    # 边界框约束
    aspect_ratio_change_threshold: float = 0.3  # 宽高比变化阈值
    
    # 跌倒状态记忆
    fall_memory_frames: int = 60           # 跌倒后保持报警的帧数
    rules_min_pass: int = 1                # 物理规则至少通过几条 (1~3)


class RuleRefinement(nn.Module):
    """
    时空一致性规则校正模块
    
    无训练参数，纯推理规则引擎。
    在 NPU 上可作为轻量后处理算子运行。
    """
    
    def __init__(self, config: Optional[RuleConfig] = None):
        super().__init__()
        self.config = config or RuleConfig()
        self.cfg = self.config
        
        # 内部状态（推理时维护）
        self.reset_state()
    
    def reset_state(self):
        """重置内部状态"""
        # 跌倒事件缓冲区
        self._fall_event_buffer = []
        # 帧计数器
        self._frame_count = 0
        # 持续报警计数器
        self._alarm_persist = 0
        # 特征历史缓冲区
        self._feature_history = []
        # 概率历史
        self._prob_history = []
        
    def forward(
        self,
        tcn_prob: torch.Tensor,           # (B, T, 1)
        motion_features: torch.Tensor,    # (B, T, C)
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 TCN 输出的概率进行规则校正
        
        Args:
            tcn_prob: (B, T, 1) TCN 输出的原始概率
            motion_features: (B, T, C) 运动特征
            debug: 是否输出调试信息
        
        Returns:
            refined_prob: (B, T, 1) 校正后的概率
            alarm_flag: (B, T, 1) 报警标志 (0/1)
        """
        B, T, _ = tcn_prob.shape
        device = tcn_prob.device
        cfg = self.cfg
        
        refined = tcn_prob.clone()
        alarm = torch.zeros_like(tcn_prob)
        
        for b in range(B):
            for t in range(T):
                prob = tcn_prob[b, t, 0].item()
                features = motion_features[b, t]
                
                # 如果 TCN 概率低于阈值，直接跳过
                if prob < cfg.tcn_prob_threshold:
                    refined[b, t, 0] = prob
                    continue
                
                # ---- 规则1: 躯干角度检查 ----
                torso_angle = features[self.TORSO_ANGLE_IDX].item()
                angle_ok = torso_angle > cfg.torso_angle_threshold
                
                # ---- 规则2: 速度峰值检查 ----
                center_speed = features[self.CENTER_SPEED_IDX].item()
                speed_ok = center_speed > cfg.velocity_peak_threshold
                
                # ---- 规则3: 边界框宽高比变化 ----
                aspect_change = features[self.ASPECT_CHANGE_IDX].item()
                aspect_ok = abs(aspect_change) > cfg.aspect_ratio_change_threshold
                
                # ---- 规则4: 躯干角度变化率 ----
                torso_change = features[self.TORSO_CHANGE_IDX].item()
                torso_change_ok = torso_change > 0.01  # 角度在增大（身体在倒下）
                
                # 综合评分
                rule_score = 0.0
                rule_weight = 0.0
                
                if angle_ok:
                    rule_score += 0.3
                rule_weight += 0.3
                
                if speed_ok:
                    rule_score += 0.25
                rule_weight += 0.25
                
                if aspect_ok:
                    rule_score += 0.25
                rule_weight += 0.25
                
                if torso_change_ok:
                    rule_score += 0.2
                rule_weight += 0.2
                
                # 规则得分归一化
                if rule_weight > 0:
                    rule_score /= rule_weight
                else:
                    rule_score = 0.0
                
                # 融合 TCN 概率和规则得分
                alpha = 0.6  # TCN 权重
                refined_prob = alpha * prob + (1 - alpha) * rule_score
                
                refined[b, t, 0] = refined_prob
                
                # ---- 最终报警判定 ----
                # 1. 概率超过阈值
                # 2. 通过的物理规则条数达到配置要求
                rules_passed = sum([angle_ok, speed_ok, aspect_ok])
                if refined_prob > cfg.tcn_prob_threshold and rules_passed >= cfg.rules_min_pass:
                    alarm[b, t, 0] = 1.0
        
        return refined, alarm
    
    # 特征索引常量（与 MotionFeatureEncoder.get_feature_names() 对应）
    ASPECT_RATIO_IDX = 0
    AREA_CHANGE_IDX = 1
    CENTER_VX_IDX = 2
    CENTER_VY_IDX = 3
    CENTER_AX_IDX = 4
    CENTER_AY_IDX = 5
    TORSO_ANGLE_IDX = 6
    TORSO_LEN_IDX = 7
    # kp velocities: 8 ~ 21 (7 points × 2)
    # conf stats: 22 ~ 24
    # distribution: 25 ~ 32
    # knee angles: 33 ~ 36
    BBOX_W_CHANGE_IDX = 37
    BBOX_H_CHANGE_IDX = 38
    CENTER_SPEED_IDX = 39
    CENTER_ACC_IDX = 40
    TORSO_CHANGE_IDX = 41
    
    @property
    def ASPECT_CHANGE_IDX(self):
        return self.BBOX_W_CHANGE_IDX


class RuleRefinementOnline:
    """
    在线推理用的规则校正模块（逐帧处理）
    
    维护内部状态缓冲区，适合视频流逐帧推理场景。
    """
    
    def __init__(self, config: Optional[RuleConfig] = None, history_len: int = 60):
        self.config = config or RuleConfig()
        self.cfg = self.config
        self.history_len = history_len
        
        # 环形缓冲区
        self._prob_buffer = []
        self._feature_buffer = []
        self._angle_buffer = []
        self._speed_buffer = []
        self._aspect_buffer = []
        
        # 状态
        self._fall_active = False
        self._fall_start_frame = -1
        self._alarm_cooldown = 0
        self._frame_idx = 0
        # ``_alarm_cooldown`` intentionally keeps the legacy model alarm
        # visible after a confirmed fall.  The application layer needs the
        # instantaneous candidate separately so it can enforce its own 0.5 s
        # continuous-signal trigger.
        self._candidate_active = False
    
    def reset(self):
        """重置所有状态"""
        self._prob_buffer.clear()
        self._feature_buffer.clear()
        self._angle_buffer.clear()
        self._speed_buffer.clear()
        self._aspect_buffer.clear()
        self._fall_active = False
        self._fall_start_frame = -1
        self._alarm_cooldown = 0
        self._frame_idx = 0
        self._candidate_active = False
    
    def update(
        self,
        tcn_prob: float,
        motion_features: np.ndarray,  # (feature_dim,)
    ) -> Tuple[float, bool]:
        """
        逐帧更新并返回校正后的概率和报警状态
        
        Args:
            tcn_prob: TCN 输出的单帧概率
            motion_features: 单帧运动特征向量
        
        Returns:
            refined_prob: 校正后的概率
            is_alarm: 是否报警
        """
        cfg = self.cfg
        self._frame_idx += 1
        
        # 更新缓冲区
        self._prob_buffer.append(tcn_prob)
        self._feature_buffer.append(motion_features)
        self._angle_buffer.append(motion_features[RuleRefinement.TORSO_ANGLE_IDX])
        self._speed_buffer.append(motion_features[RuleRefinement.CENTER_SPEED_IDX])
        self._aspect_buffer.append(motion_features[RuleRefinement.BBOX_W_CHANGE_IDX])
        
        if len(self._prob_buffer) > self.history_len:
            self._prob_buffer.pop(0)
            self._feature_buffer.pop(0)
            self._angle_buffer.pop(0)
            self._speed_buffer.pop(0)
            self._aspect_buffer.pop(0)
        
        # 如果 TCN 概率很低，直接返回
        if tcn_prob < cfg.tcn_prob_threshold:
            self._candidate_active = False
            # 检查警报冷却
            if self._alarm_cooldown > 0:
                self._alarm_cooldown -= 1
                return tcn_prob, True
            
            # 检查是否从跌倒中恢复
            if self._fall_active:
                if self._check_recovery():
                    self._fall_active = False
                    self._fall_start_frame = -1
            
            return tcn_prob, False
        
        # ---- 规则检查 ----
        angle_ok = self._angle_buffer[-1] > cfg.torso_angle_threshold
        speed_ok = self._speed_buffer[-1] > cfg.velocity_peak_threshold
        aspect_ok = abs(self._aspect_buffer[-1]) > cfg.aspect_ratio_change_threshold
        
        # 时序持续检查
        if len(self._angle_buffer) >= cfg.torso_angle_duration:
            recent_angles = self._angle_buffer[-cfg.torso_angle_duration:]
            angle_sustained = all(a > cfg.torso_angle_threshold for a in recent_angles)
        else:
            angle_sustained = angle_ok
        
        # 规则评分
        rule_score = 0.0
        if angle_sustained: rule_score += 0.4
        if speed_ok: rule_score += 0.3
        if aspect_ok: rule_score += 0.3
        
        # 融合
        alpha = 0.6
        refined = alpha * tcn_prob + (1 - alpha) * rule_score
        
        # ---- 多帧投票 ----
        if len(self._prob_buffer) >= cfg.vote_window:
            recent_probs = self._prob_buffer[-cfg.vote_window:]
            vote_passed = sum(1 for p in recent_probs if p > cfg.tcn_prob_threshold)
            vote_ratio = vote_passed / cfg.vote_window
        else:
            vote_ratio = 1.0 if tcn_prob > cfg.tcn_prob_threshold else 0.0
        
        # ---- 静止检查（对已检测到的跌倒） ----
        stillness_ok = True
        if self._fall_active:
            if len(self._speed_buffer) >= cfg.stillness_duration:
                recent_speeds = self._speed_buffer[-cfg.stillness_duration:]
                stillness_ok = all(s < cfg.stillness_threshold for s in recent_speeds)
        
        # ---- 最终判定 ----
        rules_passed = sum([angle_sustained, speed_ok, aspect_ok])
        
        is_fall = (
            refined > cfg.tcn_prob_threshold
            and vote_ratio > cfg.vote_threshold
            and rules_passed >= cfg.rules_min_pass
        )
        self._candidate_active = bool(is_fall)
        
        if is_fall:
            if not self._fall_active:
                self._fall_active = True
                self._fall_start_frame = self._frame_idx
                self._alarm_cooldown = cfg.fall_memory_frames
        
        # 检查警报冷却
        if self._alarm_cooldown > 0 and self._fall_active:
            self._alarm_cooldown -= 1
            return refined, True
        
        return refined, False

    def is_candidate_active(self) -> bool:
        """Return the current frame's fall candidate before memory hold."""

        return self._candidate_active
    
    def _check_recovery(self) -> bool:
        """检查是否从跌倒中恢复"""
        if len(self._angle_buffer) < self.config.recovery_duration:
            return False
        
        recent_angles = self._angle_buffer[-self.config.recovery_duration:]
        return all(a < self.config.recovery_angle_threshold for a in recent_angles)
    
    def get_fall_duration(self) -> int:
        """获取当前跌倒事件持续的帧数"""
        if not self._fall_active:
            return 0
        return self._frame_idx - self._fall_start_frame
    
    def is_fall_active(self) -> bool:
        return self._fall_active


def create_rule_refinement(config: Optional[RuleConfig] = None) -> RuleRefinement:
    """创建规则校正模块"""
    return RuleRefinement(config)


def create_online_refinement(
    config: Optional[RuleConfig] = None,
    history_len: int = 60,
) -> RuleRefinementOnline:
    """创建在线推理用的规则校正模块"""
    return RuleRefinementOnline(config, history_len)
