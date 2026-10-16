"""
Motion Feature Encoder (运动特征编码器)
-----------------------------------------
将 YOLOv8n-Pose 输出的 17 个关键点 + 人体边界框，
编码为紧凑的运动语义特征向量（~48 维），替代传统 CNN Feature。

这是本方案的核心创新模块之一：
- 提取人体运动学特征而非原始 RGB 特征
- 大幅降低 TCN 输入维度，从 640/1024 维 → ~48 维
- 使 TCN 计算量暴跌，适配端侧 NPU 部署

每个时间步提取的特征包括：
1. 人体框宽高比（H/W）：跌倒时迅速变化
2. 人体中心速度 (cx, cy)：计算速度和加速度
3. 躯干角度：肩-臀连线与垂直线夹角（站立≈90°，跌倒≈20°）
4. 关键点速度：头部/肩部/臀部/膝部各自的 Δx, Δy
5. 人体面积变化：跌倒后人体框变宽
6. 重心变化：所有关键点均值位置的速度
7. 手部支撑特征：手腕速度、手腕贴地度、支撑状态分
   （FOOSH 生物力学先验：保护性跌倒中手先于身体落地并保持支撑，
   作为软特征供 TCN 学习时序先后关系，不作为硬规则）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


# COCO 17关键点索引
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}


class MotionFeatureEncoder(nn.Module):
    """
    运动特征编码器
    
    输入: (B, T, 17, 3) - [batch, timesteps, keypoints, (x, y, conf)]
          (B, T, 4)     - [batch, timesteps, (x1, y1, x2, y2)] bbox
    输出: (B, T, 48)    - 48维运动特征向量
    
    模型无参数，纯计算，可完美部署到 NPU
    """
    
    def __init__(
        self,
        feature_dim: int = 48,
        frame_window: int = 3,
        velocity_channels: List[str] = None,
    ):
        """
        Args:
            feature_dim: 输出特征维度
            frame_window: 计算速度/加速度的窗口大小
            velocity_channels: 需要计算速度的关键点名称列表
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.frame_window = frame_window
        
        if velocity_channels is None:
            self.velocity_channels = [
                "nose", "left_shoulder", "right_shoulder",
                "left_hip", "right_hip",
                "left_knee", "right_knee",
            ]
        else:
            self.velocity_channels = velocity_channels
        
        # 注册为缓冲区（非参数，但随模型移动设备）
        self.register_buffer("_dummy", torch.zeros(1))
        
    def forward(
        self,
        keypoints: torch.Tensor,   # (B, T, 17, 3)
        bboxes: torch.Tensor,       # (B, T, 4)
        step_scales: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            keypoints: 关键点 (x, y, conf)，归一化到 [0, 1]
            bboxes: 边界框 (x1, y1, x2, y2)，归一化到 [0, 1]
        
        Returns:
            features: (B, T, feature_dim) 运动特征
        """
        B, T = keypoints.shape[0], keypoints.shape[1]
        device = keypoints.device
        # Live observations can straddle a bounded detector gap. Correct
        # differences for elapsed time instead of inventing intermediate poses.
        # Training/offline calls retain their original unit time steps.
        scale = step_scales if step_scales is not None else torch.ones(B, T, device=device)
        
        features_list = []
        
        # ---- 1. 人体框宽高比 H/W (1维) ----
        bbox_w = bboxes[..., 2] - bboxes[..., 0] + 1e-6
        bbox_h = bboxes[..., 3] - bboxes[..., 1] + 1e-6
        aspect_ratio = bbox_h / bbox_w  # (B, T)
        feature_aspect = aspect_ratio.unsqueeze(-1)  # (B, T, 1)
        features_list.append(feature_aspect)
        
        # ---- 2. 人体框面积变化 (1维) ----
        bbox_area = bbox_w * bbox_h  # (B, T)
        # 归一化面积变化率
        area_mean = bbox_area.mean(dim=1, keepdim=True) + 1e-6
        area_change = (bbox_area - area_mean) / area_mean
        feature_area = area_change.unsqueeze(-1)  # (B, T, 1)
        features_list.append(feature_area)
        
        # ---- 3. 人体中心速度 + 加速度 (4维) ----
        cx = (bboxes[..., 0] + bboxes[..., 2]) / 2  # (B, T)
        cy = (bboxes[..., 1] + bboxes[..., 3]) / 2  # (B, T)
        
        # 速度 (B, T-1)
        vx = cx[:, 1:] - cx[:, :-1]
        vy = cy[:, 1:] - cy[:, :-1]
        
        # 填充使维度对齐 (B, T)
        vx = torch.cat([vx[:, :1], vx], dim=1)
        vy = torch.cat([vy[:, :1], vy], dim=1)
        vx = vx * scale
        vy = vy * scale
        
        # 加速度 (B, T-2)
        ax = vx[:, 2:] - vx[:, 1:-1]
        ay = vy[:, 2:] - vy[:, 1:-1]
        ax = torch.cat([ax[:, :1], ax[:, :1], ax], dim=1)
        ay = torch.cat([ay[:, :1], ay[:, :1], ay], dim=1)
        ax = ax * scale
        ay = ay * scale
        
        feature_center = torch.stack([vx, vy, ax, ay], dim=-1)  # (B, T, 4)
        features_list.append(feature_center)
        
        # ---- 4. 躯干倾斜角度 (2维) ----
        # 躯干中线: 肩部中心 -> 臀部中心
        shoulder_center = (
            keypoints[:, :, KP["left_shoulder"]]
            + keypoints[:, :, KP["right_shoulder"]]
        ) / 2  # (B, T, 3)
        hip_center = (
            keypoints[:, :, KP["left_hip"]]
            + keypoints[:, :, KP["right_hip"]]
        ) / 2  # (B, T, 3)
        
        # 躯干向量
        torso_vec = shoulder_center - hip_center  # (B, T, 3)
        torso_x = torso_vec[..., 0]
        torso_y = torso_vec[..., 1]
        
        # 躯干与垂直线夹角 (用 arctan)
        torso_angle = torch.atan2(
            torso_x.abs(), torso_y.abs() + 1e-6
        )  # (B, T), 0=直立, pi/2=水平
        
        # 躯干长度变化（归一化）
        torso_len = torch.sqrt(torso_x**2 + torso_y**2 + 1e-6)
        torso_len_norm = torso_len / (torso_len.mean(dim=1, keepdim=True) + 1e-6)
        
        feature_torso = torch.stack([torso_angle, torso_len_norm], dim=-1)  # (B, T, 2)
        features_list.append(feature_torso)
        
        # ---- 5. 关键点速度 (7个点 × 2维 = 14维) ----
        kp_velocities = []
        for kp_name in self.velocity_channels:
            kp_idx = KP[kp_name]
            kp_data = keypoints[:, :, kp_idx]  # (B, T, 3) -> (x, y, conf)
            
            # x, y 速度
            dvx = kp_data[:, 1:, 0] - kp_data[:, :-1, 0]
            dvy = kp_data[:, 1:, 1] - kp_data[:, :-1, 1]
            
            dvx = torch.cat([dvx[:, :1], dvx], dim=1)
            dvy = torch.cat([dvy[:, :1], dvy], dim=1)
            
            # 用置信度加权
            conf_weight = kp_data[..., 2].unsqueeze(-1)  # (B, T, 1)
            
            kp_velocities.append((dvx * scale).unsqueeze(-1))
            kp_velocities.append((dvy * scale).unsqueeze(-1))
        
        feature_kp_vel = torch.cat(kp_velocities, dim=-1)  # (B, T, 14)
        features_list.append(feature_kp_vel)
        
        # ---- 6. 关键点置信度统计 (3维) ----
        kp_conf = keypoints[..., 2]  # (B, T, 17)
        conf_mean = kp_conf.mean(dim=-1, keepdim=True)      # (B, T, 1)
        conf_min = kp_conf.min(dim=-1, keepdim=True)[0]     # (B, T, 1)
        conf_std = kp_conf.std(dim=-1, keepdim=True)         # (B, T, 1)
        feature_conf = torch.cat([conf_mean, conf_min, conf_std], dim=-1)  # (B, T, 3)
        features_list.append(feature_conf)
        
        # ---- 7. 关键点分布特征 (8维) ----
        # 上半身/下半身关键点分布
        upper_kp = keypoints[:, :, :11]   # 鼻到腕
        lower_kp = keypoints[:, :, 11:]   # 臀部到脚踝
        
        upper_y_mean = upper_kp[..., 1].mean(dim=-1, keepdim=True)  # (B, T, 1)
        lower_y_mean = lower_kp[..., 1].mean(dim=-1, keepdim=True)  # (B, T, 1)
        upper_y_std = upper_kp[..., 1].std(dim=-1, keepdim=True)
        lower_y_std = lower_kp[..., 1].std(dim=-1, keepdim=True)
        
        upper_x_mean = upper_kp[..., 0].mean(dim=-1, keepdim=True)
        lower_x_mean = lower_kp[..., 0].mean(dim=-1, keepdim=True)
        upper_x_std = upper_kp[..., 0].std(dim=-1, keepdim=True)
        lower_x_std = lower_kp[..., 0].std(dim=-1, keepdim=True)
        
        feature_dist = torch.cat([
            upper_y_mean, lower_y_mean, upper_y_std, lower_y_std,
            upper_x_mean, lower_x_mean, upper_x_std, lower_x_std,
        ], dim=-1)  # (B, T, 8)
        features_list.append(feature_dist)
        
        # ---- 8. 膝关节角度 (4维) ----
        # 左膝角度: 左髋-左膝 与 左膝-左踝 的夹角
        for side, hip_k, knee_k, ankle_k in [
            ("left", KP["left_hip"], KP["left_knee"], KP["left_ankle"]),
            ("right", KP["right_hip"], KP["right_knee"], KP["right_ankle"]),
        ]:
            hip = keypoints[:, :, hip_k, :2]     # (B, T, 2)
            knee = keypoints[:, :, knee_k, :2]
            ankle = keypoints[:, :, ankle_k, :2]
            
            vec1 = hip - knee    # 髋→膝
            vec2 = ankle - knee  # 踝→膝
            
            # 夹角余弦
            dot = (vec1 * vec2).sum(dim=-1)
            norm1 = torch.sqrt((vec1**2).sum(dim=-1) + 1e-6)
            norm2 = torch.sqrt((vec2**2).sum(dim=-1) + 1e-6)
            cos_angle = dot / (norm1 * norm2 + 1e-6)
            angle = torch.acos(cos_angle.clamp(-1, 1))
            
            features_list.append(angle.unsqueeze(-1))  # (B, T, 1)
            features_list.append(cos_angle.unsqueeze(-1))  # (B, T, 1)
        
        # 总计: 1 + 1 + 4 + 2 + 14 + 3 + 8 + 4 = 37 -> 需要pad到 48
        # ---- 9. 额外统计特征 (11维，补齐到 48) ----
        # 人体框宽度变化率
        bbox_w_change = (bbox_w[:, 1:] - bbox_w[:, :-1]) / (bbox_w[:, :-1] + 1e-6)
        bbox_w_change = torch.cat([bbox_w_change[:, :1], bbox_w_change], dim=1)
        features_list.append((bbox_w_change * scale).unsqueeze(-1))  # (B, T, 1)
        
        # 人体框高度变化率
        bbox_h_change = (bbox_h[:, 1:] - bbox_h[:, :-1]) / (bbox_h[:, :-1] + 1e-6)
        bbox_h_change = torch.cat([bbox_h_change[:, :1], bbox_h_change], dim=1)
        features_list.append((bbox_h_change * scale).unsqueeze(-1))  # (B, T, 1)
        
        # 中心速度模长
        speed = torch.sqrt(vx**2 + vy**2 + 1e-6)
        features_list.append(speed.unsqueeze(-1))  # (B, T, 1)
        
        # 加速度模长
        acc = torch.sqrt(ax**2 + ay**2 + 1e-6)
        features_list.append(acc.unsqueeze(-1))  # (B, T, 1)
        
        # 躯干角度变化率
        torso_angle_change = (torso_angle[:, 1:] - torso_angle[:, :-1])
        torso_angle_change = torch.cat([torso_angle_change[:, :1], torso_angle_change], dim=1)
        features_list.append((torso_angle_change * scale).unsqueeze(-1))  # (B, T, 1)
        
        # ---- 10. 手部支撑特征 (6维，利用原 padding 空位) ----
        # 生物力学依据：保护性跌倒（FOOSH）中手通常先于身体落地，
        # 且落地后保持支撑状态。作为软特征供 TCN 学习时序先后关系，
        # 而非硬规则（晕厥/侧倒无撑手动作，硬规则会漏报最危险场景）。
        lw = keypoints[:, :, KP["left_wrist"]]    # (B, T, 3)
        rw = keypoints[:, :, KP["right_wrist"]]
        
        # 10a. 手腕速度 (2点 × 2维 = 4维)：手先落地 → 手腕速度峰值早于髋部
        wrist_vels = []
        for w in (lw, rw):
            dvx = w[:, 1:, 0] - w[:, :-1, 0]
            dvy = w[:, 1:, 1] - w[:, :-1, 1]
            dvx = torch.cat([dvx[:, :1], dvx], dim=1)
            dvy = torch.cat([dvy[:, :1], dvy], dim=1)
            wrist_vels.append((dvx * scale).unsqueeze(-1))
            wrist_vels.append((dvy * scale).unsqueeze(-1))
        features_list.append(torch.cat(wrist_vels, dim=-1))  # (B, T, 4)
        
        # 10b. 手腕贴地度 (1维)：距 bbox 底边的相对位置（取左右手最大值），
        # 1 = 贴在 bbox 底边（撑地/躺地），置信度低时衰减到 0
        bbox_y2 = bboxes[..., 3].unsqueeze(-1)  # (B, T, 1)
        bbox_hh = bbox_h.unsqueeze(-1)
        lw_low = (1.0 - (bbox_y2 - lw[..., 1:2]) / (bbox_hh + 1e-6)) * lw[..., 2:3]
        rw_low = (1.0 - (bbox_y2 - rw[..., 1:2]) / (bbox_hh + 1e-6)) * rw[..., 2:3]
        wrist_low = torch.maximum(lw_low, rw_low)  # (B, T, 1)
        features_list.append(wrist_low)
        
        # 10c. 支撑状态分 (1维)：贴地 × 手腕静止 = 保持支撑
        wrist_speed = torch.sqrt(
            wrist_vels[1].squeeze(-1)**2 + wrist_vels[3].squeeze(-1)**2 + 1e-6
        )  # 两手腕 vy 合成近似
        support_score = wrist_low.squeeze(-1) * torch.exp(-wrist_speed / 0.01)
        features_list.append(support_score.unsqueeze(-1))  # (B, T, 1)
        
        # 剩余维度用零填充或扩展
        current_dim = sum(f.shape[-1] for f in features_list)
        if current_dim < self.feature_dim:
            pad_dim = self.feature_dim - current_dim
            pad = torch.zeros(B, T, pad_dim, device=device)
            features_list.append(pad)
        
        # 拼接所有特征
        features = torch.cat(features_list, dim=-1)  # (B, T, feature_dim)
        
        # 确保维度正确
        assert features.shape[-1] == self.feature_dim, \
            f"Feature dim mismatch: expected {self.feature_dim}, got {features.shape[-1]}"
        
        return features
    
    def compute_single_frame(
        self,
        keypoints: np.ndarray,   # (17, 3)
        bbox: np.ndarray,         # (4,)
    ) -> np.ndarray:
        """
        计算单帧特征（用于在线推理时不依赖 batch）
        
        Args:
            keypoints: (17, 3) [x, y, conf]
            bbox: (4,) [x1, y1, x2, y2]
        
        Returns:
            features: (feature_dim,)
        """
        kp_t = torch.from_numpy(keypoints).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 17, 3)
        bb_t = torch.from_numpy(bbox).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
        
        with torch.no_grad():
            features = self.forward(kp_t, bb_t)  # (1, 1, feature_dim)
        
        return features.squeeze(0).squeeze(0).numpy()

    def compute_sequence(
        self,
        keypoints_seq: np.ndarray,  # (T, 17, 3)
        bboxes_seq: np.ndarray,      # (T, 4)
        step_scales: np.ndarray = None,
    ) -> np.ndarray:
        """
        计算序列特征
        
        Args:
            keypoints_seq: (T, 17, 3)
            bboxes_seq: (T, 4)
        
        Returns:
            features: (T, feature_dim)
        """
        kp_t = torch.from_numpy(keypoints_seq).float().unsqueeze(0)  # (1, T, 17, 3)
        bb_t = torch.from_numpy(bboxes_seq).float().unsqueeze(0)  # (1, T, 4)
        
        with torch.no_grad():
            scales = torch.from_numpy(step_scales).float().unsqueeze(0) if step_scales is not None else None
            features = self.forward(kp_t, bb_t, step_scales=scales)  # (1, T, feature_dim)
        
        return features.squeeze(0).numpy()

    def get_feature_names(self) -> List[str]:
        """返回特征名称列表，便于调试和可视化"""
        names = [
            "aspect_ratio",
            "area_change",
            "center_vx", "center_vy", "center_ax", "center_ay",
            "torso_angle", "torso_len_norm",
        ]
        for kp_name in self.velocity_channels:
            names.append(f"{kp_name}_vx")
            names.append(f"{kp_name}_vy")
        names.extend([
            "conf_mean", "conf_min", "conf_std",
            "upper_y_mean", "lower_y_mean", "upper_y_std", "lower_y_std",
            "upper_x_mean", "lower_x_mean", "upper_x_std", "lower_x_std",
            "left_knee_angle", "left_knee_cos",
            "right_knee_angle", "right_knee_cos",
            "bbox_w_change", "bbox_h_change",
            "center_speed", "center_acc",
            "torso_angle_change",
            # 手部支撑特征（FOOSH：手先落地 + 保持支撑）
            "left_wrist_vx", "left_wrist_vy",
            "right_wrist_vx", "right_wrist_vy",
            "wrist_low",
            "support_score",
        ])
        # 补齐到 feature_dim
        while len(names) < self.feature_dim:
            names.append(f"pad_{len(names)}")
        return names[:self.feature_dim]
