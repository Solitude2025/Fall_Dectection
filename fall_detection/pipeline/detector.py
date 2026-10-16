"""
YOLOv8n-Pose 人体检测与姿态估计封装
------------------------------------
封装 Ultralytics YOLOv8n-pose，提供人体检测 + 17关键点提取。

设计要点：
- 支持 RGB 和 红外 双模态输入
- 自适应分辨率缩放（1080P → 640x384 推理，节省算力）
- 关键点置信度过滤
- 结果归一化输出
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PersonDetection:
    """单个人体检测结果"""
    bbox: np.ndarray          # (4,) [x1, y1, x2, y2] 归一化坐标
    keypoints: np.ndarray     # (17, 3) [x, y, conf] 归一化坐标
    confidence: float         # 检测置信度
    track_id: Optional[int] = None  # 跟踪 ID
    
    @property
    def center(self) -> np.ndarray:
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])
    
    @property
    def bbox_area(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return w * h
    
    @property
    def aspect_ratio(self) -> float:
        w = self.bbox[2] - self.bbox[0] + 1e-6
        h = self.bbox[3] - self.bbox[1] + 1e-6
        return h / w


class YOLOPoseDetector:
    """
    YOLOv8n-Pose 检测器封装
    
    使用 Ultralytics 官方 YOLOv8n-pose 模型，
    支持 GPU/CPU/NPU 推理。
    
    模型参数量:
        YOLOv8n-pose.pt: ~3.3M 参数, ~6.4MB (FP16)
        满足赛题 < 20M 要求
    """
    
    # COCO 17 关键点定义
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    
    # 用于跌倒检测的核心关键点
    FALL_KEYPOINTS = [
        "nose", "left_shoulder", "right_shoulder",
        "left_hip", "right_hip", "left_knee", "right_knee",
    ]
    
    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cpu",
        input_size: Tuple[int, int] = (640, 384),  # width, height
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        kp_conf_threshold: float = 0.3,
        max_det: int = 5,
    ):
        """
        Args:
            model_path: YOLOv8n-pose 模型路径
            device: 推理设备 ("cpu", "cuda", "npu")
            input_size: 模型输入分辨率 (width, height)
            conf_threshold: 检测置信度阈值
            iou_threshold: NMS IoU 阈值
            kp_conf_threshold: 关键点置信度阈值
            max_det: 最大检测人数
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.kp_conf_threshold = kp_conf_threshold
        self.max_det = max_det
        self.device = device
        
        self._model = None
        self._model_path = model_path
    
    def load_model(self):
        """加载 YOLOv8n-pose 模型"""
        if self._model is not None:
            return
        
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            
            # 获取模型参数信息
            if hasattr(self._model, 'model'):
                total_params = sum(p.numel() for p in self._model.model.parameters())
                print(f"[YOLOPoseDetector] 模型加载成功: {total_params:,} 参数")
            else:
                print(f"[YOLOPoseDetector] 模型加载成功: {self._model_path}")
                
        except ImportError:
            raise ImportError(
                "请安装 ultralytics: pip install ultralytics\n"
                "YOLOv8n-pose 模型将自动下载: yolo pose download model=yolov8n-pose.pt"
            )
        except Exception as e:
            raise RuntimeError(f"人体检测模型加载失败: {e}") from e
    
    def detect(
        self,
        image: np.ndarray,
        original_shape: Optional[Tuple[int, int]] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> List[PersonDetection]:
        """
        检测图像中的人体
        
        Args:
            image: (H, W, 3) BGR 或 RGB 图像
            original_shape: 原始图像形状 (H, W)
        
        Returns:
            List[PersonDetection]: 检测到的人体列表
        """
        if self._model is None:
            self.load_model()
        
        h_orig, w_orig = original_shape or image.shape[:2]
        
        try:
            inference_size = input_size or self.input_size
            results = self._model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=(inference_size[1], inference_size[0]),  # Ultralytics: (height, width)
                max_det=self.max_det,
                # ``device`` used to be stored but never forwarded, which
                # silently forced the live camera onto CPU even on machines
                # with a CUDA GPU.  Keep the choice explicit and let
                # Ultralytics select the requested accelerator.
                device=self.device,
                verbose=False,
            )
            
            detections = []
            for result in results:
                if result.keypoints is None:
                    continue
                
                boxes = result.boxes
                keypoints_data = result.keypoints
                
                if boxes is None or len(boxes) == 0:
                    continue
                
                for i in range(len(boxes)):
                    # 获取边界框
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # 归一化到 [0, 1]
                    bbox_norm = np.array([
                        box[0] / w_orig, box[1] / h_orig,
                        box[2] / w_orig, box[3] / h_orig,
                    ])
                    
                    # 获取关键点
                    kp = keypoints_data.xy[i].cpu().numpy()       # (17, 2)
                    kp_conf = keypoints_data.conf[i].cpu().numpy()  # (17,)
                    
                    # 滤波低置信度关键点
                    kp_conf[kp_conf < self.kp_conf_threshold] = 0.0
                    
                    # 组合为 (17, 3)
                    kp_full = np.concatenate([
                        kp / np.array([w_orig, h_orig]),  # 归一化坐标
                        kp_conf[:, None],                   # 置信度
                    ], axis=1)
                    
                    det = PersonDetection(
                        bbox=bbox_norm,
                        keypoints=kp_full,
                        confidence=conf,
                    )
                    detections.append(det)
            
            return detections[:self.max_det]
        
        except Exception as e:
            raise RuntimeError(f"人体检测失败: {e}") from e
    
    def detect_batch(
        self,
        images: List[np.ndarray],
    ) -> List[List[PersonDetection]]:
        """批量检测"""
        return [self.detect(img) for img in images]
    
    def get_fall_relevant_keypoints(
        self,
        detection: PersonDetection,
    ) -> np.ndarray:
        """
        提取跌倒相关的核心关键点
        
        Returns:
            (7, 3) 核心关键点数组
        """
        indices = [self.KEYPOINT_NAMES.index(k) for k in self.FALL_KEYPOINTS]
        return detection.keypoints[indices]
    
    def preprocess_for_ir(
        self,
        ir_image: np.ndarray,
    ) -> np.ndarray:
        """
        红外图像预处理
        
        - 单通道 → 三通道复制
        - CLAHE 对比度增强
        - 边缘增强
        
        Args:
            ir_image: (H, W) 或 (H, W, 1) 红外图像
        
        Returns:
            (H, W, 3) 预处理后的三通道图像
        """
        if len(ir_image.shape) == 2:
            ir_image = ir_image[..., np.newaxis]
        
        if ir_image.shape[-1] == 1:
            ir_image = np.repeat(ir_image, 3, axis=-1)
        
        # 归一化到 [0, 255]
        if ir_image.max() <= 1.0:
            ir_image = (ir_image * 255).astype(np.uint8)
        
        # CLAHE 对比度增强
        try:
            import cv2
            lab = cv2.cvtColor(ir_image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            ir_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        except ImportError:
            pass  # 如果没有 cv2 就跳过
        
        return ir_image

    def preprocess_for_lowlight(
        self,
        frame: np.ndarray,
        dark_threshold: float = 60.0,
    ) -> np.ndarray:
        """
        低光场景自适应预处理（傍晚微光、逆光等）

        先计算帧平均亮度，仅当亮度低于阈值时执行增强
        （gamma 提亮 + CLAHE），正常光照帧原样返回，
        避免引入额外计算与画质损失。

        Args:
            frame: (H, W, 3) BGR/RGB uint8 图像
            dark_threshold: 平均亮度阈值（0-255），低于则判定为低光

        Returns:
            (H, W, 3) 增强后（或原样）的三通道图像
        """
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
            if frame.ndim == 3 else frame
        if float(gray.mean()) >= dark_threshold:
            return frame

        # gamma 提亮暗区
        gamma = 0.6
        table = (np.arange(256) / 255.0) ** gamma * 255.0
        table = np.clip(table, 0, 255).astype(np.uint8)
        enhanced = cv2.LUT(frame, table)

        # CLAHE 局部对比度增强
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        return enhanced


class YOLOPoseDetectorSim:
    """
    YOLOv8n-Pose 模拟器（无真实模型时的占位）
    
    用于在没有安装 ultralytics 的环境下测试整体流程。
    """
    
    def __init__(self, **kwargs):
        print("[YOLOPoseDetectorSim] 使用模拟模式，生成假关键点数据")
        self.conf_threshold = kwargs.get("conf_threshold", 0.25)
        self.max_det = kwargs.get("max_det", 5)

    def preprocess_for_lowlight(self, frame: np.ndarray) -> np.ndarray:
        """Keep the fallback detector API-compatible with the real detector."""

        return frame

    def preprocess_for_ir(self, frame: np.ndarray) -> np.ndarray:
        """Keep the fallback detector API-compatible with the real detector."""

        return frame
    
    def detect(
        self,
        image: np.ndarray,
        original_shape: Optional[Tuple[int, int]] = None,
    ) -> List[PersonDetection]:
        """生成模拟检测结果"""
        import random
        
        # 模拟一个站立的人
        bbox = np.array([0.3, 0.05, 0.7, 0.95])
        
        # 模拟 17 个关键点（站立姿势）
        kp = np.array([
            [0.50, 0.10, 0.9],  # nose
            [0.47, 0.09, 0.9],  # left_eye
            [0.53, 0.09, 0.9],  # right_eye
            [0.45, 0.10, 0.8],  # left_ear
            [0.55, 0.10, 0.8],  # right_ear
            [0.42, 0.20, 0.9],  # left_shoulder
            [0.58, 0.20, 0.9],  # right_shoulder
            [0.38, 0.35, 0.8],  # left_elbow
            [0.62, 0.35, 0.8],  # right_elbow
            [0.35, 0.50, 0.7],  # left_wrist
            [0.65, 0.50, 0.7],  # right_wrist
            [0.44, 0.55, 0.8],  # left_hip
            [0.56, 0.55, 0.8],  # right_hip
            [0.42, 0.72, 0.8],  # left_knee
            [0.58, 0.72, 0.8],  # right_knee
            [0.40, 0.90, 0.7],  # left_ankle
            [0.60, 0.90, 0.7],  # right_ankle
        ])
        
        return [PersonDetection(bbox=bbox, keypoints=kp, confidence=0.9)]
