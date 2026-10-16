"""
环境长尾增强模块
================
对视频应用可控的环境增强，模拟赛题要求的恶劣场景：
- 微光 (lowlight)：Gamma 变换 + 亮度/对比度降低
- 红外模拟 (infrared)：灰度 + CLAHE + 噪声
- 遮挡 (occlusion)：人体区域黑条/家具遮挡模拟
- 远距离 (far)：画面缩小内嵌于大画布
- 模糊 (blur)：运动模糊 + 高斯模糊
- 雨雾 (rain_fog)：雨线 + 雾化

所有增强作用于视频帧，不修改原始文件。
"""

import cv2
import numpy as np
from typing import Optional


# ═══════════════════════════════════════════════
#  1. 微光模拟
# ═══════════════════════════════════════════════
def apply_lowlight(frame: np.ndarray, gamma: float = 2.5) -> np.ndarray:
    """Gamma 变换模拟傍晚微光"""
    img = frame.copy() / 255.0
    img = np.power(img, gamma)
    img = (img * 0.6).clip(0, 1)  # 降低整体亮度
    return (img * 255).astype(np.uint8)


# ═══════════════════════════════════════════════
#  2. 红外模拟
# ═══════════════════════════════════════════════
def apply_infrared(frame: np.ndarray) -> np.ndarray:
    """RGB → 灰度 → CLAHE → Gamma → 噪声 → 模糊 → 伪红外 (黑白高对比，人体偏亮背景偏暗)"""
    # ① 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ② CLAHE 增强暗处对比度（先转回 uint8）
    gray_u8 = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_u8 = clahe.apply(gray_u8)
    gray = gray_u8.astype(np.float32)

    # ③ Gamma 提亮（模拟红外曝光，人体偏亮）
    gray = np.power(gray / 255.0, 0.6) * 255.0

    # ④ 加高斯噪声（模拟红外雪花）
    noise = np.random.normal(0, 8, gray.shape).astype(np.float32)
    gray = np.clip(gray + noise, 0, 255)

    # ⑤ 轻微模糊（模拟红外低分辨率）
    gray = cv2.GaussianBlur(gray, (3, 3), 1.0)

    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)


# ═══════════════════════════════════════════════
#  3. 遮挡模拟（下半身被家具遮挡）
# ═══════════════════════════════════════════════
def apply_occlusion(frame: np.ndarray) -> np.ndarray:
    """模拟家具/桌子遮挡人体下半部"""
    h, w = frame.shape[:2]
    img = frame.copy()
    # 下半部 40% 区域加黑条/模糊
    y_start = int(h * 0.55)
    y_end = h
    # 随机选择 2-3 个遮挡区域
    regions = [(0.1, 0.4), (0.55, 0.85)]
    for left, right in regions:
        x1, x2 = int(w * left), int(w * right)
        y1, y2 = y_start, y_end
        roi = img[y1:y2, x1:x2]
        roi_blur = cv2.GaussianBlur(roi, (31, 31), 15)
        # 混合：部分区域完全遮挡，部分区域模糊
        mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32) * 0.6
        for c in range(3):
            roi[:, :, c] = (roi[:, :, c] * (1 - mask) + roi_blur[:, :, c] * mask).astype(np.uint8)
        img[y1:y2, x1:x2] = roi
    return img


# ═══════════════════════════════════════════════
#  4. 远距离模拟
# ═══════════════════════════════════════════════
def apply_far_distance(frame: np.ndarray, scale: float = 0.35) -> np.ndarray:
    """模拟远距离监控：画面缩小并嵌入大画布"""
    h, w = frame.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # 放在画布中央偏下（监控视角）
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (40, 40, 40)  # 暗背景
    y_offset = int(h * 0.3)
    x_offset = (w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = small
    return canvas


# ═══════════════════════════════════════════════
#  5. 模糊
# ═══════════════════════════════════════════════
def apply_blur(frame: np.ndarray) -> np.ndarray:
    """运动模糊 + 高斯模糊混合"""
    img = frame.copy()
    # 运动模糊核（水平方向）
    ks = 15
    kernel = np.zeros((ks, ks))
    kernel[int((ks - 1) / 2), :] = np.ones(ks) / ks
    motion_blur = cv2.filter2D(img, -1, kernel)
    # 再加轻度高斯模糊
    return cv2.GaussianBlur(motion_blur, (5, 5), 2)


# ═══════════════════════════════════════════════
#  6. 雨雾
# ═══════════════════════════════════════════════
def apply_rain_fog(frame: np.ndarray) -> np.ndarray:
    """雨线 + 雾化合成"""
    img = frame.copy().astype(np.float32)
    h, w = img.shape[:2]

    # 雾化：增加亮度 + 降低对比度
    img = img * 0.6 + 100
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 雨线
    rain_layer = np.zeros((h, w), dtype=np.uint8)
    n_streaks = 400
    for _ in range(n_streaks):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = np.random.randint(10, 30)
        angle = np.random.uniform(-0.2, 0.2)
        dx = int(length * np.sin(angle))
        dy = int(length * np.cos(angle))
        cv2.line(rain_layer, (x, y), (max(0, x + dx), min(h - 1, y + dy)),
                  (180,), 1, cv2.LINE_AA)

    rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 1)
    rain_bgr = (cv2.cvtColor(rain_layer, cv2.COLOR_GRAY2BGR) * 0.5).astype(np.float32)
    result = cv2.add(img.astype(np.float32), rain_bgr)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# ═══════════════════════════════════════════════
#  增强注册表
# ═══════════════════════════════════════════════
ENHANCEMENTS = {
    "original":   ("原始", lambda f: f),
    "lowlight":   ("微光", apply_lowlight),
    "infrared":   ("红外模拟", apply_infrared),
    "occlusion":  ("遮挡", apply_occlusion),
    "far":        ("远距离", apply_far_distance),
    "blur":       ("模糊", apply_blur),
    "rain_fog":   ("雨雾", apply_rain_fog),
}


def get_enhanced_video_frames(video_path: str, enhancement_key: str):
    """
    生成增强后的视频帧生成器
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    enh_fn = ENHANCEMENTS.get(enhancement_key, lambda f: f)[1]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield enh_fn(frame)

    cap.release()


def get_video_info(video_path: str):
    """获取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": total, "width": w, "height": h}
