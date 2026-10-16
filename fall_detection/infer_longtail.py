"""
长尾场景批量推理脚本
====================
对 test/ 中 20 个跌倒视频，施加 7 种环境增强后逐帧推理，
统计每种场景下检出摔倒的个数，输出日志到 outputs/longtail_log.txt。

场景：
  1. 原始（基准）
  2. 红外模拟（灰度→CLAHE→Gamma→噪声→模糊，黑白高对比伪红外）
  3. 运动模糊
  4. 视角变化（随机缩放+平移）
  5. 随机遮挡
  6. 雨雾
  7. 微光（Gamma 压暗）

用法:
    python infer_longtail.py --checkpoint checkpoints/best.pth
"""

import os, sys, glob, json, time, argparse, random
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════
#  环境增强函数
# ═══════════════════════════════════════════

def enh_original(frame):
    return frame.copy()

def enh_infrared(frame):
    """RGB→灰度→CLAHE→Gamma→噪声→模糊→伪红外（黑白高对比，人体偏亮背景偏暗）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_u8 = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_u8 = clahe.apply(gray_u8)
    gray = gray_u8.astype(np.float32)
    gray = np.power(gray / 255.0, 0.6) * 255.0     # Gamma 提亮
    noise = np.random.normal(0, 8, gray.shape).astype(np.float32)
    gray = np.clip(gray + noise, 0, 255)
    gray = cv2.GaussianBlur(gray, (3, 3), 1.0)
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def enh_blur(frame):
    """运动模糊 + 高斯模糊（模拟摄像头抖动/失焦）"""
    ks = 15
    kernel = np.zeros((ks, ks))
    kernel[int((ks - 1) / 2), :] = np.ones(ks) / ks
    motion = cv2.filter2D(frame, -1, kernel)
    return cv2.GaussianBlur(motion, (5, 5), 2)

def enh_viewpoint(frame):
    """视角变化：随机缩放 0.7-1.3 倍 + 随机平移"""
    h, w = frame.shape[:2]
    scale = random.uniform(0.7, 1.3)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros_like(frame)
    if scale >= 1.0:
        x1, y1 = (new_w - w) // 2, (new_h - h) // 2
        canvas = scaled[y1:y1 + h, x1:x1 + w]
    else:
        dx = random.randint(0, w - new_w)
        dy = random.randint(0, h - new_h)
        canvas[dy:dy + new_h, dx:dx + new_w] = scaled
    return canvas

def enh_occlusion(frame):
    """随机遮挡：下半身区域模糊/黑色方块"""
    h, w = frame.shape[:2]
    img = frame.copy()
    n_blocks = random.randint(1, 3)
    for _ in range(n_blocks):
        bw = random.randint(w // 6, w // 3)
        bh = random.randint(h // 5, h // 3)
        bx = random.randint(0, w - bw)
        by = random.randint(int(h * 0.4), h - bh)
        if random.random() < 0.5:
            img[by:by + bh, bx:bx + bw] = img[by:by + bh, bx:bx + bw] // 4
        else:
            img[by:by + bh, bx:bx + bw] = 30
    return img

def enh_rain_fog(frame):
    """雨线 + 雾化"""
    img = (frame.astype(np.float32) * 0.6 + 100).clip(0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    rain = np.zeros((h, w), dtype=np.float32)
    for _ in range(300):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        length = random.randint(10, 30)
        angle = random.uniform(-0.2, 0.2)
        dx, dy = int(length * np.sin(angle)), int(length * np.cos(angle))
        cv2.line(rain, (x, y), (max(0, x + dx), min(h - 1, y + dy)), (180,), 1, cv2.LINE_AA)
    rain = cv2.GaussianBlur(rain, (3, 3), 1).astype(np.float32) * 0.4
    rain_bgr = np.stack([rain] * 3, axis=-1)
    return np.clip(img.astype(np.float32) + rain_bgr, 0, 255).astype(np.uint8)

def enh_lowlight(frame):
    """微光：Gamma 压暗 + 对比度降低"""
    img = frame.copy() / 255.0
    img = np.power(img, 2.5)
    img = (img * 0.6).clip(0, 1)
    return (img * 255).astype(np.uint8)


SCENARIOS = [
    ("原始",       "original",   enh_original),
    ("红外模拟",   "infrared",   enh_infrared),
    ("模糊",       "blur",       enh_blur),
    ("视角变化",   "viewpoint",  enh_viewpoint),
    ("随机遮挡",   "occlusion",  enh_occlusion),
    ("雨雾",       "rain_fog",   enh_rain_fog),
    ("微光",       "lowlight",   enh_lowlight),
]


# ═══════════════════════════════════════════
#  推理引擎
# ═══════════════════════════════════════════

def build_pipeline(checkpoint_path, version="large"):
    """构建推理管线（复用 infer.py 逻辑）"""
    import torch
    from models.fall_detector import create_fall_detector
    from pipeline.detector import YOLOPoseDetector
    from pipeline.inference import FallDetectionPipeline
    from models.rule_refinement import RuleConfig

    fall_detector = create_fall_detector(version=version)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k: v for k, v in sd.items()
          if not k.endswith(".total_ops") and not k.endswith(".total_params")}
    fall_detector.load_state_dict(sd, strict=False)
    fall_detector.eval()

    detector = YOLOPoseDetector(input_size=(960, 576), conf_threshold=0.15)
    detector.load_model()

    rule_config = RuleConfig(
        tcn_prob_threshold=0.3,
        vote_threshold=0.6,
        aspect_ratio_change_threshold=0.2,
        rules_min_pass=1,
        velocity_peak_threshold=0.03,
        fall_memory_frames=60,
    )

    pipeline = FallDetectionPipeline(
        detector=detector,
        fall_detector=fall_detector,
        sequence_length=32,
        detection_interval=2,
        output_dir="outputs",
        rule_config=rule_config,
    )
    return pipeline


def run_inference(pipeline, video_path, enhance_fn, out_video_path=None):
    """对视频逐帧增强后推理，返回是否检出摔倒及最高概率，可选保存标注视频"""
    pipeline.reset()
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_prob = 0.0
    any_alarm = False

    writer = None
    if out_video_path:
        os.makedirs(os.path.dirname(out_video_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        enhanced = enhance_fn(frame)
        result = pipeline.process_frame(enhanced, frame_idx)
        if result.fall_probs:
            max_prob = max(max_prob, max(result.fall_probs.values()))
        if result.alarms and any(result.alarms.values()):
            any_alarm = True
        if writer is not None:
            vis = pipeline._draw_result(enhanced, result)
            writer.write(vis)

    cap.release()
    if writer is not None:
        writer.release()

    return any_alarm, float(max_prob)


# ═══════════════════════════════════════════
#  main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="长尾场景批量推理")
    parser.add_argument("--checkpoint", default="checkpoints/best.pth", help="模型路径")
    parser.add_argument("--test_dir", default="test", help="测试视频目录")
    parser.add_argument("--output", default="outputs/longtail_log.txt", help="日志输出路径")
    parser.add_argument("--video_out", default="outputs/longtail_videos", help="标注视频输出目录")
    parser.add_argument("--version", default="large", help="模型版本")
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.test_dir, "*.mp4")))
    if not videos:
        print(f"[Error] {args.test_dir}/ 中没有视频")
        return

    print("=" * 65)
    print(f"[Longtail] 长尾场景批量推理")
    print(f"[Longtail] 视频: {len(videos)} 个 | 场景: {len(SCENARIOS)} 种")
    print(f"[Longtail] 总计: {len(videos) * len(SCENARIOS)} 次推理")
    print("=" * 65)

    # 构建管线（只建一次）
    print("[Setup] 加载模型...")
    pipeline = build_pipeline(args.checkpoint, args.version)
    print("[Setup] 完成\n")

    # 准备日志
    log_lines = []
    log_lines.append(f"长尾场景推理日志 — {len(videos)} 个视频 × {len(SCENARIOS)} 种场景")
    log_lines.append(f"模型: {args.checkpoint}  | 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_lines.append("=" * 150)
    # 表头：视频  | 原始(prob) | 红外模拟(prob) | 模糊(prob) | ...
    hdr = f"{'视频':<25s}"
    for name, _, _ in SCENARIOS:
        hdr += f"  {name:<14s}"
    log_lines.append(hdr)
    log_lines.append("-" * 150)

    # 统计
    scenario_stats = {key: {"detected": 0, "total": 0} for _, key, _ in SCENARIOS}

    # 进度条
    total_tasks = len(videos) * len(SCENARIOS)
    pbar = tqdm(total=total_tasks, desc="推理进度", unit="次", ncols=100)

    print(hdr)
    print("-" * 150)

    for vp in videos:
        vname = os.path.splitext(os.path.basename(vp))[0]
        line = f"{vname[:25]:<25s}"
        for scen_name, scen_key, enh_fn in SCENARIOS:
            out_vid = os.path.join(args.video_out, scen_key, f"{vname}.mp4") if args.video_out else None
            detected, prob = run_inference(pipeline, vp, enh_fn, out_video_path=out_vid)
            p_text = f"{prob:.3f}" if detected else "—"
            line += f"  {'YES':>3s}({p_text:>6s})" if detected else f"  {'NO ':>3s}({p_text:>6s})"

            scenario_stats[scen_key]["total"] += 1
            if detected:
                scenario_stats[scen_key]["detected"] += 1

            pbar.update(1)

        log_lines.append(line)
        print(line, flush=True)

    pbar.close()

    # 汇总
    log_lines.append("-" * 150)
    log_lines.append("")
    log_lines.append("场景汇总（每个场景 检出数 / 总数）:")
    log_lines.append("-" * 46)

    summary_lines = []
    for scen_name, scen_key, _ in SCENARIOS:
        st = scenario_stats[scen_key]
        rate = f"{st['detected'] / st['total'] * 100:.1f}%" if st['total'] > 0 else "0.0%"
        log_lines.append(f"  {scen_name:<10s}  {st['detected']:>2d} / {st['total']:>2d}  ({rate})")
        summary_lines.append(f"  {scen_name:<10s}  {st['detected']}/{st['total']} ({rate})")

    log_lines.append("=" * 150)

    # 写入文件
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n{'=' * 65}")
    print(f"  [Done]")
    for sl in summary_lines:
        print(sl)
    print(f"  日志: {args.output}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
