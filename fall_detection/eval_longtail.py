"""
长尾场景评测脚本
======================
对应赛题第三大痛点（场景长尾问题）：
傍晚微光、夜间红外、人体遮挡等恶劣场景下的检出率评测。

做法：
1. 对 test/ 每段视频生成三种变体：
   - lowlight   傍晚微光（gamma 压暗 + 噪声）
   - ir_style   红外风格（灰度 + 对比度增强 + 传感器噪声，走 --ir 管线）
   - occluded   随机遮挡（Random Erasing 模拟家具/他人遮挡）
2. 每个场景批量跑 infer.py --test，汇总各视频跌倒事件数与最高概率
3. 输出对比表（控制台 + longtail_summary.md / .json）到 eval_report/

用法：
    python eval_longtail.py --checkpoint checkpoints/best.pth
    python eval_longtail.py --checkpoint checkpoints/best.pth --test_dir test
"""

import os
import sys
import glob
import json
import argparse
import subprocess

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.augment import IRAugmentation, SpatialAugmentation
from utils.env_enhance import apply_infrared, apply_blur, apply_far_distance, apply_rain_fog

VIDEO_EXTS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]


# ═══════════════════════════════════════════════
#  变体生成
# ═══════════════════════════════════════════════

def make_lowlight(frame: np.ndarray) -> np.ndarray:
    """傍晚微光：gamma 压暗 + 高斯噪声"""
    gamma = 2.8  # gamma>1 压暗（均值降至 ~50，触发自动低光增强）
    table = ((np.arange(256) / 255.0) ** gamma * 255.0).astype(np.uint8)
    dark = cv2.LUT(frame, table)
    dark = dark.astype(np.float32) * 0.6  # 线性衰减，保证亮场景也进入低光域
    noise = np.random.randn(*dark.shape) * 6
    return np.clip(dark + noise, 0, 255).astype(np.uint8)


def make_ir_style(frame: np.ndarray) -> np.ndarray:
    """红外风格：灰度→CLAHE→Gamma→噪声→模糊 （黑白高对比，人体偏亮背景偏暗）"""
    return apply_infrared(frame)


def make_occluded(frame: np.ndarray) -> np.ndarray:
    """随机遮挡：Random Erasing"""
    return SpatialAugmentation.random_erase(
        frame, scale=(0.05, 0.25), max_boxes=3
    )


def make_blur(frame: np.ndarray) -> np.ndarray:
    """运动模糊 + 高斯模糊"""
    return apply_blur(frame)


def make_far(frame: np.ndarray) -> np.ndarray:
    """远距离：画面缩小嵌入暗背景"""
    return apply_far_distance(frame)


def make_rain_fog(frame: np.ndarray) -> np.ndarray:
    """雨雾：雨线 + 雾化"""
    return apply_rain_fog(frame)


VARIANTS = {
    "lowlight": make_lowlight,
    "ir_style": make_ir_style,
    "occluded": make_occluded,
    "blur": make_blur,
    "far": make_far,
    "rain_fog": make_rain_fog,
}


def generate_variants(test_dir: str, out_root: str):
    """为 test_dir 中每段视频生成三种变体，返回 {variant: dir}"""
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(glob.glob(os.path.join(test_dir, ext)))
    videos = sorted(videos)
    if not videos:
        print(f"[Error] {test_dir}/ 中没有视频")
        sys.exit(1)

    variant_dirs = {}
    for name in VARIANTS:
        d = os.path.join(out_root, "videos", name)
        os.makedirs(d, exist_ok=True)
        variant_dirs[name] = d

    for vp in videos:
        basename = os.path.basename(vp)
        for name, func in VARIANTS.items():
            out_path = os.path.join(variant_dirs[name], basename)
            if os.path.exists(out_path):
                continue
            cap = cv2.VideoCapture(vp)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            np.random.seed(42)  # 可复现
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(func(frame))
            cap.release()
            writer.release()
            print(f"  生成 {name}/{basename}")

    return videos, variant_dirs


# ═══════════════════════════════════════════════
#  场景推理
# ═══════════════════════════════════════════════

def run_scenario(name: str, test_dir: str, checkpoint: str,
                 out_dir: str, extra_args=None):
    """调用 infer.py --test 跑一个场景，返回 {basename: result_json}"""
    cmd = [
        sys.executable, "infer.py", "--test",
        "--test_dir", test_dir,
        "--output", out_dir,
        "--checkpoint", checkpoint,
        "--version", "large",
    ]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n[Eval] 场景 {name}: {' '.join(cmd[1:])}")
    subprocess.run(cmd, check=True)

    results = {}
    for jp in glob.glob(os.path.join(out_dir, "*_results.json")):
        with open(jp, encoding="utf-8") as f:
            data = json.load(f)
        basename = os.path.basename(jp).replace("_results.json", "")
        results[basename] = data
    return results


# ═══════════════════════════════════════════════
#  汇总
# ═══════════════════════════════════════════════

def summarize(all_results: dict, out_root: str):
    """
    all_results: {scenario: {basename: result_json}}
    输出对比表 + longtail_summary.md / .json
    """
    scenarios = list(all_results.keys())
    videos = sorted({v for r in all_results.values() for v in r})

    lines = []
    header = "| 视频 | " + " | ".join(scenarios) + " |"
    sep = "|" + "---|" * (len(scenarios) + 1)
    lines.append(header)
    lines.append(sep)

    summary = {"scenarios": scenarios, "videos": {}}
    print()
    print(header)
    print(sep)

    for v in videos:
        row = [v]
        vinfo = {}
        for s in scenarios:
            data = all_results[s].get(v)
            if data is None:
                row.append("N/A")
                continue
            n_events = len(data["fall_events"])
            max_prob = max(
                (e["max_probability"] for e in data["fall_events"]),
                default=0.0,
            )
            row.append(f"{n_events} 事件 / {max_prob:.2f}")
            vinfo[s] = {"events": n_events, "max_prob": round(max_prob, 4),
                        "frames": data["total_frames"],
                        "avg_latency_ms": round(data["avg_latency_ms"], 1)}
        summary["videos"][v] = vinfo
        line = "| " + " | ".join(row) + " |"
        lines.append(line)
        print(line)

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "longtail_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_root, "longtail_summary.md"), "w",
              encoding="utf-8") as f:
        f.write("# 长尾场景评测结果\n\n")
        f.write("单元格 = 跌倒事件数 / 事件最高概率\n\n")
        f.write("\n".join(lines) + "\n")
    print(f"\n[Eval] 汇总已保存: {out_root}/longtail_summary.{{md,json}}")


def main():
    parser = argparse.ArgumentParser(description="长尾场景评测")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--test_dir", type=str, default="test")
    parser.add_argument("--output", type=str, default="eval_report/longtail")
    parser.add_argument("--skip_generate", action="store_true",
                        help="跳过变体生成（已生成过）")
    args = parser.parse_args()

    print("=" * 60)
    print("[Eval] 长尾场景评测")
    print("=" * 60)

    if args.skip_generate:
        variant_dirs = {n: os.path.join(args.output, "videos", n)
                        for n in VARIANTS}
    else:
        print("\n[1/2] 生成场景变体...")
        _, variant_dirs = generate_variants(args.test_dir, args.output)

    print("\n[2/2] 逐场景推理...")
    all_results = {}
    # 原始场景（基线）
    all_results["original"] = run_scenario(
        "original", args.test_dir, args.checkpoint,
        os.path.join(args.output, "original"),
    )
    # 微光（自动低光增强默认开启）
    all_results["lowlight"] = run_scenario(
        "lowlight", variant_dirs["lowlight"], args.checkpoint,
        os.path.join(args.output, "lowlight"),
    )
    # 微光（关闭增强，消融对照）
    all_results["lowlight_noenh"] = run_scenario(
        "lowlight_noenh", variant_dirs["lowlight"], args.checkpoint,
        os.path.join(args.output, "lowlight_noenh"),
        extra_args=["--no_lowlight"],
    )
    # 红外风格（走 --ir 预处理）
    all_results["ir_style"] = run_scenario(
        "ir_style", variant_dirs["ir_style"], args.checkpoint,
        os.path.join(args.output, "ir_style"),
        extra_args=["--ir"],
    )
    # 遮挡
    all_results["occluded"] = run_scenario(
        "occluded", variant_dirs["occluded"], args.checkpoint,
        os.path.join(args.output, "occluded"),
    )
    # 模糊
    all_results["blur"] = run_scenario(
        "blur", variant_dirs["blur"], args.checkpoint,
        os.path.join(args.output, "blur"),
    )
    # 远距离
    all_results["far"] = run_scenario(
        "far", variant_dirs["far"], args.checkpoint,
        os.path.join(args.output, "far"),
    )
    # 雨雾
    all_results["rain_fog"] = run_scenario(
        "rain_fog", variant_dirs["rain_fog"], args.checkpoint,
        os.path.join(args.output, "rain_fog"),
    )

    summarize(all_results, args.output)


if __name__ == "__main__":
    main()
