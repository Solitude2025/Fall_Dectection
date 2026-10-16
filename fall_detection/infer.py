"""
推理脚本
--------------
支持模式：
1. 单视频推理：python infer.py --video test.mp4
2. 批量测试（test/ 目录）：python infer.py --test
3. 摄像头实时推理：python infer.py --webcam

输出：
- outputs/ 目录下保存标注后的视频（骨架+边框+Fall状态）
- outputs/ 目录下保存 JSON 推理结果

用法:
    # 处理 test/ 目录下所有视频（最常用）
    python infer.py --test --checkpoint checkpoints/best.pth
    
    # 单视频
    python infer.py --video test.mp4 --checkpoint checkpoints/best.pth
    
    # 实时摄像头
    python infer.py --webcam --checkpoint checkpoints/best.pth
    
    # GPU 加速
    python infer.py --test --checkpoint checkpoints/best.pth --device cuda
"""

import os
import sys
import argparse
import glob
import yaml
import torch
import cv2
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fall_detector import FallDetector, create_fall_detector
from pipeline.inference import FallDetectionPipeline
from pipeline.detector import YOLOPoseDetector
from models.rule_refinement import RuleConfig


def process_single_video(pipeline, video_path, output_dir):
    """处理单个视频并保存到 outputs/"""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(output_dir, f"{basename}.mp4")

    print(f"\n[Infer] 处理视频: {video_path}")

    result = pipeline.process_video(video_path)

    # 打印结果
    print(f"\n{'='*60}")
    print(f"推理完成")
    print(f"  总帧数: {result.total_frames}")
    print(f"  总耗时: {result.total_time:.2f}s")
    print(f"  FPS: {result.fps:.2f}")
    print(f"  平均延迟: {result.avg_latency_ms:.2f}ms")
    print(f"  跌倒事件数: {len(result.fall_events)}")
    for i, event in enumerate(result.fall_events):
        print(f"  事件 {i+1}: Person {event.person_id}, "
              f"帧 {event.start_frame}-{event.end_frame}, "
              f"最大概率: {event.max_probability:.3f}")
    print(f"{'='*60}")

    # 合规检查
    if result.avg_latency_ms > 100:
        print(f"\n[WARNING] 平均延迟 {result.avg_latency_ms:.1f}ms 超过 100ms 限制!")
    else:
        print(f"\n[OK] 平均延迟 {result.avg_latency_ms:.1f}ms 满足要求!")

    return result


def main():
    parser = argparse.ArgumentParser(description="跌倒检测推理")

    # ── 输入源 ──
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--test", action="store_true",
                    help="处理 test/ 目录下的所有视频")
    src.add_argument("--video", type=str, default=None,
                    help="单个视频路径")
    src.add_argument("--webcam", action="store_true",
                    help="摄像头实时推理")

    # ── 模型配置 ──
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                       help="模型检查点路径")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    parser.add_argument("--version", type=str, default="standard",
                       choices=["standard", "light", "large",
                                "transformer-tcn-standard", "transformer-tcn-efficient", "transformer-tcn-light"],
                       help="模型版本")
    parser.add_argument("--device", type=str, default="cpu",
                       help="推理设备 (cpu/cuda)")

    # ── 输入/输出目录 ──
    parser.add_argument("--test_dir", type=str, default="test",
                       help="批量推理视频目录 (配合 --test 使用)")
    parser.add_argument("--output", type=str, default="outputs",
                       help="结果输出目录")

    # ── 其他 ──
    parser.add_argument("--ir", action="store_true",
                       help="红外模式")
    parser.add_argument("--no_lowlight", action="store_true",
                       help="关闭自动低光增强（默认开启，仅暗帧触发）")
    parser.add_argument("--no_display", action="store_true",
                       help="不显示实时画面")

    args = parser.parse_args()

    # ── 加载配置 ──
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # ── 创建输出目录 ──
    os.makedirs(args.output, exist_ok=True)

    # ── 设备 ──
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Infer] CUDA 不可用，回退到 CPU")
        device = "cpu"

    # ── 加载跌倒检测模型 ──
    print(f"[Infer] 加载模型 (version={args.version})...")
    fall_detector = create_fall_detector(version=args.version)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # 过滤掉 thop profile 注入的 total_ops/total_params 键
        state_dict = {k: v for k, v in checkpoint["model_state_dict"].items()
                      if not k.endswith(".total_ops") and not k.endswith(".total_params")}
        incompatible = fall_detector.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[Infer] 缺少的权重键: {incompatible.missing_keys[:5]}...")
        if incompatible.unexpected_keys:
            print(f"[Infer] 忽略多余键: {len(incompatible.unexpected_keys)} 个 (来自 thop 分析)")
        print(f"[Infer] 加载检查点: epoch={checkpoint['epoch']}, "
              f"Best F1={checkpoint.get('best_f1', 'N/A')}")
    else:
        print(f"[Infer] 警告: 未加载模型权重 (checkpoint 不存在或未指定)，将使用随机初始化的模型")

    fall_detector.eval()
    if device == "cuda":
        fall_detector = fall_detector.to(device)

    # 模型信息
    total_params = sum(p.numel() for p in fall_detector.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"[Infer] 模型参数: {total_params:,} | 大小: {model_size_mb:.2f} MB (FP32)")

    # ── 创建 YOLO 检测器 ──
    try:
        detector = YOLOPoseDetector(
            input_size=tuple(config.get("detector", {}).get("input_size", [640, 384])),
        )
        detector.load_model()
        print("[Infer] YOLOv8n-Pose 加载成功")
    except Exception as e:
        print(f"[Infer] YOLO 加载失败 ({e})，使用模拟模式")
        from pipeline.detector import YOLOPoseDetectorSim
        detector = YOLOPoseDetectorSim()

    # ── 创建推理管线 ──
    # 从 config 加载规则配置
    rule_cfg = config.get("model", {}).get("rule", {})
    rule_config = RuleConfig(
        torso_angle_threshold=rule_cfg.get("torso_angle_threshold", 0.5),
        torso_angle_duration=rule_cfg.get("torso_angle_duration", 8),
        velocity_peak_threshold=rule_cfg.get("velocity_peak_threshold", 0.02),
        stillness_threshold=rule_cfg.get("stillness_threshold", 0.005),
        stillness_duration=rule_cfg.get("stillness_duration", 15),
        vote_window=rule_cfg.get("vote_window", 10),
        vote_threshold=rule_cfg.get("vote_threshold", 0.6),
        recovery_duration=rule_cfg.get("recovery_duration", 5),
        tcn_prob_threshold=rule_cfg.get("tcn_prob_threshold", 0.5),
        aspect_ratio_change_threshold=rule_cfg.get("aspect_ratio_change_threshold", 0.3),
        fall_memory_frames=rule_cfg.get("fall_memory_frames", 60),
        rules_min_pass=rule_cfg.get("rules_min_pass", 1),
    )
    print(f"[Infer] 规则配置: prob_thresh={rule_config.tcn_prob_threshold}, "
          f"vote_thresh={rule_config.vote_threshold}, "
          f"aspect_thresh={rule_config.aspect_ratio_change_threshold}, "
          f"rules_min_pass={rule_config.rules_min_pass}")

    pipeline = FallDetectionPipeline(
        detector=detector,
        fall_detector=fall_detector,
        sequence_length=config.get("pipeline", {}).get("sequence_length", 32),
        detection_interval=config.get("pipeline", {}).get("detection_interval", 2),
        ir_mode=args.ir,
        auto_lowlight=not args.no_lowlight,
        save_video=True,              # 始终保存视频
        output_dir=args.output,
        rule_config=rule_config,
        camera_id=config.get("pipeline", {}).get("camera_id", "laptop_camera"),
        alarm_trigger_seconds=config.get("pipeline", {}).get("alarm_trigger_seconds", 0.5),
        alarm_hold_seconds=config.get("pipeline", {}).get("alarm_hold_seconds", 30.0),
        alarm_cooldown_frames=config.get("pipeline", {}).get("alarm_cooldown_frames", 30),
    )

    stats = pipeline.get_statistics()
    print(f"[Infer] 管线配置: seq_len={stats['sequence_length']}, "
          f"det_interval={stats['detection_interval']}, "
          f"IR={stats['ir_mode']}")

    # ═══════════════════════════════════════════════════════
    # 1. test/ 目录批量推理（默认模式）
    # ═══════════════════════════════════════════════════════
    if args.test or (not args.video and not args.webcam):
        test_dir = args.test_dir
        video_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
        videos = []
        for ext in video_exts:
            videos.extend(glob.glob(os.path.join(test_dir, ext)))

        if not videos:
            print(f"\n[Error] test/ 目录中没有找到视频文件")
            print(f"  请将测试视频放入 '{test_dir}/' 目录")
            print(f"  或使用 --video 指定单个视频: python infer.py --video your_video.mp4")
            return

        print(f"\n[Infer] 批量处理 {len(videos)} 个视频 (来源: {test_dir}/)...")
        print(f"[Infer] 输出目录: {args.output}/")
        print()

        all_results = []
        for i, video_path in enumerate(videos):
            pipeline.reset()
            print(f"[{i+1}/{len(videos)}] {os.path.basename(video_path)}")
            result = pipeline.process_video(video_path)
            all_results.append(result)

            # 简要打印
            n_alarms = sum(1 for e in result.fall_events)
            print(f"      帧数={result.total_frames}, 事件={n_alarms}, "
                  f"延迟={result.avg_latency_ms:.1f}ms\n")

        # 汇总
        total_frames = sum(r.total_frames for r in all_results)
        total_events = sum(len(r.fall_events) for r in all_results)
        print(f"{'='*60}")
        print(f"批量推理完成！")
        print(f"  视频数: {len(videos)}")
        print(f"  总帧数: {total_frames}")
        print(f"  跌倒事件: {total_events}")
        print(f"  输出目录: {args.output}/")
        print(f"{'='*60}")

    # ═══════════════════════════════════════════════════════
    # 2. 单视频模式
    # ═══════════════════════════════════════════════════════
    elif args.video:
        if not os.path.exists(args.video):
            print(f"[Error] 视频不存在: {args.video}")
            return

        pipeline.reset()
        process_single_video(pipeline, args.video, args.output)

    # ═══════════════════════════════════════════════════════
    # 3. 摄像头实时推理
    # ═══════════════════════════════════════════════════════
    elif args.webcam:
        print("[Infer] 启动摄像头实时推理 (按 'q' 退出)...")
        pipeline.reset()
        for frame_result in pipeline.process_webcam(camera_id=0):
            print(f"\rFrame {frame_result.frame_idx} | "
                  f"Alarms: {sum(frame_result.alarms.values())}", end="")


if __name__ == "__main__":
    main()
