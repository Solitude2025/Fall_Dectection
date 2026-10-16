"""
评估指标工具
--------------
跌倒检测评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1 分数
- 平均检测延迟
- 误报率 (False Alarm Rate)
- 漏报率 (Miss Rate)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    average_precision_score,
)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算分类指标
    
    Args:
        predictions: (N, T) 预测概率
        labels: (N, T) 真实标签 (0/1)
        threshold: 二值化阈值
    
    Returns:
        dict: 各指标值
    """
    # 展平
    pred_flat = predictions.flatten()
    label_flat = labels.flatten()
    
    # 二值化
    pred_binary = (pred_flat >= threshold).astype(int)
    
    # 有效样本（排除 NaN）
    valid = ~np.isnan(label_flat)
    pred_binary = pred_binary[valid]
    label_flat = label_flat[valid]
    pred_flat = pred_flat[valid]
    
    if len(label_flat) == 0:
        return {}
    
    # 基础指标
    acc = accuracy_score(label_flat, pred_binary)
    prec = precision_score(label_flat, pred_binary, zero_division=0)
    rec = recall_score(label_flat, pred_binary, zero_division=0)
    f1 = f1_score(label_flat, pred_binary, zero_division=0)
    
    # 混淆矩阵（处理单类别情况）
    cm = confusion_matrix(label_flat, pred_binary)
    if cm.size == 1:
        # 只有一种类别
        if 1 in label_flat:
            # 全是正样本
            tn, fp, fn, tp = 0, 0, cm.ravel()[0], 0
        else:
            # 全是负样本
            tn, fp, fn, tp = cm.ravel()[0], 0, 0, 0
    else:
        tn, fp, fn, tp = cm.ravel()
    
    # 误报率 (FPR) = FP / (FP + TN)
    fpr = fp / (fp + tn + 1e-8)
    
    # 漏报率 (FNR/Miss Rate) = FN / (FN + TP)
    fnr = fn / (fn + tp + 1e-8)
    
    # AUC
    try:
        auc = roc_auc_score(label_flat, pred_flat)
    except:
        auc = 0.0
    
    # AP
    try:
        ap = average_precision_score(label_flat, pred_flat)
    except:
        ap = 0.0
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "miss_rate": fnr,
        "auc_roc": auc,
        "average_precision": ap,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def compute_event_metrics(
    predicted_events: List[Tuple[int, int]],    # [(start, end), ...]
    ground_truth_events: List[Tuple[int, int]],  # [(start, end), ...]
    total_frames: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    计算事件级别指标（按跌倒事件匹配）
    
    Args:
        predicted_events: 预测的跌倒事件 (start_frame, end_frame)
        ground_truth_events: 真实的跌倒事件
        total_frames: 总帧数
        iou_threshold: 事件匹配 IoU 阈值
    
    Returns:
        dict: 事件级别指标
    """
    # 将事件转为帧级别的集合
    def events_to_frame_set(events):
        frames = set()
        for start, end in events:
            frames.update(range(start, end + 1))
        return frames
    
    # 计算每个预测事件与真实事件的 IoU
    matched_pred = set()
    matched_gt = set()
    
    for pi, (ps, pe) in enumerate(predicted_events):
        pred_frames = set(range(ps, pe + 1))
        best_iou = 0
        best_gt = -1
        
        for gi, (gs, ge) in enumerate(ground_truth_events):
            if gi in matched_gt:
                continue
            gt_frames = set(range(gs, ge + 1))
            intersection = len(pred_frames & gt_frames)
            union = len(pred_frames | gt_frames)
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        
        if best_iou >= iou_threshold and best_gt >= 0:
            matched_pred.add(pi)
            matched_gt.add(best_gt)
    
    tp = len(matched_pred)
    fp = len(predicted_events) - tp
    fn = len(ground_truth_events) - len(matched_gt)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def compute_latency_metrics(
    frame_times: List[float],
) -> Dict[str, float]:
    """
    计算延迟指标
    
    Args:
        frame_times: 每帧推理时间（秒）
    
    Returns:
        dict: 延迟指标（毫秒）
    """
    if not frame_times:
        return {}
    
    times = np.array(frame_times) * 1000  # 转毫秒
    
    return {
        "mean_latency_ms": float(np.mean(times)),
        "median_latency_ms": float(np.median(times)),
        "min_latency_ms": float(np.min(times)),
        "max_latency_ms": float(np.max(times)),
        "p95_latency_ms": float(np.percentile(times, 95)),
        "p99_latency_ms": float(np.percentile(times, 99)),
        "fps": float(1.0 / np.mean(frame_times)) if np.mean(frame_times) > 0 else 0,
    }


def compute_model_efficiency(
    model: torch.nn.Module,
    input_shape: Tuple = (1, 32, 48),
) -> Dict[str, float]:
    """
    计算模型效率指标
    
    Args:
        model: PyTorch 模型
        input_shape: 输入形状
    
    Returns:
        dict: 效率指标
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 模型大小 (MB, FP32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    # 计算 FLOPs (可选，需要 thop)
    # thop 2.0.20 + Python 3.12 存在兼容性 bug（hooks残留在 Conv1d 上导致训练崩溃）
    # 因此默认禁用，仅当明确需要时手动调用
    flops_m = 0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb_fp32": model_size_mb,
        "model_size_mb_int8": model_size_mb / 4,  # INT8 量化估算
        "flops_m": flops_m,
    }
