# 轻量级时空语义跌倒检测框架

> **华为专项赛道：面向低算力端侧平台基于视觉的实时跌倒检测**

---

## 方案概述

> 应用版接入说明见 [APP_GUIDE.md](APP_GUIDE.md)。当前版本在原有端侧模型外增加了本机摄像头推理服务、Spring Boot + MySQL 事件服务和浏览器监控台。

本方案不采用传统的"YOLOv8n + TCN"直接分类跌倒的方式，而是设计了一个**完整的轻量级时空语义跌倒检测框架**，核心创新围绕三个模块展开：

| 模块 | 作用 | 创新点 |
|------|------|--------|
| **Motion Feature Encoder** | 将人体关键点序列编码为 48 维运动语义特征（含 FOOSH 手部支撑先验 6 维） | 替代高维 CNN Feature（640-1024维），TCN 计算量暴跌 |
| **Light-TCN** | 膨胀因果卷积实现轻量时序建模（32 帧输入 / 63 帧感受野） | 全卷积结构，天然支持 NPU INT8 量化，部署优势远超 GRU |
| **Rule Refinement** | 物理约束后处理，时空一致性校正 | 几乎零计算量，大幅降低"坐下/弯腰/躺卧"误报 |

### 完整链路

```
视频输入(RGB/红外)
    │
    ▼
图像预处理(Resize, 亮度自适应 CLAHE/Gamma)
    │
    ▼
YOLOv8n-Pose (人体检测+17关键点)
    │
    ▼
ByteTrack (人体ID连续跟踪)
    │
    ▼
完整性门控 (可见关键点<50%的帧不进入时序缓冲，防半入境误判)
    │
    ▼
Motion Feature Encoder (48维运动特征)
├── 人体中心速度       ├── 关键点速度
├── 人体框宽高比       ├── 躯干倾角
├── 人体面积变化       ├── 重心变化
└── 手部支撑特征       └── 手腕速度/贴地度/支撑分
    │
    ▼
Light-TCN (32帧时序建模, 63帧感受野)
    │
    ▼
Rule Refinement (规则校正)
├── 角度持续时间判断   ├── 速度峰值判断
├── 人体静止持续时间   └── 连续多帧投票
    │
    ▼
报警 / 视频记录 / 日志
```

---

## 实测性能（OmniFall 2000 视频训练，best@epoch18）

### 帧级指标（200 个测试视频，44800+ 帧）

| 指标 | 数值 |
|------|------|
| Recall（跌倒召回） | 0.737 |
| AUC-ROC | 0.819 |
| F1 | 0.627 |
| FPR | 0.255 |

### 事件级（真实视频，UR Fall + 自测）

| 场景 | 结果 |
|------|------|
| RGB 正常光（2 跌倒 + 2 ADL） | **4/4**：跌倒全检出，ADL 零误报 |
| **UR Fall 全量 20 段（10 跌倒 + 10 ADL）** | **跌倒检出 10/10（100%），ADL 8/10 无误报**（adl-04/05 躺卧类边缘误报 0.37–0.40），平均延迟 ~15ms |
| 微光（自动低光增强） | 4/4 |
| 遮挡（Random Erasing） | 4/4 |
| 老人缓慢坐地（非跌倒） | 正确不报警 |
| 红外风格（风格迁移模拟） | 事件级 0/2，**已知边界**（模型有响应 raw 0.51，规则门保守拦截，需真实红外数据标定） |

## 赛题合规

| 指标 | 限制 | 本方案 | 状态 |
|------|------|--------|------|
| 模型总参数 | ≤ 20M | 67,329 (TCN，不含 YOLO 3.3M) | ✓ |
| FP32 大小 | ≤ 80MB | 0.26MB | ✓ |
| 端侧推理延迟 | ≤ 100ms | CPU 实测 13-35ms/帧 | ✓ |
| 纯视觉方案 | 是 | ✓ | ✓ |
| 可见光+红外 | 支持 | IR 预处理+增强+评测体系（事件级检出为已知边界） | ⚠ |
| NPU 内存 | ≤ 20MB | 0.26MB（INT8 后 ~0.07MB） | ✓ |
| 相机输入 | 1080P+ | ✓（内部降采样 640×384） | ✓ |

---

## 项目结构

```
fall_detection/
├── models/                     # 核心模型
│   ├── motion_encoder.py       # 运动特征编码器（48维，含手部支撑特征）
│   ├── light_tcn.py            # 轻量时序卷积网络
│   ├── rule_refinement.py      # 时空一致性规则校正
│   ├── transformer_tcn.py      # 备选 Transformer-TCN（large 版）
│   └── fall_detector.py        # 完整跌倒检测模型
├── pipeline/                   # 推理管线
│   ├── detector.py             # YOLOv8n-Pose 封装（IR/低光自适应预处理）
│   ├── tracker.py              # ByteTrack 封装（IoU 匹配已修复标定）
│   └── inference.py            # 完整推理管线（完整性门控+概率平滑）
├── data/                       # 数据处理
│   ├── augment.py              # 长尾增强体系（截断/整帧丢失/IR退化/红外风格化等）
│   └── dataset.py              # 数据集加载器（难负样本加权采样）
├── utils/                      # 工具（评估指标、可视化）
├── configs/config.yaml         # 全部阈值配置
├── prepare_data.py             # 数据准备（下载→AV1解码→提关键点→分层抽样→分流）
├── train.py                    # 训练
├── evaluate.py                 # 评估报告（训练曲线/混淆矩阵）
├── tune_threshold.py           # 验证集阈值调优
├── eval_longtail.py            # 长尾场景评测（微光/红外/遮挡变体对比）
├── infer.py                    # 推理（批量/单视频/摄像头，输出标注视频）
├── export.py                   # 模型导出 ONNX/TorchScript
├── checkpoints/                # 模型权重（best.pth 为封版）
├── outputs/                    # 推理标注视频
├── eval_report/                # 评估图表与长尾对比表
├── export/                     # 导出产物
└── logs/                       # 运行日志
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 国内加速：pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 2. 数据准备（OmniFall）

```bash
# 完整流程：下载 -> 提取关键点 -> 80/10/10 分流
# 注意：OmniFall 视频为 AV1 编码，脚本内置 imageio-ffmpeg 回退解码
# 默认分层抽样 500 个视频（10 类动作配额，覆盖多视角），-1 为全部 12000 个
python prepare_data.py --all --max_videos 2000

# 分步执行
python prepare_data.py --download              # 只下载
python prepare_data.py --extract --max_videos 2000   # 只提取关键点
python prepare_data.py --split                 # 只分流
```

### 3. 训练

```bash
python train.py --data_dir data/train --val_dir data/val --epochs 50 --batch_size 32 --device cpu

# 训练后在验证集上调优阈值
python tune_threshold.py --checkpoint checkpoints/best.pth

# 生成评估图表
python evaluate.py --checkpoint checkpoints/best.pth
```

### 4. 推理测试

```bash
# test/ 目录批量推理（保存标注视频 + 事件 JSON 到 outputs/）
python infer.py --test --checkpoint checkpoints/best.pth

# 单视频
python infer.py --video your_video.mp4 --checkpoint checkpoints/best.pth

# 红外模式 / 关闭自动低光增强
python infer.py --test --ir --checkpoint checkpoints/best.pth
python infer.py --test --no_lowlight --checkpoint checkpoints/best.pth

# 摄像头实时
python infer.py --webcam --checkpoint checkpoints/best.pth
```

### 5. 长尾场景评测

```bash
# 对 test/ 视频生成微光/红外/遮挡变体并逐场景对比
python eval_longtail.py --checkpoint checkpoints/best.pth
# 结果：eval_report/longtail/longtail_summary.{md,json}
```

### 6. NPU 部署导出

```bash
python export.py --checkpoint checkpoints/best.pth --format onnx

# 华为 Ascend NPU 部署
atc --model=fall_detector_standard.onnx \
    --framework=5 \
    --output=fall_detector \
    --soc_version=Ascend310 \
    --input_shape="motion_features:1,32,48" \
    --input_format=ND
```

---

## 三大痛点针对性优化

### 痛点 1：动作语义混淆

**问题**：快速坐下、弯腰、躺下与跌倒视觉表征相似，YOLO 直接分类误报频发。

**解决方案**：
- **Motion Feature Encoder**：不做 RGB → 跌倒，而是提取人体运动学特征
- **手部支撑先验（FOOSH）**：保护性跌倒中手先于身体落地并保持支撑，
  提取手腕速度/贴地度/支撑分作为软特征供 TCN 学习时序先后关系
- **难负样本加权**：躯干角变化大的负样本（坐下/躺下）采样权重最高 4 倍
- **Rule Refinement**：躯干角度持续判定 + 速度峰值检查 + 多帧投票

### 痛点 2：端侧算力瓶颈

**问题**：3D-CNN / Video Transformer 推理延迟高，无法满足实时性。

**解决方案**：
- **Motion Feature Encoder**：将 640-1024 维 CNN Feature 压缩到 48 维
- **Light-TCN**：67K 参数、2.2 MFLOPs，全卷积结构，天然支持 NPU INT8 量化
- **隔帧检测优化**：YOLO 隔 2 帧运行，TCN 每帧运行
- **分辨率自适应**：1080P → 640×384 推理

### 痛点 3：长尾场景泛化差

**问题**：低光/红外/遮挡/半入境/视角变化场景下精度下降。

**解决方案**：
- **关键点级架构**：TCN 不接触像素，光照/红外天然不影响分类器
- **训练增强体系**：半入境截断、burst 整帧检测丢失、IR 结构化退化、
  红外风格迁移、Random Erasing、透视/运动模糊
- **推理自适应**：亮度 <60 自动 gamma+CLAHE（亮帧零开销）；
  关键点完整性门控（可见关键点 <50% 的帧不进入时序缓冲，防半入境误判）
- **多视角数据**：训练集按相机俯仰角/方位角/距离分层抽样

---

## 数据准备

### 参考数据集

- [OmniFall (HuggingFace)](https://huggingface.co/datasets/simplexsigil2/omnifall)
  ——本项目训练集来源（合成子集，label 1=fall / 2=fallen 视为正样本）
- [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/ds.html)
  ——`test/` 内 4 段真实测试视频来源

### 数据格式

```
data/
├── train/
│   ├── 0000/
│   │   └── person_0/
│   │       ├── keypoints.npy    # (T, 17, 3) 归一化关键点 [x, y, conf]
│   │       ├── bboxes.npy       # (T, 4)     归一化边界框
│   │       └── labels.npy       # (T,)       帧级标签 0/1
│   └── ...
├── val/
└── test/
```

---

## 模型配置

### Standard (推荐，当前封版)
- Motion Feature: 48 维
- TCN: 5层, hidden=64, dilations=[1,2,4,8,16]
- 参数: 67,329

### Light (极致轻量)
- TCN: 4层, hidden=32, depthwise separable
- 参数: ~20K

### Large (高精度，Transformer-TCN)
- Motion Feature: 64 维
- 参数: ~200K

---

## 参考文献

1. LFD-YOLO: a lightweight fall detection network with enhanced feature extraction and fusion
2. BMR-YOLO: A deep learning approach for fall detection in complex environments
3. YOLO-fall: a YOLO-based fall detection model with high precision, shrunk size, and low latency
4. ByteTrack: Multi-Object Tracking by Associating Every Detection Box
5. Temporal Convolutional Networks: A Unified Approach to Action Segmentation
