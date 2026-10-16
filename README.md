2026 HUAWEI-WHU校企联合大赛参赛作品，赛后进行了项目的进一步完善
# Fall Detection

轻量级时空语义跌倒检测系统，面向本机视频、摄像头和端侧部署场景。项目将人体姿态、运动特征、时序模型和物理规则结合起来，提供单视频检测、摄像头监测、浏览器监控台、报警事件、视频回放，以及训练、评估和 ONNX 导出工具。

> 本项目是研究与演示系统。检测结果仅用于辅助复核，不能替代人工看护、医疗判断或安全关键系统。

## 快速体验

### 环境要求

- Windows、Linux 或 macOS
- Python 3.10 或 3.11、PyTorch 2.0+
- 仅运行 Python 推理不需要 Java、Maven、MySQL 或 Docker
- 推荐 NVIDIA GPU；没有 GPU 时可使用 CPU

### 安装

```bash
git clone https://github.com/Solitude2025/Fall_Dectection.git
cd Fall_Dectection/fall_detection
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

项目包含时序分类权重 `checkpoints/best.pth`。首次运行时，Ultralytics 会自动下载 `yolov8n-pose.pt`（摄像头）或 `yolo11m-pose.pt`（上传视频）。第三方权重许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

### 单视频检测

```bash
python infer.py --video path/to/video.mp4 --checkpoint checkpoints/best.pth
```

结果默认写入 `outputs/`。批量检测本地 `test/` 目录：

```bash
python infer.py --test --checkpoint checkpoints/best.pth
```

红外/低光模式：

```bash
python infer.py --video path/to/video.mp4 --ir --checkpoint checkpoints/best.pth
python infer.py --video path/to/video.mp4 --no_lowlight --checkpoint checkpoints/best.pth
```

摄像头实时检测：

```bash
python infer.py --webcam --checkpoint checkpoints/best.pth
```

没有 CUDA 时可使用 `--device cpu`。模型、阈值、检测间隔和摄像头参数位于 `configs/config.yaml`。

## 完整网页应用

```text
浏览器监控台 :8080
        │
        ▼
Spring Boot 业务服务 :8080 ─── MySQL 或 H2
        │
        ▼
Python 推理服务 :8001 ─── 摄像头 / 视频文件
```

安装 Python、JDK 17 和 Maven 后，在项目根目录运行：

```powershell
.\start.bat
```

浏览器打开 <http://localhost:8080>。停止服务：

```powershell
.\stop.ps1
```

没有 MySQL 时可手动使用 H2：

```powershell
$env:SPRING_PROFILES_ACTIVE = "h2"
cd backend
mvn spring-boot:run
```

页面支持摄像头监测、报警确认、事件查询、视频上传和标注回放；上传视频限制为 100 MB、5 分钟和 1080p。更多说明见 [APP_GUIDE.md](APP_GUIDE.md) 与 [backend/README.md](backend/README.md)。

使用 MySQL 时不要把真实密码写入 Git：

```powershell
Copy-Item .env.example .env
docker compose up -d mysql
```

也可以设置 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD` 和 `MYSQL_ROOT_PASSWORD`。

## 算法流程

```text
视频帧 → Resize/低光增强/红外预处理
       → YOLO Pose（人体框 + 17 个关键点）
       → ByteTrack 跨帧跟踪
       → 关键点完整性门控
       → Motion Feature Encoder（48 维）
       → Light-TCN（32 帧时序建模）
       → Rule Refinement（角度、速度、静止、多帧投票）
       → 跌倒事件、标注视频和报警记录
```

核心设计包括运动学特征压缩、膨胀因果卷积、关键点完整性门控、物理规则校正，以及低光、遮挡和视角变化增强。

## 项目结构

```text
fall_detection/
├── models/                  # Motion Encoder、Light-TCN、规则校正和完整模型
├── pipeline/                # 检测、跟踪和推理流程
├── services/                # Python 推理、视频分析和摄像头服务
├── backend/                 # Spring Boot 事件服务和网页静态资源
├── data/                    # 数据集加载与增强代码
├── configs/config.yaml      # 模型、检测器、报警和部署配置
├── checkpoints/best.pth    # 本项目时序分类权重
├── tests/                   # Python 单元测试
├── scripts/                 # 回归、摄像头和服务测试脚本
├── prepare_data.py          # 数据准备
├── train.py                 # 训练
├── evaluate.py              # 评估与报告
├── infer.py                 # 视频/摄像头推理入口
├── export.py                # ONNX/TorchScript 导出
├── APP_GUIDE.md             # 网页应用详细说明
└── THIRD_PARTY_NOTICES.md   # 第三方模型与数据集说明
```

视频、原始数据、日志、数据库、缓存和推理输出默认被 `.gitignore` 忽略。请只使用你有权处理的素材。

## 训练与评估

OmniFall 不随仓库分发。请从官方来源下载并遵守其许可证：

```bash
python prepare_data.py --all --max_videos 2000
python train.py --data_dir data/train --val_dir data/val --epochs 50 --batch_size 32 --device cpu
python tune_threshold.py --checkpoint checkpoints/best.pth
python evaluate.py --checkpoint checkpoints/best.pth
python eval_longtail.py --checkpoint checkpoints/best.pth
```

导出部署模型：

```bash
python export.py --checkpoint checkpoints/best.pth --format onnx
```

## 参考结果

以下结果来自当前版本内部评测，不代表所有视频或真实养老场景的保证：

| 评测 | 结果 |
|---|---:|
| OmniFall 帧级 Recall | 0.737 |
| OmniFall 帧级 AUC-ROC | 0.819 |
| OmniFall 帧级 F1 | 0.627 |
| UR Fall 跌倒视频检出 | 10/10 |
| UR Fall ADL 无误报 | 8/10 |
| TCN 参数量 | 67,329 |
| TCN FP32 大小 | 约 0.26 MB |

测试集规模、视频来源和阈值会显著影响结果，请在自己的摄像头、视角、光照和人群上建立独立测试集。

## 已知限制

- 该系统不能保证检测到所有跌倒，也不能单独作为紧急救援依据。

## 隐私与安全

- 不要提交摄像头录像、包含人脸或个人信息的视频、生产数据库、日志、真实密码或 API 密钥。
- 使用 `.env.example` 创建本地配置，不要提交 `.env`。
- 对外部署前请增加鉴权、HTTPS、访问控制、数据保留期限和审计日志。
- 数据集、预训练权重和依赖库各自适用其许可证；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 测试

```bash
python -m pytest tests -q
```

## 许可证

项目源代码采用 [MIT License](LICENSE)。第三方数据集、预训练模型和依赖库不自动继承本项目许可证。

## 致谢

感谢 OmniFall、UR Fall、Ultralytics、PyTorch、OpenCV、ByteTrack、FastAPI 和 Spring Boot 社区提供的工具、数据或研究基础。
