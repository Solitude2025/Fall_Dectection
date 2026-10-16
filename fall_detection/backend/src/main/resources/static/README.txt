打开 http://localhost:8080/ 查看监控台。

启动顺序：
1. 启动 MySQL：在项目根目录执行 docker compose up -d mysql；没有 Docker 时也可以安装 MySQL 8，并创建 fall_detection 数据库和 fall_user 用户。
2. 启动 Spring Boot：从 backend 目录运行 mvn spring-boot:run。Python 服务会加载项目中的 checkpoints/best.pth，并按权重识别模型架构。
3. 启动 Python：在项目根目录运行 python -m services.inference_service。

没有 MySQL 时，可在启动 Spring Boot 前设置 SPRING_PROFILES_ACTIVE=h2，使用本地 H2 文件进行练习。

浏览器中的按钮会启动或停止笔记本内置摄像头。页面报警支持红色视觉提示和浏览器声音；解除按钮会将事件状态改为 cleared。
页面也支持上传视频分析：最大 100 MB、5 分钟、1080p。上传前先停止摄像头。完成后可查看进度、事件时间点及标注回放，报告单独保存在 app_data/video_jobs，不混入实时报警统计。
