# Spring Boot 后端

这个服务只做业务层工作：保存 Python 推理服务发来的跌倒事件，为前端提供事件查询、统计、摄像头启动/停止和人工解除接口。

## 接口

- `POST /api/internal/events`：Python 推送新建或解除后的事件
- `GET /api/events?status=active&limit=50`：查询事件
- `GET /api/events/{eventId}`：查看单条事件
- `POST /api/events/{eventId}/clear?reason=manual`：人工解除
- `GET /api/events/stats?days=7`：统计数据和按日趋势
- `GET/POST /api/camera/status|start|stop`：控制笔记本摄像头进程
- `GET /api/camera/stream-url`：取得前端使用的预览地址
- `GET /api/camera/stream`：Spring Boot 转发 Python 的 MJPEG 画面
- `POST /api/videos`：上传视频到 Python 异步分析（multipart 字段 `file`，不超过 100 MB）
- `GET /api/videos`、`GET /api/videos/{jobId}`：查询报告列表、任务进度与事件时间点
- `GET /api/videos/{jobId}/result`：带标注的 MP4 回放，透传 Range/Content-Range 支持拖动

页面支持设备监测和视频上传两个模式，一次只运行一种推理任务。视频限制为 5 分钟、1080p；报告与回放保存在项目 app_data/video_jobs，上传文件和中间视频处理后删除。视频报告不会混入实时报警数据与统计。

## 数据库

默认连接 MySQL，连接参数可以用 `MYSQL_HOST`、`MYSQL_PORT`、`MYSQL_DATABASE`、`MYSQL_USER`、`MYSQL_PASSWORD` 覆盖。项目根目录提供了 `docker-compose.yml`，第一次运行可以执行 `docker compose up -d mysql` 自动启动数据库。

如果暂时没有 MySQL，可以用 `SPRING_PROFILES_ACTIVE=h2` 使用文件型 H2；这只是学习备用模式，正式项目数据仍然放在 MySQL。

项目需要 JDK 17 和 Maven。当前代码把表结构交给 JPA 自动创建，学习阶段无需先写复杂 SQL。
