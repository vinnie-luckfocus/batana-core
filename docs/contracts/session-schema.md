# session-schema — 挥棒会话数据契约

> 版本：v1-draft（2026-09-17）· 定义方：batana-core · 消费方：batana（gui）/ batana-pi / batana-web

一次挥棒会话（session）是 batana 生态内传递分析结果的最小完整单元。本文件定义其结构化字段，所有仓库间的会话数据交换必须遵循本 schema。

## 设计原则

- **自描述**：schema 版本与产生它的管线档位写入会话本身，消费方无需外部上下文即可解析。
- **模态可选**：视频、姿态、IMU 等字段按档位可选出现；缺失表示该模态未采集，而非异常。
- **只增不改**：已发布字段不允许改名或改语义，新增字段必须为可选；破坏性变更升 schema 主版本。

## 顶层结构

```yaml
# session v1 示例（standard-vision 档位：仅单目视频，无 IMU）
schema_version: "1.0"            # 数据契约版本（语义化版本）

meta:                            # 会话元数据
  session_id: "sess_20260917_0001"   # 全局唯一会话 id
  created_at: "2026-09-17T10:30:00+08:00"  # ISO 8601，含时区
  device:                          # 采集设备信息
    model: "iPhone 16 Pro"
    os: "iOS 19.0"
  app:                             # 产生会话的应用
    name: "batana"
    version: "0.1.0"
  pipeline: "standard-vision"      # 处理本会话的管线档位（见 capabilities.md）
  handedness: "right"              # 打击手惯用手：right / left
  subject:                         # 受试者（可选，匿名化）
    height_cm: 175
    age_group: "adult"             # adult / teen / kid

video:                           # 视频引用（可选；standard-imu 档位无此字段）
  uri: "file:///captures/sess_20260917_0001.mp4"  # 本地或远端引用
  fps: 240
  resolution: [1280, 720]          # [宽, 高]，像素
  duration_ms: 1850
  view: "side"                     # 拍摄视角：side / front / behind / other

pose2d:                          # 2D 姿态序列（可选；能力 id: pose2d）
  model: "batana-pose-v0.1"        # 使用的姿态模型（对应 models/registry.yaml）
  frame_rate: 240                  # 与视频帧率一致
  frames:
    - frame_index: 0
      timestamp_ms: 0              # 相对会话起点的毫秒时间戳
      confidence: 0.92             # 整帧检测置信度 [0,1]
      keypoints:                   # 33 点（MediaPipe BlazePose 拓扑，顺序固定）
        - name: "nose"             # 关键点名称（枚举见下）
          x: 0.512                 # 归一化坐标 [0,1]，原点左上
          y: 0.238
          visibility: 0.98         # 可见性置信度 [0,1]
        # ... 共 33 项，缺失点 visibility=0
    # ... 后续帧

imu:                             # IMU 序列（可选；standard-imu / pro-fusion / max 档位出现）
  device_id: "cap_04a2"            # 采集设备 id（batana-cap）
  sample_rate_hz: 200
  samples:
    - timestamp_ms: 0
      accel: [0.12, -9.78, 0.45]   # 加速度，m/s²，[x,y,z]
      gyro: [0.01, 0.02, -0.15]    # 角速度，rad/s，[x,y,z]
    # ... 后续样本

phases:                          # 阶段分割（可选；能力 id: phase_split）
  granularity: "fine"              # fine（4 阶段）/ coarse（IMU 粗分割）
  segments:
    - phase: "stance"              # 准备
      start_ms: 0
      end_ms: 620
    - phase: "load_stride"         # 加速（重心加载与跨步）
      start_ms: 620
      end_ms: 980
    - phase: "swing_contact"       # 击球（挥棒至触球）
      start_ms: 980
      end_ms: 1180
    - phase: "follow_through"      # 收尾
      start_ms: 1180
      end_ms: 1850

metrics:                         # 评分指标（可选；能力 id: score_basic / bat_speed / ...）
  capabilities: ["pose2d", "phase_split", "score_basic", "bat_speed"]  # 本次实际产出的能力
  bat_speed_mps: 28.4              # 棒速估计（能力 bat_speed；精度随档位见 capabilities.md）
  swing_count: 1                   # 挥棒计数（能力 swing_count）
  tempo_ratio: 2.6                 # 准备/挥棒节奏比
  scores:                          # 分项评分，均为 0-100
    speed: 72
    angle: 65
    coordination: 70
    overall: 69                    # 综合评分
  details:                         # 档位增强指标（仅对应档位产出）
    # pro-stereo / max：bat_traj3d、pose3d、impact_point 等
    # max：kinematic_seq、biomech_report

extensions: {}                     # 预留扩展位；实验字段先进此处，稳定后升版转正
```

## 关键点枚举（pose2d.keypoints[].name）

固定 33 点，采用 MediaPipe BlazePose 拓扑与索引顺序：

`nose`, `left_eye_inner`, `left_eye`, `left_eye_outer`, `right_eye_inner`, `right_eye`, `right_eye_outer`, `left_ear`, `right_ear`, `mouth_left`, `mouth_right`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_pinky`, `right_pinky`, `left_index`, `right_index`, `left_thumb`, `right_thumb`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`

## 约束与备注

- `pose2d` 坐标为归一化值，消费方按 `video.resolution` 还原像素坐标。
- `phases` 的时间轴与 `video` / `imu` 共用同一时间原点（会话起点）；`pro-fusion` / `max` 档位的时间戳已由 `tempo_sync` 能力对齐。
- `metrics.details` 内的字段随档位与能力演进扩充，消费方必须忽略未知字段。
- 本 draft 稳定后将固化为 v1.0 并移交版本管理（见主仓 `docs/versioning.md`）。
