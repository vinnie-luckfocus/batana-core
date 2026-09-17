# session-schema — 挥棒会话数据契约

> 版本：1.0（2026-09-17 定稿）· 状态：稳定版 · 定义方：batana-core · 消费方：batana-gui / batana-pi / batana-web

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
  session_id: "sess_01J8H2K3M4N5P6Q7R8S9T0AB"  # 全局唯一会话 id（ULID 风格：sess_ 前缀 + 26 位 Crockford Base32）
  created_at: "2026-09-17T02:30:00Z"           # ISO 8601，一律 UTC（Z 后缀）；本地时区仅用于展示层换算
  duration_ms: 1850              # 必填：会话总时长（毫秒）；纯 IMU 会话（无视频）同样必须填写
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
  uri: "file:///captures/sess_01J8H2K3M4N5P6Q7R8S9T0AB.mp4"  # 本地或远端引用
  fps: 240
  resolution: [1280, 720]          # [宽, 高]，像素
  duration_ms: 1850
  view: "side"                     # 拍摄视角：side / front / behind / other
  sha256: "9f2c4a1b7e8d3f5a0c6b1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2"  # 可选；hex 小写字符串，带外上传场景填写，用于云端完整性校验

pose2d:                          # 2D 姿态序列（可选；能力 id: pose2d）
  model: "batana-pose-v0.1"        # 使用的姿态模型（对应 models/registry.yaml）
                                   # 取值规则：与 sync-api /models/manifest 的映射为 `{id}-v{major}.{minor}`
                                   # （如 manifest id=batana-pose、version=0.1.0 → batana-pose-v0.1）
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
  cap_id: "04A2"                   # 采集设备 id：batana-cap 广播名 BATANA-CAP-xx 的 xx 部分
                                   # （MAC 后两字节大写十六进制，见 ble-protocol）
  sample_rate_hz: 200
  clock_anchor:                    # 时钟锚点（可选；pro-fusion / max 必填）：
                                   # cap 单调时钟与主机时钟的映射基准，用于视觉-惯性时间融合
    device_ts_us: 1234567890       # 锚点时刻的 cap 设备时钟：拼接回绕后的 64 位单调时钟（微秒），
                                   # 见 ble-protocol 4.2（32 位 µs 时间戳回绕拼接规则）
    host_time_utc: "2026-09-17T02:30:00.000Z"  # 锚点时刻的主机 UTC 时间
    offset_ms: 0                   # 设备时钟换算到会话时间轴的偏移（毫秒）
    drift_ppm: 12.5                # 设备时钟相对主机时钟的漂移率（ppm，用于长时间外推校正）
  samples:                         # 样本值为 SI 单位（见「约束与备注」单位换算节）
    - timestamp_ms: 0
      accel: [0.12, -9.78, 0.45]   # 加速度，m/s²，[x,y,z]
      gyro: [0.01, 0.02, -0.15]    # 角速度，rad/s，[x,y,z]
    # ... 后续样本

calibration:                     # 相机标定（可选；pro-stereo / max 双目档位出现）
  format_version: "1.0"            # 标定数据结构版本，独立于 schema_version 演进
  stereo:
    baseline_mm: 62.0              # 双目基线长度（毫米）
    left:                          # 左目内参
      resolution: [1920, 1080]     # [宽, 高]，像素
      intrinsic: [[1520.4, 0, 960.2], [0, 1519.8, 540.1], [0, 0, 1]]  # 3×3 内参矩阵
      distortion: [0.08, -0.12, 0.001, -0.002, 0.0]  # 畸变系数 [k1,k2,p1,p2,k3]
    right:                         # 右目内参（结构同左目）
      resolution: [1920, 1080]
      intrinsic: [[1521.0, 0, 959.8], [0, 1520.5, 539.7], [0, 0, 1]]
      distortion: [0.07, -0.11, 0.001, -0.001, 0.0]
    extrinsic:                     # 左目 → 右目 外参
      rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]     # 3×3 旋转矩阵
      translation_mm: [62.0, 0, 0]                    # 平移向量（毫米）

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

metrics:                         # 评分指标（上传云端时必填；本地仅采集未分析时可缺失；能力 id: score_basic / bat_speed / ...）
  capabilities: ["pose2d", "phase_split", "score_basic", "bat_speed"]  # 本次实际产出的能力
  bat_speed_mps: 28.4              # 棒速估计（能力 bat_speed；精度随档位见 capabilities.md）
  swing_count: 1                   # 挥棒计数（能力 swing_count）
  tempo_ratio: 2.6                 # 准备/挥棒节奏比
  scores:                          # 分项评分，均为 0-100
    speed: 72
    angle: 65
    coordination: 70
    overall: 69                    # 综合评分
  details: {}                      # 档位增强指标（稳定后由 extensions 转正；字段定义见「扩展指标字段」节）

extensions:                        # 预留扩展位；实验字段先进此处，稳定后升版转正
  metrics_experimental:            # experimental：档位增强指标的正式定义草案（见「扩展指标字段」节）
    kinematic_seq:
      hip_shoulder_separation_deg: 42.5
```

## 关键点枚举（pose2d.keypoints[].name）

固定 33 点，采用 MediaPipe BlazePose 拓扑与索引顺序：

`nose`, `left_eye_inner`, `left_eye`, `left_eye_outer`, `right_eye_inner`, `right_eye`, `right_eye_outer`, `left_ear`, `right_ear`, `mouth_left`, `mouth_right`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_pinky`, `right_pinky`, `left_index`, `right_index`, `left_thumb`, `right_thumb`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`, `left_heel`, `right_heel`, `left_foot_index`, `right_foot_index`

## 扩展指标字段（experimental）

以下字段为档位增强指标的正式可选定义，当前处于 experimental 状态，统一置于 `extensions.metrics_experimental` 下；稳定后升版转正至 `metrics.details`。消费方必须忽略未知字段。

| 字段 | 类型 | 能力 id | 产出档位 | 说明 |
|---|---|---|---|---|
| `bat_traj3d` | object | `bat_traj3d` | pro-fusion（coarse）/ pro-stereo / max | 3D 棒轨迹：`points[]`，每项含 `t_ms`（会话时间轴毫秒）、`pos_m`（棒尖位置，米，会话坐标系 [x,y,z]）、`vel_mps`（速度矢量，m/s） |
| `pose3d` | object | `pose3d` | pro-stereo / max | 全身 3D 运动学：`frames[]`，每项含 `t_ms` 与 `joints`（33 关节 3D 坐标，拓扑同 pose2d，单位米） |
| `impact_point` | object | `impact_point` | pro-stereo（partial）/ max | 击球点/甜区估计：`t_ms`（触球时刻）、`pos_m`（击球点坐标）、`on_sweet_spot`（是否甜区命中） |
| `kinematic_seq` | object | `kinematic_seq` | pro-fusion（partial）/ pro-stereo（partial）/ max | 髋-肩-棒运动链时序；产出含髋肩分离角 `hip_shoulder_separation_deg`（峰值，度）及各环节角速度时序 |
| `biomech_report` | object | `biomech_report` | max | 完整生物力学报告：`uri`（报告文件引用）或内联结构化段落 |

## 约束与备注

- `pose2d` 坐标为归一化值，消费方按 `video.resolution` 还原像素坐标。
- `phases` 的时间轴与 `video` / `imu` 共用同一时间原点（会话起点）；`pro-fusion` / `max` 档位的时间戳已由 `tempo_sync` 能力对齐。
- **时钟域换算**：`imu` 样本的设备时钟（`device_ts_us`，cap 拼接回绕后的 64 位单调时钟，微秒，见 ble-protocol 4.2）经 `imu.clock_anchor` 换算到会话时间轴（毫秒），`clock_anchor.device_ts_us` 与样本 `device_ts_us` 使用同一 64 位时钟域：

  ```
  session_ts_ms = (device_ts_us - clock_anchor.device_ts_us) / 1000 + clock_anchor.offset_ms
  ```

  长时间会话需按 `clock_anchor.drift_ppm` 做漂移校正；`clock_anchor.host_time_utc` 记录锚点的主机 UTC 时间，用于跨设备对时与日志追溯。
- **IMU 单位换算**：BLE 链路上传输的是原始值（加速度 `mg` 量纲 i16；陀螺仪原始 LSB，灵敏度见 ble-protocol 量程配置），接收侧（gui / pi / core runtime）负责换算为 SI 单位（`m/s²`、`rad/s`）后写入本 schema 的 `imu.samples`。换算规则与量程定义见 batana-cap 仓库 `docs/contracts/ble-protocol.md`。
- 所有 ISO 8601 时间戳（`meta.created_at`、`imu.clock_anchor.host_time_utc` 等）一律使用 UTC（`Z` 后缀）；本地时区仅允许在展示层换算，不进入数据契约。
- `metrics.details` / `extensions` 内的字段随档位与能力演进扩充，消费方必须忽略未知字段。
- 本契约已定稿为 v1.0（稳定版）并移交版本管理（见主仓 `docs/versioning.md`）。

## 变更记录

- **1.0-draft（2026-09-17，评审整改）**：新增 `imu.clock_anchor` 时钟锚点与时钟域换算公式；新增可选 `calibration`（双目标定，含 `format_version`）；`meta.duration_ms` 改为必填（覆盖纯 IMU 会话）；补充 IMU 单位换算备注（BLE 原始值 → SI，交叉引用 ble-protocol）；`imu.device_id` 更名为 `imu.cap_id`（对齐 BATANA-CAP-xx 广播名规则）；`session_id` 示例统一为 ULID 风格；时间戳统一注明 UTC；`metrics.details` 注释字段正式化为 `extensions.metrics_experimental` 可选定义，髋肩分离角归入 `kinematic_seq`；版本标注统一为 `1.0-draft`；消费方统一写作 batana-gui。
- **1.0（2026-09-17）：定稿冻结，此后主版本内向后兼容**；终检整改：`metrics` 必填性裁决为「上传云端时必填；本地仅采集未分析时可缺失」（对齐 sync-api）；`video` 新增可选字段 `sha256`（带外上传完整性校验）；`imu.clock_anchor.device_ts_us` 注释明确为拼接回绕后的 64 位单调时钟并交叉引用 ble-protocol 4.2；`pose2d.model` 注明与 sync-api `/models/manifest` 的取值映射规则 `{id}-v{major}.{minor}`。
- 注意：**契约版本变更需同步更新司令塔（batana 主仓）`repos.yaml`** 中的版本兼容矩阵。
