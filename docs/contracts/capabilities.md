# capabilities — 能力注册表

> 版本：v1-draft（2026-09-17）· 定义方：batana-core · 消费方：batana（gui）/ batana-web

能力注册表描述"每条管线档位能产出哪些能力"。运行时探测当前已连接的外设，自动选择可用的最高档位；外设热插拔时无缝升降级。GUI 与 Web 查询本注册表动态显示/隐藏功能，UI 只呈现"当前可用能力"。

档位矩阵与产品侧定义以主仓 `docs/model-tiers.md` 为准，本文件为其机器可读的对齐版本。

## 档位定义

| 档位 id | 输入模态 | 典型设备组合 | 延迟目标 |
|---|---|---|---|
| `standard-vision` | `mono` | 仅手机 | 单次分析 ≤ 15s（逐步优化到 ≤ 10s） |
| `standard-imu` | `imu` | 仅 batana-cap | 棒速/计数近实时 ≤ 1s |
| `pro-fusion` | `mono` + `imu` | 手机 + batana-cap | 融合分析 ≤ 8s |
| `pro-stereo` | `stereo` | 仅 batana-pi | 3D 分析 ≤ 8s（边缘端） |
| `max` | `stereo` + `imu` | batana-pi + batana-cap | 全量分析 ≤ 8s（边缘端） |

档位选择优先级（外设齐备时从高到低）：`max` > `pro-stereo` ≈ `pro-fusion` > `standard-vision` / `standard-imu`。

## 支持度标记

| 标记 | 含义 |
|---|---|
| `full` | 完整支持，产出正式指标 |
| `partial` | 部分支持（精度受限或仅子集指标） |
| `coarse` | 粗略估计，仅供参考，不进入正式评分 |
| `rule` | 规则引擎实现（非模型） |
| `rule_data` | 规则 + 数据驱动混合 |
| `model` | 模型生成 |
| `—` | 该档位不支持此能力 |

## 能力清单

| 能力 id | 说明 | standard-vision | standard-imu | pro-fusion | pro-stereo | max |
|---|---|:-:|:-:|:-:|:-:|:-:|
| `pose2d` | 2D 身体关键点（33 点） | full | — | full | full | full |
| `phase_split` | 挥棒阶段分割（准备/加速/击球/收尾） | full | coarse | full | full | full |
| `score_basic` | 速度/角度/协调性基础评分 | full | partial | full | full | full |
| `bat_speed` | 棒速估计 | coarse | full | full（融合校正） | full（3D） | full（精确） |
| `swing_count` | 挥棒计数与节奏 | — | full | full | full | full |
| `tempo_sync` | 视觉-惯性时间戳对齐 | — | — | full | — | full |
| `bat_traj3d` | 3D 棒轨迹重建 | — | — | coarse（单目+IMU 约束） | full | full |
| `kinematic_seq` | 髋-肩-棒运动链时序 | — | — | partial | partial | full |
| `impact_point` | 击球点/击球甜区估计 | — | — | — | partial | full |
| `pose3d` | 全身 3D 运动学 | — | — | — | full | full |
| `biomech_report` | 完整生物力学报告 | — | — | — | — | full |
| `coach_advice` | AI 改进建议 | rule | rule | rule_data | rule_data | model |

## 实现约束

- 每个档位对应 `pipelines/` 下一个独立可测的管线目录；档位间共享 `operators/` 底层算子，不允许复制实现。
- `standard-imu` 的片上预处理（cap 固件内的中断唤醒、计数）只属于设备端，正式评分仍在 runtime。
- 能力 id 一旦发布不允许改名；新增能力追加行即可，语义变更升注册表版本。

## 演进方向

- v2：视频-IMU 联合标定自动化，`pro-fusion` 的 `bat_traj3d` 从 coarse 升级为 full。
- v3：`coach_advice` 由规则引擎切换为生成式模型（云端 batana-core Python 侧）。
