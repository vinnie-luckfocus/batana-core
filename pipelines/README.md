# pipelines — 分级分析管线

按模态分级的分析管线，**每个档位一个目录**，档位之间共享 `operators/` 底层算子，不允许复制实现：

- `standard-vision/`：仅单目视频（仅手机），单次分析 ≤ 15s
- `standard-imu/`：仅 IMU（仅 batana-cap），棒速/计数近实时 ≤ 1s
- `pro-fusion/`：单目视频 + IMU（手机 + batana-cap），融合分析 ≤ 8s
- `pro-stereo/`：仅双目视频（仅 batana-pi），3D 分析 ≤ 8s（边缘端）
- `max/`：双目视频 + IMU（batana-pi + batana-cap），全量分析 ≤ 8s（边缘端）

每条管线声明所需输入模态（`stereo` / `mono` / `imu`）与产出的能力 id，详见 `docs/contracts/capabilities.md`。
