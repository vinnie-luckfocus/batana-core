<p align="center">
  <img src="https://raw.githubusercontent.com/vinnie-luckfocus/batana/main/assets/logo.png" width="150" alt="batana logo">
</p>

# batana-core — 模型系统核心

batana 是棒球打击动作**捕捉 / 分析 / 评价**生态系统，本仓库是整个生态的模型与算法中枢：打击动作的捕捉、追踪、分析、评价全部在此定义与实现。

## 定位

对下屏蔽模型框架差异（TFLite / CoreML / ONNX / NPU），对上以**能力注册表**与**统一数据 schema** 服务 batana（gui）、batana-pi、batana-web。

## 边界

- ✅ 做：模型、算法、数据 schema、能力注册表、推理运行时、训练流水线
- ❌ 不做：任何 UI；任何硬件驱动（摄像头 / IMU 数据源以接口注入）；云端存储与服务端逻辑

## 技术栈

- 训练与研究：Python 3.11+ / PyTorch
- 推理运行时：C++17 / CMake（iOS / Android / macOS / Linux-arm64）
- 推理后端：TFLite、CoreML、ONNX Runtime、RKNN

## 目录结构

```
batana-core/
├── runtime/            # C++ 推理运行时 + FFI 绑定
│   ├── include/        #   公共头文件（capabilities.h、sources.h）
│   ├── backends/       #   tflite / coreml / onnx / rknn
│   └── bindings/       #   flutter FFI、python pybind
├── pipeline/           # 管线骨架（capture → preprocess → infer → score 阶段抽象 + 规则评分 stub）
├── pipelines/          # 分级管线（每个档位一个目录）
├── operators/          # 共享算子
├── tools/
│   ├── calibration/    #   ChArUco 标定板生成、双目标定 CLI
│   ├── recon3d/        #   双目三角化、网格精度评估（V0c ≤15mm@2m 判定）
│   └── validate_session.py  # session JSON 契约校验器
├── training/           # Python 训练/评估/导出
├── tests/              # pytest：合成数据端到端验证（无需真实相机）
├── models/registry.yaml# 模型工件注册表
└── docs/contracts/     # session-schema.md、capabilities.md
```

## 标定与 3D 工具链（M0 G0 / V0c）

双目模组输出 2560×800 side-by-side 整帧（左右各 1280×800），工具链自动切半；也支持 left/right 分图或 `--left-dir/--right-dir`。

```bash
pip install -e ".[dev]"                       # 建议 venv；依赖 opencv-python / numpy / pyyaml / pytest

# 1. 生成 ChArUco 标定板（默认 11×8 格、格子 100mm、A0 300dpi）
python -m tools.calibration.gen_board --out charuco_11x8_100mm.png

# 2. 双目标定：单目 + 立体标定，报告重投影误差，输出对齐契约 calibration 字段
python -m tools.calibration.calibrate_stereo frames/ --out-json calibration.json --out-yaml calibration.yaml

# 3. 三角化：左右目 2D 对应点 → 3D 点（左目坐标系，mm）
python -m tools.recon3d.triangulate --calib calibration.json --points pairs.csv --out points3d.json

# 4. 精度评估：网格模式或标定板帧回放比对，判定 ≤15mm 通过标准（退出码 0/1）
python -m tools.recon3d.eval_grid --calib calibration.json --left left.png --right right.png
python -m tools.recon3d.eval_grid --points3d points3d.json --grid-shape 7x10 --spacing-mm 100

# session 契约校验（上传云端场景加 --require-metrics）
python -m tools.validate_session session.json
```

外参约定：`extrinsic.rotation` 为左目→右目点旋转（OpenCV `stereoCalibrate` 的 R）；`extrinsic.translation_mm` 为右目光心在左目坐标系中的位置（`-Rᵀ·T_cv`，水平模组下 ≈ `[+baseline, 0, 0]`，与契约示例一致）；OpenCV 点变换由 `T_cv = -R @ translation_mm` 还原。

## 测试

```bash
pytest                 # 35 项：合成双目端到端（标定→三角化→回放比对）+ 契约校验正反例 + 管线骨架
```

合成验证不依赖真实相机：程序生成已知内外参的虚拟双目（基线 100mm、带畸变与噪声），渲染虚拟 ChArUco 板跑全链路。实测：基线恢复误差 <1mm，重投影误差 ~0.09px，2m 处 3D 重建点残差 mean ≈3mm（V0c 标准 ≤15mm@2m 余量 >2×）。


## 对外契约

- [`docs/contracts/session-schema.md`](docs/contracts/session-schema.md)：一次挥棒会话的完整结构化定义（v1.0，稳定版）
- [`docs/contracts/capabilities.md`](docs/contracts/capabilities.md)：能力注册表（v1.0，稳定版），按档位声明能力支持情况
- FFI API：`batana_runtime_create(config) → session_analyze(input) → SessionResult`

## 路线图

- 模块规格：[主仓 docs/modules/batana-core.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/modules/batana-core.md)
- 能力分级矩阵：[主仓 docs/model-tiers.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/model-tiers.md)
- 生态路线图：[主仓 docs/roadmap.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/roadmap.md)
- 版本管理：[主仓 docs/versioning.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/versioning.md)

里程碑映射：P1 v0.1（standard-vision）→ P2 v0.2（IMU + 融合）→ P3 v0.3（双目 max）→ P5 coach_advice 模型化。

## 变更记录

- **2026-09-19**：新增标定与 3D 工具链（`tools/calibration`：ChArUco 板生成 + 双目标定 CLI，输出对齐契约 `calibration` 字段；`tools/recon3d`：三角化 + V0c 精度评估）、管线骨架（`pipeline/`：capture → preprocess → infer → score 阶段抽象、推理接口桩、规则评分 stub）、session 契约校验器（`tools/validate_session.py`）；新增合成数据端到端测试（35 项 pytest，无需真实相机）。

## License

MIT，见 [LICENSE](LICENSE)。
