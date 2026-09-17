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
├── pipelines/          # 分级管线（每个档位一个目录）
├── operators/          # 共享算子
├── training/           # Python 训练/评估/导出
├── models/registry.yaml# 模型工件注册表
└── docs/contracts/     # session-schema.md、capabilities.md
```

## 对外契约

- [`docs/contracts/session-schema.md`](docs/contracts/session-schema.md)：一次挥棒会话的完整结构化定义（v1-draft）
- [`docs/contracts/capabilities.md`](docs/contracts/capabilities.md)：能力注册表（v1-draft），按档位声明能力支持情况
- FFI API：`batana_runtime_create(config) → session_analyze(input) → SessionResult`

## 路线图

- 模块规格：[主仓 docs/modules/batana-core.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/modules/batana-core.md)
- 能力分级矩阵：[主仓 docs/model-tiers.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/model-tiers.md)
- 生态路线图：[主仓 docs/roadmap.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/roadmap.md)
- 版本管理：[主仓 docs/versioning.md](https://github.com/vinnie-luckfocus/batana/blob/main/docs/versioning.md)

里程碑映射：P1 v0.1（standard-vision）→ P2 v0.2（IMU + 融合）→ P3 v0.3（双目 max）→ P5 coach_advice 模型化。

## License

MIT，见 [LICENSE](LICENSE)。
