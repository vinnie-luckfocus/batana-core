# runtime/backends — 推理后端

可插拔的模型推理后端，每个后端一个子目录：

- `tflite/`：TensorFlow Lite（移动端通用）
- `coreml/`：Apple CoreML（iOS / macOS）
- `onnx/`：ONNX Runtime（桌面 / 边缘）
- `rknn/`：瑞芯微 NPU（Linux-arm64 边缘端，如 batana-pi）

所有后端实现统一的推理接口，对上层管线屏蔽框架差异；运行时按平台与可用硬件自动选择。
