# runtime — C++ 推理运行时（batana-runtime）

跨平台 C++17 推理运行时（iOS / Android / macOS / Linux-arm64），模型后端可插拔，通过 FFI 暴露给 Flutter 等宿主。

- `include/`：公共头文件（`capabilities.h`、`sources.h` 等）
- `backends/`：TFLite / CoreML / ONNX Runtime / RKNN 等推理后端
- `bindings/`：Flutter FFI、Python pybind 绑定

数据源以接口注入：`FrameSource`（mono/stereo 帧流）与 `ImuSource`（带时间戳的 6 轴流）由调用方实现，本目录不含任何硬件驱动。
