# runtime/include — 公共头文件

batana-runtime 对外暴露的公共头文件，包括：

- `capabilities.h`：能力注册表的 C/C++ 侧声明（能力 id、档位探测）
- `sources.h`：数据源抽象接口（`FrameSource` / `ImuSource`）
- FFI 入口声明：`batana_runtime_create(config) → session_analyze(input) → SessionResult`

本目录只放接口声明，不含实现。
