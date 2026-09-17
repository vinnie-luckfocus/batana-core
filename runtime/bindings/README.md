# runtime/bindings — 语言绑定

batana-runtime 的语言绑定层：

- `flutter/`：Dart FFI 绑定，供 batana（gui）移动应用调用
- `python/`：pybind11 绑定，供训练侧离线评估与 batana-web 云端调用

绑定层只做参数编解码与生命周期管理，业务逻辑全部在 runtime 核心。
