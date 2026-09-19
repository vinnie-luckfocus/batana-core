"""推理引擎接口桩：真实姿态模型接入前的占位实现。

返回固定结构的 33 点零可见性帧，保证管线骨架在 M1.1 前可端到端联调；
接入真实模型时实现 InferEngine 协议即可替换。
"""

from __future__ import annotations

from typing import Sequence

from .keypoints import KEYPOINT_NAMES
from .stages import Frame, KeypointFrame


class PoseInferStub:
    """InferEngine 桩：产出结构合法、数值为零的关键点序列。"""

    model = "batana-pose-stub"

    def infer(self, frames: Sequence[Frame]) -> list[KeypointFrame]:
        return [
            KeypointFrame(
                frame_index=f.index,
                timestamp_ms=f.timestamp_ms,
                confidence=0.0,
                keypoints=[{"name": n, "x": 0.0, "y": 0.0, "visibility": 0.0} for n in KEYPOINT_NAMES],
            )
            for f in frames
        ]
