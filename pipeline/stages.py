"""管线阶段抽象（M1.1 骨架）：capture → preprocess → infer → score。

各阶段以 Protocol 定义接口，具体实现（视频源、姿态模型、评分引擎）按档位注入；
本包只定义骨架与数据结构，不含任何硬件驱动与模型权重。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class Frame:
    """一帧采集数据。"""

    index: int
    timestamp_ms: float
    image: np.ndarray  # 灰度或 BGR，依采集源而定
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class KeypointFrame:
    """一帧 2D 姿态结果（归一化坐标，契约同 session-schema pose2d）。"""

    frame_index: int
    timestamp_ms: float
    confidence: float
    keypoints: list[dict[str, Any]]  # [{name, x, y, visibility}]，33 点，顺序固定


@runtime_checkable
class CaptureSource(Protocol):
    """采集源：产出帧流。硬件相关实现由外部注入。"""

    fps: float

    def frames(self) -> Iterator[Frame]:
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """预处理器：帧 → 帧（如切半、去畸变、缩放）。"""

    def process(self, frame: Frame) -> Frame:
        ...


@runtime_checkable
class InferEngine(Protocol):
    """推理引擎：帧序列 → 关键点序列。模型实现由外部注入。"""

    def infer(self, frames: Sequence[Frame]) -> list[KeypointFrame]:
        ...


@runtime_checkable
class Scorer(Protocol):
    """评分引擎：关键点序列 → 结构化评分 dict（对齐 session-schema metrics）。"""

    def score(self, keypoints: Sequence[KeypointFrame], fps: float) -> dict[str, Any]:
        ...


def run_pipeline(
    capture: CaptureSource,
    preprocessors: Sequence[Preprocessor],
    infer: InferEngine,
    scorer: Scorer,
) -> dict[str, Any]:
    """骨架执行：capture → preprocess → infer → score，返回 pose2d + metrics 结构。"""
    frames = []
    for frame in capture.frames():
        for pre in preprocessors:
            frame = pre.process(frame)
        frames.append(frame)
    keypoints = infer.infer(frames)
    metrics = scorer.score(keypoints, capture.fps)
    return {
        "pose2d": {
            "frame_rate": capture.fps,
            "frames": [
                {
                    "frame_index": kf.frame_index,
                    "timestamp_ms": kf.timestamp_ms,
                    "confidence": kf.confidence,
                    "keypoints": kf.keypoints,
                }
                for kf in keypoints
            ],
        },
        "metrics": metrics,
    }
