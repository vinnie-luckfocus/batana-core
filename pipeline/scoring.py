"""规则评分引擎 stub（能力 id: score_basic，支持度 rule）。

输入关键点序列（契约 33 点拓扑），输出对齐 session-schema metrics 的结构化评分。
当前为占位规则：仅用腕部位移与髋肩间距做粗启发式打分，供 M1.1 联调；
正式规则（角度/节奏/协调性）在后续里程碑细化。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .keypoints import KEYPOINT_NAMES
from .stages import KeypointFrame

_IDX = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def _xy(kf: KeypointFrame, name: str) -> np.ndarray | None:
    kp = kf.keypoints[_IDX[name]]
    if kp["visibility"] <= 0.0:
        return None
    return np.array([kp["x"], kp["y"]])


def _clamp100(v: float) -> float:
    return float(min(100.0, max(0.0, v)))


class RuleBasedScorer:
    """score_basic 规则实现：speed / angle / coordination / overall（0-100）。"""

    capabilities = ["score_basic"]

    def score(self, keypoints: Sequence[KeypointFrame], fps: float) -> dict[str, Any]:
        visible = [kf for kf in keypoints if kf.confidence > 0]
        if len(visible) < 2 or fps <= 0:
            return {
                "capabilities": self.capabilities,
                "scores": {"speed": 0, "angle": 0, "coordination": 0, "overall": 0},
                "details": {"note": "stub: 有效帧不足，输出零分占位"},
            }

        # 速度：腕部峰值归一化位移速率（身位/秒）
        wrist_series = [
            (t, p) for kf in visible
            for t in [kf.timestamp_ms] if (p := _xy(kf, "right_wrist")) is not None
        ]
        peak_speed = 0.0
        for (t0, p0), (t1, p1) in zip(wrist_series, wrist_series[1:]):
            dt = (t1 - t0) / 1000.0
            if dt > 0:
                peak_speed = max(peak_speed, float(np.linalg.norm(p1 - p0)) / dt)
        # 经验映射：归一化坐标下峰值速率 4 身位/秒 ≈ 满分
        speed = _clamp100(peak_speed / 4.0 * 100)

        # 角度：击球期髋-肩连线与水平夹角（粗占位，取中帧）
        mid = visible[len(visible) // 2]
        ls, rs = _xy(mid, "left_shoulder"), _xy(mid, "right_shoulder")
        angle = 0.0
        if ls is not None and rs is not None:
            tilt = abs(float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0]))))
            angle = _clamp100(100 - abs(15 - tilt) * 3)  # 理想肩部倾斜约 15°

        # 协调性：左右腕速度相关性占位（有效帧比例）
        coordination = _clamp100(len(wrist_series) / len(visible) * 100)

        overall = _clamp100(0.4 * speed + 0.3 * angle + 0.3 * coordination)
        return {
            "capabilities": self.capabilities,
            "scores": {
                "speed": round(speed, 1),
                "angle": round(angle, 1),
                "coordination": round(coordination, 1),
                "overall": round(overall, 1),
            },
            "details": {
                "stub": True,
                "peak_wrist_speed_norm_s": round(peak_speed, 3),
            },
        }
