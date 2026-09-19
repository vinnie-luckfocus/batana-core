"""管线骨架冒烟测试：capture → preprocess → infer → score。"""

from __future__ import annotations

import numpy as np

from pipeline import KEYPOINT_NAMES, PoseInferStub, RuleBasedScorer, run_pipeline
from pipeline.stages import Frame


class FakeCapture:
    fps = 240.0

    def frames(self):
        for i in range(10):
            yield Frame(index=i, timestamp_ms=i * 1000.0 / 240.0, image=np.zeros((8, 8), np.uint8))


class IdentityPreprocess:
    def process(self, frame: Frame) -> Frame:
        return frame


def test_pipeline_skeleton_runs():
    result = run_pipeline(FakeCapture(), [IdentityPreprocess()], PoseInferStub(), RuleBasedScorer())
    assert len(result["pose2d"]["frames"]) == 10
    kps = result["pose2d"]["frames"][0]["keypoints"]
    assert [k["name"] for k in kps] == KEYPOINT_NAMES
    scores = result["metrics"]["scores"]
    assert set(scores) == {"speed", "angle", "coordination", "overall"}
    assert all(0 <= v <= 100 for v in scores.values())
