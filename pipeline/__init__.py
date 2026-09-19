"""batana 管线骨架（M1.1 前置）：capture → preprocess → infer → score。"""

from .infer import PoseInferStub
from .keypoints import KEYPOINT_NAMES
from .scoring import RuleBasedScorer
from .stages import (
    CaptureSource,
    Frame,
    InferEngine,
    KeypointFrame,
    Preprocessor,
    Scorer,
    run_pipeline,
)

__all__ = [
    "CaptureSource",
    "Frame",
    "InferEngine",
    "KeypointFrame",
    "KEYPOINT_NAMES",
    "PoseInferStub",
    "Preprocessor",
    "RuleBasedScorer",
    "Scorer",
    "run_pipeline",
]
