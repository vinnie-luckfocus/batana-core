"""session 校验器正反例测试。"""

from __future__ import annotations

import copy

import pytest

from pipeline.keypoints import KEYPOINT_NAMES
from tools.validate_session import validate_session


def valid_session() -> dict:
    return {
        "schema_version": "1.0",
        "meta": {
            "session_id": "sess_01J8H2K3M4N5P6Q7R8S9T0ABCD",
            "created_at": "2026-09-19T02:30:00Z",
            "duration_ms": 1850,
            "device": {"model": "iPhone 16 Pro", "os": "iOS 19.0"},
            "app": {"name": "batana", "version": "0.1.0"},
            "pipeline": "pro-fusion",
            "handedness": "right",
            "subject": {"height_cm": 175, "age_group": "adult"},
        },
        "video": {
            "uri": "file:///captures/sess.mp4",
            "fps": 240,
            "resolution": [1280, 720],
            "duration_ms": 1850,
            "view": "side",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        },
        "imu": {
            "cap_id": "04A2",
            "sample_rate_hz": 200,
            "clock_anchor": {
                "device_ts_us": 1234567890,
                "host_time_utc": "2026-09-19T02:30:00.000Z",
                "offset_ms": 0,
                "drift_ppm": 12.5,
            },
            "samples": [{"timestamp_ms": 0, "accel": [0.12, -9.78, 0.45], "gyro": [0.01, 0.02, -0.15]}],
        },
        "calibration": {
            "format_version": "1.0",
            "stereo": {
                "baseline_mm": 100.0,
                "left": {
                    "resolution": [1280, 800],
                    "intrinsic": [[1100.0, 0, 640.0], [0, 1100.0, 400.0], [0, 0, 1]],
                    "distortion": [0.05, -0.02, 0.001, -0.001, 0.0],
                },
                "right": {
                    "resolution": [1280, 800],
                    "intrinsic": [[1101.0, 0, 639.0], [0, 1100.5, 400.5], [0, 0, 1]],
                    "distortion": [0.048, -0.018, 0.001, -0.001, 0.0],
                },
                "extrinsic": {
                    "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_mm": [100.0, 0, 0],
                },
            },
        },
        "pose2d": {
            "model": "batana-pose-v0.1",
            "frame_rate": 240,
            "frames": [{
                "frame_index": 0,
                "timestamp_ms": 0,
                "confidence": 0.92,
                "keypoints": [{"name": n, "x": 0.5, "y": 0.5, "visibility": 0.9} for n in KEYPOINT_NAMES],
            }],
        },
        "phases": {
            "granularity": "fine",
            "segments": [
                {"phase": "stance", "start_ms": 0, "end_ms": 620},
                {"phase": "load_stride", "start_ms": 620, "end_ms": 980},
                {"phase": "swing_contact", "start_ms": 980, "end_ms": 1180},
                {"phase": "follow_through", "start_ms": 1180, "end_ms": 1850},
            ],
        },
        "metrics": {
            "capabilities": ["pose2d", "phase_split", "score_basic", "bat_speed"],
            "bat_speed_mps": 28.4,
            "swing_count": 1,
            "tempo_ratio": 2.6,
            "scores": {"speed": 72, "angle": 65, "coordination": 70, "overall": 69},
            "details": {},
        },
    }


def test_valid_session_passes():
    assert validate_session(valid_session()) == []


def test_minimal_session_passes():
    """仅必填字段的纯 IMU 会话（无视频/metrics 未分析）应通过。"""
    s = {
        "schema_version": "1.0",
        "meta": {
            "session_id": "sess_01J8H2K3M4N5P6Q7R8S9T0ABCD",
            "created_at": "2026-09-19T02:30:00Z",
            "duration_ms": 500,
            "device": {"model": "batana-pi", "os": "Linux"},
            "app": {"name": "batana", "version": "0.1.0"},
            "pipeline": "standard-imu",
            "handedness": "left",
        },
        "imu": {"cap_id": "00FF", "sample_rate_hz": 200},
    }
    assert validate_session(s) == []


def test_require_metrics_flag():
    s = valid_session()
    del s["metrics"]
    assert validate_session(s) == []
    errors = validate_session(s, require_metrics=True)
    assert any("metrics" in e for e in errors)


@pytest.mark.parametrize("mutate,expect", [
    (lambda s: s["meta"].__setitem__("session_id", "abc"), "session_id"),
    (lambda s: s["meta"].__setitem__("created_at", "2026-09-19 10:30:00"), "created_at"),
    (lambda s: s["meta"].__delitem__("duration_ms"), "duration_ms"),
    (lambda s: s["meta"].__setitem__("duration_ms", -5), "duration_ms"),
    (lambda s: s["meta"].__setitem__("pipeline", "ultra"), "pipeline"),
    (lambda s: s["meta"].__setitem__("handedness", "both"), "handedness"),
    (lambda s: s["video"].__setitem__("sha256", "9F2C4A1B"), "sha256"),
    (lambda s: s["video"].__setitem__("resolution", [1280]), "resolution"),
    (lambda s: s["imu"].__setitem__("cap_id", "04a2"), "cap_id"),
    (lambda s: s["imu"]["clock_anchor"].__setitem__("host_time_utc", "2026-09-19T10:30:00+08:00"), "host_time_utc"),
    (lambda s: s["imu"]["clock_anchor"].__delitem__("drift_ppm"), "drift_ppm"),
    (lambda s: s["calibration"]["stereo"].__setitem__("baseline_mm", -1), "baseline_mm"),
    (lambda s: s["calibration"]["stereo"]["left"].__setitem__("intrinsic", [[1, 0], [0, 1]]), "intrinsic"),
    (lambda s: s["calibration"]["stereo"]["left"].__setitem__("distortion", [0.1, 0.2]), "distortion"),
    (lambda s: s["calibration"]["stereo"]["extrinsic"].__setitem__("translation_mm", [1, 2]), "translation_mm"),
    (lambda s: s["calibration"].__setitem__("format_version", "2.0"), "format_version"),
    (lambda s: s["pose2d"]["frames"][0].__setitem__("keypoints", s["pose2d"]["frames"][0]["keypoints"][:32]), "keypoints"),
    (lambda s: s["pose2d"]["frames"][0]["keypoints"][0].__setitem__("x", 1.5), "x"),
    (lambda s: s["pose2d"]["frames"][0].__setitem__("confidence", 1.2), "confidence"),
    (lambda s: s["phases"]["segments"][0].__setitem__("phase", "warmup"), "phase"),
    (lambda s: s["phases"]["segments"][0].__setitem__("start_ms", 700), "start_ms"),
    (lambda s: s["metrics"].__setitem__("capabilities", ["fly_to_moon"]), "capabilities"),
    (lambda s: s["metrics"]["scores"].__setitem__("speed", 120), "speed"),
    (lambda s: s.__setitem__("schema_version", "2.0"), "schema_version"),
])
def test_invalid_sessions_rejected(mutate, expect):
    s = valid_session()
    mutate(s)
    errors = validate_session(s)
    assert errors, f"应检出 {expect} 错误"
    assert any(expect in e for e in errors), f"错误信息应指向 {expect}: {errors}"
