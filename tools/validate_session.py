"""session JSON 校验器：按 docs/contracts/session-schema.md v1.0 校验必填字段/类型/约束。

用法：
    python -m tools.validate_session session.json
    python -m tools.validate_session session.json --require-metrics   # 上传云端场景：metrics 必填

输出全部错误（路径 + 原因），退出码 0 通过 / 1 校验失败 / 2 文件或 JSON 错误。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from pipeline.keypoints import KEYPOINT_NAMES

SCHEMA_VERSION_MAJOR = "1"
SESSION_ID_RE = re.compile(r"^sess_[0-9A-HJKMNP-TV-Z]{26}$")  # Crockford Base32（无 ILOU）
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CAP_ID_RE = re.compile(r"^[0-9A-F]{4}$")
PIPELINES = {"standard-vision", "standard-imu", "pro-fusion", "pro-stereo", "max"}
PHASES = {"stance", "load_stride", "swing_contact", "follow_through"}
CAPABILITY_IDS = {
    "pose2d", "phase_split", "score_basic", "bat_speed", "swing_count",
    "tempo_sync", "bat_traj3d", "kinematic_seq", "impact_point", "pose3d",
    "biomech_report", "coach_advice",
}


class _Errors:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, msg: str) -> None:
        self.items.append(f"{path}: {msg}")

    def check(self, cond: bool, path: str, msg: str) -> bool:
        if not cond:
            self.add(path, msg)
        return cond


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _matrix(errs: _Errors, v: Any, shape: tuple[int, ...], path: str) -> None:
    if not isinstance(v, list) or len(v) != shape[0]:
        errs.add(path, f"应为 {shape} 数值数组")
        return
    if len(shape) == 1:
        if not all(_is_num(x) for x in v):
            errs.add(path, "元素必须全为数值")
        return
    for i, row in enumerate(v):
        _matrix(errs, row, shape[1:], f"{path}[{i}]")


def _validate_meta(meta: Any, errs: _Errors) -> None:
    if not isinstance(meta, dict):
        errs.add("meta", "必填，应为对象")
        return
    sid = meta.get("session_id")
    if not isinstance(sid, str) or not SESSION_ID_RE.match(sid):
        errs.add("meta.session_id", "必填，格式 sess_ + 26 位 Crockford Base32")
    created = meta.get("created_at")
    if not isinstance(created, str) or not ISO_UTC_RE.match(created):
        errs.add("meta.created_at", "必填，ISO 8601 UTC（Z 后缀）")
    dur = meta.get("duration_ms")
    if not isinstance(dur, int) or isinstance(dur, bool) or dur <= 0:
        errs.add("meta.duration_ms", "必填，正整数（毫秒；纯 IMU 会话同样必填）")
    for section, keys in (("device", ("model", "os")), ("app", ("name", "version"))):
        sec = meta.get(section)
        if not isinstance(sec, dict):
            errs.add(f"meta.{section}", "必填，应为对象")
            continue
        for k in keys:
            if not isinstance(sec.get(k), str) or not sec[k]:
                errs.add(f"meta.{section}.{k}", "必填，非空字符串")
    if meta.get("pipeline") not in PIPELINES:
        errs.add("meta.pipeline", f"必填，取值 {sorted(PIPELINES)}")
    if meta.get("handedness") not in ("right", "left"):
        errs.add("meta.handedness", "必填，right / left")
    if "subject" in meta:
        sub = meta["subject"]
        if not isinstance(sub, dict):
            errs.add("meta.subject", "应为对象")
        else:
            if "height_cm" in sub and not _is_num(sub["height_cm"]):
                errs.add("meta.subject.height_cm", "应为数值")
            if "age_group" in sub and sub["age_group"] not in ("adult", "teen", "kid"):
                errs.add("meta.subject.age_group", "取值 adult / teen / kid")


def _validate_video(video: Any, errs: _Errors) -> None:
    if not isinstance(video, dict):
        errs.add("video", "应为对象")
        return
    if not isinstance(video.get("uri"), str) or not video["uri"]:
        errs.add("video.uri", "必填，非空字符串")
    if not _is_num(video.get("fps")) or video["fps"] <= 0:
        errs.add("video.fps", "必填，正数")
    res = video.get("resolution")
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(x, int) and x > 0 for x in res)):
        errs.add("video.resolution", "必填，[宽, 高] 正整数")
    if "duration_ms" in video and (not _is_num(video["duration_ms"]) or video["duration_ms"] <= 0):
        errs.add("video.duration_ms", "应为正数（毫秒）")
    if "view" in video and video["view"] not in ("side", "front", "behind", "other"):
        errs.add("video.view", "取值 side / front / behind / other")
    if "sha256" in video:
        s = video["sha256"]
        if not isinstance(s, str) or not SHA256_RE.match(s):
            errs.add("video.sha256", "应为 64 位小写十六进制字符串")


def _validate_imu(imu: Any, errs: _Errors) -> None:
    if not isinstance(imu, dict):
        errs.add("imu", "应为对象")
        return
    cap_id = imu.get("cap_id")
    if not isinstance(cap_id, str) or not CAP_ID_RE.match(cap_id):
        errs.add("imu.cap_id", "必填，4 位大写十六进制（BATANA-CAP-xx 的 xx）")
    if not _is_num(imu.get("sample_rate_hz")) or imu["sample_rate_hz"] <= 0:
        errs.add("imu.sample_rate_hz", "必填，正数")
    if "clock_anchor" in imu:
        ca = imu["clock_anchor"]
        if not isinstance(ca, dict):
            errs.add("imu.clock_anchor", "应为对象")
        else:
            if not isinstance(ca.get("device_ts_us"), int) or isinstance(ca.get("device_ts_us"), bool) or ca["device_ts_us"] < 0:
                errs.add("imu.clock_anchor.device_ts_us", "必填，非负整数（64 位单调时钟，微秒）")
            ht = ca.get("host_time_utc")
            if not isinstance(ht, str) or not ISO_UTC_RE.match(ht):
                errs.add("imu.clock_anchor.host_time_utc", "必填，ISO 8601 UTC（Z 后缀）")
            if not _is_num(ca.get("offset_ms")):
                errs.add("imu.clock_anchor.offset_ms", "必填，数值（毫秒）")
            if not _is_num(ca.get("drift_ppm")):
                errs.add("imu.clock_anchor.drift_ppm", "必填，数值（ppm）")
    if "samples" in imu:
        if not isinstance(imu["samples"], list):
            errs.add("imu.samples", "应为数组")
        else:
            for i, s in enumerate(imu["samples"]):
                p = f"imu.samples[{i}]"
                if not isinstance(s, dict):
                    errs.add(p, "应为对象")
                    continue
                if not _is_num(s.get("timestamp_ms")) or s["timestamp_ms"] < 0:
                    errs.add(f"{p}.timestamp_ms", "应为非负数值")
                for k in ("accel", "gyro"):
                    if k in s:
                        _matrix(errs, s[k], (3,), f"{p}.{k}")


def _validate_calibration(cal: Any, errs: _Errors) -> None:
    if not isinstance(cal, dict):
        errs.add("calibration", "应为对象")
        return
    if cal.get("format_version") != "1.0":
        errs.add("calibration.format_version", "必填，当前为 \"1.0\"")
    stereo = cal.get("stereo")
    if not isinstance(stereo, dict):
        errs.add("calibration.stereo", "必填，应为对象")
        return
    if not _is_num(stereo.get("baseline_mm")) or stereo["baseline_mm"] <= 0:
        errs.add("calibration.stereo.baseline_mm", "必填，正数（毫米）")
    for side in ("left", "right"):
        cam = stereo.get(side)
        p = f"calibration.stereo.{side}"
        if not isinstance(cam, dict):
            errs.add(p, "必填，应为对象")
            continue
        res = cam.get("resolution")
        if not (isinstance(res, list) and len(res) == 2 and all(isinstance(x, int) and x > 0 for x in res)):
            errs.add(f"{p}.resolution", "必填，[宽, 高] 正整数")
        if "intrinsic" in cam:
            _matrix(errs, cam["intrinsic"], (3, 3), f"{p}.intrinsic")
        else:
            errs.add(f"{p}.intrinsic", "必填，3×3 内参矩阵")
        if "distortion" in cam:
            _matrix(errs, cam["distortion"], (5,), f"{p}.distortion")
        else:
            errs.add(f"{p}.distortion", "必填，[k1,k2,p1,p2,k3]")
    ext = stereo.get("extrinsic")
    if not isinstance(ext, dict):
        errs.add("calibration.stereo.extrinsic", "必填，应为对象")
    else:
        if "rotation" in ext:
            _matrix(errs, ext["rotation"], (3, 3), "calibration.stereo.extrinsic.rotation")
        else:
            errs.add("calibration.stereo.extrinsic.rotation", "必填，3×3 旋转矩阵")
        if "translation_mm" in ext:
            _matrix(errs, ext["translation_mm"], (3,), "calibration.stereo.extrinsic.translation_mm")
        else:
            errs.add("calibration.stereo.extrinsic.translation_mm", "必填，平移向量（毫米）")


def _validate_pose2d(pose: Any, errs: _Errors) -> None:
    if not isinstance(pose, dict):
        errs.add("pose2d", "应为对象")
        return
    if not isinstance(pose.get("model"), str) or not pose["model"]:
        errs.add("pose2d.model", "必填，非空字符串（如 batana-pose-v0.1）")
    if not _is_num(pose.get("frame_rate")) or pose["frame_rate"] <= 0:
        errs.add("pose2d.frame_rate", "必填，正数")
    frames = pose.get("frames")
    if not isinstance(frames, list):
        errs.add("pose2d.frames", "必填，应为数组")
        return
    for i, f in enumerate(frames):
        p = f"pose2d.frames[{i}]"
        if not isinstance(f, dict):
            errs.add(p, "应为对象")
            continue
        if not isinstance(f.get("frame_index"), int) or isinstance(f.get("frame_index"), bool) or f["frame_index"] < 0:
            errs.add(f"{p}.frame_index", "应为非负整数")
        if not _is_num(f.get("timestamp_ms")) or f["timestamp_ms"] < 0:
            errs.add(f"{p}.timestamp_ms", "应为非负数值")
        conf = f.get("confidence")
        if not _is_num(conf) or not 0.0 <= conf <= 1.0:
            errs.add(f"{p}.confidence", "应在 [0,1]")
        kps = f.get("keypoints")
        if not isinstance(kps, list) or len(kps) != 33:
            errs.add(f"{p}.keypoints", "应为 33 点数组（BlazePose 拓扑，顺序固定）")
            continue
        names = [k.get("name") if isinstance(k, dict) else None for k in kps]
        if names != KEYPOINT_NAMES:
            errs.add(f"{p}.keypoints", "关键点名称或顺序与契约 33 点枚举不符")
            continue
        for j, k in enumerate(kps):
            for axis in ("x", "y"):
                v = k.get(axis)
                if not _is_num(v) or not 0.0 <= v <= 1.0:
                    errs.add(f"{p}.keypoints[{j}].{axis}", "归一化坐标应在 [0,1]")
            v = k.get("visibility")
            if not _is_num(v) or not 0.0 <= v <= 1.0:
                errs.add(f"{p}.keypoints[{j}].visibility", "应在 [0,1]")


def _validate_phases(phases: Any, errs: _Errors) -> None:
    if not isinstance(phases, dict):
        errs.add("phases", "应为对象")
        return
    if phases.get("granularity") not in ("fine", "coarse"):
        errs.add("phases.granularity", "取值 fine / coarse")
    segs = phases.get("segments")
    if not isinstance(segs, list):
        errs.add("phases.segments", "应为数组")
        return
    for i, s in enumerate(segs):
        p = f"phases.segments[{i}]"
        if not isinstance(s, dict):
            errs.add(p, "应为对象")
            continue
        if s.get("phase") not in PHASES:
            errs.add(f"{p}.phase", f"取值 {sorted(PHASES)}")
        if not (_is_num(s.get("start_ms")) and _is_num(s.get("end_ms")) and 0 <= s["start_ms"] < s["end_ms"]):
            errs.add(p, "需满足 0 ≤ start_ms < end_ms")


def _validate_metrics(metrics: Any, errs: _Errors) -> None:
    if not isinstance(metrics, dict):
        errs.add("metrics", "应为对象")
        return
    caps = metrics.get("capabilities")
    if not isinstance(caps, list) or not all(isinstance(c, str) for c in caps):
        errs.add("metrics.capabilities", "必填，能力 id 字符串数组")
    else:
        unknown = [c for c in caps if c not in CAPABILITY_IDS]
        if unknown:
            errs.add("metrics.capabilities", f"未知能力 id: {unknown}（见 capabilities.md）")
    if "scores" in metrics:
        scores = metrics["scores"]
        if not isinstance(scores, dict):
            errs.add("metrics.scores", "应为对象")
        else:
            for k, v in scores.items():
                if not _is_num(v) or not 0 <= v <= 100:
                    errs.add(f"metrics.scores.{k}", "分项评分应在 [0,100]")
    for k in ("bat_speed_mps", "tempo_ratio"):
        if k in metrics and (not _is_num(metrics[k]) or metrics[k] <= 0):
            errs.add(f"metrics.{k}", "应为正数")
    if "swing_count" in metrics and (not isinstance(metrics["swing_count"], int) or metrics["swing_count"] < 0):
        errs.add("metrics.swing_count", "应为非负整数")


def validate_session(data: Any, require_metrics: bool = False) -> list[str]:
    """校验 session 数据，返回错误列表（空列表表示通过）。"""
    errs = _Errors()
    if not isinstance(data, dict):
        return ["根节点: 应为对象"]
    sv = data.get("schema_version")
    if not isinstance(sv, str) or not sv.startswith(SCHEMA_VERSION_MAJOR + "."):
        errs.add("schema_version", f"必填，语义化版本字符串，本校验器支持主版本 {SCHEMA_VERSION_MAJOR}.x")
    _validate_meta(data.get("meta"), errs)
    if "video" in data:
        _validate_video(data["video"], errs)
    if "imu" in data:
        _validate_imu(data["imu"], errs)
    if "calibration" in data:
        _validate_calibration(data["calibration"], errs)
    if "pose2d" in data:
        _validate_pose2d(data["pose2d"], errs)
    if "phases" in data:
        _validate_phases(data["phases"], errs)
    if "metrics" in data:
        _validate_metrics(data["metrics"], errs)
    elif require_metrics:
        errs.add("metrics", "上传云端时必填（本地仅采集未分析时可缺失）")
    return errs.items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="session JSON 校验器（session-schema v1.0）")
    parser.add_argument("session", help="session JSON 文件路径")
    parser.add_argument("--require-metrics", action="store_true", help="要求 metrics 必填（上传云端场景）")
    args = parser.parse_args(argv)

    try:
        data = json.loads(Path(args.session).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取/解析 {args.session}: {e}")
        return 2
    errors = validate_session(data, require_metrics=args.require_metrics)
    if errors:
        print(f"校验失败（{len(errors)} 处）:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
