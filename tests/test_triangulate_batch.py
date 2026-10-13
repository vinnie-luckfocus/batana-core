"""triangulate 批量模式（--batch）测试：往返一致性与输入校验。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.synthetic_rig import VirtualStereoRig
from tools.recon3d.triangulate import (
    load_batch_frames,
    load_calibration,
    main,
    triangulate_frames,
    triangulate_points,
)


def _write_calib(tmp_path, rig: VirtualStereoRig):
    """用 rig 真参构造标定 JSON（对齐 calibrate_stereo 输出结构）。"""
    w, h = rig.left.size
    calib = {
        "format_version": "1.0",
        "stereo": {
            "baseline_mm": rig.baseline_mm,
            "left": {
                "resolution": [w, h],
                "intrinsic": rig.left.K.tolist(),
                "distortion": rig.left.d.tolist(),
            },
            "right": {
                "resolution": [w, h],
                "intrinsic": rig.right.K.tolist(),
                "distortion": rig.right.d.tolist(),
            },
            "extrinsic": {
                "rotation": rig.R_gt.tolist(),
                "translation_mm": rig.C_right.tolist(),
            },
        },
    }
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(calib), encoding="utf-8")
    return path


def _sample_frames(rig: VirtualStereoRig, n_frames: int = 3, n_points: int = 20):
    rng = np.random.default_rng(5)
    frames = []
    truth = []
    for i in range(n_frames):
        pts = np.column_stack([
            rng.uniform(-300, 300, n_points),
            rng.uniform(-200, 200, n_points),
            np.full(n_points, 1800.0 + 100.0 * i),
        ])
        truth.append(pts)
        frames.append({
            "frame_index": i,
            "timestamp_ms": i * 8.3,
            "left": rig.left.project(pts).tolist(),
            "right": rig.right.project(pts).tolist(),
        })
    return frames, truth


def test_batch_triangulate_roundtrip(tmp_path):
    """批量三角化：多帧已知 3D 点投影往返，误差 < 1mm（无噪声投影）。"""
    rig = VirtualStereoRig()
    calib = load_calibration(_write_calib(tmp_path, rig))
    frames, truth = _sample_frames(rig)
    results = triangulate_frames(calib, frames)
    assert len(results) == len(frames)
    for res, pts in zip(results, truth):
        est = np.asarray(res["points_3d"])
        assert est.shape == pts.shape
        errs = np.linalg.norm(est - pts, axis=1)
        assert float(np.median(errs)) < 1.0
        assert "frame_index" in res and "timestamp_ms" in res


def test_batch_matches_single_mode(tmp_path):
    """批量模式逐帧结果与单组 triangulate_points 完全一致（公共函数抽取正确）。"""
    rig = VirtualStereoRig()
    calib = load_calibration(_write_calib(tmp_path, rig))
    frames, _ = _sample_frames(rig, n_frames=2)
    results = triangulate_frames(calib, frames)
    for f, res in zip(frames, results):
        single = triangulate_points(calib, np.asarray(f["left"]), np.asarray(f["right"]))
        assert np.allclose(np.asarray(res["points_3d"]), single)


def test_batch_point_count_mismatch_raises(tmp_path):
    """左右目点数不一致时报清晰错误。"""
    rig = VirtualStereoRig()
    calib = load_calibration(_write_calib(tmp_path, rig))
    frames = [{
        "frame_index": 0, "timestamp_ms": 0.0,
        "left": [[1.0, 2.0], [3.0, 4.0]],
        "right": [[1.0, 2.0]],
    }]
    with pytest.raises(ValueError, match="点数不一致"):
        triangulate_frames(calib, frames)


def test_batch_cli_writes_output(tmp_path):
    """CLI --batch：读批量输入写 frames 数组输出；结构含 frame/unit。"""
    rig = VirtualStereoRig()
    calib_path = _write_calib(tmp_path, rig)
    frames, _ = _sample_frames(rig, n_frames=2, n_points=5)
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    out_path = tmp_path / "out.json"
    assert main(["--calib", str(calib_path), "--batch", str(batch_path),
                 "--out", str(out_path)]) == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["frame"] == "left_camera" and data["unit"] == "mm"
    assert len(data["frames"]) == 2
    assert len(data["frames"][0]["points_3d"]) == 5
    assert data["frames"][1]["timestamp_ms"] == pytest.approx(8.3)


def test_load_batch_frames_validates(tmp_path):
    """批量输入结构校验：空 frames / 缺 left/right 均报错。"""
    bad1 = tmp_path / "bad1.json"
    bad1.write_text(json.dumps({"frames": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="frames"):
        load_batch_frames(bad1)
    bad2 = tmp_path / "bad2.json"
    bad2.write_text(json.dumps({"frames": [{"left": [[1, 2]]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="left/right"):
        load_batch_frames(bad2)
