"""批量三角化前处理测试：visibility 掩蔽、时序中值滤波、极线校正、schema 兼容。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.synthetic_rig import VirtualStereoRig
from tools.recon3d.preprocess import (
    correct_correspondences,
    epipolar_rms_error,
    fundamental_matrix,
    median_smooth,
    visibility_mask,
)
from tools.recon3d.triangulate import (
    load_batch_frames,
    load_calibration,
    main,
    triangulate_frames,
)
from tests.test_triangulate_batch import _sample_frames, _write_calib


@pytest.fixture(scope="module")
def rig() -> VirtualStereoRig:
    return VirtualStereoRig(baseline_mm=100.0, size=(1280, 800), seed=13)


@pytest.fixture(scope="module")
def calib(rig, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pre")
    return load_calibration(_write_calib(tmp, rig))


# ---- 掩蔽 ----


def test_visibility_mask_threshold():
    vl = np.array([[0.9, 0.3], [0.6, 0.6]])
    vr = np.array([[0.8, 0.8], [0.4, 0.9]])
    mask = visibility_mask(vl, vr, 0.5)
    assert mask.tolist() == [[True, False], [False, True]]
    with pytest.raises(ValueError, match="不一致"):
        visibility_mask(np.ones((2, 3)), np.ones((2, 2)), 0.5)


def test_batch_masking_outputs_null(rig, calib):
    """任一目低置信点输出 null，其余点正常三角化。"""
    frames, truth = _sample_frames(rig, n_frames=1, n_points=10)
    f = frames[0]
    f["left_visibility"] = [0.9] * 10
    f["right_visibility"] = [0.9] * 10
    f["left_visibility"][3] = 0.1   # 左目低置信
    f["right_visibility"][7] = 0.2  # 右目低置信
    results = triangulate_frames(calib, frames, min_vis=0.5)
    pts = results[0]["points_3d"]
    assert pts[3] is None and pts[7] is None
    ok = [p for j, p in enumerate(pts) if j not in (3, 7)]
    errs = np.linalg.norm(np.asarray(ok) - truth[0][[j for j in range(10) if j not in (3, 7)]], axis=1)
    assert float(np.median(errs)) < 1.0


def test_masked_points_stay_masked_after_smoothing(rig, calib):
    """被掩蔽帧的点平滑后仍保持掩蔽（平滑只影响未掩蔽帧的取值）。"""
    frames, _ = _sample_frames(rig, n_frames=5, n_points=4)
    for f in frames:
        f["left_visibility"] = [0.9] * 4
        f["right_visibility"] = [0.9] * 4
    frames[2]["left_visibility"][1] = 0.0  # 第 3 帧第 2 点掩蔽
    results = triangulate_frames(calib, frames, min_vis=0.5, smooth=5)
    assert results[2]["points_3d"][1] is None
    assert results[1]["points_3d"][1] is not None
    assert results[3]["points_3d"][1] is not None


def test_visibility_length_mismatch_raises(calib):
    frames = [{
        "frame_index": 0, "timestamp_ms": 0.0,
        "left": [[1.0, 2.0], [3.0, 4.0]], "right": [[1.0, 2.0], [3.0, 4.0]],
        "left_visibility": [0.9],
    }]
    with pytest.raises(ValueError, match="visibility"):
        triangulate_frames(calib, frames)


def test_batch_schema_backward_compatible(rig, calib, tmp_path):
    """无 visibility 的旧输入照常工作（缺省全部 1.0，不掩蔽）。"""
    frames, _ = _sample_frames(rig, n_frames=2, n_points=5)
    assert "left_visibility" not in frames[0]
    results = triangulate_frames(calib, frames)
    assert all(p is not None for p in results[0]["points_3d"])
    # CLI 旧输入 + 新参数组合可跑
    calib_path = _write_calib(tmp_path, rig)
    batch_path = tmp_path / "b.json"
    batch_path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
    out = tmp_path / "o.json"
    assert main(["--calib", str(calib_path), "--batch", str(batch_path),
                 "--out", str(out), "--min-vis", "0.5", "--smooth", "3",
                 "--epipolar-correct"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["frames"]) == 2


def test_load_batch_frames_validates_visibility(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"frames": [{
        "left": [[1, 2]], "right": [[1, 2]], "left_visibility": [0.9, 0.9],
    }]}), encoding="utf-8")
    with pytest.raises(ValueError, match="left_visibility"):
        load_batch_frames(bad)


# ---- 时序中值滤波 ----


def test_median_smooth_reduces_jitter():
    """含孤立尖峰噪声的轨迹：中值滤波后残差标准差显著下降。"""
    rng = np.random.default_rng(2)
    T, N = 40, 5
    t = np.arange(T)
    truth = np.stack([
        np.stack([400 + 50 * np.sin(t / 8 + j), 300 + 30 * np.cos(t / 10 + j)], -1)
        for j in range(N)
    ], 1)  # (T, N, 2) 平滑真值轨迹
    noisy = truth + rng.normal(0, 0.3, truth.shape)
    spikes = rng.random((T, N)) < 0.15  # 15% 孤立抖动
    noisy[spikes] += rng.normal(0, 15, noisy.shape)[spikes]
    valid = np.ones((T, N), dtype=bool)

    smoothed = median_smooth(noisy, valid, 5)
    err_before = float(np.std((noisy - truth).reshape(T, -1), axis=0).mean())
    err_after = float(np.std((smoothed - truth).reshape(T, -1), axis=0).mean())
    print(f"\n平滑前后残差 std px: {err_before:.2f} → {err_after:.2f}")
    assert err_after < err_before * 0.5


def test_median_smooth_ignores_invalid_frames():
    """无效帧不参与中值统计；全无效窗口保留原值。"""
    pts = np.zeros((5, 1, 2))
    pts[:, 0, 0] = [0.0, 100.0, 2.0, 4.0, 6.0]  # 帧 1 为尖峰
    valid = np.array([[True], [False], [True], [True], [True]])
    out = median_smooth(pts, valid, 3)
    assert out[1, 0, 0] == 100.0  # 无效帧保留原值
    assert out[2, 0, 0] == pytest.approx(3.0)  # 窗口 [1,2,3] 中有效值为 2、4 → 中位 3
    with pytest.raises(ValueError, match="奇数"):
        median_smooth(pts, valid, 4)


def test_smooth_requires_uniform_point_count(calib):
    frames = [
        {"frame_index": 0, "timestamp_ms": 0.0,
         "left": [[1, 2]], "right": [[1, 2]]},
        {"frame_index": 1, "timestamp_ms": 8.0,
         "left": [[1, 2], [3, 4]], "right": [[1, 2], [3, 4]]},
    ]
    with pytest.raises(ValueError, match="点数一致"):
        triangulate_frames(calib, frames, smooth=3)


# ---- 极线校正 ----


def test_epipolar_correct_reduces_error(rig, calib):
    """右目加噪后：correctMatches 校正使对称极线误差显著下降，三角化更准。"""
    rng = np.random.default_rng(9)
    gx, gy = np.meshgrid(np.arange(5) * 150.0, np.arange(4) * 150.0)
    grid = np.stack([gx.ravel() - 300, gy.ravel() - 225, np.full(20, 1800.0)], axis=1)
    pl = rig.left.project(grid)
    pr = rig.right.project(grid) + rng.normal(0, 1.5, (20, 2))  # 1.5px 噪声

    F = fundamental_matrix(calib)
    err_before = epipolar_rms_error(F, pl, pr)
    cl, cr = correct_correspondences(F, pl, pr)
    err_after = epipolar_rms_error(F, cl, cr)
    print(f"\n对称极线误差 px: {err_before:.3f} → {err_after:.3f}")
    assert err_after < err_before * 0.1

    from tools.recon3d.triangulate import triangulate_points
    tri_raw = triangulate_points(calib, pl, pr)
    tri_fix = triangulate_points(calib, cl, cr)
    e_raw = np.linalg.norm(tri_raw - grid, axis=1).mean()
    e_fix = np.linalg.norm(tri_fix - grid, axis=1).mean()
    print(f"三角化误差 mm: {e_raw:.2f} → {e_fix:.2f}")
    assert e_fix < e_raw


def test_triangulate_frames_with_epipolar_correct(rig, calib):
    """批量模式极线校正端到端：加噪轨迹校正后重建误差下降。"""
    rng = np.random.default_rng(11)
    frames, truth = _sample_frames(rig, n_frames=3, n_points=12)
    for f in frames:
        f["right"] = (np.asarray(f["right"]) + rng.normal(0, 1.0, (12, 2))).tolist()
    raw = triangulate_frames(calib, frames)
    fix = triangulate_frames(calib, frames, epipolar_correct=True)
    def errs(results):
        return np.concatenate([
            np.linalg.norm(np.asarray(r["points_3d"]) - t, axis=1)
            for r, t in zip(results, truth)
        ])
    assert errs(fix).mean() < errs(raw).mean()
