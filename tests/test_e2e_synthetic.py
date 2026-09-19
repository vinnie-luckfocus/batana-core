"""端到端合成验证：虚拟双目 → 渲染标定帧 → 标定 → 三角化 → 精度评估。

不依赖真实相机；覆盖 M0 G0 V0c 的管线全链路（硬件到货前）。
断言：外参恢复误差在容差内、合成噪声下 3D 重建误差 < 5mm。
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from tests.synthetic_rig import VirtualStereoRig
from tools.calibration.board import make_board
from tools.calibration.calibrate_stereo import (
    calibrate_mono,
    calibrate_stereo_pair,
    collect_detections,
)
from tools.recon3d.eval_grid import grid_spacing_errors, replay_board_frame
from tools.recon3d.triangulate import load_calibration, triangulate_points
from tools.validate_session import validate_session

BASELINE_MM = 100.0
N_VIEWS = 14


@pytest.fixture(scope="module")
def rig() -> VirtualStereoRig:
    return VirtualStereoRig(baseline_mm=BASELINE_MM, size=(1280, 800), seed=7)


@pytest.fixture(scope="module")
def board():
    return make_board()  # 11×8，格子 100mm


@pytest.fixture(scope="module")
def frames(rig, board):
    """渲染 N_VIEWS 组两目共视标定帧（姿态 ±35°，覆盖画幅边缘以约束畸变）。"""
    pairs = []
    while len(pairs) < N_VIEWS:
        rvec, tvec = rig.sample_board_pose()
        rvec[0] = np.deg2rad(rig.rng.uniform(-35, 35))
        rvec[1] = np.deg2rad(rig.rng.uniform(-35, 35))
        if not rig.board_fully_visible(board, rvec, tvec):
            continue
        img_l = rig.render_board(board, rvec, tvec, rig.left)
        img_r = rig.render_board(board, rvec, tvec, rig.right)
        pairs.append((f"syn_{len(pairs):02d}", img_l, img_r))
    return pairs


@pytest.fixture(scope="module")
def calib_result(rig, board, frames):
    """走与 CLI 相同的标定链路，返回结果 dict（对齐契约 calibration 字段）。"""
    left, right = collect_detections(frames, board)
    assert len(left.frame_names) >= N_VIEWS - 2, "合成帧检出率过低"
    rms_l, K1, d1 = calibrate_mono(left)
    rms_r, K2, d2 = calibrate_mono(right)
    rms_s, R, T, _ = calibrate_stereo_pair(board, left, right, K1, d1, K2, d2)
    from tools.calibration.calibrate_stereo import build_result
    return build_result(left, right, rms_l, rms_r, rms_s, K1, d1, K2, d2, R, T, board, 11, 8, 100.0)


def rotation_angle_deg(R: np.ndarray) -> float:
    rvec, _ = cv2.Rodrigues(R)
    return float(np.degrees(np.linalg.norm(rvec)))


def test_calibration_extrinsics(calib_result, rig):
    """标定恢复的外参误差在容差内：基线 <2mm，旋转 <0.3°。"""
    stereo = calib_result["stereo"]
    baseline_err = abs(stereo["baseline_mm"] - rig.baseline_mm)
    R_est = np.asarray(stereo["extrinsic"]["rotation"])
    R_err = rotation_angle_deg(R_est @ rig.R_gt.T)
    print(f"\n基线: 估计 {stereo['baseline_mm']:.2f}mm / 真值 {rig.baseline_mm}mm，误差 {baseline_err:.3f}mm")
    print(f"旋转误差: {R_err:.4f}°")
    assert baseline_err < 2.0
    assert R_err < 0.3


def test_calibration_intrinsics(calib_result, rig):
    """内参焦距恢复误差 < 3%。"""
    fx_est = calib_result["stereo"]["left"]["intrinsic"][0][0]
    fx_gt = rig.left.K[0, 0]
    rel = abs(fx_est - fx_gt) / fx_gt
    print(f"\n左目 fx: 估计 {fx_est:.1f} / 真值 {fx_gt:.1f}，相对误差 {rel:.2%}")
    assert rel < 0.03


def test_reprojection_error(calib_result):
    """合成噪声下重投影误差应 < 1px。"""
    rep = calib_result["report"]["reprojection_error_px"]
    print(f"\n重投影误差 px: {rep}")
    assert rep["left"] < 1.0 and rep["right"] < 1.0 and rep["stereo"] < 1.0


def test_triangulation_accuracy(calib_result, rig, tmp_path):
    """3D 点阵三角化：合成噪声下重建误差 < 5mm。"""
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(json.dumps(calib_result), encoding="utf-8")
    calib = load_calibration(calib_path)

    rng = np.random.default_rng(0)
    # 1.5m 处 5×4 点阵，间距 150mm（2m/100mm 基线下的深度噪声 σ≈7mm/0.2px，此处噪声 0.1px）
    gx, gy = np.meshgrid(np.arange(5) * 150.0, np.arange(4) * 150.0)
    grid = np.stack([gx.ravel() - 300, gy.ravel() - 225, np.full(20, 1500.0)], axis=1)
    grid += rng.uniform(-5, 5, grid.shape)  # 轻微扰动避免退化

    pl = rig.left.project(grid) + rng.normal(0, 0.1, grid.shape[:1] + (2,))
    pr = rig.right.project(grid) + rng.normal(0, 0.1, grid.shape[:1] + (2,))
    pts3d = triangulate_points(calib, pl, pr)
    errs = np.linalg.norm(pts3d - grid, axis=1)
    print(f"\n三角化误差 mm: mean {errs.mean():.3f} / p95 {np.percentile(errs, 95):.3f} / max {errs.max():.3f}")
    assert errs.mean() < 5.0


def test_grid_eval_and_replay(calib_result, rig, board, tmp_path):
    """网格回放比对：渲染一帧新位姿，检测→三角化→刚体对齐→误差统计。"""
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(json.dumps(calib_result), encoding="utf-8")
    calib = load_calibration(calib_path)

    # 固定确定性位姿（2.0m V0c 目标距离、小倾角），保证断言稳定可复现
    rvec = np.deg2rad(np.array([8.0, -6.0, 3.0]))
    tvec = np.array([-40.0, -330.0, 2000.0])
    assert rig.board_fully_visible(board, rvec, tvec)
    img_l = rig.render_board(board, rvec, tvec, rig.left)
    img_r = rig.render_board(board, rvec, tvec, rig.right)

    report = replay_board_frame(calib, board, img_l, img_r)
    pe = report["point_error_mm"]
    se = report["spacing_error_mm"]
    print(f"\n回放点残差 mm: mean {pe['mean']:.3f} / p95 {pe['p95']:.3f} / max {pe['max']:.3f}")
    print(f"回放间距误差 mm: mean {se['mean']:.3f} / p95 {se['p95']:.3f} / max {se['max']:.3f}")
    assert report["points_used"] >= 60
    # 合成噪声下 3D 重建误差 < 5mm（均值与间距 p95 口径）
    assert pe["mean"] < 5.0 and se["p95"] < 5.0
    # V0c 通过标准（≤15mm@2m）应大幅满足；点残差 p95 含 100mm 基线固有的视差噪声长尾
    assert pe["p95"] < 15.0 and se["max"] < 15.0


def test_grid_spacing_eval(calib_result, rig):
    """网格模式评估：真值网格投影三角化后间距误差 < 5mm。"""
    calib = {
        "K1": calib_result and np.asarray(calib_result["stereo"]["left"]["intrinsic"]),
        "d1": np.asarray(calib_result["stereo"]["left"]["distortion"]),
        "K2": np.asarray(calib_result["stereo"]["right"]["intrinsic"]),
        "d2": np.asarray(calib_result["stereo"]["right"]["distortion"]),
        "R": np.asarray(calib_result["stereo"]["extrinsic"]["rotation"]),
        "T": (-np.asarray(calib_result["stereo"]["extrinsic"]["rotation"])
             @ np.asarray(calib_result["stereo"]["extrinsic"]["translation_mm"])).reshape(3, 1),
    }
    rng = np.random.default_rng(3)
    rows, cols, spacing = 4, 5, 150.0
    gx, gy = np.meshgrid(np.arange(cols) * spacing, np.arange(rows) * spacing)
    grid = np.stack([gx.ravel() - 300, gy.ravel() - 225, np.full(rows * cols, 1500.0)], axis=1)
    pl = rig.left.project(grid) + rng.normal(0, 0.1, (rows * cols, 2))
    pr = rig.right.project(grid) + rng.normal(0, 0.1, (rows * cols, 2))
    pts3d = triangulate_points(calib, pl, pr)
    errs = grid_spacing_errors(pts3d, rows, cols, spacing)
    print(f"\n网格间距误差 mm: mean {errs.mean():.3f} / p95 {np.percentile(errs, 95):.3f} / max {errs.max():.3f}")
    assert np.percentile(errs, 95) < 5.0


def test_calib_json_passes_session_validator(calib_result):
    """标定输出嵌入 session 后须通过契约校验。"""
    session = {
        "schema_version": "1.0",
        "meta": {
            "session_id": "sess_01J8H2K3M4N5P6Q7R8S9T0ABCD",
            "created_at": "2026-09-19T02:30:00Z",
            "duration_ms": 1850,
            "device": {"model": "batana-pi", "os": "Linux"},
            "app": {"name": "batana", "version": "0.1.0"},
            "pipeline": "pro-stereo",
            "handedness": "right",
        },
        "calibration": {
            "format_version": calib_result["format_version"],
            "stereo": calib_result["stereo"],
        },
    }
    errors = validate_session(session)
    assert errors == [], f"校验失败: {errors}"
