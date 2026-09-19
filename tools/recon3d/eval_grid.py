"""三维重建精度评估：网格角点重建值 vs 已知真值，判定 ≤15mm@2m 通过标准（V0c）。

两种用法：
1. 已有 3D 网格点（如 triangulate 输出）：
   python -m tools.recon3d.eval_grid --points3d points3d.json --grid-shape 7x10 --spacing-mm 100
2. 标定板帧回放比对（自动检测角点 → 三角化 → 刚体对齐真值网格 → 残差统计）：
   python -m tools.recon3d.eval_grid --calib calibration.json --left left.png --right right.png

通过标准：默认 p95 误差 ≤ 15mm（--threshold-mm / --metric 可调）。
退出码：0 通过，1 未通过，2 用法/数据错误。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tools.calibration.board import add_board_args, board_from_args, grid_shape
from tools.recon3d.triangulate import load_calibration, triangulate_points

DEFAULT_THRESHOLD_MM = 15.0


def summarize(errors_mm: np.ndarray, threshold_mm: float = DEFAULT_THRESHOLD_MM, metric: str = "p95") -> dict:
    """误差统计：mean/p95/max（mm）与通过判定。"""
    errors_mm = np.asarray(errors_mm, dtype=np.float64).ravel()
    if errors_mm.size == 0:
        raise ValueError("无误差样本")
    stats = {
        "count": int(errors_mm.size),
        "mean": float(np.mean(errors_mm)),
        "p95": float(np.percentile(errors_mm, 95)),
        "max": float(np.max(errors_mm)),
        "threshold_mm": threshold_mm,
        "metric": metric,
    }
    stats["pass"] = bool(stats[metric] <= threshold_mm)
    return stats


def grid_spacing_errors(points3d: np.ndarray, rows: int, cols: int, spacing_mm: float) -> np.ndarray:
    """有序网格（行优先 rows×cols）相邻点间距相对真值的绝对误差。"""
    grid = np.asarray(points3d, dtype=np.float64).reshape(rows, cols, 3)
    errs = []
    if cols > 1:
        errs.append(np.abs(np.linalg.norm(np.diff(grid, axis=1), axis=2) - spacing_mm).ravel())
    if rows > 1:
        errs.append(np.abs(np.linalg.norm(np.diff(grid, axis=0), axis=2) - spacing_mm).ravel())
    return np.concatenate(errs)


def rigid_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch 刚体对齐：求 R,t 使 src @ R.T + t ≈ dst，返回 (对齐后的 src, R, t)。"""
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    H = (src - c_src).T @ (dst - c_dst)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:  # 抑制反射
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = c_dst - c_src @ R.T
    return src @ R.T + t, R, t


def replay_board_frame(calib: dict, board, left_img: np.ndarray, right_img: np.ndarray) -> dict:
    """单帧回放：检测角点 → 三角化 → 对齐真值网格 → 点残差 + 间距误差统计。"""
    detector = cv2.aruco.CharucoDetector(board)
    corners_l, ids_l, _, _ = detector.detectBoard(left_img)
    corners_r, ids_r, _, _ = detector.detectBoard(right_img)
    if ids_l is None or ids_r is None:
        raise ValueError("标定板角点检测失败")
    common = np.intersect1d(ids_l, ids_r)
    if len(common) < 4:
        raise ValueError(f"左右目共同角点过少: {len(common)}")
    lm = {int(c): i for i, c in enumerate(ids_l)}
    rm = {int(c): i for i, c in enumerate(ids_r)}
    pl = np.stack([corners_l[lm[int(c)]] for c in common])
    pr = np.stack([corners_r[rm[int(c)]] for c in common])
    pts3d = triangulate_points(calib, pl, pr)

    truth = board.getChessboardCorners()[common]  # (N,3) mm，z=0
    aligned, _, _ = rigid_align(pts3d, truth)
    point_errs = np.linalg.norm(aligned - truth, axis=1)

    rows, cols = grid_shape(board)
    square_mm = board.getSquareLength()
    pos = {int(c): i for i, c in enumerate(common)}
    spacing_errs = []
    for cid in common:
        r, c = divmod(int(cid), cols)
        for nid in ((cid + 1) if c + 1 < cols else None, (cid + cols) if r + 1 < rows else None):
            if nid is not None and int(nid) in pos:
                d = np.linalg.norm(aligned[pos[int(cid)]] - aligned[pos[int(nid)]])
                spacing_errs.append(abs(d - square_mm))
    return {
        "points_used": int(len(common)),
        "point_error_mm": summarize(point_errs),
        "spacing_error_mm": summarize(np.asarray(spacing_errs)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="三维重建精度评估（V0c 通过标准 ≤15mm@2m）")
    parser.add_argument("--calib", help="标定 JSON（回放模式必填）")
    parser.add_argument("--points3d", help="triangulate 输出的 3D 点 JSON（网格模式）")
    parser.add_argument("--grid-shape", help="网格形状 ROWSxCOLS，如 7x10（网格模式必填）")
    parser.add_argument("--spacing-mm", type=float, default=None, help="真值网格间距 mm（网格模式必填）")
    parser.add_argument("--left", help="左目标定板帧（回放模式）")
    parser.add_argument("--right", help="右目标定板帧（回放模式）")
    add_board_args(parser)
    parser.add_argument("--threshold-mm", type=float, default=DEFAULT_THRESHOLD_MM, help="通过阈值 mm（默认 15）")
    parser.add_argument("--metric", choices=["mean", "p95", "max"], default="p95", help="判定指标（默认 p95）")
    args = parser.parse_args(argv)

    if args.points3d:
        if not args.grid_shape or args.spacing_mm is None:
            parser.error("网格模式需同时给出 --points3d --grid-shape --spacing-mm")
        rows, cols = (int(v) for v in args.grid_shape.lower().split("x"))
        data = json.loads(Path(args.points3d).read_text(encoding="utf-8"))
        pts = np.asarray(data["points_3d"], dtype=np.float64)
        if len(pts) != rows * cols:
            print(f"错误: 点数 {len(pts)} 与网格 {rows}x{cols}={rows*cols} 不符")
            return 2
        errs = grid_spacing_errors(pts, rows, cols, args.spacing_mm)
        stats = summarize(errs, args.threshold_mm, args.metric)
        report = {"mode": "grid", "spacing_error_mm": stats}
    elif args.left and args.right:
        if not args.calib:
            parser.error("回放模式需给出 --calib")
        board = board_from_args(args)
        calib = load_calibration(args.calib)
        img_l = cv2.imread(args.left, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(args.right, cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            print("错误: 无法读取左右目帧")
            return 2
        report = {"mode": "replay", **replay_board_frame(calib, board, img_l, img_r)}
        for key in ("point_error_mm", "spacing_error_mm"):
            report[key].update(threshold_mm=args.threshold_mm, metric=args.metric)
            report[key]["pass"] = report[key][args.metric] <= args.threshold_mm
    else:
        parser.error("需指定 --points3d（网格模式）或 --left/--right（回放模式）")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["mode"] == "replay":
        ok = report["point_error_mm"]["pass"] and report["spacing_error_mm"]["pass"]
    else:
        ok = report["spacing_error_mm"]["pass"]
    print("判定: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
