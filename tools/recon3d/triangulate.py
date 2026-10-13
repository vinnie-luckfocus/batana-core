"""双目三角化：标定结果 + 左右目 2D 对应点 → 3D 点（左目坐标系，mm）。

外参约定与 calibrate_stereo 输出一致：文件内 translation_mm 为右目光心位置，
内部还原 OpenCV 点变换 T_cv = -R @ translation_mm 后做三角化。

输入对应点文件（JSON 或 CSV）：
- JSON: {"points": [[xl, yl, xr, yr], ...]} 或 {"left": [[x, y], ...], "right": [[x, y], ...]}
- CSV: 每行 xl,yl,xr,yr（首行允许表头）

批量模式（--batch）输入 JSON：
- {"frames": [{"frame_index": int, "timestamp_ms": float,
   "left": [[x, y], ...], "right": [[x, y], ...],
   "left_visibility": [v, ...], "right_visibility": [v, ...]}, ...]}
  每帧左右目点数须一致（允许任意点数，按数组顺序一一对应）；
  visibility 数组可选（与坐标等长，[0,1]），缺省视为全部 1.0（向后兼容）。

批量模式前处理参数：
- --min-vis FLOAT（默认 0.5）：任一目 visibility 低于阈值的点不三角化，输出 null；
- --smooth N（奇数窗口，默认 1=不平滑）：三角化前对每个关节的每目 2D 轨迹做
  时序中值滤波（只基于未掩蔽帧，被掩蔽帧保持掩蔽）；
- --epipolar-correct：用标定推导基础矩阵 F，对每对对应点做 cv2.correctMatches
  最优极线校正后再三角化。

用法：
    python -m tools.recon3d.triangulate --calib calibration.json --points pairs.json --out points3d.json
    python -m tools.recon3d.triangulate --calib calibration.json --batch frames.json --out frames3d.json \
        --min-vis 0.5 --smooth 5 --epipolar-correct
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tools.recon3d.preprocess import (
    correct_correspondences,
    fundamental_matrix,
    median_smooth,
)


def load_calibration(path: str | Path) -> dict:
    """加载标定 JSON，返回 numpy 化的内参/畸变/外参（T_cv 为 OpenCV 点变换约定，mm）。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stereo = data["stereo"]
    R = np.asarray(stereo["extrinsic"]["rotation"], dtype=np.float64)
    t_pos = np.asarray(stereo["extrinsic"]["translation_mm"], dtype=np.float64)
    return {
        "K1": np.asarray(stereo["left"]["intrinsic"], dtype=np.float64),
        "d1": np.asarray(stereo["left"]["distortion"], dtype=np.float64),
        "K2": np.asarray(stereo["right"]["intrinsic"], dtype=np.float64),
        "d2": np.asarray(stereo["right"]["distortion"], dtype=np.float64),
        "R": R,
        "T": (-R @ t_pos).reshape(3, 1),
        "resolution": tuple(stereo["left"]["resolution"]),
    }


def triangulate_points(calib: dict, pts_left: np.ndarray, pts_right: np.ndarray) -> np.ndarray:
    """三角化左右目对应点，返回 (N, 3)，左目坐标系，单位 mm。"""
    pl = np.ascontiguousarray(pts_left, dtype=np.float64).reshape(-1, 2)
    pr = np.ascontiguousarray(pts_right, dtype=np.float64).reshape(-1, 2)
    if pl.shape != pr.shape:
        raise ValueError(f"左右目点数不一致: {pl.shape} vs {pr.shape}")
    # 去畸变到归一化坐标
    nl = cv2.undistortPoints(pl, calib["K1"], calib["d1"]).reshape(-1, 2)
    nr = cv2.undistortPoints(pr, calib["K2"], calib["d2"]).reshape(-1, 2)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([calib["R"], calib["T"]])
    hom = cv2.triangulatePoints(P1, P2, nl.T, nr.T)
    pts3d = (hom[:3] / hom[3]).T
    return pts3d


def triangulate_frames(
    calib: dict,
    frames: list[dict],
    min_vis: float = 0.5,
    smooth: int = 1,
    epipolar_correct: bool = False,
) -> list[dict]:
    """批量三角化：逐帧调 triangulate_points，返回带帧元信息的结果列表。

    frames 每项含 frame_index / timestamp_ms / left / right（左右目点数须一致），
    可选 left_visibility / right_visibility（与坐标等长；缺省视为全部 1.0）。
    任一目 visibility < min_vis 的点不三角化，输出 points_3d 对应位置为 None（JSON null）。
    smooth > 1 时对每关节每目 2D 轨迹做时序中值滤波（要求各帧点数一致）；
    epipolar_correct=True 时对有效对应点做 cv2.correctMatches 极线校正。
    返回 [{"frame_index", "timestamp_ms", "points_3d": [[x, y, z] | None, ...]}]（mm）。
    """
    if smooth > 1 and smooth % 2 == 0:
        raise ValueError(f"平滑窗口须为奇数: {smooth}")

    T = len(frames)
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    vis_l: list[np.ndarray] = []
    vis_r: list[np.ndarray] = []
    for i, f in enumerate(frames):
        pl = np.asarray(f["left"], dtype=np.float64).reshape(-1, 2)
        pr = np.asarray(f["right"], dtype=np.float64).reshape(-1, 2)
        fi = f.get("frame_index", i)
        if pl.shape != pr.shape:
            raise ValueError(
                f"帧 {fi} 左右目点数不一致: {len(pl)} vs {len(pr)}"
            )
        n = len(pl)
        vl = np.asarray(f.get("left_visibility", [1.0] * n), dtype=np.float64).ravel()
        vr = np.asarray(f.get("right_visibility", [1.0] * n), dtype=np.float64).ravel()
        if len(vl) != n or len(vr) != n:
            raise ValueError(
                f"帧 {fi} visibility 与坐标数量不一致: {len(vl)}/{len(vr)} vs {n}"
            )
        lefts.append(pl)
        rights.append(pr)
        vis_l.append(vl)
        vis_r.append(vr)

    uniform = len({len(p) for p in lefts}) <= 1
    if smooth > 1 and not uniform:
        raise ValueError("时序平滑要求各帧点数一致（关节轨迹需对齐）")

    # 平滑：只基于任一目均达 min_vis 的有效帧；掩蔽帧的值平滑后仍被掩蔽（最终掩码不变）
    if smooth > 1 and T > 0:
        valid = (np.stack(vis_l) >= min_vis) & (np.stack(vis_r) >= min_vis)
        ls = median_smooth(np.stack(lefts), valid, smooth)
        rs = median_smooth(np.stack(rights), valid, smooth)
        lefts, rights = list(ls), list(rs)

    F = fundamental_matrix(calib) if epipolar_correct else None

    results: list[dict] = []
    for i, f in enumerate(frames):
        pl, pr = lefts[i], rights[i]
        valid = (vis_l[i] >= min_vis) & (vis_r[i] >= min_vis)
        pts3d: list = [None] * len(pl)
        idx = np.flatnonzero(valid)
        if idx.size:
            sel_l, sel_r = pl[idx], pr[idx]
            if F is not None:
                sel_l, sel_r = correct_correspondences(F, sel_l, sel_r)
            tri = triangulate_points(calib, sel_l, sel_r)
            for k, j in enumerate(idx):
                pts3d[j] = tri[k].tolist()
        results.append({
            "frame_index": int(f.get("frame_index", i)),
            "timestamp_ms": float(f.get("timestamp_ms", 0.0)),
            "points_3d": pts3d,
        })
    return results


def load_batch_frames(path: str | Path) -> list[dict]:
    """加载批量模式输入 JSON，校验基本结构后返回帧列表。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"批量输入需含非空 frames 数组: {path}")
    for i, f in enumerate(frames):
        if not isinstance(f, dict) or "left" not in f or "right" not in f:
            raise ValueError(f"frames[{i}] 需含 left/right 点数组")
        for key in ("left_visibility", "right_visibility"):
            if key in f and len(f[key]) != len(f["left"]):
                raise ValueError(
                    f"frames[{i}].{key} 长度须与坐标一致: {len(f[key])} vs {len(f['left'])}"
                )
    return frames


def load_point_pairs(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if "points" in data:
            arr = np.asarray(data["points"], dtype=np.float64)
            return arr[:, :2], arr[:, 2:4]
        return np.asarray(data["left"], dtype=np.float64), np.asarray(data["right"], dtype=np.float64)
    rows: list[list[float]] = []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            try:
                rows.append([float(v) for v in row])
            except ValueError:
                continue  # 跳过表头
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"CSV 需为每行 xl,yl,xr,yr 四列: {p}")
    return arr[:, :2], arr[:, 2:4]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双目三角化：2D 对应点 → 3D 点（左目坐标系，mm）")
    parser.add_argument("--calib", required=True, help="calibrate_stereo 输出的标定 JSON")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--points", help="单组对应点文件（JSON 或 CSV）")
    src.add_argument("--batch", help="批量对应点 JSON（frames 数组，逐帧三角化）")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（缺省打印到 stdout）")
    parser.add_argument("--min-vis", type=float, default=0.5,
                        help="批量模式：任一目 visibility 低于该值的点输出 null（默认 0.5）")
    parser.add_argument("--smooth", type=int, default=1,
                        help="批量模式：关节 2D 轨迹时序中值滤波窗口（奇数，默认 1=不平滑）")
    parser.add_argument("--epipolar-correct", action="store_true",
                        help="批量模式：三角化前对对应点做 cv2.correctMatches 极线校正")
    args = parser.parse_args(argv)

    calib = load_calibration(args.calib)
    if args.batch:
        frames = load_batch_frames(args.batch)
        results = triangulate_frames(
            calib, frames,
            min_vis=args.min_vis, smooth=args.smooth,
            epipolar_correct=args.epipolar_correct,
        )
        result = {
            "frame": "left_camera",
            "unit": "mm",
            "frames": results,
        }
        text = json.dumps(result, indent=2, ensure_ascii=False)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"三角化 {len(results)} 帧 -> {args.out}")
        else:
            print(text)
        return 0

    pl, pr = load_point_pairs(args.points)
    pts3d = triangulate_points(calib, pl, pr)
    result = {
        "frame": "left_camera",
        "unit": "mm",
        "count": int(len(pts3d)),
        "points_3d": pts3d.tolist(),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"三角化 {len(pts3d)} 点 -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
