"""双目三角化：标定结果 + 左右目 2D 对应点 → 3D 点（左目坐标系，mm）。

外参约定与 calibrate_stereo 输出一致：文件内 translation_mm 为右目光心位置，
内部还原 OpenCV 点变换 T_cv = -R @ translation_mm 后做三角化。

输入对应点文件（JSON 或 CSV）：
- JSON: {"points": [[xl, yl, xr, yr], ...]} 或 {"left": [[x, y], ...], "right": [[x, y], ...]}
- CSV: 每行 xl,yl,xr,yr（首行允许表头）

批量模式（--batch）输入 JSON：
- {"frames": [{"frame_index": int, "timestamp_ms": float,
   "left": [[x, y], ...], "right": [[x, y], ...]}, ...]}
  每帧左右目点数须一致（允许任意点数，按数组顺序一一对应）。

用法：
    python -m tools.recon3d.triangulate --calib calibration.json --points pairs.json --out points3d.json
    python -m tools.recon3d.triangulate --calib calibration.json --batch frames.json --out frames3d.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


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


def triangulate_frames(calib: dict, frames: list[dict]) -> list[dict]:
    """批量三角化：逐帧调 triangulate_points，返回带帧元信息的结果列表。

    frames 每项含 frame_index / timestamp_ms / left / right（左右目点数须一致）。
    返回 [{"frame_index", "timestamp_ms", "points_3d": [[x, y, z], ...]}]（mm）。
    """
    results: list[dict] = []
    for i, f in enumerate(frames):
        pl = np.asarray(f["left"], dtype=np.float64).reshape(-1, 2)
        pr = np.asarray(f["right"], dtype=np.float64).reshape(-1, 2)
        if pl.shape != pr.shape:
            raise ValueError(
                f"帧 {f.get('frame_index', i)} 左右目点数不一致: "
                f"{len(pl)} vs {len(pr)}"
            )
        pts3d = triangulate_points(calib, pl, pr)
        results.append({
            "frame_index": int(f.get("frame_index", i)),
            "timestamp_ms": float(f.get("timestamp_ms", 0.0)),
            "points_3d": pts3d.tolist(),
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
    args = parser.parse_args(argv)

    calib = load_calibration(args.calib)
    if args.batch:
        frames = load_batch_frames(args.batch)
        results = triangulate_frames(calib, frames)
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
