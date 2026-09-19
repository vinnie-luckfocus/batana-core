"""双目标定 CLI：ChArUco 标定帧 → 单目 + 双目标定，输出标定结果文件。

输入一目录标定帧：
- 2560×800 side-by-side 整帧（左右各 1280×800）自动切半；
- 或文件名含 left/right（大小写不敏感，亦识别 _l/_r 后缀）的成对分图；
- 或用 --left-dir/--right-dir 显式指定两个目录（按文件名排序配对）。

输出 JSON 结构与 docs/contracts/session-schema.md 的 `calibration` 字段对齐
（format_version / stereo.baseline_mm / left / right / extrinsic），另附 report
小节记录重投影误差等标定质量信息（契约允许消费方忽略未知字段）。

外参约定：extrinsic.rotation 为左目→右目的点旋转（OpenCV stereoCalibrate 的 R）；
extrinsic.translation_mm 为右目光心在左目坐标系中的位置（= -Rᵀ·T_cv），
水平双目模组下应接近 [+baseline, 0, 0]，与契约示例一致。
OpenCV 点变换约定 T_cv 可由 T_cv = -R @ translation_mm 还原。

用法：
    python -m tools.calibration.calibrate_stereo frames/ --out-json calib.json --out-yaml calib.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .board import add_board_args, board_from_args

MIN_CORNERS = 6  # 单帧最少有效 ChArUco 角点数
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SBS_ASPECT = 2.0  # 宽/高 ≥ 2×1.6 附近即视为 side-by-side 整帧（2560/800=3.2）


@dataclass
class ViewDetections:
    """单目相机一组帧的角点观测。"""

    image_size: tuple[int, int]  # (宽, 高)
    obj_points: list[np.ndarray] = field(default_factory=list)
    img_points: list[np.ndarray] = field(default_factory=list)
    corner_ids: list[np.ndarray] = field(default_factory=list)
    frame_names: list[str] = field(default_factory=list)

    def append(self, obj: np.ndarray, img: np.ndarray, ids: np.ndarray, name: str) -> None:
        self.obj_points.append(obj)
        self.img_points.append(img)
        self.corner_ids.append(ids)
        self.frame_names.append(name)


def detect_board(detector: cv2.aruco.CharucoDetector, board, image: np.ndarray):
    """检测 ChArUco 角点；不足 MIN_CORNERS 时返回 None。"""
    corners, ids, _, _ = detector.detectBoard(image)
    if ids is None or len(ids) < MIN_CORNERS:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj, img, ids


def _is_left_name(name: str) -> bool:
    n = name.lower()
    return "left" in n or re.search(r"_l\.[a-z]+$", n) is not None


def _is_right_name(name: str) -> bool:
    n = name.lower()
    return "right" in n or re.search(r"_r\.[a-z]+$", n) is not None


def load_frame_pairs(
    input_dir: Path,
    left_dir: Path | None,
    right_dir: Path | None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """加载左右目图像对，返回 [(帧名, 左图, 右图)]（灰度）。"""
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []

    def read_gray(p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {p}")
        return img

    if left_dir or right_dir:
        if not (left_dir and right_dir):
            raise ValueError("--left-dir 与 --right-dir 必须同时指定")
        lefts = sorted(p for p in left_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        rights = sorted(p for p in right_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if len(lefts) != len(rights) or not lefts:
            raise ValueError(f"左右目录帧数不一致或为空: {len(lefts)} vs {len(rights)}")
        for lp, rp in zip(lefts, rights):
            pairs.append((lp.stem, read_gray(lp), read_gray(rp)))
        return pairs

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not files:
        raise ValueError(f"目录中无图像文件: {input_dir}")
    left_map = {p.stem: p for p in files if _is_left_name(p.name)}
    right_map = {p.stem: p for p in files if _is_right_name(p.name)}
    if left_map or right_map:
        # 分图模式：按去前缀后的公共键配对
        def key(name: str) -> str:
            return re.sub(r"(left|right|_l$|_r$)", "", name.lower()).strip("_- ")

        right_by_key = {key(k): v for k, v in right_map.items()}
        for lname, lp in sorted(left_map.items()):
            rp = right_by_key.get(key(lname))
            if rp is None:
                raise ValueError(f"左目帧无配对右目帧: {lp.name}")
            pairs.append((lname, read_gray(lp), read_gray(rp)))
        return pairs

    # side-by-side 整帧自动切半
    for p in files:
        img = read_gray(p)
        h, w = img.shape
        if w / h < SBS_ASPECT:
            raise ValueError(
                f"{p.name} 尺寸 {w}x{h} 既非 side-by-side 整帧（宽/高≥{SBS_ASPECT}），"
                "文件名也不含 left/right 标记，无法判定左右目"
            )
        half = w // 2
        pairs.append((p.stem, img[:, :half], img[:, half:]))
    return pairs


def collect_detections(pairs, board) -> tuple[ViewDetections, ViewDetections]:
    """对所有帧对做角点检测，返回左右目观测集合。"""
    detector = cv2.aruco.CharucoDetector(board)
    left = ViewDetections(image_size=(0, 0))
    right = ViewDetections(image_size=(0, 0))
    skipped = 0
    for name, img_l, img_r in pairs:
        if left.image_size == (0, 0):
            h, w = img_l.shape
            left.image_size = right.image_size = (w, h)
        det_l = detect_board(detector, board, img_l)
        det_r = detect_board(detector, board, img_r)
        if det_l is None or det_r is None:
            skipped += 1
            continue
        left.append(*det_l, name)
        right.append(*det_r, name)
    if skipped:
        print(f"警告: {skipped} 帧角点不足（<{MIN_CORNERS}）已跳过")
    if len(left.frame_names) < 3:
        raise ValueError(f"有效标定帧过少（{len(left.frame_names)}），至少需要 3 帧不同位姿")
    return left, right


def calibrate_mono(det: ViewDetections) -> tuple[float, np.ndarray, np.ndarray]:
    """单目标定，返回 (RMS 重投影误差 px, K, dist)。"""
    ret, K, dist, _, _ = cv2.calibrateCamera(det.obj_points, det.img_points, det.image_size, None, None)
    return float(ret), K, dist


def calibrate_stereo_pair(
    board, left: ViewDetections, right: ViewDetections, K1, d1, K2, d2
) -> tuple[float, np.ndarray, np.ndarray, int]:
    """双目标定（固定单目内参）。返回 (立体对 RMS px, R, T_cv, 使用帧数)。

    R、T_cv 为 OpenCV 点变换约定：p_right = R @ p_left + T_cv（单位随板坐标，此处 mm）。
    """
    obj_pts, img_l, img_r = [], [], []
    chess = board.getChessboardCorners()
    for lo, li, lid, ro, ri, rid in zip(
        left.obj_points, left.img_points, left.corner_ids,
        right.obj_points, right.img_points, right.corner_ids,
    ):
        common = np.intersect1d(lid, rid)
        if len(common) < MIN_CORNERS:
            continue
        lm = {int(c): i for i, c in enumerate(lid)}
        rm = {int(c): i for i, c in enumerate(rid)}
        obj_pts.append(chess[common].reshape(-1, 1, 3).astype(np.float32))
        img_l.append(np.stack([li[lm[int(c)]] for c in common]).reshape(-1, 1, 2).astype(np.float32))
        img_r.append(np.stack([ri[rm[int(c)]] for c in common]).reshape(-1, 1, 2).astype(np.float32))
    if not obj_pts:
        raise ValueError("左右目无共同角点帧，无法做双目标定")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_pts, img_l, img_r, K1, d1, K2, d2, left.image_size, flags=flags
    )
    return float(ret), R, T, len(obj_pts)


def build_result(
    left: ViewDetections, right: ViewDetections,
    rms_l, rms_r, rms_s, K1, d1, K2, d2, R, T,
    board, squares_x: int, squares_y: int, square_mm: float,
) -> dict:
    """组装对齐 session-schema calibration 字段的结果字典。"""
    t_pos = (-R.T @ T).ravel()  # 右目光心在左目坐标系中的位置
    baseline = float(np.linalg.norm(T))
    w, h = left.image_size
    return {
        "format_version": "1.0",
        "stereo": {
            "baseline_mm": baseline,
            "left": {
                "resolution": [w, h],
                "intrinsic": K1.tolist(),
                "distortion": d1.ravel().tolist(),
            },
            "right": {
                "resolution": [w, h],
                "intrinsic": K2.tolist(),
                "distortion": d2.ravel().tolist(),
            },
            "extrinsic": {
                "rotation": R.tolist(),
                "translation_mm": t_pos.tolist(),
            },
        },
        "report": {
            "convention": "rotation 为左目→右目点旋转（OpenCV R）；translation_mm 为右目光心在左目坐标系中的位置（-Rᵀ·T_cv）；T_cv = -R @ translation_mm",
            "board": {"type": "charuco", "squares_x": squares_x, "squares_y": squares_y, "square_mm": square_mm},
            "views_used": len(left.frame_names),
            "reprojection_error_px": {"left": rms_l, "right": rms_r, "stereo": rms_s},
        },
    }


def run(args) -> dict:
    board = board_from_args(args)
    input_dir = Path(args.input)
    pairs = load_frame_pairs(
        input_dir,
        Path(args.left_dir) if args.left_dir else None,
        Path(args.right_dir) if args.right_dir else None,
    )
    print(f"加载帧对: {len(pairs)}")
    left, right = collect_detections(pairs, board)
    print(f"有效帧: {len(left.frame_names)}（单目分辨率 {left.image_size[0]}x{left.image_size[1]}）")

    rms_l, K1, d1 = calibrate_mono(left)
    rms_r, K2, d2 = calibrate_mono(right)
    rms_s, R, T, n_stereo = calibrate_stereo_pair(board, left, right, K1, d1, K2, d2)

    result = build_result(left, right, rms_l, rms_r, rms_s, K1, d1, K2, d2, R, T, board,
                          args.squares_x, args.squares_y, args.square_mm)
    print(f"重投影误差 px: 左目 {rms_l:.4f} / 右目 {rms_r:.4f} / 立体对 {rms_s:.4f}（{n_stereo} 帧）")
    print(f"基线: {result['stereo']['baseline_mm']:.2f} mm")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON 已写出: {args.out_json}")
    if args.out_yaml:
        Path(args.out_yaml).write_text(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"YAML 已写出: {args.out_yaml}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="双目 ChArUco 标定：输出对齐 session-schema calibration 字段的结果文件")
    parser.add_argument("input", help="标定帧目录（SBS 整帧或 left/right 分图）")
    parser.add_argument("--left-dir", help="左目帧目录（与 --right-dir 配合使用）")
    parser.add_argument("--right-dir", help="右目帧目录")
    add_board_args(parser)
    parser.add_argument("--out-json", default="calibration.json", help="JSON 输出路径（默认 calibration.json）")
    parser.add_argument("--out-yaml", default="calibration.yaml", help="YAML 输出路径（默认 calibration.yaml）")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
