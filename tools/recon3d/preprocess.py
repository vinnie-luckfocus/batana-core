"""三角化前处理：visibility 掩蔽、时序中值滤波、极线校正。

动机：MediaPipe 逐帧 2D 抖动 + 双目对应点不一致（违反极线约束）会被裸三角化
放大成骨长剧烈波动。三件套：
- 掩蔽：低置信对应点不参与三角化（输出 null）；
- 时序中值滤波：对每个关节的每目 2D 轨迹沿时间轴取窗口中值，压制孤立抖动
  （只基于未被掩蔽的帧；被掩蔽帧平滑后仍保持掩蔽）；
- 极线校正：cv2.correctMatches 把对应点拉到最近极线上，满足极线约束后再三角化。

纯 numpy/cv2，无其他依赖。
"""

from __future__ import annotations

import cv2
import numpy as np


def visibility_mask(
    vis_left: np.ndarray,
    vis_right: np.ndarray,
    min_vis: float,
) -> np.ndarray:
    """双目 visibility → 有效点掩码 (T, N)：任一目低于阈值即掩蔽。"""
    vl = np.asarray(vis_left, dtype=np.float64)
    vr = np.asarray(vis_right, dtype=np.float64)
    if vl.shape != vr.shape:
        raise ValueError(f"左右目 visibility 形状不一致: {vl.shape} vs {vr.shape}")
    return (vl >= min_vis) & (vr >= min_vis)


def median_smooth(
    points: np.ndarray,
    valid: np.ndarray,
    window: int,
) -> np.ndarray:
    """沿时间轴的逐关节中值滤波。

    points: (T, N, 2) 像素轨迹；valid: (T, N) 有效掩码（无效帧不参与中值统计）；
    window: 奇数窗口（1 = 不平滑，原样返回）。窗口内无有效帧时保留原值。
    """
    pts = np.asarray(points, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if window <= 1:
        return pts.copy()
    if window % 2 == 0:
        raise ValueError(f"平滑窗口须为奇数: {window}")
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError(f"points 须为 (T, N, 2): {pts.shape}")
    if mask.shape != pts.shape[:2]:
        raise ValueError(f"valid 形状须为 {pts.shape[:2]}: {mask.shape}")
    T, N, _ = pts.shape
    half = window // 2
    out = pts.copy()
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        wpts = pts[lo:hi]           # (W, N, 2)
        wmask = mask[lo:hi]         # (W, N)
        for j in range(N):
            if not mask[t, j]:
                continue            # 无效帧保留原值（下游仍按掩码排除）
            sel = wpts[wmask[:, j], j]  # 该关节窗口内的有效观测
            if sel.size:
                out[t, j] = np.median(sel, axis=0)
    return out


def fundamental_matrix(calib: dict) -> np.ndarray:
    """由标定推导基础矩阵 F（像素域）：F = K2^-T · [T]× · R · K1^-1。

    calib 为 load_calibration 的返回（T 为 OpenCV 点变换 T_cv，mm）；
    约定 p2ᵀ · F · p1 = 0（p1 左目像素、 p2 右目像素）。
    """
    K1, K2 = calib["K1"], calib["K2"]
    R, T = calib["R"], calib["T"].ravel()
    t_x = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    E = t_x @ R
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def correct_correspondences(
    F: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """cv2.correctMatches 最优极线校正，返回校正后的 (左, 右) 对应点。"""
    pl = np.ascontiguousarray(pts_left, dtype=np.float64).reshape(1, -1, 2)
    pr = np.ascontiguousarray(pts_right, dtype=np.float64).reshape(1, -1, 2)
    cl, cr = cv2.correctMatches(F, pl, pr)
    return cl.reshape(-1, 2), cr.reshape(-1, 2)


def epipolar_rms_error(
    F: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
) -> float:
    """对称极线误差 RMS（像素）：sqrt(mean(d₂² + d₁²)/2)，衡量对应点偏离极线程度。"""
    pl = np.asarray(pts_left, dtype=np.float64).reshape(-1, 2)
    pr = np.asarray(pts_right, dtype=np.float64).reshape(-1, 2)
    n = len(pl)
    if n == 0:
        return 0.0
    hp1 = np.hstack([pl, np.ones((n, 1))])
    hp2 = np.hstack([pr, np.ones((n, 1))])
    l2 = (F @ hp1.T).T   # 左点 → 右目极线
    l1 = (F.T @ hp2.T).T  # 右点 → 左目极线
    d2 = np.abs(np.einsum("ij,ij->i", hp2, l2)) / np.hypot(l2[:, 0], l2[:, 1])
    d1 = np.abs(np.einsum("ij,ij->i", hp1, l1)) / np.hypot(l1[:, 0], l1[:, 1])
    return float(np.sqrt(np.mean((d1 ** 2 + d2 ** 2) / 2)))
