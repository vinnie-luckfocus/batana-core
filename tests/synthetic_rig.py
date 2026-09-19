"""合成双目装置：已知内外参的虚拟相机 + 虚拟 ChArUco 板渲染。

用于无硬件的端到端验证：渲染标定帧 → 走真实检测/标定/三角化/评估链路。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VirtualCamera:
    K: np.ndarray          # 3×3 内参
    d: np.ndarray          # 5 畸变系数
    R: np.ndarray          # 世界(左目)→本目点旋转
    T: np.ndarray          # 世界(左目)→本目点平移 (3,1) mm
    size: tuple[int, int]  # (w, h)
    _mapx: np.ndarray | None = None
    _mapy: np.ndarray | None = None

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """投影（含畸变），返回 (N,2) 像素坐标。"""
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        rvec, _ = cv2.Rodrigues(self.R)
        img, _ = cv2.projectPoints(pts3d, rvec, self.T, self.K, self.d)
        return img.reshape(-1, 2)

    def _distort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        if self._mapx is None:
            w, h = self.size
            uv = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).reshape(-1, 2)
            uv = np.ascontiguousarray(uv, dtype=np.float64)
            ideal = cv2.undistortPoints(uv, self.K, self.d, P=self.K).reshape(h, w, 2)
            self._mapx = ideal[..., 0].astype(np.float32)
            self._mapy = ideal[..., 1].astype(np.float32)
        return self._mapx, self._mapy

    def distort_image(self, pinhole: np.ndarray) -> np.ndarray:
        """把理想针孔图像正向合成为带畸变图像。"""
        mapx, mapy = self._distort_maps()
        return cv2.remap(pinhole, mapx, mapy, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)


class VirtualStereoRig:
    """水平双目装置：左目位于世界原点，右目光心 +x 方向。"""

    def __init__(
        self,
        baseline_mm: float = 100.0,
        size: tuple[int, int] = (1280, 800),
        fx: float = 1100.0,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        w, h = size
        K1 = np.array([[fx, 0, w / 2 - 0.5], [0, fx + 1.0, h / 2 + 1.2], [0, 0, 1]], dtype=np.float64)
        K2 = np.array([[fx + 2.0, 0, w / 2 + 0.8], [0, fx + 1.5, h / 2 - 0.6], [0, 0, 1]], dtype=np.float64)
        d1 = np.array([0.05, -0.02, 8e-4, -6e-4, 0.0])
        d2 = np.array([0.048, -0.018, 6e-4, -7e-4, 0.0])
        # 微小安装姿态误差
        R_gt, _ = cv2.Rodrigues(np.array([0.002, 0.008, -0.003]))
        C_right = np.array([baseline_mm, 0.5, -0.3]).reshape(3, 1)  # 右目光心（左目系）
        T_cv = -R_gt @ C_right
        self.left = VirtualCamera(K1, d1, np.eye(3), np.zeros((3, 1)), size)
        self.right = VirtualCamera(K2, d2, R_gt, T_cv, size)
        self.baseline_mm = baseline_mm
        self.R_gt = R_gt
        self.C_right = C_right.ravel()
        self.rng = rng

    def render_board(self, board, rvec: np.ndarray, tvec: np.ndarray, cam: VirtualCamera,
                     texture_px_per_mm: float = 4.0, margin_mm: float = 20.0,
                     noise_sigma: float = 0.5, supersample: int = 4) -> np.ndarray:
        """渲染虚拟标定板到指定相机（超采样针孔 warp + 正向畸变 + 高斯噪声）。

        坐标约定：OpenCV 连续图像坐标以像素中心为整数点，纹理棋盘角边界位于
        (mm+m)*s-0.5；超采样网格 (u,v) → ss*(u+0.5)-0.5。先 warp 到 ss 倍
        针孔图再 INTER_AREA 降采样，避免亚像素混叠引入角点检测系统偏差。
        """
        bw, bh = board.getChessboardSize()  # (格数x, 格数y) → 板面 mm
        s = texture_px_per_mm
        m = margin_mm
        ss = supersample
        tex_w = round((bw * board.getSquareLength() + 2 * m) * s)
        tex_h = round((bh * board.getSquareLength() + 2 * m) * s)
        texture = board.generateImage((tex_w, tex_h), marginSize=round(m * s))

        # 板平面四角（板坐标 mm）→ 理想像素
        W = bw * board.getSquareLength()
        H = bh * board.getSquareLength()
        corners_mm = np.array([[-m, -m, 0], [W + m, -m, 0], [W + m, H + m, 0], [-m, H + m, 0]])
        Rb, _ = cv2.Rodrigues(rvec)
        pts_left = corners_mm @ Rb.T + tvec.ravel()      # 板坐标 → 世界（左目）系
        pts_cam = pts_left @ cam.R.T + cam.T.ravel()     # 世界 → 本目相机系
        ideal = (cam.K @ pts_cam.T).T
        ideal = ideal[:, :2] / ideal[:, 2:3]
        src = ((corners_mm[:, :2] + m) * s - 0.5).astype(np.float32)
        dst = ((ideal + 0.5) * ss - 0.5).astype(np.float32)

        w, h = cam.size
        pinhole = np.full((h * ss, w * ss), 255, dtype=np.uint8)
        Hmat = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(texture, Hmat, (w * ss, h * ss), dst=pinhole,
                            flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        if ss > 1:
            pinhole = cv2.resize(pinhole, (w, h), interpolation=cv2.INTER_AREA)
        img = cam.distort_image(pinhole)
        if noise_sigma > 0:
            noise = np.random.default_rng().normal(0, noise_sigma, img.shape)
            img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        return img

    def sample_board_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """采样一个两目共视的板位姿（板心 1.6–2.4m，姿态 ±25°）。"""
        rvec = np.array([
            np.deg2rad(self.rng.uniform(-25, 25)),
            np.deg2rad(self.rng.uniform(-25, 25)),
            np.deg2rad(self.rng.uniform(-10, 10)),
        ])
        tvec = np.array([
            self.rng.uniform(-250, 250),
            self.rng.uniform(-150, 150),
            self.rng.uniform(1600, 2400),
        ])
        return rvec, tvec

    def board_fully_visible(self, board, rvec, tvec) -> bool:
        bw, bh = board.getChessboardSize()
        W = bw * board.getSquareLength()
        H = bh * board.getSquareLength()
        corners = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], dtype=np.float64)
        Rb, _ = cv2.Rodrigues(rvec)
        pts_world = corners @ Rb.T + tvec.ravel()
        for cam in (self.left, self.right):
            uv = cam.project(pts_world)
            w, h = cam.size
            if (uv[:, 0] < 20).any() or (uv[:, 0] > w - 20).any() or \
               (uv[:, 1] < 20).any() or (uv[:, 1] > h - 20).any():
                return False
        return True
