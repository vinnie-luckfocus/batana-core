"""ChArUco 标定板共享定义。

batana 双目标定统一使用 ChArUco 板：默认 11×8 格、格子 100mm、
marker 边长为格子的 0.75 倍，字典 DICT_4X4_100（11×8 板需 44 个 marker，容量足够）。
所有几何量单位为毫米（mm）。
"""

from __future__ import annotations

import cv2

DEFAULT_SQUARES_X = 11
DEFAULT_SQUARES_Y = 8
DEFAULT_SQUARE_MM = 100.0
DEFAULT_MARKER_RATIO = 0.75
DEFAULT_DICTIONARY = "DICT_4X4_100"


def make_board(
    squares_x: int = DEFAULT_SQUARES_X,
    squares_y: int = DEFAULT_SQUARES_Y,
    square_mm: float = DEFAULT_SQUARE_MM,
    marker_ratio: float = DEFAULT_MARKER_RATIO,
    dictionary: str = DEFAULT_DICTIONARY,
) -> cv2.aruco.CharucoBoard:
    """构造 ChArUco 板对象（长度单位 mm，z=0 平面）。"""
    dict_id = getattr(cv2.aruco, dictionary, None)
    if dict_id is None:
        raise ValueError(f"未知 ArUco 字典: {dictionary}")
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_mm,
        square_mm * marker_ratio,
        cv2.aruco.getPredefinedDictionary(dict_id),
    )


def grid_shape(board: cv2.aruco.CharucoBoard) -> tuple[int, int]:
    """棋盘内角点网格形状 (rows, cols)。"""
    sx, sy = board.getChessboardSize()
    return (sy - 1, sx - 1)


def grid_object_points(board: cv2.aruco.CharucoBoard) -> "np.ndarray":
    """全部内角点的 3D 板坐标，形状 (rows*cols, 3)，行优先（y 行 × x 列），单位 mm。"""
    return board.getChessboardCorners()


def add_board_args(parser, prefix: str = "") -> None:
    """向 argparse 追加标定板参数（多个 CLI 共用）。"""
    parser.add_argument(f"--{prefix}squares-x", type=int, default=DEFAULT_SQUARES_X, help="格子列数（默认 11）")
    parser.add_argument(f"--{prefix}squares-y", type=int, default=DEFAULT_SQUARES_Y, help="格子行数（默认 8）")
    parser.add_argument(f"--{prefix}square-mm", type=float, default=DEFAULT_SQUARE_MM, help="格子边长 mm（默认 100）")
    parser.add_argument(f"--{prefix}marker-ratio", type=float, default=DEFAULT_MARKER_RATIO,
                        help="marker 边长 / 格子边长（默认 0.75）")
    parser.add_argument(f"--{prefix}dictionary", default=DEFAULT_DICTIONARY, help="ArUco 字典名（默认 DICT_4X4_100）")


def board_from_args(args, prefix: str = "") -> cv2.aruco.CharucoBoard:
    return make_board(
        squares_x=getattr(args, f"{prefix}squares_x"),
        squares_y=getattr(args, f"{prefix}squares_y"),
        square_mm=getattr(args, f"{prefix}square_mm"),
        marker_ratio=getattr(args, f"{prefix}marker_ratio"),
        dictionary=getattr(args, f"{prefix}dictionary"),
    )
