"""生成 ChArUco 标定板打印图。

默认参数对应 batana V0c 标定板：11×8 格、格子 100mm（板面 1100×800mm），
A0 幅面（1189×841mm）300dpi 打印。输出 PNG 并打印物理尺寸核对信息。

用法：
    python -m tools.calibration.gen_board --out charuco_11x8_100mm.png
    python -m tools.calibration.gen_board --square-mm 110 --dpi 300 --out board.png
"""

from __future__ import annotations

import argparse

import cv2

from .board import add_board_args, board_from_args

MM_PER_INCH = 25.4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 ChArUco 标定板打印图（PNG）")
    add_board_args(parser)
    parser.add_argument("--dpi", type=int, default=300, help="输出分辨率 dpi（默认 300）")
    parser.add_argument("--margin-mm", type=float, default=15.0, help="四周留白 mm（默认 15，需覆盖 ArUco quiet zone）")
    parser.add_argument("--out", required=True, help="输出 PNG 路径")
    args = parser.parse_args(argv)

    board = board_from_args(args)
    board_w_mm = args.squares_x * args.square_mm
    board_h_mm = args.squares_y * args.square_mm
    margin_px = round(args.margin_mm / MM_PER_INCH * args.dpi)
    w_px = round(board_w_mm / MM_PER_INCH * args.dpi)
    h_px = round(board_h_mm / MM_PER_INCH * args.dpi)

    img = board.generateImage((w_px, h_px), marginSize=margin_px)
    cv2.imwrite(args.out, img)

    total_w_mm = board_w_mm + 2 * args.margin_mm
    total_h_mm = board_h_mm + 2 * args.margin_mm
    print(f"标定板: {args.squares_x}x{args.squares_y} 格, 格子 {args.square_mm}mm, "
          f"marker {args.square_mm * args.marker_ratio:.1f}mm, 字典 {args.dictionary}")
    print(f"板面: {board_w_mm:.0f}x{board_h_mm:.0f}mm, 含留白: {total_w_mm:.0f}x{total_h_mm:.0f}mm")
    print(f"图像: {img.shape[1]}x{img.shape[0]}px @ {args.dpi}dpi -> {args.out}")
    for name, (pw, ph) in {"A0": (1189, 841), "A1": (841, 594)}.items():
        fit = "✓ 可放入" if (total_w_mm <= pw and total_h_mm <= ph) or (total_w_mm <= ph and total_h_mm <= pw) else "✗ 超出"
        print(f"幅面核对 {name} ({pw}x{ph}mm): {fit}")
    print("提示: 打印时请关闭「适应页面」缩放，打印后用直尺抽查格子边长是否等于设定值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
