"""
批量图层混合小工具。

用法示例：
  python layer_blending.py ./layers ./background.png ./out blended

参数：
  1) layers_dir    ：包含若干作为上层的 PNG 图片的目录。
  2) bottom_path   ：作为下层的 PNG 图片路径。
  3) output_dir    ：输出目录，不存在则自动创建。
  4) suffix        ：输出文件名的附加后缀，输出命名为 {layer_name}_{suffix}.png。

行为：
  - 仅处理 .png 文件。
  - 若图像尺寸或通道数不匹配，输出警告并跳过。
  - 使用标准 alpha over 公式：out_a = fg_a + bg_a*(1-fg_a)，out_rgb = fg_rgb*fg_a + bg_rgb*bg_a*(1-fg_a)；结果自动除以 out_a。
"""

import os
import sys
import cv2
import numpy as np


def load_bgra(path: str):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    if img.shape[2] == 3:
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img = np.dstack([img, alpha])
    return img


def blend_alpha(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """底层 bottom，被 top 覆盖；输入均为 float32 [0,1] BGRA，输出同规格。"""
    bg_rgb, bg_a = bottom[..., :3], bottom[..., 3:4]
    fg_rgb, fg_a = top[..., :3], top[..., 3:4]
    out_a = fg_a + bg_a * (1.0 - fg_a)
    out_rgb = fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)
    out_rgb = np.where(out_a > 1e-6, out_rgb / out_a, 0)
    return np.concatenate([out_rgb, out_a], axis=2)


def main():
    if len(sys.argv) < 5:
        print("用法: python layer_blending.py <layers_dir> <bottom_path> <output_dir> <suffix>")
        return
    layers_dir = sys.argv[1]
    bottom_path = sys.argv[2]
    output_dir = sys.argv[3]
    suffix = sys.argv[4]

    if not os.path.isdir(layers_dir):
        print(f"错误: layers_dir 不存在或不是目录: {layers_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    try:
        bottom = load_bgra(bottom_path)
    except FileNotFoundError as e:
        print(str(e))
        return
    h0, w0 = bottom.shape[:2]
    bottom_f = bottom.astype(np.float32) / 255.0

    files = [f for f in os.listdir(layers_dir) if f.lower().endswith(".png")]
    if not files:
        print("提示: layers_dir 中没有 PNG 文件。")
        return

    for name in files:
        layer_path = os.path.join(layers_dir, name)
        try:
            top = load_bgra(layer_path)
        except FileNotFoundError as e:
            print(str(e))
            continue
        if top.shape[:2] != (h0, w0):
            print(f"跳过: 尺寸不匹配 {name}, 期望 {w0}x{h0}, 实际 {top.shape[1]}x{top.shape[0]}")
            continue
        top_f = top.astype(np.float32) / 255.0
        blended = blend_alpha(bottom_f, top_f)
        out = (blended * 255.0).clip(0, 255).astype(np.uint8)
        stem = os.path.splitext(name)[0]
        out_name = f"{stem}_{suffix}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, out)
        print(f"输出: {out_path}")


if __name__ == "__main__":
    main()
