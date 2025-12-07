from notation_detect import DetectParams, extract_notes
from notation_renderer import export_notation


def main():
    # 示例参数，可按需调整
    image_path = "jianpu5.png"  # 替换成你的简谱图片路径
    key = "G"               # 调号，形如 C, G, Bb, F# 等
    target_path = "output1.png"
    target_resolution = (1920, 540)  # (width, height)
    source_offset = (200, 50)           # (x_offset, y_offset)
    bg_path = "bg.jpeg"

    params = DetectParams(
        upscale=1.0,                 # 全图 OCR 前的放大倍率（保留入口，当前未使用额外放大）
        area_min_factor=0.2,         # 连通域面积下限（相对中位数），过小判噪声
        area_max_factor=4.5,         # 连通域面积上限（相对中位数），过大判非数字
        width_height_ratio_max=1.0,  # 连通域过宽剔除阈值：w > h * ratio 认为是连线/括号
        height_ratio_max=2.0,        # 连通域过高剔除阈值：h > median_h_all * ratio 认为是小节线
        row_tolerance_factor=0.6,    # 行聚类 y 容差（系数 * 数字中位高）
        pad_ratio=0.15,              # OCR 时为候选框添加的 padding（相对 max(w,h)）
        ocr_scale=3.0,               # 单字符 OCR 放大倍率
        ocr_dilate_kernel=2,         # OCR fallback 时膨胀核尺寸（像素）
        search_width_factor=0.75,    # 点搜索区域宽度（相对数字高度）
        margin_y_factor=0.01,        # 点搜索区域上下边距（相对数字高度）
        search_height_factor=1.25,   # 点搜索区域高度（相对数字高度）
        max_dot_size_factor=0.35,    # 认定为点的最大宽/高（相对数字高度）
        cx_tol_factor=0.2,           # 点中心与数字中心的水平容差（相对数字高度）
        slur_ratio=2.0,              # 判定为连音线的宽高比阈值
    )

    result = extract_notes(image_path, params=params, key=key)

    print(f"Detected {len(result.notes)} notes:")
    for i, n in enumerate(result.notes, 1):
        cx, cy = n.center()
        print(
            f"{i:2d}. degree={n.degree}, "
            f"octave_offset={n.octave_offset:+d}, "
            f"bbox=({n.x},{n.y},{n.w},{n.h}), "
            f"center=({cx},{cy}), "
            f"pitch={n.pitch}, "
            f"articulation={'head' if n.articulation else 'tie'}"
        )
    export_notation(
        target_path=target_path,
        target_resolution=target_resolution,
        source_offset=source_offset,
        detection=result,
        bg_path=bg_path,
    )


if __name__ == "__main__":
    main()
