import cv2
import numpy as np
import os
from notation_detect import DetectionResult


def export_notation(
    target_path: str,
    target_resolution: tuple,
    source_offset: tuple,
    detection: DetectionResult,
    bg_path: str = None,
    fingering_img_path: str = None,
    fingering_img_offset: tuple = (0, 0),
    fingering_scale: float = 1.0,
) -> None:
    """
    将源图渲染到指定分辨率的透明画布上，并按键控制保存。

    参数：
        target_path: 输出图片保存路径。
        target_resolution: (width, height) 输出图片尺寸。
        source_offset: (x_offset, y_offset) 源图左上角在输出中的偏移。
        detection: extract_notes 返回的 DetectionResult 对象。
        bg_path: 背景图片路径；为空则使用不透明白底。
        fingering_img_path: 指法图目录，为空则不渲染指法图。
        fingering_img_offset: (x_offset, y_offset)，指法图顶边中点相对音符中心/行的偏移。
        fingering_scale: 指法图缩放系数，默认 1.0。
    """
    width, height = target_resolution  # 解析目标画布的宽度和高度
    x_off, y_off = source_offset  # 解析源图在目标画布上的偏移量

    # 目标画布背景（float32，0-1）
    if bg_path:
        bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        if bg is None:
            raise FileNotFoundError(f"Cannot read background image: {bg_path}")
        if bg.shape[2] < 3:
            raise ValueError("Background image must have at least 3 channels.")
        bg_bgr = bg[..., :3]
        bg_h, bg_w = bg_bgr.shape[:2]
        # 生成平铺/裁剪到目标尺寸
        rep_y = (height + bg_h - 1) // bg_h
        rep_x = (width + bg_w - 1) // bg_w
        tiled = np.tile(bg_bgr, (rep_y, rep_x, 1))
        bg_canvas = tiled[:height, :width, :]
        canvas_rgb = bg_canvas.astype(np.float32) / 255.0
        canvas_a = np.ones((height, width, 1), dtype=np.float32)  # 背景不透明
    else:
        canvas_rgb = np.ones((height, width, 3), dtype=np.float32)  # 白色
        canvas_a = np.ones((height, width, 1), dtype=np.float32)    # 不透明
    canvas = np.concatenate([canvas_rgb, canvas_a], axis=2)

    # 读取源图
    src = cv2.imread(detection.image_path, cv2.IMREAD_UNCHANGED)
    if src is None:
        raise FileNotFoundError(f"Cannot read image: {detection.image_path}")

    # 使用源图自带的透明度；若无 alpha 则视为不透明
    if src.shape[2] == 3:
        bgr = src
        alpha = np.full(src.shape[:2], 255, dtype=np.uint8)
    elif src.shape[2] == 4:
        bgr = src[..., :3]
        alpha = src[..., 3]
    else:
        raise ValueError("Unsupported source image channels.")
    src = np.dstack([bgr, alpha])

    src_h, src_w = src.shape[:2]  # 获取源图的高度和宽度

    # 计算可放置区域
    x1 = max(0, x_off)  # 计算目标画布上的起始x坐标
    y1 = max(0, y_off)  # 计算目标画布上的起始y坐标
    x2 = min(width, x_off + src_w)  # 计算目标画布上的结束x坐标
    y2 = min(height, y_off + src_h)  # 计算目标画布上的结束y坐标

    if x1 >= x2 or y1 >= y2:
        raise ValueError("Source image is completely outside the target canvas.")

    # 对应源图裁剪
    src_x1 = x1 - x_off
    src_y1 = y1 - y_off
    src_crop = src[src_y1:src_y1 + (y2 - y1), src_x1:src_x1 + (x2 - x1), :]

    # 混合：背景(dst) 与源(src)，公式：out_a = 1 - (1 - dst_a)*(1 - src_a)，out_c = dst_c * src_c
    dst_roi = canvas[y1:y2, x1:x2, :]
    src_roi = src_crop.astype(np.float32) / 255.0
    dst_rgb, dst_a = dst_roi[..., :3], dst_roi[..., 3:4]
    src_rgb, src_a = src_roi[..., :3], src_roi[..., 3:4]

    out_a = 1.0 - (1.0 - dst_a) * (1.0 - src_a)
    out_rgb = dst_rgb * src_rgb
    out = np.concatenate([out_rgb, out_a], axis=2)

    canvas[y1:y2, x1:x2, :] = out

    # 指法图渲染（仅音头）
    def blend_alpha(dst: np.ndarray, fg: np.ndarray):
        dst_rgb, dst_a = dst[..., :3], dst[..., 3:4]
        fg_rgb, fg_a = fg[..., :3], fg[..., 3:4]
        out_a = fg_a + dst_a * (1.0 - fg_a)
        out_rgb = fg_rgb * fg_a + dst_rgb * dst_a * (1.0 - fg_a)
        # 去预乘
        out_rgb = np.where(out_a > 1e-6, out_rgb / out_a, 0)
        return np.concatenate([out_rgb, out_a], axis=2)

    if fingering_img_path and detection.row_center_y is not None:
        cache = {}
        fx_off, fy_off = fingering_img_offset
        for note in detection.notes:
            if not note.articulation:
                continue
            fname = f"{note.pitch}_0_off.png"
            fpath = os.path.join(fingering_img_path, fname)
            if fpath in cache:
                fing_img = cache[fpath]
            else:
                if not os.path.exists(fpath):
                    print(f"Warning: fingering image not found: {fpath}")
                    cache[fpath] = None
                    continue
                img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Warning: cannot read fingering image: {fpath}")
                    cache[fpath] = None
                    continue
                if img.shape[2] == 3:
                    alpha_f = np.ones(img.shape[:2], dtype=np.uint8) * 255
                    img = np.dstack([img, alpha_f])
                elif img.shape[2] == 4:
                    pass
                else:
                    print(f"Warning: unsupported channels in fingering image: {fpath}")
                    cache[fpath] = None
                    continue
                if fingering_scale != 1.0:
                    new_w = max(1, int(round(img.shape[1] * fingering_scale)))
                    new_h = max(1, int(round(img.shape[0] * fingering_scale)))
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                cache[fpath] = img
                fing_img = img
            if fing_img is None:
                continue

            fg_h, fg_w = fing_img.shape[:2]
            cx_note = source_offset[0] + note.x + note.w // 2 + fx_off
            top_y = source_offset[1] + detection.row_center_y + fy_off
            tl_x = int(round(cx_note - fg_w / 2))
            tl_y = int(round(top_y))
            br_x = tl_x + fg_w
            br_y = tl_y + fg_h

            # 裁剪到画布
            clip_x1 = max(0, tl_x)
            clip_y1 = max(0, tl_y)
            clip_x2 = min(target_resolution[0], br_x)
            clip_y2 = min(target_resolution[1], br_y)
            if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                continue

            fg_x1 = clip_x1 - tl_x
            fg_y1 = clip_y1 - tl_y
            fg_x2 = fg_x1 + (clip_x2 - clip_x1)
            fg_y2 = fg_y1 + (clip_y2 - clip_y1)

            fg_crop = fing_img[fg_y1:fg_y2, fg_x1:fg_x2, :].astype(np.float32) / 255.0
            dst_roi = canvas[clip_y1:clip_y2, clip_x1:clip_x2, :]
            blended = blend_alpha(dst_roi, fg_crop)
            canvas[clip_y1:clip_y2, clip_x1:clip_x2, :] = blended

    # 显示窗口
    canvas_uint8 = (canvas * 255.0).clip(0, 255).astype(np.uint8)

    colors = [
        (0, 0, 255),     # 红
        (0, 165, 255),   # 橙
        (255, 0, 0),     # 蓝
        (0, 100, 0),     # 深绿
        (255, 0, 255),   # 紫
    ]

    def render(mode: int) -> np.ndarray:
        """
        mode 0: 仅背景+源图
        mode 1: 原始调试（所有轮廓框）
        mode 2: 最终结果调试（主行、音符框、搜索区、点）
        """
        img = canvas_uint8.copy()
        x_off, y_off = source_offset
        if mode == 0:
            return img
        if mode == 1:
            for orig_id, x, y, w, h in detection.raw_boxes:
                color = colors[(orig_id - 1) % len(colors)]
                pt1 = (x_off + x, y_off + y)
                pt2 = (x_off + x + w, y_off + y + h)
                cv2.rectangle(img, pt1, pt2, color, 1)
                cv2.putText(
                    img,
                    str(orig_id),
                    (x_off + x, max(10 + y_off, y_off + y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            return img
        # mode 2
        if detection.row_center_y is not None:
            y_line = y_off + detection.row_center_y
            cv2.line(img, (0, y_line), (img.shape[1], y_line), (255, 0, 255), 1)
        for idx, note in enumerate(detection.notes, 1):
            color = colors[(idx - 1) % len(colors)]
            for sx1, sy1, sx2, sy2 in note.search_regions:
                cv2.rectangle(
                    img,
                    (x_off + sx1, y_off + sy1),
                    (x_off + sx2, y_off + sy2),
                    color,
                    1,
                )
            pt1 = (x_off + note.x, y_off + note.y)
            pt2 = (x_off + note.x + note.w, y_off + note.h + note.y)
            thickness = 2 if note.articulation else 1
            cv2.rectangle(img, pt1, pt2, color, thickness)
            for dot in note.dots:
                cv2.circle(img, (x_off + dot[0], y_off + dot[1]), radius=3, color=color, thickness=-1)
        return img

    mode = 2  # 默认显示最终调试
    base_output = render(0)  # 仅背景+源图，作为保存用的基准

    while True:
        img_show = render(mode)
        cv2.imshow("export", img_show)
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # 空格键切换模式
            mode = (mode + 1) % 3
            continue
        if key in (13, 10):  # Enter 保存
            cv2.imwrite(target_path, base_output)
            print(f"Saved exported image to: {os.path.abspath(target_path)}")
            break
        else:
            break
    cv2.destroyAllWindows()
