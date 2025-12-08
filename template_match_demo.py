"""
Template-based digit + accidental detector demo.

Assumptions:
- Digits (0-7) and accidentals (#/b) have consistent sizes across test cases.
- Accidentals appear at the upper-left of their digit.
- Templates are stored as individual images in a folder:
  digit templates: "0.png" ... "7.png"
  accidental templates: "sharp.png", "flat.png"

Usage:
  - Set IMAGE_PATH and TEMPLATE_DIR in main()
  - Run: python template_match_demo.py
  - Press space to toggle debug overlay; Enter to save overlay; other keys exit.
"""

import os
import re
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_templates(template_dir: str) -> Dict[str, np.ndarray]:
    names = [str(i) for i in range(8)] + [
        "sharp",
        "flat",
        "slur_left1",
        "slur_right1",
        "slur_left2",
        "slur_right2",
        "slur_left3",
        "slur_right3",
        "dot",
    ]
    templates = {}
    for name in names:
        path = os.path.join(template_dir, f"{name}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: template not found: {path}")
            continue
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img = np.dstack([img, alpha])
        templates[name] = img
    return templates


def prepare_template(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (tpl, mask)：
    - tpl: 取反后的灰度（让笔画为高值，背景为低值），背景处置为 0，避免纯白区域高响应
    - mask: 前景区域 mask
    """
    if img.shape[2] == 3:
        bgr = img
        alpha = None
    else:
        bgr, alpha = img[..., :3], img[..., 3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if alpha is None:
        mask = (gray < 200).astype(np.uint8) * 255  # 将非白背景视为前景
    else:
        mask = (alpha > 0).astype(np.uint8) * 255
    tpl = (255 - gray).astype(np.uint8)
    tpl = cv2.bitwise_and(tpl, tpl, mask=mask)
    return tpl, mask


def match_template_single(gray: np.ndarray, tpl: np.ndarray, mask: np.ndarray, thresh: float, top_k: int = 200) -> List[Tuple[int, int, int, int, float]]:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(tpl) == 0:
        return []
    # mask 仅保留笔画，避免大片背景产生高响应
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    # 取局部极大值，避免满屏平原
    dilated = cv2.dilate(res, np.ones((3, 3), np.uint8))
    peaks = (res >= thresh) & (res == dilated)
    ys, xs = np.where(peaks)
    h, w = tpl.shape[:2]
    scores = [res[y, x] for y, x in zip(ys, xs)]
    # 取前 top_k
    idx_sorted = np.argsort(scores)[::-1]
    matches = []
    for idx in idx_sorted[:top_k]:
        s_val = float(scores[idx])
        if not np.isfinite(s_val):
            continue
        x, y = int(xs[idx]), int(ys[idx])
        matches.append((x, y, w, h, s_val))
    return matches


def find_replacement_groups(template_dir: str) -> List[Tuple[str, str, str, int, int]]:
    """
    返回 [(id, from_path, to_path, x_off, y_off)]，id 为 A-Z。
    命名规则（不区分大小写的扩展名，但此处用 .png 举例）：
      - fromA.png: 作为查找模板
      - toA.png 或 toA{x_offset}_{y_offset}.png: 作为替换图，中心对齐到查找到的位置，并额外施加偏移。
        偏移为整数，可为正负。若仅有一个偏移或格式不符，则忽略偏移并输出 warning。
    如果 from/to 缺失则输出 warning，不加入有效组。
    """
    entries = os.listdir(template_dir)
    from_map: Dict[str, str] = {}
    to_map: Dict[str, Tuple[str, Optional[int], Optional[int]]] = {}
    pat_from = re.compile(r"from([A-Z])\.png$")
    # 命名规则 to{id}[{x},{y}].png，[] 内偏移可选且成对出现，逗号分隔
    pat_to = re.compile(r"to([A-Z])(?:\[(?P<x>[+-]?\d+),(?P<y>[+-]?\d+)\])?\.png$")
    for name in entries:
        m = pat_from.match(name)
        if m:
            from_map[m.group(1)] = os.path.join(template_dir, name)
            continue
        m = pat_to.match(name)
        if m:
            x_raw, y_raw = m.group("x"), m.group("y")
            x_off = int(x_raw) if x_raw is not None else None
            y_off = int(y_raw) if y_raw is not None else None
            to_map[m.group(1)] = (os.path.join(template_dir, name), x_off, y_off)
            continue

    groups: List[Tuple[str, str, str, int, int]] = []
    all_ids = sorted(set(from_map.keys()) | set(to_map.keys()))
    for gid in all_ids:
        fpath = from_map.get(gid)
        tinfo = to_map.get(gid)
        if not fpath or not tinfo:
            print(f"Warning: replacement group {gid} incomplete (from={bool(fpath)}, to={bool(tinfo)})")
            continue
        tpath, x_off, y_off = tinfo
        if (x_off is None) ^ (y_off is None):
            print(f"Warning: replacement group {gid} has invalid offset format in {os.path.basename(tpath)}, offsets ignored")
            x_off = y_off = 0
        if x_off is None and y_off is None:
            x_off = y_off = 0
        groups.append((gid, fpath, tpath, x_off, y_off))
    if groups:
        print("Replacement groups:", ", ".join([g[0] for g in groups]))
    else:
        print("No complete replacement groups found.")
    return groups


def blend_alpha(dst: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """dst/fg: float [0,1], shape (...,4)"""
    dst_rgb, dst_a = dst[..., :3], dst[..., 3:4]
    fg_rgb, fg_a = fg[..., :3], fg[..., 3:4]
    out_a = fg_a + dst_a * (1.0 - fg_a)
    out_rgb = fg_rgb * fg_a + dst_rgb * dst_a * (1.0 - fg_a)
    out_rgb = np.where(out_a > 1e-6, out_rgb / out_a, 0)
    return np.concatenate([out_rgb, out_a], axis=2)


def apply_replacements(
    base_img: np.ndarray,
    gray_inv: np.ndarray,
    groups: List[Tuple[str, str, str, int, int]],
    thresh: float,
    top_k: int = 100,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float, str]]]:
    """根据 fromX/toX 模板执行查找替换，返回 (替换后的 BGR, 替换框列表)。"""
    h_img, w_img = base_img.shape[:2]
    canvas = np.dstack([base_img.astype(np.float32) / 255.0, np.ones((h_img, w_img, 1), dtype=np.float32)])
    replace_boxes: List[Tuple[int, int, int, int, float, str]] = []

    for gid, from_path, to_path, add_x, add_y in groups:
        from_img = cv2.imread(from_path, cv2.IMREAD_UNCHANGED)
        to_img = cv2.imread(to_path, cv2.IMREAD_UNCHANGED)
        if from_img is None or to_img is None:
            print(f"Warning: cannot read replacement templates for {gid}")
            continue
        if from_img.shape[2] == 3:
            from_img = np.dstack([from_img, np.ones(from_img.shape[:2], dtype=np.uint8) * 255])
        if to_img.shape[2] == 3:
            to_img = np.dstack([to_img, np.ones(to_img.shape[:2], dtype=np.uint8) * 255])

        tpl_gray, tpl_mask = prepare_template(from_img)
        matches = match_template_single(gray_inv, tpl_gray, tpl_mask, thresh, top_k=top_k)
        if not matches:
            continue

        to_float = to_img.astype(np.float32) / 255.0
        tw, th = to_img.shape[1], to_img.shape[0]
        fw, fh = tpl_gray.shape[1], tpl_gray.shape[0]

        for x, y, w, h, s in matches:
            # 扩大匹配框 1 像素用于清理底图（宽高各+2）
            pad = 1
            x_pad = max(0, x - pad)
            y_pad = max(0, y - pad)
            w_pad = min(w_img, x + w + pad) - x_pad
            h_pad = min(h_img, y + h + pad) - y_pad

            cx = x + fw / 2.0
            cy = y + fh / 2.0
            tl_x = int(round(cx - tw / 2.0 + add_x))
            tl_y = int(round(cy - th / 2.0 + add_y))
            br_x = tl_x + tw
            br_y = tl_y + th
            clip_x1 = max(0, tl_x)
            clip_y1 = max(0, tl_y)
            clip_x2 = min(w_img, br_x)
            clip_y2 = min(h_img, br_y)
            if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                continue
            fg_x1 = clip_x1 - tl_x
            fg_y1 = clip_y1 - tl_y
            fg_x2 = fg_x1 + (clip_x2 - clip_x1)
            fg_y2 = fg_y1 + (clip_y2 - clip_y1)

            fg_crop = to_float[fg_y1:fg_y2, fg_x1:fg_x2, :]
            # 先将目标区域刷白（覆盖原有内容）
            canvas[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad, :] = 1.0
            dst_roi = canvas[clip_y1:clip_y2, clip_x1:clip_x2, :]
            blended = blend_alpha(dst_roi, fg_crop)
            canvas[clip_y1:clip_y2, clip_x1:clip_x2, :] = blended
            replace_boxes.append((tl_x, tl_y, tw, th, s, gid))

    return (canvas[..., :3] * 255.0).clip(0, 255).astype(np.uint8), replace_boxes


def nms(boxes: List[Tuple[int, int, int, int, float]], iou_thresh: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        x, y, w, h, s = boxes.pop(0)
        keep.append((x, y, w, h, s))
        boxes = [
            (x2, y2, w2, h2, s2)
            for (x2, y2, w2, h2, s2) in boxes
            if iou((x, y, w, h), (x2, y2, w2, h2)) < iou_thresh
        ]
    return keep


def iou(b1, b2) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def run_matching(
    image_path: str,
    template_dir: str,
    thresh_digit: float = 0.9,
    thresh_acc: float = 0.9,
    thresh_dot: float = 0.9,
    thresh_slur1: float = 0.9,
    thresh_slur2: float = 0.9,
    thresh_slur3: float = 0.9,
    thresh_replace: float = 0.9,
) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int, int, int, float]]], np.ndarray]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = 255 - gray_img  # 让黑字变成高值，匹配取反模板
    templates = load_templates(template_dir)
    replace_groups = find_replacement_groups(template_dir)

    results: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
    all_boxes: List[Tuple[int, int, int, int, float, str]] = []
    # Digits
    for name in [str(i) for i in range(8)]:
        tpl_img = templates.get(name)
        if tpl_img is None:
            continue
        tpl_gray, tpl_mask = prepare_template(tpl_img)
        matches = match_template_single(gray_inv, tpl_gray, tpl_mask, thresh_digit)
        for m in matches:
            all_boxes.append((*m, name))
    # Accidentals
    for name, thresh in [
        ("sharp", thresh_acc),
        ("flat", thresh_acc),
        ("dot", thresh_dot),
        ("slur_left1", thresh_slur1),
        ("slur_right1", thresh_slur1),
        ("slur_left2", thresh_slur2),
        ("slur_right2", thresh_slur2),
        ("slur_left3", thresh_slur3),
        ("slur_right3", thresh_slur3),
    ]:
        tpl_img = templates.get(name)
        if tpl_img is None:
            continue
        tpl_gray, tpl_mask = prepare_template(tpl_img)
        matches = match_template_single(gray_inv, tpl_gray, tpl_mask, thresh)
        for m in matches:
            all_boxes.append((*m, name))

    # 全局 NMS 抑制跨类重叠
    if all_boxes:
        all_boxes = sorted(all_boxes, key=lambda b: b[4], reverse=True)
        kept: List[Tuple[int, int, int, int, float, str]] = []
        while all_boxes:
            x, y, w, h, s, cls = all_boxes.pop(0)
            kept.append((x, y, w, h, s, cls))
            all_boxes = [
                (x2, y2, w2, h2, s2, cls2)
                for (x2, y2, w2, h2, s2, cls2) in all_boxes
                if iou((x, y, w, h), (x2, y2, w2, h2)) < 0.3
            ]
        for x, y, w, h, s, cls in kept:
            results.setdefault(cls, []).append((x, y, w, h, s))
    replaced_img, replace_boxes = apply_replacements(img, gray_inv, replace_groups, thresh_replace, top_k=100)
    return img, results, replaced_img, replace_boxes


def visualize(
    img: np.ndarray,
    results: Dict[str, List[Tuple[int, int, int, int, float]]],
    replace_boxes: Optional[List[Tuple[int, int, int, int, float, str]]] = None,
) -> np.ndarray:
    colors = {
        "digit": (0, 0, 255),
        "sharp": (0, 165, 255),  # 橘色
        "flat": (255, 0, 0),
        "dot": (0, 255, 255),  # 黄
        "slur_left1": (0, 128, 0),   # 绿
        "slur_right1": (128, 0, 128),  # 紫
        "slur_left2": (0, 255, 0),   # 亮绿
        "slur_right2": (255, 0, 255),  # 亮紫
        "slur_left3": (0, 200, 200),   # 青
        "slur_right3": (200, 0, 200),  # 洋红
        "replace": (50, 180, 255),     # 替换框颜色
    }
    vis = img.copy()
    for name, boxes in results.items():
        if name in ["sharp", "flat"]:
            color = colors[name]
        elif name == "dot":
            color = colors["dot"]
        elif name in ["slur_left1", "slur_right1"]:
            color = colors[name]
        else:
            color = colors["digit"]
        for x, y, w, h, s in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"{name}:{s:.2f}", (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if replace_boxes:
        for x, y, w, h, s, gid in replace_boxes:
            color = colors["replace"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis, f"rep-{gid}:{s:.2f}", (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


def main():
    IMAGE_PATH = "pu11.png"          # TODO: replace with your test image
    TEMPLATE_DIR = "templates"     # TODO: directory containing 0.png..7.png, sharp.png, flat.png, dot.png, slur_left1.png, slur_right1.png, slur_left2.png, slur_right2.png, slur_left3.png, slur_right3.png, and optional replacement pairs fromA.png/toA.png...
    SAVE_PATH = "template_debug3.png"

    img, results, replaced, replace_boxes = run_matching(IMAGE_PATH, TEMPLATE_DIR)
    vis = visualize(img, results, replace_boxes=replace_boxes)

    mode = 1  # 0:原图, 1:检测框+替换框, 2:替换结果
    while True:
        if mode == 0:
            img_show = img
        elif mode == 1:
            img_show = vis
        else:
            img_show = replaced
        cv2.imshow("template_demo", img_show)
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # space toggle
            mode = (mode + 1) % 3
            continue
        if key in (13, 10):  # enter to save
            cv2.imwrite(SAVE_PATH, img_show)
            print(f"Saved debug image to {os.path.abspath(SAVE_PATH)}")
        break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
