"""
Template match & replace utility.

Usage:
  python template_match_replace.py <notation_img_path> <templates_path>

Modes (space to toggle):
  0: 原图
  1: 调试可视化（绘制所有检测框，包含替换组的 from 框）
  2: 替换结果（应用所有替换组后的图片，不显示调试框）
Enter 保存替换结果（不含调试框），命名为 {乐谱文件名}_replaced.png；同时保存匹配信息 JSON 为 {乐谱文件名}_matchinfo.json，路径与乐谱同级。

模板命名规则（目录中所有图片都会被解析，如有不符合规则的会 warning）：
  A 类替换组：
    from_{ID}_[thresh].png   ID 为 A-Z；thresh 可选，浮点；未提供则默认 0.9
    to_{ID}_[x_offset,y_offset].png  偏移可选且必须成对出现，整数，可正负；未提供则偏移 0
    若 from/to 组不完整或偏移格式不符，输出 warning
  B 类常规模板：
    {name}_[thresh].png  name 不含方括号且不可为单个大写字母；thresh 可选，浮点，默认 0.9

可将更多模板放入目录，无需修改代码。
"""

import os
import re
import json
import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

# 支持的图片扩展名
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp", ".PNG", ".JPG", ".JPEG")

# 调试颜色（稍暗，便于白底可见）
COLOR_PALETTE = [
    (0, 0, 180),
    (0, 120, 180),
    (180, 120, 0),
    (120, 0, 180),
    (0, 160, 100),
    (160, 80, 0),
    (100, 100, 180),
    (180, 60, 120),
]


def load_image_bgra(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.shape[2] == 3:
        alpha = np.ones(img.shape[:2], dtype=np.uint8) * 255
        img = np.dstack([img, alpha])
    return img


def prepare_template(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (tpl, mask)，取反灰度并用 alpha/灰度生成前景 mask，避免背景触发匹配。"""
    bgr, alpha = img[..., :3], img[..., 3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (alpha > 0).astype(np.uint8) * 255
    tpl = (255 - gray).astype(np.uint8)
    tpl = cv2.bitwise_and(tpl, tpl, mask=mask)
    return tpl, mask


def match_template_single(
    gray: np.ndarray,
    tpl: np.ndarray,
    mask: np.ndarray,
    thresh: float,
    top_k: int = 200,
) -> List[Tuple[int, int, int, int, float]]:
    if np.count_nonzero(mask) == 0 or np.count_nonzero(tpl) == 0:
        return []
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    dilated = cv2.dilate(res, np.ones((3, 3), np.uint8))
    peaks = (res >= thresh) & (res == dilated)
    ys, xs = np.where(peaks)
    h, w = tpl.shape[:2]
    scores = [res[y, x] for y, x in zip(ys, xs)]
    idx_sorted = np.argsort(scores)[::-1]
    matches = []
    for idx in idx_sorted[:top_k]:
        s_val = float(scores[idx])
        if not np.isfinite(s_val):
            continue
        x, y = int(xs[idx]), int(ys[idx])
        matches.append((x, y, w, h, s_val))
    return matches


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


def global_nms(boxes: List[Tuple[int, int, int, int, float, str]], iou_thresh: float = 0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    while boxes:
        x, y, w, h, s, cls = boxes.pop(0)
        kept.append((x, y, w, h, s, cls))
        boxes = [
            (x2, y2, w2, h2, s2, cls2)
            for (x2, y2, w2, h2, s2, cls2) in boxes
            if iou((x, y, w, h), (x2, y2, w2, h2)) < iou_thresh
        ]
    return kept


def blend_alpha(dst: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """dst/fg: float [0,1], shape (...,4)，标准 alpha over。"""
    dst_rgb, dst_a = dst[..., :3], dst[..., 3:4]
    fg_rgb, fg_a = fg[..., :3], fg[..., 3:4]
    out_a = fg_a + dst_a * (1.0 - fg_a)
    out_rgb = fg_rgb * fg_a + dst_rgb * dst_a * (1.0 - fg_a)
    out_rgb = np.where(out_a > 1e-6, out_rgb / out_a, 0)
    return np.concatenate([out_rgb, out_a], axis=2)


def parse_templates(template_dir: str, default_thresh: float = 0.9):
    """
    扫描模板目录，返回：
      replace_groups: {id: {"from": (path, thresh), "to": (path, x_off, y_off)}}
      regular_templates: {name: (path, thresh)}
    不符合规则的文件输出 warning。
    """
    replace_groups: Dict[str, Dict[str, Tuple[str, float, int, int]]] = {}
    regular_templates: Dict[str, Tuple[str, float]] = {}

    pat_from = re.compile(r"from_([A-Z])(?:_\[?([+-]?\d+(?:\.\d+)?)\]?)?\.png$", re.IGNORECASE)
    pat_to = re.compile(r"to_([A-Z])(?:_\[([+-]?\d+),([+-]?\d+)\])?\.png$", re.IGNORECASE)
    pat_regular = re.compile(r"([^ \[\]]+?)(?:_\[?([+-]?\d+(?:\.\d+)?)\]?)?\.png$", re.IGNORECASE)

    for fname in os.listdir(template_dir):
        fpath = os.path.join(template_dir, fname)
        if not fname.lower().endswith(".png"):
            continue
        m_from = pat_from.match(fname)
        if m_from:
            gid = m_from.group(1).upper()
            thresh = float(m_from.group(2)) if m_from.group(2) else default_thresh
            replace_groups.setdefault(gid, {})["from"] = (fpath, thresh, 0, 0)
            continue
        m_to = pat_to.match(fname)
        if m_to:
            gid = m_to.group(1).upper()
            x_off = int(m_to.group(2)) if m_to.group(2) else 0
            y_off = int(m_to.group(3)) if m_to.group(3) else 0
            replace_groups.setdefault(gid, {})["to"] = (fpath, default_thresh, x_off, y_off)
            continue
        m_reg = pat_regular.match(fname)
        if m_reg:
            name = m_reg.group(1)
            if re.fullmatch(r"[A-Z]", name):
                print(f"Warning: skip template '{fname}' (name cannot be single uppercase letter)")
                continue
            thresh = float(m_reg.group(2)) if m_reg.group(2) else default_thresh
            regular_templates[name] = (fpath, thresh)
            continue
        print(f"Warning: filename does not match template rules, skipped: {fname}")

    # 校验替换组完整性
    valid_replace = {}
    for gid, info in replace_groups.items():
        if "from" not in info or "to" not in info:
            print(f"Warning: replacement group {gid} incomplete (from/to missing), skipped")
            continue
        valid_replace[gid] = info
    replace_groups = valid_replace
    if replace_groups:
        print("Replacement groups:", ", ".join(sorted(replace_groups.keys())))
    else:
        print("No complete replacement groups found.")
    return replace_groups, regular_templates


def apply_replacements(
    base_img: np.ndarray,
    gray_inv: np.ndarray,
    replace_groups: Dict[str, Dict[str, Tuple[str, float, int, int]]],
    top_k: int = 100,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int, float, str]]]:
    """对所有替换组执行查找+替换，返回 (替换后 BGR, 替换框列表)。"""
    h_img, w_img = base_img.shape[:2]
    canvas = np.dstack([base_img.astype(np.float32) / 255.0, np.ones((h_img, w_img, 1), dtype=np.float32)])
    replace_boxes: List[Tuple[int, int, int, int, float, str]] = []

    for gid, info in replace_groups.items():
        from_path, thresh, _, _ = info["from"]
        to_path, _, add_x, add_y = info["to"]
        from_img = load_image_bgra(from_path)
        to_img = load_image_bgra(to_path)
        tpl_gray, tpl_mask = prepare_template(from_img)
        matches = match_template_single(gray_inv, tpl_gray, tpl_mask, thresh, top_k=top_k)
        if not matches:
            continue

        to_float = to_img.astype(np.float32) / 255.0
        tw, th = to_img.shape[1], to_img.shape[0]
        fw, fh = tpl_gray.shape[1], tpl_gray.shape[0]

        for x, y, w, h, s in matches:
            # 1) 扩大 from 框 1 像素并刷白，避免与 to 重叠残留
            pad = 1
            x_pad = max(0, x - pad)
            y_pad = max(0, y - pad)
            w_pad = min(w_img, x + w + pad) - x_pad
            h_pad = min(h_img, y + h + pad) - y_pad
            canvas[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad, :] = 1.0  # 白色不透明

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
            dst_roi = canvas[clip_y1:clip_y2, clip_x1:clip_x2, :]
            blended = blend_alpha(dst_roi, fg_crop)
            canvas[clip_y1:clip_y2, clip_x1:clip_x2, :] = blended
            replace_boxes.append((tl_x, tl_y, tw, th, s, gid))

    return (canvas[..., :3] * 255.0).clip(0, 255).astype(np.uint8), replace_boxes


def visualize(
    base_img: np.ndarray,
    results: Dict[str, List[Tuple[int, int, int, int, float]]],
    replace_boxes: Optional[List[Tuple[int, int, int, int, float, str]]],
    color_map: Dict[str, Tuple[int, int, int]],
) -> np.ndarray:
    vis = base_img.copy()
    for name, boxes in results.items():
        color = color_map.get(name, color_map.get("default", (0, 0, 200)))
        for x, y, w, h, s in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            score_txt = f"{int(round(s * 100))}"
            cv2.putText(vis, f"{name}:{score_txt}", (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # 调试模式不绘制替换后的框，避免把 to 视为检测结果
    return vis


def save_match_info(
    json_path: str,
    results: Dict[str, List[Tuple[int, int, int, int, float]]],
    replace_boxes: List[Tuple[int, int, int, int, float, str]],
    src_img_path: str,
    replaced_img_path: str,
):
    """按名称汇总所有检测到的位置并写入 JSON。"""
    data: Dict[str, List[Dict[str, float]]] = {}
    # regular / dots / slur etc.（不包含 from 框）
    for name, boxes in results.items():
        sorted_boxes = sorted(boxes, key=lambda b: b[0])
        data[name] = [
            {"x": x, "y": y, "w": w, "h": h, "center": [x + w / 2.0, y + h / 2.0], "score": s}
            for x, y, w, h, s in sorted_boxes
        ]
    # replacements: 用实际替换后的位置尺寸，名称为组 id
    rep_map: Dict[str, List[Dict[str, float]]] = {}
    for x, y, w, h, s, gid in replace_boxes:
        rep_map.setdefault(gid, []).append(
            {"x": x, "y": y, "w": w, "h": h, "center": [x + w / 2.0, y + h / 2.0], "score": s}
        )
    for k, v in rep_map.items():
        data[k] = sorted(v, key=lambda b: b["x"])

    output = {
        "src_img_path": os.path.abspath(src_img_path),
        "replaced_img_path": os.path.abspath(replaced_img_path),
        "matches": data,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved match info to {json_path}")


def build_color_map(names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    cmap = {"default": (80, 80, 200), "replace": (60, 180, 180)}
    palette_len = len(COLOR_PALETTE)
    for i, name in enumerate(sorted(names)):
        cmap[name] = COLOR_PALETTE[i % palette_len]
    return cmap


def run_template_match_replace(notation_img_path: str, templates_path: str):
    """
    执行模板匹配/替换。
    返回 (replaced_img_path, matchinfo_json_path, saved_flag)；若未按回车保存则返回 (None, None, False)。
    """
    base_img = cv2.imread(notation_img_path, cv2.IMREAD_COLOR)
    if base_img is None:
        print(f"错误: 无法读取乐谱图片 {notation_img_path}")
        return None, None, False

    replace_groups, regular_templates = parse_templates(templates_path)

    gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray_inv = 255 - gray

    results_debug: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
    results_json: Dict[str, List[Tuple[int, int, int, int, float]]] = {}
    all_boxes: List[Tuple[int, int, int, int, float, str]] = []

    # 常规模板匹配
    for name, (path, thresh) in regular_templates.items():
        tpl_img = load_image_bgra(path)
        tpl_gray, tpl_mask = prepare_template(tpl_img)
        matches = match_template_single(gray_inv, tpl_gray, tpl_mask, thresh)
        for m in matches:
            all_boxes.append((*m, name))

    # 替换组 from 匹配（用于调试显示）
    for gid, info in replace_groups.items():
        from_path, thresh, _, _ = info["from"]
        tpl_img = load_image_bgra(from_path)
        tpl_gray, tpl_mask = prepare_template(tpl_img)
        matches = match_template_single(gray_inv, tpl_gray, tpl_mask, thresh)
        for m in matches:
            all_boxes.append((*m, gid))  # 调试用，label 为组 id

    # 全局 NMS
    kept = global_nms(all_boxes, iou_thresh=0.3)
    for x, y, w, h, s, cls in kept:
        results_debug.setdefault(cls, []).append((x, y, w, h, s))
        if cls not in replace_groups:  # 仅常规模板进入 JSON
            results_json.setdefault(cls, []).append((x, y, w, h, s))

    # 应用替换（用完整匹配再跑一次，避免 NMS 削弱替换数量）
    replaced_img, replace_boxes = apply_replacements(base_img, gray_inv, replace_groups, top_k=100)

    # 调试可视化颜色映射
    color_map = build_color_map(list(results_debug.keys()) + ["replace"])
    vis_img = visualize(base_img, results_debug, None, color_map)

    saved = False
    out_img_path = None
    json_path = None
    mode = 1  # 0:原图, 1:调试框, 2:替换结果
    while True:
        if mode == 0:
            show = base_img
        elif mode == 1:
            show = vis_img
        else:
            show = replaced_img
        cv2.imshow("template_match_replace", show)
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # space
            mode = (mode + 1) % 3
            continue
        if key in (13, 10):  # enter
            base_dir = os.path.dirname(os.path.abspath(notation_img_path))
            stem = os.path.splitext(os.path.basename(notation_img_path))[0]
            out_img_path = os.path.join(base_dir, f"{stem}_replaced.png")
            cv2.imwrite(out_img_path, replaced_img)
            print(f"Saved replaced image to {out_img_path}")
            json_path = os.path.join(base_dir, f"{stem}_matchinfo.json")
            save_match_info(json_path, results_json, replace_boxes, notation_img_path, out_img_path)
            saved = True
        break
    cv2.destroyAllWindows()
    return out_img_path, json_path, saved


def main():
    if len(sys.argv) == 1:
        print("用法: python template_match_replace.py <notation_img_path> <templates_path>")
        print("说明: 模板目录中的文件需按约定命名，参见脚本顶部注释。")
        return
    if len(sys.argv) < 3:
        print("错误: 参数不足。示例: python template_match_replace.py ./score.png ./templates")
        return

    notation_img_path = sys.argv[1]
    templates_path = sys.argv[2]
    _, _, _ = run_template_match_replace(notation_img_path, templates_path)


if __name__ == "__main__":
    main()
