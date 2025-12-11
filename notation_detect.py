import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

"""
从 template_match_replace 生成的 matchinfo.json 中解析数字/符号，计算音符属性并可视化预览。
输出 *_notes.json，字段包含 midi 音高、休止符标记、连音线与行中心等信息，供渲染使用。
"""


@dataclass
class NoteInfo:
    degree: int
    accidental: int | None  # -1 flat, 0 natural, +1 sharp；休止符为 None
    octave_offset: int | None  # 休止符为 None
    pitch: int | None  # midi 编号，休止符为 None
    articulation: bool
    is_rest: bool
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    row_center_y: float
    dot_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    slur_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    acc_box: Tuple[int, int, int, int] = None
    dots_hit: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class DetectParams:
    acc_box_ratio: float = 0.6       # 升降号检测框宽高系数（相对数字高度，右下角对齐数字中心）
    dot_box_ratio_x: float = 0.6     # 上下点检测框宽系数（相对数字高度，贴靠数字上下边沿）
    dot_box_ratio_y: float = 1.2     # 上下点检测框高系数（相对数字高度，贴靠数字上下边沿）
    slur_w_ratio: float = 0.7        # 延音线检测框宽度系数（相对数字高度，位于数字上方）
    slur_h_ratio: float = 2          # 延音线检测框高度系数（相对数字高度，位于数字上方）


def normalize_key(key: str) -> str:
    return key if key in KEY_BASE_MIDI else "C"


# 调号 degree 1 基准 midi（无升降号、无八度点）
KEY_BASE_MIDI = {
    "C": 72,
    "G": 67,
    "D": 62,
    "A": 69,
    "E": 64,
    "B": 71,
    "F#": 66,
    "C#": 73,
    "F": 65,
    "Bb": 70,
    "Eb": 63,
    "Ab": 68,
    "Db": 73,
    "Gb": 66,
    "Cb": 71,
}

# 半音到升号音名（用于显示）
SEMI_TO_NOTE_SHARP = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}

# 度数+升降号相对“1”的半音偏移
INTERVALS = {
    "1b": -1,
    "1": 0,
    "1#": 1,
    "2b": 1,
    "2": 2,
    "2#": 3,
    "3b": 3,
    "3": 4,
    "4b": 4,
    "3#": 5,
    "4": 5,
    "4#": 6,
    "5b": 6,
    "5": 7,
    "5#": 8,
    "6b": 8,
    "6": 9,
    "6#": 10,
    "7b": 10,
    "7": 11,
    "7#": 12,
}


def midi_to_name(midi: int) -> str:
    """返回带升号的科学音名（用 #）。"""
    octave = (midi // 12) - 1
    semi = midi % 12
    return f"{SEMI_TO_NOTE_SHARP[semi]}{octave}"


def cluster_digits(digits: List[Tuple[int, int, int, int, float]], tol_factor: float = 0.6):
    heights = [h for _, _, _, h, _ in digits]
    median_h = np.median(heights) if heights else 0
    row_bins = []
    tol = median_h * tol_factor if median_h > 0 else 0
    for box in digits:
        _, y, _, h, _ = box
        cy = y + h / 2.0
        placed = False
        for b in row_bins:
            if abs(b["cy_mean"] - cy) <= tol:
                b["items"].append(box)
                b["cy_values"].append(cy)
                b["cy_mean"] = np.mean(b["cy_values"])
                placed = True
                break
        if not placed:
            row_bins.append({"items": [box], "cy_values": [cy], "cy_mean": cy})
    row_bins.sort(key=lambda b: len(b["items"]), reverse=True)
    if not row_bins:
        return [], None
    main = row_bins[0]
    return main["items"], main["cy_mean"]


def point_in_box(px, py, box):
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def load_matchinfo(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"matchinfo 文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def run_notation_detect(matchinfo_path: str, key: str, params: DetectParams = None):
    """
    使用 matchinfo JSON 进行乐谱解析。
    返回 (notes_json_path, saved_flag)；若未按回车保存则返回 (None, False)。
    """
    key_norm = normalize_key(key)
    if params is None:
        params = DetectParams(
            acc_box_ratio=0.6,        # 升降号检测框宽高系数（相对数字高度，右下角对齐数字中心）
            dot_box_ratio_x=0.6,      # 上下点检测框宽系数（相对数字高度，贴靠数字上下边沿）
            dot_box_ratio_y=1.2,      # 上下点检测框高系数（相对数字高度，贴靠数字上下边沿）
            slur_w_ratio=0.7,         # 延音线检测框宽度系数（相对数字高度，位于数字上方）
            slur_h_ratio=2,           # 延音线检测框高度系数（相对数字高度，位于数字上方）
        )

    try:
        info = load_matchinfo(matchinfo_path)
    except FileNotFoundError as e:
        print(str(e))
        return None, False
    except json.JSONDecodeError as e:
        print(f"错误: matchinfo 文件无法解析为 JSON: {e}")
        return None, False
    matches = info.get("matches", {})
    replaced_img_path = info.get("replaced_img_path", "")
    base_img = cv2.imread(replaced_img_path, cv2.IMREAD_COLOR)
    if base_img is None:
        print(f"错误: 无法读取替换后图片 {replaced_img_path}")
        return None, False

    # 收集数字/符号
    digit_boxes = []
    sharp_boxes = [(b["x"], b["y"], b["w"], b["h"], b.get("score", 0)) for b in matches.get("sharp", [])]
    flat_boxes = [(b["x"], b["y"], b["w"], b["h"], b.get("score", 0)) for b in matches.get("flat", [])]
    dot_boxes = matches.get("dot", [])
    dots_all = [(b["x"], b["y"], b["w"], b["h"], b.get("score", 0)) for b in dot_boxes]
    slur_left_boxes = []
    slur_right_boxes = []
    for name, boxes in matches.items():
        if name.isdigit():
            deg = int(name)
            if 0 <= deg <= 9:
                for b in boxes:
                    digit_boxes.append((deg, b["x"], b["y"], b["w"], b["h"], b.get("score", 0)))
        if name.startswith("slur_left"):
            slur_left_boxes.extend([(b["x"], b["y"], b["w"], b["h"], b.get("score", 0)) for b in boxes])
        if name.startswith("slur_right"):
            slur_right_boxes.extend([(b["x"], b["y"], b["w"], b["h"], b.get("score", 0)) for b in boxes])

    if not digit_boxes:
        print("未检测到数字。")
        return

    # 聚类行
    digit_boxes_only = [(x, y, w, h, s) for _, x, y, w, h, s in digit_boxes]
    main_row_boxes, row_center_y = cluster_digits(digit_boxes_only)
    if not main_row_boxes:
        print("未找到主行。")
        return

    # 构造 NoteInfo
    notes: List[NoteInfo] = []
    # 排序
    main_digit_boxes = []
    for deg, x, y, w, h, s in digit_boxes:
        if (x, y, w, h, s) in main_row_boxes:
            main_digit_boxes.append((deg, x, y, w, h, s))
    main_digit_boxes.sort(key=lambda b: b[1])

    for deg, x, y, w, h, s in main_digit_boxes:
        cx = x + w / 2.0
        cy = y + h / 2.0

        # 升降号检测框：右侧中点与数字中心对齐，宽高=acc_box_ratio*h
        acc_w = acc_h = h * params.acc_box_ratio
        acc_box = (int(cx - acc_w), int(cy - acc_h / 2), int(acc_w), int(acc_h))
        accidental = 0
        for bx, by, bw, bh, _ in sharp_boxes:
            if point_in_box(bx + bw, by + bh, acc_box):
                accidental = 1
                break
        if accidental == 0:
            for bx, by, bw, bh, _ in flat_boxes:
                if point_in_box(bx + bw, by + bh, acc_box):
                    accidental = -1
                    break

        # 上下点检测框
        dot_w = h * params.dot_box_ratio_x
        dot_h = h * params.dot_box_ratio_y
        above_box = (int(cx - dot_w / 2), int(y - dot_h), int(dot_w), int(dot_h))
        below_box = (int(cx - dot_w / 2), int(y + h), int(dot_w), int(dot_h))
        dots_above = 0
        dots_below = 0
        dots_hit = []
        for d in dot_boxes:
            dx, dy = d["center"][0], d["center"][1]
            if point_in_box(dx, dy, above_box):
                dots_above += 1
                dots_hit.append((dx, dy))
            elif point_in_box(dx, dy, below_box):
                dots_below += 1
                dots_hit.append((dx, dy))
        octave_offset = max(-2, min(2, dots_above - dots_below))

        # 延音线检测框（上方，宽高比例不同）
        slur_w = h * params.slur_w_ratio
        slur_h = h * params.slur_h_ratio
        slur_box = (int(cx - slur_w / 2), int(y - slur_h), int(slur_w), int(slur_h))

        # 休止符（0）不计算音高/升降号/八度点
        is_rest = deg == 0
        midi_val = None
        if not is_rest:
            deg_key = deg
            acc_key = "#" if accidental == 1 else ("b" if accidental == -1 else "")
            interval_key = f"{deg_key}{acc_key}"
            if interval_key not in INTERVALS:
                continue
            base_midi = KEY_BASE_MIDI.get(key_norm, 60)
            midi_val = base_midi + INTERVALS[interval_key] + octave_offset * 12

        notes.append(
            NoteInfo(
                degree=deg,
                accidental=None if is_rest else accidental,
                octave_offset=None if is_rest else octave_offset,
                pitch=midi_val,
                articulation=True,  # 先占位，稍后计算
                is_rest=is_rest,
                bbox=(x, y, w, h),
                center=(cx, cy),
                row_center_y=row_center_y,
                dot_boxes=[above_box, below_box],
                slur_boxes=[slur_box],
                acc_box=acc_box,
                dots_hit=dots_hit,
            )
        )

    # articulation 判定（利用 slur 检测状态）
    for i, n in enumerate(notes):
        n.slur_start = False
        n.slur_end = False
        if n.is_rest:
            continue
        # 使用已有检测结果判断左/右
        slur_box = n.slur_boxes[0]
        for bx, by, bw, bh, _ in slur_left_boxes:
            if point_in_box(bx, by + bh, slur_box):
                n.slur_start = True
                break
        for bx, by, bw, bh, _ in slur_right_boxes:
            if point_in_box(bx + bw, by + bh, slur_box):
                n.slur_end = True
                break

    for i, n in enumerate(notes):
        if i == 0:
            n.articulation = True
            continue
        prev = notes[i - 1]
        if n.is_rest:
            n.articulation = True
        else:
            n.articulation = not (n.pitch == prev.pitch and prev.slur_start and n.slur_end)

    # 可视化
    colors = [(0, 0, 255), (0, 165, 255), (255, 0, 0), (0, 128, 128), (128, 0, 128)]
    # 基本信息输出
    print(f"Row center y: {row_center_y}")
    print(f"Detected {len(notes)} notes:")
    for i, n in enumerate(notes, 1):
        if n.is_rest:
            print(
                f"{i:2d}. degree={n.degree} (rest), bbox=({n.bbox[0]},{n.bbox[1]},{n.bbox[2]},{n.bbox[3]}), "
                f"center=({n.center[0]:.1f},{n.center[1]:.1f})"
            )
        else:
            print(
                f"{i:2d}. degree={n.degree}, accidental={n.accidental:+d}, octave_offset={n.octave_offset:+d}, "
                f"pitch_midi={n.pitch}, pitch_name={midi_to_name(n.pitch)}, articulation={'head' if n.articulation else 'tie'}, "
                f"bbox=({n.bbox[0]},{n.bbox[1]},{n.bbox[2]},{n.bbox[3]}), center=({n.center[0]:.1f},{n.center[1]:.1f})"
            )
    mode = 1  # 0:原图, 1:数字+行线, 2:上下点框, 3:升降号框, 4:延音线框
    saved = False
    out_json = None
    while True:
        vis = base_img.copy()
        if mode == 0:
            show = vis
        else:
            if mode == 1:
                cv2.line(vis, (0, int(row_center_y)), (vis.shape[1], int(row_center_y)), (255, 0, 255), 1)
            # 全局渲染：按模式绘制 JSON 中的所有对象
            if mode == 2:
                for bx, by, bw, bh, _ in dots_all:
                    cv2.rectangle(vis, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 200, 200), 1)
            elif mode == 3:
                for bx, by, bw, bh, _ in sharp_boxes:
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 140, 200), 1)
                for bx, by, bw, bh, _ in flat_boxes:
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (200, 0, 0), 1)
            elif mode == 4:
                for bx, by, bw, bh, _ in slur_left_boxes:
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 160, 0), 1)
                for bx, by, bw, bh, _ in slur_right_boxes:
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (160, 0, 160), 1)
            for idx, n in enumerate(notes):
                color = colors[idx % len(colors)]
                x, y, w, h = n.bbox
                # 共用右侧 label：休止符显示 rest，其余显示科学音名
                pitch_text = "rest" if n.is_rest else midi_to_name(n.pitch)
                cv2.putText(vis, pitch_text, (x + w + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                if mode == 1:
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                elif mode == 2:
                    for bx, by, bw, bh in n.dot_boxes:
                        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 1)
                    for dx, dy in n.dots_hit:
                        cv2.circle(vis, (int(dx), int(dy)), 3, color, -1)
                    if not n.is_rest:
                        cv2.putText(vis, f"{n.octave_offset:+d}", (x + w + 2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                elif mode == 3:
                    ax, ay, aw, ah = n.acc_box
                    cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), color, 1)
                    if not n.is_rest:
                        cv2.putText(vis, f"{n.accidental:+d}", (x + w + 2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                elif mode == 4:
                    for bx, by, bw, bh in n.slur_boxes:
                        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 1)
                    if not n.is_rest:
                        slur_state = "none"
                        if n.slur_start and n.slur_end:
                            slur_state = "both"
                        elif n.slur_start:
                            slur_state = "left"
                        elif n.slur_end:
                            slur_state = "right"
                        cv2.putText(vis, slur_state, (x + w + 2, y + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            show = vis
        cv2.imshow("notation_detect", show)
        key_code = cv2.waitKey(0) & 0xFF
        if key_code == 32:
            mode = (mode + 1) % 5
            continue
        if key_code in (13, 10):
            base_dir = os.path.dirname(os.path.abspath(replaced_img_path))
            src_path = info.get("src_img_path", "")
            stem = os.path.splitext(os.path.basename(src_path if src_path else replaced_img_path))[0]
            out_json = os.path.join(base_dir, f"{stem}_{key_norm}_notes.json")
            data = {
                "replaced_img_path": os.path.abspath(replaced_img_path),
                "key": key_norm,
                "src_img_path": os.path.abspath(src_path) if src_path else os.path.abspath(replaced_img_path),
                "notes": [
                    {
                        "degree": n.degree,
                        "articulation": n.articulation,
                        "is_rest": n.is_rest,
                        "center": n.center,
                        "w": n.bbox[2],
                        "h": n.bbox[3],
                        "row_center_y": n.row_center_y,
                    }
                    for n in notes
                ],
            }
            for obj, n in zip(data["notes"], notes):
                if not n.is_rest:
                    obj["accidental"] = n.accidental
                    obj["octave_offset"] = n.octave_offset
                    obj["midi"] = n.pitch
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved notes info to {out_json}")
            saved = True
        break
    cv2.destroyAllWindows()
    return out_json, saved


def main():
    if len(sys.argv) < 3:
        print("用法: python notation_detect.py <matchinfo.json> <key>")
        return
    matchinfo_path = sys.argv[1]
    key = sys.argv[2]
    _, _ = run_notation_detect(matchinfo_path, key)


if __name__ == "__main__":
    main()
