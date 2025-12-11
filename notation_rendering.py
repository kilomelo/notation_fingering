import cv2
import json
import os
import re
import argparse
from typing import Dict, List, Tuple
import numpy as np

"""
将 notation_detect 输出的 *_notes.json 与替换后的乐谱图片进行合成，可叠加指法图（支持多变体点击切换）。
预览窗口按回车保存最终结果；输出命名为 {源图文件名}_fingering.png。
"""

# MIDI 计算用：C4 = 60
NOTE_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SEMI_TO_NOTE_SHARP = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}


def pitch_to_midi(pitch: str) -> int:
    """将如 C#4、Bb3 的音高转换为 MIDI 编号（C4 = 60）。不合法时抛出 ValueError。"""
    m = re.match(r"^([A-Ga-g])([#b]?)(\d+)$", pitch.strip())
    if not m:
        raise ValueError(f"Invalid pitch: {pitch}")
    note = m.group(1).upper()
    acc = m.group(2)
    octave = int(m.group(3))
    semi = NOTE_SEMI[note]
    if acc == "#":
        semi += 1
    elif acc == "b":
        semi -= 1
    # wrap
    while semi < 0:
        semi += 12
        octave -= 1
    while semi >= 12:
        semi -= 12
        octave += 1
    return (octave + 1) * 12 + semi


def midi_to_name(midi: int) -> str:
    """返回带升号的科学音名（用 #）。"""
    octave = (midi // 12) - 1
    semi = midi % 12
    return f"{SEMI_TO_NOTE_SHARP[semi]}{octave}"


def load_notes_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fingering_index(fingering_dir: str) -> Dict[int, List[Tuple[int, str]]]:
    """
    扫描指法目录，按 MIDI 归档所有变体。
    文件名规则：{pitch}_{variant_id}_off.png 例如 G4_0_off.png、G4_1_off.png。
    返回 midi -> [(variant_id, path)]（按 variant_id 升序）。
    """
    idx: Dict[int, List[Tuple[int, str]]] = {}
    pat = re.compile(r"^([A-Ga-g][#b]?\d+)_([0-9]+)_off\.png$")
    for name in os.listdir(fingering_dir):
        if not name.lower().endswith(".png"):
            continue
        m = pat.match(name)
        if not m:
            continue
        pitch_part = m.group(1)
        try:
            midi = pitch_to_midi(pitch_part)
        except ValueError:
            continue
        variant_id = int(m.group(2))
        idx.setdefault(midi, []).append((variant_id, os.path.join(fingering_dir, name)))
    for midi in idx:
        idx[midi].sort(key=lambda t: t[0])
    return idx


def export_notation(
    target_resolution: Tuple[int, int],
    source_offset: Tuple[int, int],
    notes_json_path: str,
    bg_path: str = None,
    fingering_img_path: str = None,
    fingering_img_offset: Tuple[int, int] = (0, 0),
    fingering_scale: float = 1.0,
) -> bool:
    """
    将乐谱（已替换后的图片）渲染到指定画布，并可叠加指法图。
    支持在预览窗口点击某个指法图区域循环切换变体（文件名中的 variant_id）。
    回车保存当前选择下的画面，返回是否已保存。

    参数：
        target_resolution: (width, height) 输出图片尺寸。
        source_offset: (x_offset, y_offset) 源图左上角在输出中的偏移。
        notes_json_path: notation_detect 输出的 *_notes.json 路径。
        bg_path: 背景图片路径；为空则使用不透明白底。
        fingering_img_path: 指法图目录，为空则不渲染指法图。
        fingering_img_offset: (x_offset, y_offset)，指法图顶边中点相对音符中心/行的偏移。
        fingering_scale: 指法图缩放系数，默认 1.0。

    输出文件：
        自动保存到与 replaced_img_path 同目录的 {src_img_name}_fingering.png
    """
    width, height = target_resolution
    x_off, y_off = source_offset

    info = load_notes_json(notes_json_path)
    replaced_img_path = info.get("replaced_img_path")
    src_img_path = info.get("src_img_path", replaced_img_path)
    notes_data = info.get("notes", [])
    if not replaced_img_path or not notes_data:
        raise ValueError("notes_json 缺少 replaced_img_path 或 notes 数据")
    # 目标文件名取自原图文件名，并包含调号；保存到与原图同名的子目录
    src_dir = os.path.dirname(src_img_path)
    stem_dir = os.path.splitext(os.path.basename(src_img_path))[0]
    save_dir = os.path.join(src_dir, stem_dir)
    os.makedirs(save_dir, exist_ok=True)
    src_stem = os.path.splitext(os.path.basename(src_img_path))[0]
    key = info.get("key", "")
    suffix = f"_{key}" if key else ""
    target_path = os.path.join(save_dir, f"{src_stem}{suffix}_fingering.png")

    base = cv2.imread(replaced_img_path, cv2.IMREAD_UNCHANGED)
    if base is None:
        raise FileNotFoundError(f"Cannot read image: {replaced_img_path}")

    # 背景
    if bg_path:
        bg = cv2.imread(bg_path, cv2.IMREAD_UNCHANGED)
        if bg is None:
            raise FileNotFoundError(f"Cannot read background image: {bg_path}")
        if bg.shape[2] < 3:
            raise ValueError("Background image must have at least 3 channels.")
        bg_bgr = bg[..., :3]
        bg_h, bg_w = bg_bgr.shape[:2]
        rep_y = (height + bg_h - 1) // bg_h
        rep_x = (width + bg_w - 1) // bg_w
        tiled = np.tile(bg_bgr, (rep_y, rep_x, 1))
        bg_canvas = tiled[:height, :width, :]
        canvas_rgb = bg_canvas.astype(np.float32) / 255.0
        canvas_a = np.ones((height, width, 1), dtype=np.float32)
    else:
        canvas_rgb = np.ones((height, width, 3), dtype=np.float32)
        canvas_a = np.ones((height, width, 1), dtype=np.float32)
    canvas = np.concatenate([canvas_rgb, canvas_a], axis=2)

    # 源图（已替换后）
    if base.shape[2] == 3:
        bgr = base
        alpha = np.full(base.shape[:2], 255, dtype=np.uint8)
    else:
        bgr = base[..., :3]
        alpha = base[..., 3]
    base_bgra = np.dstack([bgr, alpha])

    src_h, src_w = base_bgra.shape[:2]
    x1 = max(0, x_off)
    y1 = max(0, y_off)
    x2 = min(width, x_off + src_w)
    y2 = min(height, y_off + src_h)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Source image is completely outside the target canvas.")
    src_x1 = x1 - x_off
    src_y1 = y1 - y_off
    src_crop = base_bgra[src_y1:src_y1 + (y2 - y1), src_x1:src_x1 + (x2 - x1), :]

    dst_roi = canvas[y1:y2, x1:x2, :]
    src_roi = src_crop.astype(np.float32) / 255.0
    dst_rgb, dst_a = dst_roi[..., :3], dst_roi[..., 3:4]
    src_rgb, src_a = src_roi[..., :3], src_roi[..., 3:4]
    out_a = 1.0 - (1.0 - dst_a) * (1.0 - src_a)
    out_rgb = dst_rgb * src_rgb
    out = np.concatenate([out_rgb, out_a], axis=2)
    canvas[y1:y2, x1:x2, :] = out

    # 指法图索引（midi -> [(variant_id, path)]）
    fing_index: Dict[int, List[Tuple[int, str]]] = {}
    if fingering_img_path:
        fing_index = build_fingering_index(fingering_img_path)
    fx_off, fy_off = fingering_img_offset

    def blend_alpha(dst: np.ndarray, fg: np.ndarray):
        dst_rgb, dst_a = dst[..., :3], dst[..., 3:4]
        fg_rgb, fg_a = fg[..., :3], fg[..., 3:4]
        out_a = fg_a + dst_a * (1.0 - fg_a)
        out_rgb = fg_rgb * fg_a + dst_rgb * dst_a * (1.0 - fg_a)
        out_rgb = np.where(out_a > 1e-6, out_rgb / out_a, 0)
        return np.concatenate([out_rgb, out_a], axis=2)

    # 记录可点击的指法信息：每个音的可用变体及当前选择
    selections = []
    img_cache: Dict[str, np.ndarray] = {}
    if fing_index:
        for idx, n in enumerate(notes_data):
            if not n.get("articulation", True):
                continue
            if n.get("is_rest"):
                continue
            midi = n.get("midi")
            if midi is None:
                pitch = n.get("pitch", "")
                try:
                    midi = pitch_to_midi(pitch)
                except ValueError:
                    print(f"Warning: invalid pitch '{pitch}' for fingering lookup")
                    continue
            pitch_name = midi_to_name(midi)
            variants = fing_index.get(midi, [])
            if not variants:
                print(f"Warning: fingering image not found for pitch {pitch_name}")
                continue
            # 默认选用 variant_id 为 0；若不存在则取最小 variant_id
            sel_idx = 0
            for j, (vid, _) in enumerate(variants):
                if vid == 0:
                    sel_idx = j
                    break
            selections.append({"note_idx": idx, "pitch_name": pitch_name, "midi": midi, "variants": variants, "sel_idx": sel_idx})

    def load_fing(img_path: str) -> np.ndarray:
        if img_path in img_cache:
            return img_cache[img_path]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape[2] == 3:
            alpha_f = np.ones(img.shape[:2], dtype=np.uint8) * 255
            img = np.dstack([img, alpha_f])
        img_cache[img_path] = img
        return img

    def render_with_selection():
        canvas_cur = canvas.copy()
        click_boxes = []  # 记录可点击区域及所对应的 selection
        for sel in selections:
            n = notes_data[sel["note_idx"]]
            variants = sel["variants"]
            if not variants:
                continue
            variant_id, fing_path = variants[sel["sel_idx"]]
            img = load_fing(fing_path)
            if img is None:
                print(f"Warning: cannot read fingering image: {fing_path}")
                continue
            if fingering_scale != 1.0:
                new_w = max(1, int(round(img.shape[1] * fingering_scale)))
                new_h = max(1, int(round(img.shape[0] * fingering_scale)))
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_scaled = img

            fg_h, fg_w = img_scaled.shape[:2]
            cx_note = x_off + n["center"][0] + fx_off
            top_y = y_off + n["row_center_y"] + fy_off
            tl_x = int(round(cx_note - fg_w / 2))
            tl_y = int(round(top_y))
            br_x = tl_x + fg_w
            br_y = tl_y + fg_h

            clip_x1 = max(0, tl_x)
            clip_y1 = max(0, tl_y)
            clip_x2 = min(width, br_x)
            clip_y2 = min(height, br_y)
            if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                continue

            fg_x1 = clip_x1 - tl_x
            fg_y1 = clip_y1 - tl_y
            fg_x2 = fg_x1 + (clip_x2 - clip_x1)
            fg_y2 = fg_y1 + (clip_y2 - clip_y1)

            fg_crop = img_scaled[fg_y1:fg_y2, fg_x1:fg_x2, :].astype(np.float32) / 255.0
            dst_roi = canvas_cur[clip_y1:clip_y2, clip_x1:clip_x2, :]
            blended = blend_alpha(dst_roi, fg_crop)
            canvas_cur[clip_y1:clip_y2, clip_x1:clip_x2, :] = blended
            click_boxes.append(
                {
                    "bbox": (clip_x1, clip_y1, clip_x2, clip_y2),
                    "selection": sel,
                    "variant_id": variant_id,
                }
            )
        return (canvas_cur * 255.0).clip(0, 255).astype(np.uint8), click_boxes

    rendered_img, click_boxes = render_with_selection()

    # 鼠标点击切换变体
    cv2.namedWindow("export")

    def on_mouse(event, x, y, flags, param):
        nonlocal rendered_img, click_boxes
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for info in click_boxes:
            x1, y1, x2, y2 = info["bbox"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                sel = info["selection"]
                variants = sel["variants"]
                if not variants:
                    return
                if len(variants) == 1:
                    print("该音高只有一种指法，无法切换")
                    return
                sel["sel_idx"] = (sel["sel_idx"] + 1) % len(variants)
                new_vid = variants[sel["sel_idx"]][0]
                print(f"指法切换: 音高 {sel['pitch_name']} -> 变体 {new_vid}")
                rendered_img, click_boxes = render_with_selection()
                cv2.imshow("export", rendered_img)
                break

    cv2.setMouseCallback("export", on_mouse)

    saved = False
    while True:
        cv2.imshow("export", rendered_img)
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.imwrite(target_path, rendered_img)
            print(f"Saved exported image to: {os.path.abspath(target_path)}")
            saved = True
        break
    cv2.destroyAllWindows()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Render fingering overlay using *_notes.json produced by notation_detect."
    )
    parser.add_argument("notes_json_path", help="Path to *_notes.json from notation_detect.")
    parser.add_argument("--width", type=int, help="Output width. If omitted, use source image width.")
    parser.add_argument("--height", type=int, help="Output height. If omitted, use source image height.")
    parser.add_argument("--offset-x", type=int, default=0, help="Source image offset X in output.")
    parser.add_argument("--offset-y", type=int, default=0, help="Source image offset Y in output.")
    parser.add_argument("--bg", dest="bg_path", help="Background image path; default white.")
    parser.add_argument("--fingering-dir", dest="fingering_img_path", help="Directory of fingering images.")
    parser.add_argument("--fingering-offset-x", type=int, default=0, help="Fingering top-center X offset vs note center.")
    parser.add_argument("--fingering-offset-y", type=int, default=0, help="Fingering top-center Y offset vs row center.")
    parser.add_argument("--fingering-scale", type=float, default=1.0, help="Fingering image scale factor.")
    args = parser.parse_args()

    # 如果未指定输出尺寸，则沿用源图尺寸
    info = load_notes_json(args.notes_json_path)
    replaced_img_path = info.get("replaced_img_path")
    if not replaced_img_path:
        raise ValueError("notes_json 缺少 replaced_img_path")
    src_img = cv2.imread(replaced_img_path, cv2.IMREAD_UNCHANGED)
    if src_img is None:
        raise FileNotFoundError(f"Cannot read replaced image: {replaced_img_path}")
    out_w = args.width if args.width else src_img.shape[1]
    out_h = args.height if args.height else src_img.shape[0]

    export_notation(
        target_resolution=(out_w, out_h),
        source_offset=(args.offset_x, args.offset_y),
        notes_json_path=args.notes_json_path,
        bg_path=args.bg_path,
        fingering_img_path=args.fingering_img_path,
        fingering_img_offset=(args.fingering_offset_x, args.fingering_offset_y),
        fingering_scale=args.fingering_scale,
    )


if __name__ == "__main__":
    main()
