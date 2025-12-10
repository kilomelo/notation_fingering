import cv2
import json
import os
import re
import argparse
from typing import Dict, List, Tuple
import numpy as np

# MIDI 计算用：C4 = 60
NOTE_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


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


def load_notes_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fingering_index(fingering_dir: str) -> Dict[int, str]:
    """
    扫描指法目录，将文件名中的音高部分（下划线前）映射到 MIDI。文件名示例：G4_0_off.png
    若有同一 MIDI 多个文件，保留首次扫描到的。
    """
    idx: Dict[int, str] = {}
    for name in os.listdir(fingering_dir):
        if not name.lower().endswith(".png"):
            continue
        pitch_part = name.split("_")[0]
        try:
            midi = pitch_to_midi(pitch_part)
        except ValueError:
            continue
        idx.setdefault(midi, os.path.join(fingering_dir, name))
    return idx


def export_notation(
    target_resolution: Tuple[int, int],
    source_offset: Tuple[int, int],
    notes_json_path: str,
    bg_path: str = None,
    fingering_img_path: str = None,
    fingering_img_offset: Tuple[int, int] = (0, 0),
    fingering_scale: float = 1.0,
) -> None:
    """
    将乐谱（已替换后的图片）渲染到指定画布，并可叠加指法图。按空格切换模式，回车保存。

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
    notes_data = info.get("notes", [])
    if not replaced_img_path or not notes_data:
        raise ValueError("notes_json 缺少 replaced_img_path 或 notes 数据")
    # 目标文件名取自源图文件名
    src_dir = os.path.dirname(replaced_img_path)
    src_stem = os.path.splitext(os.path.basename(replaced_img_path))[0]
    target_path = os.path.join(src_dir, f"{src_stem}_fingering.png")

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

    # 指法图索引
    fing_index = {}
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

    # 渲染指法图（仅音头）
    if fing_index:
        for n in notes_data:
            if not n.get("articulation", True):
                continue
            pitch = n.get("pitch", "")
            try:
                midi = pitch_to_midi(pitch)
            except ValueError:
                print(f"Warning: invalid pitch '{pitch}' for fingering lookup")
                continue
            fing_path = fing_index.get(midi)
            if fing_path is None:
                print(f"Warning: fingering image not found for pitch {pitch}")
                continue
            img = cv2.imread(fing_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: cannot read fingering image: {fing_path}")
                continue
            if img.shape[2] == 3:
                alpha_f = np.ones(img.shape[:2], dtype=np.uint8) * 255
                img = np.dstack([img, alpha_f])
            if fingering_scale != 1.0:
                new_w = max(1, int(round(img.shape[1] * fingering_scale)))
                new_h = max(1, int(round(img.shape[0] * fingering_scale)))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            fg_h, fg_w = img.shape[:2]
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

            fg_crop = img[fg_y1:fg_y2, fg_x1:fg_x2, :].astype(np.float32) / 255.0
            dst_roi = canvas[clip_y1:clip_y2, clip_x1:clip_x2, :]
            blended = blend_alpha(dst_roi, fg_crop)
            canvas[clip_y1:clip_y2, clip_x1:clip_x2, :] = blended

    canvas_uint8 = (canvas * 255.0).clip(0, 255).astype(np.uint8)

    while True:
        cv2.imshow("export", canvas_uint8)
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):
            cv2.imwrite(target_path, canvas_uint8)
            print(f"Saved exported image to: {os.path.abspath(target_path)}")
        break
    cv2.destroyAllWindows()


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
