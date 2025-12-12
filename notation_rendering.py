import cv2
import json
import os
import re
import argparse
from typing import Dict, List, Tuple
import numpy as np

"""
将 notation_detect 输出的 *_notes.json 与替换后的乐谱图片进行合成，可叠加指法图（支持多变体点击切换）。
预览窗口按回车后，批量输出多张图片：
  - 0 号图：所有音符指法为 off
  - 之后每个音头一张：当前音头为 on，其余 off，文件名递增
输出命名为 {原图名}_{key}_fingering_{id}.png，保存到原图同名子目录。
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


def build_fingering_index(fingering_dir: str, expected_suffix: str | None = None) -> Dict[int, List[Tuple[int, str]]]:
    """
    扫描指法目录，按 MIDI 归档所有变体。
    文件名规则：{pitch}_{variant_id}_{on/off}.png 例如 G4_0_off.png、G4_1_on.png。
    expected_suffix 可指定过滤 on/off。
    返回 midi -> [(variant_id, path)]（按 variant_id 升序）。
    """
    idx: Dict[int, List[Tuple[int, str]]] = {}
    pat = re.compile(r"^([A-Ga-g][#b]?\d+)_([0-9]+)_(on|off)\.png$")
    for name in os.listdir(fingering_dir):
        if not name.lower().endswith(".png"):
            continue
        m = pat.match(name)
        if not m:
            continue
        suffix = m.group(3).lower()
        if expected_suffix and suffix != expected_suffix:
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
    fingering_off_dir: str = None,
    fingering_on_dir: str = None,
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
        fingering_off_dir: 指法图目录（off 状态），为 None 则不渲染指法图。
        fingering_on_dir: 指法图目录（on 状态），为 None 则不渲染 on 状态。
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

    # 指法图索引（off/on 分别对应 midi -> [(variant_id, path)]）
    off_index: Dict[int, List[Tuple[int, str]]] = {}
    on_index: Dict[int, List[Tuple[int, str]]] = {}
    if fingering_off_dir:
        off_index = build_fingering_index(fingering_off_dir, expected_suffix="off")
    if fingering_on_dir:
        on_index = build_fingering_index(fingering_on_dir, expected_suffix="on")
    fx_off, fy_off = fingering_img_offset

    def blend_alpha(dst: np.ndarray, fg: np.ndarray):
        dst_rgb, dst_a = dst[..., :3], dst[..., 3:4]
        fg_rgb, fg_a = fg[..., :3], fg[..., 3:4]
        out_a = fg_a + dst_a * (1.0 - fg_a)
        out_rgb = fg_rgb * fg_a + dst_rgb * dst_a * (1.0 - fg_a)
        out_rgb = np.where(out_a > 1e-6, out_rgb / out_a, 0)
        return np.concatenate([out_rgb, out_a], axis=2)

    def to_bgr(img: np.ndarray) -> np.ndarray:
        """将 BGRA/BGR 转为 BGR 便于绘制进度条。"""
        if img.shape[2] == 4:
            return img[..., :3].copy()
        return img.copy()

    # 记录可点击的指法信息：每个音的可用变体及当前选择
    selections = []
    img_cache: Dict[str, np.ndarray] = {}
    # 变体 ID 取 off/on 的并集；默认选 0，否则最小值
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
        variants_set = set()
        for lst in (off_index.get(midi, []), on_index.get(midi, [])):
            variants_set.update([vid for vid, _ in lst])
        variants_ids = sorted(list(variants_set))
        if not variants_ids:
            print(f"Warning: fingering image not found for pitch {pitch_name}")
            continue
        sel_variant_id = 0 if 0 in variants_ids else variants_ids[0]
        selections.append(
            {
                "note_idx": idx,
                "pitch_name": pitch_name,
                "midi": midi,
                "variants": variants_ids,
                "sel_variant_id": sel_variant_id,
            }
        )

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

    def find_variant(idx: Dict[int, List[Tuple[int, str]]], midi: int, vid: int):
        for v_id, path in idx.get(midi, []):
            if v_id == vid:
                return path
        return None

    def resolve_path(midi: int, vid: int, use_on: bool):
        """
        返回 (path, used_on_flag)。优先按 use_on 查找 on/off，对应缺失时按规则 fallback。
        规则：
          - off 缺失而 on 存在：用 on，输出错误信息。
          - on 缺失：若 off 存在则用 off 并 warning；都缺失则返回 None。
        """
        if use_on:
            path_on = find_variant(on_index, midi, vid)
            if path_on:
                return path_on, True
            path_off = find_variant(off_index, midi, vid)
            if path_off:
                expected = os.path.join(os.path.abspath(fingering_on_dir or ""), f"{midi_to_name(midi)}_{vid}_on.png")
                print(f"Warning: on 指法缺失，使用 off 代替 (midi={midi}, variant={vid}), 缺失的图片: {expected}")
                return path_off, False
            return None, False
        else:
            path_off = find_variant(off_index, midi, vid)
            if path_off:
                return path_off, False
            path_on = find_variant(on_index, midi, vid)
            if path_on:
                expected = os.path.join(os.path.abspath(fingering_off_dir or ""), f"{midi_to_name(midi)}_{vid}_off.png")
                print(f"错误: off 指法缺失，使用 on 代替 (midi={midi}, variant={vid}), 缺失的图片: {expected}")
                return path_on, True
            return None, False

    def render_with_selection(on_note_idx: int | None):
        """
        渲染一张图：
          - on_note_idx 指定的音（note_idx）使用 on 图片，其余使用 off。
          - 只有存在指法图的音符才会叠加。
        返回 (uint8 BGR 或 BGRA, click_boxes)
        """
        canvas_cur = canvas.copy()
        click_boxes = []  # 记录可点击区域及所对应的 selection
        for sel in selections:
            n = notes_data[sel["note_idx"]]
            use_on = on_note_idx is not None and sel["note_idx"] == on_note_idx
            vid = sel["sel_variant_id"]
            path, used_on = resolve_path(sel["midi"], vid, use_on)
            if not path:
                continue
            img = load_fing(path)
            if img is None:
                print(f"Warning: cannot read fingering image: {path}")
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
                    "variant_id": vid,
                    "used_on": used_on,
                }
            )
        return (canvas_cur * 255.0).clip(0, 255).astype(np.uint8), click_boxes

    # 预览顺序：0 号 + 按音头顺序的 on 版（懒渲染）
    head_indices = [
        (n["center"][0], idx)
        for idx, n in enumerate(notes_data)
        if n.get("articulation", True) and not n.get("is_rest")
    ]
    head_indices.sort(key=lambda t: t[0])
    preview_order = [(0, None)] + [(i + 1, note_idx) for i, (_, note_idx) in enumerate(head_indices)]
    rendered_cache: Dict[int, Tuple[np.ndarray, List[Dict]]] = {}

    def get_preview(idx: int) -> Tuple[np.ndarray, List[Dict]]:
        if idx in rendered_cache:
            return rendered_cache[idx]
        _, note_idx = preview_order[idx]
        img, boxes = render_with_selection(on_note_idx=note_idx)
        rendered_cache[idx] = (img, boxes)
        return img, boxes

    current_idx = 0
    rendered_img, click_boxes = get_preview(current_idx)

    # 鼠标点击切换变体（更新选择并清空缓存以重新渲染）
    cv2.namedWindow("export")

    def on_mouse(event, x, y, flags, param):
        nonlocal rendered_img, click_boxes, rendered_cache, current_idx
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
                cur_idx = variants.index(sel["sel_variant_id"])
                next_id = variants[(cur_idx + 1) % len(variants)]
                sel["sel_variant_id"] = next_id
                rendered_cache.clear()  # 变体变更，清空缓存以重算
                rendered_img, click_boxes = get_preview(current_idx)
                cv2.imshow("export", rendered_img)
                print(f"指法切换: 音高 {sel['pitch_name']} -> 变体 {next_id}")
                break

    cv2.setMouseCallback("export", on_mouse)

    saved = False
    while True:
        cv2.imshow("export", rendered_img)
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # space: 轮播预览
            current_idx = (current_idx + 1) % len(preview_order)
            rendered_img, click_boxes = get_preview(current_idx)
            cv2.imshow("export", rendered_img)
            continue
        if key in (13, 10):
            # 始终输出整套图片
            img0, _ = get_preview(0)
            path0 = target_path.replace("_fingering.png", "_fingering_0.png")
            cv2.imwrite(path0, img0)
            total = len(preview_order)
            print(f"开始保存指法序列，共 {total} 张 ...")
            print(f"[1/{total}] 保存 {os.path.abspath(path0)}")
            for idx_save, (out_id, _) in enumerate(preview_order[1:], start=2):
                img, _ = get_preview(out_id)
                out_path = target_path.replace("_fingering.png", f"_fingering_{out_id}.png")
                cv2.imwrite(out_path, img)
                # 进度条绘制
                progress = idx_save / total
                bar_img = to_bgr(img)
                bar_w = int(bar_img.shape[1] * 0.5)
                bar_h = 20
                x0 = (bar_img.shape[1] - bar_w) // 2
                y0 = bar_img.shape[0] - bar_h - 10
                cv2.rectangle(bar_img, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), 1)
                cv2.rectangle(bar_img, (x0 + 1, y0 + 1), (x0 + 1 + int((bar_w - 2) * progress), y0 + bar_h - 2), (0, 200, 0), -1)
                cv2.putText(bar_img, f"{int(progress * 100)}%", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
                cv2.imshow("export", bar_img)
                cv2.waitKey(1)
                print(f"[{idx_save}/{total}] 保存 {os.path.abspath(out_path)}")
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
    parser.add_argument("--fingering-off-dir", dest="fingering_off_dir", required=True, help="Directory of fingering off-state images.")
    parser.add_argument("--fingering-on-dir", dest="fingering_on_dir", required=True, help="Directory of fingering on-state images.")
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
        fingering_off_dir=args.fingering_off_dir,
        fingering_on_dir=args.fingering_on_dir,
        fingering_img_offset=(args.fingering_offset_x, args.fingering_offset_y),
        fingering_scale=args.fingering_scale,
    )


if __name__ == "__main__":
    main()
