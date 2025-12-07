import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from PIL import Image
import pytesseract

# 调试信息输出等级（越大越多），默认 1；打印时用 log(level, msg)
INFO_LEVEL = 2


def log(level: int, msg: str) -> None:
    if INFO_LEVEL >= level:
        print(msg)


# 简单的大小调调号集合（主音 + 升降号），用于合法性校验与音高拼写
KEY_SCALES = {
    "C":  ["C", "D", "E", "F", "G", "A", "B"],
    "G":  ["G", "A", "B", "C", "D", "E", "F#"],
    "D":  ["D", "E", "F#", "G", "A", "B", "C#"],
    "A":  ["A", "B", "C#", "D", "E", "F#", "G#"],
    "E":  ["E", "F#", "G#", "A", "B", "C#", "D#"],
    "B":  ["B", "C#", "D#", "E", "F#", "G#", "A#"],
    "F#": ["F#", "G#", "A#", "B", "C#", "D#", "E#"],
    "C#": ["C#", "D#", "E#", "F#", "G#", "A#", "B#"],
    "F":  ["F", "G", "A", "Bb", "C", "D", "E"],
    "Bb": ["Bb", "C", "D", "Eb", "F", "G", "A"],
    "Eb": ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
    "Ab": ["Ab", "Bb", "C", "Db", "Eb", "F", "G"],
    "Db": ["Db", "Eb", "F", "Gb", "Ab", "Bb", "C"],
    "Gb": ["Gb", "Ab", "Bb", "Cb", "Db", "Eb", "F"],
    "Cb": ["Cb", "Db", "Eb", "Fb", "Gb", "Ab", "Bb"],
}


def normalize_key(key: str) -> str:
    """校验并规范调号，不合法则返回 'C' 并输出 warning。"""
    if not isinstance(key, str):
        log(0, "Warning: key must be string, fallback to 'C'")
        return "C"
    k = key.strip()
    if len(k) == 0 or len(k) > 2:
        log(0, f"Warning: invalid key '{key}', fallback to 'C'")
        return "C"
    if len(k) == 1:
        if k in KEY_SCALES:
            return k
        log(0, f"Warning: invalid key '{key}', fallback to 'C'")
        return "C"
    # len == 2
    base, accidental = k[0], k[1]
    if base not in "CDEFGAB" or accidental not in "#b":
        log(0, f"Warning: invalid key '{key}', fallback to 'C'")
        return "C"
    if k in KEY_SCALES:
        return k
    log(0, f"Warning: unsupported key '{key}', fallback to 'C'")
    return "C"


def degree_to_pitch(key: str, degree: int, octave_offset: int) -> str:
    """按调号和八度偏移计算音高字符串（基准八度 3）。"""
    if degree <= 0:
        return "Rest"
    key_norm = normalize_key(key)
    scale = KEY_SCALES[key_norm]
    idx = (degree - 1) % 7
    note_name = scale[idx]
    base_octave = 3  # 1度所在的基准八度
    octave = base_octave + octave_offset
    return f"{note_name}{octave}"

@dataclass
class DetectParams:
    """集中配置识别阈值/系数，方便统一调整。"""
    upscale: float = 1.0                  # OCR 之前的整体缩放（目前未额外缩放，保留以便调节）
    area_min_factor: float = 0.2          # 连通域面积下限（相对中位数）
    area_max_factor: float = 4.5          # 连通域面积上限（相对中位数）
    width_height_ratio_max: float = 1.0   # 候选过宽剔除阈值（w > h * ratio）
    height_ratio_max: float = 2.0         # 候选过高剔除阈值（h > median_h_all * ratio）
    row_tolerance_factor: float = 0.6     # 行聚类 y 容差系数（乘以数字中位高）
    pad_ratio: float = 0.12               # OCR 裁剪时的边缘 padding（相对 max(w,h)）
    ocr_scale: float = 3.0                # 单字符 OCR 的放大倍数
    ocr_dilate_kernel: int = 2            # fallback 膨胀核尺寸（正方形核边长）
    search_width_factor: float = 0.6      # 点搜索区宽度系数（相对数字高）
    margin_y_factor: float = 0.2          # 点搜索区上下边距（相对数字高）
    search_height_factor: float = 0.9     # 点搜索区高度（相对数字高）
    max_dot_size_factor: float = 1 / 3    # 点尺寸上限（相对数字高）
    cx_tol_factor: float = 0.25           # 点中心与数字中心的水平容差（相对数字高）
    slur_ratio: float = 2.0               # 判定为连音线的宽高比阈值（w >= h * slur_ratio）

# 如果 Tesseract 不在 PATH，需要手动指定安装路径，例如：
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class Note:
    """表示一个简谱音符（只考虑音级 + 上下点）"""
    degree: int          # 0-7
    octave_offset: int   # 上下点数量之差：上点为正，下点为负
    x: int               # 数字框左上角 x
    y: int               # 数字框左上角 y
    w: int               # 数字框宽
    h: int               # 数字框高
    dots: List[tuple] = field(default_factory=list)  # 记录检测到的点的中心坐标列表
    search_regions: List[tuple] = field(default_factory=list)  # 点搜索区域 (x1,y1,x2,y2)
    pitch: str = ""      # 计算得到的音高字符串（如 C4、F#4 等）
    slur_start_ids: List[int] = field(default_factory=list)  # 以该音为起点的连音线 id
    slur_end_ids: List[int] = field(default_factory=list)    # 以该音为终点的连音线 id
    articulation: bool = True  # True 表示音头，False 表示延音

    def center(self):
        return (self.x + self.w // 2, self.y + self.h // 2)


def extract_notes(image_path: str, params: Optional[DetectParams] = None, key: str = "C"):
    """
    从单行简谱图片中提取音符列表。

    参数：
        image_path: 输入图片路径。
        params: DetectParams，可选，集中配置各类阈值；不传则用默认。
        key: 调号字符串，形如 "C", "G", "Bb", "F#", 默认 "C"。首字符必须是 A-G 之一，可选第二字符为 # 或 b。

    返回：
        notes: List[Note]，按 x 从左到右排序，包含 pitch、八度偏移、点等信息。
        row_center_y: 主行中心 y，用于调试行绘制。
        raw_boxes: 原始连通域列表 (id, x, y, w, h)，用于调试。
    """
    if params is None:
        params = DetectParams()
    key_norm = normalize_key(key)

    # 1. 读图
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    # 2. 为了后面找“点”，先做一个简单的二值化（黑字白底 -> 反转成白字黑底）
    #    注：你说是印刷谱，我们直接用 OTSU 阈值就行
    _, binary_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # binary_inv: 黑底白字（包括数字和点）

    # 3. 连通域分割 + 单字符 OCR：不假设行位置，先找到所有小块再逐个识别。
    contours, _ = cv2.findContours(
        binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    raw_boxes = []
    slur_boxes = []
    areas = []
    for idx, cnt in enumerate(contours, 1):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 6:  # 太小的噪声
            continue
        areas.append(area)
        raw_boxes.append((idx, x, y, w, h))  # 保留原始序号
        candidates.append({"id": idx, "x": x, "y": y, "w": w, "h": h})

    if not candidates:
        log(1, "No contours found.")
        return [], None, raw_boxes

    median_area = np.median(areas)
    median_h_all = np.median([c["h"] for c in candidates])
    log(1, f"Median area: {median_area}, median height: {median_h_all}")

    def _to_digit(ch: str, w: int, h: int, median_h: float):
        """把易混字符归一成数字，同时用尺寸过滤掉小节线等竖条。"""
        if ch in "01234567":
            return int(ch)
        if ch in ["I", "l", "|", "i"]:
            # 太细太高的，多半是小节线，丢弃
            if w < 0.25 * h or h > 2.8 * median_h:
                return None
            return 1
        return None

    def ocr_single_char(crop_gray: np.ndarray) -> str:
        """对单个字符区域做 OCR（单字符模式），带兜底增强。"""
        if crop_gray.size == 0:
            return ""

        def _try_ocr(img_gray: np.ndarray, dilate: bool) -> str:
            # resize
            scale = params.ocr_scale
            roi = cv2.resize(
                img_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            _, roi = cv2.threshold(
                roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if dilate:
                k = params.ocr_dilate_kernel
                roi = cv2.dilate(roi, np.ones((k, k), np.uint8), iterations=1)
            roi = cv2.bitwise_not(roi)  # 黑字白底
            txt = pytesseract.image_to_string(
                Image.fromarray(roi),
                config=r'--oem 3 --psm 10 -c tessedit_char_whitelist=01234567Il|'
            )
            return txt.strip()

        # 先常规，再尝试膨胀增强笔画
        txt = _try_ocr(crop_gray, dilate=False)
        if txt:
            return txt
        txt = _try_ocr(crop_gray, dilate=True)
        if txt:
            log(2, "OCR fallback (dilate) succeeded")
        return txt

    # 粗过滤极端尺寸的候选框，同时识别可能的连音线
    size_filtered = []
    for cand in candidates:
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        area = w * h
        ratio = w / h if h else float("inf")
        if ratio >= params.slur_ratio:
            slur_boxes.append({"id": cand["id"], "x": x, "y": y, "w": w, "h": h})
            continue
        if area < median_area * params.area_min_factor:
            log(2, f"Skip cand#{cand['id']} area too small ({area})")
            continue
        if area > median_area * params.area_max_factor:
            log(2, f"Skip cand#{cand['id']} area too large ({area})")
            continue
        if w > h * params.width_height_ratio_max:  # 连音线、括号等长条
            log(2, f"Skip cand#{cand['id']} too wide (w={w}, h={h})")
            continue
        if h > median_h_all * params.height_ratio_max:  # 高竖线（小节线）
            log(2, f"Skip cand#{cand['id']} too tall (h={h})")
            continue
        size_filtered.append(cand)

    if not size_filtered:
        log(0, "No candidates after size filtering.")
        return [], None, raw_boxes

    heights = [c["h"] for c in size_filtered]
    median_h = np.median(heights)

    recognized_boxes = []
    for cand in size_filtered:
        x, y, w, h = cand["x"], cand["y"], cand["w"], cand["h"]
        pad = int(max(1, params.pad_ratio * max(w, h)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        crop = gray[y1:y2, x1:x2]
        text = ocr_single_char(crop)
        if not text:
            log(2, f"Skip cand#{cand['id']} OCR empty")
            continue
        ch = text[0]
        degree = _to_digit(ch, w, h, median_h)
        if degree is None:
            log(2, f"Skip cand#{cand['id']} OCR '{ch}' not digit-like")
            continue
        recognized_boxes.append({"id": cand["id"], "degree": degree, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1})

    if not recognized_boxes:
        log(0, "No digits 0-7 detected by per-char OCR.")
        return [], None, raw_boxes

    # 按 y 聚类：同一行的数字应该在同一 y 段，取数量最多的行
    row_bins = []
    row_tolerance = median_h * params.row_tolerance_factor
    for box in recognized_boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cy = y + h / 2
        placed = False
        for bin in row_bins:
            if abs(bin["cy_mean"] - cy) <= row_tolerance:
                bin["items"].append(box)
                bin["cy_values"].append(cy)
                bin["cy_mean"] = np.mean(bin["cy_values"])
                placed = True
                break
        if not placed:
            row_bins.append({"items": [box], "cy_values": [cy], "cy_mean": cy})

    row_bins.sort(key=lambda b: len(b["items"]), reverse=True)
    main_row = row_bins[0]
    main_row_boxes = main_row["items"]

    if len(main_row_boxes) < len(recognized_boxes) // 2:
        log(0, "Warning: many detected digits are off the main row; results may be noisy.")

    digit_boxes = main_row_boxes
    row_center_y = int(main_row["cy_mean"])

    notes: List[Note] = []

    # 4. 对每个数字，在其上下邻域找“点”（连通域）
    for box in digit_boxes:
        degree = box["degree"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        # 定义搜索区域参数（可根据你实际图调）：
        search_width = max(1, int(round(h * params.search_width_factor)))
        margin_y = int(h * params.margin_y_factor)
        search_height = int(h * params.search_height_factor)
        dots = []
        search_regions = []

        cx_digit = x + w // 2
        x1 = max(0, cx_digit - search_width // 2)
        x2 = min(w_img, cx_digit + (search_width + 1) // 2)

        # 上方搜索区域
        above_y2 = max(0, y - margin_y)
        above_y1 = max(0, above_y2 - search_height)
        above_region = None
        if above_y2 > above_y1:
            search_regions.append((x1, above_y1, x2, above_y2))
            above_region = (x1, above_y1, x2, above_y2)

        # 下方搜索区域
        below_y1 = min(h_img, y + h + margin_y)
        below_y2 = min(h_img, below_y1 + search_height)
        if below_y2 > below_y1:
            search_regions.append((x1, below_y1, x2, below_y2))

        num_dots_above = 0
        num_dots_below = 0

        max_dot_size = h * params.max_dot_size_factor
        cx_tol = h * params.cx_tol_factor

        def box_inside_region(bx, by, bw, bh, rx1, ry1, rx2, ry2) -> bool:
            return bx >= rx1 and by >= ry1 and (bx + bw) <= rx2 and (by + bh) <= ry2
        def point_in_region(px, py, region) -> bool:
            x1r, y1r, x2r, y2r = region
            return x1r <= px <= x2r and y1r <= py <= y2r

        # ---- 搜索上方的点 ----
        if above_y2 > above_y1:
            for rid, bx, by, bw, bh in raw_boxes:
                if bw >= max_dot_size or bh >= max_dot_size:
                    continue
                if not box_inside_region(bx, by, bw, bh, x1, above_y1, x2, above_y2):
                    continue
                cx_box = bx + bw / 2.0
                if abs(cx_box - cx_digit) > cx_tol:
                    continue
                num_dots_above += 1
                dots.append((int(cx_box), int(by + bh / 2.0)))

        # ---- 搜索下方的点 ----
        if below_y2 > below_y1:
            for rid, bx, by, bw, bh in raw_boxes:
                if bw >= max_dot_size or bh >= max_dot_size:
                    continue
                if not box_inside_region(bx, by, bw, bh, x1, below_y1, x2, below_y2):
                    continue
                cx_box = bx + bw / 2.0
                if abs(cx_box - cx_digit) > cx_tol:
                    continue
                num_dots_below += 1
                dots.append((int(cx_box), int(by + bh / 2.0)))

        if num_dots_above > 0 and num_dots_below > 0:
            log(0, f"Warning: both upper and lower dots detected at digit x={x}, y={y}")
            num_dots_above = 0  # 规则：上点数置 0 再计算

        # 为了简单：如果检测到很多点，就裁剪到 [-2, 2] 范围内
        octave_offset = max(-2, min(2, num_dots_above - num_dots_below))
        pitch = degree_to_pitch(key_norm, degree, octave_offset)

        slur_starts = []
        slur_ends = []
        if above_region and slur_boxes:
            for slur in slur_boxes:
                sx, sy, sw, sh = slur["x"], slur["y"], slur["w"], slur["h"]
                left_bottom = (sx, sy + sh)
                right_bottom = (sx + sw, sy + sh)
                if point_in_region(*left_bottom, above_region):
                    slur_starts.append(slur["id"])
                if point_in_region(*right_bottom, above_region):
                    slur_ends.append(slur["id"])

        note = Note(
            degree=degree,
            octave_offset=octave_offset,
            x=x,
            y=y,
            w=w,
            h=h,
            dots=dots,
            search_regions=search_regions,
            pitch=pitch,
            slur_start_ids=slur_starts,
            slur_end_ids=slur_ends,
        )
        notes.append(note)

    # 5. 从左到右排序（单行简谱）
    notes.sort(key=lambda n: n.x)

    # 6. articulation：若上一音 pitch 相同且共享同一条连音线（前音为起点，当前为终点），则当前不是音头
    for i, note in enumerate(notes):
        if i == 0:
            note.articulation = True
            continue
        prev = notes[i - 1]
        shared_slur = set(prev.slur_start_ids) & set(note.slur_end_ids)
        note.articulation = not (note.pitch == prev.pitch and len(shared_slur) > 0)

    return notes, row_center_y, raw_boxes


def draw_notes_on_image(
    image_path: str,
    notes: List[Note],
    row_center_y: int = None,
    raw_boxes: List[tuple] = None,
    window_name: str = "Detected notes",
) -> None:
    """用矩形把检测结果画出来并显示原图。空格循环三个模式：原图、原始调试、最终结果调试。"""
    base_img = cv2.imread(image_path)
    if base_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    colors = [
        (0, 0, 255),     # 红
        (0, 165, 255),   # 橙
        (255, 0, 0),     # 蓝
        (0, 100, 0),     # 深绿
        (255, 0, 255),   # 紫
    ]

    def render(mode: int) -> np.ndarray:
        """
        mode 0: 原图
        mode 1: 原始调试（所有轮廓框）
        mode 2: 最终结果调试（主行、音符框、点）
        """
        img = base_img.copy()
        if mode == 0:
            return img
        if mode == 1:
            if raw_boxes:
                for orig_id, x, y, w, h in raw_boxes:
                    color = colors[(orig_id - 1) % len(colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(
                        img,
                        str(orig_id),
                        (x, max(10, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
            return img
        # mode == 2
        if row_center_y is not None:
            cv2.line(img, (0, row_center_y), (img.shape[1], row_center_y), (255, 0, 255), 1)  # 紫色线，线宽减半
        for idx, note in enumerate(notes, 1):
            color = colors[(idx - 1) % len(colors)]
            for sx1, sy1, sx2, sy2 in note.search_regions:
                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, 1)
            pt1 = (note.x, note.y)
            pt2 = (note.x + note.w, note.y + note.h)
            thickness = 2 if note.articulation else 1
            cv2.rectangle(img, pt1, pt2, color, thickness)
            for dot in note.dots:
                cv2.circle(img, dot, radius=3, color=color, thickness=-1)
        return img

    mode = 2  # 默认显示最终调试
    while True:
        img_to_show = render(mode)
        cv2.imshow(window_name, img_to_show)
        key = cv2.waitKey(0) & 0xFF
        if key == 32:  # 空格键切换模式
            mode = (mode + 1) % 3
            continue
        else:
            break
    cv2.destroyAllWindows()


def main():
    # 示例：你可以改成从命令行参数读取
    image_path = "pu6.png"  # 替换成你的简谱图片路径
    key = "G"                   # 调号，形如 C, G, Bb, F# 等
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
        margin_y_factor=0.01,         # 点搜索区域上下边距（相对数字高度）
        search_height_factor=1.25,   # 点搜索区域高度（相对数字高度）
        max_dot_size_factor=0.35,    # 认定为点的最大宽/高（相对数字高度）
        cx_tol_factor=0.2,           # 点中心与数字中心的水平容差（相对数字高度）
    )
    notes, row_center_y, raw_boxes = extract_notes(image_path, params=params, key=key)

    print(f"Detected {len(notes)} notes:")
    for i, n in enumerate(notes, 1):
        cx, cy = n.center()
        print(
            f"{i:2d}. degree={n.degree}, "
            f"octave_offset={n.octave_offset:+d}, "
            f"bbox=({n.x},{n.y},{n.w},{n.h}), "
            f"center=({cx},{cy}), "
            f"pitch={n.pitch}, "
            f"articulation={'head' if n.articulation else 'tie'}"
        )

    if notes:
        draw_notes_on_image(image_path, notes, row_center_y=row_center_y, raw_boxes=raw_boxes)
    else:
        print("No notes to draw.")


if __name__ == "__main__":
    main()
