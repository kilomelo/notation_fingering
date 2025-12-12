"""
视频渲染工具：将 notation_rendering 输出的图片序列与音频对齐，交互式标记时间点并导出带音轨的视频。

用法（独立运行）:
  python video_rendering.py <frames_dir> <audio_path>

输入：
  frames_dir  ：图片序列目录，文件名格式 XXX_{id}.png，id 从 0 开始递增。
  audio_path  ：音频路径，支持 wav；若安装 pydub+ffmpeg 可支持 mp3。

主要功能：
  - 上半区显示波形并叠加标记/控制按钮；下半区显示当前预览图片。
  - 标记状态：用“标记按钮”选择图片，点击/拖动波形确定时间点。0 号默认为 0s。
  - 播放状态：播放音频，波形背景左/右分色显示进度，下方按时间切换图片。
  - 导出：检查时间点完整、有序且间隔>=0.1s，通过后导出 mp4（使用 cv2 写视频；如安装 moviepy 则复用音轨）。
"""

import os
import sys
import time
import math
import json
import subprocess
import cv2
import numpy as np
from typing import List, Tuple, Optional

try:
    from pydub import AudioSegment
except Exception as e:
    print(f"[pydub import warning] {repr(e)}")
    AudioSegment = None  # type: ignore

# 全局弹窗状态与音频播放
overlay_info = None
audio_proc = None  # ffplay 子进程


def load_audio(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    返回 (波形 float32, 采样率)。双声道仅取左声道。
    使用 pydub 读取（支持 mp3/wav）。
    """
    if AudioSegment is None:
        raise RuntimeError("pydub 未安装，无法读取音频")
    seg = AudioSegment.from_file(audio_path)
    if seg.channels > 1:
        seg = seg.split_to_mono()[0]
    sr = seg.frame_rate
    data = np.array(seg.get_array_of_samples()).astype(np.float32)
    data /= float(1 << (8 * seg.sample_width - 1))
    return data, sr


def load_timestamps(json_path: str, count: int) -> Tuple[List[Optional[int]], int, int]:
    """
    读取毫秒级时间戳，返回 (ts_list, segment_start_ms, segment_end_ms)。
    支持新结构：
    {
      "segment_start_ms": int,
      "segment_end_ms": int,
      "timestamps": {"0": ms, ...}
    }
    兼容旧结构：直接用 {"idx": ms}。
    """
    default_ts = [0] + [None] * (count - 1)
    if not os.path.exists(json_path):
        return default_ts, 0, 0
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "timestamps" in data:
            ts_map = {int(k): int(v) for k, v in data.get("timestamps", {}).items()}
            start_ms = int(data.get("segment_start_ms", 0))
            end_ms = int(data.get("segment_end_ms", 0))
        else:
            ts_map = {int(k): int(v) for k, v in data.items()}
            start_ms = 0
            end_ms = 0
        ts_list: List[Optional[int]] = []
        for i in range(count):
            ts_list.append(ts_map.get(i))
        if ts_list and ts_list[0] is None:
            ts_list[0] = start_ms
        return ts_list, start_ms, end_ms
    except Exception as e:
        print(f"载入时间标记失败: {e}")
        return default_ts, 0, 0


def save_timestamps(json_path: str, timestamps: List[Optional[int]], segment_start_ms: int, segment_end_ms: int):
    payload = {
        "segment_start_ms": int(segment_start_ms),
        "segment_end_ms": int(segment_end_ms),
        "timestamps": {str(i): int(t) for i, t in enumerate(timestamps) if t is not None},
    }
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"已保存时间标记: {os.path.abspath(json_path)}")
    except Exception as e:
        print(f"保存时间标记失败: {e}")


def draw_waveform(
    canvas: np.ndarray,
    wave: np.ndarray,
    sr: int,
    segment_start_ms: int,
    segment_end_ms: int,
    timeline_ms: List[int],
    labels: List[int],
    active_id: int | None,
    timestamps: List[Optional[int]],
    mode: str,
    btn_states: List[bool],
    btn_area: Tuple[int, int, int, int],
    ctrl_area: Tuple[int, int, int, int],
    play_line_x: int | None = None,
    play_line_label: str = "",
):
    h, w = canvas.shape[:2]
    total_duration = max(1e-6, (segment_end_ms - segment_start_ms) / 1000.0)
    # 背景：深灰
    canvas[:] = (40, 40, 40)

    if play_line_x:
        cv2.rectangle(canvas, (0, 0), (play_line_x, h), (64, 64, 64), -1)
        if play_line_label:
            cv2.putText(canvas, play_line_label, (min(play_line_x + 6, w - 80), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA,)

    # 波形绘制（简化 subsample）
    if len(wave) > 1:
        # 进一步提高采样密度以保留更多细节
        step = max(1, int(len(wave) / (w * 6)))
        samples = wave[::step]
        norm = np.clip(samples, -1, 1)
        mid = h // 2
        scale = (h * 0.45)
        pts = np.vstack([np.arange(len(norm)), mid - norm * scale]).T.astype(np.int32)
        pts[:, 0] = (pts[:, 0] / len(norm) * w).astype(np.int32)
        cv2.polylines(canvas, [pts], False, (200, 180, 120), 1, cv2.LINE_AA)
    # 时间标记线，检测乱序：若 t 早于任意前序或晚于任意后序，标红
    rel_times = [(t - segment_start_ms) / 1000.0 for t in timeline_ms]
    for idx, (t_ms, rel_t, lbl) in enumerate(zip(timeline_ms, rel_times, labels)):
        x = int(rel_t / total_duration * w) if total_duration > 0 else 0
        earlier = any(rel_t < rel_times[j] for j in range(idx))
        later = any(rel_t > rel_times[j] for j in range(idx + 1, len(rel_times)))
        if earlier or later:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 255) if lbl % 2 == 0 else (0, 200, 0)  # 偶数黄，奇数绿
        thickness = 1
        if mode == "mark" and active_id is not None and lbl == active_id:
            thickness = 3
        cv2.line(canvas, (x, 0), (x, h), color, thickness)
        # 奇数帧整体下移标签（偏移 32 像素）
        y_base = 18 + (32 if lbl % 2 == 1 else 0)
        cv2.putText(canvas, str(lbl), (x + 2, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{t_ms/1000:.2f}s", (x + 2, y_base + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # 操作区底栏
    btn_x, btn_y, btn_w, btn_h = btn_area
    ctrl_x, ctrl_y, ctrl_w, ctrl_h = ctrl_area
    cv2.rectangle(canvas, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (ctrl_x, ctrl_y), (ctrl_x + ctrl_w, ctrl_y + ctrl_h), (255, 255, 255), -1)
    # 标记按钮：数量为 len(timestamps)-1，对应 1..N-1
    btn_count = len(timestamps) - 1
    if btn_count > 0:
        cell_w = btn_w / btn_count
        for i in range(btn_count):
            x0 = int(btn_x + i * cell_w)
            x1 = int(btn_x + (i + 1) * cell_w)
            y0, y1 = btn_y, btn_y + btn_h
            pressed = btn_states[i]
            # 颜色: 按下橘色，未按下且已标记为浅绿，未标记浅灰
            marked = timestamps[i + 1] is not None
            if pressed:
                color = (0, 165, 255)
            else:
                color = (180, 220, 180) if marked else (200, 200, 200)
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), color, -1)
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (50, 50, 50), 1)
            label = f"{i+1}"
            cv2.putText(canvas, label, (x0 + 6, y0 + btn_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    # 控制按钮（左：播放/停止，右：保存/导出），在绘制时与点击检测保持同样布局
    play_rect = (ctrl_x + 6, ctrl_y + 6, 80, ctrl_h - 12)
    stop_rect = (ctrl_x + 92, ctrl_y + 6, 80, ctrl_h - 12)
    export_rect = (ctrl_x + ctrl_w - 86, ctrl_y + 6, 80, ctrl_h - 12)
    save_rect = (export_rect[0] - 86, ctrl_y + 6, 80, ctrl_h - 12)
    for rect, text in [
        (play_rect, "Play" if mode == "mark" else "Playing"),
        (stop_rect, "Stop"),
        (save_rect, "Save"),
        (export_rect, "Export"),
    ]:
        rx, ry, rw, rh = rect
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (220, 220, 220), -1)
        cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), (50, 50, 50), 1)
        cv2.putText(canvas, text, (rx + 10, ry + rh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    # 状态提示（更新快捷键说明）
    cv2.putText(canvas, "Space switch (mark) | S save (mark) | P play | O stop | E export", (10, btn_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def validate_timestamps(ts: List[float]) -> Tuple[bool, str]:
    if any(t is None for t in ts):
        return False, "Some frames are not marked"
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            return False, "Timestamps must be strictly increasing"
        if ts[i] - ts[i - 1] < 0.1:
            return False, "Adjacent timestamps must be >= 0.1s apart"
    return True, ""


def export_video(
    frames: List[np.ndarray],
    timestamps_ms: List[Optional[int]],
    audio_path: str,
    out_path: str,
    segment_start_ms: int,
    segment_end_ms: int,
    fps: int = 30,
):
    """按時間戳（毫秒）組裝視頻，時長為截取段；再用 ffmpeg 合成截取音頻。"""
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    segment_duration_ms = max(0, segment_end_ms - segment_start_ms)
    total_duration = segment_duration_ms / 1000.0
    ts_sec = [t / 1000.0 for t in timestamps_ms if t is not None]
    # 填充 None -> segment_end_ms
    ts_full = [t if t is not None else segment_end_ms for t in timestamps_ms]
    ts_full_sec = [t / 1000.0 for t in ts_full]
    total_frames = int(math.ceil(total_duration * fps))
    for i in range(total_frames):
        t = i / fps
        # 找到當前時間應該顯示的帧（绝对时间 = segment_start + t）
        abs_t = segment_start_ms / 1000.0 + t
        idx = 0
        for j in range(len(ts_full_sec)):
            if abs_t >= ts_full_sec[j]:
                idx = j
            else:
                break
        writer.write(frames[idx])
    writer.release()
    # 使用 ffmpeg 合成音频（截取区间）
    tmp_out = out_path.replace(".mp4", "_with_audio.mp4")
    try:
        ss_arg = str(segment_start_ms / 1000.0)
        to_arg = str(segment_end_ms / 1000.0)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            out_path,  # 视频
            "-ss",
            ss_arg,
            "-to",
            to_arg,
            "-i",
            audio_path,  # 音频截取
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            tmp_out,
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp_out, out_path)
    except Exception as e:
        print(f"提示: ffmpeg 合成音频失败，输出为无音轨视频。错误: {e}")


def show_overlay_state(text: str, duration_ms: int = 2000, color=(0, 165, 255)):
    """设置全局弹窗状态（非阻塞），下一帧渲染时绘制，后续弹窗会覆盖前一个。"""
    global overlay_info
    overlay_info = {
        "text": text,
        "color": color,
        "expire": time.time() + duration_ms / 1000.0,
    }


def apply_overlay(canvas: np.ndarray, w: int, h: int):
    """根据全局 overlay_info 在画布上绘制提示。"""
    global overlay_info
    if overlay_info and time.time() < overlay_info["expire"]:
        text = overlay_info["text"]
        color = overlay_info["color"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pad = 10
        rect_w = tw + pad * 2
        rect_h = th + pad * 2
        x0 = w // 2 - rect_w // 2
        y0 = h // 2 - rect_h // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + rect_w, y0 + rect_h), color, -1)
        cv2.addWeighted(
            overlay[y0:y0 + rect_h, x0:x0 + rect_w],
            0.9,
            canvas[y0:y0 + rect_h, x0:x0 + rect_w],
            0.1,
            0,
            canvas[y0:y0 + rect_h, x0:x0 + rect_w],
        )
        cv2.putText(canvas, text, (x0 + pad, y0 + rect_h - pad - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    elif overlay_info and time.time() >= overlay_info["expire"]:
        overlay_info = None
    return canvas


def load_frames(frames_dir: str) -> List[np.ndarray]:
    files = [f for f in os.listdir(frames_dir) if f.lower().endswith(".png")]
    # 提取 id
    items = []
    for f in files:
        stem = os.path.splitext(f)[0]
        if "_" not in stem:
            continue
        try:
            idx = int(stem.split("_")[-1])
        except ValueError:
            continue
        items.append((idx, f))
    items.sort(key=lambda x: x[0])
    frames = []
    base_size = None
    for idx, name in items:
        img = cv2.imread(os.path.join(frames_dir, name), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: 無法讀取圖片 {name}，跳過。")
            continue
        if base_size is None:
            base_size = img.shape[:2]
        elif img.shape[:2] != base_size:
            print(f"Warning: 尺寸不一致，跳過 {name}")
            continue
        frames.append(img)
    if not frames:
        raise ValueError("未找到有效圖片序列")
    return frames




def main():
    if len(sys.argv) < 3:
        print("用法: python video_rendering.py <frames_dir> <audio_path>")
        return
    frames_dir = sys.argv[1]
    audio_path = sys.argv[2]

    frames = load_frames(frames_dir)
    frame_h, w = frames[0].shape[:2]
    h = frame_h  # 旧变量沿用，但下方引入拆分
    # 区间状态（毫秒）：默认全长，若 json 存在则使用其中的值
    segment_start_ms = 0
    wave_data, sr = load_audio(audio_path)
    total_duration_ms = int(len(wave_data) / sr * 1000)
    segment_end_ms = total_duration_ms
    video_path = os.path.join(frames_dir, os.path.basename(frames_dir) + ".mp4")
    ts_json_path = video_path.replace(".mp4", ".json")
    try:
        wave_data, sr = load_audio(audio_path)
    except Exception as e:
        print(f"音頻讀取失敗: {e}")
        return
    duration = len(wave_data) / sr
    current_play_time = None
    global overlay_info
    # 畫布
    canvas_h = h * 2
    canvas_w = w
    # 時間戳初始化：尝试从同名 json 读取
    video_path = os.path.join(frames_dir, os.path.basename(frames_dir) + ".mp4")
    ts_json_path = video_path.replace(".mp4", ".json")
    timestamps, segment_start_ms_loaded, segment_end_ms_loaded = load_timestamps(ts_json_path, len(frames))
    if segment_end_ms_loaded > 0:
        segment_start_ms = segment_start_ms_loaded
        segment_end_ms = segment_end_ms_loaded
    # 计算截取段采样范围
    start_idx = int(segment_start_ms * sr / 1000)
    end_idx = int(segment_end_ms * sr / 1000)
    wave_data_segment = wave_data[start_idx:end_idx] if end_idx > start_idx else wave_data[:1]
    duration = (segment_end_ms - segment_start_ms) / 1000.0
    play_ts = list(timestamps)
    active_btn = None  # 當前按下的標記按鈕 (索引 >=1)

    # 波形区域拆分：上1/3为全波形，下2/3为截取段（含按钮）
    global_wave_h = frame_h // 3
    lower_wave_h = frame_h - global_wave_h
    # 区域定义（基于下半截取段区域）
    btn_area = (0, lower_wave_h - 80, w, 40)     # 标记按钮条
    ctrl_area = (0, lower_wave_h - 40, w, 40)    # 控制按钮条

    def render_wave_with_marks(active, play_line_x=None, play_line_label=""):
        # 仅绘制截取段波形与控件，高度为下部区域
        top = np.zeros((lower_wave_h, w, 3), dtype=np.uint8)
        ts_marked = [t for t in timestamps if t is not None]
        lbls = [i for i, t in enumerate(timestamps) if t is not None]
        draw_waveform(
            top,
            wave_data_segment,
            sr,
            segment_start_ms,
            segment_end_ms,
            ts_marked if ts_marked else [segment_start_ms],
            lbls if lbls else [0],
            active,
            timestamps,
            mode,
            [active is not None and i + 1 == active for i in range(len(timestamps) - 1)],
            btn_area,
            ctrl_area,
            play_line_x=play_line_x,
            play_line_label=play_line_label,
        )
        return top

    mode = "mark"  # mark / play
    preview_idx = 0
    play_start_time = None
    play_real_start = None
    timestamps, loaded_start, loaded_end = load_timestamps(ts_json_path, len(frames))
    if loaded_end > 0:
        segment_start_ms = loaded_start
        segment_end_ms = loaded_end
    play_ts = list(timestamps)

    def get_frame_for_time(t, ts_list=None):
        idx = 0
        ts_source = ts_list if ts_list is not None else timestamps
        t_ms = int(t * 1000)
        for j, ts in enumerate(ts_source):
            if ts is not None and t_ms >= ts:
                idx = j
            else:
                break
        return frames[idx]

    # 音频播放控制：使用 ffplay 子进程，稳定且可终止
    def start_audio_playback():
        global audio_proc
        stop_audio_playback()
        try:
            dur_sec = max(0, (segment_end_ms - segment_start_ms) / 1000.0)
            audio_proc = subprocess.Popen(
                [
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    "-ss",
                    str(segment_start_ms / 1000.0),
                    "-t",
                    str(dur_sec),
                    "-i",
                    os.path.abspath(audio_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"音频播放失败，进度仅可视化: {e}")
            audio_proc = None

    def stop_audio_playback():
        global audio_proc
        if audio_proc is not None:
            try:
                audio_proc.terminate()
            except Exception:
                pass
        audio_proc = None

    drag_active = False
    drag_start_x = 0
    drag_cur_x = 0

    def x_to_ms(x_pos: int) -> int:
        ratio = max(0.0, min(1.0, x_pos / w))
        return int(ratio * total_duration_ms)

    def draw_global_wave(
        override_start_ms: int | None = None,
        override_end_ms: int | None = None,
        override_color: Tuple[int, int, int] = (80, 80, 80),
    ):
        gw = np.zeros((global_wave_h, w, 3), dtype=np.uint8)
        # 标出截取段在全波形上的位置（可被拖拽覆盖）
        rect_start = segment_start_ms if override_start_ms is None else override_start_ms
        rect_end = segment_end_ms if override_end_ms is None else override_end_ms
        if total_duration_ms > 0:
            start_ratio = max(0.0, min(1.0, rect_start / total_duration_ms))
            end_ratio = max(start_ratio, min(1.0, rect_end / total_duration_ms))
            x0 = int(start_ratio * w)
            x1 = int(end_ratio * w)
            cv2.rectangle(gw, (x0, 0), (x1, global_wave_h), override_color, thickness=-1)

        if len(wave_data) > 1:
            # 进一步提高采样密度以保留更多细节
            step = max(1, int(len(wave_data) / (w * 6)))
            samples = wave_data[::step]
            norm = np.clip(samples, -1, 1)
            mid = global_wave_h // 2
            scale = (global_wave_h * 0.45)
            pts = np.vstack([np.arange(len(norm)), mid - norm * scale]).T.astype(np.int32)
            pts[:, 0] = (pts[:, 0] / len(norm) * w).astype(np.int32)
            cv2.polylines(gw, [pts], False, (120, 180, 220), 1, cv2.LINE_AA)
        cv2.putText(gw, "Full wave", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return gw

    while True:
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        play_line_x = None
        play_line_label = ""
        bottom_frame = frames[preview_idx]
        if mode == "play" and current_play_time is not None and duration > 0:
            progress = max(0.0, min(1.0, current_play_time / duration))
            play_line_x = int(progress * w)
            play_line_label = f"{current_play_time:.2f}s"
            ts_source = play_ts if play_ts is not None else [t if t is not None else segment_end_ms for t in timestamps]
            t_abs_sec = segment_start_ms / 1000.0 + current_play_time
            bottom_frame = get_frame_for_time(t_abs_sec, ts_source)
        # 上半部分拆分：全波形+截取波形区域
        top_canvas = np.zeros((frame_h, w, 3), dtype=np.uint8)
        override_start = override_end = None
        override_color = (80, 80, 80)
        left_label = right_label = None
        if drag_active:
            x0 = min(drag_start_x, drag_cur_x)
            x1 = max(drag_start_x, drag_cur_x)
            override_start = x_to_ms(x0)
            override_end = x_to_ms(x1)
            if drag_cur_x < drag_start_x:
                override_color = (0, 0, 180)  # 深红示警
            left_label = f"{x_to_ms(drag_start_x)/1000:.2f}s"
            right_label = f"{x_to_ms(drag_cur_x)/1000:.2f}s"
        gw_img = draw_global_wave(override_start, override_end, override_color)
        # 绘制拖拽时的左右 label
        if drag_active and override_start is not None and override_end is not None and override_color != (0, 0, 180):
            x0 = min(drag_start_x, drag_cur_x)
            x1 = max(drag_start_x, drag_cur_x)
            y_text = 14
            if left_label:
                cv2.putText(gw_img, left_label, (x0, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if right_label:
                # 右侧 label 的左端对齐矩形右端
                cv2.putText(gw_img, right_label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        top_canvas[0:global_wave_h, 0:w, :] = gw_img
        lower_canvas = render_wave_with_marks(
            active_btn if mode == "mark" else None, play_line_x=play_line_x, play_line_label=play_line_label
        )
        top_canvas[global_wave_h:frame_h, 0:w, :] = lower_canvas
        canvas[0:frame_h, 0:w, :] = top_canvas
        canvas[frame_h:, 0:w, :] = bottom_frame

        # 绘制弹窗（如有）
        canvas = apply_overlay(canvas, w, h)

        cv2.imshow("video_render", canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC 退出
            stop_audio_playback()
            break
        if mode == "mark":
            # 空格切換預覽圖片
            if key == 32:
                preview_idx = (preview_idx + 1) % len(frames)
                active_btn = preview_idx if preview_idx != 0 else None
            # 保存快捷键
            if key == ord("s"):
                save_timestamps(ts_json_path, timestamps, segment_start_ms, segment_end_ms)
                abs_json = os.path.abspath(ts_json_path)
                show_overlay_state(f"Saved: {abs_json}", duration_ms=1500)
            # 播放
            if key == ord("p"):
                # 重置播放狀態
                mode = "play"
                current_play_time = 0.0
                play_real_start = time.time()
                active_btn = None
                start_audio_playback()
                play_ts = [t if t is not None else segment_end_ms for t in timestamps]
            # 鼠標事件設置在 window callback
            # 按 e 導出
            if key == ord("e"):
                ok, msg = validate_timestamps(timestamps)
                if not ok:
                    print(f"Error: {msg}")
                    show_overlay_state(f"Validation failed: {msg}", duration_ms=2000)
                    continue
                out_path = os.path.join(frames_dir, os.path.basename(frames_dir) + ".mp4")
                abs_out = os.path.abspath(out_path)
                print(f"開始導出視頻 -> {abs_out}")
                # 生成視頻（使用標記的 timestamps）
                filled_ts = [t if t is not None else segment_end_ms for t in timestamps]
                export_video(frames, filled_ts, audio_path, out_path, segment_start_ms, segment_end_ms)
                print("導出完成。")
                save_timestamps(out_path.replace(".mp4", ".json"), timestamps, segment_start_ms, segment_end_ms)
                show_overlay_state("Export completed", duration_ms=2000)
        else:  # play
            # 播放状态下忽略空格/标记/保存操作，停止键改为 o
            if key in (32, ord("s")):
                pass
            if key == ord("o"):
                stop_audio_playback()
                current_play_time = None
                mode = "mark"
                continue
            if play_real_start is not None:
                current_play_time = time.time() - play_real_start
                # print(f"[play] t_now={t_now:.3f}s / {duration:.3f}s")
                if current_play_time >= duration:
                    stop_audio_playback()
                    current_play_time = None
                    mode = "mark"
                else:
                    pass

        # 鼠標回調（標記）
        # 控件位置（用于点击检测，与绘制一致，基于截取波形区域的坐标系）
        ctrl_x, ctrl_y_rel, ctrl_w, ctrl_h = 0, lower_wave_h - 40, w, 40
        btn_y_rel = lower_wave_h - 80
        play_rect = (ctrl_x + 6, ctrl_y_rel + 6, 80, ctrl_h - 12)
        stop_rect = (ctrl_x + 92, ctrl_y_rel + 6, 80, ctrl_h - 12)
        export_rect = (ctrl_x + ctrl_w - 86, ctrl_y_rel + 6, 80, ctrl_h - 12)
        save_rect = (export_rect[0] - 86, ctrl_y_rel + 6, 80, ctrl_h - 12)

        def in_rect(px, py, rect):
            rx, ry, rw, rh = rect
            return rx <= px <= rx + rw and ry <= py <= ry + rh

        def on_mouse(event, x, y, flags, param):
            nonlocal active_btn, timestamps, preview_idx, mode, play_real_start, current_play_time, play_ts
            nonlocal segment_start_ms, segment_end_ms, wave_data_segment, duration, drag_active, drag_start_x, drag_cur_x

            # 若正在拖拽且鼠标已离开全波形区域，则立即取消
            if drag_active and y >= global_wave_h:
                drag_active = False
                drag_start_x = drag_cur_x = 0
                return

            # 顶部全波形区域交互：拖拽设定截取段
            if y < global_wave_h:
                if event == cv2.EVENT_LBUTTONDOWN:
                    drag_active = True
                    drag_start_x = x
                    drag_cur_x = x
                    active_btn = None  # 开始拖拽时取消当前标记
                elif event == cv2.EVENT_MOUSEMOVE and drag_active:
                    if y < 0 or y >= global_wave_h:
                        # 移出区域取消并重置状态
                        drag_active = False
                        drag_start_x = drag_cur_x = 0
                    else:
                        drag_cur_x = x
                elif event == cv2.EVENT_LBUTTONUP and drag_active:
                    drag_active = False
                    # 只有在全波形区域内抬起才判定
                    if 0 <= y < global_wave_h:
                        x0 = drag_start_x
                        x1 = x
                        if x1 == x0:
                            return
                        start_ms = x_to_ms(min(x0, x1))
                        end_ms = x_to_ms(max(x0, x1))
                        if end_ms - start_ms < 2000:
                            return  # 无效拖拽
                        old_start, old_end = segment_start_ms, segment_end_ms
                        # 更新截取段与波形片段
                        segment_start_ms = start_ms
                        segment_end_ms = end_ms
                        start_idx = int(segment_start_ms * sr / 1000)
                        end_idx = int(segment_end_ms * sr / 1000)
                        wave_data_segment = wave_data[start_idx:end_idx] if end_idx > start_idx else wave_data[:1]
                        duration = (segment_end_ms - segment_start_ms) / 1000.0
                        # 更新时间戳：0 号为段起点，超出范围的清除
                        timestamps[0] = segment_start_ms
                        cleared = []
                        for i, t in enumerate(timestamps):
                            if t is not None and (t < segment_start_ms or t > segment_end_ms):
                                timestamps[i] = None
                                cleared.append(i)
                        # 播放时的填充列表同步
                        play_ts = [t if t is not None else segment_end_ms for t in timestamps]
                        print(
                            f"Segment updated: {old_start}ms-{old_end}ms -> {segment_start_ms}ms-{segment_end_ms}ms; "
                            f"cleared indices: {cleared if cleared else 'none'}"
                        )
                return

            # 仅允许在截取段波形区域（下 2/3）内交互
            y_rel = y - global_wave_h
            if y_rel < 0 or y_rel >= lower_wave_h:
                return

            def set_time_from_x():
                nonlocal active_btn, preview_idx
                if active_btn is not None and mode == "mark":
                    # 将鼠标位置映射到截取区间的绝对毫秒时间
                    span_ms = max(1, segment_end_ms - segment_start_ms)
                    t_ms = int(segment_start_ms + (x / w) * span_ms)
                    timestamps[active_btn] = t_ms
                    preview_idx = active_btn

            if event == cv2.EVENT_LBUTTONDOWN:
                # 控制區: 左側播放/停止，右側導出（優先處理，避免穿透）
                if ctrl_y_rel <= y_rel < ctrl_y_rel + ctrl_h:
                    if in_rect(x, y_rel, play_rect) and mode == "mark":
                        mode = "play"
                        current_play_time = 0
                        play_real_start = time.time()
                        active_btn = None
                        play_ts = [t if t is not None else segment_end_ms for t in timestamps]
                        start_audio_playback()
                        return
                    elif in_rect(x, y_rel, stop_rect) and mode == "play":
                        stop_audio_playback()
                        mode = "mark"
                        play_real_start = None
                        return
                    elif in_rect(x, y_rel, save_rect) and mode == "mark":
                        save_timestamps(ts_json_path, timestamps, segment_start_ms, segment_end_ms)
                        abs_json = os.path.abspath(ts_json_path)
                        show_overlay_state(f"Saved: {abs_json}", duration_ms=1500)
                        return
                    elif in_rect(x, y_rel, export_rect) and mode == "mark":
                        ok, msg = validate_timestamps(timestamps)
                        if not ok:
                            print(f"Error: {msg}")
                            show_overlay_state(f"Validation failed: {msg}", duration_ms=2000)
                            return
                        out_path = os.path.join(frames_dir, os.path.basename(frames_dir) + ".mp4")
                        filled_ts = [t if t is not None else segment_end_ms for t in timestamps]
                        abs_out = os.path.abspath(out_path)
                        print(f"Exporting video -> {abs_out}")
                        show_overlay_state("Exporting...", duration_ms=10_000_000)  # 会被后续提示覆盖
                        # 立即刷新一次界面显示 Exporting 提示
                        temp_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
                        temp_top = np.zeros((frame_h, w, 3), dtype=np.uint8)
                        temp_top[0:global_wave_h, 0:w, :] = draw_global_wave()
                        temp_top[global_wave_h:frame_h, 0:w, :] = render_wave_with_marks(active_btn)
                        temp_canvas[0:frame_h, 0:w, :] = temp_top
                        temp_canvas[frame_h:, 0:w, :] = frames[preview_idx]
                        temp_canvas = apply_overlay(temp_canvas, w, h)
                        cv2.imshow("video_render", temp_canvas)
                        cv2.waitKey(1)
                        try:
                            stop_audio_playback()
                            export_video(frames, filled_ts, audio_path, out_path, segment_start_ms, segment_end_ms)
                            print("Export done.")
                            save_timestamps(out_path.replace(".mp4", ".json"), timestamps, segment_start_ms, segment_end_ms)
                            # 导出完成提示
                            show_overlay_state("Export completed", duration_ms=2000)
                        except Exception as e:
                            err_text = f"Export failed: {e}"
                            print(err_text)
                            show_overlay_state(err_text, duration_ms=2000)
                        return
                # 點擊標記按鈕區
                if btn_y_rel <= y_rel < ctrl_y_rel:
                    if mode == "play":
                        return
                    btn_count = len(frames) - 1
                    if btn_count > 0:
                        btn_w = w / btn_count
                        idx = int(x / btn_w) + 1
                        idx = min(max(1, idx), len(frames) - 1)
                        active_btn = idx
                        preview_idx = idx
                    return
                # 點擊波形區設置時間（仅标记按钮上沿以上区域）
                if 0 <= y_rel < btn_y_rel:
                    set_time_from_x()
            elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
                # 拖动时忽略按钮/控制区
                if y_rel >= btn_y_rel:
                    return
                if 0 <= y_rel < btn_y_rel:
                    set_time_from_x()

        cv2.setMouseCallback("video_render", on_mouse)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
