import argparse
import configparser
import os
import sys

"""
主入口：串联 template_match_replace -> notation_detect -> notation_rendering。
参数通过命令行指定乐谱、调号与配置文件；配置文件集中管理模板目录、指法图目录、输出尺寸与偏移等。
"""

from template_match_replace import run_template_match_replace
from notation_detect import run_notation_detect
from notation_rendering import export_notation


def load_config(path: str):
    """读取 config.ini，返回所需参数字典。"""
    cfg = configparser.ConfigParser()
    try:
        read_ok = cfg.read(path, encoding="utf-8")
    except configparser.Error as e:
        raise ValueError(f"配置文件解析失败: {e}")
    if not read_ok:
        raise FileNotFoundError(f"未找到配置文件: {path}")
    if "config" in cfg:
        section = cfg["config"]
    elif configparser.DEFAULTSECT in cfg:
        section = cfg[configparser.DEFAULTSECT]
    else:
        raise ValueError("配置文件缺少 [config] 节")

    def tup_int(val: str):
        parts = [p.strip() for p in val.split(",")]
        return (int(parts[0]), int(parts[1]))

    templates_path = section.get("templates_path", "templates")
    fingering_off_dir = section.get("fingering_off_dir", "")
    fingering_on_dir = section.get("fingering_on_dir", "")
    target_resolution = tup_int(section.get("target_resolution", "1700,550"))
    source_offset = tup_int(section.get("source_offset", "0,0"))
    bg_path = section.get("bg_path", "")
    fingering_img_offset = tup_int(section.get("fingering_img_offset", "0,0"))
    fingering_scale = section.getfloat("fingering_scale", fallback=1.0)

    return {
        "templates_path": templates_path,
        "fingering_off_dir": fingering_off_dir,
        "fingering_on_dir": fingering_on_dir,
        "target_resolution": target_resolution,
        "source_offset": source_offset,
        "bg_path": bg_path,
        "fingering_img_offset": fingering_img_offset,
        "fingering_scale": fingering_scale,
    }


def main():
    parser = argparse.ArgumentParser(description="一键串联 template_match_replace -> notation_detect -> notation_rendering")
    parser.add_argument("notation_img_path", help="原始乐谱图片路径")
    parser.add_argument("key", help="调号")
    parser.add_argument("config_path", help="配置文件 config.ini 路径")
    parser.add_argument("--audio_path", help="音频文件路径（可选，用于 video_rendering）")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config_path)
    except (ValueError, FileNotFoundError, configparser.Error) as e:
        print(f"读取配置失败: {e}")
        return
    templates_path = cfg["templates_path"]
    fingering_off_dir = cfg["fingering_off_dir"]
    fingering_on_dir = cfg["fingering_on_dir"]
    target_resolution = cfg["target_resolution"]
    source_offset = cfg["source_offset"]
    bg_path = cfg["bg_path"]
    fingering_img_offset = cfg["fingering_img_offset"]
    fingering_scale = cfg["fingering_scale"]
    audio_path = args.audio_path

    # 1) 模板匹配与替换
    replaced_img_path, matchinfo_path, saved = run_template_match_replace(args.notation_img_path, templates_path)
    if not saved or not matchinfo_path:
        print("template_match_replace 未保存结果，后续流程终止。")
        return

    # 2) 符号/音符解析
    notes_json_path, saved_detect = run_notation_detect(matchinfo_path, args.key)
    if not saved_detect or not notes_json_path:
        print("notation_detect 未保存结果，后续流程终止。")
        return

    # 3) 渲染指法图
    saved_render = export_notation(
        target_resolution=target_resolution,
        source_offset=source_offset,
        notes_json_path=notes_json_path,
        bg_path=bg_path,
        fingering_off_dir=fingering_off_dir,
        fingering_on_dir=fingering_on_dir,
        fingering_img_offset=fingering_img_offset,
        fingering_scale=fingering_scale,
    )
    if not saved_render:
        print("notation_rendering 未保存结果，流程结束。")
        return

    # 4) 可选：视频渲染
    if audio_path:
        try:
            import video_rendering

            frames_dir = os.path.dirname(os.path.abspath(notes_json_path))
            video_rendering.main_args = None  # 占位避免 lint
            print("启动 video_rendering，请按提示操作...")
            # 直接调用脚本入口
            sys.argv = ["video_rendering.py", frames_dir, audio_path]
            video_rendering.main()
        except Exception as e:
            print(f"video_rendering 调用失败: {e}")


if __name__ == "__main__":
    main()
