from template_match_replace import run_template_match_replace
from notation_detect import run_notation_detect
from notation_rendering import export_notation


def main():
    # 基本参数（按需调整）
    notation_img_path = "p.png"          # 原始乐谱图片路径
    key = "G"                            # 调号
    templates_path = "templates"         # 模板目录
    fingering_img_path = "fingering_img_cut_fade"  # 指法图目录
    target_resolution = (1500, 550)      # 输出画布 (width, height)
    source_offset = (60, 10)             # 乐谱在输出中的偏移 (x, y)
    bg_path = "bg.png"                   # 背景图路径，空则白底
    fingering_img_offset = (0, 50)       # 指法图顶边中点相对音符中心/行的偏移
    fingering_scale = 0.25               # 指法图缩放系数

    # 1) 模板匹配与替换
    replaced_img_path, matchinfo_path, saved = run_template_match_replace(notation_img_path, templates_path)
    if not saved or not matchinfo_path:
        print("template_match_replace 未保存结果，后续流程终止。")
        return

    # 2) 符号/音符解析
    notes_json_path, saved_detect = run_notation_detect(matchinfo_path, key)
    if not saved_detect or not notes_json_path:
        print("notation_detect 未保存结果，后续流程终止。")
        return

    # 3) 渲染指法图
    saved_render = export_notation(
        target_resolution=target_resolution,
        source_offset=source_offset,
        notes_json_path=notes_json_path,
        bg_path=bg_path,
        fingering_img_path=fingering_img_path,
        fingering_img_offset=fingering_img_offset,
        fingering_scale=fingering_scale,
    )
    if not saved_render:
        print("notation_rendering 未保存结果，流程结束。")


if __name__ == "__main__":
    main()
