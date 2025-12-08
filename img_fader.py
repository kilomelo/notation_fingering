import os
from PIL import Image
import argparse
import sys

def apply_fade_effect(image, fade_height_percent):
    """
    对单张图片应用上下渐变透明效果，保留源图片的alpha通道
    
    参数:
    image: PIL Image对象
    fade_height_percent: 渐变高度百分比（0-0.5）
    
    返回:
    应用了渐变效果的PIL Image对象
    """
    # 转换为RGBA模式以支持透明度
    if image.mode != 'RGBA':
        rgba_image = image.convert('RGBA')
    else:
        rgba_image = image.copy()
    
    width, height = rgba_image.size
    
    # 计算渐变区域的高度
    fade_height = int(height * fade_height_percent)
    
    # 获取原始alpha通道（如果存在）
    if image.mode == 'RGBA':
        original_alpha = rgba_image.getchannel('A')
    else:
        # 如果没有alpha通道，创建全不透明的alpha通道
        original_alpha = Image.new('L', (width, height), 255)
    
    # 创建渐变蒙版（初始为全不透明）
    gradient_mask = Image.new('L', (width, height), 255)
    
    # 应用顶部渐变（从上到下，透明度从0到255）
    for y in range(fade_height):
        # 计算当前行的透明度（从0线性增加到255）
        alpha_value = int(255 * (y / fade_height))
        # 设置整行的透明度
        for x in range(width):
            gradient_mask.putpixel((x, y), alpha_value)
    
    # 应用底部渐变（从下到上，透明度从0到255）
    for y in range(height - fade_height, height):
        # 计算当前行的透明度（从255线性减少到0）
        alpha_value = int(255 * ((height - y - 1) / fade_height))
        # 设置整行的透明度
        for x in range(width):
            gradient_mask.putpixel((x, y), alpha_value)
    
    # 计算最终的alpha通道：渐变alpha × 源图片alpha / 255
    # 因为两个alpha值都是0-255，相乘后需要除以255来归一化
    final_alpha_data = []
    
    # 获取两个通道的像素数据
    gradient_pixels = gradient_mask.getdata()
    original_alpha_pixels = original_alpha.getdata()
    
    # 逐像素计算最终alpha值
    for i in range(len(gradient_pixels)):
        grad_alpha = gradient_pixels[i]
        orig_alpha = original_alpha_pixels[i]
        # 计算乘积并归一化到0-255范围
        final_alpha = (grad_alpha * orig_alpha) // 255
        final_alpha_data.append(final_alpha)
    
    # 创建最终的alpha通道
    final_alpha_channel = Image.new('L', (width, height))
    final_alpha_channel.putdata(final_alpha_data)
    
    # 将RGB通道与最终的alpha通道合并
    r, g, b, _ = rgba_image.split()
    result_image = Image.merge('RGBA', (r, g, b, final_alpha_channel))
    
    return result_image

def batch_fade_images(input_dir, fade_height_percent):
    """
    批量处理图片添加渐变透明效果
    
    参数:
    input_dir: 图片所在目录
    fade_height_percent: 渐变高度百分比（0-0.5）
    """
    # 检查渐变高度百分比是否有效
    if not 0 <= fade_height_percent <= 0.5:
        print("错误: 渐变高度百分比必须在0到0.5之间")
        return
    
    # 创建输出目录
    base_dir = os.path.dirname(os.path.abspath(input_dir))
    dir_name = os.path.basename(os.path.abspath(input_dir))
    output_dir = os.path.join(base_dir, f"{dir_name}_fade")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"渐变高度百分比: {fade_height_percent*100}%")
    print(f"渐变区域高度: 顶部和底部各{int(fade_height_percent*100)}%")
    print(f"Alpha处理: 输出alpha = 渐变alpha × 源图片alpha / 255")
    print("-" * 50)
    
    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp', '.PNG', '.JPG', '.JPEG')
    
    # 统计处理结果
    processed_count = 0
    error_count = 0
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)
            
            # 输出文件保存为PNG格式以保留透明度
            output_path = os.path.join(output_dir, f"{name}.png")
            
            try:
                # 打开图片
                with Image.open(input_path) as img:
                    # 记录原始模式和信息
                    original_mode = img.mode
                    original_size = img.size
                    
                    # 应用渐变效果
                    faded_img = apply_fade_effect(img, fade_height_percent)
                    
                    # 保存为PNG格式以保留透明度
                    faded_img.save(output_path, 'PNG')
                    
                    # 计算渐变区域像素数
                    fade_pixels = int(original_size[1] * fade_height_percent)
                    alpha_info = "有Alpha通道" if original_mode == 'RGBA' else "无Alpha通道"
                    
                    print(f"✓ 已处理: {filename} ({original_size[0]}x{original_size[1]}, {alpha_info}, 渐变区域: {fade_pixels}像素)")
                    processed_count += 1
                    
            except Exception as e:
                print(f"✗ 处理失败: {filename} - 错误: {str(e)}")
                error_count += 1
    
    print("-" * 50)
    print(f"处理完成! 成功: {processed_count}, 失败: {error_count}")
    # 记录参数
    params_path = os.path.join(output_dir, "parameters.ini")
    try:
        with open(params_path, "w", encoding="utf-8") as f:
            f.write(f"{input_dir} {fade_height_percent}")
    except Exception as e:
        print(f"Warning: 写入参数文件失败: {params_path} - {e}")
    print(f"所有处理后的图片已保存为PNG格式以保留透明度信息")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='批量图片渐变透明效果工具（修复Alpha通道问题）')
    parser.add_argument('input_dir', help='图片所在目录路径')
    parser.add_argument('fade_percent', type=float, 
                       help='渐变高度百分比（0-0.5，如0.2表示20%）')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 目录 '{args.input_dir}' 不存在")
        return
    
    # 检查渐变百分比是否有效
    if not 0 <= args.fade_percent <= 0.5:
        print("错误: 渐变高度百分比必须在0到0.5之间")
        return
    
    # 执行批量处理
    batch_fade_images(args.input_dir, args.fade_percent)

# 测试函数
def test_alpha_preservation():
    """测试函数，验证Alpha通道保留功能"""
    print("测试Alpha通道保留功能...")
    
    # 创建一个测试图片（带自定义Alpha通道）
    test_img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))  # 半透明红色
    test_img.save('test_input.png')
    
    # 应用渐变效果
    result = apply_fade_effect(test_img, 0.2)
    result.save('test_output.png')
    
    # 验证结果
    input_alpha = test_img.getpixel((50, 50))[3]  # 源图片中心点的alpha值
    output_alpha = result.getpixel((50, 50))[3]   # 结果图片中心点的alpha值
    gradient_alpha = 255  # 中心点应该是完全不透明
    
    expected_alpha = (gradient_alpha * input_alpha) // 255
    
    print(f"输入Alpha: {input_alpha}, 输出Alpha: {output_alpha}, 期望Alpha: {expected_alpha}")
    print(f"Alpha保留测试: {'通过' if output_alpha == expected_alpha else '失败'}")
    
    # 清理测试文件
    if os.path.exists('test_input.png'):
        os.remove('test_input.png')
    if os.path.exists('test_output.png'):
        os.remove('test_output.png')

if __name__ == "__main__":
    # 如果直接运行脚本，使用示例参数（用于测试）
    if len(sys.argv) == 1:
        print("修复版图片批量渐变透明效果工具")
        print("=" * 60)
        print("主要修复: 正确处理带Alpha通道的源图片")
        print("Alpha计算: 输出Alpha = 渐变Alpha × 源图片Alpha / 255")
        print("=" * 60)
        print("使用示例:")
        print("python img_fader_fixed.py /path/to/images 0.2")
        print("\n参数说明:")
        print("input_dir: 图片所在目录")
        print("fade_percent: 渐变高度百分比（0-0.5）")
        print("            例如：0.2 表示图片顶部和底部各20%的高度区域会有渐变透明效果")
        print("\n运行测试: python image_fader_fixed.py --test")
        print("\n效果说明:")
        print("- 图片顶部：从上到下，从完全透明渐变到不透明")
        print("- 图片底部：从下到上，从完全透明渐变到不透明") 
        print("- 中间部分：保持原图透明度")
        print("- Alpha通道：正确保留源图片的透明度信息")
    elif len(sys.argv) == 2 and sys.argv[1] == '--test':
        test_alpha_preservation()
    else:
        main()
