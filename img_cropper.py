import os
from PIL import Image
import argparse

def batch_crop_images(input_dir, target_width, target_height, offset_x=0, offset_y=0):
    """
    批量裁剪图片工具
    
    参数:
    input_dir: 图片所在目录
    target_width: 目标宽度
    target_height: 目标高度
    offset_x: 中心点X轴偏移量（正数向右偏移）
    offset_y: 中心点Y轴偏移量（正数向下偏移）
    """
    # 创建输出目录
    base_dir = os.path.dirname(input_dir)
    dir_name = os.path.basename(input_dir)
    output_dir = os.path.join(base_dir, f"{dir_name}_cut")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标分辨率: {target_width}x{target_height}")
    print(f"中心点偏移: X={offset_x}, Y={offset_y}")
    print("-" * 50)
    
    # 支持的图片格式
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp')
    
    # 统计处理结果
    processed_count = 0
    error_count = 0
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 打开图片
                with Image.open(input_path) as img:
                    original_width, original_height = img.size
                    
                    # 计算源图片中心点
                    center_x = original_width / 2
                    center_y = original_height / 2
                    
                    # 计算目标中心点（应用偏移）
                    target_center_x = center_x + offset_x
                    target_center_y = center_y + offset_y
                    
                    # 计算裁剪区域
                    left = target_center_x - target_width / 2
                    top = target_center_y - target_height / 2
                    right = left + target_width
                    bottom = top + target_height
                    
                    # 确保裁剪区域不超出图片边界
                    left = max(0, left)
                    top = max(0, top)
                    right = min(original_width, right)
                    bottom = min(original_height, bottom)
                    
                    # 如果裁剪区域小于目标分辨率，进行调整
                    actual_width = right - left
                    actual_height = bottom - top
                    
                    if actual_width < target_width or actual_height < target_height:
                        # 创建目标大小的新图片（黑色背景）
                        new_img = Image.new('RGB', (target_width, target_height), (0, 0, 0))
                        
                        # 如果是RGBA模式，创建透明背景
                        if img.mode == 'RGBA':
                            new_img = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
                        
                        # 计算粘贴位置（居中）
                        paste_x = (target_width - actual_width) // 2 if actual_width < target_width else 0
                        paste_y = (target_height - actual_height) // 2 if actual_height < target_height else 0
                        
                        # 裁剪原图
                        cropped_img = img.crop((left, top, right, bottom))
                        
                        # 将裁剪的图片粘贴到新图片上
                        new_img.paste(cropped_img, (paste_x, paste_y))
                        
                        # 保存图片
                        if img.mode == 'RGBA':
                            new_img.save(output_path)
                        else:
                            # 保持原格式，但JPEG不支持透明通道
                            if filename.lower().endswith(('.jpg', '.jpeg')):
                                new_img = new_img.convert('RGB')
                            new_img.save(output_path)
                    else:
                        # 直接裁剪并保存
                        cropped_img = img.crop((left, top, right, bottom))
                        cropped_img.save(output_path)
                    
                    print(f"✓ 已处理: {filename} ({original_width}x{original_height} → {target_width}x{target_height})")
                    processed_count += 1
                    
            except Exception as e:
                print(f"✗ 处理失败: {filename} - 错误: {str(e)}")
                error_count += 1
    
    print("-" * 50)
    print(f"处理完成! 成功: {processed_count}, 失败: {error_count}")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='批量图片裁剪工具')
    parser.add_argument('input_dir', help='图片所在目录路径')
    parser.add_argument('width', type=int, help='目标宽度（像素）')
    parser.add_argument('height', type=int, help='目标高度（像素）')
    parser.add_argument('--offset_x', type=int, default=0, help='中心点X轴偏移量（默认: 0）')
    parser.add_argument('--offset_y', type=int, default=0, help='中心点Y轴偏移量（默认: 0）')
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"错误: 目录 '{args.input_dir}' 不存在")
        return
    
    # 检查目标分辨率是否有效
    if args.width <= 0 or args.height <= 0:
        print("错误: 目标宽度和高度必须大于0")
        return
    
    # 执行批量裁剪
    batch_crop_images(args.input_dir, args.width, args.height, args.offset_x, args.offset_y)

if __name__ == "__main__":
    # 如果直接运行脚本，使用示例参数（用于测试）
    if len(os.sys.argv) == 1:
        print("使用示例:")
        print("python img_cropper.py /path/to/images 800 600 --offset_x 50 --offset_y -30")
        print("\n参数说明:")
        print("input_dir: 图片所在目录")
        print("width: 目标宽度（像素）")
        print("height: 目标高度（像素）")
        print("--offset_x: 中心点水平偏移（正数向右）")
        print("--offset_y: 中心点垂直偏移（正数向下）")
    else:
        main()