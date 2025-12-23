import shutil
from pathlib import Path

# ==================== 配置区域 ====================
# 在这里直接指定路径
IMAGE_DIR = r"F:\code\utils\19-metertools\万胜永\pointer_rect_oiltemper-X\images"      # 图像文件夹路径
LABEL_DIR = r"F:\code\utils\19-metertools\万胜永\pointer\labels"      # 标签文件夹路径
OUTPUT_DIR = r"F:\code\utils\19-metertools\万胜永\pointer_rect_oiltemper-X\labels"     # 输出文件夹路径

# 支持的文件扩展名
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']
LABEL_EXTENSIONS = ['.txt']
# ================================================


def copy_matched_labels():
    """将与图像匹配的标签复制到输出目录"""

    img_path = Path(IMAGE_DIR)
    lbl_path = Path(LABEL_DIR)
    out_path = Path(OUTPUT_DIR)

    if not img_path.exists():
        print(f"错误: 图像目录不存在: {img_path}")
        return

    if not lbl_path.exists():
        print(f"错误: 标签目录不存在: {lbl_path}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    print(f"图像目录: {img_path}")
    print(f"标签目录: {lbl_path}")
    print(f"输出目录: {out_path}\n")

    # 获取图像文件名（不含扩展名）
    image_stems = set()
    for file in img_path.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_stems.add(file.stem)

    # 获取标签文件
    label_files = {}
    for file in lbl_path.iterdir():
        if file.is_file() and file.suffix.lower() in LABEL_EXTENSIONS:
            label_files[file.stem] = file

    # 找到匹配的文件
    matched_stems = image_stems & set(label_files.keys())

    print(f"图像文件: {len(image_stems)} 个")
    print(f"标签文件: {len(label_files)} 个")
    print(f"匹配文件: {len(matched_stems)} 个\n")

    if not matched_stems:
        print("没有找到匹配的文件")
        return

    # 复制匹配的标签文件
    copied = 0
    for stem in sorted(matched_stems):
        src = label_files[stem]
        dst = out_path / src.name
        try:
            shutil.copy2(src, dst)
            print(f"✓ {src.name}")
            copied += 1
        except Exception as e:
            print(f"✗ {src.name} - {e}")

    print(f"\n完成: 复制了 {copied} 个标签文件")


if __name__ == '__main__':
    copy_matched_labels()
