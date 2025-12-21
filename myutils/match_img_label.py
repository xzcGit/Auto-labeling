import os
from pathlib import Path
from typing import List, Tuple, Set
import argparse


def get_file_stems(directory: Path, extensions: List[str]) -> Set[str]:
    """
    Get file stems (filename without extension) from a directory.
    
    Args:
        directory: Path to the directory
        extensions: List of valid file extensions (e.g., ['.jpg', '.png'])
    
    Returns:
        Set of file stems
    """
    file_stems = set()
    if not directory.exists():
        return file_stems
    
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in extensions:
            file_stems.add(file.stem)
    
    return file_stems


def check_image_label_correspondence(
    root_path: str,
    image_extensions: List[str] = None,
    label_extensions: List[str] = None,
    max_display: int = 10
) -> Tuple[List[str], List[str]]:
    """
    Check correspondence between images and labels.
    
    Args:
        root_path: Root directory containing 'images' and 'labels' folders
        image_extensions: List of image file extensions (default: ['.jpg', '.jpeg', '.png', '.bmp'])
        label_extensions: List of label file extensions (default: ['.txt', '.xml', '.json'])
        max_display: Maximum number of mismatched files to display (default: 10)
    
    Returns:
        Tuple of (images_without_labels, labels_without_images)
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    if label_extensions is None:
        label_extensions = ['.txt', '.xml', '.json']
    
    root = Path(root_path)
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    
    # Check if directories exist
    if not images_dir.exists():
        print(f"错误: 图像目录不存在: {images_dir}")
        return [], []
    
    if not labels_dir.exists():
        print(f"错误: 标签目录不存在: {labels_dir}")
        return [], []
    
    # Get file stems
    print(f"正在扫描目录...")
    image_stems = get_file_stems(images_dir, image_extensions)
    label_stems = get_file_stems(labels_dir, label_extensions)
    
    # Find mismatches
    images_without_labels = sorted(image_stems - label_stems)
    labels_without_images = sorted(label_stems - image_stems)
    
    # Display statistics
    print(f"\n{'='*60}")
    print(f"检查结果统计:")
    print(f"{'='*60}")
    print(f"图像文件总数: {len(image_stems)}")
    print(f"标签文件总数: {len(label_stems)}")
    print(f"匹配的文件数: {len(image_stems & label_stems)}")
    print(f"缺少标签的图像数: {len(images_without_labels)}")
    print(f"缺少图像的标签数: {len(labels_without_images)}")
    print(f"{'='*60}\n")
    
    # Display images without labels
    if images_without_labels:
        print(f"❌ 缺少标签的图像 (共 {len(images_without_labels)} 个):")
        display_count = min(max_display, len(images_without_labels))
        for i, stem in enumerate(images_without_labels[:display_count], 1):
            # Find the actual file with extension
            actual_file = None
            for ext in image_extensions:
                if (images_dir / f"{stem}{ext}").exists():
                    actual_file = f"{stem}{ext}"
                    break
            print(f"  {i}. {actual_file or stem}")
        
        if len(images_without_labels) > max_display:
            print(f"  ... 还有 {len(images_without_labels) - max_display} 个文件未显示")
        print()
    else:
        print("✅ 所有图像都有对应的标签\n")
    
    # Display labels without images
    if labels_without_images:
        print(f"❌ 缺少图像的标签 (共 {len(labels_without_images)} 个):")
        display_count = min(max_display, len(labels_without_images))
        for i, stem in enumerate(labels_without_images[:display_count], 1):
            # Find the actual file with extension
            actual_file = None
            for ext in label_extensions:
                if (labels_dir / f"{stem}{ext}").exists():
                    actual_file = f"{stem}{ext}"
                    break
            print(f"  {i}. {actual_file or stem}")
        
        if len(labels_without_images) > max_display:
            print(f"  ... 还有 {len(labels_without_images) - max_display} 个文件未显示")
        print()
    else:
        print("✅ 所有标签都有对应的图像\n")
    
    return images_without_labels, labels_without_images


def save_mismatch_report(
    root_path: str,
    images_without_labels: List[str],
    labels_without_images: List[str],
    output_file: str = "mismatch_report.txt"
):
    """
    Save mismatch report to a file.
    
    Args:
        root_path: Root directory path
        images_without_labels: List of image stems without labels
        labels_without_images: List of label stems without images
        output_file: Output file name
    """
    output_path = Path(root_path) / output_file
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("图像与标签匹配检查报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"缺少标签的图像 (共 {len(images_without_labels)} 个):\n")
        f.write("-" * 60 + "\n")
        for stem in images_without_labels:
            f.write(f"{stem}\n")
        f.write("\n")
        
        f.write(f"缺少图像的标签 (共 {len(labels_without_images)} 个):\n")
        f.write("-" * 60 + "\n")
        for stem in labels_without_images:
            f.write(f"{stem}\n")
    
    print(f"📝 完整报告已保存至: {output_path}")


def delete_images_without_labels(
    root_path: str,
    images_without_labels: List[str],
    image_extensions: List[str] = None,
    dry_run: bool = True
) -> int:
    """
    Delete images that don't have corresponding labels.
    
    Args:
        root_path: Root directory containing 'images' folder
        images_without_labels: List of image stems without labels
        image_extensions: List of image file extensions
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        Number of files deleted (or would be deleted if dry_run=True)
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    if not images_without_labels:
        print("✅ 没有需要删除的图像")
        return 0
    
    root = Path(root_path)
    images_dir = root / 'images'
    
    deleted_count = 0
    failed_deletions = []
    
    print(f"\n{'='*60}")
    if dry_run:
        print(f"预览模式: 以下文件将被删除 (共 {len(images_without_labels)} 个)")
    else:
        print(f"正在删除缺少标签的图像 (共 {len(images_without_labels)} 个)")
    print(f"{'='*60}\n")
    
    for stem in images_without_labels:
        # Find the actual file with extension
        file_to_delete = None
        for ext in image_extensions:
            file_path = images_dir / f"{stem}{ext}"
            if file_path.exists():
                file_to_delete = file_path
                break
        
        if file_to_delete:
            if dry_run:
                print(f"  [预览] {file_to_delete.name}")
                deleted_count += 1
            else:
                try:
                    file_to_delete.unlink()
                    print(f"  ✓ 已删除: {file_to_delete.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ 删除失败: {file_to_delete.name} - {e}")
                    failed_deletions.append(file_to_delete.name)
    
    print(f"\n{'='*60}")
    if dry_run:
        print(f"预览完成: 共 {deleted_count} 个文件将被删除")
        print(f"提示: 使用 --delete 参数执行实际删除操作")
    else:
        print(f"删除完成: 成功删除 {deleted_count} 个文件")
        if failed_deletions:
            print(f"失败: {len(failed_deletions)} 个文件删除失败")
    print(f"{'='*60}\n")
    
    return deleted_count


def main():
    parser = argparse.ArgumentParser(
        description='检查图像和标签文件的对应关系',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python match_img_label.py /path/to/dataset
  python match_img_label.py /path/to/dataset --max-display 20
  python match_img_label.py /path/to/dataset --save-report
  python match_img_label.py /path/to/dataset --image-ext .jpg .png --label-ext .txt
  python match_img_label.py /path/to/dataset --delete-unmatched  # 预览删除
  python match_img_label.py /path/to/dataset --delete-unmatched --delete  # 执行删除
        """
    )
    
    parser.add_argument('root_path', type=str, help='数据集根目录路径 (包含 images 和 labels 文件夹)')
    parser.add_argument('--max-display', type=int, default=10, help='显示的最大不匹配文件数 (默认: 10)')
    parser.add_argument('--image-ext', nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'], help='图像文件扩展名列表 (默认: .jpg .jpeg .png .bmp .tif .tiff)')
    parser.add_argument('--label-ext', nargs='+', default=['.txt', '.xml', '.json'], help='标签文件扩展名列表 (默认: .txt .xml .json)')
    parser.add_argument('--save-report', action='store_true', help='保存完整的不匹配报告到文件')
    parser.add_argument('--report-name', type=str, default='mismatch_report.txt', help='报告文件名 (默认: mismatch_report.txt)')
    parser.add_argument('--delete-unmatched', action='store_true', help='删除缺少标签的图像文件')
    parser.add_argument('--delete', action='store_true', help='确认执行删除操作 (配合 --delete-unmatched 使用)')
    
    args = parser.parse_args()
    
    # Check correspondence
    images_without_labels, labels_without_images = check_image_label_correspondence(
        root_path=args.root_path,
        image_extensions=args.image_ext,
        label_extensions=args.label_ext,
        max_display=args.max_display
    )
    
    # Save report if requested
    if args.save_report:
        save_mismatch_report(
            root_path=args.root_path,
            images_without_labels=images_without_labels,
            labels_without_images=labels_without_images,
            output_file=args.report_name
        )
    
    # Delete images without labels if requested
    if args.delete_unmatched and images_without_labels:
        dry_run = not args.delete
        delete_images_without_labels(
            root_path=args.root_path,
            images_without_labels=images_without_labels,
            image_extensions=args.image_ext,
            dry_run=dry_run
        )
    
    # Exit with appropriate code
    if images_without_labels or labels_without_images:
        print("⚠️  发现不匹配的文件")
        return 1
    else:
        print("✅ 所有文件都匹配成功!")
        return 0


if __name__ == '__main__':
    exit(main())
