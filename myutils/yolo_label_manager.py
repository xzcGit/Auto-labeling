"""
YOLO标签管理工具
功能：
1. 读取指定路径下的所有YOLO格式标签文件（.txt）
2. 统计所有类别及其数量
3. 可选择性地修改指定类别标签
"""

import os
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple


class YOLOLabelManager:
    """YOLO标签管理器"""
    
    def __init__(self, label_path: Path):
        """
        初始化标签管理器
        
        Args:
            label_path: 标签文件所在目录路径
        """
        self.label_path = label_path
        self.label_files = []
        self.category_stats = Counter()
        self.total_objects = 0
        
    def scan_labels(self) -> None:
        """扫描目录下的所有txt标签文件"""
        if not self.label_path.exists():
            raise FileNotFoundError(f"路径不存在: {self.label_path}")
        
        if not self.label_path.is_dir():
            raise NotADirectoryError(f"不是有效的目录: {self.label_path}")
        
        # 获取所有.txt文件
        self.label_files = list(self.label_path.glob("*.txt"))
        print(f"找到 {len(self.label_files)} 个标签文件")
        
    def analyze_categories(self) -> Dict[int, int]:
        """
        分析所有标签文件中的类别
        
        Returns:
            类别统计字典 {类别ID: 出现次数}
        """
        self.category_stats.clear()
        self.total_objects = 0
        
        for label_file in self.label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line in lines:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:  # YOLO格式: class_id x_center y_center width height
                        try:
                            class_id = int(parts[0])
                            self.category_stats[class_id] += 1
                            self.total_objects += 1
                        except ValueError:
                            print(f"警告: 文件 {label_file.name} 中存在无效的类别ID: {parts[0]}")
                            
            except Exception as e:
                print(f"读取文件 {label_file.name} 时出错: {e}")
        
        return dict(self.category_stats)
    
    def print_statistics(self) -> None:
        """打印类别统计信息"""
        print("\n" + "="*60)
        print("类别统计结果")
        print("="*60)
        print(f"总标签文件数: {len(self.label_files)}")
        print(f"总对象数: {self.total_objects}")
        print(f"类别总数: {len(self.category_stats)}")
        print("\n类别详情:")
        print(f"{'类别ID':<10} {'数量':<10} {'占比':<10}")
        print("-"*60)
        
        # 按类别ID排序
        for class_id in sorted(self.category_stats.keys()):
            count = self.category_stats[class_id]
            percentage = (count / self.total_objects * 100) if self.total_objects > 0 else 0
            print(f"{class_id:<10} {count:<10} {percentage:.2f}%")
        
        print("="*60 + "\n")
    
    def modify_labels(self, mapping: Dict[int, int], backup: bool = True) -> Tuple[int, int]:
        """
        修改标签文件中的类别ID
        
        Args:
            mapping: 类别映射字典 {原类别ID: 新类别ID}
            backup: 是否备份原文件
            
        Returns:
            (修改的文件数, 修改的对象数)
        """
        modified_files = 0
        modified_objects = 0
        
        # 创建备份目录
        if backup:
            backup_dir = self.label_path / "backup"
            backup_dir.mkdir(exist_ok=True)
            print(f"备份目录: {backup_dir}")
        
        for label_file in self.label_files:
            try:
                # 读取原文件
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 备份原文件
                if backup:
                    backup_file = backup_dir / label_file.name
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                
                # 修改标签
                new_lines = []
                file_modified = False
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        new_lines.append('\n')
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            
                            # 如果类别在映射中，则修改
                            if class_id in mapping:
                                parts[0] = str(mapping[class_id])
                                file_modified = True
                                modified_objects += 1
                            
                            new_lines.append(' '.join(parts) + '\n')
                        except ValueError:
                            new_lines.append(line + '\n')
                    else:
                        new_lines.append(line + '\n')
                
                # 写回文件
                if file_modified:
                    with open(label_file, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    modified_files += 1
                    
            except Exception as e:
                print(f"处理文件 {label_file.name} 时出错: {e}")
        
        return modified_files, modified_objects


def main(label_path: str = None) -> None:
    """主函数"""
    print("="*60)
    print("YOLO标签管理工具")
    print("="*60)
    
    # 检查路径是否存在
    label_path = Path(label_path)
    if not label_path.exists() or not label_path.is_dir():
        print(f"错误: 路径 {label_path} 不存在或不是一个目录")
        return

    
    # 创建管理器
    manager = YOLOLabelManager(label_path)
    
    # 2. 扫描标签文件
    print("\n正在扫描标签文件...")
    manager.scan_labels()
    
    if len(manager.label_files) == 0:
        print("未找到任何标签文件！")
        return
    
    # 3. 分析类别
    print("\n正在分析类别...")
    manager.analyze_categories()
    
    # 4. 显示统计信息
    manager.print_statistics()
    
    # 5. 询问是否需要修改标签
    modify_flag = input("是否需要修改标签？(y/n): ").strip().lower()
    
    if modify_flag == 'y':
        print("\n" + "="*60)
        print("标签修改模式")
        print("="*60)
        
        # 输入映射关系
        mapping = {}
        print("\n请输入类别映射关系（输入'done'结束）:")
        print("格式: 原类别ID 新类别ID")
        print("示例: 0 1  (将类别0修改为类别1)")
        print("支持多对一映射，例如: 0->2, 1->2 (将类别0和1都修改为类别2)")
        
        while True:
            user_input = input("\n映射关系: ").strip()
            
            if user_input.lower() == 'done':
                break
            
            try:
                parts = user_input.split()
                if len(parts) == 2:
                    old_id = int(parts[0])
                    new_id = int(parts[1])
                    mapping[old_id] = new_id
                    print(f"已添加映射: {old_id} -> {new_id}")
                else:
                    print("格式错误！请输入两个整数，用空格分隔")
            except ValueError:
                print("输入错误！请输入有效的整数")
        
        if not mapping:
            print("\n未设置任何映射关系，退出修改模式")
            return
        
        # 确认映射
        print("\n" + "-"*60)
        print("映射关系汇总:")
        for old_id, new_id in mapping.items():
            count = manager.category_stats.get(old_id, 0)
            print(f"  类别 {old_id} -> {new_id} (影响 {count} 个对象)")
        print("-"*60)
        
        confirm = input("\n确认执行修改？(y/n): ").strip().lower()
        
        if confirm == 'y':
            # 询问是否备份
            backup_flag = input("是否备份原文件？(y/n，默认y): ").strip().lower()
            backup = backup_flag != 'n'
            
            print("\n正在修改标签...")
            modified_files, modified_objects = manager.modify_labels(mapping, backup)
            
            print("\n" + "="*60)
            print("修改完成！")
            print("="*60)
            print(f"修改的文件数: {modified_files}")
            print(f"修改的对象数: {modified_objects}")
            
            if backup:
                print(f"原文件已备份到: {manager.label_path / 'backup'}")
            
            # 重新分析并显示结果
            print("\n重新分析修改后的标签...")
            manager.analyze_categories()
            manager.print_statistics()
        else:
            print("\n已取消修改操作")
    else:
        print("\n仅输出统计结果，未进行修改")
    
    print("\n程序结束")


if __name__ == "__main__":
    try:
        main(label_path=r'F:\code\changzhan\万胜永\data\switch\det\labels')
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()
