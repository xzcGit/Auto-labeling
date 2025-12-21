# 文件夹构建
import os
import shutil

if __name__ == '__main__':
    # 获取类别文件夹
    parent_dir = os.path.join(os.getcwd(), 'data', 'raw')
    classes = os.listdir(parent_dir)
    # 创建类别文件夹
    for cls in classes:
        cls_dir = os.path.join(parent_dir, cls)
        if os.path.isdir(cls_dir):
            # 创建类别文件夹
            new_dir_name = cls.replace('_unlabeled', '')
            os.makedirs(os.path.join(os.getcwd(), 'data', 'raw', new_dir_name), exist_ok=True)
            os.makedirs(os.path.join(os.getcwd(), 'data', 'raw', new_dir_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(os.getcwd(), 'data', 'raw', new_dir_name, 'labels'), exist_ok=True)

        