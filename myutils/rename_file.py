import os

# 文件夹路径
parent_dir = os.path.join(os.getcwd() , 'data', 'raw')
# print(parent_dir)
# 重命名parent_dir下的所有文件夹
# for dir_name in os.listdir(parent_dir):
#     dir_path = os.path.join(parent_dir, dir_name)
#     if os.path.isdir(dir_path):
#         # 在文件夹后面添加"_unlabeled"
#         new_dir_name = dir_name + '_unlabeled'
#         new_dir_path = os.path.join(parent_dir, new_dir_name)
#         os.rename(dir_path, new_dir_path)
        

# 
for dir_name in os.listdir(parent_dir):
    dir_path = os.path.join(parent_dir, dir_name)
    if os.path.isdir(dir_path):
        # 去除文件夹后的"_unlabeled"并创建新的文件夹
        new_dir_name = dir_name.replace('_unlabeled', '')
        new_dir_path = os.path.join(parent_dir, new_dir_name)
        # 添加新的文件夹
        os.mkdir(new_dir_path)

