import os

# 设置文件夹路径，替换为实际的文件夹路径
folder_path = '../datafile/lung2_center_crop/labelsTr'

# 获取文件夹中所有 .pt 文件
file_list = [f for f in os.listdir(folder_path) if f.endswith('.pt')]

# 批量重命名文件
for idx, file_name in enumerate(file_list):
    # 生成新的文件名 (例如：file_0.pt, file_1.pt, ...)
    new_name = f"Dataset2_masks_{idx}.pt"
    
    # 获取文件的完整路径
    old_file_path = os.path.join(folder_path, file_name)
    new_file_path = os.path.join(folder_path, new_name)
    
    # 重命名文件
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {file_name} -> {new_name}")

print("Batch renaming complete!")
