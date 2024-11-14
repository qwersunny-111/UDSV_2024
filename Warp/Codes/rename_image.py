import os
import re

def extract_number(filename):
    # 使用正则表达式提取文件名中的数字
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return 0

def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    # 按照文件名中的数字排序
    files.sort(key=extract_number)
    
    # 遍历所有文件并重命名
    for i, filename in enumerate(files):
        # 获取文件扩展名
        file_extension = os.path.splitext(filename)[1]
        
        # 新文件名，编号从1开始，6位，不足补0
        new_name = f"{str(i+1).zfill(6)}{file_extension}"
        
        # 获取旧文件和新文件的完整路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f"Renamed {old_file} to {new_file}")

# 调用函数，替换 'your_folder_path' 为你的文件夹路径
rename_images('/home/B_UserData/sunleyao/UDIS2/video20240918/1_9/remain_input2')
