import os

# 设置工作目录为包含图像文件的文件夹路径
working_dir = '/home/lianghao/workspace/TCSVT/experiments/results/sequences/case1_4l_1_Bundled/'

# 获取文件夹中所有文件的列表
files = os.listdir(working_dir)

# 遍历文件夹中的每个文件
for filename in files:
    # 检查文件名是否以.jpg结尾
    if filename.endswith('.jpg'):
        # 提取文件名中的数字部分
        file_number = int(filename.split('.')[0])-1
        
        # 生成新的文件名，格式为六位数字，包含前导零
        new_filename = f'{file_number:06d}.jpg'
        
        # 构建原始文件和新文件的完整路径
        original_path = os.path.join(working_dir, filename)
        new_path = os.path.join(working_dir, new_filename)
        
        # 重命名文件
        os.rename(original_path, new_path)

print("文件重命名完成。")
