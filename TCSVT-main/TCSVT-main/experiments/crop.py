# from PIL import Image

# # 加载图片
# image_path = '/home/lianghao/workspace/TCSVT/experiments/results/sequences/case1_4l_1_Bundled/000000.jpg'
# image = Image.open(image_path)

# # 将图片转换为RGBA，如果它不是
# image = image.convert("RGBA")

# # 获取图片的数据
# datas = image.getdata()

# # 新建一个列表，如果像素不是黑色，则添加到列表中
# new_data = []
# for item in datas:
#     # 判断黑色，可以根据需要调整颜色的阈值
#     if item[0] > 20 or item[1] > 20 or item[2] > 20:
#         new_data.append(item)
#     else:
#         # 如果是黑色，则用透明替换
#         new_data.append((255, 255, 255, 0))

# # 更新图片数据
# image.putdata(new_data)

# # 裁剪图片
# # 获取图片边界
# bbox = image.getbbox()
# bbox = (
#     bbox[0] + 5,  # left 增加5个像素
#     bbox[1] + 5,  # top 增加5个像素
#     bbox[2] - 5,  # right 减少5个像素
#     bbox[3] - 5   # bottom 减少5个像素
# )
# print(bbox)
# # 裁剪图片
# image = Image.open(image_path)
# image = image.crop(bbox)
# output_path = '/home/lianghao/workspace/TCSVT/experiments/results/sequences/case1_4l_1_Bundled/000000.png'
# image.save(output_path)

# print(f"The image has been cropped and saved to {output_path}")

from PIL import Image
import glob

# 定义文件夹路径和文件模式
folder_path = '/home/lianghao/workspace/TCSVT/experiments/results/sequences/case1_4r_1_Bundled/'
file_pattern = "*.jpg"  # 匹配所有.jpg文件

# 使用glob模块找到所有匹配的文件
image_files = glob.glob(folder_path + file_pattern)

# 循环处理每个文件
for image_path in image_files:
    # 加载图片
    image = Image.open(image_path)

    # 将图片转换为RGBA格式
    image = image.convert("RGBA")

    # 获取图片的数据
    datas = image.getdata()

    # 新建一个列表，如果像素不是黑色，则添加到列表中
    new_data = []
    for item in datas:
        if item[0] > 20 or item[1] > 20 or item[2] > 20:
            new_data.append(item)
        else:
            # 如果是黑色，则用透明替换
            new_data.append((255, 255, 255, 0))

    # 更新图片数据
    image.putdata(new_data)

    # 获取图片边界并调整
    bbox = image.getbbox()
    bbox = (
        bbox[0] + 30,  # left 增加5个像素
        bbox[1] + 30,  # top 增加5个像素
        bbox[2] - 30,  # right 减少5个像素
        bbox[3] - 30   # bottom 减少5个像素
    )
    
    # 裁剪图片
    image = Image.open(image_path)
    image = image.crop(bbox)

    # 定义输出路径，并保存修改后的图片
    # output_path = image_path.replace('.jpg', '.png')  # 将.jpg替换为.png
    image.save(image_path)

    print(f"The image {image_path} has been cropped and saved to {image_path}")

