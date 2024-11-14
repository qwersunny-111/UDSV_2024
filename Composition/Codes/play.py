import cv2
import os

# 图片文件夹路径
folder_path = '/home/sunleyao/sly/UDIS2-main/UDIS2++-control points/Composition/composition'

# 获取文件夹中的所有文件名并按编号排序
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png', 'bmp'))],
                     key=lambda x: int(os.path.splitext(x)[0]))

# 创建一个窗口
cv2.namedWindow('Image Sequence', cv2.WINDOW_NORMAL)

# 逐一读取和显示图片
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image {image_path}")
        continue

    cv2.imshow('Image Sequence', image)

    # 等待一定时间（毫秒），这里设置为1000毫秒，即1秒
    key = cv2.waitKey(1000)
    
    # 如果按下ESC键，退出循环
    if key == 27:
        break

# 释放窗口
cv2.destroyAllWindows()
