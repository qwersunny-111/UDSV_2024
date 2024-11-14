import cv2
import numpy as np

def harris_corner_detection(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算 Harris 角点
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 只保留角点，将其阈值化
    image[dst > 0.01 * dst.max()] = [0, 255, 0]

    # 绘制绿色空心圆圈标注角点
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    for point in keypoints:
        cv2.circle(image, tuple(point[::-1]), 3, (0, 255, 0), 1)  # 绘制绿色空心圆圈

    # 保存结果图像
    cv2.imwrite(output_path, image)

    print(f"已将带有 Harris 角点的图像保存到: {output_path}")

# 测试函数
image_path = '/home/B_UserData/sunleyao/UDIS2/testing/20240427/input2/56.jpg'
output_path = '/home/sunleyao/sly/UDIS2-main/UDIS2-main-y-5/Warp/image_harris.jpg'
harris_corner_detection(image_path, output_path)
