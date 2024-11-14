import cv2

def resize_image(reference_image_path, target_image_path, output_image_path):
    # 读取参考图像
    reference_image = cv2.imread(reference_image_path)

    # 读取目标图像
    target_image = cv2.imread(target_image_path)

    # 调整目标图像的大小和分辨率
    resized_target_image = cv2.resize(target_image, (reference_image.shape[1], reference_image.shape[0]))
    print(reference_image.shape)
    # 保存结果
    cv2.imwrite(output_image_path, resized_target_image)

resize_image("../../Carpark-DHW-0/input1.jpg", "../../Carpark-DHW-1/input1.jpg", "../../Carpark-DHW/input1.jpg")
resize_image("../../Carpark-DHW-0/input2.jpg", "../../Carpark-DHW-1/input2.jpg", "../../Carpark-DHW/input2.jpg")
