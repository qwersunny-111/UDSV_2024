import os
import numpy as np
import cv2

img_path = "/home/lianghao/workspace/TCSVT/experiments/data/sequences/case1_4l_1/000000.jpg"
# K = np.array([[794.7633, 0.0, 635.6471], [0.0, 794.8898, 369.4420], [0.0, 0.0, 1.0]]).astype(np.float32)
K = np.array([[684.2892, 0.0, 668.6719], [0.0, 684.1945, 358.0719], [0.0, 0.0, 1.0]]).astype(np.float32)
R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).astype(np.float32)
img = cv2.imread(img_path)
mask = cv2.UMat(255*np.ones((img.shape[0],img.shape[1]), np.uint8))
print(img.shape)
cv2.imwrite("image.jpg", img)

warper = cv2.PyRotationWarper("cylindrical", 720)
# corner, image_wp =warper.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
# print(corner, image_wp.shape)
# cv2.imwrite("image_wp.jpg", image_wp)
# p, mask_wp =warper.warp(mask, K, R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
# print(p, mask_wp.get().shape)
# cv2.imwrite("mask_wp.jpg", mask_wp.get())

src_size = (img.shape[1], img.shape[0])
retval, xmap, ymap = warper.buildMaps(src_size, K, R)
img_remap = cv2.remap(img, xmap, ymap, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
print(retval, img_remap.shape)
cv2.imwrite("img_remap.jpg", img_remap)