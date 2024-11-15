import cv2
import numpy as np
import tqdm

W= 960
H= 540
# frame_index = 125

# frontview = cv2.imread('real_09/final_3/front/seamcut/' + '{:0{}d}'.format(frame_index,3) + '.jpg')
# rearview = cv2.imread('real_09/final_3/rear/seamcut/' + '{:0{}d}'.format(frame_index,3) + '.jpg')
# left = cv2.imread('real_09/final_3/125_2_1_seam.jpg')
# right = cv2.imread('real_09/final_3/125_7_6_seam.jpg')

# ll = rearview[:, int(rearview.shape[1]/2):-W//2, :]
# l = left[:, W//2:-W//2, :]
# mid = frontview[:, W//2:-W//2+120, :]
# r = right[:, W//2+105:-W//2+15, :]
# rr = rearview[:, W//2+10:int(rearview.shape[1]/2), :]

# img = np.concatenate((ll, l, mid, r, rr), axis=1)

# frontview_band = cv2.imread('real_09/final_3/front/multiband/' + '{:0{}d}'.format(frame_index,3) + '.jpg')
# rearview_band = cv2.imread('real_09/final_3/rear/multiband/' + '{:0{}d}'.format(frame_index,3) + '.jpg')
# left_band = cv2.imread('real_09/final_3/125_2_1_multiband.jpg')
# right_band = cv2.imread('real_09/final_3/125_7_6_multiband.jpg')

# ll_band = rearview_band[:, int(rearview_band.shape[1]/2):-W//2+5, :]
# l_band = left_band[:, W//2:-W//2, :]
# mid_band = frontview_band[:, W//2:-W//2+10, :]
# r_band = right_band[:, W//2:-W//2+10, :]
# rr_band = rearview_band[:, W//2+8:int(rearview_band.shape[1]/2), :]

# img_band = np.concatenate((ll_band, l_band, mid_band, r_band, rr_band), axis=1)

# # cv2.imshow('final', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite('real_09/final_3/res_seam.jpg', img)
# cv2.imwrite('real_09/final_3/res_multiband.jpg', img_band)

start = 15
img_count = 300

front_img_path = 'real_09/final_5/front/multiband'
mid_img_path = 'real_09/final_5/pano/rear_left/band1'
rear_img_path = 'real_09/final_5/rear/multiband'
output_path = 'real_09/final_5/pano/rear_left/res1'

front_frames = []
mid_frames = []
rear_frames = []


for i in tqdm.trange(start, start + img_count):
    frame1 = cv2.imread(front_img_path + '/' + '{:0{}d}'.format(i,3) + '.jpg')
    front_frames.append(frame1)

    frame2 = cv2.imread(mid_img_path + '/' + '{:0{}d}'.format(i,3) + '.jpg')
    mid_frames.append(frame2)

    frame3 = cv2.imread(rear_img_path + '/' + '{:0{}d}'.format(i,3) + '.jpg')
    rear_frames.append(frame3)

print('Read Done!')

W_p = front_frames[0].shape[1] + mid_frames[0].shape[1] + rear_frames[0].shape[1]
H_p = front_frames[0].shape[0]

#3 600
#6 530
W_p = W_p  - 2 * W - 540

res = []

for i in tqdm.trange(img_count):
    # front = front_frames[i][:, :-W, :]
    # w1 = front.shape[1]
    # mid = mid_frames[i]
    # w2 = mid.shape[1]
    # rear = rear_frames[i][:,W:W_p - w1 - w2 + W, :]
    m = mid_frames[i]
    w2 = m.shape[1]
    r = front_frames[i][:,W:, :]
    w1 = r.shape[1]
    l = rear_frames[i][:, - W_p + w1 + w2 - W:-W, :]
    img = np.concatenate((l, m, r), axis=1)
    res.append(img)
    cv2.imwrite(output_path + '/' + '{:0{}d}'.format(i+start,3) + '.jpg', img)

print('Write Done!')

# 写视频
video_out = 'real_09/final_5/pano/rear_left/res1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20
frame_height, frame_width, _ = res[0].shape

video = cv2.VideoWriter(video_out, fourcc, fps, (frame_width, frame_height))
for i in tqdm.trange(img_count):
    video.write(res[i])
video.release()
