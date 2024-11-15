import numpy as np
import cv2
import math
import tqdm
import argparse

def cylinder(img, w, h):
    width = img.shape[1]
    height = img.shape[0]
    center_x = width/2
    center_y = height/2
    alpha = math.pi/3
    f = width/(2*math.tan(alpha/2))
    # print(f)
    #half_cyc = math.atan(width/(2*f))
    #print(half_cyc)

    img1 = np.zeros((height, width, 3), np.uint8)
    new_width = []

    for h in range(0, height):
        for w in range(0, width):
            theta = math.atan((w-center_x)/f)
            dist_x = np.int32(f * (alpha/2 + theta))
            dist_y = np.int32(f * (h - center_y)/math.sqrt((w-center_x)**2 + f**2) + center_y)
            if dist_x >= 0 and dist_x < width and dist_y >= 0 and dist_y < height:
                img1[dist_y,dist_x] = img[h,w]
                new_width.append(dist_x)
            else:
                print('err: spill border')
                exit()
    res_width = max(new_width)
    # 50是为了去除黑边 需根据不同视频调整
    res_img = img1[30:-30, 0:res_width, :]
    res = cv2.resize(res_img, (w,h), interpolation=cv2.INTER_CUBIC)
    return res

# for i in range(H):
#    if img1[i, 0, 0] != 0:
#       print(i)
#       break

if __name__ == '__main__':
    W = 720
    H = 480
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-l", "--left", type=str, default="real_09/calib_0_8/09_0_8_4_100.mp4", help="path to the left video")
    # args = vars(ap.parse_args())

    # vs = cv2.VideoCapture(args["left"])
    # num_frames = np.int32(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    # with tqdm.trange(num_frames) as t:
    #     t.set_description(f'Reading video from <{args["left"]}>')
    #     left_frames = []
    #     for frame_index in t:
    #         success, pixels = vs.read()
    #         if success:
    #             # 规范分辨率
    #             unstabilized_frame = cv2.resize(pixels, (W, H))
    #         else:
    #             print('capture error')
    #             exit()
    #         if unstabilized_frame is None:
    #             raise IOError(
    #                 f'Video at <{args["left"]}> did not have frame {frame_index} of '
    #                 f'{num_frames} (indexed from 0).'
    #             )
    #         left_frames.append(unstabilized_frame)
    # vs.release()

    # frame_idx = 100

    # img = left_frames[frame_idx]
    # img = cv2.imread('real_09/ex_2_0/video0/000100.jpg')
    # res_img = cylinder(img, W, H)


    # cv2.imshow('img', img)
    # cv2.imshow('res_img', res_img)
    # cv2.waitKey(0)
    input_path = 'real_09/final_7/camera5_1_calib_300_cut_3.mp4'

    vs = cv2.VideoCapture(input_path)
    num_frames = np.int32(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm.trange(num_frames) as t:
        t.set_description(f'Reading video from <{input_path}>')
        frames = []
        for frame_index in t:
            success, pixels = vs.read()
            if success:
                # 规范分辨率
                unstabilized_frame = cv2.resize(pixels, (W, H))
            else:
                print('capture error')
                exit()
            if unstabilized_frame is None:
                raise IOError(
                    f'Video at <{input_path}> did not have frame {frame_index} of '
                    f'{num_frames} (indexed from 0).'
                )
            frames.append(unstabilized_frame)
    
    res_imgs = []

    for i in tqdm.trange(num_frames):
        img = frames[i]
        res_img = cylinder(img, W, H)
        res_imgs.append(res_img)
        cv2.imwrite('real_09/final_7/video5/' + '{:0{}d}'.format(i,3) + '.jpg', res_img)

    # 写视频
    video_out = 'real_09/final_7/case7_5.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20
    frame_height, frame_width, _ = res_imgs[0].shape

    video = cv2.VideoWriter(video_out, fourcc, fps, (frame_width, frame_height))
    for i in tqdm.trange(num_frames):
        video.write(res_imgs[i])
    video.release()
