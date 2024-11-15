import numpy as np
import cv2
import sys
import argparse
import tqdm


def preprocess(img1, img2, overlap_w, flag_half, need_mask=False):
    if img1.shape[0] != img2.shape[0]:
        print ("error: image dimension error")
        sys.exit()
    if overlap_w > img1.shape[1] or overlap_w > img2.shape[1]:
        print ("error: overlapped area too large")
        sys.exit()

    w1 = img1.shape[1]
    w2 = img2.shape[1]

    if flag_half:
        shape = np.array(img1.shape)
        shape[1] = w1 // 2 + w2 // 2

        subA = np.zeros(shape)
        subA[:, :w1 // 2 + overlap_w // 2] = img1[:, :w1 // 2 + overlap_w // 2]
        subB = np.zeros(shape)
        subB[:, w1 // 2 - overlap_w // 2:] = img2[:,
                                                w2 - (w2 // 2 + overlap_w // 2):]
        if need_mask:
            mask = np.zeros(shape)
            mask[:, :w1 // 2] = 1
            return subA, subB, mask
    else:
        shape = np.array(img1.shape)
        shape[1] = w1 + w2 - overlap_w

        subA = np.zeros(shape)
        subA[:, :w1] = img1
        subB = np.zeros(shape)
        subB[:, w1 - overlap_w:] = img2
        if need_mask:
            mask = np.zeros(shape)
            mask[:, :w1 - overlap_w // 2] = 1
            return subA, subB, mask

    return subA, subB, None


def GaussianPyramid(img, leveln):
    GP = [img]
    for i in range(leveln - 1):
        GP.append(cv2.pyrDown(GP[i]))
    return GP


def LaplacianPyramid(img, leveln):
    LP = []
    for i in range(leveln - 1):
        next_img = cv2.pyrDown(img)
        LP.append(img - cv2.resize(cv2.pyrUp(next_img, img.shape[1::-1]),img.shape[1::-1]))
        img = next_img
    LP.append(img)
    return LP


def blend_pyramid(LPA, LPB, MP):
    blended = []
    for i, M in enumerate(MP):
        blended.append(LPA[i] * M + LPB[i] * (1.0 - M))
    return blended


def reconstruct(LS):
    img = LS[-1]
    for lev_img in LS[-2::-1]:
        img = cv2.resize(cv2.pyrUp(img, lev_img.shape[1::-1]),lev_img.shape[1::-1])
        img += lev_img
    return img


def multi_band_blending(img1, img2, mask, overlap_w, leveln=None, flag_half=False, need_mask=False):
    if overlap_w < 0:
        print ("error: overlap_w should be a positive integer")
        sys.exit()

    if need_mask:  # no input mask
        subA, subB, mask = preprocess(img1, img2, overlap_w, flag_half, True)
    else:  # have input mask
        subA, subB, _ = preprocess(img1, img2, overlap_w, flag_half)

    max_leveln = int(np.floor(np.log2(min(img1.shape[0], img1.shape[1],
                                          img2.shape[0], img2.shape[1]))))
    if leveln is None:
        leveln = max_leveln
    if leveln < 1 or leveln > max_leveln:
        print ("warning: inappropriate number of leveln")
        leveln = max_leveln

    # Get Gaussian pyramid and Laplacian pyramid
    MP = GaussianPyramid(mask, leveln)
    LPA = LaplacianPyramid(subA, leveln)
    LPB = LaplacianPyramid(subB, leveln)

    # Blend two Laplacian pyramidspass
    blended = blend_pyramid(LPA, LPB, MP)

    # Reconstruction process
    result = reconstruct(blended)
    result[result > 255] = 255
    result[result < 0] = 0

    return result.astype(np.uint8)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser(
    #     description="A Python implementation of multi-band blending")
    # ap.add_argument('-f', '--first', required=True,
    #                 help="path to the first (left) image")
    # ap.add_argument('-s', '--second', required=True,
    #                 help="path to the second (right) image")
    # ap.add_argument('-m', '--mask', required=False,
    #                 help="path to the mask image")
    # ap.add_argument('-o', '--overlap', required=True, type=int,
    #                 help="width of the overlapped area between two images, \
    #                       even number recommended")
    # ap.add_argument('-l', '--leveln', required=False, type=int,
    #                 help="number of levels of multi-band blending, \
    #                       calculated from image size if not provided")
    # ap.add_argument('-H', '--half', required=False, action='store_true',
    #                 help="option to blend the left half of the first image \
    #                       and the right half of the second image")
    # args = vars(ap.parse_args())

    W = 960
    H = 540
    overlap_w = 445
    leveln = 5

    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", type=str, default="ref/0_3/0525_0_5_500_3_test_origin_warp.avi", help="path to the left video")
    ap.add_argument("-r", "--right", type=str, default="ref/0_3/0525_0_4_500_3_test_origin_warp.avi", help="path to the right video")
    args = vars(ap.parse_args())
    # 读取视频
    vs = cv2.VideoCapture(args["left"])
    num_frames = np.int32(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm.trange(num_frames) as t:
        t.set_description(f'Reading video from <{args["left"]}>')
        left_frames = []
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
                    f'Video at <{args["left"]}> did not have frame {frame_index} of '
                    f'{num_frames} (indexed from 0).'
                )
            left_frames.append(unstabilized_frame)
    vs.release()

    vs = cv2.VideoCapture(args["right"])
    num_frames = np.int32(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm.trange(num_frames) as t:
        t.set_description(f'Reading video from <{args["right"]}>')
        right_frames = []
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
                    f'Video at <{args["right"]}> did not have frame {frame_index} of '
                    f'{num_frames} (indexed from 0).'
                )
            right_frames.append(unstabilized_frame)
    vs.release()

    # frame_index = 20

    flag_half = False
    mask = None
    need_mask =True

    
    with tqdm.trange(num_frames) as t:
        t.set_description(f'stitching frames')
        stitched_frames = []
        for frame_index in t:
            img_l = left_frames[frame_index]
            img_r = right_frames[frame_index]
            # left = np.zeros((480,1140,3),np.uint8)
            # right = np.zeros((480,1140,3), np.uint8)
            # left[:,:840,:] = img_l
            # right[:,300:,:] =img_r
            result = multi_band_blending(img_l, img_r, mask, overlap_w, leveln, flag_half, need_mask)
            stitched_frames.append(result)
    
    output = 'ref/0_3/0525_0_5_4_500_3_test_origin_multiband.avi'
    frame_height, frame_width = stitched_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video = cv2.VideoWriter(output, fourcc, 20, (frame_width, frame_height))
    with tqdm.trange(num_frames) as t:
        t.set_description(f'Writing stabilized video to <{output}>')
        for frame_index in t:
            video.write(stitched_frames[frame_index])
    video.release()
    # img1 = left_frames[frame_index]
    # img2 = right_frames[frame_index]

    # left = np.zeros((480,800,3),np.uint8)
    # right = np.zeros((480,800,3),np.uint8)
    # left[:,:640,:] = img1
    # right[:,160:,:] = img2

    # cv2.imshow('left',left)
    # cv2.imshow('right',right)
    # cv2.waitKey(0)
    # if args['mask'] != None:
    #     mask = cv2.imread(args['mask'])
    #     mask = mask//255
    #     need_mask = False
    # else:
    # mask = None
    # need_mask =True
    # overlap_w = 480
    # # leveln = args['leveln']
    # # print('args: ', args)
    # leveln = None
    
    # result = multi_band_blending(img1, img2, mask, overlap_w, leveln, flag_half, need_mask)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)
    # print("blending result has been saved in 'result.png'")