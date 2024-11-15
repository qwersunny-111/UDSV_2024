import torch
import torch.nn as nn
import numpy as np
import os
import math
import cv2
import sys
parentddir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parentddir)

from models.DUT.DUT import DUT
from tqdm import tqdm
from utils.WarpUtils import warpListImage
from configs.config import cfg
import argparse

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser(description='Control for stabilization model')
    parser.add_argument('--SmootherPath', help='the path to pretrained smoother model, blank for jacobi solver', default='ckpt/smoother.pth')
    parser.add_argument('--RFDetPath', help='pretrained RFNet path, blank for corner detection', default='ckpt/RFDet_640.pth.tar')
    parser.add_argument('--PWCNetPath', help='pretrained pwcnet path, blank for KTL tracker', default='ckpt/network-default.pytorch')
    parser.add_argument('--MotionProPath', help='pretrained motion propagation model path, blank for median', default='ckpt/MotionPro.pth')
    parser.add_argument('--SingleHomo', help='whether use multi homograph to do motion estimation', action='store_true')
    # parser.add_argument('--InputBasePath', help='path to input videos (cliped as frames)', default='images/')
    parser.add_argument('--InputBasePath', help='path to input videos (cliped as frames)', default='/home/lianghao/workspace/TCSVT/experiments/sequences/case5/')
    # parser.add_argument('--OutputBasePath', help='path to save output stable videos', default='results/')
    parser.add_argument('--OutputBasePath', help='path to save output stable videos', default='/home/lianghao/workspace/TCSVT/experiments/case5/')
    parser.add_argument('--OutNamePrefix', help='prefix name before the output video name', default='case5_')
    parser.add_argument('--MaxLength', help='max number of frames can be dealt with one time', type=int, default=1200)
    parser.add_argument('--Repeat', help='max number of frames can be dealt with one time', type=int, default=50)
    return parser.parse_args()

def display_unstablilized_and_cropped_video_loop(num_frames, frames_per_second, unstabilized_frames, cropped_frames):
        
    # 每帧的显示时长(ms)
    milliseconds_per_frame = int(1000/frames_per_second)
    while True:
        for i in range(num_frames):
            # 按垂直方向(行顺序)堆叠数组
            cv2.imshow('unstabilized and stabilized video', np.vstack(
                (unstabilized_frames[i], cropped_frames[i])))
            if cv2.waitKey(milliseconds_per_frame) & 0xFF == ord('q'):
                return

 ##  RANSAC去噪并获得局部网格内的特征点坐标(相对于整张图像的全局坐标)  ##
def get_features_in_subframe(early_subframe, late_subframe, subframe_offset):

    # gather all features that track between frames
    early_features_including_outliers, late_features_including_outliers = get_all_matched_features_between_subframes(early_subframe, late_subframe)
    if early_features_including_outliers is None or len(early_features_including_outliers) < 12 :
        return (None, None)

    # eliminate outlying features using RANSAC
    # 局部网格内部使用RANSAC滤点，去除outliers
    _, outlier_features = cv2.findHomography(early_features_including_outliers, late_features_including_outliers, cv2.USAC_MAGSAC, 0.4, 0.999, 100000)
    outlier_features_mask = outlier_features.flatten().astype(dtype=bool)
    early_features = early_features_including_outliers[outlier_features_mask]
    late_features = late_features_including_outliers[outlier_features_mask]

    # Add a constant offset to feature coordinates to express them
    # relative to the original frame's top left corner, not the subframe's
    # 返回特征点相对于全局的图像坐标
    return (early_features + subframe_offset, late_features + subframe_offset)

##  获取相邻帧某局部网格内对应特征点(FAST检测+光流跟踪) ##
def get_all_matched_features_between_subframes(early_subframe, late_subframe):
       
    # convert a KeyPoint list into a CV_32FC2 array containing the coordinates of each KeyPoint;
    # see https://stackoverflow.com/a/55398871 and https://stackoverflow.com/a/47617999

    # FAST特征点检测
    feature_detector = cv2.FastFeatureDetector_create()
    early_keypoints = feature_detector.detect(early_subframe)
    if len(early_keypoints) < 6:
        return (None, None)

    early_features_including_unmatched = np.float32(cv2.KeyPoint_convert(early_keypoints)[:, np.newaxis, :])

    # 光流法跟踪前帧的特征点
    late_features_including_unmatched, matched_features, _ = cv2.calcOpticalFlowPyrLK(early_subframe, late_subframe, 
                                                                                          early_features_including_unmatched, None)

    # 创建掩膜，仅提取出匹配完成的特征点
    matched_features_mask = matched_features.flatten().astype(dtype=bool)
    early_features = early_features_including_unmatched[matched_features_mask]
    late_features = late_features_including_unmatched[matched_features_mask]

    if len(early_features) < 6:
        return (None, None)

    return (early_features, late_features)

##  获得帧间单应变换供评估cropping ratio  ##
def get_matched_homography_for_evaluation(early_frame, late_frame):

    frame_height, frame_width = early_frame.shape[:2]
    # 局部网格宽高
    subframe_width = math.ceil(frame_width / 4)
    subframe_height = math.ceil(frame_height / 4)

    # early_features_by_subframe[i] contains a CV_32FC2 array of the early features in the
    # frame's i^th subframe;
    # late_features_by_subframe[i] is defined similarly
    # 依次存储局部网格中的特征点
    early_features_by_subframe = []
    late_features_by_subframe = []

    # TODO parallelize
    for subframe_left_x in range(0, frame_width, subframe_width):
        for subframe_top_y in range(0, frame_height, subframe_height):
            # 依次取局部网格，顺序先行后列
            early_subframe = early_frame[subframe_top_y:subframe_top_y+subframe_height,
                                         subframe_left_x:subframe_left_x+subframe_width]
            late_subframe = late_frame[subframe_top_y:subframe_top_y+subframe_height,
                                       subframe_left_x:subframe_left_x+subframe_width]
            # 局部网格左上角坐标
            subframe_offset = [subframe_left_x, subframe_top_y]
            # 获得局部网格中的特征点
            subframe_early_features, subframe_late_features = get_features_in_subframe(early_subframe, late_subframe, subframe_offset)

            # 特征点存入数组
            if subframe_early_features is not None:
                early_features_by_subframe.append(subframe_early_features)
            if subframe_late_features is not None:
                late_features_by_subframe.append(subframe_late_features)

    # 数组拼接(默认axis=0,纵向拼接)，获得全局特征点
    early_features = np.concatenate(early_features_by_subframe, axis=0)
    late_features = np.concatenate(late_features_by_subframe, axis=0)

    if len(early_features) < 6:
        return (None, None, None)

    # 获取单应性矩阵
    early_to_late_homography, _ = cv2.findHomography(early_features, late_features)

    return early_to_late_homography

def compute_cropping_ratio_and_distortion_score(num_frames, unstabilized_frames, cropped_frames):
        '''
        Helper function for stabilize.

        Compute the cropping ratio and distortion score for the given stabilization using the
        definitions of these metrics in the original paper.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * stabilized_frames: A list of the stabilized frames, each represented as a NumPy array.

        Output:

        A tuple of the following items in order.

        * cropping_ratio: The cropping ratio of the stabilized video. Per the original paper, the
            cropping ratio of each frame is the scale component of its unstabilized-to-cropped
            homography, and the cropping ratio of the overall video is the average of the frames'
            cropping ratios.
        * distortion_score: The distortion score of the stabilized video. Per the original paper,
            the distortion score of each frame is ratio of the two largest eigenvalues of the
            affine part of its unstabilized-to-cropped homography, and the distortion score of the
            overall video is the greatest of its frames' distortion scores.
        '''

        cropping_ratios = np.empty((num_frames), dtype=np.float32)
        distortion_scores = np.empty((num_frames), dtype=np.float32)
       
        with tqdm(range(num_frames)) as t:
            t.set_description('Computing cropping ratio and distortion score')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]
                cropped_frame = cropped_frames[frame_index]
                unstabilized_to_cropped_homography = get_matched_homography_for_evaluation(unstabilized_frame, cropped_frame)

                # the scaling component has x-component cropped_to_unstabilized_homography[0][0]
                # and y-component cropped_to_unstabilized_homography[1][1],
                # so the fraction of the enlarged video that actually fits in the frame is
                # 1 / (cropped_to_unstabilized_homography[0][0] * cropped_to_unstabilized_homography[1][1])
                cropping_ratio = 1 / (unstabilized_to_cropped_homography[0][0] * unstabilized_to_cropped_homography[1][1])
                cropping_ratios[frame_index] = cropping_ratio

                affine_component = np.copy(unstabilized_to_cropped_homography)
                affine_component[2] = [0, 0, 1]
                eigenvalue_magnitudes = np.sort(
                    np.abs(np.linalg.eigvals(affine_component)))
                distortion_score = eigenvalue_magnitudes[-2] /  eigenvalue_magnitudes[-1]
                distortion_scores[frame_index] = distortion_score

        return (np.mean(cropping_ratios), np.min(distortion_scores))
#（number frame,row+1,col+1,2） (:,:,:,0)是x
def compute_stability_score(vertex_stabilized_displacements_by_frame_index):
        '''
        Helper function for stabilize.

        Compute the stability score for the given stabilization using the definitions of these
        metrics in the original paper.

        Input:

        * num_frames: The number of frames in the video.
        * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
            stabilized displacements of each vertex in the MeshFlow mesh, as generated by
            _get_stabilized_vertex_displacements.

        Output:

        * stability_score: The stability score of the stabilized video. Per the original paper, the
            stability score for each vertex is derived from the representation of its vertex profile
            (vector of velocities) in the frequency domain. Specifically, it is the fraction of the
            representation's total energy that is contained within its second to sixth lowest
            frequencies. The stability score of the overall video is the average of the vertices'
            average x- and y-stability scores.
        '''

        vertex_stabilized_x_dispacements_by_row_and_col, vertex_stabilized_y_dispacements_by_row_and_col = np.swapaxes(
            vertex_stabilized_displacements_by_frame_index, 0, 3)
        vertex_x_profiles_by_row_and_col = np.diff(
            vertex_stabilized_x_dispacements_by_row_and_col)
        vertex_y_profiles_by_row_and_col = np.diff(
            vertex_stabilized_y_dispacements_by_row_and_col)

        vertex_x_freq_energies_by_row_and_col = np.square(
            np.abs(np.fft.fft(vertex_x_profiles_by_row_and_col)))
        vertex_y_freq_energies_by_row_and_col = np.square(
            np.abs(np.fft.fft(vertex_y_profiles_by_row_and_col)))

        vertex_x_total_freq_energy_by_row_and_col = np.sum(
            vertex_x_freq_energies_by_row_and_col, axis=2)
        vertex_y_total_freq_energy_by_row_and_col = np.sum(
            vertex_y_freq_energies_by_row_and_col, axis=2)

        vertex_x_low_freq_energy_by_row_and_col = np.sum(
            vertex_x_freq_energies_by_row_and_col[:, :, 1:6], axis=2)
        vertex_y_low_freq_energy_by_row_and_col = np.sum(
            vertex_y_freq_energies_by_row_and_col[:, :, 1:6], axis=2)

        x_stability_scores_by_row_and_col = vertex_x_low_freq_energy_by_row_and_col / \
            vertex_x_total_freq_energy_by_row_and_col
        y_stability_scores_by_row_and_col = vertex_y_low_freq_energy_by_row_and_col / \
            vertex_y_total_freq_energy_by_row_and_col

        x_stability_score = np.mean(x_stability_scores_by_row_and_col)
        y_stability_score = np.mean(y_stability_scores_by_row_and_col)

        return (x_stability_score + y_stability_score) 
def generateStable(model, base_path, outPath, outPrefix, max_length, args):

    image_base_path = base_path
    image_len = min(len([ele for ele in os.listdir(image_base_path) if ele[-4:] == '.jpg']), max_length)
    # read input video
    unstableimages = []
    images = []
    rgbimages = []
    for i in range(image_len):
        image = cv2.imread(os.path.join(image_base_path, '{:06d}.jpg'.format(i)), 0)
        image = image * (1. / 255.)
        image = cv2.resize(image, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        images.append(image.reshape(1, 1, cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH))

        image = cv2.imread(os.path.join(image_base_path, '{:06d}.jpg'.format(i)))
        image = cv2.resize(image, (cfg.MODEL.HEIGHT, cfg.MODEL.WIDTH))
        unstableimages.append(image)
        image = cv2.imread(os.path.join(image_base_path, '{:06d}.jpg'.format(i)))
        image = cv2.resize(image, (cfg.MODEL.WIDTH, cfg.MODEL.HEIGHT))
        rgbimages.append(np.expand_dims(np.transpose(image, (2, 0, 1)), 0))

    x = np.concatenate(images, 1).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0)

    x_RGB = np.concatenate(rgbimages, 0).astype(np.float32)
    x_RGB = torch.from_numpy(x_RGB).unsqueeze(0)

    with torch.no_grad():
        origin_motion, smoothPath = model.inference(x.cuda(), x_RGB.cuda(), repeat=args.Repeat)

    origin_motion = origin_motion.cpu().numpy()
    smoothPath = smoothPath.cpu().numpy()
    origin_motion = np.transpose(origin_motion[0], (2, 3, 1, 0))
    smoothPath = np.transpose(smoothPath[0], (2, 3, 1, 0))
    print(smoothPath.shape)

    x_paths = origin_motion[:, :, :, 0]
    y_paths = origin_motion[:, :, :, 1]
    sx_paths = smoothPath[:, :, :, 0]
    sy_paths = smoothPath[:, :, :, 1]

    frame_rate = 25
    frame_width = cfg.MODEL.WIDTH
    frame_height = cfg.MODEL.HEIGHT
    
    print("generate stabilized video...")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(outPath, outPrefix + 'DUT_stable.mp4'), fourcc, frame_rate, (frame_width, frame_height))

    new_x_motion_meshes = sx_paths - x_paths
    new_y_motion_meshes = sy_paths - y_paths

    outImages = warpListImage(rgbimages, new_x_motion_meshes, new_y_motion_meshes)
    outImages = outImages.numpy().astype(np.uint8)
    outImages = [np.transpose(outImages[idx], (1, 2, 0)) for idx in range(outImages.shape[0])]
    stableimages = []
    for frame in tqdm(outImages):
        VERTICAL_BORDER = 60
        HORIZONTAL_BORDER = 80

        new_frame = frame[VERTICAL_BORDER:-VERTICAL_BORDER, HORIZONTAL_BORDER:-HORIZONTAL_BORDER]
        # new_frame = cv2.resize(new_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        new_frame = cv2.resize(new_frame, (frame.shape[0], frame.shape[1]), interpolation=cv2.INTER_CUBIC)
        stableimages.append(new_frame)
        out.write(new_frame)
    out.release()

    # display_unstablilized_and_cropped_video_loop(200, 20, unstableimages, stableimages)
    smoothPath = np.transpose(smoothPath, (2, 1, 0, 3))
    print(smoothPath.shape)
    stability_score = compute_stability_score(smoothPath)
    cropping_ratio, distortion_score = compute_cropping_ratio_and_distortion_score(
        smoothPath.shape[0], unstableimages, stableimages)
    print('cropping ratio:', cropping_ratio)
    print('distortion score:', distortion_score)
    print('stability score:', stability_score)

if __name__ == "__main__":

    args = parse_args()
    print(args)

    smootherPath = args.SmootherPath
    RFDetPath = args.RFDetPath
    PWCNetPath = args.PWCNetPath
    MotionProPath = args.MotionProPath
    homo = not args.SingleHomo
    inPath = args.InputBasePath
    outPath = args.OutputBasePath
    outPrefix = args.OutNamePrefix
    maxlength = args.MaxLength

    model = DUT(SmootherPath=smootherPath, RFDetPath=RFDetPath, PWCNetPath=PWCNetPath, MotionProPath=MotionProPath, homo=homo)
    model.cuda()
    model.eval()

    generateStable(model, inPath, outPath, outPrefix, maxlength, args)
