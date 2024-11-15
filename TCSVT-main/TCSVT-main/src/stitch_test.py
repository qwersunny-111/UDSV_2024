import cv2
import numpy as np
import tqdm
import argparse
import time
import stitch_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter

from multiband import multi_band_blending

from cylinder import cylinder

# W = 640
# H = 720
# O = 240

# mesh_row_count = 12
# mesh_col_count = 16

# feature_ellipse_row_count = 12
# feature_ellipse_col_count = 12

# W = 1280
# W = 1080
# W = 640
# H = 720
# O = 400

W = 960
H = 540

# mesh_row_count = 12
# mesh_col_count = 20
mesh_row_count = 10
mesh_col_count = 16

feature_ellipse_col_count = 10
feature_ellipse_row_count = 6


frame_idx = 20

def measure_performance(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(method.__name__+' has taken: '+str(end_time-start_time)+' sec')
        return result
    return timed

# @measure_performance
def seamcut(src, dst):

    import maxflow
    from get_energy_map.energy import get_energy_map
    img_pixel1,img_pixel2,left,right,up,down = get_energy_map(src, dst)

    g = maxflow.GraphFloat()
    img_pixel1 = img_pixel1.astype(float)
    img_pixel1 = img_pixel1*1e10
    img_pixel2 = img_pixel2.astype(float)
    img_pixel2 = img_pixel2*1e10
    nodeids = g.add_grid_nodes(img_pixel1.shape)
    # print(img_pixel1.shape)
    g.add_grid_tedges(nodeids,img_pixel1,img_pixel2)
    structure_left = np.array([[0,0,0],
                            [0,0,1],
                            [0,0,0]])
    g.add_grid_edges(nodeids,weights=left,structure=structure_left,symmetric=False)
    structure_right = np.array([[0,0,0],
                            [1,0,0],
                            [0,0,0]])
    g.add_grid_edges(nodeids,weights=right,structure=structure_right,symmetric=False)
    structure_up = np.array([[0,0,0],
                            [0,0,0],
                            [0,1,0]])
    g.add_grid_edges(nodeids,weights=up,structure=structure_up,symmetric=False)
    structure_down = np.array([[0,1,0],
                            [0,0,0],
                            [0,0,0]])
    g.add_grid_edges(nodeids,weights=down,structure=structure_down,symmetric=False)
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)

    # The labels should be 1 where sgm is False and 0 otherwise.
    img2 = np.int_(np.logical_not(sgm))
    src_mask = img2.astype(np.uint8)
    dst_mask = np.logical_not(img2).astype(np.uint8)
    src_mask = np.stack((src_mask,src_mask,src_mask),axis=-1)
    dst_mask = np.stack((dst_mask,dst_mask,dst_mask),axis=-1)

    src = src*src_mask
    dst = dst*dst_mask

    result = src+dst
    return result

def plot_motionfield(motion_field_origin, motion_field_filtered):
  
    stabilized_mediafiltered_motion_mesh_x = motion_field_filtered[:, :, 0]
    stabilized_mediafiltered_motion_mesh_y = motion_field_filtered[:, :, 1]
    
    stabilized_motion_mesh_unfiltered_x = motion_field_origin[:, :, 0]
    stabilized_motion_mesh_unfiltered_y = motion_field_origin[:, :, 1]
    
    fig = plt.figure()

    x = np.arange(0,1280+64,64)
    y = np.arange(0,720+60,60)
    x, y = np.meshgrid(x, y)

    ax1 = fig.add_subplot(221, projection="3d")
    surf = ax1.plot_surface(x, y, stabilized_motion_mesh_unfiltered_x, rstride=1, cstride=1, alpha=0.9, cmap=plt.cm.coolwarm)
    ax1.set_zlim(-400, -100)
    ax1.set_xlabel("W")
    ax1.set_ylabel("H")
    ax1.set_zlabel("pixel")
    # ax1.set_title("stabilized_unfiltered_motion_mesh_x")
    ax1.contourf(x, y, stabilized_mediafiltered_motion_mesh_y, zdir='z',offset=-400)

    ax2 = fig.add_subplot(222, projection="3d")
    surf = ax2.plot_surface(x, y, stabilized_mediafiltered_motion_mesh_x, rstride=1, cstride=1, alpha=0.9, cmap=plt.cm.coolwarm)
    ax2.set_zlim(-400, -100)
    cax1 = fig.add_axes([ax2.get_position().x1+0.05, ax2.get_position().y0, 0.02, ax2.get_position().height])
    fig.colorbar(surf, cax=cax1)
    ax2.set_xlabel("W")
    ax2.set_ylabel("H")
    ax2.set_zlabel("pixel")
    # ax2.set_title("stabilized_mediafiltered_motion_mesh_x")
    ax2.contourf(x, y, stabilized_mediafiltered_motion_mesh_y, zdir='z',offset=-400)
    
    ax3 = fig.add_subplot(223, projection="3d")
    surf = ax3.plot_surface(x, y, stabilized_motion_mesh_unfiltered_y, rstride=1, cstride=1, alpha=0.9, cmap=plt.cm.coolwarm)
    ax3.set_zlim(-100, 100)
    ax3.set_xlabel("W")
    ax3.set_ylabel("H")
    ax3.set_zlabel("pixel")
    # ax3.set_title("stabilized_unfiltered_motion_mesh_y")
    ax3.contourf(x, y, stabilized_motion_mesh_unfiltered_y, zdir='z',offset=-100)
    
    ax4 = fig.add_subplot(224, projection="3d")
    surf = ax4.plot_surface(x, y, stabilized_mediafiltered_motion_mesh_y, rstride=1, cstride=1, alpha=0.9, cmap=plt.cm.coolwarm)
    ax4.set_zlim(-100, 100)
    cax2 = fig.add_axes([ax4.get_position().x1+0.05, ax4.get_position().y0, 0.02, ax4.get_position().height])
    fig.colorbar(surf, cax=cax2)
    ax4.set_xlabel("W")
    ax4.set_ylabel("H")
    ax4.set_zlabel("pixel")
    # ax4.set_title("stabilized_mediafiltered_motion_mesh_y")
    ax4.contourf(x, y, stabilized_mediafiltered_motion_mesh_y, zdir='z',offset=-100)
    
    plt.show()

def motion_field_filter(left_velocity, right_velocity):
    # 中值滤波器去噪
    left_velocity = median_filter(left_velocity, size=3)
    right_velocity = median_filter(right_velocity, size=3)

    # 运动场的平滑过程 调试中
    print(np.median(left_velocity[:, 14, 0]))
    print(np.median(right_velocity[:, 2, 0]))
    O = int((- np.ceil(np.median(left_velocity[:, 14, 0])) + np.floor(np.median(right_velocity[:, 2, 0]))) // 2)
    print(O)
    O_l = -int(np.ceil(np.median(left_velocity[:, 14, 0])))
    print(O_l)
    O_r = int(np.floor(np.median(right_velocity[:, 2, 0])))

    # 边缘的运动场平滑 根据平移量置固定值
    vertex_motion_x_l = left_velocity[:, :16, 0]
    vertex_motion_y_l = left_velocity[:, :16, 1]

    vertex_motion_x_r = right_velocity[:, -16:, 0]
    vertex_motion_y_r = right_velocity[:, -16:, 1]

    vertex_motion_y_l[:, :11] = 0
    vertex_motion_x_l[:, :11] = -O_l
    # # vertex_motion_x_l[:, 1] = -O 
    # vertex_motion_x_l[:, 2] = -O - 15

    vertex_motion_y_r[:, -11:] = 0
    vertex_motion_x_r[:, -11:] = O_r
    # vertex_motion_x_r[:, -2] = O 
    # vertex_motion_x_r[:, -3] = O + 15

    # 均值滤波器
    vertex_motion_x_l_filter = uniform_filter(vertex_motion_x_l, size=5)
    vertex_motion_y_l_filter = uniform_filter(vertex_motion_y_l, size=5)

    vertex_motion_x_r_filter = uniform_filter(vertex_motion_x_r, size=5)
    vertex_motion_y_r_filter = uniform_filter(vertex_motion_y_r, size=5)

    # vertex_motion_x_l_filter[:, :5] = -O
    # vertex_motion_x_r_filter[:, -5:] = O

    # vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l_filter))
    # left_velocity[:, :12, :] = vertex_motion_l_no

    # vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r_filter))
    # right_velocity[:, -12:, :] = vertex_motion_r_no

    vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l_filter))
    # vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l))
    left_velocity[:, :16, :] = vertex_motion_l_no

    vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r_filter))
    # vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r))
    right_velocity[:, -16:, :] = vertex_motion_r_no

    return left_velocity, right_velocity, O_l, O_r, O

if __name__=='__main__':
    
    # 从视频流中读取帧
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-l", "--left", type=str, default="240329/l.mp4", help="path to the left video")
    # ap.add_argument("-r", "--right", type=str, default="240329/r.mp4", help="path to the right video")
    # # ap.add_argument("-i", "--frameIdx", type=int, default=150, help="the selected frame index")
    # args = vars(ap.parse_args())

    # # 读取视频
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
    #             # unstabilized_frame = cylinder(pixels, W, H)
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

    # vs = cv2.VideoCapture(args["right"])
    # num_frames = np.int32(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    # with tqdm.trange(num_frames) as t:
    #     t.set_description(f'Reading video from <{args["right"]}>')
    #     right_frames = []
    #     for frame_index in t:
    #         success, pixels = vs.read()
    #         if success:
    #             # 规范分辨率
    #             unstabilized_frame = cv2.resize(pixels, (W, H))
    #             # unstabilized_frame = cylinder(pixels, W, H)
    #         else:
    #             print('capture error')
    #             exit()
    #         if unstabilized_frame is None:
    #             raise IOError(
    #                 f'Video at <{args["right"]}> did not have frame {frame_index} of '
    #                 f'{num_frames} (indexed from 0).'
    #             )
    #         right_frames.append(unstabilized_frame)
    # vs.release()

    # # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # # fps = 30
    # # frame_height, frame_width, _ = left_frames[0].shape
    # # video = cv2.VideoWriter('real_09/calib_0_0/09_0_0_4_100_c.mp4', fourcc, fps, (frame_width, frame_height))
    # # for i in tqdm.trange(num_frames):
    # #     video.write(left_frames[i])
    # # video.release()
    # # video = cv2.VideoWriter('real_09/calib_0_0/09_0_0_3_100_c.mp4', fourcc, fps, (frame_width, frame_height))
    # # for i in tqdm.trange(num_frames):
    # #     video.write(right_frames[i])
    # # video.release()
    # # print('Write Done!')

    # left_frame = left_frames[frame_idx]
    # right_frame = right_frames[frame_idx]


    # 从图片中读取帧
    # left_frame = cv2.imread("real_09/final_0/video2/640.jpg")
    # right_frame = cv2.imread("real_09/final_0/video1/640.jpg")
    left_frame = cv2.imread("real_09/final_5/video3/180.jpg")
    right_frame = cv2.imread("real_09/final_5/video2/180.jpg")
    left_frame_1 = cv2.imread("real_09/final_6/video3/555.jpg")
    right_frame_1 = cv2.imread("real_09/final_6/video2/555.jpg")
    # left_frame = cv2.resize(left_frame, (W, H))
    # right_frame = cv2.resize(right_frame, (W, H))

    stitcher = stitch_utils.stitch_utils(mesh_row_count=mesh_row_count, mesh_col_count=mesh_col_count, 
                                         feature_ellipse_row_count=feature_ellipse_row_count, feature_ellipse_col_count=feature_ellipse_col_count)
                                        #  overlap_region=O)

    # 获取匹配特征点 特征点对中点 以及全局单应矩阵
    left_features, right_features, middle_features, early_to_late_homography_l, early_to_late_homography_r = stitcher.get_matched_features_and_homography_for_stitch(left_frame, right_frame) 
    # 获取拼接使用的网格运动场
    left_velocity, _ = stitcher.get_velocities_for_stitch(left_frame, left_features, middle_features, early_to_late_homography_l)
    right_velocity, _ = stitcher.get_velocities_for_stitch(left_frame, right_features, middle_features, early_to_late_homography_r)
    
    # print(left_features.shape)
    # print(left_velocity.shape)
    # err = stitcher.proj_err(W, H, left_features, middle_features, left_velocity)

    # 运动场可视化
    left_velocity_origin = left_velocity.copy()
    right_velocity_origin = right_velocity.copy()

    ##  原始运动场的可视化部分  ##
    meshes = stitcher.get_vertex_x_y(frame_width=W, frame_height=H)
    origin_vertex = np.reshape(meshes, (mesh_row_count + 1, mesh_col_count + 1, 2))
    for mesh in meshes:
        left_frame_mesh = cv2.circle(left_frame, (int(mesh[0, 0]), int(mesh[0, 1])), 2, (240, 100, 0), -1)
        right_frame_mesh = cv2.circle(right_frame, (int(mesh[0, 0]), int(mesh[0, 1])), 2, (100, 240, 0), -1)
    
    final_vertex_l = origin_vertex + 0.5 * left_velocity
    final_vertex_r = origin_vertex + 0.5 * right_velocity

    left_frame_mesh_1 = left_frame_mesh.copy()
    left_frame_mesh_2 = left_frame_mesh.copy()
    right_frame_mesh_1 = right_frame_mesh.copy()
    right_frame_mesh_2 = right_frame_mesh.copy()

    for i in range(mesh_row_count + 1):
        for j in range(mesh_col_count + 1):
            motion_field_l = cv2.line(left_frame_mesh_1, (int(origin_vertex[i,j,0]), int(origin_vertex[i,j,1])), (int(final_vertex_l[i,j,0]), int(final_vertex_l[i,j,1])), (0,0,255), 1)
            motion_field_r = cv2.line(right_frame_mesh_1, (int(origin_vertex[i,j,0]), int(origin_vertex[i,j,1])), (int(final_vertex_r[i,j,0]), int(final_vertex_r[i,j,1])), (0,0,255), 1)


    # 运动场的平滑过程
    left_velocity, right_velocity, O_l, O_r, O = motion_field_filter(left_velocity, right_velocity)


    # 运动场可视化
    # plot_motionfield(left_velocity_origin, left_velocity)
    # O = int((- np.ceil(np.median(left_velocity[:, 16, 0])) + np.floor(np.median(right_velocity[:, 2, 0]))) // 2)
    # print(O)
    # O_l = -int(np.ceil(np.median(left_velocity[:, 16, 0])))
    # print(O_l)
    # O_r = int(np.floor(np.median(right_velocity[:, 2, 0])))

    #  平滑后运动场的可视化部分  ##
    final_vertex_l_filter = origin_vertex + 0.5 * left_velocity
    final_vertex_r_filter = origin_vertex + 0.5 * right_velocity  

    for i in range(mesh_row_count + 1):
        for j in range(mesh_col_count + 1):
            motion_field_filter_l = cv2.line(left_frame_mesh_2, (int(origin_vertex[i,j,0]), int(origin_vertex[i,j,1])), (int(final_vertex_l_filter[i,j,0]), int(final_vertex_l_filter[i,j,1])), (0,0,255), 1)
            motion_field_filter_r = cv2.line(right_frame_mesh_2, (int(origin_vertex[i,j,0]), int(origin_vertex[i,j,1])), (int(final_vertex_r_filter[i,j,0]), int(final_vertex_r_filter[i,j,1])), (0,0,255), 1)

    cv2.imshow('motion field of left', motion_field_l)
    cv2.imshow('motion field filtered of left', motion_field_filter_l)
    cv2.imshow('motion field of right', motion_field_r)
    cv2.imshow('motion field filtered of right', motion_field_filter_r)
    cv2.waitKey(0)

    ## 图片展示 ##
    # # 网格变形
    img_l = stitcher.get_warped_frames_for_stitch(0, left_frame_1, left_velocity, O_l)
    img_r = stitcher.get_warped_frames_for_stitch(1, right_frame_1, right_velocity, O_r)
    # 15
    O = O+40

    # 缝合线选取
    l = np.zeros((H, W + 2 * O, 3), np.uint8)
    r = np.zeros((H, W + 2 * O, 3), np.uint8)
    l[:, :W, :] = img_l
    r[:, 2 * O:, :] = img_r
    stitched_seam = seamcut(l, r)

    # 多频段融合
    flag_half = False
    mask = None
    need_mask =True
    overlap_w = W-2*O#480#880#440#590 #595  # 140  #245
    leveln = 5
    stitched_band = multi_band_blending(img_l, img_r, mask, overlap_w, leveln, flag_half, need_mask)

    # # 显示结果
    cv2.imshow('warp_left', img_l)
    cv2.imshow('warp_right', img_r)
    # cv2.imshow('left', l)
    # cv2.imshow('right', r)
    cv2.imshow('stitched', stitched_seam)
    cv2.imshow('stitched_multiband', stitched_band)
    cv2.waitKey(0)
    # cv2.imwrite('real_09/final_3/125_2_1_seam.jpg', stitched_seam)
    # cv2.imwrite('real_09/final_3/125_2_1_multiband.jpg', stitched_band)



    ## 视频处理 ##
    # with tqdm.trange(num_frames) as t:
    #     t.set_description(f'stitching frames')
    #     stitched_frames = []
    #     stitched_frames_multiband = []
    #     left_warp = []
    #     right_warp = []
    #     for frame_index in t:
    #         left_frame_ = left_frames[frame_index]
    #         right_frame_ = right_frames[frame_index]
    #         # 网格变形
    #         img_l = stitcher.get_warped_frames_for_stitch(0, left_frame_, left_velocity, O_l)
    #         img_r = stitcher.get_warped_frames_for_stitch(1, right_frame_, right_velocity, O_r)
    #         left_warp.append(img_l)
    #         right_warp.append(img_r)
    #         # 缝合线选取
    #         l_img = np.zeros((H, W + 2 * O, 3), np.uint8)
    #         r_img = np.zeros((H, W + 2 * O, 3), np.uint8)
    #         l_img[:, :W, :] = img_l
    #         r_img[:, 2 * O:, :] = img_r
    #         stitched = seamcut(l_img, r_img)
    #         stitched_multiband = multi_band_blending(img_l, img_r, mask, overlap_w, leveln, flag_half, need_mask)
    #         stitched_frames.append(stitched)
    #         stitched_frames_multiband.append(stitched_multiband)

    # # 保存视频文件
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fps = 30

    # frame_height, frame_width, _ = left_warp[0].shape
    # video = cv2.VideoWriter('real_09/calib_0_6/09_0_6_4_3_l_warp_c.mp4', fourcc, fps, (frame_width, frame_height))
    # for i in tqdm.trange(num_frames):
    #     video.write(left_warp[i])
    # video.release()
    # video = cv2.VideoWriter('real_09/calib_0_6/09_0_6_4_3_r_warp_c.mp4', fourcc, fps, (frame_width, frame_height))
    # for i in tqdm.trange(num_frames):
    #     video.write(right_warp[i])
    # video.release()

    # frame_height, frame_width, _ = stitched_frames[0].shape
    # video = cv2.VideoWriter('real_09/calib_0_6/09_0_6_4_3_c.mp4', fourcc, fps, (frame_width, frame_height))
    # for i in tqdm.trange(num_frames):
    #     video.write(stitched_frames[i])
    # video.release()
    
    # frame_height, frame_width, _ = stitched_frames_multiband[0].shape
    # video = cv2.VideoWriter('real_09/calib_0_6/09_0_6_4_3_multiband_c.mp4', fourcc, fps, (frame_width, frame_height))
    # for i in tqdm.trange(num_frames):
    #     video.write(stitched_frames_multiband[i])
    # video.release()
    # print('Write Done!')