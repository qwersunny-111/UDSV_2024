import cv2
import math
import numpy as np
import statistics
import tqdm
import multiprocessing as mp
from functools import partial
import time
import matplotlib.pyplot as plt
import argparse
import stitch_utils
from multiband import multi_band_blending

from scipy.ndimage import uniform_filter
from scipy.ndimage import median_filter

mesh_row_count = 10
mesh_col_count = 16
multicore = 4
W = 960
H = 540
O = 30

class MeshFlowStabilizer:

    ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL = 0
    ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED = 1
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH = 2
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW = 3

    # The adaptive weights' constant high and low values(100 and 1).
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH_VALUE = 80
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW_VALUE = 2

    ##  初始化参数  ##
    def __init__(self, mesh_row_count=10, mesh_col_count=16,  # 网格行数与列数，定点数各加1
                 mesh_outlier_subframe_row_count=4, mesh_outlier_subframe_col_count=4,  # 图像划分为4*4，设立局部阈值RANSAC
                 feature_ellipse_row_count=8, feature_ellipse_col_count=10,  # 每个特征点所占椭圆覆盖的行\列
                 homography_min_number_corresponding_features=6,
                 temporal_smoothing_radius=10,  # \Omega_t 时域平滑半径
                 optimization_num_iterations=100,  # 雅可比方法最小化能量函数时的迭代次数
                 color_outside_image_area_bgr=(0, 0, 0),  # 稳定图像后设置背景色，避免图像无法覆盖窗口
                 multicore=4,
                 visualize=False):
        '''
        Constructor.

        Input:

        * mesh_row_count: The number of rows contained in the mesh.
            NOTE There are 1 + mesh_row_count vertices per row.
        * mesh_col_count: The number of cols contained in the mesh.
            NOTE There are 1 + mesh_col_count vertices per column.
        * mesh_outlier_subframe_row_count: The height in rows of each subframe when breaking down
            the image into subframes to eliminate outlying features.
        * mesh_outlier_subframe_col_count: The width of columns of each subframe when breaking
            down the image into subframes to eliminate outlying features.
        * feature_ellipse_row_count: The height in rows of the ellipse drawn around each feature
            to match it with vertices in the mesh.
        * feature_ellipse_col_count: The width in columns of the ellipse drawn around each feature
            to match it with vertices in the mesh.
        * homography_min_number_corresponding_features: The minimum number of features that must
            correspond between two frames to perform a homography.
        * temporal_smoothing_radius: In the energy function used to smooth the image, the number of
            frames to inspect both before and after each frame when computing that frame's
            regularization term. Thus, the regularization term involves a sum over up to
            2 * temporal_smoothing_radius frame indexes.
            NOTE This constant is denoted as \Omega_{t} in the original paper.
        * optimization_num_iterations: The number of iterations of the Jacobi method to perform when
            minimizing the energy function.
        * color_outside_image_area_bgr: The color, expressed in BGR, to display behind the
            stabilized footage in the output.
            NOTE This color should be removed during cropping, but is customizable just in case.
        * visualize: Whether or not to display a video loop of the unstabilized and cropped,
            stabilized videos after saving the stabilized video. Pressing Q closes the window.

        Output:

        (A MeshFlowStabilizer object.)
        '''

        self.mesh_col_count = mesh_col_count
        self.mesh_row_count = mesh_row_count
        self.mesh_outlier_subframe_row_count = mesh_outlier_subframe_row_count
        self.mesh_outlier_subframe_col_count = mesh_outlier_subframe_col_count
        self.feature_ellipse_row_count = feature_ellipse_row_count
        self.feature_ellipse_col_count = feature_ellipse_col_count
        self.homography_min_number_corresponding_features = homography_min_number_corresponding_features
        self.temporal_smoothing_radius = temporal_smoothing_radius
        self.optimization_num_iterations = optimization_num_iterations
        self.color_outside_image_area_bgr = color_outside_image_area_bgr
        self.multicore = multicore
        self.visualize = visualize


    ##  主功能函数 稳定图像  ##
    def stabilize(self, input_path, output_path, adaptive_weights_definition = ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path.

        Input:

        * input_path: The path to a video.
        * output_path: The path where the stabilized version of the video should be placed.
        * adaptive_weights_definition: Which method to use for computing the energy function's adaptive
            weights.

        Output:

        (The stabilized video is saved to output_path.)

        In addition, the function returns a tuple of the following items in order.

        * cropping_ratio: The cropping ratio of the stabilized video. Per the original paper, the
            cropping ratio of each frame is the scale component of its unstabilized-to-cropped
            homography, and the cropping ratio of the overall video is the average of the frames'
            cropping ratios.
        * distortion_score: The distortion score of the stabilized video. Per the original paper,
            the distortion score of each frame is ratio of the two largest eigenvalues of the
            affine part of its unstabilized-to-cropped homography, and the distortion score of the
            overall video is the greatest of its frames' distortion scores.
        * stability_score: The stability score of the stabilized video. Per the original paper, the
            stability score for each vertex is derived from the representation of its vertex profile
            (vector of velocities) in the frequency domain. Specifically, it is the fraction of the
            representation's total energy that is contained within its second to sixth lowest
            frequencies. The stability score of the overall video is the average of the vertices'
            stability scores.
        '''

        if not (adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or
                adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED or
                adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH or
                adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW):
            raise ValueError(
                'Invalid value for `adaptive_weights_definition`. Expecting value of '
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL`, '
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED`, '
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH`, or'
                '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW`.'
            )

        # 取图像、总帧数、帧率、编码方式
        unstabilized_frames, num_frames, frames_per_second = self._get_unstabilized_frames_and_video_features(input_path)
        #
        vertex_unstabilized_displacements_by_frame_index, homographies = self._get_unstabilized_vertex_displacements_and_homographies(
            num_frames, unstabilized_frames)
        #
        vertex_stabilized_displacements_by_frame_index = self._get_stabilized_vertex_displacements(
            num_frames, unstabilized_frames, adaptive_weights_definition,
            vertex_unstabilized_displacements_by_frame_index, homographies
        )
        #
        stabilized_frames, crop_boundaries = self._get_stabilized_frames_and_crop_boundaries_with_multiprocessing(
            num_frames, unstabilized_frames,
            vertex_unstabilized_displacements_by_frame_index,
            vertex_stabilized_displacements_by_frame_index
        )
        #
        cropped_frames = self._crop_frames(stabilized_frames, crop_boundaries)

        # 输出评价参数
        cropping_ratio, distortion_score = self._compute_cropping_ratio_and_distortion_score(
            num_frames, unstabilized_frames, cropped_frames)
        stability_score = self._compute_stability_score(vertex_stabilized_displacements_by_frame_index)

        # 写输出视频
        self._write_stabilized_video(output_path, num_frames, frames_per_second, cropped_frames)

        # 图窗展示原视频与稳定视频
        if self.visualize:
            self._display_unstablilized_and_cropped_video_loop(num_frames, frames_per_second, unstabilized_frames, cropped_frames)

        return (cropping_ratio, distortion_score, stability_score)


    ##  读入视频 获得视频序列以及参数  ##
    def _get_unstabilized_frames_and_video_features(self, input_path):
        '''
        Helper method for stabilize.
        Return each frame of the input video as a NumPy array along with miscellaneous video
        features.

        Input:

        * input_path: The path to the unstabilized video.

        Output:

        A tuple of the following items in order.

        * unstabilized_frames: A list of the frames in the unstabilized video, each represented as a
            NumPy array.
        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * codec: The video codec.
        '''

        unstabilized_video = cv2.VideoCapture(input_path)
        # num_frames = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames = np.int32(unstabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = unstabilized_video.get(cv2.CAP_PROP_FPS)

        with tqdm.trange(num_frames) as t:
            t.set_description(f'Reading video from <{input_path}>')

            unstabilized_frames = []
            for frame_index in t:
                success, pixels = unstabilized_video.read()
                if success:
                    # 规范分辨率
                    unstabilized_frame = cv2.resize(pixels, (W, H))
                    # unstabilized_frame = cv2.resize(pixels, (640,360))
                else:
                    print('capture error')
                    exit()
                if unstabilized_frame is None:
                    raise IOError(
                        f'Video at <{input_path}> did not have frame {frame_index} of '
                        f'{num_frames} (indexed from 0).'
                    )
                unstabilized_frames.append(unstabilized_frame)

        unstabilized_video.release()

        return (unstabilized_frames, num_frames, frames_per_second)


    ##  获取各顶点偏移量以及帧间的全局H变换  ##
    def _get_unstabilized_vertex_displacements_and_homographies(self, num_frames, unstabilized_frames):
        '''
        Helper method for stabilize.
        Return the displacements for the given unstabilized frames.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.

        Output:

        A tuple of the following items in order.

        * vertex_unstabilized_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.mesh_row_count, self.mesh_col_count, 2)
            containing the unstabilized displacements of each vertex in the MeshFlow mesh.
            In particular,
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][0]
            contains the x-displacement of the mesh vertex at the given row and col from frame 0 to
            frame frame_index, both inclusive.
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
        * homographies: A NumPy array of shape
            (num_frames, 3, 3)
            containing global homographies between frames.
            In particular, homographies[frame_index] contains a homography matrix between frames
            frame_index and frame_index + 1 (that is, the homography to construct frame_index + 1).
            Since no frame comes after num_frames - 1, homographies[num_frames-1] is the identity homography.
        '''

        vertex_unstabilized_displacements_by_frame_index = np.empty((num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        vertex_unstabilized_displacements_by_frame_index[0].fill(0)

        homographies = np.empty((num_frames, 3, 3))
        homographies[-1] = np.identity(3)

        # 即数组中从0遍历到num_frame-2
        with tqdm.trange(num_frames - 1) as t:
            t.set_description('Computing unstabilized mesh displacements')
            for current_index in t:
                current_frame, next_frame = unstabilized_frames[current_index:current_index + 2]  # 取current_index与current_index+1
                # 获取各顶点运动向量以及全局单应矩阵
                current_velocity, homography = self._get_unstabilized_vertex_velocities(current_frame, next_frame)
                # 逐帧叠加，第一帧置0
                vertex_unstabilized_displacements_by_frame_index[current_index + 1] = vertex_unstabilized_displacements_by_frame_index[current_index] + current_velocity
                # 最后一帧置单位阵
                homographies[current_index] = homography

        return (vertex_unstabilized_displacements_by_frame_index, homographies)

    
    ##
    def _get_stitch_vertex_displacements_and_homographies(self, num_frames, unstabilized_frames_1, unstabilized_frames_2):

        left_stitch_vertex_displacements_by_frame_index = np.empty((num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        right_stitch_vertex_displacements_by_frame_index = np.empty((num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2))

        # 即数组中从0遍历到num_frame
        with tqdm.trange(num_frames) as t:
            t.set_description('Computing stitch mesh displacements')
            for current_index in t:
                left_frame = unstabilized_frames_1[current_index]
                right_frame = unstabilized_frames_2[current_index]
                # 获取各顶点运动向量以及全局单应矩阵，这里的middle是左右特征点的平均值，左右特征点的单应矩阵分别计算其和middle的单应矩阵
                left_features, right_features, middle_features, early_to_late_homography_l, early_to_late_homography_r = self.get_matched_features_and_homography_for_stitch(left_frame, right_frame) 
                # 生成拼接顶点运动场
                left_velocity, _ = self.get_unstabilized_vertex_velocities_for_stitch(left_frame, left_features, middle_features, early_to_late_homography_l)
                right_velocity, _ = self.get_unstabilized_vertex_velocities_for_stitch(left_frame, right_features, middle_features, early_to_late_homography_r)



                left_stitch_vertex_displacements_by_frame_index[current_index] = left_velocity
                right_stitch_vertex_displacements_by_frame_index[current_index] = right_velocity

        return left_stitch_vertex_displacements_by_frame_index, right_stitch_vertex_displacements_by_frame_index
    ##


    ##  获取中值滤波后的帧间顶点位移(MeshFlow Vertex Profiles)及全局H变换  ##
    def _get_unstabilized_vertex_velocities(self, early_frame, late_frame): 
        '''
        Helper method for _get_unstabilized_vertex_displacements_and_homographies.

        Given two adjacent frames (the "early" and "late" frames), estimate the velocities of the
        vertices in the early frame.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.

        Output:

        A tuple of the following items in order.

        * mesh_velocities: A NumPy array of shape
            (mesh_row_count + 1, mesh_col_count + 1, 2)
            where the entry mesh_velocities[row][col][0]
            contains the x-velocity of the mesh vertex at the given row and col during early_frame,
            and mesh_velocities[row][col][1] contains the corresponding y-velocity.
            NOTE since time is discrete and in units of frames, a vertex's velocity during
            early_frame is the same as its displacement from early_frame to late_frame.
        * early_to_late_homography: A NumPy array of shape (3, 3) representing the homography
            between early_frame and late_frame.
        '''

        # applying this homography to a coordinate in the early frame maps it to where it will be
        # in the late frame, assuming the point is not undergoing motion

        # 两帧对应特征点以及帧间单应矩阵
        early_features, late_features, early_to_late_homography = self._get_matched_features_and_homography(early_frame, late_frame) 

        # Each vertex started in the early frame at a position given by vertex_x_y_by_row_coland.
        # If it has no velocity relative to the scene (i.e., the vertex is shaking with it), then to
        # get its position in the late frame, we apply early_to_late_homography to its early
        # position.
        # The displacement between these positions is its global motion.

        frame_height, frame_width = early_frame.shape[:2]

        # 取各顶点图像坐标，结构((mesh_row_count + 1)*(mesh_col_count + 1), 1, 2)
        vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)
        # 各顶点由全局运动引起的运动矢量
        vertex_global_velocities = cv2.perspectiveTransform(vertex_x_y, early_to_late_homography) - vertex_x_y
        # 变换矩阵结构reshape2(mesh_row_count + 1, mesh_col_count + 1, 2)
        vertex_global_velocities_by_row_col = np.reshape(vertex_global_velocities, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        vertex_global_x_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 0]
        vertex_global_y_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 1]

        # In addition to the above motion (which moves each vertex to its spot in the mesh in
        # late_frame), each vertex may undergo additional residual motion to match its nearby
        # features.
        # After gathering these velocities, perform first median filter:
        # sort each vertex's velocities by x-component, then by y-component, and use the median
        # element as the vertex's velocity.

        # 获取各顶点在约定范围(椭圆)内由各特征点引起的运动向量
        vertex_nearby_feature_residual_x_velocities_by_row_col, vertex_nearby_feature_residual_y_velocities_by_row_col = self._get_vertex_nearby_feature_residual_velocities(
            frame_width, frame_height, early_features, late_features, early_to_late_homography)
        
        # 各顶点累积的运动向量取中位数放回
        vertex_residual_x_velocities_by_row_col = np.array([
            [
                statistics.median(x_velocities)
                if x_velocities else 0
                for x_velocities in row
            ]
            for row in vertex_nearby_feature_residual_x_velocities_by_row_col
        ])
        vertex_residual_y_velocities_by_row_col = np.array([
            [
                statistics.median(y_velocities)
                if y_velocities else 0
                for y_velocities in row
            ]
            for row in vertex_nearby_feature_residual_y_velocities_by_row_col
        ])

        # 累加全局运动量
        vertex_x_velocities_by_row_col = (vertex_global_x_velocities_by_row_col + vertex_residual_x_velocities_by_row_col).astype(np.float32)
        vertex_y_velocities_by_row_col = (vertex_global_y_velocities_by_row_col + vertex_residual_y_velocities_by_row_col).astype(np.float32)

        # Perform second median filter:
        # replace each vertex's velocity with the median velocity of its neighbors.
        # 顶点运动向量中值滤波
        vertex_smoothed_x_velocities_by_row_col = cv2.medianBlur(vertex_x_velocities_by_row_col, 3)
        vertex_smoothed_y_velocities_by_row_col = cv2.medianBlur(vertex_y_velocities_by_row_col, 3)

        # a = np.array([1,2,3,4,5])
        # b = np.array([6,7,8,9,10])
        # c = np.dstack((a,b))
        # >>> c
        # array([[[ 1,  6],
        #         [ 2,  7],
        #         [ 3,  8],
        #         [ 4,  9],
        #         [ 5, 10]]]
        # >>> c.shape
        # (1, 5, 2)
        # 调整数据结构
        vertex_smoothed_velocities_by_row_col = np.dstack((vertex_smoothed_x_velocities_by_row_col, vertex_smoothed_y_velocities_by_row_col))

        return (vertex_smoothed_velocities_by_row_col, early_to_late_homography)


    ##  用于拼接的运动场生成
    def get_unstabilized_vertex_velocities_for_stitch(self, early_frame, early_features, late_features, early_to_late_homography): 

        # applying this homography to a coordinate in the early frame maps it to where it will be
        # in the late frame, assuming the point is not undergoing motion 

        # Each vertex started in the early frame at a position given by vertex_x_y_by_row_coland.
        # If it has no velocity relative to the scene (i.e., the vertex is shaking with it), then to
        # get its position in the late frame, we apply early_to_late_homography to its early
        # position.
        # The displacement between these positions is its global motion.

        frame_height, frame_width = early_frame.shape[:2]

        # 取各顶点图像坐标，结构((mesh_row_count + 1)*(mesh_col_count + 1), 1, 2)
        vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)
        # 各顶点由全局运动引起的运动矢量
        vertex_global_velocities = cv2.perspectiveTransform(vertex_x_y, early_to_late_homography) - vertex_x_y
        # 变换矩阵结构reshape2(mesh_row_count + 1, mesh_col_count + 1, 2)
        vertex_global_velocities_by_row_col = np.reshape(vertex_global_velocities, (mesh_row_count + 1, mesh_col_count + 1, 2))
        vertex_global_x_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 0]
        vertex_global_y_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 1]

        # In addition to the above motion (which moves each vertex to its spot in the mesh in
        # late_frame), each vertex may undergo additional residual motion to match its nearby
        # features.
        # After gathering these velocities, perform first median filter:
        # sort each vertex's velocities by x-component, then by y-component, and use the median
        # element as the vertex's velocity.

        # 获取各顶点在约定范围(椭圆)内由各特征点引起的运动向量，先计算特征点的残余运动矢量，然后通过椭圆传播到顶点，然后输出顶点的残余运动矢量
        vertex_nearby_feature_residual_x_velocities_by_row_col, vertex_nearby_feature_residual_y_velocities_by_row_col = self._get_vertex_nearby_feature_residual_velocities(
            frame_width, frame_height, early_features, late_features, early_to_late_homography)

        # for row in vertex_nearby_feature_residual_x_velocities_by_row_col:
        #     for col in row:
        #         print(len(col))
        # print(vertex_nearby_feature_residual_x_velocities_by_row_col[7][8])
        # print(vertex_nearby_feature_residual_x_velocities_by_row_col[7][9])
        # 各顶点累积的运动向量取中位数放回
        vertex_residual_x_velocities_by_row_col = np.array([
            [
                statistics.median(x_velocities)
                if x_velocities else 0
                for x_velocities in row
            ]
            for row in vertex_nearby_feature_residual_x_velocities_by_row_col
        ])
        # print(vertex_residual_x_velocities_by_row_col)
        vertex_residual_y_velocities_by_row_col = np.array([
            [
                statistics.median(y_velocities)
                if y_velocities else 0
                for y_velocities in row
            ]
            for row in vertex_nearby_feature_residual_y_velocities_by_row_col
        ])

        # 累加全局运动量
        vertex_x_velocities_by_row_col = (vertex_global_x_velocities_by_row_col + vertex_residual_x_velocities_by_row_col).astype(np.float32)
        vertex_y_velocities_by_row_col = (vertex_global_y_velocities_by_row_col + vertex_residual_y_velocities_by_row_col).astype(np.float32)

        # Perform second median filter:
        # replace each vertex's velocity with the median velocity of its neighbors.
        # 顶点运动向量中值滤波
        # vertex_smoothed_x_velocities_by_row_col = cv2.medianBlur(vertex_x_velocities_by_row_col, 3)
        # vertex_smoothed_y_velocities_by_row_col = cv2.medianBlur(vertex_y_velocities_by_row_col, 3)

        # 调整数据结构
        # vertex_smoothed_velocities_by_row_col = np.dstack((vertex_smoothed_x_velocities_by_row_col, vertex_smoothed_y_velocities_by_row_col))
        vertex_smoothed_velocities_by_row_col = np.dstack((vertex_x_velocities_by_row_col, vertex_y_velocities_by_row_col))
        # print(vertex_smoothed_velocities_by_row_col.shape)
        return (vertex_smoothed_velocities_by_row_col, early_to_late_homography)
    ##


    ##
    def get_matched_features_and_homography_for_stitch(self, early_frame, late_frame):

        # get features that have had outliers removed by applying homographies to sub-frames
        frame_height, frame_width = early_frame.shape[:2]
        # # 局部网格宽高
        # subframe_width = math.ceil(frame_width / mesh_outlier_subframe_col_count)
        # subframe_height = math.ceil(frame_height / mesh_outlier_subframe_row_count)

        # early_features_by_subframe[i] contains a CV_32FC2 array of the early features in the
        # frame's i^th subframe;
        # late_features_by_subframe[i] is defined similarly
        # 依次存储局部网格中的特征点
        # early_features_by_subframe = []
        # late_features_by_subframe = []

        common_field = 800

        # # TODO parallelize
        # for subframe_left_x in range(0, frame_width, subframe_width):
        #     for subframe_top_y in range(0, frame_height, subframe_height):
        # 依次取局部网格，顺序先行后列
        early_subframe = early_frame[:, (frame_width - common_field): frame_width]
        late_subframe = late_frame[:, 0: common_field]
        # # 局部网格左上角坐标
        # subframe_offset = [subframe_left_x, subframe_top_y]
        subframe_offset = [(frame_width - common_field), 0]
        #此时early_features已经增加了偏移
        early_features, late_features = self.loftr_for_stitch(early_subframe, late_subframe, subframe_offset)
        # subframe_early_features, subframe_late_features = get_features_in_subframe(early_subframe, late_subframe, subframe_offset)
        # 特征点存入数组
        # if subframe_early_features is not None:
        #     early_features_by_subframe.append(subframe_early_features)
        # if subframe_late_features is not None:
        #     late_features_by_subframe.append(subframe_late_features)


        # early_features, late_features = loftr_in_subframe(early_subframe, late_subframe, subframe_offset)
        # 数组拼接(默认axis=0,纵向拼接)，获得全局特征点
        # early_features = np.concatenate(early_features_by_subframe, axis=0)
        # late_features = np.concatenate(late_features_by_subframe, axis=0)

        middle_points = (early_features + late_features) / 2

        if len(early_features) < 12:
            print('No enough features!')
            return (None, None, None)

        # 获取单应性矩阵 MAGSAC替代RANSAC
        # early_to_late_homography, _ = cv2.findHomography(early_features, late_features)
        early_to_late_homography_l, _ = cv2.findHomography(early_features, middle_points, cv2.USAC_MAGSAC)
        early_to_late_homography_r, _ = cv2.findHomography(late_features, middle_points, cv2.USAC_MAGSAC)

        return (early_features, late_features, middle_points, early_to_late_homography_l, early_to_late_homography_r)
    ##


    ##  获取各顶点对应图像坐标  ##
    def _get_vertex_x_y(self, frame_width, frame_height):
        '''
        Helper method for _get_stabilized_frames_and_crop_boundaries_and_crop_boundaries and _get_unstabilized_vertex_velocities.
        Return a NumPy array that maps [row, col] coordinates to [x, y] coordinates.

        Input:

        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.

        Output:

        row_col_to_vertex_x_y: A CV_32FC2 array (see https://stackoverflow.com/a/47617999)
            containing the coordinates [x, y] of vertices in the mesh. This array is ordered
            so that when this array is reshaped to
            (self.mesh_row_count + 1, self.mesh_col_count + 1, 2),
            the resulting entry in [row, col] contains the coordinates [x, y] of the vertex in the
            top left corner of the cell at the mesh's given row and col.
        '''
        # CV_32FC2 32位浮点型双通道矩阵
        # points = np.array([(1, 2), (2, 3)])
        # points = np.float32(points[:, np.newaxis, :])
        # array([[[ 1.,  2.]],
        #        [[ 2.,  3.]], dtype=float32)
        # ceil 向上取整
        # [[[0. , 0.]],
        #  [[a. , 0.]],
        #  ...
        #  [[na. , 0.]],
        #  [[0. , b.]],
        #  [[a. , b.]],
        #  ...
        # ]
        # 维度((mesh_row_count + 1)*(mesh_col_count + 1), 1, 2)

        return np.array([
            [[math.ceil((frame_width - 1) * (col / (self.mesh_col_count))), math.ceil((frame_height - 1) * (row / (self.mesh_row_count)))]]
            for row in range(self.mesh_row_count + 1)
            for col in range(self.mesh_col_count + 1)
        ], dtype=np.float32)
    
    ##  获取各顶点在约定范围(椭圆)内由各特征点引起的运动向量  ##
    def _get_vertex_nearby_feature_residual_velocities(self, frame_width, frame_height, early_features, late_features, early_to_late_homography): 
        '''
        Helper method for _get_unstabilized_vertex_velocities.

        Given two adjacent frames, return a list that maps each vertex in the mesh to the residual
        velocities of its nearby features.

        Input:

        * frame_width: the width of the windows' frames.
        * frame_height: the height of the windows' frames.
        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in early_subframe that was
            successfully tracked in late_subframe. These coordinates are expressed relative to the
            frame, not the window. If fewer than self.homography_min_number_corresponding_features such 
            features were found, early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in late_subframe that was
            successfully tracked from early_subframe. These coordinates are expressed relative to the
            frame, not the window. If fewer than self.homography_min_number_corresponding_features such 
            features were found, late_features is None.
        * early_to_late_homography: A homography matrix that maps a point in early_frame to its
            corresponding location in late_frame, assuming the point is not undergoing motion

        Output:

        A tuple of the following items in order.

        * vertex_nearby_feature_x_velocities_by_row_col: A list
            where entry vertex_nearby_feature_x_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.
        * vertex_nearby_feature_y_velocities_by_row_col: A list
            where entry vertex_nearby_feature_y_velocities_by_row_col[row, col] contains a list of
            the y-velocities of all the features nearby the vertex at the given row and col.
        '''
        #  初始化数组
        vertex_nearby_feature_x_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]
        vertex_nearby_feature_y_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]

        if early_features is not None:
            # calculate features' velocities; see https://stackoverflow.com/a/44409124 for
            # combining the positions and velocities into one matrix

            # If a point were undergoing no motion, then its position in the late frame would be
            # found by applying early_to_late_homography to its position in the early frame.
            # The point's additional motion is what takes it from that position to its actual
            # position.

            # 计算特征点自身的运动(去除全局运动) v_p~ = p - F_t*p^
            feature_residual_velocities = late_features - cv2.perspectiveTransform(early_features, early_to_late_homography)

            # 数据类型转换
            # a = array([1, 2, 3])
            # b = array([ 6, 7, 8])
            # >>> np.c_[a, b]
            # array([[ 1,  6],
            #        [ 2,  7],
            #        [ 3,  8]])
            # 首列存当前帧特征点的位置 次列存帧间相对运动
            feature_positions_and_residual_velocities = np.c_[late_features, feature_residual_velocities]

            # apply features' velocities to nearby mesh vertices
            for feature_position_and_residual_velocity in feature_positions_and_residual_velocities:
                feature_x, feature_y, feature_residual_x_velocity, feature_residual_y_velocity = feature_position_and_residual_velocity[0]
                feature_row = (feature_y / frame_height) * self.mesh_row_count
                feature_col = (feature_x / frame_width) * self.mesh_col_count

                # Draw an ellipse around each feature of width self.feature_ellipse_col_count
                # and height self.feature_ellipse_row_count,
                # and apply the feature's velocity to all mesh vertices that fall within this ellipse.
                # To do this, we can iterate through all the rows that the ellipse covers.
                # For each row, we can use the equation for an ellipse centered on the
                # feature to determine which columns the ellipse covers. The resulting
                # (row, column) pairs correspond to the vertices in the ellipse.

                # 顶行
                ellipse_top_row_inclusive = max(0, math.ceil(feature_row - self.feature_ellipse_row_count / 2))
                # 底行
                ellipse_bottom_row_exclusive = 1 + min(self.mesh_row_count, math.floor(feature_row + self.feature_ellipse_row_count / 2))

                for vertex_row in range(ellipse_top_row_inclusive, ellipse_bottom_row_exclusive):

                    # half-width derived from ellipse equation
                    # 计算每一行所包含的列
                    ellipse_slice_half_width = self.feature_ellipse_col_count * \
                        math.sqrt((1/4) - ((vertex_row - feature_row) / self.feature_ellipse_row_count) ** 2)
                    # 左列
                    ellipse_left_col_inclusive = max(0, math.ceil(feature_col - ellipse_slice_half_width))
                    # 右列
                    ellipse_right_col_exclusive = 1 + min(self.mesh_col_count, math.floor(feature_col + ellipse_slice_half_width))
                    # 向对应网格点中存入运动向量
                    for vertex_col in range(ellipse_left_col_inclusive, ellipse_right_col_exclusive):
                        vertex_nearby_feature_x_velocities_by_row_col[vertex_row][vertex_col].append(feature_residual_x_velocity)
                        vertex_nearby_feature_y_velocities_by_row_col[vertex_row][vertex_col].append(feature_residual_y_velocity)

        return (vertex_nearby_feature_x_velocities_by_row_col, vertex_nearby_feature_y_velocities_by_row_col)

    ##  设立局部网格，根据不同阈值获得整张图像特征点以及帧间单应性变换  ##
    def _get_matched_features_and_homography(self, early_frame, late_frame):
        '''
        Helper method for _get_unstabilized_vertex_velocities and _compute_cropping_ratio_and_distortion_score.

        Detect features in the early window using the MeshFlowStabilizer's feature_detector
        and track them into the late window using cv2.calcOpticalFlowPyrLK.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.

        Output:

        A tuple of the following items in order.

        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in early_frame that was
            successfully tracked in late_frame. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array of positions containing the coordinates of each feature in late_frame 
            that was successfully tracked from early_frame. These coordinates are expressed relative to the
            window. If fewer than self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        * early_to_late_homography: A homography matrix that maps a point in early_frame to its
            corresponding location in late_frame, assuming the point is not undergoing motion.
            If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_to_late_homography is None.
        '''

        # get features that have had outliers removed by applying homographies to sub-frames

        frame_height, frame_width = early_frame.shape[:2]
        # 局部网格宽高
        subframe_width = math.ceil(frame_width / self.mesh_outlier_subframe_col_count)
        subframe_height = math.ceil(frame_height / self.mesh_outlier_subframe_row_count)

        # early_features_by_subframe[i] contains a CV_32FC2 array of early features in frame's i^th subframe
        # late_features_by_subframe[i] is defined similarly
        # 依次存储局部网格中的特征点
        early_features_by_subframe = []
        late_features_by_subframe = []
        # feature_in_bbox = []             # BBox内特征点的索引

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
                subframe_early_features, subframe_late_features = self._get_features_in_subframe(early_subframe, late_subframe, subframe_offset)

                # 特征点存入数组
                if subframe_early_features is not None:
                    early_features_by_subframe.append(subframe_early_features)
                if subframe_late_features is not None:
                    late_features_by_subframe.append(subframe_late_features)

        # 数组拼接(默认axis=0,纵向拼接)，获得全局特征点
        early_features = np.concatenate(early_features_by_subframe, axis=0)
        late_features = np.concatenate(late_features_by_subframe, axis=0)
        
        ##
        # 剔除BBox内的特征点
        # startx, starty, endx, endy = sum_box[frame_idx]

        # for i in range(len(early_features)):
        #     if startx < early_features[i, 0, 0] < endx and starty < early_features[i, 0, 1] < endy:
        #         feature_in_bbox.append(i)
        
        # early_features = np.delete(early_features, feature_in_bbox, axis=0)
        # late_features = np.delete(late_features, feature_in_bbox, axis=0)
        ##

        if len(early_features) < 12:
            return (None, None, None)

        # 获取单应性矩阵
        early_to_late_homography, _ = cv2.findHomography(early_features, late_features, cv2.USAC_MAGSAC)

        return (early_features, late_features, early_to_late_homography)

    ##  获得帧间单应变换供评估cropping ratio  ##
    def _get_matched_homography_for_evaluation(self, early_frame, late_frame):

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
        # feature_in_bbox = []             # BBox内特征点的索引

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
                subframe_early_features, subframe_late_features = self._get_features_in_subframe(early_subframe, late_subframe, subframe_offset)

                # 特征点存入数组
                if subframe_early_features is not None:
                    early_features_by_subframe.append(subframe_early_features)
                if subframe_late_features is not None:
                    late_features_by_subframe.append(subframe_late_features)

        # 数组拼接(默认axis=0,纵向拼接)，获得全局特征点
        early_features = np.concatenate(early_features_by_subframe, axis=0)
        late_features = np.concatenate(late_features_by_subframe, axis=0)

        if len(early_features) < self.homography_min_number_corresponding_features:
            return (None, None, None)

        # 获取单应性矩阵
        early_to_late_homography, _ = cv2.findHomography(early_features, late_features)

        return early_to_late_homography

    ## LoFTR方法获取局部网格内的特征点坐标  ##
    def loftr_in_subframe(self, early_subframe, late_subframe, subframe_offset):

        import kornia as K
        import kornia.feature as KF
        import torch

        img1 = K.image_to_tensor(early_subframe, False).float() /255.
        img1 = K.color.bgr_to_rgb(img1)

        img2 = K.image_to_tensor(late_subframe, False).float() /255.
        img2 = K.color.bgr_to_rgb(img2)

        matcher = KF.LoFTR(pretrained='outdoor')
        matcher.cuda()
        matcher.eval()

        input_dict = {"image0": K.color.rgb_to_grayscale(img1.cuda()), # LofTR works on grayscale images only 
                    "image1": K.color.rgb_to_grayscale(img2.cuda())}

        with torch.no_grad():
            correspondences = matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.4, 0.999, 100000)

        filter_mask = inliers.flatten().astype(dtype = bool)
        mkpts0_filtered = mkpts0[filter_mask]
        mkpts1_filtered = mkpts1[filter_mask]

        mkpts0_filtered = np.float32(mkpts0_filtered[:, np.newaxis, :])
        mkpts1_filtered = np.float32(mkpts1_filtered[:, np.newaxis, :])

        return (mkpts0_filtered + subframe_offset, mkpts1_filtered + subframe_offset)


    ##
    def loftr_for_stitch(self, early_subframe, late_subframe, subframe_offset):

        import kornia as K
        import kornia.feature as KF
        import torch

        img1 = K.image_to_tensor(early_subframe, False).float() /255.
        img1 = K.color.bgr_to_rgb(img1)

        img2 = K.image_to_tensor(late_subframe, False).float() /255.
        img2 = K.color.bgr_to_rgb(img2)

        matcher = KF.LoFTR(pretrained='outdoor')
        matcher.cuda()
        matcher.eval()

        input_dict = {"image0": K.color.rgb_to_grayscale(img1.cuda()), # LofTR works on grayscale images only 
                    "image1": K.color.rgb_to_grayscale(img2.cuda())}

        with torch.no_grad():
            correspondences = matcher(input_dict)

        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.4, 0.999, 100000)

        filter_mask = inliers.flatten().astype(dtype = bool)
        mkpts0_filtered = mkpts0[filter_mask]
        mkpts1_filtered = mkpts1[filter_mask]

        mkpts0_filtered = np.float32(mkpts0_filtered[:, np.newaxis, :])
        mkpts1_filtered = np.float32(mkpts1_filtered[:, np.newaxis, :])

        return (mkpts0_filtered + subframe_offset, mkpts1_filtered)
    ##


    ##  RANSAC去噪并获得局部网格内的特征点坐标(相对于整张图像的全局坐标)  ##
    def _get_features_in_subframe(self, early_subframe, late_subframe, subframe_offset):
        '''
        Helper method for _get_matched_features_and_homography.
        Track and return features that appear between the two given frames, eliminating outliers
        by applying a homography using RANSAC.

        Input:

        * early_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_subframe.
        * late_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_subframe.
        * offset_location: A tuple (x, y) representing the offset of the subframe within its frame,
            relative to the frame's top left corner.

        Output:

        A tuple of the following items in order.
        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in early_subframe that was
            successfully tracked in late_subframe. These coordinates are expressed relative to the
            frame, not the subframe. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in late_subframe that was
            successfully tracked from early_subframe. These coordinates are expressed relative to the
            frame, not the subframe. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        '''

        # gather all features that track between frames
        early_features_including_outliers, late_features_including_outliers = self._get_all_matched_features_between_subframes(early_subframe, late_subframe)
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
    def _get_all_matched_features_between_subframes(self, early_subframe, late_subframe):
        '''
        Helper method for _get_features_in_subframe.
        Detect features in the early subframe using the MeshFlowStabilizer's feature_detector
        and track them into the late subframe using cv2.calcOpticalFlowPyrLK.

        Input:

        * early_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_subframe.
        * late_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_subframe.

        Output:

        A tuple of the following items in order.
        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in early_subframe that was
            successfully tracked in late_subframe. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in late_subframe that was
            successfully tracked from early_subframe. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        '''

        # convert a KeyPoint list into a CV_32FC2 array containing the coordinates of each KeyPoint;
        # see https://stackoverflow.com/a/55398871 and https://stackoverflow.com/a/47617999

        # FAST特征点检测
        feature_detector = cv2.FastFeatureDetector_create()
        early_keypoints = feature_detector.detect(early_subframe)
        if len(early_keypoints) < self.homography_min_number_corresponding_features:
            return (None, None)
        
        # 从KeyPoint格式中提取出(x,y)坐标，并转化为CV_32FC2
        # >>> points = np.array([(1, 2), (2, 3), (3, 4)])
        # >>> points = np.float32(points[:, np.newaxis, :])
        # >>> points
        # array([[[ 1.,  2.]],
        #        [[ 2.,  3.]],
        #        [[ 3.,  4.]]], dtype=float32)
        early_features_including_unmatched = np.float32(cv2.KeyPoint_convert(early_keypoints)[:, np.newaxis, :])

        # 光流法跟踪前帧的特征点
        late_features_including_unmatched, matched_features, _ = cv2.calcOpticalFlowPyrLK(early_subframe, late_subframe, 
                                                                                          early_features_including_unmatched, None)

        # >>> a=array([[1,2],[3,4],[5,6]])
        # >>> a.flatten()
        # array([1, 2, 3, 4, 5, 6])
        # 创建掩膜，仅提取出匹配完成的特征点
        matched_features_mask = matched_features.flatten().astype(dtype=bool)
        early_features = early_features_including_unmatched[matched_features_mask]
        late_features = late_features_including_unmatched[matched_features_mask]

        if len(early_features) < self.homography_min_number_corresponding_features:
            return (None, None)

        return (early_features, late_features)


    ##  使用雅可比方法 获得稳定后的各帧顶点运动向量 其数据格式与未稳定类似 (num_frames, row + 1, col + 1, 2)
    def _get_stabilized_vertex_displacements(self, num_frames, unstabilized_frames, adaptive_weights_definition, vertex_unstabilized_displacements_by_frame_index, homographies):
        '''
        Helper method for stabilize.

        Return each vertex's displacement at each frame in the stabilized video.

        Specifically, find the displacements that minimize an energy function.
        The energy function takes displacements as input and outputs a number corresponding
        to how shaky the input is.

        The output array of stabilized displacements is calculated using the
        Jacobi method. For each mesh vertex, the method solves the equation
        A p = b
        for vector p,
        where entry p[i] contains the vertex's stabilized displacement at frame i.
        The entries in matrix A and vector b were derived by finding the partial derivative of the
        energy function with respect to each p[i] and setting them all to 0. Thus, solving for p in
        A p = b results in displacements that produce a local extremum (which we can safely
        assume is a local minimum) in the energy function.

        Input:

        * num_frames: The number of frames in the video.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive weights.
        * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
            unstabilized displacements of each vertex in the MeshFlow mesh, as outputted
            by _get_unstabilized_vertex_displacements_and_homographies.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        * vertex_stabilized_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.mesh_row_count, self.mesh_col_count, 2)
            containing the stabilized displacements of each vertex in the MeshFlow mesh.
            In particular,
            vertex_stabilized_displacements_by_frame_index[frame_index][row][col][0]
            contains the x-displacement (the x-displacement in addition to any imposed by
            global homographies) of the mesh vertex at the given row and col from frame 0 to frame
            frame_index, both inclusive.
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
        '''
        ##
        frame_height, frame_width = unstabilized_frames[0].shape[:2]
        # 非对角系数 对角系数
        off_diagonal_coefficients, on_diagonal_coefficients = self._get_jacobi_method_input(
            num_frames, frame_width, frame_height, adaptive_weights_definition, homographies)
        ##

        # vertex_unstabilized_displacements_by_frame_index is indexed by
        # frame_index, then row, then col, then velocity component.
        # Instead, vertex_unstabilized_displacements_by_coord is indexed by
        # row, then col, then frame_index, then velocity component;
        # this rearrangement should allow for faster access during the optimization step.
        # 坐标轴0位置换坐标轴2 其他轴位置不变 (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
        vertex_unstabilized_displacements_by_coord = np.moveaxis(
            vertex_unstabilized_displacements_by_frame_index, 0, 2)
        
        # 初始化稳定后的顶点位移
        vertex_stabilized_displacements_by_coord = np.empty(
            vertex_unstabilized_displacements_by_coord.shape)

        # TODO parallelize
        with tqdm.trange((self.mesh_row_count + 1) * (self.mesh_col_count + 1)) as t:
            t.set_description('Computing stabilized mesh displacements')
            for mesh_coords_flattened in t:
                # 先列后行 (0,0),(0,1),...,(1,0),(1,1),...
                ##
                mesh_row = mesh_coords_flattened // (self.mesh_col_count + 1)   # 商向前取整
                mesh_col = mesh_coords_flattened % (self.mesh_col_count + 1)    # 取余
                ##
                # 取出对应的 (num_frames, 2)
                vertex_unstabilized_displacements = vertex_unstabilized_displacements_by_coord[mesh_row][mesh_col]
                vertex_stabilized_displacements = self._get_jacobi_method_output(
                    off_diagonal_coefficients, on_diagonal_coefficients,
                    vertex_unstabilized_displacements,
                    vertex_unstabilized_displacements
                )
                vertex_stabilized_displacements_by_coord[mesh_row][mesh_col] = vertex_stabilized_displacements

            # 还原
            vertex_stabilized_displacements_by_frame_index = np.moveaxis(
                vertex_stabilized_displacements_by_coord, 2, 0
            )

        return vertex_stabilized_displacements_by_frame_index


    ##
    def _get_unified_stabilized_vertex_displacements(self, num_frames, vertex_for_stitch,
                                                     vertex_unstabilized_displacements_by_frame_index, vertex_stabilized_displacements_by_frame_index):
            ##
            # frame_height, frame_width = unstabilized_frames[0].shape[:2]
            # 非对角系数 对角系数
            off_diagonal_coefficients, on_diagonal_coefficients = self._get_jacobi_method_input_for_stitch(num_frames)
            ##

            # vertex_unstabilized_displacements_by_frame_index is indexed by
            # frame_index, then row, then col, then velocity component.
            # Instead, vertex_unstabilized_displacements_by_coord is indexed by
            # row, then col, then frame_index, then velocity component;
            # this rearrangement should allow for faster access during the optimization step.
            # 坐标轴0位置换坐标轴2 其他轴位置不变 (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
            vertex_for_stitch_coord = np.moveaxis(vertex_for_stitch, 0, 2)
            vertex_for_start = vertex_for_stitch + vertex_unstabilized_displacements_by_frame_index - vertex_stabilized_displacements_by_frame_index
            vertex_for_start_coord = np.moveaxis(vertex_for_start, 0, 2)
                      
            # 初始化稳定后的顶点位移
            vertex_stabilized_displacements_by_coord = np.empty(vertex_for_stitch_coord.shape)

            # TODO parallelize
            with tqdm.trange((self.mesh_row_count + 1) * (self.mesh_col_count + 1)) as t:
                t.set_description('Computing unified mesh displacements')
                for mesh_coords_flattened in t:
                    # 先列后行 (0,0),(0,1),...,(1,0),(1,1),...
                    ##
                    mesh_row = mesh_coords_flattened // (self.mesh_col_count + 1)   # 商向前取整
                    mesh_col = mesh_coords_flattened % (self.mesh_col_count + 1)    # 取余
                    ##
                    # 取出对应的 (num_frames, 2)
                    # vertex_unstabilized_displacements = vertex_unstabilized_displacements_by_coord[mesh_row][mesh_col]
                    vertex_for_stitch_displacements = vertex_for_stitch_coord[mesh_row][mesh_col]
                    vertex_for_start = vertex_for_start_coord[mesh_row][mesh_col]
                    vertex_stabilized_displacements = self._get_jacobi_method_output(
                        off_diagonal_coefficients, on_diagonal_coefficients,
                        vertex_for_stitch_displacements, vertex_for_start)
                    vertex_stabilized_displacements_by_coord[mesh_row][mesh_col] = vertex_stabilized_displacements

                # 还原
                vertex_stabilized_displacements_by_frame_index = np.moveaxis(
                    vertex_stabilized_displacements_by_coord, 2, 0)

            return vertex_stabilized_displacements_by_frame_index
    
    ##
    def _get_stitch_vertex_displacements(self, num_frames, vertex_for_stitch):
        ##
        # frame_height, frame_width = unstabilized_frames[0].shape[:2]
        # 非对角系数 对角系数
        off_diagonal_coefficients, on_diagonal_coefficients = self._get_jacobi_method_input_for_stitch(num_frames)
        ##

        # vertex_unstabilized_displacements_by_frame_index is indexed by
        # frame_index, then row, then col, then velocity component.
        # Instead, vertex_unstabilized_displacements_by_coord is indexed by
        # row, then col, then frame_index, then velocity component;
        # this rearrangement should allow for faster access during the optimization step.
        # 坐标轴0位置换坐标轴2 其他轴位置不变 (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
        vertex_for_stitch_coord = np.moveaxis(vertex_for_stitch, 0, 2)
        vertex_for_start = vertex_for_stitch
        vertex_for_start_coord = np.moveaxis(vertex_for_start, 0, 2)
                    
        # 初始化稳定后的顶点位移
        vertex_stabilized_displacements_by_coord = np.empty(vertex_for_stitch_coord.shape)

        # TODO parallelize
        with tqdm.trange((self.mesh_row_count + 1) * (self.mesh_col_count + 1)) as t:
            t.set_description('Computing unified mesh displacements')
            for mesh_coords_flattened in t:
                # 先列后行 (0,0),(0,1),...,(1,0),(1,1),...
                ##
                mesh_row = mesh_coords_flattened // (self.mesh_col_count + 1)   # 商向前取整
                mesh_col = mesh_coords_flattened % (self.mesh_col_count + 1)    # 取余
                ##
                # 取出对应的 (num_frames, 2)
                # vertex_unstabilized_displacements = vertex_unstabilized_displacements_by_coord[mesh_row][mesh_col]
                vertex_for_stitch_displacements = vertex_for_stitch_coord[mesh_row][mesh_col]
                vertex_for_start = vertex_for_start_coord[mesh_row][mesh_col]
                vertex_stabilized_displacements = self._get_jacobi_method_output(
                    off_diagonal_coefficients, on_diagonal_coefficients,
                    vertex_for_stitch_displacements, vertex_for_start)
                vertex_stabilized_displacements_by_coord[mesh_row][mesh_col] = vertex_stabilized_displacements

            # 还原
            vertex_stabilized_displacements_by_frame_index = np.moveaxis(
                vertex_stabilized_displacements_by_coord, 2, 0)

        return vertex_stabilized_displacements_by_frame_index

    def _get_jacobi_method_input_for_stitch(self, num_frames):

        # row_indexes[row][col] = row, col_indexes[row][col] = col
        # 返回一个表示索引的数组 维度为输入视频的帧总数
        row_indexes, col_indexes = np.indices((num_frames, num_frames))

        # regularization_weights[t, r] is a weight constant applied to the regularization term.
        # In the paper, regularization_weights[t, r] is denoted as w_{t,r}.
        # NOTE that regularization_weights[i, i] = 0.
        # row_indexes - col_indexes
        # [[ 0 -1 -2 -3]
        #  [ 1  0 -1 -2]
        #  [ 2  1  0 -1]
        #  [ 3  2  1  0]]
        # \omega_(t,r)
        regularization_weights = np.exp(-np.square((3 / self.temporal_smoothing_radius)* (row_indexes - col_indexes)))

        # adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
        # regularization term corresponding to the frame at index t
        # Note that the paper does not specify the weight to apply to the last frame (which does not
        # have a velocity), so we assume it is the same as the second-to-last frame.
        # In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        # adaptive_weights = self._get_adaptive_weights(num_frames, frame_width, frame_height, adaptive_weights_definition, 
        #                                               homographies)
        adaptive_weights = np.full((num_frames), 2)
        # combined_adaptive_regularization_weights[t, r] = \lambda_{t} w_{t, r}
        # adaptive_weights转化为对角阵，矩阵相乘
        # [[2 0 0]    [[0 1 4]    [[0 2 8]
        #  [0 2 0]  *  [1 0 1]  =  [2 0 2]
        #  [0 0 2]]    [4 1 0]]    [8 2 0]] 
        combined_adaptive_regularization_weights = np.matmul(np.diag(adaptive_weights), regularization_weights)

        # the off-diagonal entry at cell [t, r] is written as  -2 * \lambda_{t} w_{t, r}
        off_diagonal_coefficients = -2 * combined_adaptive_regularization_weights

        # the on-diagonal entry at cell [t, t] is written as
        # 1 + 2 * \sum_{r \in \Omega_{t}, r \neq t} \lambda_{t} w_{t, r}.
        # NOTE Since w_{t, t} = 0,
        # we can ignore the r \neq t constraint on the sum and write the on-diagonal entry at cell [t, t] as
        # 1 + 2 * \sum{r \in \Omega_{t}} \lambda_{t} w_{t, r}.
        # 按行求和 生成一维数组
        on_diagonal_coefficients = 1 + 2 * np.sum(combined_adaptive_regularization_weights, axis=1)

        # set coefficients to 0 for appropriate t, r; see https://stackoverflow.com/a/36247680
        # 将对角线上下一部分元素置1 即将各t对应r范围外的元素置零
        off_diagonal_mask = np.zeros(off_diagonal_coefficients.shape)
        for i in range(-self.temporal_smoothing_radius, self.temporal_smoothing_radius + 1):
            off_diagonal_mask += np.diag(np.ones(num_frames - abs(i)), i)
        off_diagonal_coefficients = np.where(off_diagonal_mask, off_diagonal_coefficients, 0)

        return (off_diagonal_coefficients, on_diagonal_coefficients)
    ##

    ##  获得Jacobi方法中的A矩阵 A的列方向对应整个视频序列 行对应稳定每一帧所需前后一定范围r的图像  ##
    def _get_jacobi_method_input(self, num_frames, frame_width, frame_height, adaptive_weights_definition, homographies):
        '''
        Helper method for _get_stabilized_displacements.
        The Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximates a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.
        Return the values in matrix A given the video's features and the user's chosen method.

        Input:

        * num_frames: The number of frames in the video.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive weights.
        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        A tuple of the following items in order.

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
        * on_diagonal_coefficients: A 1D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i] = A_{i, i}.
        '''

        # row_indexes[row][col] = row, col_indexes[row][col] = col
        # 返回一个表示索引的数组 维度为输入视频的帧总数
        #  np.indices((2, 3))
        #  [[0 0 0]
        #   [1 1 1]]
        #  [[0 1 2]
        #   [0 1 2]]
        row_indexes, col_indexes = np.indices((num_frames, num_frames))

        # regularization_weights[t, r] is a weight constant applied to the regularization term.
        # In the paper, regularization_weights[t, r] is denoted as w_{t,r}.
        # NOTE that regularization_weights[i, i] = 0.
        # row_indexes - col_indexes
        # [[ 0 -1 -2 -3]
        #  [ 1  0 -1 -2]
        #  [ 2  1  0 -1]
        #  [ 3  2  1  0]]
        # \omega_(t,r)
        regularization_weights = np.exp(-np.square((3 / self.temporal_smoothing_radius)* (row_indexes - col_indexes)))

        # adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
        # regularization term corresponding to the frame at index t
        # Note that the paper does not specify the weight to apply to the last frame (which does not
        # have a velocity), so we assume it is the same as the second-to-last frame.
        # In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        adaptive_weights = self._get_adaptive_weights(num_frames, frame_width, frame_height, adaptive_weights_definition, 
                                                      homographies)

        # combined_adaptive_regularization_weights[t, r] = \lambda_{t} w_{t, r}
        # adaptive_weights转化为对角阵，矩阵相乘
        # [[2 0 0]    [[0 1 4]    [[0 2 8]
        #  [0 2 0]  *  [1 0 1]  =  [2 0 2]
        #  [0 0 2]]    [4 1 0]]    [8 2 0]] 
        combined_adaptive_regularization_weights = np.matmul(np.diag(adaptive_weights), regularization_weights)

        # 以下两个方程参考文章《Bundled Camera Paths for Video Stabilization》 Page5 Eq.6
        # the off-diagonal entry at cell [t, r] is written as  -2 * \lambda_{t} w_{t, r}
        off_diagonal_coefficients = -2 * combined_adaptive_regularization_weights

        # the on-diagonal entry at cell [t, t] is written as
        # 1 + 2 * \sum_{r \in \Omega_{t}, r \neq t} \lambda_{t} w_{t, r}.
        # NOTE Since w_{t, t} = 0,
        # we can ignore the r \neq t constraint on the sum and write the on-diagonal entry at cell [t, t] as
        # 1 + 2 * \sum{r \in \Omega_{t}} \lambda_{t} w_{t, r}.
        # 按行求和 生成一维数组
        on_diagonal_coefficients = 1 + 2 * np.sum(combined_adaptive_regularization_weights, axis=1)

        # set coefficients to 0 for appropriate t, r; see https://stackoverflow.com/a/36247680
        # 将对角线上下一部分元素置1 即将各t对应r范围外的元素置零
        off_diagonal_mask = np.zeros(off_diagonal_coefficients.shape)
        for i in range(-self.temporal_smoothing_radius, self.temporal_smoothing_radius + 1):
            off_diagonal_mask += np.diag(np.ones(num_frames - abs(i)), i)
        off_diagonal_coefficients = np.where(off_diagonal_mask, off_diagonal_coefficients, 0)

        return (off_diagonal_coefficients, on_diagonal_coefficients)
    
    ##  计算\Lambda_t  ##
    def _get_adaptive_weights(self, num_frames, frame_width, frame_height, adaptive_weights_definition, homographies):
        '''
        Helper method for _get_jacobi_method_input.
        Return the array of adaptive weights for use in the energy function.

        Input:

        * num_frames: The number of frames in the video.
        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        * adaptive_weights: A NumPy array of size
            (num_frames,).
            adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
            regularization term corresponding to the frame at index t.
            Note that the paper does not specify the weight to apply to the last frame (which does
            not have a velocity), so we assume it is the same as the second-to-last frame.
            In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        '''

        if adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED:
            # the adaptive weights are determined by plugging the eigenvalues of each homography's affine component into a linear model
            homography_affine_components = homographies.copy()
            # 将所有H矩阵第三行置为[0, 0, 1]，此时单应变换退化为仿射变换
            homography_affine_components[:, 2, :] = [0, 0, 1]

            # 建立长度为num_frames的一维数组
            adaptive_weights = np.empty((num_frames))

            for frame_index in range(num_frames):
                homography = homography_affine_components[frame_index]
                # 按行排序(小到大) 绝对值 特征值
                sorted_eigenvalue_magnitudes = np.sort(np.abs(np.linalg.eigvals(homography)))

                # 平移部分(通过图像宽高进行归一化)
                translational_element = math.sqrt((homography[0, 2] / frame_width) ** 2 + (homography[1, 2] / frame_height) ** 2)
                # 仿射部分
                affine_component = sorted_eigenvalue_magnitudes[-2] / sorted_eigenvalue_magnitudes[-1]

                adaptive_weight_candidate_1 = -1.93 * translational_element + 0.95

                if adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL:
                    adaptive_weight_candidate_2 = 5.83 * affine_component + 4.88
                else:  # ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED
                    adaptive_weight_candidate_2 = 5.83 * affine_component - 4.88

                adaptive_weights[frame_index] = max( min(adaptive_weight_candidate_1,adaptive_weight_candidate_2), 0)
        
        # 直接赋值
        elif adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH:
            adaptive_weights = np.full((num_frames), self.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH_VALUE)
        elif adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW:
            adaptive_weights = np.full((num_frames), self.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW_VALUE)

        return adaptive_weights

    ##  雅可比方法求解最优局部x Ax=b
    def _get_jacobi_method_output(self, off_diagonal_coefficients, on_diagonal_coefficients, x_start, b):
        '''
        Helper method for _get_stabilized_displacements.
        Using the Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximate a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.

        Return a value of x after performing self.optimization_num_iterations of the Jacobi method.

        Input:

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
         * on_diagonal_coefficients: A 1D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i] = A_{i, i}.
            In the Wikipedia link, this array is the diagonal entries of D.
        * x_start: A NumPy array containing an initial estimate for x.
        * b: A NumPy array containing the constant vector b.

        Output:

        * x: A NumPy array containing the value of x computed with the Jacobi method.
        '''

        x = x_start.copy()

        # np.reciprocal取倒数 构造对角矩阵 即D^(-1)
        reciprocal_on_diagonal_coefficients_matrix = np.diag(np.reciprocal(on_diagonal_coefficients))
         
        # x^(k+1) = D^(-1) * (b - (L + U) * x^(k))
        for _ in range(self.optimization_num_iterations):
            x = np.matmul(reciprocal_on_diagonal_coefficients_matrix,
                          b - np.matmul(off_diagonal_coefficients, x))

        return x



        # unstabilized_vertex_x_y and stabilized_vertex_x_y are CV_32FC2 NumPy arrays
        # of the coordinates of the mesh nodes in the stabilized video, indexed from the top left
        # corner and moving left-to-right, top-to-bottom.
        total_frames = len(unstabilized_frames)
        count = total_frames // multicore
        frame_height, frame_width = unstabilized_frames[0].shape[:2]

        if index == self.multicore - 1:
            unstabilized_frames_in_snipped = unstabilized_frames[index * count: total_frames]
            stabilized_motion_mesh_in_snipped = stabilized_motion_mesh_by_frame_index[index * count: total_frames]
        else:
            unstabilized_frames_in_snipped = unstabilized_frames[index * count: (index + 1) * count]
            stabilized_motion_mesh_in_snipped = stabilized_motion_mesh_by_frame_index[index * count: (index + 1) * count]
        
        num_frames = len(unstabilized_frames_in_snipped)

        unstabilized_vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)

        # row_col_to_unstabilized_vertex_x_y[row, col] and
        # row_col_to_stabilized_vertex_x_y[row, col]
        # contain the x and y positions of the vertex at the given row and col
        # 原格式 ((self.mesh_row_count + 1)* (self.mesh_col_count + 1), 1, 2)
        row_col_to_unstabilized_vertex_x_y = np.reshape(
            unstabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))

        # Construct map from the stabilized frame to the unstabilized frame.
        # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized video, then
        # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
        # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
        # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
        # NOTE the inverted coordinate order. This setup allows us to index into map just like
        # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
        # order so we can easily apply homographies to those points.
        # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
        # coordinate falls outside the stabilized image (so in the output image, that image
        # should be filled with a border color).
        # Since these arrays' default values fall outside the unstabilized image, remap will
        # fill in those coordinates in the stabilized image with the border color as desired.
        # 数组形状 (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
        frame_stabilized_y_x_to_unstabilized_x_template = np.full((frame_height, frame_width), frame_width + 1)
        frame_stabilized_y_x_to_unstabilized_y_template = np.full((frame_height, frame_width), frame_height + 1)
        
        # shape(frame_stabilized_y_x_to_stabilized_x_y_template) = (frame_height, frame_width, 2)
        # shape(frame_stabilized_x_y_template) = (frame_height * frame_width, 1, 2)
        # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
        frame_stabilized_x_y_template = np.swapaxes(
            np.indices((frame_width, frame_height), dtype=np.float32), 0, 2).reshape((-1, 1, 2))

        # left_crop_x_by_frame_index[frame_index] contains the x-value where the left edge
        # where frame frame_index would be cropped to produce a rectangular image;
        # right_crop_x_by_frame_index, top_crop_y_by_frame_index, and
        # bottom_crop_y_by_frame_index are analogous
        # 稳定后图像的边界点坐标 长度为num_frames的一维数组
        left_crop_x_by_frame_index = np.full(num_frames, 0)
        right_crop_x_by_frame_index = np.full(num_frames, frame_width - 1)
        top_crop_y_by_frame_index = np.full(num_frames, 0)
        bottom_crop_y_by_frame_index = np.full(num_frames, frame_height - 1)

        stabilized_frames = []
        # with tqdm.trange(num_frames) as t:
        #     t.set_description('Warping frames')
        #     for frame_index in t:
        for frame_index in range(num_frames):
            unstabilized_frame = unstabilized_frames_in_snipped[frame_index]

            # Construct map from the stabilized frame to the unstabilized frame.
            # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized
            # video, then
            # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
            # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
            # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
            # NOTE the inverted coordinate order. This setup allows us to index into map just like
            # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
            # order so we can easily apply homographies to those points.
            # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
            # coordinate falls outside the stabilized image (so in the output image, that image
            # should be filled with a border color).
            # Since these arrays' default values fall outside the unstabilized image, remap will
            # fill in those coordinates in the stabilized image with the border color as desired.
            # np.full((frame_height, frame_width), frame_width + 1)
            frame_stabilized_y_x_to_unstabilized_x = np.copy(frame_stabilized_y_x_to_unstabilized_x_template)
            # np.full((frame_height, frame_width), frame_height + 1)
            frame_stabilized_y_x_to_unstabilized_y = np.copy(frame_stabilized_y_x_to_unstabilized_y_template)
            # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
            # shape(-1, 1, 2)
            frame_stabilized_x_y = np.copy(frame_stabilized_x_y_template)

            # Determine the coordinates of the mesh vertices in the stabilized video.
            # The current displacements are given by vertex_unstabilized_displacements, and
            # the desired displacements are given by vertex_stabilized_displacements,
            # so adding the difference of the two transforms the frame as desired.
            # 将顶点稳定前后运动向量的差值叠加到各顶点坐标上 得到稳定后的顶点坐标
            # unstabilized_vertex_x_y ((self.mesh_row_count + 1)* (self.mesh_col_count + 1), 1, 2)
            # stabilized_motion_mesh_by_frame_index (num_frames, self.mesh_row_count + 1 * self.mesh_col_count + 1, 1, 2)
            stabilized_vertex_x_y = unstabilized_vertex_x_y + stabilized_motion_mesh_in_snipped[frame_index]

            # 重构数组结构
            row_col_to_stabilized_vertex_x_y = np.reshape(
                stabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
            # Look at each face of the mesh. Since we know the original and transformed coordinates
            # of its four vertices, we can construct a homography to fill in the remaining pixels
            # TODO parallelize
            # 由稳定后的顶点坐标和稳定后的顶点坐标，计算单应性矩阵
            for cell_top_left_row in range(self.mesh_row_count):
                for cell_top_left_col in range(self.mesh_col_count):

                    # Construct a mask representing the stabilized cell.
                    # Since we know the cell's boundaries before and after stabilization, we can
                    # construct a homography representing this cell's warp and then apply it to
                    # the unstabilized cell (which is just a rectangle) to construct the stabilized
                    # cell.
                    # top_left, top_right, bottom_left, bottom_right
                    # unstabilized为原图像各网格顶点坐标 stabilized为叠加稳定向量后的各网格顶点坐标
                    unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[
                        cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                    stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[
                        cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                    # 计算单应性矩阵
                    unstabilized_to_stabilized_homography, _ = cv2.findHomography(
                        unstabilized_cell_bounds, stabilized_cell_bounds)
                    stabilized_to_unstabilized_homography, _ = cv2.findHomography(
                        stabilized_cell_bounds, unstabilized_cell_bounds)

                    # 行列互换 先列后行
                    # [[ 80. 120.  80. 120.]    列变换 
                    #  [ 30.  30.  60.  60.]]   行变化
                    unstabilized_cell_x_bounds, unstabilized_cell_y_bounds = np.transpose(unstabilized_cell_bounds)
                    # 向下取整
                    unstabilized_cell_left_x = math.floor(np.min(unstabilized_cell_x_bounds))
                    # 向上取整
                    unstabilized_cell_right_x = math.ceil(np.max(unstabilized_cell_x_bounds))
                    unstabilized_cell_top_y = math.floor(np.min(unstabilized_cell_y_bounds))
                    unstabilized_cell_bottom_y = math.ceil(np.max(unstabilized_cell_y_bounds))
                    # 设置掩膜 取原始图像(未稳定)中对应的某一网格
                    unstabilized_cell_mask = np.zeros((frame_height, frame_width))
                    unstabilized_cell_mask[unstabilized_cell_top_y:unstabilized_cell_bottom_y + 1,
                                            unstabilized_cell_left_x:unstabilized_cell_right_x + 1] = 255
                    # dsize (x, y)
                    stabilized_cell_mask = cv2.warpPerspective(
                        unstabilized_cell_mask, unstabilized_to_stabilized_homography, (frame_width, frame_height))
                    
                    # 以下取全局变化
                    # 以稳定后的图像为基准 通过逆单应变换获得稳定后图像与稳定前图像间各像素的对应关系
                    # frame_stalizied_x_y中存储每个像素所在位置的索引 shape(-1, 1, 2)
                    # shape(cell_unstabilized_x_y) = (frame_height * frame_width, 1, 2)
                    cell_unstabilized_x_y = cv2.perspectiveTransform(frame_stabilized_x_y, stabilized_to_unstabilized_homography)
                    # shape(cell_stabilized_y_x_to_unstabilized_x_y) = (frame_height, frame_width, 2)
                    cell_stabilized_y_x_to_unstabilized_x_y = cell_unstabilized_x_y.reshape((frame_height, frame_width, 2))
                    # (2, frame_width, frame_height)
                    cell_stabilized_y_x_to_unstabilized_x, cell_stabilized_y_x_to_unstabilized_y = np.moveaxis(
                        cell_stabilized_y_x_to_unstabilized_x_y, 2, 0)

                    # update the overall stabilized-to-unstabilized map, applying this cell's
                    # transformation only to those pixels that are actually part of this cell、
                    # 以下取局部 将掩膜内的部分转换为经逆单应矩阵变换后的点的坐标
                    # frame_stabilized_y_x_to_unstabilized_x  np.full((frame_height, frame_width), frame_width + 1)
                    # (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
                    # 向各网格在稳定图像中所对应的区域内存入各像素点对应未稳定图像像素的索引
                    frame_stabilized_y_x_to_unstabilized_x = np.where(
                        stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_x, frame_stabilized_y_x_to_unstabilized_x)
                    frame_stabilized_y_x_to_unstabilized_y = np.where(
                        stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_y, frame_stabilized_y_x_to_unstabilized_y)
            
            # cv2.remap(img,map1,map2,interpolation) img源图像 map1表示CV_32FC2类型(x,y)点的x map2表示CV_32FC2类型(x,y)点的y
            stabilized_frame = cv2.remap(
                unstabilized_frame,
                frame_stabilized_y_x_to_unstabilized_x.reshape((frame_height, frame_width, 1)).astype(np.float32),
                frame_stabilized_y_x_to_unstabilized_y.reshape((frame_height, frame_width, 1)).astype(np.float32),
                cv2.INTER_LINEAR,
                borderValue=self.color_outside_image_area_bgr
            )

            # crop the frame
            # left edge: the maximum stabilized x_s that corresponds to the unstabilized x_u = 0
            # np.abs返回每个元素的绝对值
            # np.where 返回符合条件的元素的索引 [0]为所在行数 [1]为所在列数
            # 即取稳定后图像的最大内接矩
            stabilized_image_x_matching_unstabilized_left_edge = np.where(
                np.abs(frame_stabilized_y_x_to_unstabilized_x - 0) < 1)[1]
            if stabilized_image_x_matching_unstabilized_left_edge.size > 0:
                left_crop_x_by_frame_index[frame_index] = np.max(stabilized_image_x_matching_unstabilized_left_edge)

            # right edge: the minimum stabilized x_s that corresponds to the stabilized
            # x_u = frame_width - 1

            stabilized_image_x_matching_unstabilized_right_edge = np.where(
                np.abs(frame_stabilized_y_x_to_unstabilized_x - (frame_width - 1)) < 1)[1]
            if stabilized_image_x_matching_unstabilized_right_edge.size > 0:
                right_crop_x_by_frame_index[frame_index] = np.min(stabilized_image_x_matching_unstabilized_right_edge)

            # top edge: the maximum stabilized y_s that corresponds to the unstabilized
            # y_u = 01

            stabilized_image_y_matching_unstabilized_top_edge = np.where(
                np.abs(frame_stabilized_y_x_to_unstabilized_y - 0) < 1)[0]
            if stabilized_image_y_matching_unstabilized_top_edge.size > 0:
                top_crop_y_by_frame_index[frame_index] = np.max(stabilized_image_y_matching_unstabilized_top_edge)

            # bottom edge: the minimum stabilized y_s that corresponds to the unstabilized
            # y_u = frame_height - 1

            stabilized_image_y_matching_unstabilized_bottom_edge = np.where(
                np.abs(frame_stabilized_y_x_to_unstabilized_y - (frame_height - 1)) < 1)[0]
            if stabilized_image_y_matching_unstabilized_bottom_edge.size > 0:
                bottom_crop_y_by_frame_index[frame_index] = np.min(stabilized_image_y_matching_unstabilized_bottom_edge)

            stabilized_frames.append(stabilized_frame)

        return stabilized_frames, left_crop_x_by_frame_index, right_crop_x_by_frame_index, top_crop_y_by_frame_index, bottom_crop_y_by_frame_index


    ##  根据四个顶点稳定前后的运动向量之差计算单应矩阵并进行网格变形 以稳定后的图像为基准 确定稳定后的图像各像素与源图像各像素的对应关系  ##
    def _get_stabilized_frames_and_crop_boundaries(self, num_frames, unstabilized_frames, vertex_unstabilized_displacements_by_frame_index, vertex_stabilized_displacements_by_frame_index):
        '''
        Helper method for stabilize.

        Return stabilized copies of the given unstabilized frames warping them according to the
        given transformation data, as well as boundaries representing how to crop these stabilized
        frames.

        Inspired by the Python MeshFlow implementation available at
        https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization/blob/5780fe750cf7dc35e5cfcd0b4a56d408ce3a9e53/src/MeshFlow.py#L117.

        Input:

        * num_frames: The number of frames in the unstabilized video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
            unstabilized displacements of each vertex in the MeshFlow mesh, as generated by
            _get_unstabilized_vertex_displacements_and_homographies.
        * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
            stabilized displacements of each vertex in the MeshFlow mesh, as generated by
            _get_stabilized_vertex_displacements.

        Output:

        A tuple of the following items in order.

        * stabilized_frames: A list of the frames in the stabilized video, each represented as a
            NumPy array.
        * crop_boundaries: A tuple of the form
            (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
            representing the x- and y-boundaries (all inclusive) of the cropped video.
        '''

        frame_height, frame_width = unstabilized_frames[0].shape[:2]

        # unstabilized_vertex_x_y and stabilized_vertex_x_y are CV_32FC2 NumPy arrays
        # (see https://stackoverflow.com/a/47617999)
        # of the coordinates of the mesh nodes in the stabilized video, indexed from the top left
        # corner and moving left-to-right, top-to-bottom.
        # 例 [[[ 0., 0.]]
        #     [[40., 0.]]
        #     ...
        #     [[ 0., 40.]]
        #     [[40., 40.]]
        #     ...
        #     [[600., 479.]]
        #     [[639., 479.]]]
        unstabilized_vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)

        # row_col_to_unstabilized_vertex_x_y[row, col] and
        # row_col_to_stabilized_vertex_x_y[row, col]
        # contain the x and y positions of the vertex at the given row and col
        # 原格式 ((self.mesh_row_count + 1)* (self.mesh_col_count + 1), 1, 2)
        # [[[  0.   0.]
        #   [ 40.   0.]]
        #    ...
        #   [639.   0.]]
        #
        #  [[  0.  60.]
        #   [ 40.  60.]
        #    ...
        #   [639.  60.]]
        #    ...
        #  [[  0. 479.]
        #   [ 40. 479.]
        #    ...
        #   [639. 479.]]]
        row_col_to_unstabilized_vertex_x_y = np.reshape(
            unstabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))

        # stabilized_motion_mesh_by_frame_index[frame_index] is a CV_32FC2 NumPy array
        # (see https://stackoverflow.com/a/47617999) containing the amount to add to each vertex coordinate to transform it 
        # from its unstabilized position at frame frame_index to its stabilized position at frame frame_index.
        # Since the current displacements are given by vertex_unstabilized_displacements[frame_index],
        # and the final displacements are given by vertex_stabilized_displacements[frame_index], 
        # adding the difference of the two produces the desired result.
        # 数据格式  (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
        # np.reshape -1 代表自动匹配元素数量
        stabilized_motion_mesh_by_frame_index = np.reshape(
            vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index,
            (num_frames, -1, 1, 2)
        )
        
        # Construct map from the stabilized frame to the unstabilized frame.
        # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized
        # video, then
        # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
        # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
        # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
        # NOTE the inverted coordinate order. This setup allows us to index into map just like
        # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
        # order so we can easily apply homographies to those points.
        # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
        # coordinate falls outside the stabilized image (so in the output image, that image
        # should be filled with a border color).
        # Since these arrays' default values fall outside the unstabilized image, remap will
        # fill in those coordinates in the stabilized image with the border color as desired.
        # 数组形状 (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
        frame_stabilized_y_x_to_unstabilized_x_template = np.full((frame_height, frame_width), frame_width + 1)
        frame_stabilized_y_x_to_unstabilized_y_template = np.full((frame_height, frame_width), frame_height + 1)
        
        # a = np.indices((2, 3), dtype=np.float32)
        # [[[0. 0. 0.]
        #   [1. 1. 1.]]
        #  [[0. 1. 2.]
        #   [0. 1. 2.]]]
        # shape(a) = (2, 2, 3)

        # b = np.swapaxes(a, 0, 2)
        #   [[[0. 0.]
        #     [1. 0.]]
        #    [[0. 1.]
        #     [1. 1.]]
        #    [[0. 2.]
        #     [1. 2.]]]
        # shape(b) = (3, 2, 2)

        # shape(frame_stabilized_y_x_to_stabilized_x_y_template) = (frame_height, frame_width, 2)
        frame_stabilized_y_x_to_stabilized_x_y_template = np.swapaxes(
            np.indices((frame_width, frame_height), dtype=np.float32), 0, 2)
        # shape(frame_stabilized_x_y_template) = (frame_height * frame_width, 1, 2)
        # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
        # [[[0. 0.]]
        #  [[1. 0.]]
        #  ...
        #  [[0. 1.]]
        #  [[1. 1.]]
        #  ...
        #  [[0. 4.]]
        #  ...
        #  [[3. 4.]]]
        frame_stabilized_x_y_template = frame_stabilized_y_x_to_stabilized_x_y_template.reshape((-1, 1, 2))

        # left_crop_x_by_frame_index[frame_index] contains the x-value where the left edge
        # where frame frame_index would be cropped to produce a rectangular image;
        # right_crop_x_by_frame_index, top_crop_y_by_frame_index, and
        # bottom_crop_y_by_frame_index are analogous
        # 稳定后图像的边界点坐标 长度为num_frames的一维数组
        left_crop_x_by_frame_index = np.full(num_frames, 0)
        right_crop_x_by_frame_index = np.full(num_frames, frame_width - 1)
        top_crop_y_by_frame_index = np.full(num_frames, 0)
        bottom_crop_y_by_frame_index = np.full(num_frames, frame_height - 1)

        stabilized_frames = []
        with tqdm.trange(num_frames) as t:
            t.set_description('Warping frames')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]

                # Construct map from the stabilized frame to the unstabilized frame.
                # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized
                # video, then
                # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
                # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
                # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
                # NOTE the inverted coordinate order. This setup allows us to index into map just like
                # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
                # order so we can easily apply homographies to those points.
                # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
                # coordinate falls outside the stabilized image (so in the output image, that image
                # should be filled with a border color).
                # Since these arrays' default values fall outside the unstabilized image, remap will
                # fill in those coordinates in the stabilized image with the border color as desired.
                # np.full((frame_height, frame_width), frame_width + 1)
                frame_stabilized_y_x_to_unstabilized_x = np.copy(frame_stabilized_y_x_to_unstabilized_x_template)
                # np.full((frame_height, frame_width), frame_height + 1)
                frame_stabilized_y_x_to_unstabilized_y = np.copy(frame_stabilized_y_x_to_unstabilized_y_template)
                # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
                # shape(-1, 1, 2)
                frame_stabilized_x_y = np.copy(frame_stabilized_x_y_template)

                # Determine the coordinates of the mesh vertices in the stabilized video.
                # The current displacements are given by vertex_unstabilized_displacements, and
                # the desired displacements are given by vertex_stabilized_displacements,
                # so adding the difference of the two transforms the frame as desired.
                # 将顶点稳定前后运动向量的差值叠加到各顶点坐标上 得到稳定后的顶点坐标
                # unstabilized_vertex_x_y ((self.mesh_row_count + 1)* (self.mesh_col_count + 1), 1, 2)
                # stabilized_motion_mesh_by_frame_index (num_frames, self.mesh_row_count + 1 * self.mesh_col_count + 1, 1, 2)
                stabilized_vertex_x_y = unstabilized_vertex_x_y + stabilized_motion_mesh_by_frame_index[frame_index]

                # 重构数组结构
                row_col_to_stabilized_vertex_x_y = np.reshape(
                    stabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
                # Look at each face of the mesh. Since we know the original and transformed coordinates
                # of its four vertices, we can construct a homography to fill in the remaining pixels
                # TODO parallelize
                # 由稳定后的顶点坐标和稳定后的顶点坐标，计算单应性矩阵
                for cell_top_left_row in range(self.mesh_row_count):
                    for cell_top_left_col in range(self.mesh_col_count):

                        # Construct a mask representing the stabilized cell.
                        # Since we know the cell's boundaries before and after stabilization, we can
                        # construct a homography representing this cell's warp and then apply it to
                        # the unstabilized cell (which is just a rectangle) to construct the stabilized
                        # cell.
                        # top_left, top_right, bottom_left, bottom_right
                        # unstabilized为原图像各网格顶点坐标 stabilized为叠加稳定向量后的各网格顶点坐标
                        unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[
                            cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                        stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[
                            cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                        # 计算单应性矩阵
                        unstabilized_to_stabilized_homography, _ = cv2.findHomography(
                            unstabilized_cell_bounds, stabilized_cell_bounds)
                        stabilized_to_unstabilized_homography, _ = cv2.findHomography(
                            stabilized_cell_bounds, unstabilized_cell_bounds)

                        # 行列互换 先列后行
                        # [[ 80. 120.  80. 120.]    列变换 
                        #  [ 30.  30.  60.  60.]]   行变化
                        unstabilized_cell_x_bounds, unstabilized_cell_y_bounds = np.transpose(unstabilized_cell_bounds)
                        # 向下取整
                        unstabilized_cell_left_x = math.floor(np.min(unstabilized_cell_x_bounds))
                        # 向上取整
                        unstabilized_cell_right_x = math.ceil(np.max(unstabilized_cell_x_bounds))
                        unstabilized_cell_top_y = math.floor(np.min(unstabilized_cell_y_bounds))
                        unstabilized_cell_bottom_y = math.ceil(np.max(unstabilized_cell_y_bounds))
                        # 设置掩膜 取原始图像(未稳定)中对应的某一网格
                        unstabilized_cell_mask = np.zeros((frame_height, frame_width))
                        unstabilized_cell_mask[unstabilized_cell_top_y:unstabilized_cell_bottom_y + 1,
                                               unstabilized_cell_left_x:unstabilized_cell_right_x + 1] = 255
                        #图像位置平移
                        # transform_dist = [-50, 0]
                        # transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])
                        # dsize (x, y)
                        stabilized_cell_mask = cv2.warpPerspective(
                            unstabilized_cell_mask, unstabilized_to_stabilized_homography, (frame_width, frame_height))
                        
                        # 以下取全局变化
                        # 以稳定后的图像为基准 通过逆单应变换获得稳定后图像与稳定前图像间各像素的对应关系
                        # frame_stalizied_x_y中存储每个像素所在位置的索引 shape(-1, 1, 2)
                        # shape(cell_unstabilized_x_y) = (frame_height * frame_width, 1, 2)
                        cell_unstabilized_x_y = cv2.perspectiveTransform(frame_stabilized_x_y, stabilized_to_unstabilized_homography)
                        # cell_unstabilized_x_y = frame_stabilized_x_y
                        # shape(cell_stabilized_y_x_to_unstabilized_x_y) = (frame_height, frame_width, 2)
                        cell_stabilized_y_x_to_unstabilized_x_y = cell_unstabilized_x_y.reshape((frame_height, frame_width, 2))
                        # (2, frame_width, frame_height)
                        cell_stabilized_y_x_to_unstabilized_x, cell_stabilized_y_x_to_unstabilized_y = np.moveaxis(
                            cell_stabilized_y_x_to_unstabilized_x_y, 2, 0)

                        # update the overall stabilized-to-unstabilized map, applying this cell's
                        # transformation only to those pixels that are actually part of this cell、
                        # 以下取局部 将掩膜内的部分转换为经逆单应矩阵变换后的点的坐标
                        # frame_stabilized_y_x_to_unstabilized_x  np.full((frame_height, frame_width), frame_width + 1)
                        # (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
                        # 向各网格在稳定图像中所对应的区域内存入各像素点对应未稳定图像像素的索引
                        frame_stabilized_y_x_to_unstabilized_x = np.where(
                            stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_x, frame_stabilized_y_x_to_unstabilized_x)
                        frame_stabilized_y_x_to_unstabilized_y = np.where(
                            stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_y, frame_stabilized_y_x_to_unstabilized_y)
                
                # cv2.remap(img,map1,map2,interpolation) img源图像 map1表示CV_32FC2类型(x,y)点的x map2表示y
                stabilized_frame = cv2.remap(
                    unstabilized_frame,
                    frame_stabilized_y_x_to_unstabilized_x.reshape((frame_height, frame_width, 1)).astype(np.float32),
                    frame_stabilized_y_x_to_unstabilized_y.reshape((frame_height, frame_width, 1)).astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderValue=self.color_outside_image_area_bgr
                )

                # crop the frame
                # left edge: the maximum stabilized x_s that corresponds to the unstabilized x_u = 0
                # np.abs返回每个元素的绝对值
                # np.where 返回符合条件的元素的索引 [0]为所在行数 [1]为所在列数
                # 即取稳定后图像的最大内接矩
                stabilized_image_x_matching_unstabilized_left_edge = np.where(
                    np.abs(frame_stabilized_y_x_to_unstabilized_x - 0) < 1)[1]
                if stabilized_image_x_matching_unstabilized_left_edge.size > 0:
                    left_crop_x_by_frame_index[frame_index] = np.max(stabilized_image_x_matching_unstabilized_left_edge)

                # right edge: the minimum stabilized x_s that corresponds to the stabilized
                # x_u = frame_width - 1
                stabilized_image_x_matching_unstabilized_right_edge = np.where(
                    np.abs(frame_stabilized_y_x_to_unstabilized_x - (frame_width - 1)) < 1)[1]
                if stabilized_image_x_matching_unstabilized_right_edge.size > 0:
                    right_crop_x_by_frame_index[frame_index] = np.min(stabilized_image_x_matching_unstabilized_right_edge)

                # top edge: the maximum stabilized y_s that corresponds to the unstabilized
                # y_u = 01
                stabilized_image_y_matching_unstabilized_top_edge = np.where(
                    np.abs(frame_stabilized_y_x_to_unstabilized_y - 0) < 1)[0]
                if stabilized_image_y_matching_unstabilized_top_edge.size > 0:
                    top_crop_y_by_frame_index[frame_index] = np.max(stabilized_image_y_matching_unstabilized_top_edge)

                # bottom edge: the minimum stabilized y_s that corresponds to the unstabilized
                # y_u = frame_height - 1
                stabilized_image_y_matching_unstabilized_bottom_edge = np.where(
                    np.abs(frame_stabilized_y_x_to_unstabilized_y - (frame_height - 1)) < 1)[0]
                if stabilized_image_y_matching_unstabilized_bottom_edge.size > 0:
                    bottom_crop_y_by_frame_index[frame_index] = np.min(stabilized_image_y_matching_unstabilized_bottom_edge)

                stabilized_frames.append(stabilized_frame)

        # the final video crop is the one that would adequately crop every single frame
        # 取稳定后图像中能够映射到源图像的部分 即取各帧最大内接矩的交
        left_crop_x = np.max(left_crop_x_by_frame_index)
        right_crop_x = np.min(right_crop_x_by_frame_index)
        top_crop_y = np.max(top_crop_y_by_frame_index)
        bottom_crop_y = np.min(bottom_crop_y_by_frame_index)

        return (stabilized_frames, (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y))


    ##  裁切稳定后的图像 在保持长宽比的前提下使之充满图窗  ##
    def _crop_frames(self, uncropped_frames, crop_boundaries):
        '''
        Return copies of the given frames that have been cropped according to the given crop
        boundaries.

        Input:

        * uncropped_frames: A list of the frames to crop, each represented as a NumPy array.
        * crop_boundaries: A tuple of the form
            (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
            representing the x- and y-boundaries (all inclusive) of the cropped video.

        Output:

        * cropped_frames: A list of the frames cropped according to the crop boundaries.

        '''

        frame_height, frame_width = uncropped_frames[0].shape[:2]
        left_crop_x, top_crop_y, right_crop_x, bottom_crop_y = crop_boundaries

        # There are two ways to scale up the image: increase its width to fill the original width,
        # scaling the height appropriately, or increase its height to fill the original height,
        # scaling the width appropriately. At least one of these options will result in the image
        # completely filling the frame.
        uncropped_aspect_ratio = frame_width / frame_height
        cropped_aspect_ratio = (right_crop_x + 1 - left_crop_x) / (bottom_crop_y + 1 - top_crop_y)

        if cropped_aspect_ratio >= uncropped_aspect_ratio:
            # the cropped image is proportionally wider than the original, so to completely fill the
            # frame, it must be scaled so its height matches the frame height
            # 图像过宽
            uncropped_to_cropped_scale_factor = frame_height / (bottom_crop_y + 1 - top_crop_y)
        else:
            # the cropped image is proportionally taller than the original, so to completely fill
            # the frame, it must be scaled so its width matches the frame width
            # 图像过高
            uncropped_to_cropped_scale_factor = frame_width / (right_crop_x + 1 - left_crop_x)

        # 图像resize 保持稳定图像比例不变的同时使之充满图窗
        cropped_frames = []
        for uncropped_frame in uncropped_frames:
            cropped_frames.append(cv2.resize(
                uncropped_frame[top_crop_y:bottom_crop_y + 1, left_crop_x:right_crop_x + 1],    # 输入图像
                (frame_width, frame_height),                                                    # 输出图像尺寸
                fx = uncropped_to_cropped_scale_factor,                                         # 水平轴比例因子
                fy = uncropped_to_cropped_scale_factor                                          # 垂直轴比例因子
            ))

        return cropped_frames

  
    def _compute_cropping_ratio_and_distortion_score(self, num_frames, unstabilized_frames, cropped_frames):
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

        with tqdm.trange(num_frames) as t:
            t.set_description('Computing cropping ratio and distortion score')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]
                cropped_frame = cropped_frames[frame_index]
                unstabilized_to_cropped_homography = self._get_matched_homography_for_evaluation(unstabilized_frame, cropped_frame)

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

    def _compute_stability_score(self, vertex_stabilized_displacements_by_frame_index):
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

        return (x_stability_score + y_stability_score) / 2.0


    ##  在窗格中显示稳定前与稳定后的图像  ##  
    def _display_unstablilized_and_cropped_video_loop(self, num_frames, frames_per_second, unstabilized_frames, cropped_frames):
        '''
        Helper function for stabilize.

        Display a loop of the unstabilized and cropped, stabilized videos.

        Input:

        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * cropped_frames: A list of the cropped, stabilized frames, each represented as a NumPy
            array.

        Output:

        (The unstabilized and cropped, stabilized videos loop in a new window. Pressing the Q key
        closes the window.)
        '''
        
        # 每帧的显示时长(ms)
        milliseconds_per_frame = int(1000/frames_per_second)
        while True:
            for i in range(num_frames):
                # 按垂直方向(行顺序)堆叠数组
                cv2.imshow('unstabilized and stabilized video', np.vstack(
                    (unstabilized_frames[i], cropped_frames[i])))
                if cv2.waitKey(milliseconds_per_frame) & 0xFF == ord('q'):
                    return


    ##  写稳定后的视频  ##
    def _write_stabilized_video(self, output_path, num_frames, frames_per_second, stabilized_frames):
        '''
        Helper method for stabilize.
        Write the given stabilized frames as a video to the given path.

        Input:
        * output_path: The path where the stabilized version of the video should be placed.
        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * codec: The video codec.
        * stabilized_frames: A list of the frames in the stabilized video, each represented as a
            NumPy array.

        Output:

        (The video is saved to output_path.)
        '''

        # adapted from https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        frame_height, frame_width = stabilized_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        video = cv2.VideoWriter(output_path, fourcc, frames_per_second, (frame_width, frame_height))

        with tqdm.trange(num_frames) as t:
            t.set_description(f'Writing stabilized video to <{output_path}>')
            for frame_index in t:
                video.write(stabilized_frames[frame_index])

        video.release()



##  调用多进程 根据四个顶点稳定前后的运动向量之差计算单应矩阵并进行网格变形  ##
def get_stabilized_frames_and_crop_boundaries_with_multiprocessing(pos, num_frames, unstabilized_frames, vertex_for_stitch, vertex_unstabilized_displacements_by_frame_index, vertex_stabilized_displacements_by_frame_index):
    '''
    Helper method for stabilize.

    Return stabilized copies of the given unstabilized frames warping them according to the
    given transformation data, as well as boundaries representing how to crop these stabilized
    frames.

    Input:

    * num_frames: The number of frames in the unstabilized video.
    * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
    * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
        unstabilized displacements of each vertex in the MeshFlow mesh, as generated by
        _get_unstabilized_vertex_displacements_and_homographies.
    * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
        stabilized displacements of each vertex in the MeshFlow mesh, as generated by
        _get_stabilized_vertex_displacements.

    Output:

    A tuple of the following items in order.

    * stabilized_frames: A list of the frames in the stabilized video, each represented as a
        NumPy array.
    * crop_boundaries: A tuple of the form
        (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
        representing the x- and y-boundaries (all inclusive) of the cropped video.
    '''

    # stabilized_motion_mesh_by_frame_index[frame_index] is a CV_32FC2 NumPy array containing the amount to add to 
    # each vertex coordinate to transform it from its unstabilized position at frame frame_index to its stabilized 
    # position at frame frame_index.
    # Since the current displacements are given by vertex_unstabilized_displacements[frame_index],
    # and the final displacements are given by vertex_stabilized_displacements[frame_index], 
    # adding the difference of the two produces the desired result.
    # shape  (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
    # stabilized_motion_mesh_by_frame_index = np.reshape(
    #     vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index,
    #     (num_frames, -1, 1, 2)
    # )
    
    ##
    # stabilized_motion_mesh_by_frame_index_unfiltered = vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index
    # stabilized_motion_mesh_by_frame_index = []
    # for stabilized_motion_mesh_unfiltered in stabilized_motion_mesh_by_frame_index_unfiltered:
    #     vertex_stabilized_x_velocities_by_row_col = stabilized_motion_mesh_unfiltered[:, :, 0].astype(np.float32)
    #     vertex_stabilized_y_velocities_by_row_col = stabilized_motion_mesh_unfiltered[:, :, 1].astype(np.float32)
    #     vertex_smoothed_x_velocities_by_row_col = cv2.medianBlur(vertex_stabilized_x_velocities_by_row_col, 3)
    #     vertex_smoothed_y_velocities_by_row_col = cv2.medianBlur(vertex_stabilized_y_velocities_by_row_col, 3)
    #     # vertex_smoothed_x_velocities_by_row_col = signal.medfilt2d(vertex_stabilized_x_velocities_by_row_col, kernel_size = 3)
    #     # vertex_smoothed_y_velocities_by_row_col = signal.medfilt2d(vertex_stabilized_y_velocities_by_row_col, kernel_size = 3)
    #     stabilized_motion_mesh_by_frame_index.append(np.dstack((vertex_smoothed_x_velocities_by_row_col, vertex_smoothed_y_velocities_by_row_col)))
    # # print(stabilized_motion_mesh_by_frame_index[100])
    # stabilized_mediafiltered_motion_mesh_by_frame_index = np.reshape(stabilized_motion_mesh_by_frame_index, (num_frames, -1, 1, 2))
    ##

    stabilized_motion_mesh_by_frame_index = np.reshape(vertex_for_stitch + vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index, (num_frames, -1, 1, 2)) 
    print('multicore for frame warp start!')
    t1 = time.time()

    job = partial(multiprocessing_job_for_warp, pos, unstabilized_frames, stabilized_motion_mesh_by_frame_index)
    pool = mp.Pool(processes=multicore)
    res = pool.map(job, range(multicore))

    print('multicore for frame warp finish!')
    t2 = time.time()
    print('cost: %.2f'%(t2-t1))

    stabilized_frames = []
    for result in res:
        stabilized_frames.append(result)

    stabilized_frames = np.concatenate(stabilized_frames, 0)

    return stabilized_frames

def multiprocessing_job_for_warp(pos, unstabilized_frames, stabilized_motion_mesh_by_frame_index, index):
    # unstabilized_vertex_x_y and stabilized_vertex_x_y are CV_32FC2 NumPy arrays
    # of the coordinates of the mesh nodes in the stabilized video, indexed from the top left
    # corner and moving left-to-right, top-to-bottom.
    total_frames = len(unstabilized_frames)
    count = total_frames // multicore
    frame_height, frame_width = unstabilized_frames[0].shape[:2]

    if index == multicore - 1:
        unstabilized_frames_in_snipped = unstabilized_frames[index * count: total_frames]
        stabilized_motion_mesh_in_snipped = stabilized_motion_mesh_by_frame_index[index * count: total_frames]
    else:
        unstabilized_frames_in_snipped = unstabilized_frames[index * count: (index + 1) * count]
        stabilized_motion_mesh_in_snipped = stabilized_motion_mesh_by_frame_index[index * count: (index + 1) * count]
    
    num_frames = len(unstabilized_frames_in_snipped)
    
    unstabilized_vertex_x_y = np.array([
            [[math.ceil((frame_width - 1) * (col / (mesh_col_count))), math.ceil((frame_height - 1) * (row / (mesh_row_count)))]]
            for row in range(mesh_row_count + 1)
            for col in range(mesh_col_count + 1)], dtype=np.float32)

    # row_col_to_unstabilized_vertex_x_y[row, col] and
    # row_col_to_stabilized_vertex_x_y[row, col]
    # contain the x and y positions of the vertex at the given row and col
    row_col_to_unstabilized_vertex_x_y = np.reshape(
        unstabilized_vertex_x_y, (mesh_row_count + 1, mesh_col_count + 1, 2))

    # Construct map from the stabilized frame to the unstabilized frame.
    # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized video, then
    # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
    # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
    # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
    # NOTE the inverted coordinate order. This setup allows us to index into map just like
    # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
    # order so we can easily apply homographies to those points.
    # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
    # coordinate falls outside the stabilized image (so in the output image, that image
    # should be filled with a border color).
    # Since these arrays' default values fall outside the unstabilized image, remap will
    # fill in those coordinates in the stabilized image with the border color as desired.
    # 数组形状 (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
    frame_stabilized_y_x_to_unstabilized_x_template = np.full((frame_height, frame_width), frame_width + 1)
    frame_stabilized_y_x_to_unstabilized_y_template = np.full((frame_height, frame_width), frame_height + 1)
    
    # shape(frame_stabilized_y_x_to_stabilized_x_y_template) = (frame_height, frame_width, 2)
    # shape(frame_stabilized_x_y_template) = (frame_height * frame_width, 1, 2)
    # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
    frame_stabilized_x_y_template = np.swapaxes(
        np.indices((frame_width, frame_height), dtype=np.float32), 0, 2).reshape((-1, 1, 2))

    stabilized_frames = []

    x_displacement = O
    if pos > 0:
        x_displacement = -x_displacement
    transform_dist = [x_displacement, 0]
    transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])
    transform_array1 = np.array([[1, 0, -transform_dist[0]], [0, 1, -transform_dist[1]], [0, 0, 1]])

    for frame_index in range(num_frames):
        unstabilized_frame = unstabilized_frames_in_snipped[frame_index]

        # Construct map from the stabilized frame to the unstabilized frame.
        # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized video, then
        # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
        # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
        # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
        # NOTE the inverted coordinate order. This setup allows us to index into map just like
        # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
        # order so we can easily apply homographies to those points.
        # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
        # coordinate falls outside the stabilized image (so in the output image, that image
        # should be filled with a border color).
        # Since these arrays' default values fall outside the unstabilized image, remap will
        # fill in those coordinates in the stabilized image with the border color as desired.
        # np.full((frame_height, frame_width), frame_width + 1)
        frame_stabilized_y_x_to_unstabilized_x = np.copy(frame_stabilized_y_x_to_unstabilized_x_template)
        # np.full((frame_height, frame_width), frame_height + 1)
        frame_stabilized_y_x_to_unstabilized_y = np.copy(frame_stabilized_y_x_to_unstabilized_y_template)
        # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
        # shape(-1, 1, 2)
        frame_stabilized_x_y = np.copy(frame_stabilized_x_y_template)

        # Determine the coordinates of the mesh vertices in the stabilized video.
        # The current displacements are given by vertex_unstabilized_displacements, and
        # the desired displacements are given by vertex_stabilized_displacements,
        # so adding the difference of the two transforms the frame as desired.
        # 将顶点稳定前后运动向量的差值叠加到各顶点坐标上 得到稳定后的顶点坐标
        # unstabilized_vertex_x_y ((self.mesh_row_count + 1)* (self.mesh_col_count + 1), 1, 2)
        # stabilized_motion_mesh_by_frame_index (num_frames, self.mesh_row_count + 1 * self.mesh_col_count + 1, 1, 2)
        stabilized_vertex_x_y = unstabilized_vertex_x_y + stabilized_motion_mesh_in_snipped[frame_index]

        row_col_to_stabilized_vertex_x_y = np.reshape(
            stabilized_vertex_x_y, (mesh_row_count + 1, mesh_col_count + 1, 2))

        # Look at each face of the mesh. Since we know the original and transformed coordinates
        # of its four vertices, we can construct a homography to fill in the remaining pixels
        # TODO parallelize
        # 由稳定后的顶点坐标和稳定后的顶点坐标，计算单应性矩阵
        for cell_top_left_row in range(mesh_row_count):
            for cell_top_left_col in range(mesh_col_count):

                # Construct a mask representing the stabilized cell.
                # Since we know the cell's boundaries before and after stabilization, we can
                # construct a homography representing this cell's warp and then apply it to the 
                # unstabilized cell (which is just a rectangle) to construct the stabilized cell.
                # top_left, top_right, bottom_left, bottom_right
                # unstabilized为原图像各网格顶点坐标 stabilized为叠加稳定向量后的各网格顶点坐标
                unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[
                    cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[
                    cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)

                unstabilized_to_stabilized_homography, _ = cv2.findHomography(
                    unstabilized_cell_bounds, stabilized_cell_bounds)
                stabilized_to_unstabilized_homography, _ = cv2.findHomography(
                    stabilized_cell_bounds, unstabilized_cell_bounds)

                # 行列互换 先列后行
                unstabilized_cell_x_bounds, unstabilized_cell_y_bounds = np.transpose(unstabilized_cell_bounds)
                # 向下取整
                unstabilized_cell_left_x = math.floor(np.min(unstabilized_cell_x_bounds))
                # 向上取整
                unstabilized_cell_right_x = math.ceil(np.max(unstabilized_cell_x_bounds))
                unstabilized_cell_top_y = math.floor(np.min(unstabilized_cell_y_bounds))
                unstabilized_cell_bottom_y = math.ceil(np.max(unstabilized_cell_y_bounds))
                # 设置掩膜 取原始图像(未稳定)中对应的某一网格
                unstabilized_cell_mask = np.zeros((frame_height, frame_width))
                unstabilized_cell_mask[unstabilized_cell_top_y:unstabilized_cell_bottom_y + 1,
                                        unstabilized_cell_left_x:unstabilized_cell_right_x + 1] = 255
                # dsize (x, y)
                stabilized_cell_mask = cv2.warpPerspective(
                    unstabilized_cell_mask, transform_array.dot(unstabilized_to_stabilized_homography), (frame_width, frame_height))
                
                # 以下取全局变化
                # 以稳定后的图像为基准 通过逆单应变换获得稳定后图像与稳定前图像间各像素的对应关系
                # frame_stalizied_x_y中存储每个像素所在位置的索引 shape(-1, 1, 2)
                # shape(cell_unstabilized_x_y) = (frame_height * frame_width, 1, 2)
                cell_unstabilized_x_y = cv2.perspectiveTransform(frame_stabilized_x_y, stabilized_to_unstabilized_homography.dot(transform_array1))
                # shape(cell_stabilized_y_x_to_unstabilized_x_y) = (frame_height, frame_width, 2)
                cell_stabilized_y_x_to_unstabilized_x_y = cell_unstabilized_x_y.reshape((frame_height, frame_width, 2))
                cell_stabilized_y_x_to_unstabilized_x, cell_stabilized_y_x_to_unstabilized_y = np.moveaxis(
                    cell_stabilized_y_x_to_unstabilized_x_y, 2, 0)

                # update the overall stabilized-to-unstabilized map, applying this cell's
                # transformation only to those pixels that are actually part of this cell、
                # 以下取局部 将掩膜内的部分转换为经逆单应矩阵变换后的点的坐标
                # frame_stabilized_y_x_to_unstabilized_x  np.full((frame_height, frame_width), frame_width + 1)
                # (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
                # 向各网格在稳定图像中所对应的区域内存入各像素点对应未稳定图像像素的索引
                frame_stabilized_y_x_to_unstabilized_x = np.where(
                    stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_x, frame_stabilized_y_x_to_unstabilized_x)
                frame_stabilized_y_x_to_unstabilized_y = np.where(
                    stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_y, frame_stabilized_y_x_to_unstabilized_y)
            
        # cv2.remap(img,map1,map2,interpolation) img源图像 map1表示CV_32FC2类型(x,y)点的x map2表示点的y
        stabilized_frame = cv2.remap(
            unstabilized_frame,
            frame_stabilized_y_x_to_unstabilized_x.reshape((frame_height, frame_width, 1)).astype(np.float32),
            frame_stabilized_y_x_to_unstabilized_y.reshape((frame_height, frame_width, 1)).astype(np.float32),
            cv2.INTER_LINEAR, borderValue=(0,0,0)
        )

        stabilized_frames.append(stabilized_frame)

    return stabilized_frames

def multiprocessing_job_for_warp_boost(pos, unstabilized_frames, stabilized_motion_mesh_by_frame_index, index):
    # unstabilized_vertex_x_y and stabilized_vertex_x_y are CV_32FC2 NumPy arrays
    # of the coordinates of the mesh nodes in the stabilized video, indexed from the top left
    # corner and moving left-to-right, top-to-bottom.
    total_frames = len(unstabilized_frames)
    count = total_frames // multicore
    frame_height, frame_width = unstabilized_frames[0].shape[:2]

    if index == multicore - 1:
        unstabilized_frames_in_snipped = unstabilized_frames[index * count: total_frames]
        stabilized_motion_mesh_in_snipped = stabilized_motion_mesh_by_frame_index[index * count: total_frames]
    else:
        unstabilized_frames_in_snipped = unstabilized_frames[index * count: (index + 1) * count]
        stabilized_motion_mesh_in_snipped = stabilized_motion_mesh_by_frame_index[index * count: (index + 1) * count]
    
    num_frames = len(unstabilized_frames_in_snipped)
    
    unstabilized_vertex_x_y = np.array([
            [[math.ceil((frame_width - 1) * (col / (mesh_col_count))), math.ceil((frame_height - 1) * (row / (mesh_row_count)))]]
            for row in range(mesh_row_count + 1)
            for col in range(mesh_col_count + 1)], dtype=np.float32)

    # # row_col_to_unstabilized_vertex_x_y[row, col] and
    # # row_col_to_stabilized_vertex_x_y[row, col]
    # # contain the x and y positions of the vertex at the given row and col
    # row_col_to_unstabilized_vertex_x_y = np.reshape(
    #     unstabilized_vertex_x_y, (mesh_row_count + 1, mesh_col_count + 1, 2))

    # # Construct map from the stabilized frame to the unstabilized frame.
    # # If (x_s, y_s) in the stabilized video is taken from (x_u, y_u) in the unstabilized video, then
    # # stabilized_y_x_to_unstabilized_x[y_s, x_s] = x_u,
    # # stabilized_y_x_to_unstabilized_y[y_s, x_s] = y_u, and
    # # frame_stabilized_y_x_to_stabilized_x_y[y_s, x_s] = [x_u, y_u].
    # # NOTE the inverted coordinate order. This setup allows us to index into map just like
    # # we index into the image. Each point [x_u, y_u] in the array is in OpenCV's expected
    # # order so we can easily apply homographies to those points.
    # # NOTE If a given coordinate's value is not changed by the subsequent steps, then that
    # # coordinate falls outside the stabilized image (so in the output image, that image
    # # should be filled with a border color).
    # # Since these arrays' default values fall outside the unstabilized image, remap will
    # # fill in those coordinates in the stabilized image with the border color as desired.
    # # 数组形状 (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
    # frame_stabilized_y_x_to_unstabilized_x_template = np.full((frame_height, frame_width), frame_width + 1)
    # frame_stabilized_y_x_to_unstabilized_y_template = np.full((frame_height, frame_width), frame_height + 1)
    
    # # shape(frame_stabilized_y_x_to_stabilized_x_y_template) = (frame_height, frame_width, 2)
    # # shape(frame_stabilized_x_y_template) = (frame_height * frame_width, 1, 2)
    # # 元组中存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
    # frame_stabilized_x_y_template = np.swapaxes(
    #     np.indices((frame_width, frame_height), dtype=np.float32), 0, 2).reshape((-1, 1, 2))

    stabilized_frames = []

    x_displacement = O
    if pos > 0:
        x_displacement = -x_displacement
    transform_dist = [x_displacement, 0]
    print(transform_dist)
    transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])
    transform_array1 = np.array([[1, 0, -transform_dist[0]], [0, 1, -transform_dist[1]], [0, 0, 1]])

    for frame_index in range(num_frames):
        unstabilized_frame = unstabilized_frames_in_snipped[frame_index]

        # define handles on mesh in x and y direction
        map_x = np.zeros((frame_height, frame_width), np.float32)
        map_y = np.zeros((frame_height, frame_width), np.float32)
        #
        mesh = unstabilized_vertex_x_y.reshape((mesh_row_count + 1, mesh_col_count + 1, 2))

        row_col_to_stabilized_vertex_x_y = np.reshape(stabilized_motion_mesh_in_snipped[frame_index], (mesh_row_count + 1, mesh_col_count + 1, 2))
        x_motion_mesh, y_motion_mesh = np.moveaxis(row_col_to_stabilized_vertex_x_y, 2, 0)
        # print(x_motion_mesh[0,0])

        pixel_indices_x, pixel_indices_y = np.meshgrid(np.arange(frame_width), np.arange(frame_height))
        pixel_indices = np.concatenate((pixel_indices_x.reshape(-1,1), pixel_indices_y.reshape(-1,1)), axis=1).astype(np.float32).reshape((frame_height, frame_width, 2))
        
        for i in range(mesh_row_count):
            for j in range(mesh_col_count):
                    
                src = [mesh[i, j], mesh[i+1, j],
                        mesh[i, j+1], mesh[i+1, j+1]]
                src = np.asarray(src)

                dst = [[mesh[i, j, 0] + x_motion_mesh[i, j], mesh[i, j, 1] + y_motion_mesh[i, j]],
                    [mesh[i+1, j, 0] + x_motion_mesh[i+1, j], mesh[i+1, j, 1] + y_motion_mesh[i+1, j]],
                    [mesh[i, j+1, 0] + x_motion_mesh[i, j+1], mesh[i, j+1, 1] + y_motion_mesh[i, j+1]],
                    [mesh[i+1, j+1, 0] + x_motion_mesh[i+1, j+1], mesh[i+1, j+1, 1] + y_motion_mesh[i+1, j+1]]]
                dst = np.asarray(dst)
                dst = dst.reshape((4, 2))
                    
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
                
                chosen_pixel = pixel_indices[int(mesh[i, j, 1]): int(mesh[i+1, j, 1]), 
                                             int(mesh[i, j, 0]): int(mesh[i, j+1, 0]), :]

                warped_pixel = cv2.perspectiveTransform(chosen_pixel, transform_array.dot(H))
                # warped_pixel = cv2.perspectiveTransform(chosen_pixel, H)
                map_x[int(mesh[i, j, 1]): int(mesh[i+1, j, 1]),
                      int(mesh[i, j, 0]): int(mesh[i, j+1, 0])] = warped_pixel[:, :, 0]
                map_y[int(mesh[i, j, 1]): int(mesh[i+1, j, 1]), 
                      int(mesh[i, j, 0]): int(mesh[i, j+1, 0])] = warped_pixel[:, :, 1]
        
        stabilized_frame = cv2.remap(unstabilized_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)\
        
        stabilized_frames.append(stabilized_frame)
        #

    return stabilized_frames

def plot_vertex_profiles(origin_paths_1, origin_paths_2, stabilized_paths_1,stabilized_paths_2):
    """
    @param: x_paths is original mesh vertex profiles
    @param: sx_paths is optimized mesh vertex profiles

    Return:
            saves equally spaced mesh vertex profiles
            in directory '<PWD>/results/'
    """

    # plot some vertex profiles
    for i in range(0, origin_paths_1.shape[1]):
        for j in range(0, origin_paths_1.shape[2], 10):
            plt.plot(origin_paths_1[:, i, j, 0], label='original profile A')
            plt.plot(origin_paths_2[:, i, j, 0], label='original profile B')
            plt.plot(stabilized_paths_1[:, i, j, 0], label='optimize with stabilization A')
            plt.plot(stabilized_paths_2[:, i, j, 0], label='optimize with stabilization B')
            plt.xlabel("Frame")
            plt.ylabel("Pixel")
            plt.legend(loc='best')
            plt.grid(visible=True ,color = 'k', linestyle = '--', linewidth = 0.5)
            plt.savefig('pic/paths_unified_lambda2/stabilized'+str(i)+'_'+str(j)+'.png')
            plt.clf()

def plot_motionfield(frame_index, vertex_stabilized_displacements_by_frame_index, vertex_unstabilized_displacements_by_frame_index):
  
    stabilized_motion_mesh_by_frame_index_unfiltered = vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index

    stabilized_motion_mesh_by_frame_index = []
    for stabilized_motion_mesh_unfiltered in stabilized_motion_mesh_by_frame_index_unfiltered:
        # vertex_stabilized_x_velocities_by_row_col = stabilized_motion_mesh_unfiltered[:, :, 0].astype(np.float32)
        # vertex_stabilized_y_velocities_by_row_col = stabilized_motion_mesh_unfiltered[:, :, 1].astype(np.float32)
        # vertex_stabilized_x_velocities_by_row_col = (stabilized_motion_mesh_unfiltered[:, :, 0] + 50).astype(np.uint8)
        # vertex_stabilized_y_velocities_by_row_col = (stabilized_motion_mesh_unfiltered[:, :, 1] + 50).astype(np.uint8)
        vertex_stabilized_x_velocities_by_row_col = (stabilized_motion_mesh_unfiltered[:, :, 0] + 50).astype(np.uint8)
        vertex_stabilized_y_velocities_by_row_col = (stabilized_motion_mesh_unfiltered[:, :, 1] + 50).astype(np.uint8)
        vertex_smoothed_x_velocities_by_row_col = cv2.medianBlur(vertex_stabilized_x_velocities_by_row_col, 9)
        vertex_smoothed_y_velocities_by_row_col = cv2.medianBlur(vertex_stabilized_y_velocities_by_row_col, 9)
        # vertex_smoothed_x_velocities_by_row_col = signal.medfilt2d(vertex_stabilized_x_velocities_by_row_col, kernel_size = 3)
        # vertex_smoothed_y_velocities_by_row_col = signal.medfilt2d(vertex_stabilized_y_velocities_by_row_col, kernel_size = 3)
        stabilized_motion_mesh_by_frame_index.append(np.dstack((vertex_smoothed_x_velocities_by_row_col.astype(np.float32)-50, vertex_smoothed_y_velocities_by_row_col.astype(np.float32)-50)))
    
    stabilized_mediafiltered_motion_mesh = stabilized_motion_mesh_by_frame_index[frame_index]
    stabilized_mediafiltered_motion_mesh_x = stabilized_mediafiltered_motion_mesh[:, :, 0]
    stabilized_mediafiltered_motion_mesh_y = stabilized_mediafiltered_motion_mesh[:, :, 1]
    
    stabilized_motion_mesh_unfiltered = stabilized_motion_mesh_by_frame_index_unfiltered[frame_index]
    stabilized_motion_mesh_unfiltered_x = stabilized_motion_mesh_unfiltered[:, :, 0]
    stabilized_motion_mesh_unfiltered_y = stabilized_motion_mesh_unfiltered[:, :, 1]
    
    fig = plt.figure()

    x = np.arange(0,640+40,40)
    y = np.arange(0,480+30,30)
    x, y = np.meshgrid(x, y)

    ax1 = fig.add_subplot(221, projection="3d")
    surf = ax1.plot_surface(x, y, stabilized_motion_mesh_unfiltered_x, rstride=1, cstride=1, cmap=plt.cm.viridis)
    ax1.set_xlabel("X Label")
    ax1.set_ylabel("Y Label")
    ax1.set_zlabel("Z Label")
    ax1.set_title("stabilized_unfiltered_motion_mesh_x")

    ax2 = fig.add_subplot(222, projection="3d")
    surf = ax2.plot_surface(x, y, stabilized_mediafiltered_motion_mesh_x, rstride=1, cstride=1, cmap=plt.cm.viridis)
    cax1 = fig.add_axes([ax2.get_position().x1+0.05, ax2.get_position().y0, 0.02, ax2.get_position().height])
    fig.colorbar(surf, cax=cax1)
    ax2.set_xlabel("X Label")
    ax2.set_ylabel("Y Label")
    ax2.set_zlabel("Z Label")
    ax2.set_title("stabilized_mediafiltered_motion_mesh_x")
    
    ax3 = fig.add_subplot(223, projection="3d")
    surf = ax3.plot_surface(x, y, stabilized_motion_mesh_unfiltered_x, rstride=1, cstride=1, cmap=plt.cm.viridis)
    ax3.set_xlabel("X Label")
    ax3.set_ylabel("Y Label")
    ax3.set_zlabel("Z Label")
    ax3.set_title("stabilized_unfiltered_motion_mesh_y")
    
    ax4 = fig.add_subplot(224, projection="3d")
    surf = ax4.plot_surface(x, y, stabilized_mediafiltered_motion_mesh_x, rstride=1, cstride=1, cmap=plt.cm.viridis)
    cax2 = fig.add_axes([ax4.get_position().x1+0.05, ax4.get_position().y0, 0.02, ax4.get_position().height])
    fig.colorbar(surf, cax=cax2)
    ax4.set_xlabel("X Label")
    ax4.set_ylabel("Y Label")
    ax4.set_zlabel("Z Label")
    ax4.set_title("stabilized_mediafiltered_motion_mesh_y")
    
    plt.show()

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

def main():
    # TODO get video path from command line args
    input_path = '20221015/6l.mp4'
    output_path = '20221015/6l_origin_10_4-4_10-10_MAGSAC_multi.avi'
    stabilizer = MeshFlowStabilizer(visualize=True)
    cropping_ratio, distortion_score, stability_score = stabilizer.stabilize(
        input_path, output_path,
        adaptive_weights_definition = MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL
    )
    # stability_score = stabilizer.stabilize(
    #     input_path, output_path,
    #     adaptive_weights_definition = MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL
    # )
    print('cropping ratio:', cropping_ratio)
    print('distortion score:', distortion_score)
    print('stability score:', stability_score)


if __name__ == '__main__':

    stabilizer = MeshFlowStabilizer(visualize=True)
    adaptive_weights_definition = MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED

    if not (adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or
        adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED or
        adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH or
        adaptive_weights_definition == MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW):
        raise ValueError(
        'Invalid value for `adaptive_weights_definition`. Expecting value of '
        '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL`, '
        '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED`, '
        '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH`, or'
        '`MeshFlowStabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW`.'
    )

    # 从视频流中读取帧
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--left", type=str, default="/home/sunleyao/sly/UVSScompare/comparedZCH/3/back/output.mp4", help="path to the left video")
    ap.add_argument("-r", "--right", type=str, default="/home/sunleyao/sly/UVSScompare/comparedZCH/3/front/output.mp4", help="path to the right video")
    # ap.add_argument("-l", "--left", type=str, default="real_09/final_3/rear/case3_rear_multiband.mp4", help="path to the left video")
    # ap.add_argument("-r", "--right", type=str, default="real_09/final_3/front/case3_front_multiband.mp4", help="path to the right video")
    args = vars(ap.parse_args())

    vs = cv2.VideoCapture(args["left"])
    num_frames = np.int32(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm.trange(num_frames) as t:
        t.set_description(f'Reading video from <{args["left"]}>')
        left_frames = []
        for frame_index in t:
            success, pixels = vs.read()
            if success:
                # 规范分辨率
                unstabilized_frame = pixels[:, -2560:, :]
                unstabilized_frame = cv2.resize(unstabilized_frame, (2560, 512))
                # unstabilized_frame = cylinder(pixels, W, H)
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
                unstabilized_frame = pixels[:, :1536, :]
                unstabilized_frame = cv2.resize(unstabilized_frame, (1536, 512))
                # unstabilized_frame = cylinder(pixels, W, H)
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
    #前视图后视图,拼接运动场生成
    vertex_left, vertex_right = stabilizer._get_stitch_vertex_displacements_and_homographies(num_frames, left_frames, right_frames)
    #使用雅可比方法，计算网格顶点稳定后的运动场
    vertex_stabilized_stitched_by_frame_index_1 = stabilizer._get_stitch_vertex_displacements(
        num_frames, vertex_left
    )

    vertex_stabilized_stitched_by_frame_index_2 = stabilizer._get_stitch_vertex_displacements(
        num_frames, vertex_right
    )


    stitcher = stitch_utils.stitch_utils(mesh_row_count=mesh_row_count, mesh_col_count=mesh_col_count, 
                                         feature_ellipse_row_count=8, feature_ellipse_col_count=10)

    def motion_field_filter(left_velocity, right_velocity):
        # 中值滤波器去噪
        left_velocity = median_filter(left_velocity, size=3)
        right_velocity = median_filter(right_velocity, size=3)

        # 运动场的平滑过程 调试中
        O = int((- np.ceil(np.median(left_velocity[:, 15, 0])) + np.floor(np.median(right_velocity[:, 1, 0]))) // 2)
        O_l = -int(np.ceil(np.median(left_velocity[:, 15, 0])))
        O_r = int(np.floor(np.median(right_velocity[:, 1, 0])))

        # 边缘的运动场平滑 根据平移量置固定值
        vertex_motion_x_l = left_velocity[:, :15, 0]
        vertex_motion_y_l = left_velocity[:, :15, 1]

        vertex_motion_x_r = right_velocity[:, -15:, 0]
        vertex_motion_y_r = right_velocity[:, -15:, 1]

        vertex_motion_y_l[:, :10] = 0
        vertex_motion_x_l[:, :10] = -O_l
        # # vertex_motion_x_l[:, 1] = -O 
        # vertex_motion_x_l[:, 2] = -O - 15

        vertex_motion_y_r[:, -10:] = 0
        vertex_motion_x_r[:, -10:] = O_r
        # vertex_motion_x_r[:, -2] = O 
        # vertex_motion_x_r[:, -3] = O + 15

        # 均值滤波器
        vertex_motion_x_l_filter = uniform_filter(vertex_motion_x_l, size=3)
        vertex_motion_y_l_filter = uniform_filter(vertex_motion_y_l, size=3)
        # vertex_motion_y_l_filter = vertex_motion_y_l

        vertex_motion_x_r_filter = uniform_filter(vertex_motion_x_r, size=3)
        vertex_motion_y_r_filter = uniform_filter(vertex_motion_y_r, size=3)
        # vertex_motion_y_r_filter = vertex_motion_y_r

        # vertex_motion_x_l_filter[:, :5] = -O
        # vertex_motion_x_r_filter[:, -5:] = O

        # vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l_filter))
        # left_velocity[:, :12, :] = vertex_motion_l_no

        # vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r_filter))
        # right_velocity[:, -12:, :] = vertex_motion_r_no

        vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l_filter))
        # vertex_motion_l_no = np.dstack((vertex_motion_x_l_filter, vertex_motion_y_l))
        left_velocity[:, :15, :] = vertex_motion_l_no

        vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r_filter))
        # vertex_motion_r_no = np.dstack((vertex_motion_x_r_filter, vertex_motion_y_r))
        right_velocity[:, -15:, :] = vertex_motion_r_no

        # vertex_motion_y_l[:, :10] = 0
        # vertex_motion_x_l[:, :10] = -O_l

        # vertex_motion_y_r[:, -10:] = 0
        # vertex_motion_x_r[:, -10:] = O_r

        return left_velocity, right_velocity, O_l, O_r, O

    with tqdm.trange(num_frames) as t:
        t.set_description(f'stitching frames')
        # stitched_frames = []
        # stitched_frames_multiband = []
        for frame_index in t:
            left_frame_ = left_frames[frame_index]
            right_frame_ = right_frames[frame_index]
            
            left_velocity_ = vertex_stabilized_stitched_by_frame_index_1[frame_index]
            right_velocity_ = vertex_stabilized_stitched_by_frame_index_2[frame_index]
            left_velocity_1_filter, right_velocity_1_filter, O_l_1, O_r_1, O_1 = motion_field_filter(left_velocity_, right_velocity_)
            O_1 = O_1 + 5
            # 网格变形
            img_l_1 = stitcher.get_warped_frames_for_stitch(0, left_frame_, left_velocity_1_filter, O_l_1)
            img_r_1 = stitcher.get_warped_frames_for_stitch(1, right_frame_, right_velocity_1_filter, O_r_1)

            # 缝合线选取
            l = np.zeros((512, 2560 + 2 * O_1, 3), np.uint8)
            r = np.zeros((512, 1536 + 2 * O_1, 3), np.uint8)
            l[:, :2560, :] = img_l_1
            r[:, 2 * O_1:, :] = img_r_1
            # stitched_seam_1 = seamcut(l, r)

            # 多频段融合
            flag_half = False
            mask = None
            need_mask =True     
            leveln = 5

            overlap_w = 2560-2*O_1
            stitched_band_1 = multi_band_blending(img_l_1, img_r_1, mask, overlap_w, leveln, flag_half, need_mask)

            # cv2.imwrite('/home/sunleyao/sly/UVSScompare/comparedZCH/1/seam1/'+'{:0{}d}'.format(frame_index + 0, 3) + '.jpg', stitched_seam_1)
            cv2.imwrite('/home/sunleyao/sly/UVSScompare/comparedZCH/3/band1/'+'{:0{}d}'.format(frame_index + 0, 3) + '.jpg', stitched_band_1)

