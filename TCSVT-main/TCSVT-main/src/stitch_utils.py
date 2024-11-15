import numpy as np
import cv2
import kornia as K
import kornia.feature as KF
import torch
import math
import statistics
import time

def measure_performance(method):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        print(method.__name__+' has taken: '+str(end_time-start_time)+' sec')
        return result
    return timed

class stitch_utils:
    def __init__(self, mesh_row_count=12, mesh_col_count=8,  # 网格行数与列数，顶点数各加1
                 feature_ellipse_row_count=8, feature_ellipse_col_count=6,  # 每个特征点所占椭圆覆盖的行\列
                 homography_min_number_corresponding_features=12,
                 color_outside_image_area_bgr=(0, 0, 0)  # 稳定图像后设置背景色，避免图像无法覆盖窗口
                #  overlap_region = 150
                 ):
        self.mesh_col_count = mesh_col_count
        self.mesh_row_count = mesh_row_count
        self.feature_ellipse_row_count = feature_ellipse_row_count
        self.feature_ellipse_col_count = feature_ellipse_col_count
        self.homography_min_number_corresponding_features = homography_min_number_corresponding_features
        self.color_outside_image_area_bgr = color_outside_image_area_bgr
        # self.overlap_region = overlap_region


    # LoFTR进行inter-frame的特征匹配 subframe_offset是截取subframe的偏移量
    def loftr_in_subframe(self, early_subframe, late_subframe, subframe_offset):

        img1 = K.image_to_tensor(early_subframe, False).float() /255.
        img1 = K.color.bgr_to_rgb(img1)

        img2 = K.image_to_tensor(late_subframe, False).float() /255.
        img2 = K.color.bgr_to_rgb(img2)
        
        matcher = KF.LoFTR(pretrained='outdoor')
        # matcher.cuda()
        matcher = matcher.cuda() if torch.cuda.is_available() else matcher
        matcher.eval()

        # input_dict = {"image0": K.color.rgb_to_grayscale(img1.cuda()), # LofTR works on grayscale images only 
        #             "image1": K.color.rgb_to_grayscale(img2.cuda())}
        input_dict = {
            "image0": K.color.rgb_to_grayscale(img1.cuda() if torch.cuda.is_available() else img1),
            "image1": K.color.rgb_to_grayscale(img2.cuda() if torch.cuda.is_available() else img2)
        }
        #在不计算梯度的上下文中进行特征匹配，以节省内存和计算资源。
        with torch.no_grad():
            correspondences = matcher(input_dict)

        # for k,v in correspondences.items():
        #     print (k)
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

        _, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.8, 0.9999, 100000)
        # print(inliers.flatten().astype(dtype = bool).shape)
        filter_mask = inliers.flatten().astype(dtype = bool)
        mkpts0_filtered = mkpts0[filter_mask]
        mkpts1_filtered = mkpts1[filter_mask]

        mkpts0_filtered = np.float32(mkpts0_filtered[:, np.newaxis, :])
        mkpts1_filtered = np.float32(mkpts1_filtered[:, np.newaxis, :])

        return (mkpts0_filtered + subframe_offset, mkpts1_filtered)


    #  获取各顶点对应图像坐标  #
    def get_vertex_x_y(self, frame_width, frame_height):

        # CV_32FC2 32位浮点型双通道矩阵
        # 维度((mesh_row_count + 1)*(mesh_col_count + 1), 1, 2)
        return np.array([
            [[math.ceil((frame_width - 1) * (col / (self.mesh_col_count))), math.ceil((frame_height - 1) * (row / (self.mesh_row_count)))]]
            for row in range(self.mesh_row_count + 1)
            for col in range(self.mesh_col_count + 1)
        ], dtype=np.float32)
    

    #  获取重叠区域(裁切common field)对应的匹配特征点对以及全局H  #
    def get_matched_features_and_homography_for_stitch(self, early_frame, late_frame):

        frame_height, frame_width = early_frame.shape[:2]
        # 裁切重叠区域
        common_field = 540
        early_subframe = early_frame[0: frame_height, (frame_width - common_field): frame_width]
        late_subframe = late_frame[0: frame_height, 0: common_field]
        # 左上角坐标
        subframe_offset = [(frame_width - common_field), 0]
        # 获得匹配特征点对
        early_features, late_features = self.loftr_in_subframe(early_subframe, late_subframe, subframe_offset)
        # 特征点对的虚拟中点
        middle_points = (early_features + late_features) / 2

        if len(early_features) < self.homography_min_number_corresponding_features:
            print('No enough features!')
            return (None, None, None)

        # 获取单应性矩阵 MAGSAC替代RANSAC
        early_to_late_homography_l, _ = cv2.findHomography(early_features, middle_points, cv2.USAC_MAGSAC)
        early_to_late_homography_r, _ = cv2.findHomography(late_features, middle_points, cv2.USAC_MAGSAC)

        return (early_features, late_features, middle_points, early_to_late_homography_l, early_to_late_homography_r)
    

    #  获取各顶点在约定范围(椭圆)内由各特征点引起的运动向量  #
    def get_vertex_nearby_feature_residual_velocities(self, frame_width, frame_height, early_features, late_features, early_to_late_homography): 
    
        #  初始化数组 存储各顶点在椭圆范围内传播的运动向量
        vertex_nearby_feature_x_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]
        vertex_nearby_feature_y_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]

        if early_features is not None:

            # 计算特征点自身的运动(去除全局运动) v_p~ = p - F_t*p^
            feature_residual_velocities = late_features - cv2.perspectiveTransform(early_features, early_to_late_homography)
            # 首列存当前帧特征点的位置 次列存帧间相对运动
            feature_positions_and_residual_velocities = np.c_[late_features, feature_residual_velocities]

            # 特征运动传播
            for feature_position_and_residual_velocity in feature_positions_and_residual_velocities:
                feature_x, feature_y, feature_residual_x_velocity, feature_residual_y_velocity = feature_position_and_residual_velocity[0]
                feature_row = (feature_y / frame_height) * self.mesh_row_count
                feature_col = (feature_x / frame_width) * self.mesh_col_count

                # 顶行
                ellipse_top_row_inclusive = max(0, math.ceil(feature_row - self.feature_ellipse_row_count / 2))
                # 底行
                ellipse_bottom_row_exclusive = 1 + min(self.mesh_row_count, math.floor(feature_row + self.feature_ellipse_row_count / 2))

                for vertex_row in range(ellipse_top_row_inclusive, ellipse_bottom_row_exclusive):
                    # 基于椭圆参数方程 计算每一行所包含的列
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
    

    #  获取用于拼接的各网格顶点运动向量  #
    # @measure_performance
    def get_velocities_for_stitch(self, early_frame, early_features, late_features, early_to_late_homography): 

        frame_height, frame_width = early_frame.shape[:2]
        vertex_x_y = self.get_vertex_x_y(frame_width, frame_height)
        # 各顶点由全局运动引起的运动矢量
        vertex_global_velocities = cv2.perspectiveTransform(vertex_x_y, early_to_late_homography) - vertex_x_y
        # 变换矩阵结构reshape2(mesh_row_count + 1, mesh_col_count + 1, 2)
        vertex_global_velocities_by_row_col = np.reshape(vertex_global_velocities, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        vertex_global_x_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 0]
        vertex_global_y_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 1]

        # 获取各顶点在约定范围(椭圆)内由各特征点引起的运动向量
        vertex_nearby_feature_residual_x_velocities_by_row_col, vertex_nearby_feature_residual_y_velocities_by_row_col = self.get_vertex_nearby_feature_residual_velocities(
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
        # 合并
        vertex_smoothed_velocities_by_row_col = np.dstack((vertex_x_velocities_by_row_col, vertex_y_velocities_by_row_col))

        return (vertex_smoothed_velocities_by_row_col, early_to_late_homography)
    

    #  网格变形  #
    # @measure_performance
    def get_warped_frames_for_stitch(self, pos, unstabilized_frame, stabilized_motion_mesh, x_displacement):

        frame_height, frame_width = unstabilized_frame.shape[:2]

        unstabilized_vertex_x_y = self.get_vertex_x_y(frame_width, frame_height)
        # shape ((mesh_row_count + 1)* (mesh_col_count + 1), 1, 2) -> (mesh_row_count + 1, mesh_col_count + 1, 2)
        row_col_to_unstabilized_vertex_x_y = np.reshape(
            unstabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        # shape  (mesh_row_count + 1 * mesh_col_count, 1, 2)
        # stabilized_motion_mesh = np.reshape(stabilized_motion_mesh, (-1, 1, 2))

        # 将顶点前后运动向量的差值叠加到各顶点坐标上 得到拼接需求的顶点坐标
        # unstabilized_vertex_x_y (mesh_row_count + 1 * mesh_col_count + 1, 1, 2)
        # stabilized_motion_mesh (mesh_row_count + 1 * mesh_col_count + 1, 1, 2)
        # stabilized_vertex_x_y = unstabilized_vertex_x_y + stabilized_motion_mesh

        # 重构数组结构
        # row_col_to_stabilized_vertex_x_y = np.reshape( stabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        row_col_to_stabilized_vertex_x_y = row_col_to_unstabilized_vertex_x_y + stabilized_motion_mesh

        # 数组形状 (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
        frame_stabilized_y_x_to_unstabilized_x = np.full((frame_height, frame_width), frame_width + 1)
        frame_stabilized_y_x_to_unstabilized_y = np.full((frame_height, frame_width), frame_height + 1)

        # shape(frame_stabilized_y_x_to_stabilized_x_y) = (frame_height, frame_width, 2)
        frame_stabilized_y_x_to_stabilized_x_y = np.swapaxes(
            np.indices((frame_width, frame_height), dtype=np.float32), 0, 2)
        # shape(frame_stabilized_x_y) = (frame_height * frame_width, 1, 2)
        # 存入像素点对应索引[frame_width列 frame_height行] 从左上角像素点开始 从左至右 先列后行
        frame_stabilized_x_y = frame_stabilized_y_x_to_stabilized_x_y.reshape((-1, 1, 2))

        # 平移矩阵
        # x_displacement = self.overlap_region
        if pos > 0:
            x_displacement = -x_displacement
        transform_dist = [x_displacement, 0]
        transform_array = np.array([[1, 0, transform_dist[0]], [0, 1, transform_dist[1]], [0, 0, 1]])
        transform_array1 = np.array([[1, 0, -transform_dist[0]], [0, 1, -transform_dist[1]], [0, 0, 1]])

        # TODO parallelize
        # 由稳定后的顶点坐标和稳定后的顶点坐标，计算单应性矩阵
        for cell_top_left_row in range(self.mesh_row_count):
            for cell_top_left_col in range(self.mesh_col_count):

                # 计算通过运动向量变换后的网格的MASK 后续计算MASK中各像素与原图像对应网格中各像素的对应
                # 取出4个顶点坐标 依次为top_left, top_right, bottom_left, bottom_right
                # unstabilized为原图像各网格顶点坐标 stabilized为叠加运动向量后的各网格顶点坐标
                unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[
                    cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[
                    cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                # 计算单应性矩阵
                unstabilized_to_stabilized_homography, _ = cv2.findHomography(unstabilized_cell_bounds, stabilized_cell_bounds)
                stabilized_to_unstabilized_homography, _ = cv2.findHomography(stabilized_cell_bounds, unstabilized_cell_bounds)

                # 行列互换 先列后行
                unstabilized_cell_x_bounds, unstabilized_cell_y_bounds = np.transpose(unstabilized_cell_bounds)
                unstabilized_cell_left_x = math.floor(np.min(unstabilized_cell_x_bounds))
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
                # (2, frame_width, frame_height)
                cell_stabilized_y_x_to_unstabilized_x, cell_stabilized_y_x_to_unstabilized_y = np.moveaxis(
                    cell_stabilized_y_x_to_unstabilized_x_y, 2, 0)

                # 以下取局部 将掩膜内的部分转换为经逆单应矩阵变换后的点的坐标
                # frame_stabilized_y_x_to_unstabilized_x  np.full((frame_height, frame_width), frame_width + 1)
                # (frame_height, frame_width) 填充值为frame_width + 1/frame_height + 1
                # 向各网格在稳定图像中所对应的区域内存入各像素点对应未稳定图像像素的索引
                frame_stabilized_y_x_to_unstabilized_x = np.where(
                    stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_x, frame_stabilized_y_x_to_unstabilized_x)
                frame_stabilized_y_x_to_unstabilized_y = np.where(
                    stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_y, frame_stabilized_y_x_to_unstabilized_y)

        # cv2.remap(img,map1,map2,interpolation) map1表示CV_32FC2类型(x,y)点的x map2表示CV_32FC2类型(x,y)点的y
        warped_frame = cv2.remap(
            unstabilized_frame,
            frame_stabilized_y_x_to_unstabilized_x.reshape((frame_height, frame_width, 1)).astype(np.float32),
            frame_stabilized_y_x_to_unstabilized_y.reshape((frame_height, frame_width, 1)).astype(np.float32),
            cv2.INTER_LINEAR,
            borderValue=(0, 0, 0)
        )

        return warped_frame
    
    def proj_err(self, w, h, early_features, late_features, velocity):
        row_size = h // self.mesh_row_count
        col_size = w // self.mesh_col_count

        unstabilized_vertex_x_y = self.get_vertex_x_y(w, h)
        row_col_to_unstabilized_vertex_x_y = np.reshape(unstabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        row_col_to_stabilized_vertex_x_y = row_col_to_unstabilized_vertex_x_y + velocity

        H_index = np.empty((self.mesh_row_count, self.mesh_col_count), dtype=object)
        for cell_top_left_row in range(self.mesh_row_count):
            for cell_top_left_col in range(self.mesh_col_count):

                # 计算通过运动向量变换后的网格的MASK 后续计算MASK中各像素与原图像对应网格中各像素的对应
                # 取出4个顶点坐标 依次为top_left, top_right, bottom_left, bottom_right
                # unstabilized为原图像各网格顶点坐标 stabilized为叠加运动向量后的各网格顶点坐标
                unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[
                    cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[
                    cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                # 计算单应性矩阵
                unstabilized_to_stabilized_homography, _ = cv2.findHomography(unstabilized_cell_bounds, stabilized_cell_bounds)
                H_index[cell_top_left_row, cell_top_left_col] = unstabilized_to_stabilized_homography
        # print(H_index[0,0])
        sum = 0
        for i in range(early_features.shape[0]):
            left = int(early_features[i,0,0] // col_size)
            top = int(early_features[i,0,1] // row_size)
            point = np.reshape(early_features[i,0], (1,1,2))
            # print(point)
            displace = cv2.perspectiveTransform(point, H_index[top,left]) - point
            # print(displace.shape)
            err = late_features[i,0] - displace[0,0] - early_features[i,0]
            err1 = math.sqrt(err[0]**2 + err[1]**2)
            sum = sum + err1
        print(sum/early_features.shape[0])
        return sum/early_features.shape[0]
