import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_tps_transform as torch_tps_transform
import ssl
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import os
import scipy.ndimage

import torchvision.transforms as T
resize_512 = T.Resize((512,512))

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

from PIL import Image

import matplotlib.pyplot as plt


# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)

    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp


#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2



# random augmentation
# it seems to do nothing to the performance
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7,1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug  *= color_image

    random_colors = torch.randn(3).uniform_(0.7,1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug  *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug

class MeshMotionManager:
    def __init__(self, window_size=3, sigma=1):
        self.window_size = window_size
        self.sigma = sigma
        self.motion_history = []

    def add_motion(self, mesh_motion):
        if len(self.motion_history) >= self.window_size:
            self.motion_history.pop(0)
        self.motion_history.append(mesh_motion.cpu().detach().numpy())

    def get_smoothed_motion(self):
        if len(self.motion_history) < self.window_size:
            return None  # Not enough data to smooth

        # Stack motions to create a [window_size, batch_size, 13, 13, 2] array
        motions = np.stack(self.motion_history, axis=0)
        smoothed_motions = np.zeros_like(motions)

        for b in range(motions.shape[1]):  # batch_size
            for i in range(motions.shape[2]):  # height
                for j in range(motions.shape[3]):  # width
                    for k in range(motions.shape[4]):  # x, y coordinates
                        # Extract the trajectory for each control point
                        trajectory = motions[:, b, i, j, k]
                        # Apply Gaussian filter
                        smoothed_trajectory = scipy.ndimage.gaussian_filter1d(trajectory, sigma=self.sigma)
                        smoothed_motions[:, b, i, j, k] = smoothed_trajectory

        return torch.tensor(smoothed_motions[-1])  # Return the smoothed current motion

# Initialize MeshMotionManager
mesh_motion_manager = MeshMotionManager(window_size=9, sigma=5)

# for train.py / test.py
def build_model(net, input1_tensor, input2_tensor, is_training = True):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # network
    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(input1_tensor, input2_tensor)
        H_motion, mesh_motion = net(aug_input1_tensor, aug_input2_tensor)
    else:
        H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])

    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()

    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_H = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat, (img_h, img_w))

    H_inv_mat = torch.matmul(torch.matmul(M_tile_inv, torch.inverse(H)), M_tile)
    output_H_inv = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_inv_mat, (img_h, img_w))

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    # 获取第 1 行的 y 坐标数值
    line_1_y = mesh[:, 0, :, 1]
    # 获取第 13 行的 y 坐标数值
    line_13_y = mesh[:, 12, :, 1]

    mean_values_1_y = torch.mean(line_1_y, dim=1, keepdim=True)
    mean_values_13_y = torch.mean(line_13_y, dim=1, keepdim=True)

    mesh[:, 0, :, 1] = mean_values_1_y[:, 0].unsqueeze(-1)
    # print(mesh[:, 0, :, 1])
    # print(mesh[:, 1, :, 1]-mesh[:, 0, :, 1])
    # print(mean_values_13_y-mean_values_1_y)
    # mesh[:, 1, :, 1] = mesh[:, 0, :, 1] + (mesh[:, 1, :, 1]-mesh[:, 0, :, 1])/(mean_values_13_y-mean_values_1_y)*512
    mesh[:, 12, :, 1] = mean_values_13_y[:, 0].unsqueeze(-1)
    
    # mesh[:, 0, :, 1] = rigid_mesh[:, 0, :, 1]
    # mesh[:, 12, :, 1] = rigid_mesh[:, 12, :, 1]

    mesh[:, :, 12, 1] = rigid_mesh[:, :, 12, 1]
    # 获取第 13 列的 x 坐标数值
    column_13_x = mesh[:, :, 12, 0]
    #column_12_x = mesh[:, :, 11, 0]

    mean_values_13 = torch.mean(column_13_x, dim=1, keepdim=True)
    #mean_values_12 = torch.mean(column_12_x, dim=1, keepdim=True)

    # 将第 13 列的 x 坐标设置为 column_13_x 中的第一个值
    mesh[:, :, 12, 0] = mean_values_13[:, 0].unsqueeze(-1)
    #mesh[:, :, 11, 0] = mean_values_12[:, 0].unsqueeze(-1)


    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]

    # calculate the overlapping regions to apply shape-preserving constraints
    overlap = torch_tps_transform.transformer(warp_mesh_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = overlap.permute(0, 2, 3, 1).unfold(1, int(img_h/grid_h), int(img_h/grid_h)).unfold(2, int(img_w/grid_w), int(img_w/grid_w))
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap<0.9, overlap_one, overlap_zero)

    out_dict = {}
    out_dict.update(output_H=output_H, output_H_inv = output_H_inv, warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, mesh1 = rigid_mesh, mesh2 = mesh, overlap = overlap)


    return out_dict

# for train_ft.py
def build_new_ft_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    H_motion, mesh_motion = net(input1_tensor, input2_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    #H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)

    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)
    #mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    mesh[:, 0, :, 1] = rigid_mesh[:, 0, :, 1]
    mesh[:, 12, :, 1] = rigid_mesh[:, 12, :, 1]

    mesh[:, :, 12, 1] = rigid_mesh[:, :, 12, 1]
    # 获取第 13 列的 x 坐标数值
    column_13_x = mesh[:, :, 12, 0]

    mean_values_13 = torch.mean(column_13_x, dim=1, keepdim=True)

    # 将第 13 列的 x 坐标设置为 column_13_x 中的第一个值
    mesh[:, :, 12, 0] = mean_values_13[:, 0].unsqueeze(-1)

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh, norm_rigid_mesh, (img_h, img_w))
    warp_mesh = output_tps[:,0:3,...]
    warp_mesh_mask = output_tps[:,3:6,...]


    out_dict = {}
    out_dict.update(warp_mesh = warp_mesh, warp_mesh_mask = warp_mesh_mask, rigid_mesh = rigid_mesh, mesh = mesh)


    return out_dict

# for train_ft.py
def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh):
    batch_size, _, img_h, img_w = input1_tensor.size()

    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh = torch.stack([mesh[...,0]*img_w/512, mesh[...,1]*img_h/512], 3)

    ######################################
    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    #print(out_width)
    #print(out_height)

    warp1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    warp1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = (input1_tensor+1)*127.5

    mask1 = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    mask1[:,:, int(torch.abs(height_min)):int(torch.abs(height_min))+img_h,  int(torch.abs(width_min)):int(torch.abs(width_min))+img_w] = 255

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()

    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    stitch_tps_out = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask], 1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))
    warp2 = stitch_tps_out[:,0:3,:,:]*127.5
    mask2 = stitch_tps_out[:,3:6,:,:]*255

    stitched = warp1*(warp1/(warp1+warp2+1e-6)) + warp2*(warp2/(warp1+warp2+1e-6))

    #stitched_mesh = draw_mesh_on_warp(stitched[0].cpu().detach().numpy().transpose(1,2,0), mesh_trans[0].cpu().detach().numpy())

    out_dict = {}
    out_dict.update(warp1 = warp1, mask1 = mask1, warp2 = warp2, mask2 = mask2, stitched = stitched)

    return out_dict


# for test_output.py
def build_output_model(net, input1_tensor, input2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
    H_motion, mesh_motion = net(resized_input1, resized_input2)

    H_motion = H_motion.reshape(-1, 4, 2)
    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h+1, grid_w+1, 2)

    # Add current mesh_motion to manager and get smoothed motion
    # mesh_motion_manager.add_motion(mesh_motion)
    # smoothed_mesh_motion = mesh_motion_manager.get_smoothed_motion()

    # if smoothed_mesh_motion is not None:
    #     mesh_motion = smoothed_mesh_motion.cuda()

    mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)


    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = H2Mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    mesh[:, 0, :, 1] = rigid_mesh[:, 0, :, 1]
    mesh[:, 12, :, 1] = rigid_mesh[:, 12, :, 1]

    mesh[:, :, 12, 1] = rigid_mesh[:, :, 12, 1]
    # 获取第 13 列的 x 坐标数值
    column_13_x = mesh[:, :, 12, 0]
    #column_12_x = mesh[:, :, 11, 0]

    mean_values_13 = torch.mean(column_13_x, dim=1, keepdim=True)
    #mean_values_12 = torch.mean(column_12_x, dim=1, keepdim=True)

    # 将第 13 列的 x 坐标设置为 column_13_x 中的第一个值
    mesh[:, :, 12, 0] = mean_values_13[:, 0].unsqueeze(-1)

    width_max = torch.max(mesh[...,0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[...,0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[...,1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[...,1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    #print(out_width)
    #print(out_height)

    # get warped img1
    M_tensor = torch.tensor([[out_width / 2.0, 0., out_width / 2.0],
                      [0., out_height / 2.0, out_height / 2.0],
                      [0., 0., 1.]])
    N_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]])
    if torch.cuda.is_available():
        M_tensor = M_tensor.cuda()
        N_tensor = N_tensor.cuda()
    N_tensor_inv = torch.inverse(N_tensor)

    I_ = torch.tensor([[1., 0., width_min],
                      [0., 1., height_min],
                      [0., 0., 1.]])#.unsqueeze(0)
    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        I_ = I_.cuda()
        mask = mask.cuda()
    I_mat = torch.matmul(torch.matmul(N_tensor_inv, I_), M_tensor).unsqueeze(0)

    homo_output = torch_homo_transform.transformer(torch.cat((input1_tensor+1, mask), 1), I_mat, (out_height.int(), out_width.int()))

    torch.cuda.empty_cache()
    # get warped img2
    mesh_trans = torch.stack([mesh[...,0]-width_min, mesh[...,1]-height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    tps_output = torch_tps_transform.transformer(torch.cat([input2_tensor+1, mask],1), norm_mesh, norm_rigid_mesh, (out_height.int(), out_width.int()))


    out_dict = {}
    out_dict.update(final_warp1=homo_output[:, 0:3, ...]-1, final_warp1_mask = homo_output[:, 3:6, ...], final_warp2=tps_output[:, 0:3, ...]-1, final_warp2_mask = tps_output[:, 3:6, ...], mesh1=rigid_mesh, mesh2=mesh_trans)

    return out_dict



# define and forward
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet1_part2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )


        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)

        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        ssl._create_default_https_context = ssl._create_unverified_context
        resnet50_model = models.resnet.resnet50(pretrained=True)

        if torch.cuda.is_available():
            resnet50_model = resnet50_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2, self.feature_extractor_stage3 = self.get_res50_FeatureMap(resnet50_model)
        #-----------------------------------------

    def get_res50_FeatureMap(self, resnet50_model):

        layers_list = []

        layers_list.append(resnet50_model.conv1)
        layers_list.append(resnet50_model.bn1)
        layers_list.append(resnet50_model.relu)
        layers_list.append(resnet50_model.maxpool)
        
        layers_list.append(resnet50_model.layer1)
        layers_list.append(resnet50_model.layer2)

        feature_extractor_stage1 = nn.Sequential(*layers_list)

        feature_extractor_stage2 = nn.Sequential(resnet50_model.layer3)

        feature_extractor_stage3 = nn.Sequential(resnet50_model.layer4)

        return feature_extractor_stage1, feature_extractor_stage2, feature_extractor_stage3

    # forward
    def forward(self, input1_tesnor, input2_tesnor):
        batch_size, _, img_h, img_w = input1_tesnor.size()
        # print(f'input1_tesnor: {input1_tesnor.shape}')
        # print(f'input2_tesnor: {input2_tesnor.shape}')

        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)
        # print(f'feature_1_64: {feature_1_64.shape}')
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)
        # print(f'feature_1_32: {feature_1_32.shape}')
        feature_1_16 = self.feature_extractor_stage3(feature_1_32)
        # print(f'feature_1_16: {feature_1_16.shape}')

        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)
        # print(f'feature_2_64: {feature_2_64.shape}')
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)
        # print(f'feature_2_32: {feature_2_32.shape}')
        feature_2_16 = self.feature_extractor_stage3(feature_2_32)
        # print(f'feature_2_16: {feature_2_16.shape}')

        # feature_map_np = feature_2_32.detach().cpu().numpy()

        # ## 将特征图保存到文件夹中
        # # 获取特征图的形状信息
        # batch_size, num_channels, height, width = feature_map_np.shape

        # # 创建保存特征图的目录
        # save_dir = '/home/sunleyao/sly/UDIS2-main/UDIS2-main-y-5/Warp/feature_maps'
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # # 将每个通道的特征图分别保存到指定目录
        # for channel in range(num_channels):
        #     # 获取当前通道的特征图并进行缩放和类型转换
        #     current_channel_image = feature_map_np[0, channel]
        #     scaled_channel_image = ((current_channel_image - np.min(current_channel_image)) /
        #                             (np.max(current_channel_image) - np.min(current_channel_image)) * 255).astype(np.uint8)

        #     # 创建 PIL 图像对象
        #     channel_image = Image.fromarray(scaled_channel_image)

        #     # 生成保存文件的路径
        #     save_path = os.path.join(save_dir, f'feature_channel_{channel + 1}.png')

        #     # 保存当前通道的特征图
        #     channel_image.save(save_path)

        # print(f'All feature maps have been saved to {save_dir}')

        # 计算多尺度特征的匹配
        correlation_32 = self.CCL(feature_1_32, feature_2_32)
        # print(f'correlation_32: {correlation_32.shape}')
        correlation_16 = self.CCL(feature_1_16, feature_2_16)
        # print(f'correlation_16: {correlation_16.shape}')

        # 融合多尺度特征
        
        # ######### stage 1 ###############
        # correlation_32 = self.CCL(feature_1_32, feature_2_32)
        # print(f'correlation_32: {correlation_32.shape}')
        # temp_1 = self.regressNet1_part1(correlation_32)
        # print(f'correlation_32: {correlation_32.shape}')

        # 上采样correlation_16到correlation_32的尺寸
        correlation_16_upsampled = F.interpolate(correlation_16, size=correlation_32.shape[2:], mode='bilinear', align_corners=True)
        # print(f'correlation_16_upsampled: {correlation_16_upsampled.shape}')

        # 加法融合
        correlation_fused = correlation_32 + correlation_16_upsampled
        # print(f'correlation_fused: {correlation_fused.shape}')

        ######### stage 1 ###############
        temp_1 = self.regressNet1_part1(correlation_fused)
        # print(f'temp_1: {temp_1.shape}')
        
        # temp_1 = self.regressNet1_part1(correlation)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        offset_1 = self.regressNet1_part2(temp_1)
        H_motion_1 = offset_1.reshape(-1, 4, 2)

        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]])

        if torch.cuda.is_available():
            M_tensor = M_tensor.cuda()

        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        warp_feature_2_64 = torch_homo_transform.transformer(feature_2_64, H_mat, (int(img_h/8), int(img_w/8)))

        feature_map_np = feature_2_16.detach().cpu().numpy()
        
        
        ## 将特征图保存到文件夹中
        # 获取特征图的形状信息
        batch_size, num_channels, height, width = feature_map_np.shape

        # 创建保存特征图的目录
        save_dir = '/home/sunleyao/sly/UDIS2-main/UDIS2++-experiment-visualFMT/Warp/feature_maps_2_16'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 将每个通道的特征图分别保存到指定目录
        for channel in range(num_channels):
            # 获取当前通道的特征图并进行缩放和类型转换
            current_channel_image = feature_map_np[0, channel]
            scaled_channel_image = ((current_channel_image - np.min(current_channel_image)) /
                                    (np.max(current_channel_image) - np.min(current_channel_image)) * 255).astype(np.uint8)

            # 创建 PIL 图像对象
            channel_image = Image.fromarray(scaled_channel_image)

            # 生成保存文件的路径
            save_path = os.path.join(save_dir, f'feature_channel_{channel + 1}.png')

            # 保存当前通道的特征图
            channel_image.save(save_path)

        print(f'All feature maps have been saved to {save_dir}')

        
        ######### stage 2
        correlation_64 = self.CCL(feature_1_64, warp_feature_2_64)
        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        offset_2 = self.regressNet2_part2(temp_2)

        return offset_1, offset_2


    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)
        #print(match_vol .size())

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow
