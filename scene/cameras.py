# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_inria.md file.
#
# For inquiries contact  george.drettakis@inria.fr


import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View3

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, intrinsics=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 do_grad=False, is_co3d=False,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.intrinsics = intrinsics.astype(np.float32)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        
        self.gt_alpha_mask = gt_alpha_mask

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if is_co3d:
            self.world_view_transform = torch.tensor(getWorld2View3(R, T, trans, scale)).transpose(0, 1).cuda()
            w, h = self.image_width, self.image_height
            fx, fy, cx, cy = self.intrinsics[0, 0], self.intrinsics[1, 1], \
                                self.intrinsics[0, 2], self.intrinsics[1, 2]
            far, near = self.zfar, self.znear
            opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                        [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                        [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                        [0.0, 0.0, 1.0, 0.0]]).cuda().float().transpose(0, 1)
            self.projection_matrix = opengl_proj

        else:
            self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()

        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

