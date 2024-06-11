# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import numpy as np
from lietorch import SO3, SE3
class CameraModel(nn.Module):
    def __init__(self, num_cams, pose_dim=9, init_pose=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param cfg: config argument options
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(CameraModel, self).__init__()
        self.register_buffer('init_pose', init_pose)

        img_emb_dim = 1
        pose_dim = 9
        self.img_embedding = nn.Embedding(num_cams, img_emb_dim)
        self.cam_mlp = nn.Sequential(
                        nn.Linear(img_emb_dim+pose_dim, 32),
                        nn.ReLU(),
                        nn.Linear(32, pose_dim))
        self.factor = 0.01

    def forward(self, cam_id):
        cam_id = torch.int(cam_id).cuda()
        img_emb = self.img_embedding(cam_id)
        init_cam = self.init_pose[cam_id]
        cam_residual = self.cam_mlp(torch.cat([img_emb, init_cam], dim=1))
        cam = init_cam + self.factor * cam_residual
        cam = SE3(cam)

        return cam


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)

def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w
    
def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output