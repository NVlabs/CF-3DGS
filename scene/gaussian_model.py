# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_inria.md file.
#
# For inquiries contact  george.drettakis@inria.fr


import torch
from lietorch import SO3, SE3, Sim3
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh
import math


from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.camera_conversion import (
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_to_matrix,
    rotation_6d_to_matrix,
    lie,
)

import pdb


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, rot_type='6d', view_dependent=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.rotate_xyz = False
        self.rotate_seq = False
        self.rot_type = rot_type
        self.seq_idx = 0
        self.view_dependent = view_dependent
        print("Rotation type : {}".format(self.rot_type))
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        # if self.rotate_seq:
        #     R = self.R[self.seq_idx]
        #     _rotation = quaternion_mul(R[None].repeat(self._rotation.shape[0], 1),
        #                                self._rotation)
        #     return self.rotation_activation(_rotation)
        # elif self.rotate_xyz:
        #     num_parts = self.labels.max().item() + 1
        #     _rotation = self._rotation.clone()
        #     for pid in range(1, num_parts):
        #         label_select = self.labels == pid
        #         _rotation[label_select] = quaternion_mul(self.R[pid-1][None].repeat(label_select.sum(), 1),
        #                                self._rotation[label_select])
        #     return self.rotation_activation(_rotation)
        # else:
        #     return self.rotation_activation(self._rotation)
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        # return (self.R @ self._xyz.t()).t() + self.T
        if self.rotate_xyz:
            num_parts = self.labels.max().item() + 1
            xyz = self._xyz.clone()
            for pid in range(1, num_parts):
                if self.rot_type == '6d':
                    R = rotation_6d_to_matrix(self.R[pid-1])
                elif self.rot_type == 'axis':
                    R = lie.so3_to_SO3(self.R[pid-1])
                elif self.rot_type == 'quat':
                    R = quaternion_to_matrix(
                        self.rotation_activation(self.R[pid-1][None]))[0]
                else:
                    R = self.R[pid-1]
                try:
                    xyz = (R @ xyz.t()).t() + self.T[pid-1]

                except:
                    pdb.set_trace()
            return xyz
        elif self.rotate_seq:
            if self.rot_type == '6d':
                R = rotation_6d_to_matrix(self.current_R)
            elif self.rot_type == 'axis':
                # R = axis_angle_to_matrix(self.current_R)
                R = lie.so3_to_SO3(self.current_R)
            elif self.rot_type == 'quat':
                R = quaternion_to_matrix(
                    self.rotation_activation(self.current_R))
            else:
                R = self.current_R

            T = self.current_T
            xyz_old = self._xyz.clone()
            xyz = (R @ xyz_old.t()).t() + T
            return xyz
        else:
            return self._xyz

    def get_RT(self, idx=None):
        if getattr(self, 'R', None) is None:
            return torch.eye(4, device="cuda")

        if self.rotate_xyz:
            idx = 0
        else:
            idx = self.seq_idx if idx is None else idx

        if self.rot_type == "6d":
            R = rotation_6d_to_matrix(self.R[idx])
        elif self.rot_type == "axis":
            R = lie.so3_to_SO3(self.R[idx])
        elif self.rot_type == "quat":
            R = quaternion_to_matrix(
                self.rotation_activation(self.R[idx][None]))[0]
        else:
            R = self.R[idx]
        t = self.T[idx].squeeze()
        if t.dim() == 1:
            t = t[:, None]
        Rt = torch.cat([R, t], dim=-1)
        Rt = torch.cat([Rt, torch.tensor([[0, 0, 0, 1]], dtype=Rt.dtype,
                                         device=Rt.device)], dim=0)
        return Rt

    def set_seq_idx(self, idx):
        if idx < 0:
            self.rotate_seq = False
            self.rotate_xyz = False
        else:
            self.seq_idx = idx
            self.current_R = self.R[idx]
            self.current_T = self.T[idx]

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_noview(self):
        features_dc = self._features_dc.squeeze()
        return features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if self.view_dependent:
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color[:, :3]
        features[:, 3:, 1:] = 0.0
        # features = torch.cat([features, features], dim=1)
        print("Number of points at initialisation : ",
              fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(
            np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(
            1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def fix_position(self):

        self._xyz = nn.Parameter(self._xyz.detach().requires_grad_(False))
        self._features_dc = nn.Parameter(
            self._features_dc.detach().requires_grad_(False))
        self._features_rest = nn.Parameter(
            self._features_rest.detach().requires_grad_(False))

        self._scaling = nn.Parameter(
            self._scaling.detach().requires_grad_(False))
        self._rotation = nn.Parameter(
            self._rotation.detach().requires_grad_(False))
        self._opacity = nn.Parameter(
            self._opacity.detach().requires_grad_(False))
        _xyz = self._xyz.detach().clone()

    def training_setup(self, training_args, fix_pos=False,
                       fix_feat=False, fit_pose=False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        l = []

        # if not fix_pos:
        # l.append({'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"})

        _xyz_lr = training_args.position_lr_init * \
            self.spatial_lr_scale if not fix_pos else 0.0
        feat_lr_factor = 1.0 if not fix_feat else 0.0
        # _rotation_lr = training_args.rotation_lr if not fix_pos else 0.0
        l += [
            {'params': [self._xyz], 'lr': _xyz_lr, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr *
                feat_lr_factor, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr /
                20.0 * feat_lr_factor, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr *
                feat_lr_factor, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr *
                feat_lr_factor, "name": "scaling"},
            {'params': [self._rotation],
                'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        if fit_pose:
            rotation_lr_factor = 0.1 if (
                not fix_pos) or (not fix_feat) else 1.0
            if self.rotate_seq:
                self.camera_optimizer = []
                for idx in range(len(self.R)):
                    l_cam = [
                        {'params': [self.R[idx]],
                            'lr': training_args.rotation_lr, "name": "R"},
                        {'params': [self.T[idx]],
                            'lr': training_args.rotation_lr, "name": "T"},
                    ]
                    self.camera_optimizer.append(
                        torch.optim.Adam(l_cam, lr=0.0, eps=1e-15))
            else:
                l_cam = [
                    {'params': [self.R],
                        'lr': training_args.rotation_lr, "name": "R"},
                    {'params': [self.T],
                        'lr': training_args.rotation_lr, "name": "T"},
                ]
                self.camera_optimizer = [
                    torch.optim.Adam(l_cam, lr=0.0, eps=1e-15)]
            self.camera_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                           lr_final=training_args.rotation_lr * 0.1,
                                                           lr_delay_mult=0.1,
                                                           max_steps=training_args.position_lr_max_steps)
        else:
            self.camera_optimizer = None

    def training_setup_fix_position(self, training_args, gaussian_rot=True):
        # self.percent_dense = training_args.percent_dense
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if gaussian_rot:
            lr_factor = 1.0
        else:
            lr_factor = 0.1

        l = [
            # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self.R], 'lr': training_args.rotation_lr, "name": "R"},
            {'params': [self.T], 'lr': training_args.rotation_lr, "name": "T"},
        ]
        if gaussian_rot:
            l += [
                {'params': [self._rotation],
                    'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def init_RT(self, pcd=None, pose=None):
        if pcd is None:
            labels = torch.ones(
                (self.get_xyz.shape[0]), dtype=torch.long, device="cuda")
        else:
            labels = pcd.labels
            labels = torch.tensor(labels, dtype=torch.long, device="cuda")

        num_parts = labels.max().item() + 1
        self.labels = labels
        # R = torch.stack([torch.eye(3, device="cuda") for _ in range(1, num_parts)])
        # if self.quaternion:
        #     R = matrix_to_quaternion(R)

        if self.rot_type == "axis":
            if pose is not None:
                # R_init = matrix_to_axis_angle(pose[:3, :3])
                R_init = lie.SO3_to_so3(pose[:3, :3])
            else:
                R_init = torch.zeros(3, device="cuda", dtype=torch.float32)
            R = torch.stack([R_init for _ in range(1, num_parts)])
        elif self.rot_type == "6d":
            if pose is not None:
                R_init = matrix_to_rotation_6d(pose[:3, :3])
            else:
                R_init = torch.eye(2, 3, device="cuda").reshape(-1)
            R = torch.stack([R_init for _ in range(1, num_parts)])
        elif self.rot_type == "quat":
            if pose is not None:
                R_init = matrix_to_quaternion(pose[:3, :3])
            else:
                R_init = torch.tensor(
                    [1, 0, 0, 0], dtype=torch.float32, device="cuda")
            R = torch.stack([R_init for _ in range(1, num_parts)])
        else:
            if pose is not None:
                R_init = pose[:3, :3]
            else:
                R_init = torch.eye(3, device="cuda")
            R = torch.stack([R_init for _ in range(1, num_parts)])

        if pose is not None:
            T_init = pose[:3, 3].unsqueeze(0)
        else:
            T_init = torch.zeros(1, 3, device="cuda")

        T = torch.cat([T_init for _ in range(1, num_parts)])
        self.R = nn.Parameter(R.requires_grad_(True))
        self.T = nn.Parameter(T.requires_grad_(True))

        self.rotate_xyz = True
        self.rotate_seq = False

    def init_RT_seq(self, seq_len, pose=None):
        if pose is None:
            # T = torch.stack([torch.zeros(1, 3, device="cuda") for _ in range(seq_len)])
            if self.rot_type == "axis":
                R = torch.zeros(3, device="cuda", dtype=torch.float32)
            elif self.rot_type == "6d":
                R = torch.eye(2, 3, device="cuda").reshape(-1)
            elif self.rot_type == "quat":
                # R = torch.zeros(seq_len, 4, device="cuda", dtype=torch.float32)
                R = torch.tensor(
                    [1, 0, 0, 0], dtype=torch.float32, device="cuda")
            else:
                R = torch.eye(3, device="cuda")

            T = torch.zeros(1, 3, device="cuda", dtype=torch.float32)
            self.R = [nn.Parameter(R.requires_grad_(True))
                      for _ in range(seq_len)]
            self.T = [nn.Parameter(T.requires_grad_(True))
                      for _ in range(seq_len)]
        else:
            assert pose.shape[0] == seq_len
            pose = pose.cuda()
            R = pose[:, :3, :3]
            if self.rot_type == "axis":
                # R = matrix_to_axis_angle(R)
                # R = lie.SO3_to_so3(R)
                R = lie
            elif self.rot_type == "6d":
                R = matrix_to_rotation_6d(R)
            elif self.rot_type == "quat":
                R = matrix_to_quaternion(R)
            T = pose[:, :3, 3].unsqueeze(1)
            self.R = [nn.Parameter(R[i].requires_grad_(True))
                      for i in range(seq_len)]
            self.T = [nn.Parameter(T[i].requires_grad_(True))
                      for i in range(seq_len)]

        self.rotate_seq = True
        self.rotate_xyz = False

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
        # if getattr(self, "camera_optimizer", None) is not None:
        #     for param_group in self.camera_optimizer.param_groups:
        #         if param_group["name"] == "R":
        #             lr = self.xyz_scheduler_args(iteration)
        #             param_group['lr'] = lr
        #             return lr

    def update_learning_rate_camera(self, cam_idx, iteration):
        ''' Learning rate scheduling per step '''
        if isinstance(self.camera_optimizer, list):
            for param_group in self.camera_optimizer[cam_idx].param_groups:
                lr = self.camera_scheduler_args(iteration)
                param_group['lr'] = lr
        else:
            for param_group in self.camera_optimizer.param_groups:
                lr = self.camera_scheduler_args(iteration)
                param_group['lr'] = lr

    def freeze_camera(self):
        for param_group in self.camera_optimizer.param_groups:
            param_group['lr'] = 0.0

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(
            1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(
            1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4')
                      for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def export_gaussian(self, ):
        gassuian = {}
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        max_radii2D = self.max_radii2D.detach().cpu().numpy()
        gassuian['xyz'] = xyz
        gassuian['normals'] = normals
        gassuian['f_dc'] = f_dc
        gassuian['f_rest'] = f_rest
        gassuian['opacities'] = opacities
        gassuian['scale'] = scale
        gassuian['rotation'] = rotation
        gassuian['max_radii2D'] = max_radii2D

        return gassuian

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, num_gauss=-1):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if num_gauss == -1:
            num_gauss = xyz.shape[0]
        self._xyz = nn.Parameter(torch.tensor(
            xyz[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(
            features_dc[:num_gauss], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(
            features_extra[:num_gauss], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(
            opacities[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(
            scales[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(
            rots[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))

        R = torch.eye(3, device="cuda")
        T = torch.zeros(1, 3, device="cuda")
        self.R = nn.Parameter(R.requires_grad_(True))
        self.T = nn.Parameter(T.requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(
                    group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # if group["name"] not in ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]:
            #     continue
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # if group["name"] not in tensors_dict:
            #     continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # if "xyz" in optimizable_tensors:
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        # if "rotation" in optimizable_tensors:
        self._rotation = optimizable_tensors["rotation"]
        # self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(
                prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(
                prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class GS_Render:
    def __init__(self, sh_degree=3, white_background=False,
                 radius=1, rot_type='6d', view_dependent=False):

        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.rot_type = rot_type
        self.view_dependent = view_dependent

        self.gaussians = GaussianModel(sh_degree,
                                       rot_type=self.rot_type,
                                       view_dependent=self.view_dependent)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

    def init_model(self, input=None, num_pts=10000, radius=1.0):

        if input is None:
            # init from random points
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
            self.radius = radius.max()

        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            radius = np.linalg.norm(input.points, axis=1).max()
            self.gaussians.create_from_pcd(input, 1)
            self.radius = radius
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def reset_model(self):
        self.gaussians = GaussianModel(
            self.sh_degree, self.rot_type, self.view_dependent)

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        """
        Render the scene. 

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if not invert_bg_color else 1 - self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                if self.view_dependent:
                    shs_view = self.gaussians.get_features.transpose(1, 2).view(
                        -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                    )
                    # camera_center = viewpoint_camera.camera_center.repeat(
                    #     self.gaussians.get_features.shape[0], 1
                    # )
                    fidx = viewpoint_camera.uid
                    camera_center = self.gaussians.get_RT(fidx).inverse()[
                        :3, 3].detach()
                    camera_center = camera_center[None].repeat(
                        self.gaussians.get_features.shape[0], 1)
                    dir_pp = self.gaussians.get_xyz - camera_center
                    dir_pp_normalized = dir_pp / \
                        dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(
                        self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                    )
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    colors_precomp = self.gaussians.get_features_noview
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if len(out) == 4:
            rendered_image, radii, rendered_depth, rendered_alpha = out
            rendered_image = rendered_image.clamp(0, 1)

            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            return {
                "image": rendered_image,
                "depth": rendered_depth,
                "alpha": rendered_alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
            }
        elif len(out) == 3:
            rendered_image, radii, rendered_depth = out

            rendered_image = rendered_image.clamp(0, 1)

            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            return {
                "image": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
            }
