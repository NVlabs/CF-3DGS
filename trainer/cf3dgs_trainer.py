# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from tqdm import tqdm
from random import randint
import math
import numpy as np
import random
from collections import defaultdict, OrderedDict
import json
import gzip
import torch
import torch.nn.functional as F
from torchvision import io
from PIL import Image
from einops import rearrange
import pickle
import scipy
import imageio
import glob
import cv2
import open3d as o3d

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model_cf import CF3DGS_Render as GS_Render

from utils.graphics_utils import BasicPointCloud, focal2fov, procrustes
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr, colorize
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE

from kornia.geometry.depth import depth_to_3d, depth_to_normals
from kornia.geometry.camera import project_points

import pdb

from .trainer import GaussianTrainer
from .losses import Loss, compute_scale_and_shift

from copy import copy
from utils.vis_utils import interp_poses_bspline, generate_spiral_nerf, plot_pose


def contruct_pose(poses):
    n_trgt = poses.shape[0]
    for i in range(n_trgt-1, 0, -1):
        poses = torch.cat(
            (poses[:i], poses[[i-1]]@poses[i:]), 0)
    return poses


class CFGaussianTrainer(GaussianTrainer):
    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg):
        super().__init__(data_root, model_cfg, pipe_cfg, optim_cfg)
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg

        self.gs_render = GS_Render(white_background=False,
                                   view_dependent=model_cfg.view_dependent,)
        self.gs_render_local = GS_Render(white_background=False,
                                         view_dependent=model_cfg.view_dependent,)
        self.use_mask = self.pipe_cfg.use_mask
        self.use_mono = self.pipe_cfg.use_mono
        self.near = 0.01
        self.setup_losses()

    def setup_losses(self):
        self.loss_func = Loss(self.optim_cfg)

    def train_step(self,
                   gs_render,
                   viewpoint_cam,
                   iteration,
                   pipe,
                   optim_opt,
                   colors_precomp=None,
                   update_gaussians=True,
                   update_cam=True,
                   update_distort=False,
                   densify=True,
                   prev_gaussians=None,
                   use_reproject=False,
                   use_matcher=False,
                   ref_fidx=None,
                   reset=True,
                   reproj_loss=None,
                   **kwargs,
                   ):
        # Render
        render_pkg = gs_render.render(
            viewpoint_cam,
            compute_cov3D_python=pipe.compute_cov3D_python,
            convert_SHs_python=pipe.convert_SHs_python,
            override_color=colors_precomp)

        if prev_gaussians is not None:
            with torch.no_grad():
                # Render
                render_pkg_prev = prev_gaussians.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python,
                    override_color=colors_precomp)
            mask = (render_pkg["alpha"] > 0.5).float()
            render_pkg["image"] = render_pkg["image"] * \
                mask + render_pkg_prev["image"] * (1 - mask)
            render_pkg["depth"] = render_pkg["depth"] * \
                mask + render_pkg_prev["depth"] * (1 - mask)

        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["image"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss_dict = self.compute_loss(render_pkg, viewpoint_cam,
                                      pipe, iteration,
                                      use_reproject, use_matcher,
                                      ref_fidx, **kwargs)

        loss = loss_dict['loss']
        loss.backward()

        with torch.no_grad():
            # Progress bar
            # try:
            #     self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            # except:
            #     pdb.set_trace()
            # mask = visibility_filter.reshape(gt_image.shape[1:])[None]
            psnr_train = psnr(image, gt_image).mean().double()
            self.just_reset = False
            if iteration < optim_opt.densify_until_iter and densify:
                # Keep track of max radii in image-space for pruning
                try:
                    gs_render.gaussians.max_radii2D[visibility_filter] = torch.max(gs_render.gaussians.max_radii2D[visibility_filter],
                                                                                   radii[visibility_filter])
                except:
                    pdb.set_trace()
                gs_render.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    self.gs_render.gaussians.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                               gs_render.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0 and reset and iteration < optim_opt.reset_until_iter:
                    gs_render.gaussians.reset_opacity()
                    self.just_reset = True

            if update_gaussians:
                gs_render.gaussians.optimizer.step()
                gs_render.gaussians.optimizer.zero_grad(set_to_none=True)
            if getattr(gs_render.gaussians, "camera_optimizer", None) is not None and update_cam:
                current_fidx = gs_render.gaussians.seq_idx
                gs_render.gaussians.camera_optimizer[current_fidx].step()
                gs_render.gaussians.camera_optimizer[current_fidx].zero_grad(
                    set_to_none=True)


        return loss_dict, render_pkg, psnr_train

    def init_two_view(self, view_idx_1, view_idx_2, pipe, optim_opt):
        # prepare data
        self.loss_func.depth_loss_type = "invariant"
        cam_info, pcd, viewpoint_cam = self.prepare_data(view_idx_1,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()

        # Initialize gaussians
        self.gs_render.reset_model()
        self.gs_render.init_model(pcd,)
        # self.gs_render.init_model(num_pts=300_000,)
        self.gs_render.gaussians.init_RT_seq(self.seq_len)
        self.gs_render.gaussians.set_seq_idx(view_idx_1)
        self.gs_render.gaussians.rotate_seq = False
        # Fit relative pose
        print(f"optimizing frame {view_idx_1:03d}")
        optim_opt.iterations = 1000
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress")
        self.gs_render.gaussians.training_setup(optim_opt, fix_pos=True,)
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          depth_gt=self.mono_depth[view_idx_1],
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        self.pcd_stack = []
        self.pcd_stack.append(self.gs_render.gaussians.get_xyz.detach())
        model_params = self.gs_render.gaussians.capture()
        return model_params

    def add_view_v2(self, view_idx, view_idx_prev, reverse=False):
        # Initialize gaussians
        self.loss_func.depth_loss_type = "invariant"
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        # prepare data
        cam_info, pcd, viewpoint_cam = self.prepare_data(view_idx_prev,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()
        self.gs_render_local.reset_model()
        self.gs_render_local.init_model(pcd)
        # Fit current gaussian
        optim_opt.iterations = 1000
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress")
        self.gs_render_local.gaussians.training_setup(
            optim_opt, fix_pos=True,)
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render_local.gaussians.update_learning_rate(iteration)
            loss, rend_dict, psnr_train = self.train_step(self.gs_render_local,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          #   depth_gt=self.mono_depth[view_idx_prev],
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          updata_distort=False,
                                                          densify=False,
                                                          )
            if psnr_train > 35 and iteration > 500:
                progress_bar.close()
                break

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        print(f"optimizing frame {view_idx:03d}")
        viewpoint_cam_ref = self.load_viewpoint_cam(view_idx,
                                                    load_depth=True)
        optim_opt.iterations = 300
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        self.gs_render_local.gaussians.init_RT(None)
        self.gs_render_local.gaussians.training_setup_fix_position(
            optim_opt, gaussian_rot=False)

        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress")
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render_local.gaussians.update_learning_rate(iteration)
            loss, rend_dict_ref, psnr_train = self.train_step(self.gs_render_local,
                                                              viewpoint_cam_ref, iteration,
                                                              pipe, optim_opt,
                                                              densify=False,
                                                              )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        # self.visualize(rend_dict_ref, "vis/render_optim.png",
        #                gt_image=viewpoint_cam_ref.original_image.cuda(),
        #                gt_depth=self.mono_depth[view_idx_prev])
        local_model_params = self.gs_render_local.gaussians.capture()

        # pcd under view_idx_prev frame
        pcd = self.gs_render_local.gaussians._xyz.detach()
        rel_pose = self.gs_render_local.gaussians.get_RT().detach()
        pose = rel_pose @ self.gs_render.gaussians.get_RT(
            view_idx_prev).detach()
        self.gs_render.gaussians.update_RT_seq(pose, view_idx)

        self.gs_render.gaussians.rotate_seq = False
        pipe.convert_SHs_python = self.gs_render.gaussians.rotate_seq

        if self.just_reset:
            num_iterations = 500
            self.just_reset = False
            for iteration in range(1, num_iterations):
                fidx = randint(0, view_idx_prev)
                self.global_iteration += 1
                self.gs_render.gaussians.update_learning_rate(
                    self.global_iteration)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True)
                loss, rend_dict_ref, psnr_train = self.train_step(self.gs_render,
                                                                  viewpoint_cam,
                                                                  self.global_iteration,
                                                                  pipe, self.optim_cfg,
                                                                  update_gaussians=True,
                                                                  update_cam=False,
                                                                  #   depth_gt=self.mono_depth[fidx],
                                                                  update_distort=False,
                                                                  )

        num_iterations = self.single_step
        if max(view_idx, view_idx_prev) > min(int(self.seq_len * 0.8), self.seq_len-5):
            num_iterations = 1000
        elif min(view_idx, view_idx_prev) < int(self.single_step // 100):
            num_iterations = 100

        progress_bar = tqdm(range(num_iterations), desc="Training progress")

        for iteration in range(1, num_iterations+1):

            last_frame = max(1, view_idx//2)
            if random.random() < 0.7:
                fidx = randint(last_frame, view_idx)
            else:
                fidx = randint(1, last_frame)

            self.global_iteration += 1
            if self.gs_render.gaussians.rotate_seq:
                self.gs_render.gaussians.set_seq_idx(fidx)
            viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                    pose=self.gs_render.gaussians.get_RT(
                                                        fidx).detach().cpu()
                                                    if not self.gs_render.gaussians.rotate_seq
                                                    else None,
                                                    load_depth=True)
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(
                self.global_iteration)

            loss, rend_dict_ref, psnr_train = self.train_step(self.gs_render,
                                                              viewpoint_cam,
                                                              self.global_iteration,
                                                              pipe, self.optim_cfg,
                                                              update_gaussians=True,
                                                              update_cam=False,
                                                              #   depth_gt=self.mono_depth[fidx],
                                                              update_distort=self.pipe_cfg.distortion,
                                                              )

            if self.global_iteration % 1000 == 0:
                self.gs_render.gaussians.oneupSHdegree()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)

            if iteration == num_iterations:
                progress_bar.close()

        return pcd, local_model_params

    def create_pcd_from_render(self, render_dict, viewpoint_cam):
        intrinsics = torch.from_numpy(viewpoint_cam.intrinsics).float().cuda()
        depth = render_dict["depth"].squeeze()
        image = render_dict["image"]
        pts = depth_to_3d(depth[None, None],
                          intrinsics[None],
                          normalize_points=False)
        points = pts.squeeze().permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        colors = image.permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data = pcd_data.farthest_point_down_sample(num_samples=30_000)
        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)
        return pcd

    def train_from_progressive(self, ):
        pipe = copy(self.pipe_cfg)
        self.single_step = 500 # 300 for faster training; 500 for better results

        num_iterations = self.single_step * (self.seq_len // 10) * 10
        self.optim_cfg.iterations = num_iterations
        self.optim_cfg.position_lr_max_steps = num_iterations
        self.optim_cfg.opacity_reset_interval = num_iterations // 10
        self.optim_cfg.densify_until_iter = num_iterations
        self.optim_cfg.reset_until_iter = int(num_iterations * 0.8)
        self.optim_cfg.densify_from_iter = 1000
        self.optim_cfg.densify_from_iter = self.single_step


        if pipe.expname == "":
            expname = "progressive"
        else:
            expname = pipe.expname
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        result_path = f"output/{expname}/{self.category}_{self.seq_name}"
        os.makedirs(result_path, exist_ok=True)

        pose_dict = dict()
        poses_gt = []
        for seq_data in self.data:
            if self.data_type == "co3d":
                R, t, _, _, _ = self.load_camera(seq_data)
            else:
                try:
                    R = seq_data.R.transpose()
                    t = seq_data.T
                except:
                    R = np.eye(3)
                    t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses_gt.append(torch.from_numpy(pose))
        pose_dict["poses_gt"] = torch.stack(poses_gt)
        max_frame = self.seq_len
        start_frame = 1
        end_frame = max_frame



        os.makedirs(f"{result_path}/pose", exist_ok=True)
        os.makedirs(f"{result_path}/mesh", exist_ok=True)

        num_eppch = 1
        reverse = False
        for epoch in range(num_eppch):
            gauss_params = self.init_two_view(
                0, end_frame, pipe, copy(self.optim_cfg))
            
            self.global_iteration = 0
            optim_opt = copy(self.optim_cfg)
            self.gs_render.gaussians.rotate_seq = True
            self.gs_render.gaussians.training_setup(self.optim_cfg,
                                                    fit_pose=True,)
            self.match_results = OrderedDict()
            for fidx in range(start_frame, end_frame):
                # pcd_new, local_gauss_params = self.add_view(
                #     None, fidx, fidx-1, pipe, optim_opt, reverse=reverse)
                pcd_new, local_gauss_params = self.add_view_v2(
                    fidx, fidx-1)
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        )
                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=pipe.compute_cov3D_python,
                                                    convert_SHs_python=pipe.convert_SHs_python)
                gt_image = viewpoint_cam.original_image.cuda()
                psnr_train = psnr(render_dict["image"],
                                    gt_image).mean().double()
                print(
                    'Frames {:03d}/{:03d}, PSNR : {:.03f}'.format(fidx, self.seq_len-1, psnr_train))
                self.visualize(render_dict,
                                f"{result_path}/train/{self.global_iteration:06d}_{fidx:03d}.png",
                                gt_image=gt_image, save_ply=False)

            with torch.no_grad():
                psnr_test = 0.0
                pose_dict["poses_pred"] = []
                self.render_depth = OrderedDict()
                self.gs_render.gaussians.rotate_seq = False
                self.gs_render.gaussians.rotate_xyz = False

                for val_idx in range(end_frame):
                    viewpoint_cam = self.load_viewpoint_cam(val_idx,
                                                            pose=self.gs_render.gaussians.get_RT(
                                                                val_idx).detach().cpu(),
                                                            )
                    render_dict = self.gs_render.render(viewpoint_cam,
                                                        compute_cov3D_python=pipe.compute_cov3D_python,
                                                        convert_SHs_python=pipe.convert_SHs_python)
                    self.render_depth[val_idx] = render_dict["depth"]
                    gt_image = viewpoint_cam.original_image.cuda()
                    psnr_test += psnr(render_dict["image"],
                                        gt_image).mean().double()
                    self.visualize(render_dict,
                                    f"{result_path}/eval/ep{epoch:02d}_{self.global_iteration:06d}_{val_idx:03d}.png",
                                    gt_image=gt_image, save_ply=False)
                print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}'.format(
                    start_frame,
                    end_frame,
                    psnr_test / (end_frame)))

                for idx in range(self.seq_len):
                    pose = self.gs_render.gaussians.get_RT(idx)
                    pose_dict["poses_pred"].append(pose.detach().cpu())

            pose_dict["poses_pred"] = torch.stack(pose_dict["poses_pred"])
            pose_dict["poses_gt"] = torch.stack(poses_gt)
            pose_dict["match_results"] = self.match_results
            torch.save(
                pose_dict, f"{result_path}/pose/ep{epoch:02d}_init.pth")
            os.makedirs(f"{result_path}/chkpnt", exist_ok=True)
            torch.save(self.gs_render.gaussians.capture(),
                        f"{result_path}/chkpnt/ep{epoch:02d}_init.pth")



    def eval_nvs(self, ):
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        num_epochs = 200
        num_iterations = num_epochs * self.seq_len
        optim_opt.iterations = num_iterations
        optim_opt.position_lr_max_steps = num_iterations
        optim_opt.densify_until_iter = num_iterations // 2
        optim_opt.reset_until_iter = num_iterations // 2
        optim_opt.opacity_reset_interval = num_iterations // 10
        optim_opt.densification_interval = 100
        optim_opt.densify_from_iter = 500
        # self.optim_cfg.densification_interval = 100

        if pipe.expname == "":
            expname = "progressive"
        else:
            expname = pipe.expname
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        # result_path = f"vis/{expname}/{self.category}_{self.seq_name}"
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'test')
        os.makedirs(result_path, exist_ok=True)

        pose_dict = dict()
        pose_dict["poses_gt"] = []
        for seq_data in self.data:
            if self.data_type == "co3d":
                R, t, _, _, _ = self.load_camera(seq_data)
            else:
                try:
                    R = seq_data.R.transpose()
                    t = seq_data.T
                except:
                    R = np.eye(3)
                    t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            pose_dict["poses_gt"].append(torch.from_numpy(pose))

        max_frame = self.seq_len
        start_frame = 0
        end_frame = max_frame
        if self.model_cfg.model_path != "":
            self.gs_render.gaussians.restore(
                torch.load(self.model_cfg.model_path), self.optim_cfg)
            pose_dict_train = torch.load(
                self.model_cfg.model_path.replace('chkpnt', 'pose'))
            self.gs_render.gaussians.rotate_seq = True

        sample_rate = 2 if "Family" in result_path else 8
        pose_test_init = pose_dict_train['poses_pred'][int(
            sample_rate/2)::sample_rate-1][:max_frame]
        self.gs_render.gaussians.init_RT_seq(
            self.seq_len, pose_test_init.float())
        self.gs_render.gaussians.rotate_seq = True
        self.gs_render.gaussians.training_setup(optim_opt,
                                                fix_pos=True,
                                                fix_feat=True,
                                                fit_pose=True,)
        progress_bar = tqdm(range(num_iterations),
                            desc="Training progress")

        iteration = 0
        for epoch in range(num_epochs):
            for fidx in range(self.seq_len):
                iteration += 1
                self.gs_render.gaussians.rotate_seq = True
                self.gs_render.gaussians.set_seq_idx(fidx)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=None,
                                                        load_depth=True,
                                                        )
                # self.gs_render.gaussians.update_learning_rate_camera(
                #     fidx, iteration)
                loss_dict, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                                   viewpoint_cam,
                                                                   iteration, pipe, optim_opt,
                                                                   densify=False,
                                                                   depth_gt=None,
                                                                   update_cam=True,
                                                                   update_gaussians=False,
                                                                   reset=False,
                                                                   )
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}"})
                    progress_bar.update(10)
                if iteration == optim_opt.iterations:
                    progress_bar.close()

        psnr_test = 0
        ssim_test = 0
        lpips_test = 0
        with torch.no_grad():
            for fidx in range(self.seq_len):
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True,
                                                        )

                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=False,
                                                    convert_SHs_python=False)
                gt_image = viewpoint_cam.original_image.cuda()
                psnr_test += psnr(render_dict["image"],
                                  gt_image).mean().double()
                ssim_test += ssim(render_dict["image"],
                                  gt_image).mean().double()
                lpips_test += lpips(render_dict["image"],
                                    gt_image, net_type="vgg").mean().double()
                self.visualize(render_dict,
                               f"{result_path}/test/{fidx:04d}.png",
                               gt_image=gt_image, save_ply=False)
        with open(f"{result_path}/test.txt", 'w') as f:
            f.write('PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
                    psnr_test / end_frame,
                    ssim_test / end_frame,
                    lpips_test / end_frame))
            f.close()

        print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
            start_frame,
            end_frame,
            psnr_test / end_frame,
            ssim_test / end_frame,
            lpips_test / end_frame))

    def eval_pose(self, ):
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'pose')
        os.makedirs(result_path, exist_ok=True)
        pose_path = os.path.join(result_path, 'ep00_init.pth')
        poses = torch.load(pose_path)
        poses_pred = poses['poses_pred'].inverse().cpu()
        poses_gt_c2w = poses['poses_gt'].inverse().cpu()
        poses_gt = poses_gt_c2w[:len(poses_pred)].clone()
        # align scale first (we do this because scale differennt a lot)
        trans_gt_align, trans_est_align, _ = self.align_pose(poses_gt[:, :3, -1].numpy(),
                                                             poses_pred[:, :3, -1].numpy())
        poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
        poses_pred[:, :3, -1] = torch.from_numpy(trans_est_align)

        c2ws_est_aligned = align_ate_c2b_use_a2b(poses_pred, poses_gt)
        ate = compute_ATE(poses_gt.cpu().numpy(),
                          c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(
            poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        print("{0:.3f}".format(rpe_trans*100),
              '&' "{0:.3f}".format(rpe_rot * 180 / np.pi),
              '&', "{0:.3f}".format(ate))
        plot_pose(poses_gt, c2ws_est_aligned, pose_path)
        pdb.set_trace()
        with open(f"{result_path}/pose_eval.txt", 'w') as f:
            f.write("RPE_trans: {:.03f}, RPE_rot: {:.03f}, ATE: {:.03f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))
            f.close()

    def align_pose(self, pose1, pose2):
        mtx1 = np.array(pose1, dtype=np.double, copy=True)
        mtx2 = np.array(pose2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R

    def render_nvs(self, traj_opt='bspline', N_novel_imgs=120, degree=100):
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'nvs')
        os.makedirs(result_path, exist_ok=True)
        self.gs_render.gaussians.restore(
            torch.load(self.model_cfg.model_path), self.optim_cfg)
        pose_dict_train = torch.load(
            self.model_cfg.model_path.replace('chkpnt', 'pose'))
        poses_pred_w2c_train = pose_dict_train['poses_pred'].cpu()
        if traj_opt == 'bspline':
            i_train = self.i_train
            if "co3d" in self.model_cfg.source_path:
                poses_pred_w2c_train = poses_pred_w2c_train[:100]
                i_train = self.i_train[:100]
            c2ws = interp_poses_bspline(poses_pred_w2c_train.inverse(), N_novel_imgs,
                                        i_train, degree)
            w2cs = c2ws.inverse()

        self.gs_render.gaussians.rotate_seq = False
        render_dir = f"{result_path}/{traj_opt}"
        os.makedirs(render_dir, exist_ok=True)
        for fidx, pose in enumerate(w2cs):
            viewpoint_cam = self.load_viewpoint_cam(10,
                                                    pose=pose,
                                                    )
            render_dict = self.gs_render.render(viewpoint_cam,
                                                compute_cov3D_python=False,
                                                convert_SHs_python=False)
            self.visualize(render_dict,
                           f"{render_dir}/img_out/{fidx:04d}.png",
                           save_ply=False)

        imgs = []
        for img in sorted(glob.glob(os.path.join(render_dir, "img_out", "*.png"))):
            if "depth" in img:
                continue
            rgb = cv2.imread(img)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(img.replace(".png", "_depth.png"))
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            rgb = np.hstack([rgb, depth])
            imgs.append(rgb)

        imgs = np.stack(imgs, axis=0)

        video_out_dir = os.path.join(render_dir, 'video_out')
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        imageio.mimwrite(os.path.join(
            video_out_dir, f'{self.category}_{self.seq_name}_ours.mp4'), imgs, fps=30, quality=9)

    def save_model(self, epoch):
        pass

    def compute_loss(self,
                     render_dict,
                     viewpoint_cam,
                     pipe_opt,
                     iteration,
                     use_reproject=False,
                     use_matcher=False,
                     ref_fidx=None,
                     **kwargs):
        loss = 0.0
        if "image" in render_dict:
            image = render_dict["image"]
            gt_image = viewpoint_cam.original_image.cuda()
        if "depth" in render_dict:
            depth = render_dict["depth"]
            depth[depth < self.near] = self.near
            fidx = viewpoint_cam.uid
            kwargs['depth_pred'] = depth

        loss_dict = self.loss_func(image, gt_image, **kwargs)
        return loss_dict

    def visualize(self, render_pkg, filename, gt_image=None, gt_depth=None, save_ply=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if "depth" in render_pkg:
            rend_depth = Image.fromarray(
                colorize(render_pkg["depth"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            if gt_depth is not None:
                gt_depth = Image.fromarray(
                    colorize(gt_depth.detach().cpu().numpy(),
                             cmap='magma_r')).convert("RGB")
                rend_depth = Image.fromarray(np.hstack([np.asarray(gt_depth),
                                                        np.asarray(rend_depth)]))
            rend_depth.save(filename.replace(".png", "_depth.png"))
        if "acc" in render_pkg:
            rend_acc = Image.fromarray(
                colorize(render_pkg["acc"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_acc.save(filename.replace(".png", "_acc.png"))

        rend_img = Image.fromarray(
            np.asarray(render_pkg["image"].detach().cpu().permute(1, 2, 0).numpy()
                       * 255.0, dtype=np.uint8)).convert("RGB")
        if gt_image is not None:
            gt_image = Image.fromarray(
                np.asarray(
                    gt_image.permute(1, 2, 0).cpu().numpy() * 255.0,
                    dtype=np.uint8)).convert("RGB")
            rend_img = Image.fromarray(np.hstack([np.asarray(gt_image),
                                                  np.asarray(rend_img)]))
        rend_img.save(filename)

        if save_ply:
            points = self.gs_render.gaussians._xyz.detach().cpu().numpy()
            pcd_data = o3d.geometry.PointCloud()
            pcd_data.points = o3d.utility.Vector3dVector(points)
            pcd_data.colors = o3d.utility.Vector3dVector(np.ones_like(points))
            o3d.io.write_point_cloud(
                filename.replace('.png', '.ply'), pcd_data)

    
    def construct_point(self, gs_model, poses, iteration, result_path, stop_frame=-1):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=3 * 0.01,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        pipe = self.pipe_cfg
        optim_opt = self.optim_cfg
        pipe.convert_SHs_python = True
        if poses is None:
            poses = torch.stack(
                [gs_model.gaussians.get_RT(idx).detach().cpu()
                 for idx in range(self.seq_len)])

        self.gs_render.gaussians.rotate_seq = False
        stop_frame = len(poses) if stop_frame == -1 else stop_frame
        with torch.no_grad():
            progress_bar = tqdm(range(self.seq_len),
                                desc="Reconstructing point cloud")
            for idx in range(len(poses)):
                if idx > stop_frame:
                    break

                viewpoint_cam = self.load_viewpoint_cam(
                    idx, pose=poses[idx], load_depth=True)
                # if idx not in self.render_depth:
                render_dict = gs_model.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python)
                render_depth = render_dict['depth'].detach().squeeze()

                rgb = viewpoint_cam.original_image.cuda().permute(1, 2, 0).detach().cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)

                depth = render_depth.detach().cpu().numpy()


                H, W = depth.shape
                intrinsic = viewpoint_cam.intrinsics
                fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

                rgb = o3d.geometry.Image(rgb)
                depth = o3d.geometry.Image(depth)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
                # pose = self.gs_render.gaussians.get_RT(idx).detach().cpu().numpy()
                volume.integrate(rgbd, intrinsic, poses[idx])
                progress_bar.update(1)
        progress_bar.close()

        self.gs_render.gaussians.rotate_seq = True
        mesh = volume.extract_triangle_mesh()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            f"{result_path}/{self.gs_render.rot_type}_{iteration:06d}.ply", mesh)

        points = np.asarray(mesh.vertices, dtype=np.float32)
        colors = np.asarray(mesh.vertex_colors)
        normals = np.asarray(mesh.vertex_normals)
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data.normals = o3d.utility.Vector3dVector(normals)

        pcd_data = pcd_data.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(
            f"{result_path}/{self.gs_render.rot_type}_{iteration:06d}.ply", pcd_data)
        points = np.asarray(pcd_data.points)
        colors = np.asarray(pcd_data.colors)
        normals = np.asarray(pcd_data.normals)
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        return pcd