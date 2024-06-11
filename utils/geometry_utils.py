# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np
import logging
from matplotlib import pyplot as plt
import os
import shutil
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                   device=torch.device("cpu")):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        device (torch.device): device to use
    '''
    h, w = resolution
    
    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, h, device=device), torch.arange(0, w, device=device))
    pixel_locations = torch.stack(
        [pixel_locations[1], pixel_locations[0]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0])/ 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc
    return pixel_locations, pixel_scaled

def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor


def get_mask(tensor):
    ''' Returns mask of non-illegal values for tensor.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
    '''
    tensor, is_numpy = to_pytorch(tensor, True)
    mask = ((abs(tensor) != np.inf) & (torch.isnan(tensor) == False))
    mask = mask.bool()
    if is_numpy:
        mask = mask.numpy()

    return mask




def transform_to_world(pixels, depth, camera_mat, world_mat=None, scale_mat=None,
                       invert=True, device=torch.device("cuda")):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)
    if world_mat is None:
        world_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]], dtype=torch.float32, device=device)
    if scale_mat is None:
        scale_mat = torch.tensor([[[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]], dtype=torch.float32, device=device)
    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    scale_mat = to_pytorch(scale_mat)
    
    
    # Invert camera matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)

    # Project pixels into camera space
    # pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    pixels_depth = pixels.clone()
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels_depth

    # Transform p_world back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)

    if is_numpy:
        p_world = p_world.numpy()
    return p_world


def transform_to_camera_space(p_world, camera_mat, world_mat, scale_mat):
    ''' Transforms world points to camera space.
        Args:
        p_world (tensor): world points tensor of size B x N x 3
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
    '''
    batch_size, n_p, _ = p_world.shape
    device = p_world.device

    # Transform world points to homogen coordinates
    p_world = torch.cat([p_world, torch.ones(
        batch_size, n_p, 1).to(device)], dim=-1).permute(0, 2, 1)

    # Apply matrices to transform p_world to camera space
    p_cam = camera_mat @ world_mat @ scale_mat @ p_world

    # Transform points back to 3D coordinates
    p_cam = p_cam[:, :3].permute(0, 2, 1)
    return p_cam


def origin_to_world(n_points, camera_mat, world_mat, scale_mat, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''
    batch_size = camera_mat.shape[0]
    device = camera_mat.device

    # Create origin in homogen coordinates
    p = torch.zeros(batch_size, 4, n_points, device=device)
    p[:, -1] = 1.

    # Invert matrices
    if invert:
        camera_mat = torch.inverse(camera_mat)
        world_mat = torch.inverse(world_mat)
        scale_mat = torch.inverse(scale_mat)

    # Apply transformation
    p_world = scale_mat @ world_mat @ camera_mat @ p

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world


def image_points_to_world(image_points, camera_mat, world_mat, scale_mat,
                          invert=True):
    ''' Transforms points on image plane to world coordinates.

    In contrast to transform_to_world, no depth value is needed as points on
    the image plane have a fixed depth of 1.

    Args:
        image_points (tensor): image points tensor of size B x N x 2
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    d_image = torch.ones(batch_size, n_pts, 1).to(device)
    return transform_to_world(image_points, d_image, camera_mat, world_mat,
                              scale_mat, invert=invert)


def check_weights(params):
    ''' Checks weights for illegal values.

    Args:
        params (tensor): parameter tensor
    '''
    for k, v in params.items():
        if torch.isnan(v).any():
            logger_py.warn('NaN Values detected in model weight %s.' % k)


def check_tensor(tensor, tensorname='', input_tensor=None):
    ''' Checks tensor for illegal values.

    Args:
        tensor (tensor): tensor
        tensorname (string): name of tensor
        input_tensor (tensor): previous input
    '''
    if torch.isnan(tensor).any():
        logger_py.warn('Tensor %s contains nan values.' % tensorname)
        if input_tensor is not None:
            logger_py.warn('Input was:', input_tensor)


def normalize_tensor(tensor, min_norm=1e-5, feat_dim=-1):
    ''' Normalizes the tensor.

    Args:
        tensor (tensor): tensor
        min_norm (float): minimum norm for numerical stability
        feat_dim (int): feature dimension in tensor (default: -1)
    '''
    norm_tensor = torch.clamp(torch.norm(tensor, dim=feat_dim, keepdim=True),
                              min=min_norm)
    normed_tensor = tensor / norm_tensor
    return normed_tensor
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

# https://github.dev/kwea123/ngp_pl
def create_spheric_poses(radius, mean_h, n_poses=120):
    
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
        mean_h: mean camera height
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,2*mean_h],
            [0,0,1,-t]
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0],
            [0,np.cos(phi),-np.sin(phi)],
            [0,np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th)],
            [0,1,0],
            [np.sin(th),0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0],[0,0,1],[0,1,0]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/12, radius)]
    return np.stack(spheric_poses, 0)

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # c = np.dot(c2w[:3,:4], np.array([0.7*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        # c = np.dot(c2w[:3,:4], np.array([0.3*np.cos(theta) , -0.3*np.sin(theta) , -np.sin(theta*zrate) *0.01, 1.]) * rads)
        c = np.dot(c2w[:3,:4], np.array([0.2*np.cos(theta) , -0.2*np.sin(theta) , -np.sin(theta*zrate) *0.1, 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def reprojection(pixels, depth, Rt_ref, world_mat, camera_mat):
   
    assert(pixels.shape[-1] == 2)

    # Convert to pytorch
    pixels, is_numpy = to_pytorch(pixels, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    Rt_ref = to_pytorch(Rt_ref)

    # Transform pixels to homogen coordinates
    pixels = pixels.permute(0, 2, 1)
    pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)
    # Project pixels into camera space
    # pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    pixels_depth = pixels.clone()
    depth = depth.view(1, -1, 1)
    pixels_depth[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)

    # Transform pixels to world space
    xy_ref = camera_mat @ Rt_ref @ torch.inverse(world_mat) @ torch.inverse(camera_mat) @ pixels_depth

    # Transform p_world back to 3D coordinates
    xy_ref = xy_ref[:, :3].permute(0, 2, 1)
    xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]
    
    valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(-1).float()
    if is_numpy:
        xy_ref = xy_ref.numpy()
    return xy_ref, valid_mask
def project_to_cam(points, camera_mat, device):
    '''
    points: (B, N, 3)
    camera_mat: (B, 4, 4)
    '''
    # breakpoint()
    B, N, D = points.size()
    points, is_numpy = to_pytorch(points, True)
    points = points.permute(0, 2, 1)
    # points = torch.cat([points, torch.ones(B, 1, N, device=device)], dim=1)

    xy_ref = camera_mat @ points

    xy_ref = xy_ref[:, :3].permute(0, 2, 1)
    xy_ref = xy_ref[..., :2] / xy_ref[..., 2:]
    
    
    valid_points = xy_ref.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(-1).bool()
    if is_numpy:
        xy_ref = xy_ref.numpy()
    return xy_ref, valid_mask

def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx

def taylor_A(x,nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i>0: denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans
def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans




def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scaled = pred * scale
    return pred_scaled, scale

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale

def align_scale_shift_numpy(pred: np.array, target: np.array):
    mask = target > 0
    target_mask = target[mask]
    pred_mask = pred[mask]
    if np.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = np.median(target[mask]) / (np.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale