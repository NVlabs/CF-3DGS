# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_inria.md file.
#
# For inquiries contact  george.drettakis@inria.fr


import torch
import math
import numpy as np
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

    def update(self, points):
        self.points = points


class SegmentedPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array
    labels: np.array


class SimilarityTransform(NamedTuple):
    R: torch.Tensor
    T: torch.Tensor
    s: torch.Tensor


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getWorld2View3(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def procrustes(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # print('X1', X1.shape)

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1 ** 2)

    # print('var', var1.shape)

    # 3. The outer product of X1 and X2.
    K = X1.mm(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)
    # V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[0], device=S1.device)
    Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # Construct R.
    R = V.mm(Z.mm(U.T))

    # print('R', X1.shape)

    # 5. Recover scale.
    scale = torch.trace(R.mm(K)) / var1
    # print(R.shape, mu1.shape)
    # 6. Recover translation.
    t = mu2 - scale * (R.mm(mu1))
    # t = mu2 - (R.mm(mu1))
    # print(t.shape)

    # 7. Error:
    S1_hat = scale * R.mm(S1) + t
    # S1_hat = R.mm(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    R_ = torch.eye(4).to(S1)
    R_[:3, :3] = R
    T_ = torch.eye(4).to(S1)
    T_[:3, -1] = t.squeeze(-1)
    S_ = torch.eye(4).to(S1)
    transf = T_@S_@R_

    return S1_hat, transf


# def procrustes(S1, S2,weights=None):
    # '''
    # Computes a similarity transform (sR, t) that takes
    # a set of 3D points S1 (BxNx3) closest to a set of 3D points, S2,
    # where R is an 3x3 rotation matrix, t 3x1 translation, s scale. / mod : assuming scale is 1
    # i.e. solves the orthogonal Procrutes problem.
    # '''
    # transposed = False
    # if S1.shape[0] != 3 and S1.shape[0] != 2:
    #     S1 = S1.T
    #     S2 = S2.T
    #     transposed = True
    # assert (S2.shape[1] == S1.shape[1])

    # if weights is None:
    #     weights = torch.ones_like(S1[:1,:])

    # # 1. Remove mean.
    # weights_norm = weights/(weights.sum(-1, keepdim=True) + 1e-6)
    # mu1 = (S1*weights_norm).sum(1, keepdim=True)
    # mu2 = (S2*weights_norm).sum(1, keepdim=True)

    # X1 = S1 - mu1
    # X2 = S2 - mu2

    # # diags = torch.stack([torch.diag(w.squeeze(0)) for w in weights]) # does batched version exist?
    # diags = torch.diag(weights.squeeze())

    # # 3. The outer product of X1 and X2.
    # K = (X1@diags).mm(X2.T)
    # # K = (X1@diags).bmm(X2.permute(0,2,1))

    # # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    # U, s, V = torch.svd(K)

    # # Construct Z that fixes the orientation of R to get det(R)=1.
    # Z = torch.eye(U.shape[0], device=S1.device)
    # Z[-1, -1] *= torch.sign(torch.det(U @ V.T))
    # # Construct R.
    # R = V.mm(Z.mm(U.T))

    # # 6. Recover translation.
    # t = mu2 - ((R.mm(mu1)))

    # # 7. Error:
    # S1_hat = R.mm(S1) + t
    # if transposed:
    #     S1_hat = S1_hat.T

    # # Combine recovered transformation as single matrix
    # R_=torch.eye(4).to(S1)
    # R_[:3, :3]=R
    # T_=torch.eye(4).to(S1)
    # T_[:3, -1]=t.squeeze(-1)
    # S_=torch.eye(4).to(S1)
    # transf = T_@S_@R_

    # return (S1_hat-S2).square().mean(),transf


def convert3x4_4x4(input):
    """
    Make into homogeneous cordinates by adding [0, 0, 0, 1] to the bottom.
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(
                input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor(
                [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def align_umeyama(model, data, known_scale=False):
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation
    of Transformation Parameters Between Two Point Patterns,
    IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Input:
    model -- first trajectory (nx3), numpy array type
    data -- second trajectory (nx3), numpy array type

    Output:
    s -- scale factor (scalar)
    R -- rotation matrix (3x3)
    t -- translation vector (3x1)
    t_error -- translational error per point (1xn)

    """

    # substract mean
    mu_M = model.mean(0)
    mu_D = data.mean(0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = np.shape(model)[0]

    # correlation
    C = 1.0/n*np.dot(model_zerocentered.transpose(), data_zerocentered)
    sigma2 = 1.0/n*np.multiply(data_zerocentered, data_zerocentered).sum()
    U_svd, D_svd, V_svd = np.linalg.linalg.svd(C)

    D_svd = np.diag(D_svd)
    V_svd = np.transpose(V_svd)

    S = np.eye(3)
    if (np.linalg.det(U_svd)*np.linalg.det(V_svd) < 0):
        S[2, 2] = -1

    R = np.dot(U_svd, np.dot(S, np.transpose(V_svd)))

    if known_scale:
        s = 1
    else:
        s = 1.0/sigma2*np.trace(np.dot(D_svd, S))

    t = mu_M-s*np.dot(R, mu_D)

    return s, R, t


def _getIndices(n_aligned, total_n):
    if n_aligned == -1:
        idxs = np.arange(0, total_n)
    else:
        assert n_aligned <= total_n and n_aligned >= 1
        idxs = np.arange(0, n_aligned)
    return idxs


# align by similarity transformation
def align_sim3(p_es, p_gt, n_aligned=-1):
    '''
    calculate s, R, t so that:
        gt = R * s * est + t
    '''
    idxs = _getIndices(n_aligned, p_es.shape[0])
    est_pos = p_es[idxs, 0:3]
    gt_pos = p_gt[idxs, 0:3]
    try:
        s, R, t = align_umeyama(gt_pos, est_pos)  # note the order
    except:
        print('[WARNING] align_poses.py: SVD did not converge!')
        s, R, t = 1.0, np.eye(3), np.zeros(3)
    return s, R, t


def align_ate_c2b_use_a2b(traj_a, traj_b, traj_c=None):
    """Align c to b using the sim3 from a to b.
    :param traj_a:  (N0, 3/4, 4) torch tensor
    :param traj_b:  (N0, 3/4, 4) torch tensor
    :param traj_c:  None or (N1, 3/4, 4) torch tensor
    :return:        (N1, 4,   4) torch tensor
    """
    device = traj_a.device
    if traj_c is None:
        traj_c = traj_a.clone()

    traj_a = traj_a.float().cpu().numpy()
    traj_b = traj_b.float().cpu().numpy()
    traj_c = traj_c.float().cpu().numpy()

    R_a = traj_a[:, :3, :3]  # (N0, 3, 3)
    t_a = traj_a[:, :3, 3]  # (N0, 3)

    R_b = traj_b[:, :3, :3]  # (N0, 3, 3)
    t_b = traj_b[:, :3, 3]  # (N0, 3)

    # This function works in quaternion.
    # scalar, (3, 3), (3, ) gt = R * s * est + t.
    s, R, t = align_sim3(t_a, t_b)

    # reshape tensors
    R = R[None, :, :].astype(np.float32)  # (1, 3, 3)
    t = t[None, :, None].astype(np.float32)  # (1, 3, 1)
    s = float(s)

    R_c = traj_c[:, :3, :3]  # (N1, 3, 3)
    t_c = traj_c[:, :3, 3:4]  # (N1, 3, 1)

    R_c_aligned = R @ R_c  # (N1, 3, 3)
    t_c_aligned = s * (R @ t_c) + t  # (N1, 3, 1)
    traj_c_aligned = np.concatenate(
        [R_c_aligned, t_c_aligned], axis=2)  # (N1, 3, 4)

    # append the last row
    traj_c_aligned = convert3x4_4x4(traj_c_aligned)  # (N1, 4, 4)

    traj_c_aligned = torch.from_numpy(traj_c_aligned).to(device)

    return traj_c_aligned  # (N1, 4, 4)
