# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from math import exp


import pdb


class Loss_Eval(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb_pred, rgb_gt):
        loss = F.mse_loss(rgb_pred, rgb_gt)
        return_dict = {
            'loss': loss
        }
        return return_dict


class Loss(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        self.depth_loss_type = cfg.depth_loss_type

        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')
        self.scale_inv_loss = ScaleAndShiftInvariantLoss(alpha=0.5,
                                                         scales=1)

        # ssim_loss = ssim
        self.ssim_loss = SSIM_V2()

        self.cfg = cfg

    def get_rgb_full_loss(self, rgb_values, rgb_gt, rgb_loss_type='l2'):
        num_pixels = rgb_values.shape[1] * rgb_values.shape[2]
        if rgb_loss_type == 'l1':
            rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(num_pixels)
        elif rgb_loss_type == 'l2':
            rgb_loss = self.l2_loss(rgb_values, rgb_gt) / float(num_pixels)
        return rgb_loss

    def depth_loss_dpt(self, pred_depth, gt_depth, weight=None):
        """
        :param pred_depth:  (H, W)
        :param gt_depth:    (H, W)
        :param weight:      (H, W)
        :return:            scalar
        """
        t_pred = torch.median(pred_depth)
        s_pred = torch.mean(torch.abs(pred_depth - t_pred))

        t_gt = torch.median(gt_depth)
        s_gt = torch.mean(torch.abs(gt_depth - t_gt))

        pred_depth_n = (pred_depth - t_pred) / s_pred
        gt_depth_n = (gt_depth - t_gt) / s_gt

        if weight is not None:
            loss = F.mse_loss(pred_depth_n, gt_depth_n, reduction='none')
            loss = loss * weight
            loss = loss.sum() / (weight.sum() + 1e-8)
        else:

            depth_error = (pred_depth_n - gt_depth_n) ** 2
            depth_error[depth_error > torch.quantile(depth_error, 0.8)] = 0
            loss = depth_error.mean()
            # loss = F.mse_loss(pred_depth_n, gt_depth_n)

        return loss

    def get_depth_loss(self, depth_pred, depth_gt):
        num_pixels = depth_pred.shape[0] * depth_pred.shape[1]
        if self.depth_loss_type == 'l1':
            loss = self.l1_loss(depth_pred, depth_gt) / float(num_pixels)
        elif self.depth_loss_type == 'invariant':
            # loss = self.depth_loss_dpt(1.0/depth_pred, 1.0/depth_gt)
            mask = (depth_gt > 0.02).float()
            loss = self.scale_inv_loss(
                depth_pred[None], depth_gt[None], mask[None])
        return loss


    def forward(self, rgb_pred, rgb_gt, depth_pred=None, depth_gt=None,
                rgb_loss_type='l1', **kwargs):

        rgb_gt = rgb_gt.cuda()

        lambda_dssim = self.cfg.lambda_dssim
        lambda_depth = self.cfg.lambda_depth

        # rgb_full_loss = self.get_rgb_full_loss(rgb_pred, rgb_gt, rgb_loss_type)
        rgb_full_loss = (1 - lambda_dssim) * l1_loss(rgb_pred, rgb_gt)
        if lambda_dssim != 0.0:
            # dssim_loss = compute_ssim_loss(rgb_pred, rgb_gt)
            # pdb.set_trace()
            # dssim_loss = 1 - ssim(rgb_pred, rgb_gt)
            dssim_loss = 1 - self.ssim_loss(rgb_pred, rgb_gt)

        if lambda_depth != 0.0 and depth_pred is not None and depth_gt is not None:
            depth_gt = depth_gt.cuda()
            depth_pred[depth_pred < 0.02] = 0.02
            depth_pred[depth_pred > 20.0] = 20.0
            depth_loss = self.get_depth_loss(
                depth_pred.squeeze(), depth_gt.squeeze())
        else:
            depth_loss = torch.tensor(0.0).cuda().float()

        loss = rgb_full_loss + lambda_dssim * dssim_loss +\
            lambda_depth * depth_loss

        if torch.isnan(loss):
            breakpoint()

        return_dict = {
            'loss': loss,
            'loss_rgb': rgb_full_loss,
            'loss_dssim': dssim_loss,
            'loss_depth': depth_loss,
        }

        return return_dict


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous())
    return window




def _ssim(
    img1, img2, window, window_size, channel, mask=None, size_average=True
):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if not (mask is None):
        b = mask.size(0)
        ssim_map = ssim_map.mean(dim=1, keepdim=True) * mask
        ssim_map = ssim_map.view(b, -1).sum(dim=1) / mask.view(b, -1).sum(
            dim=1
        ).clamp(min=1)
        return ssim_map

    # import pdb

    # pdb.set_trace

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM_V2(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_V2, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, mask=None):
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        (_, channel, _, _) = img1.size()

        if (
            channel == self.channel
            and self.window.data.type() == img1.data.type()
        ):
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(
            img1,
            img2,
            window,
            self.window_size,
            channel,
            mask,
            self.size_average,
        )






# copy from MiDaS and MonoSDF
def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] -
                  a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] +
                  a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(
            scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * \
            prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * \
                self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)
