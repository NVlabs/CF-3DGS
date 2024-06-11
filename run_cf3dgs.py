# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
from argparse import ArgumentParser, Namespace

from trainer.cf3dgs_trainer import CFGaussianTrainer
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch
import pdb
from datetime import datetime


def contruct_pose(poses):
    n_trgt = poses.shape[0]
    # for i in range(n_trgt-1):
    #     poses[i+1] = poses[i+1] @ torch.inverse(poses[i])
    for i in range(n_trgt-1, 0, -1):
        poses = torch.cat(
            (poses[:i], poses[[i-1]]@poses[i:]), 0)
    return poses


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    model_cfg = lp.extract(args)
    pipe_cfg = pp.extract(args)
    optim_cfg = op.extract(args)
    # hydrant/615_99120_197713
    # hydrant/106_12648_23157
    # teddybear/34_1403_4393
    data_path = model_cfg.source_path
    trainer = CFGaussianTrainer(data_path, model_cfg, pipe_cfg, optim_cfg)
    start_time = datetime.now()
    if model_cfg.mode == "train":
        trainer.train_from_progressive()
    elif model_cfg.mode == "render":
        trainer.render_nvs(traj_opt=model_cfg.traj_opt)
    elif model_cfg.mode == "eval_nvs":
        trainer.eval_nvs()
    elif model_cfg.mode == "eval_pose":
        trainer.eval_pose()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
