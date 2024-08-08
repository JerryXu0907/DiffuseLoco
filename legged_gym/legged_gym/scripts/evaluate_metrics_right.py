# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, export_policy_as_onnx, task_registry, Logger

import numpy as np
import torch
from tqdm import tqdm
import pickle

import escnn
from escnn.nn import FieldType
from hydra import compose, initialize
from morpho_symm.utils.robot_utils import load_symmetric_system
from morpho_symm.nn.test_EMLP import get_kinematic_three_rep_two, get_ground_reaction_forces_rep_two
from rsl_rl.modules import actor_critic_symmetric
from rsl_rl.algorithms import ppo_augment
def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.task == "go1_highlevel":
        low_env_cfg = env_cfg.low_env
    else:
        low_env_cfg = env_cfg
    low_env_cfg.env.num_envs = 2000
    low_env_cfg.record.record = RECORD_FRAMES
    low_env_cfg.record.folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', str(args.checkpoint))
    low_env_cfg.terrain.curriculum = False
    low_env_cfg.rewards.curriculum = False
    low_env_cfg.mode = "play"
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.com_displacement_range = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    if "stand_dance" in args.task:
        low_env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
        low_env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        low_env_cfg.commands.ranges.heading = [-0.5 * np.pi, -0.5 * np.pi]
    elif "push_door" in args.task:
        low_env_cfg.asset.left_or_right = 0 # 0: right, 1: left
    elif "walk_slope" in args.task:
        low_env_cfg.terrain.curriculum = True
        low_env_cfg.commands.ranges.lin_vel_x = [0.2, 0.2]
        low_env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
        low_env_cfg.commands.ranges.heading = [0., 0.]

    if os.path.exists(low_env_cfg.record.folder):
        shutil.rmtree(low_env_cfg.record.folder)
    os.makedirs(low_env_cfg.record.folder, exist_ok=True)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg, is_highlevel=(args.task == "go1_highlevel"))
    low_env = env.low_level_env if args.task == "go1_highlevel" else env
    # obs = env.get_observations()
    obs, *_ = env.reset()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # currently, both high and low level shares the same number of obs
        if hasattr(ppo_runner.alg.actor_critic, "adaptation_module"):
            input_dim = env.num_obs * env.num_history + env.num_obs * env.num_stacked_obs
        else:
            input_dim = env.num_obs
        export_policy_as_onnx(ppo_runner.alg.actor_critic, input_dim, path)
        print('Exported policy as jit script to: ', path)
    # path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    # torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), os.path.join(path, 'policy.pt'))
    # print('Exported policy to: ', os.path.join(path, 'policy.pt'))
    if args.play_mirror:
        if 'emlp' in args.task:
            G = actor_critic_symmetric.G
        elif 'aug' in args.task:
            G = ppo_augment.G
        else:
            initialize(config_path="../../../MorphoSymm/morpho_symm/cfg/robot", version_base='1.3')
            robot_name = 'a1'  # or any of the robots in the library (see `/morpho_symm/cfg/robot`)
            robot_cfg = compose(config_name=f"{robot_name}.yaml")
            robot, G = load_symmetric_system(robot_cfg=robot_cfg)
        rep_QJ = G.representations["Q_js"]  # Used to transform joint-space position coordinates q_js ∈ Q_js
        rep_TqQJ = G.representations["TqQ_js"]  # Used to transform joint-space velocity coordinates v_js ∈ TqQ_js
        rep_O3 = G.representations["Rd"]  # Used to transform the linear momentum l ∈ R3
        rep_O3_pseudo = G.representations["Rd_pseudo"]  # Used to transform the angular momentum k ∈ R3
        trivial_rep = G.trivial_representation
        rep_kin_three = get_kinematic_three_rep_two(G)
        rep_hands_pos = get_ground_reaction_forces_rep_two(G, rep_kin_three) # Used to transform hands position
        gspace = escnn.gspaces.no_base_space(G)
        in_field_type =  FieldType(gspace, [rep_O3, rep_O3, rep_TqQJ, rep_TqQJ, rep_kin_three, rep_O3, rep_O3, rep_O3, rep_kin_three]*5)
        out_field_type = FieldType(gspace, [rep_TqQJ])

    total_steps = 1000000000000

    num_epi_record = torch.zeros(low_env_cfg.env.num_envs)
    init_dones = torch.zeros(low_env_cfg.env.num_envs, dtype=torch.bool)
    metrics = {}

    
    with tqdm(total=13) as pbar:

        for i in range(total_steps): 
            with torch.no_grad():
                if args.play_mirror:
                    obs = in_field_type.transform_fibers(obs, G.elements[1])
                actions = policy(obs)
            if args.play_mirror:
                actions = out_field_type.transform_fibers(actions, G.elements[1])
            obs, _, rews, dones, infos = env.step(actions.detach())

            init_dones = num_epi_record > 1
            num_epi_record += dones.cpu()
            mask = (num_epi_record < 12) & (init_dones)
            # gather metrics, extend each key by env.metrics but only for mask env. don't collect for not masked
            for key in env.metrics.keys():
                if key not in metrics:
                    metrics[key] = torch.zeros(low_env_cfg.env.num_envs)
                metrics[key] += env.metrics[key].cpu() * mask
            
            # print("num_epi_record", torch.mean(num_epi_record), torch.min(num_epi_record))
            if torch.all(num_epi_record >= 12):
                break
            pbar.update(torch.min(num_epi_record).item()- pbar.n)


    tracking_error = metrics['tracking_ang_vel'].cpu() / metrics['time'] * env.dt
    print('tracking_error:', torch.mean(tracking_error))
    CoT = metrics['energy'].cpu() / metrics['base_ang_vel'].cpu()
    print('CoT:', torch.mean(CoT))
    alive_time = metrics['time'] / (num_epi_record - 2)
    print('alive_time:', torch.mean(alive_time))

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
