import numpy as np
import torch
import tqdm

from diffusion_policy.policy.base_policy import BaseLowdimPolicy
from diffusion_policy.env_runner.base_runner import BaseLowdimRunner

import zarr, time

from legged_gym.envs import *
from legged_gym.utils import task_registry



class LeggedRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            task="g1_amp",
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            device=None,
        ):
        super().__init__(output_dir)

        self.task = task
        
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.task)
        # override some parameters for testing
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

        train_cfg.runner.amp_num_preload_transitions = 1

        # prepare environment        
        env_cfg.env.num_envs = 100
        
        # breakpoint()
        env, _ = task_registry.make_env(name=self.task, args=None, env_cfg=env_cfg)
        
        self.env = env
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
    
    def run(self, policy: BaseLowdimPolicy, online=False, generate_data=False):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        env.max_episode_length = int(env.max_episode_length)

        # plan for rollout
        obs, _ = env.reset()
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval IsaacGym", 
            leave=False, mininterval=self.tqdm_interval_sec)
        
        history = self.n_obs_steps
        
        state_history = torch.zeros((env.num_envs, history+1, 98), dtype=torch.float32, device=device)
        action_history = torch.zeros((env.num_envs, history, env.num_actions), dtype=torch.float32, device=device)
        
        # state_history[:,:,:] = obs[:,None,:]
        state_history[:,:,:] = env.get_diffusion_observation().to(device)[:, None, :] # (env.num_envs, 1, env.num_observations)
        
        obs_dict = {"obs": state_history[:, :]}
        single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")}
            
            
        episode_ends = []
        action_error = []
        idx = 0    
        saved_idx = 0 
        evaluate = False

        if evaluate:
            record_done = torch.zeros(100)
            record_episode_length = torch.zeros(100)

        action = torch.zeros((env.num_envs, 1, env.num_actions), dtype=torch.float32, device=device)
        while(True):
            # run policy
            with torch.no_grad():

                if "hop" in self.task:
                    state_history[:, -policy.n_obs_steps-1:-1, 6:9] = torch.tensor([.7, 0., 0.])
                if "bounce" in self.task:
                    state_history[:, -policy.n_obs_steps-1:-1, 6:9] = torch.tensor([1.5, 0., 0.])
                elif "walk" in self.task:
                    state_history[:, -policy.n_obs_steps-1:-1, 6:9] = torch.tensor([.7, 1., 0.])

                # USE DELAYED INPUTS s_t-h-1:s_t-1
                obs_dict = {"obs": state_history[:, -policy.n_obs_steps-1:-1, :]}
                t1 = time.perf_counter()
                action_dict = policy.predict_action(obs_dict)
                t2 = time.perf_counter()
                print("time spent diffusion step: ", t2-t1)
                
                pred_action = action_dict["action_pred"]

                # USE THE NEXT PREDICTED ACTION
                # RHC Framework -- only use the first action
                action = pred_action[:,history:history+1,:]

            # step env
            self.n_action_steps = action.shape[1]
            temp_length_buf = env.episode_length_buf.clone()
            for i in range(self.n_action_steps):
                action_step = action[:, i, :]
                temp_length_buf[:] = env.episode_length_buf.clone()
                _, _, _, done, self.extras, reset_env_ids, terminal_amp_states = env.step(action_step)
            
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                action_history = torch.roll(action_history, shifts=-1, dims=1)
                
                state_history[:, -1, :] = env.get_diffusion_observation().to(device)
                # action_history[:, -1, :] = env.get_diffusion_action().to(device)
                single_obs_dict = {"obs": state_history[:, -1, :].to("cuda:0")}
            
                idx += 1
            # reset env
            env_ids = torch.nonzero(done, as_tuple=False).squeeze(1).int()
            if len(env_ids) > 0:
                state_history[env_ids,:,:] = single_obs_dict["obs"][env_ids].to(state_history.device)[:,None,:]
                action_history[env_ids,:,:] = 0.0

                idx = 0                        

            if len(env_ids) > 0 and evaluate:
                mask = (record_done < 1) & done.cpu()
                # print(env_ids)
                record_episode_length[mask] += temp_length_buf[mask].cpu().float()
                record_done += mask
                if (record_done == 1).all():
                    print(torch.mean(record_episode_length))
                    print(torch.sum(record_episode_length >= 1000), torch.sum(mask) / 100)
                    break
                    
            done = done.cpu().numpy()
            done = np.all(done)

            # update pbar
            if online:
                pbar.update(action.shape[1])
            else:
                pbar.update(env.num_envs)

        # clear out video buffer
        _ = env.reset()
        