"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

try:
    from isaacgym.torch_utils import *
except:
    print("Isaac Gym Not Installed")

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--headless", default=False)
@click.option("--online", default=True)
@click.option("--generate_data", default=False)
def main(checkpoint, output_dir, device, online, generate_data, **kwargs):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]


    OmegaConf.set_struct(cfg, False)

    cfg.task.env_runner["device"] = device
    # cfg["task"]["env_runner"]["_target_"] = "diffusion_policy.env_runner.cyber_runner.LeggedRunner"
    print("Using {0} number of observation steps.".format(cfg["task"]["env_runner"]["n_obs_steps"]))
    
    cls = hydra.utils.get_class(cfg._target_)
    print(f"Loading workspace {cls.__name__}")
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    task="cyber2_stand_dance_aug"
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        task=task)
    env_runner.run(policy, online=online, generate_data=generate_data)
    

if __name__ == "__main__":
    main()
