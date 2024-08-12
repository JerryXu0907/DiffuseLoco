# DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets


## Codebase Structure

1. Model Defination:
 ```diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py```

2. Evaluation Script:
```scripts/eval.py```

3. Config File:
```diffusion_policy/config_files/cyber_diffusion_policy_n=8.yaml```

4. Environment for evaluation and source policy training:
```legged_gym/envs/cyberdog2```

5. Environment Wrapper (RHC, Delayed Inputs, Uniform Obs Space):
 ```diffusion_policy/diffusion_policy/env_runner/cyber_runner.py```

6. Deploy on real robots (This section is not completed yet) :
```legged_gym/legged_gym/scripts``` and
```csrc``` and ```scripts/pytorch_save.py```


## Getting Started

First, create the conda environment:

```bash
conda create -n diffuseloco python=3.8
```

followed by 

```bash
conda activate diffuseloco
```

Install necessary system packages:

```bash
sudo apt install cmake
```

Then, install the python dependencies:

```bash
cd DiffuseLoco

pip install -r requirements.txt
```

Then, install IsaacGym for simulation environment:

> Note: in the public repo, this should come from NVIDIA's official source. We provide a zip file for easier review purpose only. 

```bash
unzip isaacgym.zip

cd isaacgym/python

pip install -e .
```

Finally, install the package

```bash
cd ../..

bash ./install.sh
```

## Evaluate Pre-trained Checkpoints

Bipedal Walking Task

```bash
source env.sh

python ./scripts/eval.py --checkpoint=./checkpoints/cyberdog_final.ckpt --task=cyber2_stand
```

Hop Task

```bash
source env.sh

python ./scripts/eval.py --checkpoint=./checkpoints/cyberdog_final.ckpt --task=cyber2_hop
```

Bounce Task

```bash
source env.sh

python ./scripts/eval.py --checkpoint=./checkpoints/cyberdog_final.ckpt --task=cyber2_bounce
```

Walk Task

```bash
source env.sh

python ./scripts/eval.py --checkpoint=./checkpoints/cyberdog_final.ckpt --task=cyber2_walk
```

## Training

```bash
source env.sh

python scripts/train.py
```


## Compatibility

The codebase is tested on the following systems:

### System 1

- NVIDIA RTX 4060M
- Ubuntu 20.04
- NVIDIA driver version: 535 (535.129.03)
- CUDA version: 12.1.1
- cuDNN version: 8.9.7 for CUDA 12.X
- TensorRT version: 8.6 GA

### System 2

- NVIDIA RTX 4070
- Ubuntu 22.04
- NVIDIA driver version: 550 (550.90.07)
- CUDA version: 12.4
- cuDNN version: 8.9.7 for CUDA 12.4
- TensorRT version: 10.3.0.26 GA


## Accelerating for Real-Time Deployment

We use [TensorRT](https://developer.nvidia.com/tensorrt) to accelerate the policy inference and meet the real-time requirement.

Before installation, verify that the latest CUDA and cuDNN are installed on the system.

Download the "TensorRT 10.3 GA for Linux x86_64 and CUDA 12.0 to 12.5 TAR Package" and the "TensorRT 10.3 GA for Ubuntu 22.04 and CUDA 12.0 to 12.5 DEB local repo Package" installation package.

Install with the following commands:

```bash
cd ~/Downloads/
sudo dpkg -i ./nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5/nv-tensorrt-local-620E7D29-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5
```

We also need to link the libraries. Unpack the tar package:

```bash
cd ~/Downloads/
tar xzvf ./TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
```

Then. move the unpacked directory to the installation path (here, we will use `$TRT_INSTALL_PATH`), and add the following lines to bashrc

```bash
# TensorRT
export TRT_LIBPATH=$TRT_INSTALL_PATH/targets/x86_64-linux-gnu/lib/
export LD_LIBRARY_PATH=$TRT_INSTALL_PATH/lib/:$TRT_LIBPATH:$LD_LIBRARY_PATH
```

Finally, install the Python binding using the following command

```bash
cd $TRT_INSTALL_PATH/python/
pip install ./tensorrt-10.3.0-cp38-none-linux_x86_64.whl
```
