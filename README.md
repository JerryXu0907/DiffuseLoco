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
5. Deploy on real robots (This section is not completed yet) :
```legged_gym/legged_gym/scripts``` and
```csrc``` and ```scripts/pytorch_save.py```







## Setup

Tested on:
- Ubuntu 22.04
- NVIDIA driver version: 535 (535.129.03)
- CUDA version: 12.1.1
- cuDNN version: 8.9.7 for CUDA 12.X
- TensorRT version: 8.6 GA

### Environment

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

note: in the public repo, this should come from nVidia's official source. We provide a zip file for easier review purpose only. 
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

Walk Task (Some bugs still exist when merging environments)
```bash
source env.sh

python ./scripts/eval.py --checkpoint=./checkpoints/cyberdog_final.ckpt --task=cyber2_walk
```

### TensorRT
Goto https://developer.nvidia.com/tensorrt

Download both the "TensorRT 10.3 GA for Linux x86_64 and CUDA 12.0 to 12.5 TAR Package" and the DEB package
Install the DEB package with Software Install.

Alternatively, do the following commands

```
sudo dpkg -i ./nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5/nv-tensorrt-local-620E7D29-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5
```
We also need to link the libraries. Unpack the tar package:

```
tar xzvf ./TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
```

Then. move the unpacked directory to the installation path (~/Documents/), and add to bashrc

```

# TensorRT
export TRT_LIBPATH="/home/tk/Documents/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/"
export LD_LIBRARY_PATH="/home/tk/Documents/TensorRT-10.3.0.26/lib/:$TRT_LIBPATH:$LD_LIBRARY_PATH"
```


Install to Python using the following command

```
cd ~/Documents/TensorRT-10.3.0.26/python/
pip install ./tensorrt-10.3.0-cp38-none-linux_x86_64.whl
```







