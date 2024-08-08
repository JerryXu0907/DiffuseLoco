# DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets



## Codebase Tutorial
This codebase is combined with two parts, diffusion_policy and AMP_for_hardware.
### Diffusion policy contains 
1. Diffusion model architecture.
2. Training and evaluation dataset and script.
3. Configs for different tasks.
4. Environment Runner.


### AMP_for_hardware
1. Gather source policy and training data.
2. Environment for legged gym for evaluation and training.
3. Deploy on real robots.(This section is not demonstrated in the codebase yet.)





## Setup

System requirements:
- Ubuntu 22.04
- NVIDIA driver version: 535 (535.129.03)
- CUDA version: 12.1.1
- cuDNN version: 8.9.7 for CUDA 12.X
- TensorRT version: 8.6 GA

### Environment

First, create the conda environment:

```bash
conda create -n diff python=3.8
```

Then, install the python dependencies:

```bash
pip install -r requirements.txt
```
### TensorRT

Goto https://developer.nvidia.com/tensorrt

Download both the "TensorRT 10.3 GA for Linux x86_64 and CUDA 12.0 to 12.5 TAR Package" and the DEB package
Install the DEB package with Software Install.

Alternatively, do the following commands

```bash
sudo dpkg -i ./nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5/nv-tensorrt-local-620E7D29-keyring.gpg /usr/share/keyrings/

sudo apt update
sudo apt install nv-tensorrt-local-repo-ubuntu2204-10.3.0-cuda-12.5
```

We also need to link the libraries. Unpack the tar package:

```bash
tar xzvf ./TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
```

Then. move the unpacked directory to the installation path (~/Documents/), and add to bashrc

```bash
# TensorRT
export TRT_LIBPATH="/home/tk/Documents/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/"
export LD_LIBRARY_PATH="/home/tk/Documents/TensorRT-10.3.0.26/lib/:$TRT_LIBPATH:$LD_LIBRARY_PATH"
```


Install to Python using the following command

```bash
cd ~/Documents/TensorRT-10.3.0.26/python/
pip install ./tensorrt-10.3.0-cp38-none-linux_x86_64.whl
```


## Evaluate Pre-trained Checkpoints

Hop + Trot transition demo:

### Need a new ckpt, this one doesn't work til the end....

```bash
python ./scripts/eval.py --checkpoint=checkpoints/very_large.ckpt -o eval_output_dir/ --generate_data=False 
```







