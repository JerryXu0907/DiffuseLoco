# DiffuseLoco: Real-Time Legged Locomotion Control with Diffusion from Offline Datasets


[[Paper]](https://arxiv.org/abs/2404.19264)




[Xiaoyu Huang](http://cheng-chi.github.io/)<sup>1</sup>,
[Yufeng Chi](https://www.cs.cmu.edu/~sfeng/)<sup>1</sup>,
[Ruofeng Wang](https://yilundu.github.io/)<sup>1</sup>,
[Zhongyu Li](https://www.zhenjiaxu.com/)<sup>1</sup>,
[Xuebin Peng](https://www.eacousineau.com/)<sup>2</sup>,
[Sophia Shao](http://www.benburchfiel.com/)<sup>1</sup>,
[Borivoje Nikolic](https://www.cs.columbia.edu/~shurans/)<sup>1</sup>
[Koushil Sreenath](https://www.cs.columbia.edu/~shurans/)<sup>1</sup>


<sup>1</sup>UC Berkeley,
<sup>2</sup>Simon Fraser University

## Setup


### Simulation

First, create the conda environment:

```bash
conda create -n diff python=3.8
```

Then, install the python dependencies:

```bash
pip install -r requirements.txt
```




## 🆕 Evaluate Pre-trained Checkpoints
Hop + Trot transition demo:

### Need a new ckpt, this one doesn't work til the end....
```
python ./scripts/eval.py --checkpoint=checkpoints/very_large.ckpt -o eval_output_dir/ --generate_data=False 
```


## 🗺️ Codebase Tutorial
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







