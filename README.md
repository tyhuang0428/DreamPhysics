# DreamPhysics
*DreamPhysics: Learning Physical Properties of Dynamic 3D Gaussians from Video Diffusion Priors.* A demo for optimizing physical parameters of dynamic 3D Gaussians via the distillation of video diffusion prior.

[![arXiv](https://img.shields.io/badge/arXiv-2406.01476-b31b1b.svg)](https://arxiv.org/abs/2406.01476)

https://github.com/tyhuang0428/DreamPhysics/assets/56391988/08d12ffa-d5fc-4f5b-940b-4c062ab3dd48


### Installation
Since we use original gaussian-splatting as a submodule, please clone this repo and then clone gaussian-splatting as follows:

```shell
git clone https://github.com/tyhuang0428/DreamPhysics
cd DreamPhysics
git clone https://github.com/graphdeco-inria/gaussian-splatting
```

The implementation is mainly based on [PhysGaussian](https://github.com/XPandora/PhysGaussian) and [threestudio](https://github.com/threestudio-project/threestudio). Therefore, the required packages from these two repos should be included by the following commands:
```shell
conda create -n DreamPhysics python=3.9
conda activate DreamPhysics

pip install -r requirements.txt
pip install -e gaussian-splatting/submodules/diff-gaussian-rasterization/
pip install -e gaussian-splatting/submodules/simple-knn/
```

### Quick Start
We support using text-to-video ([ModelScope](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b)) and image-to-video ([Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)) diffusion models to guide the optimization of physical parameters. You can refer to the following command:
```shell
# text-to-video
python ms_simulation.py --model_path <path to gs model> --prompt <input text prompt> --output_path <path to output folder> --physics_config <path to physics-related config file> --guidance_config <path to video-diffusion config file>

# image-to-video
python svd_simulation.py --model_path <path to gs model> --prompt <input image prompt> --output_path <path to output folder> --physics_config <path to physics-related config file> --guidance_config <path to video-diffusion config file>
```

You can download prepared [models](https://huggingface.co/datasets/tyhuang/DreamPhysics/tree/main/model) and corresponding config files, and then have a quick try:
```shell
python ms_simulation.py --model_path ./model/ficus_whitebg-trained/ --prompt "a ficus swaying in the wind" --output_path ./output_ms --physics_config ./config/physics/ficus_config.json

python svd_simulation.py --model_path ./model/ball/ --prompt ./model/ball/input.png --output_path ./output_svd --physics_config ./config/physics/ball_config.json
```

We will keep increasing the scale of our 3D assets. You can also load your own 3D Gaussian pre-trained models to this pipeline. For the setting details of physical configs, you can refer to [PhysGaussian](https://github.com/XPandora/PhysGaussian). Note that, to optimize Young's modulus (E), we rescale the value of E (reduced by 1e7 times, e.g., 1.0 represents 1e7). The common Young's modulus is between 1e4 and 1e7, so make sure that you set an appropriate value.

### Limitation
- Only support collision and rotation of soft-body objects.
- Only support the optimization of Young's modulus (E).
- The simulation is unstable as the physical property changes.

### TODO
- Add more types of physical animation.
- Implement the optimization of other physical properties.
- Increase the scale of available 3D assets.

**Since it is a simple demo for optimizing physical parameters of dynamic 3D Gaussians, we are willing to discuss and improve the implementation.**

### Acknowledgement
This repo is built based on several open-sourced projects:

- 3D generation project [threestudio](https://github.com/threestudio-project/threestudio) and its extension for [Animate124](https://github.com/HeliosZhao/Animate124/tree/threestudio).
- WARP-based MPM solver [warp-mpm](https://github.com/zeshunzong/warp-mpm) and physics-based 3D Gaussian method [PhysGaussian](https://github.com/XPandora/PhysGaussian).

We also use [LGM](https://github.com/3DTopia/LGM) to generate 3D assets with 3D GS representation.

### Citation
```
@article{huang2024dreamphysics,
  title={DreamPhysics: Learning Physical Properties of Dynamic 3D Gaussians with Video Diffusion Priors},
  author={Huang, Tianyu and Zeng, Yihan and Li, Hui and Zuo, Wangmeng and Lau, Rynson WH},
  journal={arXiv preprint arXiv:2406.01476},
  year={2024}
}
```
