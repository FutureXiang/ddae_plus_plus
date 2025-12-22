# DDAE++ × Lightning-DiT

<p align="center">
  <img src="https://github.com/user-attachments/assets/e94248c2-cf6d-4ad1-b1e2-7703e6278689" width="800">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/769cf308-9b89-4286-951c-ff84df93eeac" width="800">
</p>

This is a multi-gpu PyTorch implementation of the latent-space experiments in paper [DDAE++: Enhancing Diffusion Models Towards Unified Generative and Discriminative Learning](https://arxiv.org/abs/2505.10999).

## TL;DR
- **Self-conditioning:** We enhance diffusion architectures (e.g., UNet, UViT, DiT) by conditioning the decoding layers on features encoded by themselves. It is simple to implement, yet jointly **improves both generation and representation quality at almost no cost**.
- **Contrastive self-distillation:** This also facilitates effective integration of contrastive regularizations, which further boost linear probing accuracy without sacrificing FID.

## Connection to concurrent works
- Enhancing diffusion models *with external knowledge*
  - [(REPA) Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think](https://arxiv.org/abs/2410.06940)
  - [(LSEP) No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models](https://arxiv.org/abs/2509.21565)
- Enhancing diffusion models *in self-supervised manner*
  - [(SRA) No Other Representation Component Is Needed: Diffusion Transformers Can Provide Representation Guidance by Themselves](https://arxiv.org/abs/2505.02831)
  - [(DispLoss) Diffuse and Disperse: Image Generation with Representation Regularization](https://arxiv.org/abs/2506.09027)

## Results on ImageNet-1k (256x256)
Our approach concurrently enhances generation quality (FID and IS, w/ and w/o CFG) and representation quality (linear probing accuracy and ADE20K/VOC2012 segmentation mIoU). The effectiveness is demonstrated across DiT-B/L/XL models in class-conditional/unconditional settings. See more results in the paper.

**Comparison with Lightning-DiT**
<p align="center">
  <img src="https://github.com/user-attachments/assets/232b5f92-fdb6-4827-82d0-b0df9f0688bf" width="800">
</p>

**Class-conditional models**

| Model |     Method    | Epoch | FID w/o CFG↓ | FID w/ CFG↓ | IN-1k Linear Acc↑ |
|:------|:--------------|:-----:|:------------:|:-----------:|:-----------------:|
| DiT-B | Lightning-DiT |  100  |    12.70     |    2.87     |       62.01       |
| DiT-B | + Self-Cond   |  100  |    11.89     |    2.67     |       62.60       |
| DiT-L | Lightning-DiT |  100  |     5.53     |    2.09     |       65.34       |
| DiT-L | + Self-Cond   |  100  |     5.30     |    2.04     |       66.68       |
| DiT-XL| Lightning-DiT |  100  |     4.72     |    1.79     |       66.74       |
| DiT-XL| + Self-Cond   |  100  |   **4.52**   |  **1.73**   |       67.15       |

**Unconditional models**

| Model |     Method    | Epoch |   FID↓  | IN-1k Linear Acc↑ | ADE20K Linear mIoU↑ | VOC2012 Linear mIoU↑ |
|:------|:--------------|:-----:|:-------:|:-----------------:|:-------------------:|:--------------------:|
| DiT-B | Lightning-DiT |  400  |  18.86  |       64.93       |        30.48        |         64.34        |
| DiT-B | + Self-Cond   |  400  |  17.94  |       66.55       |        30.75        |         65.61        |
| DiT-L | Lightning-DiT |  400  |   8.73  |       70.15       |        33.04        |         70.52        |
| DiT-L | + Self-Cond   |  400  |   8.07  |     **71.33**     |      **33.77**      |       **71.34**      |

## Repository overview

This repo contains:
- [x] A simplified, easier-to-use version of [LightningDiT](https://github.com/hustvl/LightningDiT), including training and sampling
- [x] Linear probing evaluation of DiT representations
- [x] Our **self-conditioning** implementation on DiT (less than 10 lines of code!)
- [ ] Contrastive self-distillation on latent space
- [ ] (Maybe will) try the Dispersive Loss

## Requirements
In addition to PyTorch environments, please install:
```sh
pip install pyyaml tensorboard timm einops torchdiffeq safetensors omegaconf scipy tqdm
```
Minimal GPU requirement: 6 x NVIDIA RTX 4090 GPU. Batch sizes and activation checkpointing are configurable in yaml.

## Usage

### VA-VAE tokenizer
Download the [pre-trained VA-VAE tokenizer](https://github.com/hustvl/LightningDiT?tab=readme-ov-file#inference-with-pre-trained-models) (`vavae-imagenet256-f16d32-dinov2.pt`) and place it in the current directory.

### ImageNet-1k pre-processing
```sh
# Suppose that data/imagenet1k/[train|val]/ has 1000 folders like "n01440764"
# This will store latent files under data/imagenet1k-vavae/[train_256|valid_256]
CUDA_VISIBLE_DEVICES=0,1 ./runtask.sh extract --data_path data/imagenet1k/train/ --data_split train --output_path data/imagenet1k-vavae
CUDA_VISIBLE_DEVICES=0,1 ./runtask.sh extract --data_path data/imagenet1k/val/   --data_split valid --output_path data/imagenet1k-vavae

# You may want to rename the directories to match the "data_path" & "valid_data_path" in the yaml
```

### ADE20K & VOC2012 pre-processing (optional, for segmentation)
```sh
# ADE20K (ADEChallengeData2016.zip)
CUDA_VISIBLE_DEVICES=0,1 ./runtask.sh extract_seg --data_path data/ADEChallengeData2016/ --split_str train --data_split train --output_path data/ade20k-vavae
CUDA_VISIBLE_DEVICES=0,1 ./runtask.sh extract_seg --data_path data/ADEChallengeData2016/ --split_str valid --data_split valid --output_path data/ade20k-vavae

# VOC2012 (VOCtrainval_11-May-2012.tar)
CUDA_VISIBLE_DEVICES=0,1 ./runtask.sh extract_seg --data_path data/VOCdevkit/VOC2012/    --split_str train --data_split train --output_path data/voc2012-vavae
CUDA_VISIBLE_DEVICES=0,1 ./runtask.sh extract_seg --data_path data/VOCdevkit/VOC2012/    --split_str valid --data_split valid --output_path data/voc2012-vavae

# You may want to rename the directories to match the "train_data_path" & "valid_data_path" in the yaml
```

### Training
```sh
# Lightning-DiT (e.g., unconditional DiT-B, 400 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh train     configs/imagenet1k/BASEunconditional.yaml

# + Self-conditioning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh train     configs/imagenet1k/BASEunconditional_selfcond9.yaml
```

### Sampling and FID/IS calculation
```sh
# Sampling (e.g., checkpoint at 200 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh inference configs/imagenet1k/BASEunconditional.yaml --epoch 200
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh inference configs/imagenet1k/BASEunconditional_selfcond9.yaml --epoch 200
```
Please use the [pytorch-fid](https://github.com/mseitzer/pytorch-fid) package, or ADM's [evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to calculate FID/IS metrics. Note that only 10k samples are generated by default for efficiency.

### Linear probing (classification, segmentation)
```sh
# Linear classification (e.g., checkpoint at 400 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh linear    configs/imagenet1k/BASEunconditional.yaml --epoch 399
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh linear    configs/imagenet1k/BASEunconditional_selfcond9.yaml --epoch 399

# Linear segmentation (e.g., checkpoint at 400 epochs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh seg       configs/imagenet1k/BASEunconditional.yaml --epoch 399
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./runtask.sh seg       configs/imagenet1k/BASEunconditional_selfcond9.yaml --epoch 399
```

## Citation

If you find our work useful, please cite our related paper:

```bibtex
@article{xiang2025ddaepp,
  title={DDAE++: Enhancing Diffusion Models Towards Unified Generative and Discriminative Learning},
  author={Xiang, Weilai and Yang, Hongyu and Huang, Di and Wang, Yunhong},
  journal={arXiv preprint arXiv:2505.10999},
  year={2025}
}

@inproceedings{xiang2023ddae,
  title={Denoising Diffusion Autoencoders are Unified Self-supervised Learners},
  author={Xiang, Weilai and Yang, Hongyu and Huang, Di and Wang, Yunhong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## Acknowledgments
This repository is built on numerous open-source codebases such as [LightningDiT](https://github.com/hustvl/LightningDiT) and [ADM](https://github.com/openai/guided-diffusion).
