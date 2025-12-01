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
Our approach concurrently enhances generation quality (FID and IS, w/ and w/o CFG) and representation quality (linear probing accuracy and ADE20K/VOC2012 segmentation mIoU). The effectiveness is demonstrated across DiT-B/L models in unconditional/class-conditional settings. See more results in the paper.

**Comparison with Lightning-DiT**
<p align="center">
  <img src="https://github.com/user-attachments/assets/3dfb374d-ab65-48c6-8e51-d47b4691c3fd" width="800">
</p>

## Repository overview

This repo contains:
- [x] A simplified, easier-to-use version of [LightningDiT](https://github.com/hustvl/LightningDiT), including training and sampling
- [x] Linear probing evaluation of DiT representations
- [x] Our **self-conditioning** implementation on DiT (less than 10 lines of code!)
- [ ] Contrastive self-distillation on latent space
- [ ] (Maybe will) try the Dispersive Loss

## Acknowledgments
This repository is built on numerous open-source codebases such as [LightningDiT](https://github.com/hustvl/LightningDiT) and [ADM](https://github.com/openai/guided-diffusion).
