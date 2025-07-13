# DDAE++ × LightningDiT

<p align="center">
  <img src="https://github.com/user-attachments/assets/e94248c2-cf6d-4ad1-b1e2-7703e6278689" width="800">
</p>

This is a multi-gpu PyTorch implementation of the latent-space experiments in paper [DDAE++: Enhancing Diffusion Models Towards Unified Generative and Discriminative Learning](https://arxiv.org/abs/2505.10999):

## TL;DR
- **Self-conditioning:** We enhance diffusion architectures (e.g., UNet & LightningDiT), by conditioning the decoding process on features learned by themselves. It is simple to implement, yet jointly **improves both FID and linear accuracy at almost no cost**.
- **Contrastive self-distillation:** This also facilitates an effective integration of contrastive regularizations, which further boost linear accuracy without sacrificing FID.

## Connection to other works
<p align="center">
  <img src="https://github.com/user-attachments/assets/902acd42-7952-469c-bbf6-c4720e0fbcda" width="500">
</p>

## Results on ImageNet-1k-256x256
Our approach concurrently enhances linear probing accuracy and generation quality, as evaluated by FID and Inception Score (with and w/o CFG). The effectiveness is demonstrated across both **DiT-Base and DiT-Large** models in **class-conditional and unconditional** settings.
#### DiT-Base, class-conditional, 100 epochs
<p align="center">
<img src="https://github.com/user-attachments/assets/22236c30-d5a7-428f-8f18-1aeb1000f972" height="250">
<img src="https://github.com/user-attachments/assets/74668422-c971-4442-9099-82aa3d20cd6f" height="250">
</p>

#### DiT-Base, unconditional, 400 epochs
<p align="center">
<img src="https://github.com/user-attachments/assets/73460267-a7dd-4616-a90c-b0c729000b94" height="250">
</p>

#### DiT-Large, class-conditional, 100 epochs
<p align="center">
<img src="https://github.com/user-attachments/assets/cfc3a1c9-8716-44ae-a2ab-6598f1a7c405" height="250">
<img src="https://github.com/user-attachments/assets/58cd6796-3fec-4298-8287-45fb0c2653d8" height="250">
</p>

#### DiT-Large, unconditional, 200 epochs


## Repository overview

This repo contains:
- [x] A simplified, easier-to-use version of [LightningDiT](https://github.com/hustvl/LightningDiT), including training and sampling
- [x] Linear probing evaluation of DiT representations, including optimal layer-noise hyper-parameter selection
- [x] Our **self-conditioning** implementation on DiT (less than 10 lines of code!)
- [ ] Contrastive self-distillation on latent space
- [ ] (Maybe will) try the Dispersive Loss

## Acknowledgments
This repository is built on numerous open-source codebases such as [LightningDiT](https://github.com/hustvl/LightningDiT) and [ADM](https://github.com/openai/guided-diffusion).
