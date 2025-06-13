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
  <img src="https://github.com/user-attachments/assets/902acd42-7952-469c-bbf6-c4720e0fbcda" width="600">
</p>

## Results on ImageNet-1k-256x256
We achieve **66.55%** linear probing accuracy using DiT-base on ImageNet-1k, while also improving generation FID and Inception Score.

<table><thead>
  <tr>
    <th>Unconditional, DiT-Base</th>
    <th colspan="2">LightningDiT Baseline</th>
    <th colspan="2">+Self-conditioning</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Epoch</td>
    <td>FID_10k</td>
    <td>LP Acc.</td>
    <td>FID_10k</td>
    <td>LP Acc.</td>
  </tr>
  <tr>
    <td>20</td>
    <td>43.44 </td>
    <td>61.54 </td>
    <td>42.63 </td>
    <td>62.05 </td>
  </tr>
  <tr>
    <td>40</td>
    <td>35.39 </td>
    <td>62.87 </td>
    <td>34.22 </td>
    <td>62.99 </td>
  </tr>
  <tr>
    <td>60</td>
    <td>31.66 </td>
    <td>63.20 </td>
    <td>30.75 </td>
    <td>63.74 </td>
  </tr>
  <tr>
    <td>80</td>
    <td>29.31 </td>
    <td>63.45 </td>
    <td>28.60 </td>
    <td>64.36 </td>
  </tr>
  <tr>
    <td>100</td>
    <td>27.76 </td>
    <td>63.77 </td>
    <td>26.85 </td>
    <td>64.59 </td>
  </tr>
  <tr>
    <td>200</td>
    <td>24.19 </td>
    <td>64.50 </td>
    <td>23.14 </td>
    <td>65.50 </td>
  </tr>
  <tr>
    <td>300</td>
    <td>22.55 </td>
    <td>64.80 </td>
    <td>21.73 </td>
    <td>66.22 </td>
  </tr>
  <tr>
    <td>400</td>
    <td>21.26 </td>
    <td>64.93 </td>
    <td>20.85 </td>
    <td>66.55 </td>
  </tr>
</tbody></table>

<table><thead>
  <tr>
    <th>Conditional, DiT-Base</th>
    <th colspan="4">LightningDiT Baseline</th>
    <th colspan="4">+Self-conditioning</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Epoch</td>
    <td>FID_10k</td>
    <td>LP Acc.</td>
    <td>FID_10k_CFG</td>
    <td>IS_10k_CFG</td>
    <td>FID_10k</td>
    <td>LP Acc.</td>
    <td>FID_10k_CFG</td>
    <td>IS_10k_CFG</td>
  </tr>
  <tr>
    <td>20</td>
    <td>31.64 </td>
    <td>60.90 </td>
    <td>15.35 </td>
    <td>83.16 </td>
    <td>29.82 </td>
    <td>61.05 </td>
    <td>13.31 </td>
    <td>93.70 </td>
  </tr>
  <tr>
    <td>40</td>
    <td>22.16 </td>
    <td>61.35 </td>
    <td>8.62 </td>
    <td>124.99 </td>
    <td>21.39 </td>
    <td>61.44 </td>
    <td>7.86 </td>
    <td>139.14 </td>
  </tr>
  <tr>
    <td>60</td>
    <td>18.44 </td>
    <td>61.63 </td>
    <td>6.69 </td>
    <td>151.21 </td>
    <td>17.54 </td>
    <td>61.92 </td>
    <td>6.28 </td>
    <td>166.32 </td>
  </tr>
  <tr>
    <td>80</td>
    <td>16.58 </td>
    <td>61.77 </td>
    <td>6.13 </td>
    <td>162.90 </td>
    <td>15.74 </td>
    <td>62.17 </td>
    <td>5.82 </td>
    <td>179.47 </td>
  </tr>
  <tr>
    <td>100</td>
    <td>15.51 </td>
    <td>62.01 </td>
    <td>5.72 </td>
    <td>172.01 </td>
    <td>14.52 </td>
    <td>62.39 </td>
    <td>5.58 </td>
    <td>189.66 </td>
  </tr>
</tbody></table>

## Repository overview

This repo contains:
- [x] A simplified, easier-to-use version of [LightningDiT](https://github.com/hustvl/LightningDiT), including training and sampling
- [x] Linear probing evaluation of DiT representations, including optimal layer-noise hyper-parameter selection
- [x] Our **self-conditioning** implementation on DiT (less than 10 lines of code!)
- [ ] Contrastive self-distillation on latent space
- [ ] (Maybe will) try the Dispersive Loss
