import argparse
from pathlib import Path
import math
import os
from functools import partial
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.distributed as dist
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from models.lightningdit import LightningDiT_models
from models.RectifiedFlow import RectifiedFlow
from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE
from utils.misc import Config, init_seeds, gather_tensor, print0, load_checkpoint_ema


def make_grid_numpy(array, columns=4):
    n, h, w, c = array.shape

    x = min(columns, n)
    y = n // x
    assert x * y == n
    grid = np.zeros((y * h, x * w, c), dtype=array.dtype)

    for idx, arr in enumerate(array):
        i, j = divmod(idx, x)
        grid[i*h: (i+1)*h, j*w: (j+1)*w] = arr
    return grid


def sample(opt):

    # ===== config =====

    yaml_path = opt.config
    local_rank = opt.local_rank
    ep = opt.epoch

    opt = Config(yaml_path)

    # ===== model =====

    device = "cuda:%d" % local_rank
    latent_size = opt.data['latent_size']
    num_classes = opt.data['num_classes']
    model = LightningDiT_models[opt.model['model_type']](
        input_size=latent_size,
        num_classes=num_classes,
        use_qknorm=opt.model['use_qknorm'],
        in_channels=opt.model['in_chans'],
        btnk_blockname=opt.model['btnk_blockname'] if 'btnk_blockname' in opt.model else None
    )
    model.to(device)

    if hasattr(opt, 'load_path'):
        target = opt.load_path
    else:
        if ep == -1:
            ep = opt.train['n_epoch'] - 1
        target = os.path.join(opt.save_dir, "ckpts", f"model_{ep}.pth")
    print0("Loading from checkpoint:", target, "for sampling")
    checkpoint = torch.load(target, map_location=device, weights_only=True)
    load_checkpoint_ema(model, checkpoint)
    model.eval()

    vae = VA_VAE(f'tokenizer/configs/{opt.data["vae_model_name"]}.yaml')

    # ===== CFG and Sampler =====

    method = opt.sample['sampling_method']
    steps = opt.sample['num_sampling_steps']
    timestep_shift = opt.sample_arg('timestep_shift', None)

    if num_classes > 0:
        cfg_scale = opt.sample_arg('cfg_scale', None)
        interval_start = opt.sample_arg('cfg_interval_start', None)
        print0(f"Class-conditional, using timeshift={timestep_shift}, cfg={cfg_scale}, interval_start={interval_start}")
        using_cfg = (cfg_scale is not None) and (cfg_scale > 1.0)
    else:
        print0(f"Unconditional, using timeshift={timestep_shift}")

    rf = RectifiedFlow(device)
    sample_fn = partial(rf.sample, sampler_type=method, num_steps=steps, timestep_shift=timestep_shift)

    # ===== making folders =====

    if local_rank == 0:
        image_out_name = "Generated"

        if hasattr(opt, 'load_path'):
            exp_dir = os.path.join("output", "load_" + Path(opt.load_path).stem)
        else:
            exp_dir = opt.save_dir
            image_out_name += f"_ep{ep}"

        image_out_name += f"_{method}"
        if method == 'euler':
            image_out_name += f"_steps{steps}"

        if timestep_shift is not None:
            image_out_name += f"_shift{timestep_shift:.2f}"

        if num_classes > 0 and using_cfg:
            image_out_name += f"_cfg{cfg_scale}"
            if interval_start is not None:
                image_out_name += f"_interval{interval_start:.2f}"

        gen_dir = os.path.join(exp_dir, image_out_name)
        os.makedirs(gen_dir)
        gen_dir_png = os.path.join(gen_dir, "pngs")
        os.makedirs(gen_dir_png)
        res = []

    # ===== latent un-norm =====

    dataset = ImgLatentDataset(data_dir=opt.data['data_path'], latent_norm=opt.data['latent_norm'])
    latent_mean, latent_std = dataset.get_latent_stats()
    latent_mean = latent_mean.clone().detach().to(device)
    latent_std = latent_std.clone().detach().to(device)

    # ===== sampling loop =====

    gpus = dist.get_world_size()
    local_bs = opt.sample['local_batch_size']
    global_bs = local_bs * gpus
    total_samples = int(math.ceil(opt.sample['fid_num'] / global_bs) * global_bs)
    local_samples = total_samples // gpus
    batches = local_samples // local_bs

    for batch in tqdm(range(batches), disable=(local_rank != 0)):
        with torch.no_grad():
            z = torch.randn(local_bs, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, num_classes, (local_bs,), device=device) if num_classes > 0 else None

            if num_classes > 0 and using_cfg:
                model_kwargs = dict(y=y, cfg_scale=cfg_scale, cfg_interval=True, cfg_interval_start=interval_start)
                model_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                model_fn = model.forward

            samples = sample_fn(model_fn, z, model_kwargs)

            samples = samples * latent_std + latent_mean
            samples = vae.decode_to_images(samples) # 0-255, nhwc, cpu, np array

        dist.barrier(device_ids=[local_rank])
        samples = gather_tensor(torch.as_tensor(samples, device=device).contiguous()).cpu().numpy()
        if local_rank == 0:
            res.append(samples)
            if batch >= 20:
                continue
            # pick a few images to for a quick preview
            few = min(8, samples.shape[0])
            # suppose that few % 4 == 0
            grid = make_grid_numpy(samples[:few], columns=4)
            png_path = os.path.join(gen_dir, f"grid_{batch}.png")
            Image.fromarray(grid).save(png_path)

    if local_rank == 0:
        res = np.concat(res)
        for no, img in tqdm(enumerate(res)):
            png_path = os.path.join(gen_dir_png, f"{no}.png")
            Image.fromarray(img).save(png_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--epoch", type=int, default=-1)
    opt = parser.parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    print0(opt)

    init_seeds(RANDOM_SEED=0, no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    sample(opt)
