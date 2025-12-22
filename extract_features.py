import argparse
import os
from safetensors.torch import save_file
from datetime import datetime

import torch
import torch.distributed as dist
from torchvision.datasets import ImageFolder
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from datasets.img_latent_dataset import ImgLatentDataset
from tokenizer.vavae import VA_VAE
from utils.misc import init_seeds, DataLoaderDDP, print0


def main(opt):

    # ===== config =====

    local_rank = opt.local_rank

    output_dir = os.path.join(opt.output_path, f'{opt.data_split}_{opt.image_size}')
    if local_rank == 0:
        os.makedirs(output_dir)

    # ===== model =====

    tokenizer = VA_VAE(opt.config)

    # ===== data =====

    datasets = [
        ImageFolder(opt.data_path, transform=tokenizer.img_transform(p_hflip=0.0)),
        ImageFolder(opt.data_path, transform=tokenizer.img_transform(p_hflip=1.0))
    ]
    loaders = [
        DataLoaderDDP(
            dataset,
            global_bs=opt.batch_size * dist.get_world_size(),
            shuffle=False
        )[0] for dataset in datasets
    ]
    total_data_in_loop = len(loaders[0].dataset)
    print0(f"Total data in one loop: {total_data_in_loop}")

    # ===== extracting loop =====

    run_images = 0
    saved_files = 0
    latents = []
    latents_flip = []
    labels = []
    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0:
            print0(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')

        for loader_idx, data in enumerate(batch_data):
            x = data[0]
            y = data[1]  # (N,)

            z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

            if batch_idx == 0:
                print0('latent shape', z.shape, 'dtype', z.dtype)

            if loader_idx == 0:
                latents.append(z)
                labels.append(y)
            else:
                latents_flip.append(z)

        if len(latents) == 10000 // opt.batch_size:
            latents = torch.cat(latents, dim=0)
            latents_flip = torch.cat(latents_flip, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                'latents': latents,
                'latents_flip': latents_flip,
                'labels': labels
            }
            for key in save_dict:
                print0(key, save_dict[key].shape)
            save_filename = os.path.join(output_dir, f'latents_rank{local_rank:02d}_shard{saved_files:03d}.safetensors')
            save_file(
                save_dict,
                save_filename,
                metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
            )
            print0(f'Saved {save_filename}')
            
            latents = []
            latents_flip = []
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        latents_flip = torch.cat(latents_flip, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            'latents': latents,
            'latents_flip': latents_flip,
            'labels': labels
        }
        for key in save_dict:
            print0(key, save_dict[key].shape)
        save_filename = os.path.join(output_dir, f'latents_rank{local_rank:02d}_shard{saved_files:03d}.safetensors')
        save_file(
            save_dict,
            save_filename,
            metadata={'total_size': f'{latents.shape[0]}', 'dtype': f'{latents.dtype}', 'device': f'{latents.device}'}
        )
        print0(f'Saved {save_filename}')

    # Calculate latents stats
    dist.barrier()
    if local_rank == 0:
        ImgLatentDataset(output_dir, latent_norm=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/path/to/your/data')
    parser.add_argument("--data_split", type=str, default='imagenet_train')
    parser.add_argument("--output_path", type=str, default="/path/to/your/output")
    parser.add_argument("--config", type=str)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=20)
    opt = parser.parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    print0(opt)

    init_seeds(RANDOM_SEED=42, no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    main(opt)
