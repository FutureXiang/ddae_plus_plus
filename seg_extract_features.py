import argparse
import os
from safetensors.torch import save_file
from datetime import datetime

import torch
import torch.distributed as dist
from datasets.ade20k_voc2012_dataset import ADE20KDataset, VOC2012Dataset
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

    ade_ = 'ade' in opt.data_path.lower()
    voc_ = 'voc' in opt.data_path.lower()
    if ade_ and not voc_:
        DATASET = ADE20KDataset
    elif voc_ and not ade_:
        DATASET = VOC2012Dataset
    else:
        raise NotImplementedError()

    # for image + mask, NO flip for simplicity
    dataset = DATASET(opt.data_path, split=opt.split_str, transform=tokenizer.img_transform(p_hflip=0.0))
    loader = DataLoaderDDP(
        dataset,
        global_bs=opt.batch_size * dist.get_world_size(),
        shuffle=False
    )[0]
    total_data_in_loop = len(loader.dataset)
    print0(f"Total data in one loop: {total_data_in_loop}")

    # ===== extracting loop =====

    run_images = 0
    saved_files = 0
    latents = []
    labels = []
    for batch_idx, (x, y) in enumerate(loader):
        run_images += x.shape[0]
        if run_images % 100 == 0:
            print0(f'{datetime.now()} processing {run_images} of {total_data_in_loop} images')

        z = tokenizer.encode_images(x).detach().cpu()  # (N, C, H, W)

        if batch_idx == 0:
            print0('latent shape', z.shape, 'dtype', z.dtype)
            print0('mask shape', y.shape, 'dtype', y.dtype)

        latents.append(z)
        labels.append(y)

        if len(latents) == 10000 // opt.batch_size:
            latents = torch.cat(latents, dim=0)
            labels = torch.cat(labels, dim=0)
            save_dict = {
                'latents': latents,
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
            labels = []
            saved_files += 1

    # save remainder latents that are fewer than 10000 images
    if len(latents) > 0:
        latents = torch.cat(latents, dim=0)
        labels = torch.cat(labels, dim=0)
        save_dict = {
            'latents': latents,
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
    parser.add_argument("--data_path", type=str, default='/path/to/ADEChallengeData2016_or_VOC2012')
    parser.add_argument("--split_str", type=str, default='training')
    parser.add_argument("--data_split", type=str, default='ade20k_train')
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
