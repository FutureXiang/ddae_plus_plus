import argparse
import json
import os
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from models.lightningdit import LightningDiT_models
from models.RectifiedFlow import RectifiedFlow
from datasets.img_latent_dataset import ImgLatentDataset
from utils.ema import EMA
from utils.misc import Meter, Config, init_seeds, reduce_tensor, DataLoaderDDP, print0


def train(opt):

    # ===== config =====

    yaml_path = opt.config
    local_rank = opt.local_rank
    use_bf16 = opt.use_bf16

    opt = Config(yaml_path)

    model_dir = os.path.join(opt.save_dir, "ckpts")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)

        tsbd_dir = os.path.join(opt.save_dir, "tensorboard")
        writer = SummaryWriter(log_dir=tsbd_dir)
        config_str=json.dumps(opt._original_content, indent=2)
        writer.add_text('training configs', config_str, global_step=0)

    # ===== model =====

    device = "cuda:%d" % local_rank
    model = LightningDiT_models[opt.model['model_type']](
        input_size=opt.data['latent_size'],
        num_classes=opt.data['num_classes'],
        use_qknorm=opt.model['use_qknorm'],
        in_channels=opt.model['in_chans'],
        use_checkpoint=opt.train['use_checkpoint'],
        btnk_blockname=opt.model['btnk_blockname'] if 'btnk_blockname' in opt.model else None
    )
    model.to(device)
    if local_rank == 0:
        ema = EMA(model, beta=opt.train['ema'])
        ema.to(device)
        ema.eval()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank)

    rf = RectifiedFlow(device, use_lognorm=opt.train['use_lognorm'])

    # ===== optim =====

    optim = torch.optim.AdamW(model.parameters(),
                              lr=opt.optimizer['lr'],
                              weight_decay=opt.optimizer['weight_decay'],
                              betas=(opt.optimizer['beta1'], opt.optimizer['beta2']))

    # ===== resume? =====

    if opt.load_epoch != -1:
        target = os.path.join(model_dir, f"model_{opt.load_epoch}.pth")
        print0("Resuming training from checkpoint:", target)
        checkpoint = torch.load(target, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        if local_rank == 0:
            ema.load_state_dict(checkpoint['ema'])
        optim.load_state_dict(checkpoint['opt'])

    # ===== data =====

    dataset = ImgLatentDataset(data_dir=opt.data['data_path'], latent_norm=opt.data['latent_norm'])
    print0("train dataset:", len(dataset))
    loader, sampler = DataLoaderDDP(dataset, global_bs=opt.train['global_batch_size'], shuffle=True)
    print0(f"Global (total) learning rate = {opt.optimizer['lr']}")

    # ===== training loop =====

    for ep in range(opt.load_epoch + 1, opt.train['n_epoch']):
        sampler.set_epoch(ep)
        dist.barrier()

        # training
        model.train()
        if local_rank == 0:
            now_lr = optim.param_groups[0]['lr']
            meter = Meter(n_items=3)
            pbar = tqdm(loader)
        else:
            pbar = loader
        for x, y in pbar:
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss_dict = rf(model, x, dict(y=y))
            mse_loss = loss_dict["loss"].mean()
            cos_loss = loss_dict["cos_loss"].mean()
            loss = mse_loss + cos_loss
            loss.backward()
            optim.step()

            # logging
            dist.barrier()
            loss = reduce_tensor(loss)
            if local_rank == 0:
                ema.update()
                meter.update([loss.item(), mse_loss.item(), cos_loss.item()])
                pbar.set_description(f"[epoch {ep}, lr {now_lr:.5f}] loss: {meter.get(0):.4f} = {meter.get(1):.4f} + {meter.get(2):.4f}")

        if local_rank == 0:
            writer.add_scalar('lr', now_lr, ep)
            writer.add_scalar('loss', meter.get(0), ep)
            writer.add_scalar('mse_loss', meter.get(1), ep)
            writer.add_scalar('cos_loss', meter.get(2), ep)

            if ep % opt.train['save_every'] == 0 or ep == opt.train['n_epoch'] - 1:
                pass
            else:
                continue

            checkpoint = {
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'opt': optim.state_dict(),
            }
            save_path = os.path.join(model_dir, f"model_{ep}.pth")
            torch.save(checkpoint, save_path)
            print('saved model at', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_bf16", action='store_true', default=False)
    opt = parser.parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    print0(opt)

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    train(opt)
