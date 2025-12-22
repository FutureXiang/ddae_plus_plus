import argparse
import os
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.lightningdit import LightningDiT_models
from models.RectifiedFlow import RectifiedFlow
from datasets.img_latent_dataset import ImgLatentDataset
from utils.misc import Config, init_seeds, gather_tensor, DataLoaderDDP, print0, load_checkpoint_ema


class Classifier(nn.Module):
    def __init__(self, feat_fn, time_list, name_list, base_lr, epoch, input_shape, local_rank, num_classes, use_BN):
        super(Classifier, self).__init__()
        self.feat_fn = feat_fn
        self.times = time_list
        self.names = name_list
        self.classifiers = nn.ModuleDict()
        self.optims = {}
        self.schedulers = {}
        self.loss_fn = nn.CrossEntropyLoss()
        device = "cuda:%d" % local_rank

        for time in self.times:
            feats = self.feat_fn(torch.zeros(1, *input_shape).to(device), time)
            if self.names is None:
                self.names = list(feats.keys()) # all available names

            for name in self.names:
                key = self.make_key(time, name)
                layers = nn.Linear(feats[name].shape[1], num_classes)
                if use_BN:
                    layers = torch.nn.SyncBatchNorm.convert_sync_batchnorm(nn.Sequential(
                        nn.BatchNorm1d(num_features=feats[name].shape[1], affine=False),
                        layers
                    ))
                layers = torch.nn.parallel.DistributedDataParallel(
                    layers.to(device), device_ids=[local_rank], output_device=local_rank)
                optimizer = torch.optim.Adam(layers.parameters(), lr=base_lr)
                scheduler = CosineAnnealingLR(optimizer, epoch)
                self.classifiers[key] = layers
                self.optims[key] = optimizer
                self.schedulers[key] = scheduler

    def train(self, x, y):
        self.classifiers.train()
        for time in self.times:
            feats = self.feat_fn(x, time)
            for name in self.names:
                key = self.make_key(time, name)
                representation = feats[name].detach()
                logit = self.classifiers[key](representation)
                loss = self.loss_fn(logit, y)

                self.optims[key].zero_grad()
                loss.backward()
                self.optims[key].step()
        return loss

    def test(self, x):
        outputs = {}
        with torch.no_grad():
            self.classifiers.eval()
            for time in self.times:
                feats = self.feat_fn(x, time)
                for name in self.names:
                    key = self.make_key(time, name)
                    representation = feats[name].detach()
                    logit = self.classifiers[key](representation)
                    pred = logit.argmax(dim=-1)
                    outputs[key] = pred
        return outputs

    def make_key(self, t, n):
        return ("time %.2f, name %s" % (t, n)).replace('.', '_')

    def get_lr(self):
        key = self.make_key(self.times[0], self.names[0])
        optim = self.optims[key]
        return optim.param_groups[0]['lr']

    def schedule_step(self):
        for time in self.times:
            for name in self.names:
                key = self.make_key(time, name)
                self.schedulers[key].step()


def train(opt):

    def test():
        preds = {k: [] for k in classifiers.optims.keys()}
        accs = {}
        labels = []
        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            outputs = classifiers.test(image.to(device))
            for key in outputs:
                preds[key].append(outputs[key])
            labels.append(label.to(device))

        for key in preds:
            preds[key] = torch.cat(preds[key])
        labels = torch.cat(labels)
        dist.barrier()
        labels = gather_tensor(labels)
        for key in preds:
            pred = gather_tensor(preds[key])
            accs[key] = (pred == labels).sum().item() / len(labels)
        return accs


    # ===== config =====

    yaml_path = opt.config
    local_rank = opt.local_rank
    use_bf16 = opt.use_bf16
    ep = opt.epoch
    grid_search = opt.grid

    opt = Config(yaml_path)

    # ===== model =====

    device = "cuda:%d" % local_rank
    latent_size = opt.data['latent_size']
    in_chans = opt.model['in_chans']
    latent_shape = (in_chans, latent_size, latent_size)
    model = LightningDiT_models[opt.model['model_type']](
        input_size=latent_size,
        num_classes=opt.data['num_classes'],
        use_qknorm=opt.model['use_qknorm'],
        in_channels=in_chans,
        btnk_blockname=opt.model['btnk_blockname'] if 'btnk_blockname' in opt.model else None
    )
    model.to(device)

    if hasattr(opt, 'load_path'):
        target = opt.load_path
    else:
        if ep == -1:
            ep = opt.train['n_epoch'] - 1
        target = os.path.join(opt.save_dir, "ckpts", f"model_{ep}.pth")
    print0("Loading from checkpoint:", target, "for linear probing")
    checkpoint = torch.load(target, map_location=device, weights_only=True)
    load_checkpoint_ema(model, checkpoint)
    model.eval()

    rf = RectifiedFlow(device)

    def feat_fn(x, t):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            with torch.no_grad():
                return rf.get_feature(model.forward_get_act, x, t, clean=False)

    # ===== linear args =====

    epoch = opt.linear['n_epoch']
    global_bs = opt.linear['global_batch_size']
    lr = opt.linear['lr']

    if grid_search:
        time_list = [opt.linear['timestep'] - 0.05, opt.linear['timestep'], opt.linear['timestep'] + 0.05]
        name_list = None
    else:
        time_list = [opt.linear['timestep']]
        name_list = [opt.linear['blockname']]

    # ===== data =====

    train_set = ImgLatentDataset(data_dir=opt.data['data_path'], latent_norm=opt.data['latent_norm'])
    print0("train dataset:", len(train_set))
    valid_set = ImgLatentDataset(data_dir=opt.linear['valid_data_path'], latent_norm=opt.data['latent_norm'])
    print0("valid dataset:", len(valid_set))

    train_loader, sampler = DataLoaderDDP(train_set, global_bs=global_bs, shuffle=True)
    valid_loader, _ = DataLoaderDDP(valid_set, global_bs=global_bs, shuffle=False)

    # ===== linear probing loop =====

    classifiers = Classifier(feat_fn, time_list, name_list,
                             lr, epoch, latent_shape, local_rank, opt.linear['cls_num_classes'],
                             opt.linear['use_BN'] if 'use_BN' in opt.linear else False).to(device)

    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            loss = classifiers.train(image.to(device), label.to(device))
            if grid_search:
                pbar.set_description("[epoch %d / iter %d]: lr: %.1e" % (e, i, classifiers.get_lr()))
            else:
                pbar.set_description("[epoch %d / iter %d]: lr: %.1e loss: %.4f" % (e, i, classifiers.get_lr(), loss.item()))
        classifiers.schedule_step()

    accs = test()
    for key, acc in accs.items():
        print0("[%s]: Test acc: %.2f" % (key, acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_bf16", action='store_true', default=False)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--grid", action='store_true', default=False)
    opt = parser.parse_args()
    opt.local_rank = int(os.environ['LOCAL_RANK'])
    print0(opt)

    init_seeds(no=opt.local_rank)
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    train(opt)
