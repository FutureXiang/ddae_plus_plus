import argparse
import os
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.lightningdit import LightningDiT_models
from models.RectifiedFlow import RectifiedFlow
from datasets.img_latent_dataset import ImgLatentSegDataset
from utils.misc import Config, init_seeds, DataLoaderDDP, print0, load_checkpoint_ema


def unpack_multi_name(name):
    # "out_57" -> ["out_5", "out_7"]
    numbers = name.split('_')[1:]
    return [f"out_{i}" for i in numbers]


def label_mapping(label, data_path):
    # [0, N-1]: labels to predict
    # 255: ignore in CrossEntropy and mIoU

    ade_ = 'ade' in data_path.lower()
    voc_ = 'voc' in data_path.lower()
    if ade_ and not voc_:
        # [1, 150]: labels  -> [0, 149]
        # 0: ignore         -> 255
        processed_label = label.clone()
        processed_label[processed_label > 0] = processed_label[processed_label > 0] - 1
        processed_label[label == 0] = 255
        return processed_label
    elif voc_ and not ade_:
        # 0: background, but still needs to predict
        # [1, 20]: foreground labels
        # 255: ignore
        return label.clone()
    else:
        raise NotImplementedError()


class Classifier(nn.Module):
    def __init__(self, feat_fn, time_list, name_list, base_lr, epoch, input_shape, local_rank, num_classes, use_LN, interp_size):
        super(Classifier, self).__init__()
        self.feat_fn = feat_fn
        self.times = time_list
        self.names = name_list
        self.classifiers = nn.ModuleDict()
        self.optims = {}
        self.schedulers = {}
        self.interp_size = interp_size
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        device = "cuda:%d" % local_rank

        for time in self.times:
            feats = self.feat_fn(torch.zeros(1, *input_shape).to(device), time)
            if self.names is None:
                self.names = list(feats.keys()) # all available names

            for name in self.names:
                key = self.make_key(time, name)
                # "out_57" -> ["out_5", "out_7"]
                multi_name = unpack_multi_name(name)
                dim = sum([feats[i].shape[2] for i in multi_name])
                layers = nn.Linear(dim, num_classes)
                if use_LN:
                    layers = nn.Sequential(
                        nn.LayerNorm(dim),
                        layers
                    )
                layers = torch.nn.parallel.DistributedDataParallel(
                    layers.to(device), device_ids=[local_rank], output_device=local_rank)
                optimizer = torch.optim.Adam(layers.parameters(), lr=base_lr)
                scheduler = CosineAnnealingLR(optimizer, epoch)
                self.classifiers[key] = layers
                self.optims[key] = optimizer
                self.schedulers[key] = scheduler

    def forward_head(self, representation, key):
        B, L, D = representation.shape
        H = W = int(L**0.5)

        logit = self.classifiers[key](representation)

        logit = logit.permute(0, 2, 1).view(B, -1, H, W) # (B, num_classes, H, W)
        logit = torch.nn.functional.interpolate(logit, size=self.interp_size, mode='bilinear', align_corners=False)
        return logit

    def train(self, x, y):
        self.classifiers.train()
        for time in self.times:
            feats = self.feat_fn(x, time)
            for name in self.names:
                key = self.make_key(time, name)
                # "out_57" -> ["out_5", "out_7"]
                multi_name = unpack_multi_name(name)
                multi_feat = [feats[i].detach() for i in multi_name]
                representation = torch.cat(multi_feat, dim=-1)
                logit = self.forward_head(representation, key)
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
                    # "out_57" -> ["out_5", "out_7"]
                    multi_name = unpack_multi_name(name)
                    multi_feat = [feats[i].detach() for i in multi_name]
                    representation = torch.cat(multi_feat, dim=-1)
                    logit = self.forward_head(representation, key)
                    pred = logit.argmax(dim=1)
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
        def compute_miou_from_confusion_matrix(confusion_matrix):
            # confusion_matrix: (num_classes, num_classes)
            # axis=0: label
            # axis=1: pred
            intersection = torch.diag(confusion_matrix)
            union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
            iou = intersection / (union + 1e-10)

            valid_classes = union > 0
            miou = iou[valid_classes].mean()
            return miou.item()

        confusion_matrices = {
            k: torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
            for k in classifiers.optims.keys()
        }

        for image, label in tqdm(valid_loader, disable=(local_rank!=0)):
            processed_label = label_mapping(label.to(device), opt.seg['valid_data_path'])

            outputs = classifiers.test(image.to(device))
            for key in outputs:
                # (B, H, W)
                valid_mask = (processed_label != 255)
                valid_pred = outputs[key][valid_mask]
                valid_label = processed_label[valid_mask]

                indices = valid_label * num_classes + valid_pred
                confusion_matrices[key] += torch.bincount(
                    indices, 
                    minlength=num_classes * num_classes
                ).reshape(num_classes, num_classes)

        dist.barrier()
        for key in confusion_matrices:
            dist.all_reduce(confusion_matrices[key], op=dist.ReduceOp.SUM)

        mious = {}
        for key, cm in confusion_matrices.items():
            mious[key] = compute_miou_from_confusion_matrix(cm)
        return mious


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
    print0("Loading from checkpoint:", target, "for segmentation")
    checkpoint = torch.load(target, map_location=device, weights_only=True)
    load_checkpoint_ema(model, checkpoint)
    model.eval()

    rf = RectifiedFlow(device)

    def feat_fn(x, t):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            with torch.no_grad():
                return rf.get_feature(model.forward_get_act, x, t, clean=False, use_gap=False)

    # ===== seg args =====

    epoch = opt.seg['n_epoch']
    global_bs = opt.seg['global_batch_size']
    lr = opt.seg['lr']
    num_classes = opt.seg['seg_num_classes']
    interp_size = opt.seg['interp_size']

    if grid_search:
        time_list = [opt.seg['timestep'] - 0.05, opt.seg['timestep'], opt.seg['timestep'] + 0.05]
        name_list = None
    else:
        time_list = [opt.seg['timestep']]
        name_list = [opt.seg['blockname']]

    # ===== data =====

    train_set = ImgLatentSegDataset(data_dir=opt.seg['train_data_path'], latent_norm=opt.data['latent_norm'])
    print0("train dataset:", len(train_set))
    valid_set = ImgLatentSegDataset(data_dir=opt.seg['valid_data_path'], latent_norm=opt.data['latent_norm'])
    print0("valid dataset:", len(valid_set))

    train_loader, sampler = DataLoaderDDP(train_set, global_bs=global_bs, shuffle=True)
    valid_loader, _ = DataLoaderDDP(valid_set, global_bs=global_bs, shuffle=False)

    # ===== seg loop =====

    classifiers = Classifier(feat_fn, time_list, name_list,
                             lr, epoch, latent_shape, local_rank, num_classes,
                             opt.seg['use_LN'] if 'use_LN' in opt.seg else False, interp_size).to(device)

    for e in range(epoch):
        sampler.set_epoch(e)
        pbar = tqdm(train_loader, disable=(local_rank!=0))
        for i, (image, label) in enumerate(pbar):
            processed_label = label_mapping(label.to(device), opt.seg['train_data_path'])
            loss = classifiers.train(image.to(device), processed_label)
            if grid_search:
                pbar.set_description("[epoch %d / iter %d]: lr: %.1e" % (e, i, classifiers.get_lr()))
            else:
                pbar.set_description("[epoch %d / iter %d]: lr: %.1e loss: %.4f" % (e, i, classifiers.get_lr(), loss.item()))
        classifiers.schedule_step()

    mious = test()
    for key, miou in mious.items():
        print0("[%s]: Test mIoU: %.2f" % (key, miou * 100))


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
    torch.use_deterministic_algorithms(True, warn_only=True) # for CrossEntropyLoss
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    train(opt)
