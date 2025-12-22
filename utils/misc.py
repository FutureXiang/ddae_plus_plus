import os
import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


class Meter:
    def __init__(self, n_items, ema=0.95):
        self.data = [None for _ in range(n_items)]
        self.ema = ema
    
    def update(self, x):
        for i in range(len(x)):
            if self.data[i] is None:
                self.data[i] = x[i]
            else:
                self.data[i] = self.ema * self.data[i] + (1 - self.ema) * x[i]

    def get(self, i):
        return self.data[i]


class Config(object):
    def __init__(self, yaml_path):
        with open(yaml_path, 'r') as f:
            opt = yaml.safe_load(f)
        print0(opt)
        self._original_content = opt

        for key in opt:
            setattr(self, key, opt[key])

    def sample_arg(self, key, default):
        return self.sample[key] if key in self.sample else default


def init_seeds(RANDOM_SEED=1337, no=0):
    RANDOM_SEED += no
    print("local_rank = {}, seed = {}".format(no, RANDOM_SEED))
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def gather_tensor(tensor):
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list


def DataLoaderDDP(dataset, global_bs, shuffle=True):
    gpus = dist.get_world_size()
    local_bs = global_bs // gpus
    global_real = local_bs * gpus
    print0(f"Batch size per gpu = {local_bs}, with global (effective) batch size = {global_real}")

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=local_bs,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=shuffle,
    )
    return dataloader, sampler


def print0(*args, **kwargs):
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        print(*args, **kwargs)


def load_checkpoint_ema(raw_model, checkpoint):
    assert 'ema' in checkpoint.keys()
    old_state_dict = checkpoint['ema']
    filtered_raw_state_dict = {}
    filtered_ema_renamed_state_dict = {}
    for k, v in old_state_dict.items():
        if k.startswith('ema_model.'):
            # len('ema_model.') == 10
            filtered_ema_renamed_state_dict[k[10:]] = v
        else:
            filtered_raw_state_dict[k] = v

    # one more buffer named "step". see utils/ema.py
    if len(filtered_raw_state_dict) == len(filtered_ema_renamed_state_dict) + 1:
        print0("EMA weights detected, dumped by our EMA class in train.py")
        raw_model.load_state_dict(filtered_ema_renamed_state_dict)
    elif len(filtered_ema_renamed_state_dict) == 0:
        print0("EMA weights detected, loaded from official checkpoint")
        raw_model.load_state_dict(filtered_raw_state_dict)
    else:
        raise NotImplementedError()
    return raw_model
