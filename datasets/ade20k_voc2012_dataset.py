"""
ADEChallengeData2016 and PascalVOC2012 reader.
"""

import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


def center_crop_arr_NEAREST(pil_image, image_size):
    # modified from: tokenizer/vavae.py -> def center_crop_arr()

    # Use NEAREST for mask downsampling and resizing to avoid creating new labels
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.NEAREST
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.NEAREST
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class ADE20KDataset(Dataset):
    def __init__(self, data_dir, split='training', transform=None, img_size=256):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # modified from: tokenizer/vavae.py -> def img_transform()
        self.mask_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr_NEAREST(pil_image, img_size)),
            transforms.PILToTensor(), # (1, 256, 256), uint8
            transforms.Lambda(lambda x: x.long().squeeze(0)) # (256, 256), long
        ])

        # 'training' or 'validation', defined by ADEChallengeData2016.zip
        if split in ['training', 'train']:
            split = 'training'
        elif split in ['validation', 'valid', 'val']:
            split = 'validation'
        self.img_dir = os.path.join(data_dir, 'images', split)
        self.ann_dir = os.path.join(data_dir, 'annotations', split)

        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = sorted(os.listdir(self.ann_dir))

        # Make sure image and annotation lists match
        assert len(self.images) == len(self.annotations)
        assert all(img.replace('.jpg', '.png') == ann for img, ann in zip(self.images, self.annotations))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(ann_path)
        assert mask.mode == 'L'

        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask


class VOC2012Dataset(Dataset):
    def __init__(self, data_dir, split='training', transform=None, img_size=256):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.mask_transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr_NEAREST(pil_image, img_size)),
            transforms.PILToTensor(), # (1, 256, 256), uint8
            transforms.Lambda(lambda x: x.long().squeeze(0)) # (256, 256), long
        ])

        self.img_dir = os.path.join(data_dir, 'JPEGImages')
        self.ann_dir = os.path.join(data_dir, 'SegmentationClass')

        # 'train' or 'val', defined by VOCtrainval_11-May-2012.tar
        if split in ['training', 'train']:
            filename = 'train.txt'
        elif split in ['validation', 'valid', 'val']:
            filename = 'val.txt'
        split_file = os.path.join(data_dir, 'ImageSets', 'Segmentation', filename)
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        ann_path = os.path.join(self.ann_dir, f'{img_id}.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(ann_path)
        assert mask.mode == 'P'

        if self.transform:
            image = self.transform(image)
        mask = self.mask_transform(mask)
        return image, mask
