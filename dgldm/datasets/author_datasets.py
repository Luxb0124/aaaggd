import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


def get_nonorm_transform(resolution):
    nonorm_transform = transforms.Compose(
        [transforms.Resize((resolution, resolution),
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor()])
    return nonorm_transform


def get_transformers(content_image_size=96, style_image_size=96, resolution=96):
    content_transforms = transforms.Compose(
        [transforms.Resize(content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    style_transforms = transforms.Compose(
        [transforms.Resize(style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])

    target_transforms = transforms.Compose(
        [transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    return content_transforms, style_transforms, target_transforms


class FontDataset(Dataset):
    def __init__(self, data_root, phase, transforms=None, scr=False, num_neg=None, resolution=96):
        super().__init__()
        self.root = data_root
        self.phase = phase
        self.scr = scr
        if self.scr:
            self.num_neg = num_neg

        # Get Data path
        self.get_path()
        if transforms == 'default':
            content_transforms, style_transforms, target_transforms = get_transformers(
                content_image_size=resolution,
                style_image_size=resolution,
                resolution=resolution)
            self.transforms = [content_transforms, style_transforms, target_transforms]
        else:
            self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(resolution)

    def get_path(self):
        self.target_images = []
        # images with related style
        self.style_to_images = {}
        target_image_dir = f"{self.root}/{self.phase}/TargetImage"
        for style in os.listdir(target_image_dir):
            images_related_style = []
            for img in os.listdir(f"{target_image_dir}/{style}"):
                img_path = f"{target_image_dir}/{style}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)
            self.style_to_images[style] = images_related_style

    def __getitem__(self, index):
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split('/')[-1]
        style, content = target_image_name.split('.')[0].split('+')

        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"
        content_image = Image.open(content_image_path).convert('RGB')

        # Random sample used for style image
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        style_image_path = random.choice(images_related_style)
        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image}

        if self.scr:
            # Get neg image from the different style of the same content
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)
            choose_neg_names = []
            for i in range(self.num_neg):
                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)
                choose_neg_name = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.png"
                choose_neg_names.append(choose_neg_name)

            # Load neg_images
            for i, neg_name in enumerate(choose_neg_names):
                neg_image = Image.open(neg_name).convert("RGB")
                if self.transforms is not None:
                    neg_image = self.transforms[2](neg_image)
                if i == 0:
                    neg_images = neg_image[None, :, :, :]
                else:
                    neg_images = torch.cat([neg_images, neg_image[None, :, :, :]], dim=0)
            sample["neg_images"] = neg_images
        return sample

    def __len__(self):
        return len(self.target_images)


def check_data(dict_data):
    for k in dict_data.keys():
        v = dict_data[k]
        print(type(v))
        if isinstance(v, np.ndarray):
            print('check', k, v.shape, v.min(), v.max())
        elif isinstance(v, str):
            print('check', k, v)
        elif isinstance(v, float):
            print('check', k, v)
        elif isinstance(v, list):
            print('check', k, v)
        elif isinstance(v, torch.Tensor):
            print('check', k, v.shape, v.min(), v.max(), v.device)
        else:
            assert 1 == 2


if __name__ == '__main__':
    resolution = 96
    content_image_size = 96
    style_image_size = 96
    content_transforms, style_transforms, target_transforms = get_transformers(content_image_size=content_image_size,
                                                                               style_image_size=style_image_size,
                                                                               resolution=resolution)
    crt_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(crt_dir, "../../../datasets/font_datasets/SEPARATE/korean_for_FontDiffuser/")
    train_font_dataset = FontDataset(data_root=data_root, phase='train', scr=False, resolution=resolution,
                                     transforms='default',)
    item = train_font_dataset[0]
    print('datasets len', len(train_font_dataset))
    check_data(item)
    # content_image torch.Size([3, 96, 96]) tensor(-1.) tensor(1.)
    # style_image torch.Size([3, 96, 96]) tensor(-0.1765) tensor(1.)
    # target_image torch.Size([3, 96, 96]) tensor(-0.1608) tensor(1.)
    # nonorm_target_image torch.Size([3, 96, 96]) tensor(0.4196) tensor(1.)
