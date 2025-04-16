import os
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image


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


class CustomizedDataset(data.Dataset):
    def __init__(self, data_root=None, standard_root=None, resolution=96, transforms=None,
                 rec_char_dict_path=None):
        self.data_root = data_root
        self.standard_root = standard_root
        self.resolution = resolution
        print(os.path.exists(self.data_root), self.data_root)
        print(os.path.exists(self.standard_root), self.standard_root)
        assert os.path.exists(self.data_root) and os.path.exists(self.standard_root)

        # to save memory
        self.styles = [os.path.basename(b) for b in glob.glob('%s/*' %(self.data_root))]
        self.chars = [os.path.basename(b)[0] for b in glob.glob('%s/*' %(os.path.join(self.data_root, self.styles[0])))]
        print('style nums: %d, char nums: %d.' %(len(self.styles), len(self.chars)))

        if transforms == 'default':
            content_transforms, style_transforms, target_transforms = get_transformers(
                content_image_size=resolution,
                style_image_size=resolution,
                resolution=resolution)
            self.transforms = [content_transforms, style_transforms, target_transforms]
            self.nonorm_transforms = get_nonorm_transform(resolution)
        else:
            self.transforms = transforms
            self.nonorm_transforms = None

        if rec_char_dict_path is None:
            ocr_weight_dir = os.path.join(os.path.dirname(__file__), '../aux_ocrs', 'ocr_weights')
            rec_char_dict_path = os.path.join(ocr_weight_dir, 'ppocr_keys_v1.txt')
        self.all_rec_chars = self.get_char_dict(rec_char_dict_path)
        self.char2id = {x: i for i, x in enumerate(self.all_rec_chars)}
        self.all_rec_chars_len = len(self.all_rec_chars)

    def get_char_dict(self, character_dict_path):
        character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character_str.append(line)
        dict_character = list(character_str)
        dict_character = ['sos'] + dict_character + [' ']  # eos is space
        return dict_character

    def __len__(self):
        return len(self.styles) * len(self.chars)

    def __getitem__(self, idx):
        style_index = idx // len(self.chars)
        content_index = idx % len(self.chars)
        style = self.styles[style_index]
        src_char = random.choice(self.chars)
        ref_char = self.chars[content_index]

        src_char_id = torch.tensor(self.char2id.get(src_char, self.all_rec_chars_len - 1))
        ref_char_id = torch.tensor(self.char2id.get(ref_char, self.all_rec_chars_len - 1))
        src_char_len = torch.tensor(1)
        ref_char_len = torch.tensor(1)

        std_src_path = os.path.join(self.standard_root, '%s.png' %src_char)
        std_ref_path = os.path.join(self.standard_root, '%s.png' %ref_char)
        sty_src_path = os.path.join(self.data_root, style, '%s.png' %src_char)
        sty_ref_path = os.path.join(self.data_root, style, '%s.png' %ref_char)

        std_src_img = Image.open(std_src_path).convert('RGB').resize((self.resolution, self.resolution))
        std_ref_img = Image.open(std_ref_path).convert('RGB').resize((self.resolution, self.resolution))
        sty_src_img = Image.open(sty_src_path).convert('RGB').resize((self.resolution, self.resolution))
        sty_ref_img = Image.open(sty_ref_path).convert('RGB').resize((self.resolution, self.resolution))

        if self.transforms is None:
            std_src_img = np.asarray(std_src_img)
            std_ref_img = np.asarray(std_ref_img)
            sty_src_img = np.asarray(sty_src_img)
            sty_ref_img = np.asarray(sty_ref_img)
            std_src_img = (std_src_img.astype(np.float32) / 127.5) - 1.0
            std_ref_img = (std_ref_img.astype(np.float32) / 127.5) - 1.0
            sty_src_img = (sty_src_img.astype(np.float32) / 127.5) - 1.0
            sty_ref_img = (sty_ref_img.astype(np.float32) / 127.5) - 1.0
        else:
            # content_transforms, style_transforms, target_transforms
            std_src_img = self.transforms[0](std_src_img)
            std_ref_img = self.transforms[1](std_ref_img)
            sty_src_img = self.transforms[0](sty_src_img)
            sty_ref_img = self.transforms[0](sty_ref_img)

        rt_dict = {}
        rt_dict['std_src_img'] = std_src_img
        rt_dict['std_ref_img'] = std_ref_img
        rt_dict['sty_ref_img'] = sty_ref_img
        rt_dict['sty_src_img'] = sty_src_img
        rt_dict['src_char'] = src_char
        rt_dict['ref_char'] = ref_char
        rt_dict['src_char_id'] = src_char_id
        rt_dict['ref_char_id'] = ref_char_id
        rt_dict['src_char_len'] = src_char_len
        rt_dict['ref_char_len'] = ref_char_len
        rt_dict['sty_class_id'] = torch.tensor(style_index)
        return rt_dict


def check_data(dict_data):
    show_imgs = []
    for k in dict_data.keys():
        v = dict_data[k]
        print(type(v))
        if isinstance(v, np.ndarray):
            print('check', k, v.shape, v.min(), v.max(), v.mean())
            if len(v.shape) == 3:
                show_imgs.append(v)
                print('====================k', k)
        elif isinstance(v, str):
            print('check', k, v)
        elif isinstance(v, float):
            print('check', k, v)
        elif isinstance(v, list):
            print('check', k, v)
        elif isinstance(v, torch.Tensor):
            try:
                print('check', k, v.shape, v.min(), v.max(), v.mean(), v.device)
            except:
                print('check', k, v.shape, v.min(), v.max(), v.device)
            if len(v.shape) == 3:
                show_imgs.append(v.unsqueeze(0))
                print('====================k', k)
        else:
            print('check error', k)
            assert 1 == 2
    if len(show_imgs) > 0:
        try:
            _show_imgs = np.concatenate(show_imgs, axis=1)
            _show_imgs = (((_show_imgs + 1) / 2) * 255).astype(np.uint8)
            print(_show_imgs.shape)
            _show_imgs = Image.fromarray(_show_imgs)
            _show_imgs.show()
        except:
            __show_imgs = torch.cat(show_imgs, 0)
            grid = make_grid(__show_imgs)
            grid = (grid + 1) / 2
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            __show_imgs = Image.fromarray(ndarr)
            __show_imgs.show()


def check_train_dataset():
    crt_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(crt_dir, "../../../../datasets/sub_chinese/train")
    standard_root = os.path.join(crt_dir, "../../../../datasets/sub_chinese/standard")
    # transforms = None
    transforms = 'default'
    dataset = CustomizedDataset(data_root=data_root, standard_root=standard_root, resolution=96,
                                transforms=transforms)
    item = dataset[600]
    print('datasets len', len(dataset))
    check_data(item)


def get_train_dataloder(batch_size=16):
    crt_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(crt_dir, "../../../../datasets/sub_chinese2/train")
    standard_root = os.path.join(crt_dir, "../../../../datasets/sub_chinese2/standard")
    # transforms = None
    transforms = 'default'
    dataset = CustomizedDataset(data_root=data_root, standard_root=standard_root, resolution=96,
                                transforms=transforms)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return dataloader, dataset.all_rec_chars


if __name__ == '__main__':
    check_train_dataset()
